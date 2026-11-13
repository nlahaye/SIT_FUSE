"""
Copyright [2022-26], by the California Institute of Technology and Chapman University.
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.

Local job launcher: runs a YAML plan of shell-command jobs on one machine (no scheduler needed), packing them onto
the GPUs and CPU cores, honoring dependencies, and resuming where it stopped. Generic: any SIT-FUSE use case (or any
command) can be described as a plan.

    python -m sit_fuse.local_launcher run    plan.yaml [--dry-run] [--only GLOB ...] [--rerun-failed] [--force]
    python -m sit_fuse.local_launcher status plan.yaml
    python -m sit_fuse.local_launcher list   plan.yaml            # expanded jobs, resources and dependencies

Plan
----
    launcher:
      state_dir: /data/me/launch/sharks   # logs/<job>.log, markers/<job>.done|.failed, claims/, status.csv
      gpus: auto            # auto (nvidia-smi / CUDA_VISIBLE_DEVICES) | N (ids 0..N-1) | [0, 2, 3] | 0 = none
      cpus: auto            # CPU cores the launcher may hand out: auto | int | fraction | -N (all but N)
      pin_cpus: false       # true: give each job its own core ids (sched_setaffinity)
      max_parallel: null    # cap on concurrent jobs
      retries: 0            # extra attempts per job after a failure
      poll_sec: 2
      nice: 0
      gpu_mem_headroom: 0.95  # XLA_PYTHON_CLIENT_MEM_FRACTION = headroom * gpu share
      env: {JAX_COMPILATION_CACHE_DIR: /data/me/jax_cache}   # added to every job
    vars: {cfg: /data/me/config.yaml, py: python}     # template values (may reference each other)
    jobs:
      - name: fit
        cmd: "{py} -m pkg.fit -y {cfg} --K {K} --fold {fold}"
        grid: {K: [2, 3, 4], fold: "range(5)"}         # cartesian product; also cases: [{...}, ...]
        cpus: 4             # cores (thread env vars OMP/MKL/OPENBLAS/NUMBA... are set to this)
        gpu: 0              # 0 = CPU job (CUDA_VISIBLE_DEVICES=""), 0.25 = a quarter of one GPU (four such jobs
                            # share a device), 1 = one whole GPU, 2 = two whole GPUs
        creates: ["{out}/fit_K{K}_F{fold}.npz"]       # optional: skip the job when all of these exist
        env: {MY_FLAG: "1"}
        priority: 0         # higher first among ready jobs (default: plan order)
        timeout: null       # seconds
        id_keys: [K, fold]  # params used in the job id (default: all; use when some params are long paths)
      - name: score
        cmd: "..."
        grid: {K: [2, 3, 4], fold: "range(5)"}
        after: [fit]        # depends on the fit jobs that share its grid values (here the same K and fold);
                            # no shared keys = all of them. {name: fit, on: [K]} joins on K only.
        needs: ["{out}/fit_K{K}_F{fold}.npz"]   # and / or: depends on the jobs whose provides / creates list
                            # these paths (warning if no job provides a path that does not exist yet)
        provides: [...]     # paths this job writes (for needs; unlike creates, not used to skip the job)

Job ids are <name>.<key><value>_..., and each job gets LAUNCHER_JOB, LAUNCHER_GPUS, LAUNCHER_CPUS in its env.

Resuming / several launchers
----------------------------
A job is done when markers/<id>.done exists with the same command (edit the command and it reruns) or all of its
`creates` paths exist. Failed jobs are skipped on the next run unless --rerun-failed; their dependents are
reported as blocked. Before starting a job the launcher claims it (claims/<id>, atomic create), so several launcher
processes can share one state_dir: e.g. one per machine on a shared filesystem, each with its own gpus / cpus, or a
second launcher started later on the same machine with spare devices. Claims of dead processes on the same host are
released automatically; `run --release-claims` drops all claims (only when no other launcher is running).
Ctrl-C / SIGTERM stops the running jobs (their claims are released, nothing is marked done).
"""

import argparse
import fnmatch
import glob
import hashlib
import itertools
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time

try:
    from sit_fuse.compute_resources import _THREAD_ENV_VARS, available_cpus, resolve_num_threads
except Exception:  # noqa: BLE001  (standalone use)
    _THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "NUMBA_NUM_THREADS",
                        "GDAL_NUM_THREADS", "LOKY_MAX_CPU_COUNT")

    def available_cpus():
        try:
            return len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            return os.cpu_count() or 1

    def resolve_num_threads(spec, available=None):
        available = available or available_cpus()
        if spec in (None, "auto"):
            return available
        spec = float(spec) if isinstance(spec, str) and "." in spec else int(spec) if isinstance(spec, str) else spec
        if isinstance(spec, float):
            return max(1, int(round(spec * available)))
        return max(1, available + spec if spec < 0 else min(spec, available))

HOST = socket.gethostname()
DONE, FAILED, BLOCKED, PENDING, RUNNING, ELSEWHERE = "done", "failed", "blocked", "pending", "running", "elsewhere"


# ---------------------------------------------------------------------------
# Plan expansion
# ---------------------------------------------------------------------------

class Job:
    def __init__(self, name, params, cmd, cpus, gpu, creates, env, priority, timeout, order, id_keys=None):
        self.name, self.params, self.cmd = name, params, cmd
        self.cpus, self.gpu, self.creates, self.env = cpus, gpu, creates, env
        self.priority, self.timeout, self.order = priority, timeout, order
        tail = "_".join(f"{k}{v}" for k, v in params.items() if id_keys is None or k in id_keys)
        self.id = re.sub(r"[^A-Za-z0-9_.=+-]", "-", f"{name}.{tail}" if tail else name)
        self.deps = []
        self.hash = hashlib.sha1(cmd.encode()).hexdigest()[:16]


def _values(v):
    if isinstance(v, str):
        m = re.fullmatch(r"\s*range\(([^)]*)\)\s*", v)
        if m:
            return list(range(*[int(x) for x in m.group(1).split(",")]))
        return [v]
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _expand_params(spec):
    cases = spec.get("cases") or [{}]
    grid = spec.get("grid") or {}
    keys = list(grid)
    out = []
    for case in cases:
        for combo in itertools.product(*[_values(grid[k]) for k in keys]):
            p = dict(case)
            p.update(zip(keys, combo))
            out.append(p)
    return out


def _fmt(s, ctx):
    return str(s).format_map(ctx) if isinstance(s, str) else s


def _resolve_vars(vars_):
    out = {k: v for k, v in (vars_ or {}).items()}
    for _ in range(5):                  # vars may reference other vars
        new = {k: _fmt(v, out) for k, v in out.items()}
        if new == out:
            break
        out = new
    return out


def load_plan(path):
    import yaml
    with open(path) as f:
        plan = yaml.safe_load(f)
    st = dict(plan.get("launcher") or {})
    st.setdefault("state_dir", os.path.join(os.path.dirname(os.path.abspath(path)), "launch_state"))
    vars_ = _resolve_vars(plan.get("vars"))
    st = {k: _fmt(v, vars_) for k, v in st.items()}
    jobs, by_name, specs, order = [], {}, [], 0
    for spec in plan.get("jobs") or []:
        name = spec["name"]
        if name in by_name:
            raise ValueError(f"duplicate job name {name!r}")
        by_name[name] = []
        for p in _expand_params(spec):
            ctx = dict(vars_, **p)
            env = {k: str(_fmt(v, ctx)) for k, v in (spec.get("env") or {}).items()}
            j = Job(name, p, _fmt(spec["cmd"], ctx), int(spec.get("cpus", 1)), float(spec.get("gpu", 0) or 0),
                    [_fmt(c, ctx) for c in _values(spec.get("creates") or [])], env,
                    float(spec.get("priority", 0)), spec.get("timeout"), order, spec.get("id_keys"))
            # realpath: a needed symlink (e.g. an init file linked into another out_dir) matches its target's job
            j.provides = [os.path.realpath(f) for f in (_fmt(c, ctx) for c in _values(spec.get("provides") or []))
                          if f] + [os.path.realpath(c) for c in j.creates if c]
            j.needs = [os.path.realpath(f) for f in (_fmt(c, ctx) for c in _values(spec.get("needs") or [])) if f]
            order += 1
            jobs.append(j)
            by_name[name].append(j)
        specs.append(spec)
    for spec in specs:                  # after: natural join on shared grid keys (or `on`)
        name = spec["name"]
        for dep in _values(spec.get("after") or []):
            dname, on = (dep, None) if isinstance(dep, str) else (dep["name"], dep.get("on"))
            if dname not in by_name:
                raise ValueError(f"{name}: after {dname!r} is not a job name")
            for j in by_name[name]:
                for d in by_name[dname]:
                    keys = set(j.params) & set(d.params) if on is None else set(_values(on))
                    if all(str(j.params.get(k)) == str(d.params.get(k)) for k in keys):
                        j.deps.append(d)
    providers = {}
    for j in jobs:                      # needs: the jobs that provide / create those paths
        for f in j.provides:
            providers.setdefault(f, []).append(j)
    missing = []
    for j in jobs:
        for f in j.needs:
            prov = [d for d in providers.get(f, []) if d is not j]
            if prov:
                j.deps += [d for d in prov if d not in j.deps]
            elif not os.path.exists(f):
                missing.append((j.id, f))
    st["_missing_needs"] = missing
    ids = [j.id for j in jobs]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"job ids not unique (add a grid key or rename): {dup[:5]}")
    return st, _toposort(jobs)


def _toposort(jobs):
    """Plan order where possible, every job after its dependencies; raises on cycles."""
    indeg = {j.id: len(set(d.id for d in j.deps)) for j in jobs}
    users = {}
    for j in jobs:
        for d in set(d.id for d in j.deps):
            users.setdefault(d, []).append(j)
    import heapq
    heap = [(j.order, j.id, j) for j in jobs if indeg[j.id] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        _, _, j = heapq.heappop(heap)
        out.append(j)
        for u in users.get(j.id, []):
            indeg[u.id] -= 1
            if indeg[u.id] == 0:
                heapq.heappush(heap, (u.order, u.id, u))
    if len(out) != len(jobs):
        raise ValueError("dependency cycle among: " + ", ".join(sorted(j.id for j in jobs if indeg[j.id] > 0))[:500])
    return out


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

def detect_gpus(spec="auto"):
    if spec in (None, 0, "0", False, "none"):
        return []
    if isinstance(spec, (list, tuple)):
        return [str(g) for g in spec]
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        return [str(i) for i in range(int(spec))]
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis not in (None, ""):
        return [g.strip() for g in vis.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20)
        return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()] if out.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        return []


class Pool:
    """GPU shares (1.0 per device) and CPU cores."""

    def __init__(self, gpus, cpus, pin_cpus=False):
        self.gpu_free = {g: 1.0 for g in gpus}
        self.ncpu = cpus
        self.cpu_free = cpus
        try:
            allowed = sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            allowed = list(range(os.cpu_count() or 1))
        self.cores = allowed[:cpus] if pin_cpus else None

    def fits(self, job):
        return self.check(job) is None

    def check(self, job):
        """None if the job can never run on this pool, else a reason (used to reject impossible jobs)."""
        if job.cpus > self.ncpu:
            return f"needs {job.cpus} cpus, pool has {self.ncpu}"
        if job.gpu > 0 and not self.gpu_free:
            return "needs a GPU, none configured"
        if job.gpu > 1 and int(round(job.gpu)) > len(self.gpu_free):
            return f"needs {job.gpu:g} GPUs, pool has {len(self.gpu_free)}"
        return None

    def take(self, job):
        """-> (gpu ids, core ids) or None when it does not fit now."""
        if job.cpus > self.cpu_free:
            return None
        gpus = []
        if job.gpu > 0:
            eps = 1e-9
            if job.gpu <= 1:
                # best fit: the device with the least free share that still fits (keeps whole devices free)
                cand = [(f, g) for g, f in self.gpu_free.items() if f + eps >= job.gpu]
                if not cand:
                    return None
                gpus = [min(cand)[1]]
            else:
                whole = [g for g, f in self.gpu_free.items() if f >= 1 - eps]
                n = int(round(job.gpu))
                if len(whole) < n:
                    return None
                gpus = whole[:n]
        share = job.gpu if job.gpu <= 1 else 1.0
        for g in gpus:
            self.gpu_free[g] -= share
        self.cpu_free -= job.cpus
        cores = None
        if self.cores is not None:
            cores, self.cores = self.cores[:job.cpus], self.cores[job.cpus:]
        return gpus, cores

    def give(self, job, gpus, cores):
        share = job.gpu if job.gpu <= 1 else 1.0
        for g in gpus:
            self.gpu_free[g] = min(1.0, self.gpu_free[g] + share)
        self.cpu_free += job.cpus
        if cores is not None:
            self.cores = sorted(self.cores + list(cores))


# ---------------------------------------------------------------------------
# State on disk
# ---------------------------------------------------------------------------

class State:
    def __init__(self, state_dir):
        self.dir = state_dir
        for sub in ("logs", "markers", "claims"):
            os.makedirs(os.path.join(state_dir, sub), exist_ok=True)

    def p(self, sub, name):
        return os.path.join(self.dir, sub, name)

    def status(self, job):
        f = self.p("markers", job.id + ".done")
        if os.path.exists(f):
            with open(f) as fh:
                try:
                    if json.load(fh).get("hash") == job.hash:
                        return DONE
                except ValueError:
                    pass
        if job.creates and all(glob.glob(c) for c in job.creates):
            return DONE
        if os.path.exists(self.p("markers", job.id + ".failed")):
            return FAILED
        return None

    def mark(self, job, kind, **info):
        other = self.p("markers", job.id + (".failed" if kind == DONE else ".done"))
        if kind == DONE and os.path.exists(other):
            os.remove(other)
        _atomic_json(self.p("markers", job.id + "." + kind), dict(hash=job.hash, cmd=job.cmd, host=HOST, **info))

    def clear_failed(self, job):
        f = self.p("markers", job.id + ".failed")
        if os.path.exists(f):
            os.remove(f)

    def claim(self, job):
        path = self.p("claims", job.id)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if self._stale(path):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                return self.claim(job)
            return False
        with os.fdopen(fd, "w") as fh:
            json.dump(dict(host=HOST, pid=os.getpid(), t=time.time()), fh)
        return True

    def claimed(self, job):
        path = self.p("claims", job.id)
        return os.path.exists(path) and not self._stale(path)

    def release(self, job):
        try:
            os.remove(self.p("claims", job.id))
        except FileNotFoundError:
            pass

    def release_all(self):
        for f in glob.glob(self.p("claims", "*")):
            os.remove(f)

    @staticmethod
    def _stale(path):
        try:
            with open(path) as fh:
                c = json.load(fh)
        except (OSError, ValueError):
            return False                # being written
        if c.get("host") != HOST:
            return False
        try:
            os.kill(int(c["pid"]), 0)
            return False
        except ProcessLookupError:
            return True
        except PermissionError:
            return False


def _atomic_json(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def job_env(job, gpus, st, base=None):
    env = dict(os.environ if base is None else base)
    env.update({k: str(v) for k, v in (st.get("env") or {}).items()})
    for v in _THREAD_ENV_VARS:
        env[v] = str(job.cpus)
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        if job.gpu < 1:
            env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION",
                           f"{float(st.get('gpu_mem_headroom', 0.95)) * job.gpu:.3f}")
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env.update({"LAUNCHER_JOB": job.id, "LAUNCHER_GPUS": ",".join(gpus), "LAUNCHER_CPUS": str(job.cpus)})
    env.update(job.env)
    return env


class Launcher:
    def __init__(self, plan_path, only=None, rerun_failed=False, force=False, dry_run=False, max_parallel=None,
                 gpus=None, cpus=None, verbose=True):
        self.st, self.jobs = load_plan(plan_path)
        if gpus is not None:
            self.st["gpus"] = gpus
        if cpus is not None:
            self.st["cpus"] = cpus
        self.state = State(self.st["state_dir"])
        self.dry_run, self.force, self.verbose = dry_run, force, verbose
        sel = self.jobs
        if only:
            sel = [j for j in self.jobs if any(fnmatch.fnmatch(j.id, pat) or fnmatch.fnmatch(j.name, pat)
                                               for pat in only)]
        self.selected = {j.id for j in sel}
        self.rerun_failed = rerun_failed
        self.forced = set(self.selected) if force else set()   # ignore done markers / creates until rerun here
        spec = self.st.get("cpus", "auto")
        if isinstance(spec, int) and spec > 0 or isinstance(spec, str) and spec.isdigit():
            ncpu = int(spec)            # explicit core count: taken as given (e.g. a plan listed on another machine)
        else:
            ncpu = resolve_num_threads(spec, available_cpus()) or available_cpus()
        self.pool = Pool(detect_gpus(self.st.get("gpus", "auto")), ncpu, bool(self.st.get("pin_cpus", False)))
        mp = max_parallel if max_parallel is not None else self.st.get("max_parallel")
        self.max_parallel = int(mp) if mp else None
        self.running = {}               # id -> (job, Popen, gpus, cores, t0, logfh, attempt)
        self.attempts = {}
        self.rows = {}
        self._stop = False

    def log(self, *a):
        if self.verbose:
            print(time.strftime("%H:%M:%S"), *a, flush=True)

    # status of every job given markers / claims (no side effects)
    def statuses(self):
        out = {}
        for j in self.jobs:            # topological order (load_plan)
            s = None if j.id in self.forced else self.state.status(j)
            if s == FAILED and self.rerun_failed and j.id in self.selected:
                s = None
            if s is None:
                if j.id in self.running:
                    s = RUNNING
                elif self.state.claimed(j):
                    s = ELSEWHERE
                elif any(out[d.id] in (FAILED, BLOCKED) for d in j.deps):
                    s = BLOCKED
                else:
                    s = PENDING
            out[j.id] = s
        return out

    def run(self):
        bad = [(j.id, self.pool.check(j)) for j in self.jobs if j.id in self.selected and self.pool.check(j)]
        if bad and not self.dry_run:
            raise SystemExit("jobs that cannot run with this pool:\n  " + "\n  ".join(f"{i}: {r}" for i, r in bad[:20]))
        self.log(f"pool: gpus={list(self.pool.gpu_free)} cpus={self.pool.ncpu} state_dir={self.state.dir}")
        for jid, f in self.st.get("_missing_needs", [])[:20]:
            self.log(f"WARNING: {jid} needs {f}: no job provides it and it does not exist")
        if self.forced and not self.dry_run:
            for jid in self.forced:
                for k in ("done", "failed"):
                    f = self.state.p("markers", f"{jid}.{k}")
                    if os.path.exists(f):
                        os.remove(f)
        if self.dry_run:
            return self._dry()
        prev = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        for s in prev:
            signal.signal(s, self._on_signal)
        try:
            self._loop()
        finally:
            self._kill_all()
            for s, h in prev.items():
                signal.signal(s, h)
            self.write_status()
        st = self.statuses()
        sel = [st[i] for i in st if i in self.selected]
        summary = {k: sel.count(k) for k in (DONE, FAILED, BLOCKED, PENDING, ELSEWHERE) if sel.count(k)}
        self.log("finished:", summary)
        return 0 if not any(st[i] in (FAILED, BLOCKED) for i in st if i in self.selected) else 1

    def _dry(self):
        st = self.statuses()
        for j in self.jobs:
            if j.id in self.selected:
                print(f"[{st[j.id]:9s}] {j.id}  cpus={j.cpus} gpu={j.gpu:g}  after={[d.id for d in j.deps]}\n"
                      f"            {j.cmd}")
        return 0

    def _ready(self, st):
        ready = []
        for j in self.jobs:
            if j.id not in self.selected or st[j.id] != PENDING:
                continue
            if all(st[d.id] == DONE for d in j.deps):
                ready.append(j)
        # waiting on deps outside the selection that are not done and will not run: treat as blocked
        return sorted(ready, key=lambda j: (-j.priority, j.order))

    def _loop(self):
        poll = float(self.st.get("poll_sec", 2))
        while not self._stop:
            self._reap()
            st = self.statuses()
            ready = self._ready(st)
            for j in ready:
                if self.max_parallel and len(self.running) >= self.max_parallel:
                    break
                if not self.pool.fits(j):
                    continue
                res = self.pool.take(j)
                if res is None:
                    continue           # smaller jobs further down may still fit (backfill)
                if not self.state.claim(j):
                    self.pool.give(j, *res)
                    continue
                if self.state.status(j) == DONE:        # finished by another launcher meanwhile
                    self.state.release(j)
                    self.pool.give(j, *res)
                    continue
                self._start(j, *res)
            st = self.statuses()
            active = [i for i in st if i in self.selected and st[i] in (RUNNING, ELSEWHERE)]
            waiting = [j for j in self.jobs if j.id in self.selected and st[j.id] == PENDING]
            if not active and not self.running:
                if not waiting:
                    break
                # pending jobs whose deps are outside the selection and not done
                stuck = [j.id for j in waiting if not all(st[d.id] == DONE for d in j.deps)]
                if len(stuck) == len(waiting):
                    self.log("cannot start (dependencies not selected / not done):", stuck[:10])
                    break
            time.sleep(poll)

    def _start(self, j, gpus, cores):
        a = self.attempts.get(j.id, 0) + 1
        self.attempts[j.id] = a
        logf = open(self.state.p("logs", j.id + ".log"), "a")
        logf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} host={HOST} attempt={a} gpus={gpus} "
                   f"cpus={j.cpus} ===\n$ {j.cmd}\n")
        logf.flush()
        nice = int(self.st.get("nice", 0) or 0)

        def pre():
            os.setsid()
            if cores is not None and hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, cores)
            if nice:
                os.nice(nice)
        proc = subprocess.Popen(j.cmd, shell=True, env=job_env(j, gpus, self.st), stdout=logf,
                                stderr=subprocess.STDOUT, preexec_fn=pre, executable="/bin/bash")
        self.running[j.id] = (j, proc, gpus, cores, time.time(), logf, a)
        self.state.clear_failed(j)
        self.log(f"start {j.id} gpus={','.join(gpus) or '-'} cpus={j.cpus} (attempt {a})")

    def _reap(self):
        retries = int(self.st.get("retries", 0) or 0)
        for jid in list(self.running):
            j, proc, gpus, cores, t0, logf, a = self.running[jid]
            rc = proc.poll()
            if rc is None and j.timeout and time.time() - t0 > float(j.timeout):
                self._kill(proc)
                rc = proc.wait()
                logf.write(f"\n=== timeout after {j.timeout} s ===\n")
            if rc is None:
                continue
            wall = time.time() - t0
            logf.write(f"\n=== exit {rc} after {wall:.1f} s ===\n")
            logf.close()
            self.pool.give(j, gpus, cores)
            del self.running[jid]
            self.rows[jid] = dict(job=jid, name=j.name, rc=rc, wall_sec=round(wall, 1), gpus=",".join(gpus),
                                  cpus=j.cpus, attempt=a, host=HOST, end=time.strftime("%Y-%m-%d %H:%M:%S"),
                                  **{f"p_{k}": v for k, v in j.params.items()})
            if self._stop:
                self.state.release(j)
                continue
            if rc == 0:
                self.forced.discard(jid)
                self.state.mark(j, DONE, wall_sec=wall, gpus=gpus)
                self.log(f"done  {jid} ({wall:.0f} s)")
            elif a <= retries:
                self.log(f"fail  {jid} rc={rc}; retry {a}/{retries}")
            else:
                self.forced.discard(jid)
                self.state.mark(j, FAILED, rc=rc, wall_sec=wall)
                self.log(f"FAIL  {jid} rc={rc} (log: {self.state.p('logs', jid + '.log')})")
            self.state.release(j)
            self.write_status()

    @staticmethod
    def _kill(proc):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _kill_all(self):
        for jid, (j, proc, gpus, cores, t0, logf, a) in list(self.running.items()):
            self._kill(proc)
            logf.write("\n=== stopped by launcher ===\n")
            logf.close()
            self.pool.give(j, gpus, cores)
            self.state.release(j)
            del self.running[jid]

    def _on_signal(self, *_):
        self.log("stopping: terminating running jobs")
        self._stop = True

    def write_status(self):
        import csv
        st = self.statuses()
        path = os.path.join(self.state.dir, "status.csv")
        old = {}
        if os.path.exists(path):
            with open(path) as fh:
                old = {r["job"]: r for r in csv.DictReader(fh)}
        keys = ["job", "name", "status", "rc", "wall_sec", "gpus", "cpus", "attempt", "host", "end"]
        rows = []
        for j in self.jobs:
            r = dict(old.get(j.id, {}))
            r.update(self.rows.get(j.id, {}))
            r.update(job=j.id, name=j.name, status=st[j.id])
            rows.append({k: r.get(k, "") for k in keys})
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
        return rows


def print_status(plan_path, only=None):
    L = Launcher(plan_path, only=only, verbose=False)
    st = L.statuses()
    by = {}
    for j in L.jobs:
        if j.id in L.selected:
            by.setdefault(j.name, []).append(st[j.id])
    for name, ss in by.items():
        counts = ", ".join(f"{k}={ss.count(k)}" for k in (DONE, RUNNING, ELSEWHERE, PENDING, FAILED, BLOCKED)
                           if ss.count(k))
        print(f"{name:32s} {len(ss):5d} jobs  {counts}")
    fails = [j.id for j in L.jobs if j.id in L.selected and st[j.id] == FAILED]
    for f in fails[:20]:
        print("  failed:", f, "->", L.state.p("logs", f + ".log"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("action", choices=["run", "status", "list"])
    ap.add_argument("plan")
    ap.add_argument("--only", nargs="*", default=None, help="job ids / names (glob patterns)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rerun-failed", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun the selected jobs even if done")
    ap.add_argument("--max-parallel", type=int, default=None)
    ap.add_argument("--gpus", default=None, help="override launcher.gpus: N (= ids 0..N-1) | id list 2,3 (one id: 3,) "
                    "| none | auto")
    ap.add_argument("--cpus", default=None, help="override launcher.cpus")
    ap.add_argument("--release-claims", action="store_true", help="drop all claims first (no other launcher running)")
    a = ap.parse_args(argv)
    gpus = None
    if a.gpus is not None:
        gpus = ([g for g in a.gpus.split(",") if g] if "," in a.gpus else
                int(a.gpus) if a.gpus.isdigit() else a.gpus)
    if a.action == "status":
        return print_status(a.plan, a.only)
    L = Launcher(a.plan, only=a.only, rerun_failed=a.rerun_failed, force=a.force,
                 dry_run=a.dry_run or a.action == "list", max_parallel=a.max_parallel, gpus=gpus, cpus=a.cpus)
    if a.release_claims:
        L.state.release_all()
    return L.run()


if __name__ == "__main__":
    sys.exit(main() or 0)
