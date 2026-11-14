"""
Copyright [2022-26], by the California Institute of Technology and Chapman University.
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.

Small helpers for running independent pieces of one SIT-FUSE step in parallel (in-process workers) or as
separate launcher jobs (shards), and for files that several jobs update. Ported from NatureNet's
models/irl/parallel.py; thread counts are resolved with sit_fuse.compute_resources.

cpu_map(fn, items, n_jobs, threads)  ordered results of fn(item) (joblib / loky processes; BLAS / OpenMP pools
                                     capped at `threads` per worker; large numpy arguments are memory-mapped).
                                     n_jobs <= 1 runs serially in this process (identical results; the default).
                                     fn must be a module-level function.
worker_plan(cfg, workers, threads)   (n_workers, threads_per_worker) from --workers / env SIT_FUSE_NUM_WORKERS /
                                     YAML compute.num_workers, splitting the thread budget of apply_compute_limits.
parse_shard("i/N")                   (i, N) with 0 <= i < N; None for None / "" / "0/1".
shard_indices(n, shard, mode)        indices of shard i of N over n items: "stride" (i, i+N, ...; balances scene
                                     cost) or "block" (contiguous; keeps neighbouring frames together).
shard_path(dir, stem, shard, ext)    <dir>/<stem>.shard_0003_of_0008<ext>
shard_files(dir, stem, n, ext)       all N shard files; raises listing the missing ones.
locked(path)                         exclusive advisory lock (fcntl, <path>.lock) around a read-modify-write.
atomic_write_bytes / update_json     atomic replace; locked read -> merge(old or None) -> atomic write.
"""
import contextlib
import json
import os
from pathlib import Path


def _resolve(spec, available=None):
    try:
        from sit_fuse.compute_resources import available_cpus, resolve_num_threads
    except Exception:  # noqa: BLE001  (standalone use)
        def available_cpus():
            try:
                return len(os.sched_getaffinity(0))
            except (AttributeError, OSError):
                return os.cpu_count() or 1

        def resolve_num_threads(s, available=None):
            a = available or available_cpus()
            if s in (None, "auto"):
                return None
            s = int(s)
            return max(1, a + s if s < 0 else min(s, a))
    return resolve_num_threads(spec, available or available_cpus()), available_cpus()


def worker_plan(cfg=None, workers=None, threads=None):
    """
    (n_workers, threads_per_worker). workers: CLI value, else env SIT_FUSE_NUM_WORKERS, else
    compute.num_workers, else 1 (accepts the same forms as num_threads: int, -N, fraction, low/medium/high/max).
    threads: the per-process budget returned by apply_compute_limits (None = all available cores).
    """
    spec = workers
    if spec in (None, ""):
        spec = os.environ.get("SIT_FUSE_NUM_WORKERS")
    if spec in (None, ""):
        spec = ((cfg or {}).get("compute") or {}).get("num_workers")
    budget = int(threads) if threads else _resolve(None)[1]
    if spec in (None, "", 1, "1"):
        return 1, budget
    n, _ = _resolve(spec, budget)
    n = max(1, min(int(n or 1), budget))
    return n, max(1, budget // n)


def setup_compute(cfg=None, num_threads=None, workers=None, verbose=False):
    """
    apply_compute_limits (YAML compute: block, -n/--num-threads, SIT_FUSE_NUM_THREADS) for this process, then
    worker_plan. Returns (n_workers, threads_per_worker).
    """
    n = None
    try:
        from sit_fuse.compute_resources import apply_compute_limits
        n = apply_compute_limits(cfg, num_threads=num_threads, verbose=verbose)
    except ImportError:
        pass
    return worker_plan(cfg, workers, n)


def add_parallel_args(parser, shard=True, merge=True):
    """Standard flags: -n/--num-threads, --workers and (optionally) --shard i/N and --merge."""
    parser.add_argument("-n", "--num-threads", default=None,
                        help="CPU thread budget for this process (int, -N, fraction, low/medium/high/max); "
                             "overrides compute.num_threads")
    parser.add_argument("--workers", default=None,
                        help="in-process worker processes sharing the thread budget; overrides compute.num_workers")
    if shard:
        parser.add_argument("--shard", default=None, metavar="i/N",
                            help="process only shard i of N (0-based) and write a partial output for --merge")
    if merge:
        parser.add_argument("--merge", action="store_true", help="combine the shard outputs written by --shard")
        parser.add_argument("--n-shards", type=int, default=None,
                            help="with --merge: number of shards (default: inferred from the shard files)")
    return parser


def cpu_map(fn, items, n_jobs=1, threads=1):
    items = list(items)
    call = lambda it: fn(*it) if isinstance(it, tuple) else fn(it)  # noqa: E731
    if n_jobs is None or int(n_jobs) <= 1 or len(items) <= 1:
        return [call(it) for it in items]
    from joblib import Parallel, delayed, parallel_config
    n = min(int(n_jobs), len(items))
    with parallel_config(backend="loky", inner_max_num_threads=max(1, int(threads or 1))):
        return Parallel(n_jobs=n)(delayed(fn)(*it) if isinstance(it, tuple) else delayed(fn)(it) for it in items)


def parse_shard(spec):
    if spec in (None, "", "0/1"):
        return None
    i, n = (int(x) for x in str(spec).split("/"))
    if n < 1 or not 0 <= i < n:
        raise ValueError(f"--shard {spec!r}: expected i/N with 0 <= i < N")
    return i, n


def shard_indices(n_items, shard, mode="stride"):
    if shard is None:
        return list(range(n_items))
    i, n = shard
    if mode == "block":
        lo, hi = (n_items * i) // n, (n_items * (i + 1)) // n
        return list(range(lo, hi))
    return list(range(i, n_items, n))


def shard_path(directory, stem, shard, ext=".pkl"):
    i, n = shard
    return Path(directory) / f"{stem}.shard_{i:04d}_of_{n:04d}{ext}"


def shard_files(directory, stem, n_shards, ext=".pkl"):
    paths = [shard_path(directory, stem, (i, n_shards), ext) for i in range(n_shards)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {n_shards} shard outputs missing in {directory}: "
                                f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")
    return paths


def infer_n_shards(directory, stem, ext=".pkl"):
    """N from existing <stem>.shard_*_of_N files (the single N present), else None."""
    ns = {p.name.rsplit("_of_", 1)[1][:-len(ext)] for p in Path(directory).glob(f"{stem}.shard_*_of_*{ext}")}
    if len(ns) > 1:
        raise ValueError(f"shard files for several shard counts in {directory} ({sorted(ns)}); pass --n-shards")
    return int(ns.pop()) if ns else None


@contextlib.contextmanager
def locked(path):
    import fcntl
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".lock", "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def atomic_write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def update_json(path, merge):
    """merge(old_obj_or_None) -> new object, written atomically while holding the lock. Returns the new object."""
    with locked(path):
        old = json.loads(Path(path).read_text()) if Path(path).exists() else None
        new = merge(old)
        atomic_write_bytes(path, json.dumps(new, indent=2, default=str).encode())
    return new
