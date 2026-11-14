"""
Copyright [2022-26], by the California Institute of Technology and Chapman University.
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.

CPU / thread throttling for SIT-FUSE.

Most of SIT-FUSE's CPU saturation does not come from explicit worker pools, but
from implicit, library-level thread pools that default to "all cores":

    * PyTorch intra-op (OpenMP) and inter-op pools
    * BLAS / LAPACK / OpenMP pools behind numpy, scipy, sklearn (OpenBLAS, MKL, libgomp)
    * dask's threaded scheduler (e.g. da.to_zarr, xarray .compute())
    * OpenCV, GDAL, numba

This module resolves a single user-facing setting into a thread count and
applies it to all of those pools, optionally also pinning the process to a
CPU subset (hard cap) and lowering its scheduling priority.

YAML (all keys optional; omitting the section keeps the old behaviour):

    compute:
      num_threads: medium     # null | int | negative int (= all but N) |
                              # float fraction in (0, 1] | low | medium | high | max
      num_threads_scope: total  # total = split num_threads across DDP ranks | per_process
      interop_threads: null   # torch inter-op pool; default = min(num_threads, 4)
      pin_cpus: false         # false | true (N cores, offset by LOCAL_RANK) | [explicit core ids]
      nice: 0                 # 0-19, higher = lower priority (POSIX only)
      dask_scheduler: threads # threads | synchronous | processes
      use_gpu: auto           # auto | true | false
      inference_batch_size: null  # override generate_output's default (10000, tiled: 5)

Overrides (highest priority first): explicit function args / CLI flag
(--num-threads), env var SIT_FUSE_NUM_THREADS, YAML compute.num_threads.

NOTE: this module deliberately imports nothing heavy at module level so that
`limit_env_early()` can run *before* numpy / torch are imported.
"""

import os
import sys

PRESETS = {
    "low": 0.25,
    "med": 0.5,
    "medium": 0.5,
    "high": 0.75,
    "max": 1.0,
    "all": 1.0,
}

# Env vars read by native thread pools when the library is first loaded.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "NUMBA_NUM_THREADS",
    "GDAL_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",   # joblib/sklearn n_jobs=-1 (e.g. RandomForest) resolves to this
)

_STATE = {"applied": None, "threadpool_limiter": None}


def available_cpus():
    """CPUs this process may actually use (affinity mask / SLURM cpuset / cgroup quota)."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1

    # cgroup v2 CPU quota (containers, k8s, some HPC schedulers)
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(int(quota) // int(period))))
    except (OSError, ValueError):
        pass

    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        n = min(n, int(slurm))
    return max(1, n)


def resolve_num_threads(spec, available=None):
    """Turn a user spec into an int thread count, or None (= no limit)."""
    if available is None:
        available = available_cpus()
    if spec is None:
        return None
    if isinstance(spec, bool):
        raise ValueError("compute.num_threads must not be a boolean")

    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in ("", "none", "null", "auto", "unlimited"):
            return None
        if s in PRESETS:
            spec = PRESETS[s]
        else:
            try:
                spec = float(s) if "." in s else int(s)
            except ValueError:
                raise ValueError(
                    "Unrecognized compute.num_threads %r; use an int, a fraction, "
                    "or one of %s" % (spec, sorted(PRESETS)))

    if isinstance(spec, float):
        if not 0.0 < spec <= 1.0:
            raise ValueError("Fractional compute.num_threads must be in (0, 1]")
        n = int(round(spec * available))
    else:
        n = int(spec)
        if n == 0:
            return None
        if n < 0:  # all but |n|
            n = available + n
    return max(1, min(n, available))


def _set_env(n):
    for var in _THREAD_ENV_VARS:
        os.environ[var] = str(n)


def limit_env_early(num_threads=None, yaml_path=None):
    """
    Set thread env vars before numpy/torch are imported (the only way some
    BLAS builds can be capped at pool-creation time). Safe no-op if nothing
    is configured. Uses only the stdlib + PyYAML.
    """
    spec = num_threads
    if spec is None:
        spec = os.environ.get("SIT_FUSE_NUM_THREADS")
    if spec is None and yaml_path is not None and os.path.exists(yaml_path):
        try:
            import yaml
            with open(yaml_path) as f:
                conf = yaml.safe_load(f) or {}
            spec = (conf.get("compute") or {}).get("num_threads")
        except Exception:
            spec = None
    try:
        n = resolve_num_threads(spec)
    except ValueError:
        return None
    if n is not None:
        _set_env(n)
    return n


def limit_env_from_argv(argv=None):
    """Early hook for CLI scripts: reads -y/--yaml and --num-threads from argv."""
    argv = sys.argv[1:] if argv is None else argv
    yaml_path, nt = None, None
    for i, a in enumerate(argv):
        if a in ("-y", "--yaml") and i + 1 < len(argv):
            yaml_path = argv[i + 1]
        elif a.startswith("--yaml="):
            yaml_path = a.split("=", 1)[1]
        elif a in ("-n", "--num-threads") and i + 1 < len(argv):
            nt = argv[i + 1]
        elif a.startswith("--num-threads="):
            nt = a.split("=", 1)[1]
    return limit_env_early(nt, yaml_path)


def add_num_threads_arg(parser):
    """Standard -n/--num-threads CLI flag for SIT-FUSE entry points."""
    parser.add_argument("-n", "--num-threads", default=None,
        help="CPU thread cap: int, negative int (all but N), fraction (0-1], or "
             "low/medium/high/max. Overrides compute.num_threads in the YAML.")
    return parser


def apply_num_threads_arg(args):
    """Make --num-threads the highest-priority override (inherited by subprocesses / DDP ranks)."""
    nt = getattr(args, "num_threads", None)
    if nt is not None:
        os.environ["SIT_FUSE_NUM_THREADS"] = str(nt)
    return nt


def merge_compute_conf(src_conf, dst_conf):
    """
    Copy src_conf's compute: section into dst_conf (e.g. a pipeline YAML into
    a generated model YAML). Keys already in dst_conf["compute"] are kept
    unless src_conf sets them too (src wins). Returns dst_conf.
    """
    if dst_conf is None:
        return dst_conf
    merged = dict(dst_conf.get("compute") or {})
    merged.update((src_conf or {}).get("compute") or {})
    if merged:
        dst_conf["compute"] = merged
    return dst_conf


def get_compute_conf(yml_conf):
    if not yml_conf:
        return {}
    return yml_conf.get("compute") or {}


def resolve_use_gpu(yml_conf):
    import torch
    val = get_compute_conf(yml_conf).get("use_gpu", "auto")
    if isinstance(val, str) and val.strip().lower() == "auto":
        return torch.cuda.is_available()
    use = bool(val)
    if use and not torch.cuda.is_available():
        print("SIT-FUSE compute: use_gpu=true but CUDA is unavailable; falling back to CPU")
        return False
    return use


def num_local_processes(devices, accelerator="auto"):
    """Number of processes Lightning will launch on this node for `devices`."""
    if isinstance(devices, (list, tuple)):
        return max(1, len(devices))
    if isinstance(devices, str):
        d = devices.strip().lower()
        if d.lstrip("-").isdigit():
            devices = int(d)
        elif "," in d:
            return max(1, len([x for x in d.split(",") if x.strip()]))
        else:
            devices = -1  # "auto"
    try:
        devices = int(devices)
    except (TypeError, ValueError):
        return 1
    if devices > 0:
        return devices
    acc = str(accelerator).lower()
    if acc in ("gpu", "cuda", "auto"):
        try:
            import torch
            if torch.cuda.is_available():
                return max(1, torch.cuda.device_count())
        except ImportError:
            pass
    return 1


def apply_compute_limits(yml_conf=None, num_threads=None, num_processes=1,
                         default_num_threads=None, verbose=True):
    """
    Apply thread / CPU limits at runtime. Idempotent; call once per process
    (prep_model() does this automatically). Returns the resolved per-process
    thread count, or None if no limit was requested.

    num_processes: processes sharing the budget on this node (e.g. DDP ranks).
        With compute.num_threads_scope == "total" (default) the budget is split
        evenly across them; with "per_process" each process gets the full value.
    default_num_threads: used when nothing is configured anywhere (lets callers
        keep a legacy hard-coded value as the fallback).
    """
    cconf = get_compute_conf(yml_conf)
    spec = num_threads
    if spec is None:
        spec = os.environ.get("SIT_FUSE_NUM_THREADS")
    if spec is None:
        spec = cconf.get("num_threads")
    if spec is None:
        spec = default_num_threads

    avail = available_cpus()
    n = resolve_num_threads(spec, avail)
    num_processes = max(1, int(num_processes or 1))
    scope = str(cconf.get("num_threads_scope", "total")).lower()
    if n is not None and num_processes > 1 and scope == "total":
        n = max(1, n // num_processes)

    key = (n, repr(cconf.get("pin_cpus")), cconf.get("nice"), cconf.get("dask_scheduler"))
    if _STATE["applied"] == key:
        return n

    # Priority (soft): applies even without a thread cap.
    nice_inc = int(cconf.get("nice", 0) or 0)
    if nice_inc > 0 and _STATE["applied"] is None and hasattr(os, "nice"):
        try:
            os.nice(min(nice_inc, 19))
        except OSError:
            pass

    if n is None:
        _STATE["applied"] = key
        return None

    _set_env(n)  # for libraries loaded later and for subprocesses

    # CPU affinity (hard cap; Linux). Off by default: with `true`, concurrent
    # jobs on the same node would all land on the same first-N cores.
    pin = cconf.get("pin_cpus", False)
    if pin and hasattr(os, "sched_setaffinity"):
        try:
            allowed = sorted(os.sched_getaffinity(0))
            if isinstance(pin, (list, tuple)):
                cores = [c for c in pin if c in allowed]
            else:
                # Offset by local rank so DDP ranks don't stack on the same cores.
                rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
                start = (rank * n) % len(allowed)
                cores = (allowed[start:] + allowed[:start])[:n]
            if cores:
                os.sched_setaffinity(0, cores)
                n = min(n, len(cores))
        except OSError as e:
            print("SIT-FUSE compute: could not set CPU affinity:", e)

    # BLAS / OpenMP pools already loaded (numpy, scipy, sklearn, torch's libgomp)
    try:
        from threadpoolctl import threadpool_limits
        _STATE["threadpool_limiter"] = threadpool_limits(limits=n)
    except Exception:
        pass

    # PyTorch
    try:
        import torch
        torch.set_num_threads(n)
        interop = cconf.get("interop_threads")
        interop = min(n, 4) if interop is None else max(1, min(int(interop), n))
        try:
            torch.set_num_interop_threads(interop)
        except RuntimeError:
            pass  # can only be set once, before inter-op work starts
    except ImportError:
        pass

    # dask (da.to_zarr, xarray .compute(), dask="parallelized" ufuncs)
    try:
        import dask
        sched = cconf.get("dask_scheduler", "threads") or "threads"
        dask.config.set(scheduler=sched, num_workers=n)
    except ImportError:
        pass

    # OpenCV, GDAL, numba (only touch them if already imported / cheap)
    if "cv2" in sys.modules:
        try:
            sys.modules["cv2"].setNumThreads(n)
        except Exception:
            pass
    try:
        from osgeo import gdal
        gdal.SetConfigOption("GDAL_NUM_THREADS", str(n))
    except Exception:
        pass
    if "numba" in sys.modules:
        try:
            sys.modules["numba"].set_num_threads(n)
        except Exception:
            pass

    _STATE["applied"] = key
    if verbose:
        print("SIT-FUSE compute: limiting to %d of %d available CPU threads%s%s"
              % (n, avail, " per process (x%d processes)" % num_processes if num_processes > 1 else "",
                 " (pinned)" if pin else ""))
    return n
