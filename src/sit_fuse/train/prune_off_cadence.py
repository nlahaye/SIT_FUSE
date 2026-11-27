"""
Free storage when a run moves to a coarser GOES cadence (e.g. 5 min -> 30 min): delete the per-scene outputs of
scenes the new cadence no longer uses, from head export onwards (head outputs, inference layers and scene
checkpoints, geo rasters, fused rasters). Dry run by default; --apply deletes.

    python prune_off_cadence.py prune_off_cadence_30min.yaml            # report only (CSV + summary)
    python prune_off_cadence.py prune_off_cadence_30min.yaml --apply    # delete, then fix manifests / links

Keep set (per target, `keep:`), the scenes the 30-min run keeps, computed with the same selection the scripts use:
  matchups: {file, splits}    UQ runs: sf_uq_utils.load_matchups(every_minutes=E) = classifier_inference_uq.py
  production: {config, sensor} production runs: build_production_run.list_inputs with sensors.<s>.every_minutes=E
  self                        fused outputs etc.: first scene time per E-min UTC bin among the files found
A file is deleted only when (1) its name carries a scene time (ABI _sYYYYJJJHHMMSS or YYYYmmddTHHMMSS), (2) that
time is a known scene of the keep source (matchups / production: any of its scenes at full cadence) and not a kept
one (within tolerance_seconds), (3) another scene of the same E-min bin is kept, and (4) it matches no exclude
pattern (TEMPO names, manifests, tables are never touched) and no protected head output (protect: heads of the
given matchup splits, e.g. dev + calibration, so the features / training can be rebuilt; symlink targets resolved).
Files without a scene time (vector files, tables, shard pickles, models) are kept.

Symlinks are unlinked, never followed (the target stays). After --apply: list entries pointing at deleted files are
dropped from geo_products_manifest.json / inference_manifest.json in the target dirs (mtimes kept, so
plan_uq_jobs.py's resume checks still see finished stages), and dangling symlinks left in the target dirs (e.g.
heads linked from a pruned run) are removed. Writes <report_dir>/prune_<name>.csv (path, bytes, target, action) and
prints a per-target summary. *.zarr directories count as one file.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN_DIR))

from uq_viz.overlay import time_from_name, utc  # noqa: E402

DEFAULT_EXCLUDE = ["*TEMPO*", "*.json", "*.csv", "*.yaml", "*.yml", "*.log", "*.txt", "*.md", "*.parquet",
                   "*.gpkg", "*.mp4", "*.gif"]


def _abs(base: Path, p) -> Path:
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else (base / p)


def _times_matchups(spec: dict, base: Path, every, off):
    from sf_uq_utils import load_matchups
    f = _abs(base, spec["file"])
    splits = spec.get("splits")
    allr = load_matchups(f, require_truth=False, splits=splits, missing_heads="keep")
    keep = load_matchups(f, require_truth=False, splits=splits, every_minutes=every, offset_minutes=off,
                         missing_heads=spec.get("missing_heads", "skip"))
    t = lambda rs: [x for x in (time_from_name(r["pair_id"]) or time_from_name(r["head_outputs"]) for r in rs) if x]
    return t(keep), t(allr)


def _times_production(spec: dict, base: Path, every, off):
    import build_production_run as B
    cpath = _abs(base, spec["config"])
    cfg = yaml.safe_load(cpath.read_text())
    cbase = cpath.resolve().parent
    s = spec.get("sensor", "goes")
    sc = dict(cfg["sensors"][s])
    t0, t1 = utc(cfg["start"]), utc(cfg["end"])
    bbox = [float(v) for v in cfg["bbox"]] if cfg.get("bbox") else None
    full, _ = B.list_inputs(dict(sc, every_minutes=None), t0, t1, bbox, cbase, s)
    kept, _ = B.list_inputs(dict(sc, every_minutes=every, offset_minutes=off), t0, t1, bbox, cbase, s)
    return [t for _, t in kept], [t for _, t in full]


def _protected(spec: dict | None, base: Path) -> tuple[set[str], set[str]]:
    """Head outputs of the protected matchup splits: (record paths, resolved files)."""
    paths, real = set(), set()
    if not spec:
        return paths, real
    from sf_uq_utils import load_matchups
    for f in spec.get("matchups") or []:
        for r in load_matchups(_abs(base, f), require_truth=False, splits=spec.get("splits"), missing_heads="keep"):
            h = Path(str(r["head_outputs"]))
            paths.add(os.path.abspath(h))
            if h.exists():
                real.add(os.path.realpath(h))
    return paths, real


def _is_protected(p: Path, protected) -> bool:
    """A record's own path is kept; a regular file is kept when a protected record resolves to it. Other symlinks
    may go: unlinking them leaves the target in place."""
    paths, real = protected
    if os.path.abspath(p) in paths:
        return True
    return not p.is_symlink() and p.exists() and os.path.realpath(p) in real


def _walk(d: Path):
    """Files (and *.zarr dirs as units) under d; symlinks are listed, not followed."""
    for root, dirs, files in os.walk(d, followlinks=False):
        z = [x for x in dirs if x.endswith(".zarr")]
        for x in z:
            yield Path(root) / x
        dirs[:] = [x for x in dirs if not x.endswith(".zarr")]
        for x in files:
            yield Path(root) / x


def _size(p: Path) -> int:
    if p.is_symlink():
        return 0
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _bin(t, every, off):
    return (t - off).floor(f"{int(every)}min")


def plan_target(tg: dict, base: Path, every, off, tol, exclude, protected):
    name = tg["name"]
    keep_spec = tg.get("keep", "self")
    dirs = [_abs(base, d) for d in (tg["dirs"] if isinstance(tg["dirs"], list) else [tg["dirs"]])]
    include = tg.get("include") or ["*"]
    excl = list(exclude) + list(tg.get("exclude") or [])
    files = []
    for d in dirs:
        if not d.is_dir():
            print(f"  {name}: {d} not found; skipped")
            continue
        for p in _walk(d):
            fn = p.name
            if not any(fnmatch.fnmatch(fn, pt) for pt in include) or any(fnmatch.fnmatch(fn, pt) for pt in excl):
                continue
            t = time_from_name(fn)
            if t is not None:
                files.append((p, t))
    if keep_spec == "self":
        times = sorted({t for _, t in files})
        first = {}
        for t in times:
            first.setdefault(_bin(t, every, off), t)
        kept, known = list(first.values()), times
    elif "matchups" in keep_spec:
        kept, known = _times_matchups(keep_spec["matchups"], base, every, off.total_seconds() / 60)
    elif "production" in keep_spec:
        kept, known = _times_production(keep_spec["production"], base, every, off.total_seconds() / 60)
    else:
        raise ValueError(f"target {name}: keep must be self, {{matchups: ...}} or {{production: ...}}")
    tol = pd.Timedelta(seconds=float(tol))
    kept_s = pd.Series(sorted(utc(t) for t in kept)) if kept else pd.Series([], dtype="datetime64[ns, UTC]")
    known_s = pd.Series(sorted(utc(t) for t in known)) if known else kept_s
    kept_bins = {_bin(t, every, off) for t in kept_s}

    def near(s, t):
        if s.empty:
            return False
        i = s.searchsorted(t)
        return any(0 <= j < len(s) and abs(s.iloc[j] - t) <= tol for j in (i - 1, i))

    rows, n_keep = [], 0
    for p, t in files:
        t = utc(t)
        why = None
        if near(kept_s, t):
            why = "kept scene"
        elif not near(known_s, t):
            why = "unknown scene"
        elif _bin(t, every, off) not in kept_bins:
            why = "no kept scene in bin"
        elif _is_protected(p, protected):
            why = "protected head"
        if why:
            n_keep += 1
            continue
        rows.append({"target": name, "path": str(p), "scene_time": t.isoformat(), "bytes": _size(p),
                     "symlink": p.is_symlink()})
    print(f"  {name}: {len(files)} timed files, {len(rows)} to delete "
          f"({sum(r['bytes'] for r in rows) / 1e9:.2f} GB), {n_keep} kept; "
          f"{len(kept_s)} kept / {len(known_s)} known scenes")
    return rows, dirs


def _drop_deleted(obj, gone: set[str], mdir: Path):
    """Lists in a manifest: drop elements that are (or contain) a deleted path."""
    def hit(v):
        if isinstance(v, str):
            if v in gone:
                return True
            if ("/" in v or v.endswith((".tif", ".pkl"))) and os.path.abspath(_abs(mdir, v)) in gone:
                return True
            return False
        if isinstance(v, dict):
            return any(hit(x) for x in v.values())
        if isinstance(v, list):
            return False
        return False
    n = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k], m = _drop_deleted(v, gone, mdir)
            n += m
    elif isinstance(obj, list):
        keep = [v for v in obj if not hit(v)]
        n += len(obj) - len(keep)
        out = []
        for v in keep:
            v2, m = _drop_deleted(v, gone, mdir)
            n += m
            out.append(v2)
        obj = out
    return obj, n


def fix_manifests(dirs, gone: set[str]):
    for d in dirs:
        if not d.is_dir():
            continue
        for m in list(d.rglob("geo_products_manifest.json")) + list(d.rglob("inference_manifest.json")):
            try:
                st = m.stat()
                doc = json.loads(m.read_text())
            except (OSError, ValueError) as e:
                print(f"  WARNING {m}: not read ({e})")
                continue
            doc, n = _drop_deleted(doc, gone, m.parent)
            if n:
                tmp = m.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(doc, indent=2, default=str))
                os.replace(tmp, m)
                os.utime(m, (st.st_atime, st.st_mtime))      # resume checks compare mtimes
                print(f"  {m}: {n} entries of deleted files dropped")


def remove_dangling(dirs) -> int:
    n = 0
    for d in dirs:
        if not d.is_dir():
            continue
        for root, ds, fs in os.walk(d, followlinks=False):
            for x in fs + ds:
                p = Path(root) / x
                if p.is_symlink() and not p.exists():
                    p.unlink()
                    n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--apply", action="store_true", help="delete (default: dry run, report only)")
    ap.add_argument("--only", nargs="+", help="target names")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text())
    base = a.config.resolve().parent
    every = float(cfg["every_minutes"])
    off = pd.Timedelta(minutes=float(cfg.get("offset_minutes", 0) or 0))
    tol = cfg.get("tolerance_seconds", 60)
    exclude = cfg.get("exclude", DEFAULT_EXCLUDE)
    name = cfg.get("name", a.config.stem)
    rdir = _abs(base, cfg.get("report_dir", "."))
    rdir.mkdir(parents=True, exist_ok=True)
    protected = _protected(cfg.get("protect"), base)
    print(f"{name}: every {every:g} min, {'APPLY' if a.apply else 'dry run'}; {len(protected[0])} protected head "
          f"outputs")
    rows, alldirs = [], []
    for tg in cfg["targets"]:
        if tg.get("enabled", True) is False or (a.only and tg["name"] not in a.only):
            continue
        r, d = plan_target(tg, base, every, off, tol, exclude, protected)
        rows += r
        alldirs += d
    total = sum(r["bytes"] for r in rows)
    print(f"total: {len(rows)} files / links, {total / 1e9:.2f} GB")
    rep = rdir / f"prune_{name}.csv"
    with rep.open("w", newline="") as fh:
        w = csv.DictWriter(fh, ["target", "path", "scene_time", "bytes", "symlink", "action"])
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, action="deleted" if a.apply else "would delete"))
    print(f"report: {rep}")
    if not a.apply:
        print("dry run: nothing deleted (rerun with --apply)")
        return 0
    gone, failed = set(), 0
    for r in rows:
        p = Path(r["path"])
        try:
            if p.is_symlink() or p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
            gone.add(os.path.abspath(p))
        except OSError as e:
            failed += 1
            print(f"  WARNING {p}: {e}")
    fix_manifests(alldirs, gone)
    n = remove_dangling(alldirs)
    print(f"deleted {len(gone)} ({total / 1e9:.2f} GB), {failed} failed, {n} dangling links removed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
