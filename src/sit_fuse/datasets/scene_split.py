"""
Copyright [2022-26], by the California Institute of Technology and Chapman University.
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.

Balanced, spatio-temporally stratified train / val / test scene selection for SIT-FUSE.

Generic: works for any list of scenes that carry a date (column, file-name regex, zarr attribute,
or a time index plus an origin date) and, optionally, a space key (tile, region, sensor, ...).

  1. Every scene gets a stratum = time bins (year, season, month, ...) [+ space key].
  2. Within a stratum, scenes are grouped into contiguous temporal units (block_days), so
     neighbouring (autocorrelated) days are never split across train and test.
  3. A test (and optional val) fraction of units is drawn per stratum, spread over time
     (systematic sampling with a random offset), so every stratum is represented in each split.
  4. Train scenes within buffer_days of a held-out scene (same space key) are dropped.
  5. Train scenes are balanced across strata: proportional (all), equal (cap every stratum at
     the smallest), and / or capped at max_train_scenes with temporally spaced selection.
  6. data.sampling.groups = {basename: stratum} is written into the derived config, so pixel
     sampling (data.sampling.scene_balance) balances strata, and data.sampling.spatial_block /
     spatial_balance balance space within scenes.

CLI:
  python -m sit_fuse.datasets.scene_split split  -y config/split/whale_scene_split_v2.yaml
  python -m sit_fuse.datasets.scene_split report -y config/split/whale_scene_split_v2.yaml [--train_dir DIR]

`report` reads <train_dir>/train_data.indices.npy + train_data.files.json (data.sampling.cache_train)
and writes the achieved per-stratum and per-spatial-block sample counts.

YAML (see config/split/*.yaml):
  scenes:
    table: scene_index.csv     # or files: [...] / glob: "..." / from_config: model.yaml
    path_column: path
    filter: {complete: true}   # keep rows whose column equals the value (table only)
    root: null                 # prefix for relative paths
  time:
    column: date               # table column, or
    regex: "(\\d{8})"          #   regex on basename (group 1) + format: "%Y%m%d", or
    attr: date                 #   zarr attribute (.zattrs / zarr.json), or
    index_regex: "_t(\\d+)"    #   integer day index + origin: "2015-01-01" (origin_json: meta.json, origin_key: time_start)
    step_days: 1
  space:                       # optional
    column / regex / attr
  strata:
    time_bins: [year, season]  # year | season | month | quarter | week | doy_bin:N | none
    seasons: {DJF: [12, 1, 2], MAM: [3, 4, 5], JJA: [6, 7, 8], SON: [9, 10, 11]}
    include_space: true
  split:
    block_days: 14
    test_fraction: 0.2
    val_fraction: 0.0
    buffer_days: 0
    balance: proportional      # proportional | equal
    max_train_scenes: null
    min_units_for_test: 2      # strata with fewer units stay all-train (flagged in the summary)
    seed: 0
  output:
    dir: /path/split
    tag: whales
    base_config: config/model/x.yaml
    derived_config: config/model/x_split.yaml
    files_test: holdout        # holdout | all | train+holdout | val+holdout
    set_groups: true
    sampling: {}               # extra data.sampling keys merged into the derived config
"""
import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import re
from collections import Counter, OrderedDict, defaultdict

import numpy as np
import yaml

DEFAULT_SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
_TRUE = {"true", "1", "yes", "y", "t"}


PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # src/sit_fuse


def resolve_path(p):
    """Relative config paths (e.g. config/model/x.yaml) are relative to the sit_fuse package dir, so the
    command works from any directory; an existing path relative to the cwd is used as is."""
    if p is None or os.path.isabs(p) or os.path.exists(p):
        return p
    return os.path.join(PKG_DIR, p)


# --------------------------------------------------------------------------- #
# Scene discovery
# --------------------------------------------------------------------------- #
def _basename(f):
    while isinstance(f, (list, tuple)) and f:
        f = f[0]
    return os.path.basename(str(f).rstrip("/"))


def _flatten_files(files):
    out = []
    for f in files or []:
        if isinstance(f, str) and any(c in f for c in "*?["):
            out.extend(sorted(glob.glob(f)))
        else:
            out.append(f)
    return out


def _match(v, want):
    if isinstance(want, bool):
        return str(v).strip().lower() in _TRUE if want else str(v).strip().lower() not in _TRUE
    if isinstance(want, (list, tuple)):
        return str(v) in {str(w) for w in want}
    return str(v) == str(want)


def load_scenes(sc):
    """List of dicts with key 'path' (str or list) and, for tables, every column."""
    root = sc.get("root")
    rows = []
    if sc.get("table"):
        with open(sc["table"], newline="") as f:
            for r in csv.DictReader(f):
                if all(_match(r.get(k), v) for k, v in (sc.get("filter") or {}).items()):
                    rows.append(dict(r))
        pc = sc.get("path_column", "path")
        for r in rows:
            r["path"] = r[pc]
    else:
        files = []
        if sc.get("from_config"):
            with open(resolve_path(sc["from_config"])) as f:
                c = yaml.safe_load(f)
            seen = set()
            for k in ("files_train", "files_test"):
                for p in _flatten_files(c["data"].get(k)):
                    key = json.dumps(p)
                    if key not in seen:
                        seen.add(key)
                        files.append(p)
        files += _flatten_files(sc.get("files"))
        if sc.get("glob"):
            files += sorted(glob.glob(sc["glob"]))
        rows = [{"path": p} for p in files]
    if root:
        for r in rows:
            if isinstance(r["path"], str) and not os.path.isabs(r["path"]):
                r["path"] = os.path.join(root, r["path"])
    if not rows:
        raise ValueError("scene_split: no scenes found")
    return rows


def _zarr_attrs(path):
    p = path[0] if isinstance(path, (list, tuple)) else path
    for name in (".zattrs", "zarr.json"):
        fp = os.path.join(str(p), name)
        if os.path.exists(fp):
            with open(fp) as f:
                a = json.load(f)
            return a.get("attributes", a) if name == "zarr.json" else a
    return {}


def _parse_date(v, fmt=None):
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if isinstance(v, (dt.date, dt.datetime)):
        return v.date() if isinstance(v, dt.datetime) else v
    s = str(v).strip()
    if fmt:
        return dt.datetime.strptime(s, fmt).date()
    for f in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S", "%Y%j", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s[:len(dt.datetime(2000, 1, 1).strftime(f))], f).date()
        except ValueError:
            continue
    return dt.date.fromisoformat(s[:10])


def _origin(tc):
    if tc.get("origin"):
        return _parse_date(tc["origin"])
    if tc.get("origin_json"):
        with open(tc["origin_json"]) as f:
            return _parse_date(json.load(f)[tc.get("origin_key", "time_start")])
    raise ValueError("time.index_regex needs time.origin or time.origin_json")


def scene_date(row, tc):
    b = _basename(row["path"])
    if tc.get("column"):
        return _parse_date(row.get(tc["column"]), tc.get("format"))
    if tc.get("regex"):
        m = re.search(tc["regex"], b)
        return _parse_date(m.group(1) if m and m.groups() else (m.group(0) if m else None), tc.get("format"))
    if tc.get("attr"):
        return _parse_date(_zarr_attrs(row["path"]).get(tc["attr"]), tc.get("format"))
    if tc.get("index_regex"):
        m = re.search(tc["index_regex"], b)
        if not m:
            return None
        return _origin(tc) + dt.timedelta(days=int(m.group(1)) * float(tc.get("step_days", 1)))
    raise ValueError("time: set one of column / regex / attr / index_regex")


def scene_space(row, sp):
    if not sp:
        return None
    if sp.get("column"):
        return str(row.get(sp["column"]))
    if sp.get("regex"):
        m = re.search(sp["regex"], _basename(row["path"]))
        return (m.group(1) if m.groups() else m.group(0)) if m else "__none__"
    if sp.get("attr"):
        return str(_zarr_attrs(row["path"]).get(sp["attr"]))
    return None


# --------------------------------------------------------------------------- #
# Strata and temporal units
# --------------------------------------------------------------------------- #
def season_of(month, seasons):
    for name, months in seasons.items():
        if month in months:
            return name
    return "none"


def season_year(d, seasons):
    """Year a season belongs to (DJF of Dec 2016 -> 2017) so seasons are contiguous."""
    s = season_of(d.month, seasons)
    months = seasons.get(s, [])
    if months and d.month == months[0] and months[0] > months[-1]:
        return d.year + 1
    return d.year


def time_bin(d, b, seasons):
    if b in (None, "none"):
        return ""
    if b == "year":
        return str(d.year)
    if b == "season":
        return season_of(d.month, seasons)
    if b == "season_year":
        return "%d%s" % (season_year(d, seasons), season_of(d.month, seasons))
    if b == "month":
        return "%02d" % d.month
    if b == "quarter":
        return "Q%d" % ((d.month - 1) // 3 + 1)
    if b == "week":
        return "W%02d" % d.isocalendar()[1]
    if b.startswith("doy_bin"):
        n = int(b.split(":")[1]) if ":" in b else 4
        return "D%d" % (min((d.timetuple().tm_yday - 1) * n // 366, n - 1))
    raise ValueError("unknown time bin: %s" % b)


def assign_strata(scenes, strata_conf):
    bins = strata_conf.get("time_bins", ["year", "season"])
    seasons = {k: [int(m) for m in v] for k, v in (strata_conf.get("seasons") or DEFAULT_SEASONS).items()}
    use_space = strata_conf.get("include_space", True)
    for s in scenes:
        parts = [time_bin(s["date"], b, seasons) for b in bins]
        if use_space and s.get("space") is not None:
            parts.append(str(s["space"]))
        s["stratum"] = "|".join(p for p in parts if p != "") or "all"


def assign_units(scenes, block_days):
    """Contiguous temporal units within (stratum, space): same unit -> same split."""
    block_days = max(int(block_days or 1), 1)
    by = defaultdict(list)
    for i, s in enumerate(scenes):
        by[(s["stratum"], s.get("space"))].append(i)
    for (st, sp), idx in by.items():
        idx.sort(key=lambda i: scenes[i]["date"])
        t0 = scenes[idx[0]]["date"]
        for i in idx:
            scenes[i]["unit"] = "%s#%s#%d" % (st, sp, (scenes[i]["date"] - t0).days // block_days)


def spaced_pick(n, k, rng):
    """k of n ordered items, evenly spread (systematic sampling with a random offset)."""
    k = int(min(max(k, 0), n))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if k == n:
        return np.arange(n)
    step = n / k
    pos = np.floor(rng.random() * step + step * np.arange(k)).astype(np.int64)
    return np.unique(np.clip(pos, 0, n - 1))


def largest_remainder(weights, total):
    w = np.asarray(weights, dtype=np.float64)
    if total <= 0 or w.sum() <= 0:
        return np.zeros(len(w), dtype=np.int64)
    q = w / w.sum() * total
    out = np.floor(q).astype(np.int64)
    for j in np.argsort(-(q - out))[: int(total - out.sum())]:
        out[j] += 1
    return out


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def split_scenes(scenes, conf):
    rng = np.random.default_rng(int(conf.get("seed", 0)))
    test_f, val_f = float(conf.get("test_fraction", 0.2)), float(conf.get("val_fraction", 0.0))
    min_units = int(conf.get("min_units_for_test", 2))
    units = OrderedDict()
    for i, s in enumerate(scenes):
        units.setdefault(s["stratum"], OrderedDict()).setdefault(s["unit"], []).append(i)
    flags = {}
    for st, us in units.items():
        keys = sorted(us, key=lambda u: min(scenes[i]["date"] for i in us[u]))
        n = len(keys)
        role = {u: "train" for u in keys}
        if n >= min_units and (test_f > 0 or val_f > 0):
            n_test = max(1, int(round(n * test_f))) if test_f > 0 else 0
            n_val = max(1, int(round(n * val_f))) if val_f > 0 else 0
            n_hold = min(n - 1, n_test + n_val)
            hold = [keys[j] for j in spaced_pick(n, n_hold, rng)]
            n_val = min(n_val, len(hold) - (1 if n_test and len(hold) > 1 else 0))
            val = set(hold[j] for j in rng.permutation(len(hold))[:max(n_val, 0)])
            for u in hold:
                role[u] = "val" if u in val else "test"
        else:
            flags[st] = "too_few_units_all_train"
        for u in keys:
            for i in us[u]:
                scenes[i]["split"] = role[u]

    buf = int(conf.get("buffer_days", 0) or 0)
    if buf > 0:
        held = defaultdict(list)
        for s in scenes:
            if s["split"] in ("test", "val"):
                held[s.get("space")].append(s["date"].toordinal())
        for sp in held:
            held[sp] = np.sort(np.asarray(held[sp]))
        for s in scenes:
            h = held.get(s.get("space"))
            if s["split"] == "train" and h is not None and len(h):
                t = s["date"].toordinal()
                j = np.searchsorted(h, t)
                d = min(abs(t - h[min(j, len(h) - 1)]), abs(t - h[max(j - 1, 0)]))
                if d <= buf:
                    s["split"] = "buffer"

    by = OrderedDict()
    for i, s in enumerate(scenes):
        if s["split"] == "train":
            by.setdefault(s["stratum"], []).append(i)
    for st in by:
        by[st].sort(key=lambda i: scenes[i]["date"])
    counts = np.array([len(v) for v in by.values()])
    target = counts.copy()
    if conf.get("balance", "proportional") == "equal" and len(counts):
        target = np.full(len(counts), counts.min())
    cap = conf.get("max_train_scenes")
    if cap is not None and target.sum() > int(cap):
        w = target if conf.get("balance", "proportional") != "equal" else np.ones(len(target))
        alloc = largest_remainder(w, int(cap))
        # redistribute allocation that exceeds a stratum's size
        for _ in range(len(alloc)):
            over = np.maximum(alloc - target, 0)
            if over.sum() == 0:
                break
            alloc = np.minimum(alloc, target)
            room = target - alloc
            alloc += largest_remainder(room, min(int(over.sum()), int(room.sum())))
        target = np.minimum(alloc, target)
    for (st, idx), k in zip(by.items(), target):
        keep = set(spaced_pick(len(idx), k, rng).tolist())
        for j, i in enumerate(idx):
            if j not in keep:
                scenes[i]["split"] = "unused"
    return flags


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def summarize(scenes, flags):
    tab = defaultdict(Counter)
    for s in scenes:
        tab[s["stratum"]][s["split"]] += 1
    strata = {st: dict(c) for st, c in sorted(tab.items())}
    tot = Counter(s["split"] for s in scenes)
    tr = np.array([c.get("train", 0) for c in tab.values()], dtype=float)
    out = dict(n_scenes=len(scenes), totals=dict(tot), strata=strata, flags=flags,
               train_balance=dict(min=float(tr.min()) if len(tr) else 0, max=float(tr.max()) if len(tr) else 0,
                                  cv=float(tr.std() / tr.mean()) if len(tr) and tr.mean() > 0 else 0.0),
               date_range={k: [str(min(s["date"] for s in scenes if s["split"] == k)),
                               str(max(s["date"] for s in scenes if s["split"] == k))]
                           for k in tot})
    return out


def _rel(path, base):
    return path if base is None else os.path.relpath(path, os.path.dirname(os.path.abspath(base)))


def write_outputs(scenes, summary, conf):
    oc = conf.get("output", {})
    out_dir = oc.get("dir") or "."
    tag = oc.get("tag", "scenes")
    os.makedirs(out_dir, exist_ok=True)
    man = os.path.join(out_dir, "%s_scene_split.csv" % tag)
    with open(man, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "basename", "date", "space", "stratum", "unit", "split"])
        for s in sorted(scenes, key=lambda s: (s["date"], _basename(s["path"]))):
            w.writerow([json.dumps(s["path"]) if not isinstance(s["path"], str) else s["path"], _basename(s["path"]),
                        s["date"].isoformat(), s.get("space") or "", s["stratum"], s["unit"], s["split"]])
    with open(os.path.join(out_dir, "%s_scene_split_summary.json" % tag), "w") as f:
        json.dump(summary, f, indent=1)
    print("WROTE", man)

    if not oc.get("base_config"):
        return man, None
    with open(resolve_path(oc["base_config"])) as f:
        base = yaml.safe_load(f)
    ordered = sorted(scenes, key=lambda s: (s["date"], _basename(s["path"])))
    pick = lambda roles: [s["path"] for s in ordered if s["split"] in roles]
    data = base.setdefault("data", {})
    data["files_train"] = pick({"train"})
    data["files_val_scenes"] = pick({"val"})
    data["files_holdout"] = pick({"test"})
    mode = oc.get("files_test", "holdout")
    data["files_test"] = {"holdout": pick({"test"}), "all": [s["path"] for s in ordered],
                          "train+holdout": pick({"train", "test"}),
                          "val+holdout": pick({"val", "test"})}[mode]
    samp = data.setdefault("sampling", {}) or {}
    data["sampling"] = samp
    if oc.get("set_groups", True):
        samp["groups"] = {_basename(s["path"]): s["stratum"] for s in ordered if s["split"] == "train"}
    samp.update(oc.get("sampling") or {})
    base.setdefault("scene_split", {}).update(dict(manifest=os.path.abspath(man), seed=conf.get("split", {}).get("seed", 0),
                                                   files_test=mode, totals=summary["totals"]))
    dst = resolve_path(oc.get("derived_config") or os.path.splitext(oc["base_config"])[0] + "_split.yaml")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w") as f:
        f.write("# Derived by sit_fuse.datasets.scene_split from %s; do not edit the file lists by hand.\n" % oc["base_config"])
        yaml.safe_dump(base, f, sort_keys=False, default_flow_style=None, width=200)
    print("WROTE", dst)
    return man, dst


def run_split(conf):
    scenes = load_scenes(conf.get("scenes", {}))
    tc, sp = conf.get("time", {}), conf.get("space")
    keep = []
    for r in scenes:
        d = scene_date(r, tc)
        if d is None:
            print("SKIP (no date):", r["path"])
            continue
        r["date"], r["space"] = d, scene_space(r, sp)
        keep.append(r)
    assign_strata(keep, conf.get("strata", {}))
    assign_units(keep, conf.get("split", {}).get("block_days", 14))
    flags = split_scenes(keep, conf.get("split", {}))
    summary = summarize(keep, flags)
    print(json.dumps(dict(totals=summary["totals"], train_balance=summary["train_balance"], flags=flags)))
    write_outputs(keep, summary, conf)
    return keep, summary


# --------------------------------------------------------------------------- #
# Report on the achieved training sample
# --------------------------------------------------------------------------- #
def report(conf, train_dir=None, spatial_block=None):
    oc = conf.get("output", {})
    train_dir = train_dir or oc.get("train_dir")
    with open(os.path.join(train_dir, "train_data.files.json")) as f:
        meta = json.load(f)
    T = np.load(os.path.join(train_dir, "train_data.indices.npy"), mmap_mode="r")
    files = meta["files"]
    sidx = meta.get("scene_index") or [[i, 0] for i in range(len(files))]
    man = os.path.join(oc.get("dir") or ".", "%s_scene_split.csv" % oc.get("tag", "scenes"))
    stratum = {}
    if os.path.exists(man):
        with open(man) as f:
            for r in csv.DictReader(f):
                stratum[r["basename"]] = r["stratum"]
    r_col = np.asarray(T[:, 0]).astype(np.int64)
    n_r = np.bincount(r_col, minlength=len(sidx))
    per_scene, per_stratum = Counter(), Counter()
    for r, n in enumerate(n_r):
        if n == 0 or r >= len(sidx):
            continue
        b = _basename(files[int(sidx[r][0])])
        per_scene[b] += int(n)
        per_stratum[stratum.get(b, "__unknown__")] += int(n)
    out = dict(n_samples=int(len(T)), n_scenes_sampled=len(per_scene), per_stratum=dict(sorted(per_stratum.items())),
               per_scene_cv=float(np.std(list(per_scene.values())) / max(np.mean(list(per_scene.values())), 1e-9)))
    spatial_block = spatial_block or conf.get("report", {}).get("spatial_block")
    if spatial_block and T.shape[1] >= 3:
        from sit_fuse.datasets.scene_sampling import block_shape
        by, bx = block_shape(spatial_block)
        p = int(meta.get("pixel_padding", 0))
        rows, cols = np.asarray(T[:, 1]) - p, np.asarray(T[:, 2]) - p
        blk = Counter(zip((rows // by).tolist(), (cols // bx).tolist()))
        v = np.array(list(blk.values()), dtype=float)
        out["spatial_blocks"] = dict(n_blocks=len(blk), min=float(v.min()), max=float(v.max()),
                                     cv=float(v.std() / v.mean()))
    dst = os.path.join(train_dir, "train_sample_report.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_stratum"}))
    print("WROTE", dst)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["split", "report"])
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--train_dir", default=None, help="report: SIT-FUSE save dir holding train_data.npy (default output.train_dir)")
    ap.add_argument("--spatial_block", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="override split.seed")
    a = ap.parse_args(argv)
    with open(a.yaml) as f:
        conf = yaml.safe_load(f)
    if a.seed is not None:
        conf.setdefault("split", {})["seed"] = a.seed
    if a.cmd == "split":
        run_split(conf)
    else:
        report(conf, a.train_dir, a.spatial_block)


if __name__ == "__main__":
    main()
