"""
Copyright [2022-26], by the California Institute of Technology and Chapman University.
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.

Generic data audit for SIT-FUSE products: regex-driven scene discovery, completeness checks and the
inputs / configs for sit_fuse.datasets.scene_split and sf_dataset (data.files_* + data.sampling).

  1. Files under inputs.roots are matched against one or more member regexes. Named groups:
       time     scene time (time_format; required on the anchor member)
       version  processing version (e.g. ABI _c creation time); the latest copy of a file wins
       slot     band / channel / granule part that must be present `count` times per scene
       prefix   HAB-style entry prefix (entry.type: prefix)
       space    optional space key (tile, granule, region)
     any other named group is part of the scene key (scene.key, default [time]).
  2. Members join the anchor scene by identical key groups, or by nearest time (join: nearest,
     tolerance_s), e.g. a GOES ABI scan paired with a TEMPO granule.
  3. A scene is complete when every required member has `count` slots, files pass the size / open
     checks and the time lies in inputs.time_window (and not in exclude_dates).
  4. Outputs in output.dir:
       <tag>_scene_index.csv        one row per scene (path, date, datetime, space, complete, issues, ...)
       <tag>_audit_summary.json     counts, incomplete reasons, duplicates, gaps, strata / unit preview
       <tag>_scene_split.yaml       ready-to-run scene_split config (table: index, filter: complete)
     With --split (or run_split: true) the split is run, writing the manifest and, when
     output.base_config is set, the derived model config for sf_dataset / training.

CLI:
  python -m sit_fuse.datasets.data_audit -y config/audit/goes18_la_fire_audit.yaml [--split]
  python -m sit_fuse.datasets.scene_split report -y <output.dir>/<tag>_scene_split.yaml

YAML:
  product: goes18_la_fire
  inputs:
    roots: [/data/GOES_LA/]
    recursive: true
    time_window: [2025-01-05, 2025-01-14]      # inclusive days (optional)
    exclude_dates: []
  members:
    abi:
      regex: 'OR_ABI-L1b-RadC-M6C01_G18_s(?P<time>\\d{13})\\d_e\\d{14}_c(?P<version>\\d{14})\\.tif$'
      time_format: '%Y%j%H%M%S'
      anchor: true
      count: 1
      required: true
      min_bytes: 1024
      roots: null                              # per-member override of inputs.roots
      join: key                                # key | nearest (non-anchor members)
      tolerance_s: 300                         # nearest join window
  scene:
    key: [time]                                # named groups forming the scene id
    day_shift_hours: 0                         # e.g. -8 to assign scenes to local (PST) days
  space: {group: space, split: true}           # or {dir_level: -1} / {value: LA} / omit; split: false keeps
                                               # the key in the index only (units and buffers by day)
  entry:
    type: anchor                               # anchor | prefix | list
    members: [abi]                             # list: order of member paths in each entry
  checks:
    open: none                                 # none | auto | netcdf | gdal | zarr
  gap_days: 1                                  # report day gaps longer than this
  strata: {time_bins: [none]}                  # passed to scene_split
  split: {block_days: 1, test_fraction: 0.2, buffer_days: 0, pin: {}, seed: 0}
  output:
    dir: /data/audit/goes18_la_fire/
    tag: goes18_la_fire
    base_config: config/model/goes_la_fire.yaml
    derived_config: config/model/goes_la_fire_split.yaml
    files_test: all
    sampling: {scene_balance: 0.5}
  run_split: false
"""
import argparse
import copy
import csv
import datetime as dt
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import yaml

from sit_fuse.datasets import scene_split as ss

RESERVED = {"time", "version", "slot", "prefix", "space"}


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _parse_time(v, fmt):
    if fmt:
        return dt.datetime.strptime(v, fmt)
    for f in ("%Y%m%dT%H%M%S", "%Y%j%H%M%S", "%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%S", "%Y%m%d", "%Y%j", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(v, f)
        except ValueError:
            continue
    raise ValueError("cannot parse time %r (set time_format)" % v)


def member_specs(conf):
    ms = conf.get("members") or {}
    if not ms:
        raise ValueError("data_audit: members is empty")
    specs = []
    for name, m in ms.items():
        m = dict(m)
        m["name"] = name
        m["rx"] = re.compile(m["regex"])
        m.setdefault("count", 1)
        m.setdefault("required", True)
        m.setdefault("join", "key")
        specs.append(m)
    anchors = [m for m in specs if m.get("anchor")]
    if not anchors:
        specs[0]["anchor"] = True
    elif len(anchors) > 1:
        raise ValueError("data_audit: only one anchor member")
    return specs


def scan_files(conf, specs):
    """Matched file records per member, plus a count of unparsable names."""
    inp = conf.get("inputs") or {}
    recursive = inp.get("recursive", True)
    recs, bad = defaultdict(list), Counter()
    for m in specs:
        roots = m.get("roots") or inp.get("roots") or []
        roots = [roots] if isinstance(roots, str) else roots
        on_path = m.get("match", "basename") == "path"
        seen = set()
        for root in roots:
            for d, dirs, files in os.walk(root):
                names = list(files) + [x for x in dirs if x.endswith(".zarr")]   # zarr stores are dirs
                for fle in names:
                    path = os.path.join(d, fle)
                    if path in seen:
                        continue
                    mt = m["rx"].search(path if on_path else fle)
                    if not mt:
                        continue
                    seen.add(path)
                    g = {k: v for k, v in mt.groupdict().items() if v is not None}
                    t = None
                    if "time" in g:
                        try:
                            t = _parse_time(g["time"], m.get("time_format"))
                        except ValueError:
                            bad[m["name"]] += 1
                            continue
                    recs[m["name"]].append(dict(path=path, dir=d, groups=g, time=t,
                                                version=g.get("version", ""), slot=g.get("slot")))
                if not recursive:
                    dirs[:] = []
                else:
                    dirs[:] = [x for x in dirs if not x.endswith(".zarr")]
    return recs, bad


def _scene_key(rec, key_groups):
    out = []
    for k in key_groups:
        if k == "time":
            out.append(rec["time"].isoformat() if rec["time"] else "")
        else:
            out.append(str(rec["groups"].get(k, "")))
    return tuple(out)


def _dedupe(recs):
    """Latest version per (key, slot); returns kept records and the number of superseded copies."""
    best, dup = {}, 0
    for r in recs:
        k = (r["_key"], r["slot"])
        if k in best:
            dup += 1
            if str(r["version"]) > str(best[k]["version"]):
                best[k] = r
        else:
            best[k] = r
    return list(best.values()), dup


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def _open_ok(path, how):
    if how in (None, "none"):
        return True
    if how == "auto":
        how = "zarr" if path.rstrip("/").endswith(".zarr") else ("netcdf" if path.endswith((".nc", ".nc4", ".h5", ".hdf")) else "gdal")
    try:
        if how == "netcdf":
            from netCDF4 import Dataset
            Dataset(path).close()
        elif how == "gdal":
            from osgeo import gdal
            if gdal.Open(path) is None:
                return False
        elif how == "zarr":
            import zarr
            zarr.open(path, mode="r")
        return True
    except Exception:
        return False


def _size(path):
    if os.path.isdir(path):
        return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
    return os.path.getsize(path)


# --------------------------------------------------------------------------- #
# Scenes
# --------------------------------------------------------------------------- #
def _space(rec, sp):
    if not sp:
        return None
    if "group" in sp:
        return rec["groups"].get(sp["group"], "__none__")
    if "dir_level" in sp:
        parts = os.path.normpath(rec["dir"]).split(os.sep)
        return parts[int(sp["dir_level"])]
    if "value" in sp:
        return str(sp["value"])
    return None


def _entry(scene, conf, anchor):
    e = conf.get("entry") or {}
    typ = e.get("type", "anchor")
    mem = scene["members"]
    if typ == "anchor":
        paths = sorted(mem.get(anchor, []))
        return paths[0] if len(paths) == 1 else paths
    if typ == "prefix":
        r = scene["_anchor_rec"]
        return os.path.join(r["dir"], r["groups"]["prefix"])
    if typ == "list":
        out = []
        for name in e.get("members") or list(mem):
            ps = sorted(mem.get(name, []))
            out.extend(ps) if len(ps) != 1 else out.append(ps[0])
        return out
    raise ValueError("entry.type must be anchor | prefix | list")


def build_scenes(conf, specs, recs):
    key_groups = (conf.get("scene") or {}).get("key", ["time"])
    shift = dt.timedelta(hours=float((conf.get("scene") or {}).get("day_shift_hours", 0) or 0))
    anchor = next(m for m in specs if m.get("anchor"))
    dups = {}
    for m in specs:
        for r in recs.get(m["name"], []):
            r["_key"] = _scene_key(r, key_groups if m is anchor or m["join"] == "key" else ["time"])
        recs[m["name"]], dups[m["name"]] = _dedupe(recs.get(m["name"], []))

    scenes = {}
    for r in recs.get(anchor["name"], []):
        if r["time"] is None:
            raise ValueError("data_audit: anchor regex needs a (?P<time>...) group")
        s = scenes.setdefault(r["_key"], dict(key=r["_key"], time=r["time"], members=defaultdict(list),
                                              slots=defaultdict(set), _anchor_rec=r, issues=[]))
        s["members"][anchor["name"]].append(r["path"])
        s["slots"][anchor["name"]].add(r["slot"] if r["slot"] is not None else r["path"])

    for m in specs:
        if m is anchor:
            continue
        rs = recs.get(m["name"], [])
        if m["join"] == "key":
            by = defaultdict(list)
            for r in rs:
                by[r["_key"]].append(r)
            for k, s in scenes.items():
                for r in by.get(k, []):
                    s["members"][m["name"]].append(r["path"])
                    s["slots"][m["name"]].add(r["slot"] if r["slot"] is not None else r["path"])
        else:   # nearest in time within tolerance_s; one time stamp (all its slots) per scene
            tol = float(m.get("tolerance_s", 300))
            by_t = defaultdict(list)
            for r in rs:
                by_t[r["time"]].append(r)
            times = np.array(sorted(by_t), dtype="datetime64[us]")
            keys = sorted(by_t)
            for s in scenes.values():
                if not len(times):
                    break
                t = np.datetime64(s["time"], "us")
                j = int(np.searchsorted(times, t))
                cand = [c for c in (j - 1, j) if 0 <= c < len(times)]
                c = min(cand, key=lambda c: abs((times[c] - t) / np.timedelta64(1, "s")))
                dts = abs((times[c] - t) / np.timedelta64(1, "s"))
                if dts <= tol:
                    for r in by_t[keys[c]]:
                        s["members"][m["name"]].append(r["path"])
                        s["slots"][m["name"]].add(r["slot"] if r["slot"] is not None else r["path"])
                    s.setdefault("join_dt_s", {})[m["name"]] = float(dts)

    inp = conf.get("inputs") or {}
    win = inp.get("time_window")
    lo, hi = (ss._parse_date(win[0]), ss._parse_date(win[1])) if win else (None, None)
    excl = {ss._parse_date(d) for d in inp.get("exclude_dates") or []}
    ck = conf.get("checks") or {}
    out = []
    for s in sorted(scenes.values(), key=lambda s: (s["time"], s["key"])):
        s["date"] = (s["time"] + shift).date()
        for m in specs:
            n = len(s["slots"].get(m["name"], ()))
            if n < int(m["count"]):
                if m["required"]:
                    s["issues"].append("missing:%s(%d/%d)" % (m["name"], n, int(m["count"])))
            elif n > int(m["count"]):
                s["issues"].append("extra:%s(%d/%d)" % (m["name"], n, int(m["count"])))
            for p in s["members"].get(m["name"], []):
                if m.get("min_bytes") and _size(p) < int(m["min_bytes"]):
                    s["issues"].append("small:%s" % m["name"])
                if not _open_ok(p, m.get("open", ck.get("open", "none"))):
                    s["issues"].append("unreadable:%s" % m["name"])
        if (lo and s["date"] < lo) or (hi and s["date"] > hi):
            s["issues"].append("outside_window")
        if s["date"] in excl:
            s["issues"].append("excluded_date")
        s["space"] = _space(s["_anchor_rec"], conf.get("space"))
        s["complete"] = not s["issues"]
        s["path"] = _entry(s, conf, anchor["name"])
        out.append(s)
    return out, dups


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def summarize(scenes, dups, bad, conf):
    comp = [s for s in scenes if s["complete"]]
    days = sorted({s["date"] for s in comp})
    gap = int(conf.get("gap_days", 1))
    gaps = [[str(a), str(b), (b - a).days - 1] for a, b in zip(days[:-1], days[1:]) if (b - a).days - 1 >= gap]
    reasons = Counter(i.split("(")[0] for s in scenes for i in s["issues"])
    per_day = Counter(str(s["date"]) for s in comp)
    out = dict(product=conf.get("product"), n_scenes=len(scenes), n_complete=len(comp),
               n_days=len(days), date_range=[str(days[0]), str(days[-1])] if days else None,
               incomplete_reasons=dict(reasons), superseded_versions=dups, unparsable_names=dict(bad),
               scenes_per_day=dict(min=min(per_day.values()) if per_day else 0, max=max(per_day.values()) if per_day else 0,
                                   median=float(np.median(list(per_day.values()))) if per_day else 0.0),
               per_space=dict(Counter(str(s["space"]) for s in comp)), gaps=gaps)
    # preview of strata / temporal units, so too-small strata show up before the split
    if comp:
        use_sp = bool(conf.get("space")) and conf["space"].get("split", True)
        prev = [dict(path=s["path"], date=s["date"], space=s["space"] if use_sp else None) for s in comp]
        ss.assign_strata(prev, conf.get("strata") or {})
        ss.assign_units(prev, (conf.get("split") or {}).get("block_days", 14))
        units = defaultdict(set)
        for p in prev:
            units[p["stratum"]].add(p["unit"])
        mu = int((conf.get("split") or {}).get("min_units_for_test", 2))
        out["strata_preview"] = {st: dict(scenes=sum(1 for p in prev if p["stratum"] == st), units=len(u),
                                          all_train=len(u) < mu) for st, u in sorted(units.items())}
    return out


def write_index(scenes, specs, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    names = [m["name"] for m in specs]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "path", "date", "datetime", "space", "complete", "issues"] + ["n_" + n for n in names])
        for s in scenes:
            p = s["path"] if isinstance(s["path"], str) else json.dumps(s["path"])
            w.writerow(["|".join(s["key"]), p, s["date"].isoformat(), s["time"].isoformat(), s["space"] or "",
                        str(bool(s["complete"])).lower(), ";".join(s["issues"])] +
                       [len(s["slots"].get(n, ())) for n in names])


def split_conf(conf, index_path):
    oc = copy.deepcopy(conf.get("output") or {})
    tag = oc.get("tag", conf.get("product", "scenes"))
    sc = {"scenes": {"table": os.path.abspath(index_path), "path_column": "path", "filter": {"complete": True}},
          "time": {"column": "date"},
          "strata": copy.deepcopy(conf.get("strata") or {"time_bins": ["season_year"]}),
          "split": copy.deepcopy(conf.get("split") or {}),
          "output": {k: v for k, v in dict(oc, tag=tag).items() if k in
                     ("dir", "tag", "base_config", "derived_config", "files_test", "set_groups", "sampling", "train_dir")}}
    if conf.get("space") and conf["space"].get("split", True):
        sc["space"] = {"column": "space"}       # units / buffers per space key
    if conf.get("report"):
        sc["report"] = copy.deepcopy(conf["report"])
    return sc


def run_audit(conf, split=None):
    """Audit, write index / summary / scene_split YAML; run the split when requested."""
    specs = member_specs(conf)
    recs, bad = scan_files(conf, specs)
    scenes, dups = build_scenes(conf, specs, recs)
    if not scenes:
        raise ValueError("data_audit: no anchor files matched (check inputs.roots and members.*.regex)")
    oc = conf.get("output") or {}
    out_dir = oc.get("dir") or "."
    tag = oc.get("tag", conf.get("product", "scenes"))
    os.makedirs(out_dir, exist_ok=True)
    index = os.path.join(out_dir, "%s_scene_index.csv" % tag)
    write_index(scenes, specs, index)
    summary = summarize(scenes, dups, bad, conf)
    with open(os.path.join(out_dir, "%s_audit_summary.json" % tag), "w") as f:
        json.dump(summary, f, indent=1)
    sconf = split_conf(conf, index)
    split_yaml = os.path.join(out_dir, "%s_scene_split.yaml" % tag)
    with open(split_yaml, "w") as f:
        f.write("# Written by sit_fuse.datasets.data_audit (%s)\n" % conf.get("product", tag))
        yaml.safe_dump(sconf, f, sort_keys=False, default_flow_style=None, width=200)
    print(json.dumps({k: summary[k] for k in ("n_scenes", "n_complete", "n_days", "date_range", "incomplete_reasons")}))
    print("WROTE", index, "\nWROTE", split_yaml)
    if (conf.get("run_split", False) if split is None else split) and summary["n_complete"]:
        ss.run_split(sconf)
    return scenes, summary, split_yaml


def main(argv=None):
    ap = argparse.ArgumentParser(description="Regex-driven scene audit -> scene_split / sf_dataset inputs")
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--split", action="store_true", help="also run scene_split (manifest + derived model config)")
    a = ap.parse_args(argv)
    with open(a.yaml) as f:
        conf = yaml.safe_load(f)
    run_audit(conf, split=True if a.split else None)


if __name__ == "__main__":
    main()
