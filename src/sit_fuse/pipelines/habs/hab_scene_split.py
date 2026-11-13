"""
Copyright [2022-26], by the California Institute of Technology and Chapman University.
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.

Spatio-temporally stratified scene split (sit_fuse.datasets.scene_split) for the HAB pipelines.

HAB scenes are daily L3m grids over one ROI, one per sensor and day, so the split is made once over
calendar days and shared by every sensor. The same held-out days are then used by:

  training          data.files_train / files_val_scenes / files_holdout / files_test and
                    data.sampling.groups ({basename: stratum}) in train_model_<inst>[_trop].yaml
  context assign.   in situ matchups restricted to `context_splits` days (default: train)
  validation        in situ matchups restricted to `validation_splits` days (default: test)
  UQ                uq.splits: test = held-out days, calibration = val days (or a seeded share of
                    train days), buffer / unused days excluded

The day-level manifest (<output.dir>/<output.tag>_scene_split.csv) is reused when it exists, so
training, post-inference and validation-only runs see the identical split. Set `regenerate: true`
to rebuild it.

Pipeline YAML:
  scene_split:
    enabled: true
    manifest: null              # existing day-level manifest; default <output.dir>/<output.tag>_scene_split.csv
    regenerate: false
    dates_from: auto            # auto | inputs (instruments.*.input_oc_dir) | outputs (instruments.*.out_dir)
    date_range: null            # [YYYY-MM-DD, YYYY-MM-DD]; default start_date / end_date when present
    time: {regex: "(?<!\\d)(\\d{8})(?!\\d)", format: "%Y%m%d"}    # date in file names (dates_from)
    strata: {time_bins: [season_year]}
    split: {block_days: 14, test_fraction: 0.2, val_fraction: 0.0, buffer_days: 3, seed: 0}
    output: {dir: /path/scene_split, tag: hab}
    files_test: rest            # rest (all non-train; inference also writes files_train) | holdout | all
    sampling: {}                # data.sampling keys for training, e.g. {scene_balance: 0.5}
    context_splits: [train]
    validation_splits: [test]
    uq: true                    # drive uq.splits from the day split
"""
import copy
import csv
import datetime as dt
import os
import re

from sit_fuse.datasets import scene_split as ss

DATE_RE = r"(?<!\d)(\d{8})(?!\d)"
GOES_DATE_RE = (r"_s(\d{7})", "%Y%j")


def scene_split_conf(yml_conf):
    """The pipeline's scene_split block, or None when absent / disabled."""
    c = (yml_conf or {}).get("scene_split")
    if not c or not c.get("enabled", True):
        return None
    return c


def _day(v):
    return ss._parse_date(v)


def _date_range(yml_conf, c):
    rng = c.get("date_range")
    if rng:
        return _day(rng[0]), _day(rng[1])
    if yml_conf.get("start_date") and yml_conf.get("end_date"):
        return _day(yml_conf["start_date"]), _day(yml_conf["end_date"])
    return None, None


def _manifest_path(c):
    if c.get("manifest"):
        return c["manifest"]
    oc = c.get("output") or {}
    return os.path.join(oc.get("dir") or ".", "%s_scene_split.csv" % oc.get("tag", "hab"))


def _time_pattern(c, instrument):
    tc = c.get("time") or {}
    over = (c.get("time_overrides") or {}).get(instrument)
    if over:
        return over["regex"], over.get("format")
    if "GOES" in str(instrument).upper() and not tc.get("regex"):
        return GOES_DATE_RE
    return tc.get("regex", DATE_RE), tc.get("format", "%Y%m%d")


def collect_dates(yml_conf, c=None):
    """Calendar days with data for any instrument: training inputs (input_oc_dir + input_oc_re),
    or inference outputs (out_dir) for post-inference configs."""
    c = c or scene_split_conf(yml_conf) or {}
    src = c.get("dates_from", "auto")
    lo, hi = _date_range(yml_conf, c)
    days = set()
    for inst, keys in (yml_conf.get("instruments") or {}).items():
        for key, kc in (keys or {}).items():
            if not isinstance(kc, dict):
                continue
            use_inputs = src == "inputs" or (src == "auto" and kc.get("input_oc_dir"))
            fdir = kc.get("input_oc_dir") if use_inputs else kc.get("out_dir")
            if not fdir or not os.path.isdir(fdir):
                continue
            if use_inputs and kc.get("input_oc_re"):
                pat = re.compile(str(kc["input_oc_re"]))
                fmt = "%Y%j%H%M%S" if "GOES" in inst else "%Y%m%d"
                grp = 2
            else:
                rx, fmt = _time_pattern(c, inst)
                pat, grp = re.compile(rx), 1
            for root, dirs, files in os.walk(fdir):
                for fle in list(files) + list(dirs):      # zarr outputs are directories
                    m = pat.search(fle)
                    if not m:
                        continue
                    try:
                        d = dt.datetime.strptime(m.group(grp), fmt).date()
                    except ValueError:
                        continue
                    if (lo is None or d >= lo) and (hi is None or d <= hi):
                        days.add(d)
    return sorted(days)


def read_manifest(path):
    """{YYYYMMDD: {"split", "stratum"}} from a day-level (or scene-level) scene_split manifest."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = _day(r["date"]).strftime("%Y%m%d")
            prev = out.get(d)
            # scene-level manifests: a day is held out if any of its scenes is
            if prev is None or prev["split"] == "train":
                out[d] = {"split": r["split"], "stratum": r["stratum"]}
    return out


def build_day_split(yml_conf, c=None, days=None):
    """Run sit_fuse.datasets.scene_split over calendar days and write the day-level manifest."""
    c = c or scene_split_conf(yml_conf)
    days = days if days is not None else collect_dates(yml_conf, c)
    if not days:
        raise ValueError("scene_split: no dated HAB scenes found (check dates_from / input_oc_dir / out_dir)")
    oc = dict(c.get("output") or {})
    oc.setdefault("tag", "hab")
    if c.get("manifest"):
        oc["dir"] = os.path.dirname(os.path.abspath(c["manifest"]))
        oc["tag"] = os.path.basename(c["manifest"]).replace("_scene_split.csv", "")
    oc.pop("base_config", None)          # file lists are applied per instrument below
    conf = {"scenes": {"files": [d.strftime("%Y%m%d") for d in days]},
            "time": {"regex": r"(\d{8})", "format": "%Y%m%d"},
            "strata": copy.deepcopy(c.get("strata") or {"time_bins": ["season_year"]}),
            "split": copy.deepcopy(c.get("split") or {}),
            "output": oc}
    conf["strata"].setdefault("include_space", False)
    ss.run_split(conf)
    return _manifest_path(c) if c.get("manifest") else os.path.join(oc.get("dir") or ".", "%s_scene_split.csv" % oc["tag"])


def day_roles(yml_conf):
    """{YYYYMMDD: {"split", "stratum"}} for the configured split (cached on yml_conf), or None."""
    c = scene_split_conf(yml_conf)
    if c is None:
        return None
    if "_scene_split_roles" in yml_conf:
        return yml_conf["_scene_split_roles"]
    man = _manifest_path(c)
    if c.get("regenerate", False) or not os.path.exists(man):
        man = build_day_split(yml_conf, c)
    roles = read_manifest(man)
    yml_conf["_scene_split_roles"] = roles
    yml_conf["_scene_split_manifest"] = os.path.abspath(man)
    print("scene_split:", os.path.abspath(man), {k: sum(1 for v in roles.values() if v["split"] == k)
                                                for k in sorted({v["split"] for v in roles.values()})})
    return roles


def dates_for(yml_conf, splits):
    """Sorted YYYYMMDD days whose split is in `splits`, or None when no split is configured."""
    roles = day_roles(yml_conf)
    if roles is None:
        return None
    want = set(splits)
    return sorted(d for d, r in roles.items() if r["split"] in want)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def apply_training_split(yml_conf, config_dict, entries, entry_dates):
    """
    Assign complete HAB training entries (path prefix or [prefix, trop_fname]) to the day split and
    write data.files_* / data.sampling / scene_split into the model config (as scene_split.write_outputs).
    entry_dates: {basename: datetime.date}. Days missing from the manifest go to files_test only.
    """
    c = scene_split_conf(yml_conf)
    roles = day_roles(yml_conf)
    order = sorted(entries, key=lambda e: (entry_dates[ss._basename(e)], ss._basename(e)))
    split_of = {}
    for e in order:
        r = roles.get(entry_dates[ss._basename(e)].strftime("%Y%m%d"))
        split_of[ss._basename(e)] = (r["split"], r["stratum"]) if r else ("unassigned", None)
    pick = lambda want: [e for e in order if split_of[ss._basename(e)][0] in want]

    data = config_dict.setdefault("data", {})
    data["files_train"] = pick({"train"})
    data["files_val_scenes"] = pick({"val"})
    data["files_holdout"] = pick({"test"})
    mode = c.get("files_test", "rest")
    data["files_test"] = {"rest": [e for e in order if split_of[ss._basename(e)][0] != "train"],
                          "holdout": data["files_holdout"],
                          "all": list(order)}[mode]
    samp = data.get("sampling") or {}
    samp.update(copy.deepcopy(c.get("sampling") or {}))
    if c.get("set_groups", True):
        samp["groups"] = {ss._basename(e): split_of[ss._basename(e)][1] for e in data["files_train"]}
    data["sampling"] = samp
    totals = {}
    for s, _ in split_of.values():
        totals[s] = totals.get(s, 0) + 1
    config_dict["scene_split"] = {"manifest": yml_conf.get("_scene_split_manifest"),
                                  "seed": (c.get("split") or {}).get("seed", 0),
                                  "files_test": mode, "totals": totals}
    print("scene_split training lists:", totals)
    return config_dict


# --------------------------------------------------------------------------- #
# UQ
# --------------------------------------------------------------------------- #
def uq_date_roles(yml_conf):
    """{YYYYMMDD: train | calibration | test | excluded} for uq.splits, or None."""
    c = scene_split_conf(yml_conf)
    if c is None or not c.get("uq", True):
        return None
    m = {"train": "train", "val": "calibration", "test": "test"}
    return {d: m.get(r["split"], "excluded") for d, r in day_roles(yml_conf).items()}
