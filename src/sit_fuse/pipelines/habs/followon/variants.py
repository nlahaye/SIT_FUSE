"""Derived HAB configurations for follow-on experiments (one YAML per variant, one launcher job each).

Subcommands
  hab     HAB pipeline YAML per variant from the production YAML + a variant spec
  train   SIT-FUSE training YAML (hab_training_pipeline) from a training base YAML + spec
  model   model YAML derived from an existing one (file-list filter / rewrite, out_dir, reader_kwargs)

hab spec (YAML):
  base_yaml: /path/hab_reflectance_geotiff_gen_s_ca_2023_2026_full_CA.yaml
  out_root: /mnt/data1/HAB_COMBINED_RUNS/MERGED_TEST_111526/followon/ancillary
  variants:
    baseline: {reuse_out_dir: true}             # production uq.out_dir (no retraining needed)
    anc_all:  {features_add: [ancillary]}
    phys_only: {features_set: [physical, ancillary]}
    olci_only: {sensors: [s3a, s3b]}
    pace_1km:
      sensors: [pace]
      overrides: {instruments: {pace: {no_trop: {out_dir: ...}}}, head_export: {georef: {...}}}
  common: {...}                                 # optional: merged into every variant (variant keys win)
  Each variant gets   uq.out_dir = <out_root>/<name>/uq   (unless reuse_out_dir)
                      config_dir = <out_root>/<name>/config  (model YAMLs written by training stay isolated)
  Splits are untouched (uq.date_roles_file, splits.*), so test samples are shared across
  variants and paired comparisons (variant_compare) are valid.

train spec (YAML):
  base_yaml: /path/hab_reflectance_model_train_s_ca.yaml
  out: /path/train_OLCI_S3A.yaml
  output: /mnt/data1/HAB_MODEL_WEIGHTS/FOLLOWON/OLCI_S3A/     # model weights root
  config_dir: /mnt/.../followon/same_instrument/OLCI_S3A/config/
  instruments: [OLCI]                      # keep only these instrument keys
  platforms: {OLCI: [S3A]}                 # keep input sources whose dir / regex has the token
  set: {train_min_date: "20240301", instruments.OLCI.no_trop.run_preprocess_and_scale: true}

  python -m sit_fuse.pipelines.habs.followon.variants hab -s spec.yaml [--only a,b]
  python -m sit_fuse.pipelines.habs.followon.variants train -s train_spec.yaml
  python -m sit_fuse.pipelines.habs.followon.variants model --src train_model_OLCI.yaml --out m.yaml \
      --keep 'S3B_OLCI' --out-dir /weights/OLCI_S3A/ --set data.reader_kwargs.nrt=true
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


def read_yaml(p) -> dict:
    with open(p) as f:
        return yaml.safe_load(f)


def write_yaml(obj, p) -> str:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
    os.replace(tmp, p)
    return str(p)


def deep_merge(base: dict, over: dict | None) -> dict:
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _parse_value(v: str):
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def set_dotted(d: dict, key: str, value) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


# ----------------------------------------------------------------------------- HAB variants
def hab_variant(base: dict, name: str, spec: dict, out_root: str) -> dict:
    y = copy.deepcopy(base)
    uq = y.setdefault("uq", {})
    root = Path(out_root) / name
    if not spec.get("reuse_out_dir", False):
        uq["out_dir"] = str(root / "uq")
        y["config_dir"] = str(root / "config") + "/"
    feats = uq.setdefault("features", {})
    groups = list(feats.get("groups") or [])
    if "features_set" in spec:
        groups = list(spec["features_set"])
    for g in spec.get("features_add") or []:
        if g not in groups:
            groups.append(g)
    groups = [g for g in groups if g not in set(spec.get("features_drop") or [])]
    if groups:
        feats["groups"] = groups
    if spec.get("sensors"):
        uq["sensors"] = list(spec["sensors"])
    if spec.get("species_run"):
        uq["species_run"] = list(spec["species_run"])
    deep_merge(y, spec.get("overrides"))
    for k, v in (spec.get("set") or {}).items():
        set_dotted(y, k, v)
    y.setdefault("followon", {})["variant"] = {"name": name, "spec": spec}
    return y


def run_hab(spec_path: str, only: list[str] | None = None) -> dict[str, str]:
    spec = read_yaml(spec_path)
    base = read_yaml(spec["base_yaml"])
    out = {}
    common = spec.get("common") or {}
    for name, vs in (spec.get("variants") or {}).items():
        if only and name not in only:
            continue
        vs = deep_merge(copy.deepcopy(common), vs or {})      # variant keys win over common
        y = hab_variant(base, name, vs, spec["out_root"])
        p = Path(spec["out_root"]) / name / f"hab_{name}.yaml"
        out[name] = write_yaml(y, p)
        Path(y["uq"]["out_dir"]).mkdir(parents=True, exist_ok=True)
    write_yaml({"variants": out}, Path(spec["out_root"]) / "variants_index.yaml")
    return out


# ----------------------------------------------------------------------------- training YAMLs
def _filter_sources(entry: dict, tokens: list[str]) -> dict:
    """Keep input_oc_dir / input_oc_re sources containing any platform token (lists stay aligned)."""
    dirs, res = entry.get("input_oc_dir"), entry.get("input_oc_re")
    if not isinstance(dirs, list):
        return entry
    res_l = res if isinstance(res, list) else [res] * len(dirs)
    keep = [i for i, (d, r) in enumerate(zip(dirs, res_l)) if any(t in str(d) or t in str(r) for t in tokens)]
    if not keep:
        raise ValueError(f"no input source matches {tokens}: {dirs}")
    e = dict(entry)
    e["input_oc_dir"] = [dirs[i] for i in keep] if len(keep) > 1 else dirs[keep[0]]
    e["input_oc_re"] = [res_l[i] for i in keep] if len(keep) > 1 else res_l[keep[0]]
    if "input_dir" in e and isinstance(e["input_dir"], list):
        e["input_dir"] = [e["input_dir"][i] for i in keep]
    return e


def train_variant(base: dict, spec: dict) -> dict:
    y = copy.deepcopy(base)
    if spec.get("instruments"):
        y["instruments"] = {k: v for k, v in y["instruments"].items() if k in set(spec["instruments"])}
        missing = set(spec["instruments"]) - set(y["instruments"])
        if missing:
            raise KeyError(f"instruments not in base training YAML: {sorted(missing)}")
    for inst, toks in (spec.get("platforms") or {}).items():
        for key, entry in y["instruments"][inst].items():
            y["instruments"][inst][key] = _filter_sources(entry, list(toks))
    for k in ("output", "config_dir", "train_min_date", "train_max_date"):
        if k in spec:
            y[k] = spec[k]
    deep_merge(y, spec.get("overrides"))
    for k, v in (spec.get("set") or {}).items():
        set_dotted(y, k, v)
    if spec.get("config_dir"):
        Path(spec["config_dir"], "model").mkdir(parents=True, exist_ok=True)
    return y


def run_train(spec_path: str) -> str:
    spec = read_yaml(spec_path)
    return write_yaml(train_variant(read_yaml(spec["base_yaml"]), spec), spec["out"])


# ----------------------------------------------------------------------------- model YAMLs
def _entry_str(e) -> str:
    return " ".join(map(str, e)) if isinstance(e, (list, tuple)) else str(e)


def _rewrite(e, subs):
    if isinstance(e, (list, tuple)):
        return [_rewrite(x, subs) for x in e]
    for a, b in subs:
        e = re.sub(a, b, str(e))
    return e


def model_variant(src: dict, keep: str | None = None, drop: str | None = None, subs=(), out_dir: str | None = None,
                  sets: dict | None = None, extra_lists: dict | None = None) -> dict:
    m = copy.deepcopy(src)
    for k in ("files_train", "files_test"):
        lst = list(m["data"].get(k) or [])
        lst += list((extra_lists or {}).get(k) or [])
        if keep:
            lst = [e for e in lst if re.search(keep, _entry_str(e))]
        if drop:
            lst = [e for e in lst if not re.search(drop, _entry_str(e))]
        m["data"][k] = [_rewrite(e, subs) for e in lst]
    if out_dir:
        m.setdefault("output", {})["out_dir"] = out_dir
    for k, v in (sets or {}).items():
        set_dotted(m, k, v)
    return m


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hab")
    h.add_argument("-s", "--spec", required=True)
    h.add_argument("--only", default=None)
    t = sub.add_parser("train")
    t.add_argument("-s", "--spec", required=True)
    m = sub.add_parser("model")
    m.add_argument("--src", required=True)
    m.add_argument("--add-lists-from", action="append", default=[],
                   help="model YAML whose files_train / files_test are appended before filtering")
    m.add_argument("--out", required=True)
    m.add_argument("--keep", default=None, help="regex: keep entries matching")
    m.add_argument("--drop", default=None, help="regex: drop entries matching")
    m.add_argument("--sub", action="append", default=[], help="'pattern=>replacement' applied to entries")
    m.add_argument("--out-dir", default=None)
    m.add_argument("--set", action="append", default=[], help="dotted.key=value (YAML value)")
    a = p.parse_args(argv)
    if a.cmd == "hab":
        print(json.dumps(run_hab(a.spec, a.only.split(",") if a.only else None), indent=1))
    elif a.cmd == "train":
        print(run_train(a.spec))
    else:
        extra = {"files_train": [], "files_test": []}
        for f in a.add_lists_from:
            d = read_yaml(f)["data"]
            for k in extra:
                extra[k] += list(d.get(k) or [])
        subs = [tuple(s.split("=>", 1)) for s in a.sub]
        sets = {k: _parse_value(v) for k, v in (s.split("=", 1) for s in a.set)}
        mv = model_variant(read_yaml(a.src), a.keep, a.drop, subs, a.out_dir, sets, extra)
        n = {k: len(mv["data"][k]) for k in ("files_train", "files_test")}
        if not sum(n.values()):
            raise SystemExit(f"model variant has no scenes after filtering: {n}")
        print(write_yaml(mv, a.out), n)


if __name__ == "__main__":
    main()
