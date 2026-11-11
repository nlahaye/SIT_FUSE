"""
Build a full SIT-FUSE config for the xai tools: <base model yml> + <species xai block>.

    python -m sit_fuse.xai.make_xai_config -b config/model/copernicus_whale.yaml \
        -x config/xai/whale_xai_block.yaml -o config/model/copernicus_whale_xai.yaml

- Deep-merges the block into the base config (block wins; lists replaced).
- Fills xai.interpret.label_files from xai.interpret.label_pattern, one per data file
  (default data.files_train). Placeholders: {out_dir} (output.out_dir), {stem} (scene file name without
  directory), {name} (scene file name), {n_classes} (cluster.num_classes).
- Sets data.number_channels = len(feature_names) + len(delete_chans) in the merged copy (the base value is not
  used by non-convolutional DBN training, so e.g. copernicus_shark.yaml says 6 for a 29-channel model; the xai
  tools fall back to it only when no names are given). Checks that every name used by feature_groups / external
  exists. Warns (does not fail) for missing label files, so the config can be
  built before inference finishes.
"""
import argparse
import copy
import os
import sys

import yaml


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def fill_label_files(conf):
    ip = conf.setdefault("xai", {}).setdefault("interpret", {})
    pat = ip.pop("label_pattern", None)
    data_files = ip.get("data_files") or conf["data"]["files_train"]
    if pat and not ip.get("label_files"):
        fmt = dict(out_dir=conf["output"]["out_dir"].rstrip("/"), n_classes=conf.get("cluster", {}).get("num_classes"))
        ip["label_files"] = [pat.format(stem=os.path.splitext(os.path.basename(f))[0], name=os.path.basename(f), **fmt)
                             for f in data_files]
    missing = [f for f in ip.get("label_files", []) if not os.path.exists(f)]
    if missing:
        print(f"WARNING: {len(missing)}/{len(ip['label_files'])} label files not found yet, e.g. {missing[0]}",
              file=sys.stderr)
    return conf


def check(conf):
    x, d = conf["xai"], conf["data"]
    names = list(x["feature_names"])
    n_new = len(names) + len(d.get("delete_chans") or [])
    if int(d.get("number_channels", -1)) != n_new:
        print(f"NOTE: data.number_channels {d.get('number_channels')} -> {n_new} (= len(feature_names) + "
              "len(delete_chans)) in the merged config", file=sys.stderr)
        d["number_channels"] = n_new
    if len(set(names)) != len(names):
        raise ValueError("duplicate feature_names")
    used = [m for g in (x.get("feature_groups") or {}).values() for m in g]
    ext = (x.get("interpret") or {}).get("external") or {}
    used += [ext[k] for k in ("sst", "u", "v") if ext.get(k)]
    bad = sorted(set(used) - set(names))
    if bad:
        raise ValueError(f"names not in feature_names: {bad}")
    vn = (x.get("interpret") or {}).get("variable_names")
    if vn is not None and list(vn) != names:
        raise ValueError("xai.interpret.variable_names differs from xai.feature_names")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("-b", "--base", required=True, help="SIT-FUSE model yml used to train / run the encoder")
    p.add_argument("-x", "--xai", required=True, help="species xai block yml")
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args(argv)
    with open(a.base) as f:
        base = yaml.safe_load(f)
    with open(a.xai) as f:
        block = yaml.safe_load(f)
    conf = fill_label_files(deep_merge(base, block))
    check(conf)
    with open(a.out, "w") as f:
        yaml.safe_dump(conf, f, sort_keys=False, default_flow_style=None, width=120)
    print("wrote", a.out, "-", len(conf["xai"]["feature_names"]), "channels,",
          len(conf["xai"]["interpret"].get("label_files", [])), "label files")


if __name__ == "__main__":
    main()
