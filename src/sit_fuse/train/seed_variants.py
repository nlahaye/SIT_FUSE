"""
Copyright [2022-26], by the California Institute of Technology and Chapman University.
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.

Seed variants of a SIT-FUSE model, for sit_fuse.xai.cluster_stability.

The main model (base config) is left untouched. For each seed this writes a derived config with its own
out_dir and seed, optionally shares the main run's training sample, and writes a shell script that runs
the training / inference stages for every variant. After inference, `stability` writes the
cluster_stability YAML (one label raster per scene and run, same scenes in the same order).

  python -m sit_fuse.train.seed_variants make      -y config/seeds/whale_seed_variants_v2.yaml
  bash <config_dir>/run_seed_variants.sh            # or run the stages per config by hand
  python -m sit_fuse.train.seed_variants stability -y config/seeds/whale_seed_variants_v2.yaml
  python -m sit_fuse.xai.cluster_stability -y <stability.yaml>

seed_scope controls what the seed changes:
  model  (default) model initialisation, shuffling, dropout, Gibbs sampling; same training sample
         (the main run's train_data*.npy, train_data.files.json and encoder_scaler.pkl are linked or
         copied into the variant out_dir; needs data.sampling.cache_train in the main run)
  data   training-sample selection only (data.sampling.seed = seed); model seed = base seed
  both   both of the above

YAML:
  base_config: config/model/x_split.yaml   # main model config (e.g. written by scene_split)
  seeds: [1, 2, 3, 4]
  seed_scope: model                        # model | data | both
  share_train_cache: link                  # link | copy | none   (seed_scope: model)
  out_dir_pattern: "{base}_seed{seed}/"    # {base} = main out_dir without trailing slash
  config_dir: config/model/seeds
  files_test: holdout                      # holdout | base | <list of files>  (inference scenes per variant)
  stages: [pretrain_encoder, finetune_dc_mlp_head, generate_output]   # + train_child_heads for fine level
  python: python
  num_threads: null
  overrides: {}                            # nested dict merged into every variant config
  stability:
    yaml: config/xai/x_cluster_stability_seeds.yaml
    template: config/xai/x_cluster_stability_v2.yaml   # other keys copied (scenes, label_scale, ...)
    include_main: true
    level: parent                          # parent (.no_heir.clust.data) | fine (.clust.data)
    raster: no_geo                         # no_geo (values / 1000) | full_geo
    out_dir: /path/stability
"""
import argparse
import copy
import glob
import os
import shutil

import yaml

STAGE_MODULES = {
    "pretrain_encoder": "sit_fuse.train.pretrain_encoder",
    "finetune_dc_mlp_head": "sit_fuse.train.finetune_dc_mlp_head",
    "train_heirarchichal_deep_cluster": "sit_fuse.train.train_heirarchichal_deep_cluster",
    "train_child_heads": "sit_fuse.train.train_child_heads",
    "generate_output": "sit_fuse.inference.generate_output",
}
CACHE_FILES = ("train_data.npy", "train_data.indices.npy", "train_data.files.json", "encoder_scaler.pkl")


PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # src/sit_fuse


def resolve_path(p):
    """Relative config paths (e.g. config/model/x.yaml) are relative to the sit_fuse package dir, so the
    command works from any directory; an existing path relative to the cwd is used as is."""
    if p is None or os.path.isabs(p) or os.path.exists(p):
        return p
    return os.path.join(PKG_DIR, p)


def _merge(dst, src):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def _load(p):
    with open(resolve_path(p)) as f:
        return yaml.safe_load(f)


def _dump(obj, p, header=None):
    p = resolve_path(p)
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w") as f:
        if header:
            f.write(header)
        yaml.safe_dump(obj, f, sort_keys=False, default_flow_style=None, width=200)


def _drop_stage_seeds(conf):
    """Stage-level seeds win over the top-level seed (train.seeding), so remove them in variants."""
    for path in (("encoder", "training"), ("cluster", "training"), ("cluster", "heir", "training")):
        cur = conf
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
        if isinstance(cur, dict):
            cur.pop("seed", None)


def variant_out_dir(base_conf, seed, pattern):
    base = base_conf["output"]["out_dir"].rstrip("/")
    return pattern.format(base=base, seed=seed, name=os.path.basename(base))


def variant_config(base_conf, seed, sv):
    c = copy.deepcopy(base_conf)
    scope = sv.get("seed_scope", "model")
    if scope not in ("model", "data", "both"):
        raise ValueError("seed_scope must be model | data | both")
    _drop_stage_seeds(c)
    samp = c["data"].setdefault("sampling", {}) or {}
    c["data"]["sampling"] = samp
    base_seed = base_conf.get("seed")
    c["seed"] = int(seed) if scope in ("model", "both") else base_seed
    if scope in ("data", "both"):
        samp["seed"] = int(seed)
    samp["cache_train"] = True
    c["output"]["out_dir"] = variant_out_dir(base_conf, seed, sv.get("out_dir_pattern", "{base}_seed{seed}/"))
    ft = sv.get("files_test", "holdout")
    if ft == "holdout":
        if c["data"].get("files_holdout"):
            c["data"]["files_test"] = list(c["data"]["files_holdout"])
    elif isinstance(ft, list):
        c["data"]["files_test"] = ft
    c["output"].pop("heir_model_dir", None)
    _merge(c, sv.get("overrides"))
    c["seed_variant"] = dict(seed=int(seed), seed_scope=scope, base_out_dir=base_conf["output"]["out_dir"])
    return c


def share_cache(base_conf, var_conf, mode):
    if mode in (None, "none"):
        return []
    src_dir, dst_dir = base_conf["output"]["out_dir"], var_conf["output"]["out_dir"]
    os.makedirs(dst_dir, exist_ok=True)
    done = []
    missing = [f for f in CACHE_FILES[:2] if not os.path.exists(os.path.join(src_dir, f))]
    if missing:
        print("WARNING: main run has no cached training sample in", src_dir,
              "(set data.sampling.cache_train: true and run pretrain_encoder for the main model first);",
              "the variant will draw its own sample with data.sampling.seed =",
              (var_conf["data"].get("sampling") or {}).get("seed"))
        return done
    for f in CACHE_FILES:
        s, d = os.path.join(src_dir, f), os.path.join(dst_dir, f)
        if not os.path.exists(s) or os.path.lexists(d):
            continue
        if mode == "link":
            os.symlink(os.path.abspath(s), d)
        else:
            shutil.copy2(s, d)
        done.append(d)
    return done


def write_script(paths, sv, dst):
    py = sv.get("python", "python")
    nt = sv.get("num_threads")
    stages = sv.get("stages", ["pretrain_encoder", "finetune_dc_mlp_head", "generate_output"])
    lines = ["#!/usr/bin/env bash", "# Written by sit_fuse.train.seed_variants; one block per seed variant.", "set -euo pipefail", ""]
    for p in paths:
        lines.append("# %s" % p)
        for st in stages:
            mod = STAGE_MODULES.get(st, st)
            lines.append("%s -m %s -y %s%s" % (py, mod, p, " -n %s" % nt if nt else ""))
        lines.append("")
    with open(dst, "w") as f:
        f.write("\n".join(lines))
    os.chmod(dst, 0o755)
    return dst


def make(sv):
    base = _load(sv["base_config"])
    if (base["data"].get("sampling") or {}).get("seed") is None and sv.get("seed_scope", "model") == "model":
        print("WARNING: base config has no data.sampling.seed; with seed_scope: model the variants rely on the "
              "cached sample of the main run (data.sampling.cache_train).")
    cdir = resolve_path(sv.get("config_dir") or os.path.join(os.path.dirname(resolve_path(sv["base_config"])), "seeds"))
    stem = os.path.splitext(os.path.basename(sv["base_config"]))[0]
    paths = []
    for s in sv["seeds"]:
        c = variant_config(base, s, sv)
        if sv.get("seed_scope", "model") == "model":
            share_cache(base, c, sv.get("share_train_cache", "link"))
        p = os.path.join(cdir, "%s_seed%d.yaml" % (stem, int(s)))
        _dump(c, p, "# Seed variant %d of %s (sit_fuse.train.seed_variants).\n" % (int(s), sv["base_config"]))
        paths.append(p)
        print("WROTE", p, "->", c["output"]["out_dir"])
    script = write_script(paths, sv, os.path.join(cdir, "run_seed_variants.sh"))
    print("WROTE", script)
    return paths, script


def _scene_name(f):
    while isinstance(f, (list, tuple)) and f:
        f = f[0]
    return os.path.basename(str(f).rstrip("/"))


def label_raster(out_dir, scene, level="parent", raster="no_geo"):
    tag = ".no_heir.clust.data" if level == "parent" else ".clust.data"
    suffix = ".no_geo.tif" if raster == "no_geo" else ".zarr.full_geo.tif"
    hits = sorted(glob.glob(os.path.join(glob.escape(out_dir), glob.escape(scene + tag) + "_*clusters" + suffix)),
                  key=os.path.getmtime)
    return hits[-1] if hits else None


def stability(sv):
    st = sv.get("stability", {})
    base = _load(sv["base_config"])
    runs = []
    if st.get("include_main", True):
        runs.append(("main", base["output"]["out_dir"]))
    for s in sv["seeds"]:
        runs.append(("seed%d" % int(s), variant_out_dir(base, s, sv.get("out_dir_pattern", "{base}_seed{seed}/"))))
    var0 = variant_config(base, sv["seeds"][0], sv) if sv.get("seeds") else base
    scenes = [_scene_name(f) for f in var0["data"]["files_test"]]
    level, raster = st.get("level", "parent"), st.get("raster", "no_geo")
    table, dropped = {name: [] for name, _ in runs}, []
    for sc in scenes:
        hits = {name: label_raster(d, sc, level, raster) for name, d in runs}
        if any(v is None for v in hits.values()):
            dropped.append((sc, [k for k, v in hits.items() if v is None]))
            continue
        for k, v in hits.items():
            table[k].append(v)
    if dropped:
        print("WARNING: %d scenes missing in some runs (dropped), e.g. %s" % (len(dropped), dropped[:3]))
    n = len(next(iter(table.values()))) if table else 0
    if n == 0:
        raise RuntimeError("no scene has label rasters in every run; run inference (generate_output) first")
    out = _load(st["template"]) if st.get("template") else {}
    out["runs"] = table
    if out.get("scenes"):
        keep = {sc for sc in scenes if sc not in {d[0] for d in dropped}}
        out["scenes"] = [p for p in out["scenes"] if _scene_name(p) in keep]
    out["label_scale"] = 1000 if raster == "no_geo" else out.get("label_scale", 1)
    out.setdefault("label_fill", -1)
    if st.get("out_dir"):
        out["out_dir"] = st["out_dir"]
    out["seed_variants"] = dict(level=level, raster=raster, seed_scope=sv.get("seed_scope", "model"),
                                n_scenes=n, n_dropped=len(dropped))
    _dump(out, st["yaml"], "# Written by sit_fuse.train.seed_variants stability.\n")
    print("WROTE", resolve_path(st["yaml"]), "(%d runs x %d scenes)" % (len(table), n))
    return resolve_path(st["yaml"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Seed variants of a SIT-FUSE model for cluster_stability.")
    ap.add_argument("cmd", choices=["make", "commands", "stability"])
    ap.add_argument("-y", "--yaml", required=True)
    a = ap.parse_args(argv)
    sv = _load(a.yaml)
    if a.cmd == "make":
        make(sv)
    elif a.cmd == "commands":
        base = _load(sv["base_config"])
        cdir = resolve_path(sv.get("config_dir") or os.path.join(os.path.dirname(resolve_path(sv["base_config"])), "seeds"))
        stem = os.path.splitext(os.path.basename(sv["base_config"]))[0]
        paths = [os.path.join(cdir, "%s_seed%d.yaml" % (stem, int(s))) for s in sv["seeds"]]
        print(open(write_script(paths, sv, os.path.join(cdir, "run_seed_variants.sh"))).read())
    else:
        stability(sv)


if __name__ == "__main__":
    main()
