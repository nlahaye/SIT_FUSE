#!/usr/bin/env python3
"""
Write the georeferenced SIT-FUSE cluster rasters (<input>.clust.data_<N>clusters.zarr.full_geo.tif) for every
scene of one or more SIT-FUSE inference YAMLs, using SIT-FUSE's own writer
(postprocessing/generate_cluster_geotiffs.run_geotiff_gen, the step normally driven by a
config/postprocess/*geotiff_gen*.yaml). The matchup generators (build_fdcc_matchups.py, the FIRMS / HMS
generators) read these files as the scene rasters and scene grids, so they must exist before the matchups.

Scene list: data.<input_lists> of each sit_fuse_yaml; cluster outputs: output.out_dir (or out_dir below) /
basename(input) + ".clust.data_*clusters.zarr". The input file supplies the grid (geotransform / GCPs), as in
config/postprocess/goes_tempo_geotiff_gen_4.yaml. No context assignment.

    python make_full_geo.py goes18/g18_full_geo_2026.yaml --list            # counts only
    python make_full_geo.py goes18/g18_full_geo_2026.yaml --shard 0/8       # one of 8 parallel parts
    python make_full_geo.py goes18/g18_full_geo_2026.yaml                   # everything in one process

YAML:
    sit_fuse_yamls: [/.../config/model/goes_tempo_test_3_2026.yaml]   # inference YAMLs (inputs + out_dir)
    input_lists: [files_test]        # data.* lists to read (add files_train for encoder-training scenes)
    out_dir: null                    # override output.out_dir of the YAMLs
    include_regex: "_s2026"          # optional: only inputs whose basename matches (e.g. one year)
    exclude_tokens: [proba, no_heir] # cluster outputs whose name contains these are skipped
    skip_existing: true              # leave scenes whose .full_geo.tif already exists
    clust_reader_type: zarr_to_numpy
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys

import yaml

LOG = logging.getLogger("make-full-geo")


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def collect(cfg: dict, shard: tuple[int, int] | None = None) -> tuple[list[str], list[str], dict]:
    """(cluster outputs, matching input GeoTIFFs, counts) for the scenes still to write. Shards split the full
    scene list (before skip_existing), so a restarted shard keeps the same scenes."""
    lists = cfg.get("input_lists") or ["files_test"]
    rx = re.compile(cfg["include_regex"]) if cfg.get("include_regex") else None
    excl = [str(t) for t in (cfg.get("exclude_tokens") or ["proba", "no_heir"])]
    skip = bool(cfg.get("skip_existing", True))
    clust, gtiff, seen = [], [], set()
    n = dict(inputs=0, no_output=0, existing=0, todo=0)
    for sy in cfg["sit_fuse_yamls"]:
        sf = load_yaml(sy)
        out_dir = cfg.get("out_dir") or sf["output"]["out_dir"]
        files = []
        for key in lists:
            files += list((sf.get("data") or {}).get(key) or [])
        for inp in files:
            base = os.path.basename(inp)
            if rx and not rx.search(base):
                continue
            n["inputs"] += 1
            outs = sorted(p for p in glob.glob(os.path.join(out_dir, glob.escape(base) + ".clust.data_*clusters.zarr"))
                          if not any(t in os.path.basename(p) for t in excl))
            if not outs:
                n["no_output"] += 1
                continue
            for p in outs:
                if p in seen:
                    continue
                seen.add(p)
                if shard and (len(seen) - 1) % shard[1] != shard[0]:
                    continue
                if skip and os.path.exists(p + ".full_geo.tif"):
                    n["existing"] += 1
                    continue
                clust.append(p)
                gtiff.append(inp)
                n["todo"] += 1
    return clust, gtiff, n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--shard", default=None, help="i/n: write every n-th scene starting at i")
    ap.add_argument("--list", action="store_true", help="report counts, write nothing")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_yaml(a.config)
    shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
    clust, gtiff, n = collect(cfg, shard)
    LOG.info("%d inputs: %d without cluster output; %s%d full_geo already present, %d to write",
             n["inputs"], n["no_output"], f"shard {shard[0]}/{shard[1]}: " if shard else "", n["existing"], n["todo"])
    if a.list or not clust:
        return 0

    try:
        from sit_fuse.postprocessing.generate_cluster_geotiffs import run_geotiff_gen
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "postprocessing"))
        from generate_cluster_geotiffs import run_geotiff_gen

    run_geotiff_gen({
        "gen_from_geotiffs": False,
        "data": {"clust_reader_type": cfg.get("clust_reader_type", "zarr_to_numpy"),
                 "reader_kwargs": {"no_arg": "no_arg"}, "cluster_fnames": clust, "gtiff_data": gtiff,
                 "create_separate": False, "subset_inds": []},
        "context": {"apply_context": False, "generate_union": False, "clusters": [], "name": "none",
                    "compare_truth": False},
    })
    missing = [p for p in clust if not os.path.exists(p + ".full_geo.tif")]
    if missing:
        LOG.error("%d of %d full_geo rasters not written, e.g. %s", len(missing), len(clust), missing[0])
        return 1
    LOG.info("wrote %d full_geo rasters", len(clust))
    return 0


if __name__ == "__main__":
    sys.exit(main())
