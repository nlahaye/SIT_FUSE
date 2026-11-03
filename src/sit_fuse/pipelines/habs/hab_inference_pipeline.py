"""
HAB inference pipeline: runs between hab_training_pipeline.py and
hab_post_inference_pipeline.py.

    python -m sit_fuse.pipelines.habs.hab_inference_pipeline -y <hab yaml> [--stages clusters,heads]

Per instrument / key (trop, no_trop) it loads the model YAML written by
training (config_dir/model/train_model_<instrument>[_trop].yaml, or
inference.model_configs[instrument][key]) and, with one model load:

  clusters   standard SIT-FUSE inference (generate_output): writes
             <input>.clust.data_<N>clusters.zarr, _proba.zarr and, with
             no_heir, <input>.no_heir.clust.data_*. These are the inputs of
             run_context_free_geotiff_generation in post-inference.
  heads      encoder-side probabilistic outputs (optional, run_head_export):
             export_head_outputs.head_distributions (top-K parent / child
             distributions, entropies, margins, joint probability), plus
             optional MC input-noise bands and embedding-distance OOD bands
             (uq/hab_head_export.py). The stack is resampled onto the same
             geo_zarr grid with the same nearest-neighbour template as the
             context-free cluster GeoTIFFs and written as
             {PREFIX}.{YYYYMMDD}.L3m.DAY.head_outputs.tif, which
             hab_uq_common.find_product("head_outputs") picks up
             (uq.input_mode: auto | head_outputs).
  ood_stats  per-model embedding statistics (fit once on a sample of
             data.files_train, saved as .npz; run automatically by `heads`
             when head_export.ood.enabled and no stats file exists).

YAML (top level, all optional):
  run_inference: true          used by hab_full_pipeline
  run_head_export: false       heads stage (also honoured by post-inference)
  inference:
    stages: [clusters, heads]
    input_lists: [files_test, files_train]
    start_date / end_date      restrict scenes by YYYYMMDD in the file name
    overwrite: false
    model_configs: {modis: {no_trop: /path/train_model_modis.yaml}}
  head_export:
    top_k_parent: 5
    top_k_child: 3
    n_parent_ranks: 2
    mc: {samples: 0, input_noise: null, joint: true, seed: 0}
    ood: {enabled: false, stats_path: null, max_files: 30,
          max_pixels_per_file: 20000, shrinkage: 50.0}
    keep_no_geo: false
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sit_fuse.pipelines.habs.hab_post_inference_constants import (
    INSTRUMENT_PREFIX, RE_STR_TIFF, YAML_TEMPLATE_CF_GTIFF)
from sit_fuse.pipelines.habs.uq.hab_head_export import (
    BAND_DESCRIPTIONS, EmbeddingStats, batch_bands, extra_band_names, input_noise_sigma)

LOG = logging.getLogger("hab-inference")

HEAD_SUFFIX = ".head_outputs.tif"
CF_NODATA = -999999.0        # colocate_and_resample / toGeotiff fill value
CLUSTER_GEOTIFF_SCALE = 1000.0   # plot_clusters stores joint_label / 1000


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _read_yaml(path):
    from sit_fuse.utils import read_yaml
    return read_yaml(path)


def geo_zarr_path(yml_conf: dict[str, Any]) -> str:
    """Training uses geo_zarr: {out_dir, out_uid}; post-inference a plain path."""
    g = yml_conf["geo_zarr"]
    if isinstance(g, dict):
        return os.path.join(g["out_dir"], g["out_uid"] + ".zarr")
    return g


def model_config_path(yml_conf, instrument, key) -> str:
    over = (yml_conf.get("inference", {}).get("model_configs", {}) or {}).get(instrument, {})
    if isinstance(over, str):
        return over
    if key in over:
        return over[key]
    # same name as hab_training_utils.build_config_fname_data_prep_and_train
    # (not imported: that module pulls in the training stack)
    fname = os.path.join(yml_conf["config_dir"], "model", "train_model_" + instrument)
    return fname + ("_trop" if "no" not in key else "") + ".yaml"


def instrument_out_dir(yml_conf, instrument, key, sf_yaml) -> str:
    return yml_conf["instruments"][instrument][key].get("out_dir") or sf_yaml["output"]["out_dir"]


def _entry_base(entry) -> str:
    return os.path.basename(entry[0] if isinstance(entry, (list, tuple)) else entry)


def _entry_date(entry):
    m = re.search(r"(\d{8})", _entry_base(entry))
    return m.group(1) if m else None


def scene_entries(sf_yaml, inf_cfg) -> list:
    lists = inf_cfg.get("input_lists")
    if lists is None:
        lists = ["files_test"] + (["files_train"] if sf_yaml["output"].get("generate_train_output", True) else [])
    start = str(inf_cfg["start_date"]).replace("-", "") if inf_cfg.get("start_date") else None
    end = str(inf_cfg["end_date"]).replace("-", "") if inf_cfg.get("end_date") else None
    out, seen = [], set()
    for k in lists:
        for e in sf_yaml["data"].get(k) or []:
            d = _entry_date(e)
            if (start and d and d < start) or (end and d and d > end):
                continue
            if str(e) not in seen:
                seen.add(str(e))
                out.append(e)
    return out


def head_output_name(instrument, entry) -> str:
    base = _entry_base(entry)
    prefix = INSTRUMENT_PREFIX.get(instrument, "")
    m = re.search("(" + prefix + RE_STR_TIFF, base)
    return (m.group(1) if m else base) + HEAD_SUFFIX


# ---------------------------------------------------------------------------
# Stage 1: standard SIT-FUSE cluster inference
# ---------------------------------------------------------------------------

def _cuda():
    import torch
    return torch.cuda.is_available()


def cluster_outputs_exist(out_dir, base, no_heir) -> bool:
    ok = bool(glob.glob(os.path.join(out_dir, base + ".clust.data_*clusters.zarr")))
    if no_heir:
        ok &= bool(glob.glob(os.path.join(out_dir, base + ".no_heir.clust.data_*clusters.zarr")))
    return ok


def run_cluster_inference_scene(model, data, sf_yaml, out_dir, base, no_heir) -> None:
    from sit_fuse.inference.generate_output import generate_output
    tiled = bool(sf_yaml["data"].get("tile", False))
    use_gpu = _cuda()
    if getattr(model, "clust_tree_ckpt", None) is not None:
        generate_output(data, model, use_gpu, out_dir, base + ".clust.data", tiled=tiled)
    if no_heir:
        generate_output(data, model.pretrained_model, use_gpu, out_dir, base + ".no_heir.clust.data", tiled=tiled)


# ---------------------------------------------------------------------------
# Stage 2: head outputs (+ MC, OOD)
# ---------------------------------------------------------------------------

def _device(model):
    return next(model.pretrained_model.parameters()).device


def iter_batches(data, tiled, batch=None):
    import torch
    from torch.utils.data import DataLoader
    loader = DataLoader(data, batch_size=int(batch or (5 if tiled else 10000)), shuffle=False,
                        num_workers=0, drop_last=False)
    for xb in loader:
        yield xb[0] if isinstance(xb, (list, tuple)) else xb


def scene_head_stack(model, data, ex_cfg, sigma=None, emb_stats=None, tiled=False):
    """(n_bands, H, W) no-geo head stack, placed like generate_output.plot_clusters."""
    import torch
    from sit_fuse.train.export_head_outputs import assemble_scene
    dev = _device(model)
    chunks, tile_hw = [], None
    for xb in iter_batches(data, tiled, ex_cfg.get("batch_size")):
        x = xb.to(device=dev, dtype=torch.float32)
        rows = batch_bands(model, x, ex_cfg, sigma=sigma, emb_stats=emb_stats)
        if rows is None:
            raise RuntimeError("Heir_DC.forward returned None")
        if x.ndim == 4:
            tile_hw = (x.shape[2], x.shape[3])
        chunks.append(rows)
    rows = np.concatenate(chunks, axis=0)
    return assemble_scene(np.asarray(data.targets_full), rows, tile_hw)


def georeference_like_cf_gtiffs(stack, geo_zarr, work_zarr, out_tif) -> tuple[np.ndarray, dict]:
    """
    Resample a (n_bands, H, W) no-geo stack exactly as the context-free cluster
    GeoTIFFs are produced (YAML_TEMPLATE_CF_GTIFF: geo_zarr swath, nearest
    neighbour, 0.055 deg EPSG:4326), so head_outputs.tif and {PREFIX}.{date}.tif
    share one grid. Returns (stack_on_grid, rasterio profile).
    """
    import dask.array as da
    import rasterio
    from sit_fuse.preprocessing.colocate_and_resample import resample_or_fuse_data

    da.to_zarr(da.from_array(np.moveaxis(stack, 0, -1).astype(np.float32)), work_zarr, overwrite=True)
    cfg = copy.deepcopy(YAML_TEMPLATE_CF_GTIFF)
    cfg["low_res"]["data"]["filenames"] = [work_zarr]
    cfg["low_res"]["data"]["geo_filenames"] = [geo_zarr]
    tmp = str(out_tif) + ".tmp.tif"
    cfg["output_files"] = [tmp]
    resample_or_fuse_data(cfg)
    with rasterio.open(tmp) as src:
        arr = src.read().astype(np.float32)
        prof = src.profile.copy()
    os.remove(tmp)
    return arr, prof


def write_head_tif(out_tif, arr, prof, names, tags, cf_nodata=CF_NODATA):
    from sit_fuse.train.sf_uq_utils import HEAD_NODATA, write_band_stack
    arr = np.where(~np.isfinite(arr) | (arr <= cf_nodata + 1), HEAD_NODATA, arr).astype(np.float32)
    tags = dict(tags)
    tags["band_descriptions"] = json.dumps({n: BAND_DESCRIPTIONS[n] for n in names if n in BAND_DESCRIPTIONS})
    write_band_stack(out_tif, dict(zip(names, arr)), prof, tags, nodata=HEAD_NODATA)
    return str(out_tif)


def embedding_stats_path(yml_conf, instrument, key, out_dir) -> str:
    ood = yml_conf.get("head_export", {}).get("ood", {}) or {}
    p = ood.get("stats_path")
    if isinstance(p, dict):
        p = (p.get(instrument) or {}).get(key) if isinstance(p.get(instrument), dict) else p.get(instrument)
    return p or os.path.join(out_dir, "head_export", f"embedding_stats_{instrument}_{key}.npz")


def fit_embedding_stats(model, sf_yaml, ood_cfg, load_dataset: Callable, seed=0) -> EmbeddingStats:
    """
    Pass 1 (even-indexed training files): per-parent sums. Pass 2 (odd-indexed):
    distances -> percentile table, so percentiles are not computed on the
    pixels used for the means/variances.
    """
    import torch
    from sit_fuse.pipelines.habs.uq.hab_head_export import _embed, _root_probs
    rng = np.random.default_rng(seed)
    files = list(sf_yaml["data"].get("files_train") or [])
    rng.shuffle(files)
    files = files[: int(ood_cfg.get("max_files", 30))]
    if not files:
        raise RuntimeError("No data.files_train for embedding statistics")
    max_px = int(ood_cfg.get("max_pixels_per_file", 20000))
    tiled = bool(sf_yaml["data"].get("tile", False))
    dev = _device(model)

    def sample(entry):
        data = load_dataset(entry)
        if data is None or data.data_full is None:
            return None, None
        Es, Ps = [], []
        with torch.no_grad():
            for xb in iter_batches(data, tiled):
                x = xb.to(device=dev, dtype=torch.float32)
                P = _root_probs(model, x)
                E = _embed(model, x)
                Es.append(E.cpu().numpy())
                Ps.append(P.argmax(1).cpu().numpy())
        E, p = np.concatenate(Es), np.concatenate(Ps)
        if len(p) > max_px:
            idx = rng.choice(len(p), max_px, replace=False)
            E, p = E[idx], p[idx]
        return E, p

    fit_files = files[0::2] or files
    q_files = files[1::2] or files
    st = None
    for f in fit_files:
        E, p = sample(f)
        if E is None:
            continue
        if st is None:
            n_parent = int(sf_yaml.get("cluster", {}).get("num_classes", int(p.max()) + 1))
            st = EmbeddingStats(max(n_parent, int(p.max()) + 1), E.shape[1],
                                shrinkage=float(ood_cfg.get("shrinkage", 50.0)))
        st.update(E, p)
    if st is None:
        raise RuntimeError("No valid training samples for embedding statistics")
    st.finalize()
    Eq, pq = [], []
    for f in q_files:
        E, p = sample(f)
        if E is not None:
            Eq.append(E)
            pq.append(p)
    if Eq:
        st.calibrate_quantiles(np.concatenate(Eq), np.concatenate(pq))
    return st


# ---------------------------------------------------------------------------
# Consistency check (head export vs production cluster GeoTIFF)
# ---------------------------------------------------------------------------

def head_consistency(head_tif, cf_tif) -> dict[str, Any]:
    """Fraction of co-valid pixels where joint_label == round(cluster_tif * 1000)."""
    import rasterio
    from sit_fuse.train.sf_uq_utils import read_head_outputs
    bands, prof, _ = read_head_outputs(head_tif)
    with rasterio.open(cf_tif) as src:
        cf = src.read(1).astype(np.float64)
    j = bands["joint_label"]
    if cf.shape != j.shape:
        return {"head": str(head_tif), "cf": str(cf_tif), "error": f"shape {j.shape} vs {cf.shape}"}
    ok = (j >= 0) & np.isfinite(j) & (cf > CF_NODATA + 1) & (cf >= 0)
    n = int(ok.sum())
    agree = float((np.rint(cf[ok] * CLUSTER_GEOTIFF_SCALE) == np.rint(j[ok])).mean()) if n else np.nan
    return {"head": str(head_tif), "cf": str(cf_tif), "n_pixels": n, "joint_agreement": agree}


def validate_head_outputs(yml_conf, min_agreement=0.99) -> list[dict[str, Any]]:
    out = []
    for instrument, keys in yml_conf["instruments"].items():
        for key, blk in keys.items():
            d = blk.get("out_dir")
            if not d:
                continue
            for h in sorted(glob.glob(os.path.join(d, "**", "*" + HEAD_SUFFIX), recursive=True)):
                cf = h[: -len(HEAD_SUFFIX)] + ".tif"
                if os.path.exists(cf):
                    r = head_consistency(h, cf)
                    r.update(instrument=instrument, key=key)
                    if r.get("joint_agreement", 1.0) < min_agreement:
                        LOG.warning("%s: joint agreement %.4f < %.2f (grid or model mismatch)",
                                    h, r["joint_agreement"], min_agreement)
                    out.append(r)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _tags(sf_cfg_path, entry, model, ex, names, geo, mode, stats_path):
    from sit_fuse.train.sf_uq_utils import FORMAT_VERSION
    return {
        "format_version": FORMAT_VERSION,
        "num_child_classes": int(model.num_classes),
        "num_parent_classes": int(ex.get("_num_parent_classes", -1)),
        "top_k_parent": int(ex.get("top_k_parent", 5)), "top_k_child": int(ex.get("top_k_child", 3)),
        "n_parent_ranks": int(ex.get("n_parent_ranks", 2)),
        "extra_bands": ",".join(extra_band_names(ex)),
        "mc_samples": int((ex.get("mc") or {}).get("samples", 0)),
        "embedding_stats": stats_path or "none",
        "sit_fuse_yaml": sf_cfg_path, "source_file": str(entry),
        "georeferenced": geo is not None, "georef_source": geo or "none", "georef_mode": mode,
    }


def run_instrument(yml_conf, instrument, key, stages, model=None, load_dataset=None,
                   georef_fn=None) -> dict[str, Any]:
    """
    model / load_dataset / georef_fn are injectable for testing; by default the
    model comes from prep_model(sf_yaml), datasets from get_prediction_dataset
    and georeferencing from georeference_like_cf_gtiffs.
    """
    from sit_fuse.train.sf_uq_utils import head_band_names
    inf = yml_conf.get("inference", {}) or {}
    ex = copy.deepcopy(yml_conf.get("head_export", {}) or {})
    cfg_path = model_config_path(yml_conf, instrument, key)
    sf_yaml = _read_yaml(cfg_path)
    out_dir = instrument_out_dir(yml_conf, instrument, key, sf_yaml)
    sf_yaml["output"]["out_dir"] = out_dir
    os.makedirs(out_dir, exist_ok=True)
    no_heir = bool(yml_conf.get("no_heir", sf_yaml["output"].get("generate_intermediate_output", True)))
    overwrite = bool(inf.get("overwrite", False))
    tiled = bool(sf_yaml["data"].get("tile", False))
    entries = scene_entries(sf_yaml, inf)
    summary = {"instrument": instrument, "key": key, "n_scenes": len(entries), "clusters": [], "heads": []}

    if load_dataset is None:
        from sit_fuse.datasets.dataset_utils import get_prediction_dataset

        def load_dataset(e):
            return get_prediction_dataset(sf_yaml, e)[0]
    if model is None:
        from sit_fuse.inference.generate_output import prep_model
        model = prep_model(sf_yaml)
    model.eval()

    do_heads = "heads" in stages
    names, sigma, stats, stats_path = None, None, None, None
    if do_heads:
        names = head_band_names(int(ex.get("top_k_parent", 5)), int(ex.get("top_k_child", 3)),
                                int(ex.get("n_parent_ranks", 2))) + extra_band_names(ex)
        mc = ex.get("mc", {}) or {}
        if int(mc.get("samples", 0)) > 0:
            import torch
            torch.manual_seed(int(mc.get("seed", 0)))
        if (ex.get("ood", {}) or {}).get("enabled", False):
            stats_path = embedding_stats_path(yml_conf, instrument, key, out_dir)
            if os.path.exists(stats_path) and not (ex["ood"].get("refit", False)):
                stats = EmbeddingStats.load(stats_path)
            else:
                stats = fit_embedding_stats(model, sf_yaml, ex["ood"], load_dataset,
                                            seed=int(ex["ood"].get("seed", 0)))
                stats.save(stats_path)
                LOG.info("%s/%s: embedding stats -> %s", instrument, key, stats_path)
    if "ood_stats" in stages and not do_heads:
        ood = ex.get("ood", {}) or {}
        stats_path = embedding_stats_path(yml_conf, instrument, key, out_dir)
        fit_embedding_stats(model, sf_yaml, ood, load_dataset, seed=int(ood.get("seed", 0))).save(stats_path)
        summary["embedding_stats"] = stats_path

    geo = geo_zarr_path(yml_conf) if yml_conf.get("geo_zarr") else None
    georef_fn = georef_fn or georeference_like_cf_gtiffs
    for entry in entries:
        base = _entry_base(entry)
        head_tif = os.path.join(out_dir, head_output_name(instrument, entry))
        need_heads = do_heads and (overwrite or not os.path.exists(head_tif))
        need_clusters = "clusters" in stages and (overwrite or not cluster_outputs_exist(out_dir, base, no_heir))
        if not (need_heads or need_clusters):
            continue
        data = load_dataset(entry)
        if data is None or data.data_full is None:
            LOG.warning("Skipping %s: no valid samples", base)
            continue
        if need_heads:   # before cluster inference: run_inference pads data_full in place
            if sigma is None and int((ex.get("mc") or {}).get("samples", 0)) > 0:
                sigma = input_noise_sigma(ex, sf_yaml, int(np.asarray(data.data_full).shape[1]))
            stack = scene_head_stack(model, data, ex, sigma=sigma, emb_stats=stats, tiled=tiled)
            if geo is not None:
                work = os.path.join(out_dir, base + ".clust.head_outputs.zarr")
                arr, prof = georef_fn(stack, geo, work, head_tif)
                if not ex.get("keep_no_geo", False):
                    import shutil
                    shutil.rmtree(work, ignore_errors=True)
                mode = "cf_gtiff_template"
            else:
                from rasterio.transform import Affine
                arr = stack
                prof = {"height": stack.shape[1], "width": stack.shape[2], "transform": Affine.identity(),
                        "crs": None}
                mode = "none"
            write_head_tif(head_tif, arr, prof, names,
                           _tags(cfg_path, entry, model, ex, names, geo, mode, stats_path))
            summary["heads"].append(head_tif)
        if need_clusters:
            run_cluster_inference_scene(model, data, sf_yaml, out_dir, base, no_heir)
            summary["clusters"].append(base)
    return summary


def run_hab_inference_pipeline(yml_conf, stages=None, **inject) -> dict[str, Any]:
    inf = yml_conf.get("inference", {}) or {}
    if stages is None:
        stages = list(inf.get("stages", ["clusters"]))
        if yml_conf.get("run_head_export", False) and "heads" not in stages:
            stages.append("heads")
    stages = tuple(stages)
    out = {}
    for instrument, keys in yml_conf["instruments"].items():
        for key in keys:
            LOG.info("Inference %s/%s stages=%s", instrument, key, stages)
            out[f"{instrument}_{key}"] = run_instrument(yml_conf, instrument, key, stages, **inject)
    yml_conf["inference_summary"] = out
    return out


def run_head_output_export(yml_conf, **inject):
    """Head stage only (post-inference hook); existing head_outputs.tif are kept."""
    return run_hab_inference_pipeline(yml_conf, stages=("heads",), **inject)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yaml", required=True, help="HAB pipeline YAML")
    ap.add_argument("--stages", default=None, help="comma list of clusters,heads,ood_stats")
    a = ap.parse_args()
    conf = _read_yaml(a.yaml)
    run_hab_inference_pipeline(conf, None if a.stages is None else a.stages.split(","))
