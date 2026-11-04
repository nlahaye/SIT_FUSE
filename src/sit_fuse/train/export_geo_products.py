#!/usr/bin/env python
"""
Georeferenced, GIS-ready products from classifier_inference_uq.py outputs.

classifier_inference_uq.py writes one single-band GeoTIFF per layer per scene
on the head-output grid (<inference_dir>/<model>_<pair_id>_<layer>.tif). This
script turns them into products that open directly in QGIS / GDAL:

  rasters/<model>_<pair_id>.tif     one multi-band COG per scene (selected
                                    layers, band descriptions, units, scene
                                    time and model provenance in tags);
                                    swath (GCP) grids are warped to dst_crs
  <model>_smoke_polygons.parquet    connected smoke regions from mask_layer,
  (+ .gpkg)                         one row per region: scene_time (UTC),
                                    pair_id, area_km2, n_pixels, mean/max
                                    probability, mean uncertainty layers,
                                    selective retained fraction
  <model>_scene_footprints.parquet  valid-data footprint per scene with
  (+ .gpkg)                         manifest area estimates / intervals,
                                    excluded fraction, raster path and,
                                    when available, evaluation metrics
  <model>_pixels.parquet            optional point table (lon/lat + layers,
                                    strided) for analysis outside GIS
  geo_products_manifest.json

All vector outputs share one CRS (vectors.crs, default EPSG:4326) and carry a
`scene_time` timestamp, so QGIS's Temporal Controller can animate them.

Georeferencing, first that works per scene:
  1. the inference layer itself (CRS + non-identity transform)
  2. the head-output GeoTIFF (affine or GCPs; GCPs are lost in the
     single-band inference layers)
  3. the scene's SIT-FUSE context-free GeoTIFF (sit_fuse_context_free in the
     scene file)
Scenes without georeferencing are skipped with a warning.

Usage:
  python export_geo_products.py uq_config.yaml [--models random_forest ...]
         [--inference-dir DIR] [--output-dir DIR]

Config (all optional):
  geo_export:
    output_dir: <output_dir>/geo
    inference_dir_template: <evaluation.inference_dir_template or output_dir/inference/{model}>
    scene_file: <inference.scene_file>      # for sit_fuse_context_free / scene_time
    evaluation_dir: <evaluation.output_dir> # joins per_scene_metrics.csv when present
    layers: [p_1, pred_label, pred_label_selective, selective_flag, confidence,
             predictive_entropy, ensemble_mutual_information, ensemble_variance,
             ensemble_p1_q05, ensemble_p1_q95, excluded_cluster_mask]   # null = all
    raster:
      enabled: true
      cog: true
      dst_crs: null            # null: keep affine grids, warp GCP grids to EPSG:4326
      resolution: null         # dst_crs units; null = GDAL default
      resampling: nearest      # continuous layers; categorical layers always nearest
    vectors:
      enabled: true
      crs: EPSG:4326
      mask_layer: pred_label   # or pred_label_selective (abstentions dropped)
      positive_value: 1
      min_pixels: 1
      simplify: null           # tolerance in vectors.crs units
      summary_layers: [p_1, predictive_entropy, ensemble_mutual_information,
                       ensemble_variance, selective_flag]
      formats: [parquet, gpkg]
    footprints: true
    # Scene-area intervals on the footprints:
    #   conformal - area_expected_km2 +/- q_hat * area_analytic_sd_km2, q_hat from
    #               <evaluation_dir>/<model>/area_interval_calibration.json (group
    #               q_hat when the scene's group was calibrated, else pooled);
    #               analytic lo/hi columns are dropped (they undercover)
    #   analytic  - keep the independent-pixel analytic interval only
    #   all       - both;  none - drop all interval columns
    area_intervals: conformal
    area_calibration: null     # path override for area_interval_calibration.json
    pixels:
      enabled: false
      stride: 4
      layers: null             # null = geo_export.layers
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sit_fuse.train.classifier_utils import load_json, load_yaml

from sf_uq_utils import (OUT_NODATA, conformal_group_key, layer_info, load_matchups, pixel_area_km2,
                         save_json)

LOG = logging.getLogger("sf-uq-geo")

DEFAULT_LAYERS = [
    "p_1", "pred_label", "pred_label_selective", "selective_flag", "confidence",
    "predictive_entropy", "ensemble_mutual_information", "ensemble_variance",
    "ensemble_p1_q05", "ensemble_p1_q95", "excluded_cluster_mask",
]
DEFAULT_SUMMARY = ["p_1", "predictive_entropy", "ensemble_mutual_information",
                   "ensemble_variance", "selective_flag"]
CATEGORICAL = ("pred_label", "pred_label_selective", "selective_flag", "excluded_cluster_mask")
TIME_RE = re.compile(r"(\d{8}T\d{6})Z?")


# ---------------------------------------------------------------------------
# Georeferencing
# ---------------------------------------------------------------------------

def _grid_of(path) -> dict | None:
    """{'crs','transform','height','width'} or {'gcps','gcp_crs',...}; None if not georeferenced."""
    import rasterio
    from rasterio.errors import RasterioError
    from rasterio.transform import Affine

    if not path or not Path(path).exists():
        return None
    try:
        with rasterio.open(path) as src:
            g = {"height": src.height, "width": src.width, "crs": src.crs, "transform": src.transform}
            gcps, gcp_crs = src.gcps
    except (RasterioError, OSError, ValueError):
        return None
    if g["crs"] is not None and g["transform"] != Affine.identity():
        return g
    if gcps and gcp_crs is not None:
        return {**g, "crs": None, "transform": None, "gcps": list(gcps), "gcp_crs": gcp_crs}
    return None


def resolve_grid(scene: dict, layer_path, record: dict | None):
    """First georeferenced grid among layer / head outputs / context-free GeoTIFF."""
    shape = None
    for src_name, path in (("layer", layer_path), ("head_outputs", scene.get("head_outputs")),
                           ("context_free", (record or {}).get("sit_fuse_context_free"))):
        g = _grid_of(path)
        if g is None:
            continue
        if shape is None:
            import rasterio
            with rasterio.open(layer_path) as src:
                shape = (src.height, src.width)
        if (g["height"], g["width"]) != shape:
            LOG.warning("%s: %s grid %s != layer grid %s; not used", scene["pair_id"], src_name,
                        (g["height"], g["width"]), shape)
            continue
        return src_name, g
    return None, None


def target_grid(g: dict, dst_crs, resolution):
    """Output grid: native affine grid unless warping is needed/requested."""
    from rasterio.crs import CRS
    from rasterio.warp import calculate_default_transform

    if "gcps" not in g and dst_crs is None:
        return {"crs": g["crs"], "transform": g["transform"], "height": g["height"],
                "width": g["width"], "warp": False}
    dst = CRS.from_user_input(dst_crs or "EPSG:4326")
    kw = {"resolution": resolution} if resolution else {}
    if "gcps" in g:
        tr, w, h = calculate_default_transform(g["gcp_crs"], dst, g["width"], g["height"], gcps=g["gcps"], **kw)
    else:
        if CRS.from_user_input(g["crs"]) == dst and not resolution:
            return {"crs": g["crs"], "transform": g["transform"], "height": g["height"],
                    "width": g["width"], "warp": False}
        from rasterio.transform import array_bounds
        b = array_bounds(g["height"], g["width"], g["transform"])
        tr, w, h = calculate_default_transform(g["crs"], dst, g["width"], g["height"],
                                               left=b[0], bottom=b[1], right=b[2], top=b[3], **kw)
    return {"crs": dst, "transform": tr, "height": h, "width": w, "warp": True}


def warp(arr: np.ndarray, g: dict, tg: dict, categorical: bool, resampling: str) -> np.ndarray:
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    if not tg["warp"]:
        return arr
    out = np.full((tg["height"], tg["width"]), OUT_NODATA, dtype=np.float32)
    rs = Resampling.nearest if categorical else Resampling[resampling]
    src = np.where(np.isfinite(arr), arr, OUT_NODATA).astype(np.float32)
    kw = ({"gcps": g["gcps"], "src_crs": g["gcp_crs"]} if "gcps" in g
          else {"src_transform": g["transform"], "src_crs": g["crs"]})
    reproject(src, out, dst_transform=tg["transform"], dst_crs=tg["crs"], src_nodata=OUT_NODATA,
              dst_nodata=OUT_NODATA, resampling=rs, **kw)
    return out


# ---------------------------------------------------------------------------
# Per-scene products
# ---------------------------------------------------------------------------

def read_layer(path) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        nd = src.nodata
    if nd is not None:
        a[a == nd] = np.nan
    return a


def scene_time(scene: dict, record: dict | None):
    for v in ((record or {}).get("sit_fuse_timestamp"), (record or {}).get("scene_time")):
        if v:
            return pd.Timestamp(v).tz_convert("UTC") if pd.Timestamp(v).tzinfo else pd.Timestamp(v, tz="UTC")
    for s in (scene.get("pair_id", ""), scene.get("head_outputs", "")):
        m = TIME_RE.search(Path(str(s)).name)
        if m:
            return pd.Timestamp(pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S"), tz="UTC")
    return pd.NaT


def write_scene_raster(path: Path, arrays: dict, tg: dict, tags: dict, cog: bool) -> None:
    import rasterio
    import rasterio.shutil

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.tif") if cog else path
    prof = {"driver": "GTiff", "height": tg["height"], "width": tg["width"], "count": len(arrays),
            "dtype": "float32", "nodata": OUT_NODATA, "crs": tg["crs"], "transform": tg["transform"],
            "compress": "deflate", "tiled": True, "BIGTIFF": "IF_SAFER"}
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
        for i, (name, a) in enumerate(arrays.items(), start=1):
            dst.write(np.where(np.isfinite(a), a, OUT_NODATA).astype(np.float32), i)
            info = layer_info(name)
            dst.set_band_description(i, name)
            dst.update_tags(i, layer=name, **info)
    if cog:
        rasterio.shutil.copy(tmp, path, driver="COG", compress="DEFLATE")
        os.remove(tmp)


def region_polygons(mask: np.ndarray, layers: dict, summary: list, tg: dict, area: np.ndarray | None,
                    min_pixels: int) -> list[dict]:
    """Connected regions of `mask` -> polygons with per-region statistics."""
    from rasterio.features import shapes
    from scipy.ndimage import label as cc_label
    from shapely.geometry import shape
    from shapely.ops import unary_union

    lab, n = cc_label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return []
    idx = lab.ravel()
    npx = np.bincount(idx, minlength=n + 1)
    stats = {"n_pixels": npx}
    if area is not None:
        stats["area_km2"] = np.bincount(idx, weights=np.nan_to_num(area.ravel()), minlength=n + 1)
    for name in summary:
        a = layers.get(name)
        if a is None:
            continue
        v = a.ravel()
        ok = np.isfinite(v)
        s = np.bincount(idx[ok], weights=v[ok], minlength=n + 1)
        c = np.bincount(idx[ok], minlength=n + 1)
        key = "selective_retained_fraction" if name == "selective_flag" else f"mean_{name}"
        with np.errstate(invalid="ignore", divide="ignore"):
            stats[key] = s / c
        if name == "p_1":
            mx = np.full(n + 1, np.nan)
            np.fmax.at(mx, idx[ok], v[ok])
            stats["max_p_1"] = mx
    geoms: dict[int, list] = {}
    for geom, val in shapes(lab.astype(np.int32), mask=mask, transform=tg["transform"], connectivity=8):
        geoms.setdefault(int(val), []).append(shape(geom))
    rows = []
    for k, parts in geoms.items():
        if npx[k] < min_pixels:
            continue
        rows.append({"region_id": k, "geometry": unary_union(parts),
                     **{s: (float(v[k]) if np.isfinite(v[k]) else None) for s, v in stats.items()}})
    return rows


def footprint(valid: np.ndarray, tg: dict):
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union

    parts = [shape(g) for g, v in shapes(valid.astype(np.uint8), mask=valid, transform=tg["transform"])]
    if not parts:
        return None
    res = abs(tg["transform"].a)
    return unary_union(parts).buffer(res * 0.01).simplify(res, preserve_topology=True)


def pixel_points(arrays: dict, tg: dict, stride: int) -> pd.DataFrame:
    from rasterio.transform import xy

    ref = next(iter(arrays.values()))
    rr, cc = np.mgrid[0:ref.shape[0]:stride, 0:ref.shape[1]:stride]
    rr, cc = rr.ravel(), cc.ravel()
    vals = {k: a[rr, cc] for k, a in arrays.items()}
    ok = np.zeros(rr.size, bool)
    for a in vals.values():
        ok |= np.isfinite(a)
    x, y = xy(tg["transform"], rr[ok], cc[ok])
    return pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y), "row": rr[ok], "col": cc[ok],
                         **{k: v[ok] for k, v in vals.items()}})


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _write_vectors(gdf, stem: Path, formats) -> list[str]:
    out = []
    if gdf is None or gdf.empty:
        return out
    if "parquet" in formats:
        gdf.to_parquet(f"{stem}.parquet", index=False)
        out.append(f"{stem}.parquet")
    if "gpkg" in formats:
        g = gdf.copy()
        for c in g.columns:  # GPKG has no tz-aware datetime type
            if isinstance(g[c].dtype, pd.DatetimeTZDtype):
                g[c] = g[c].dt.tz_convert("UTC").dt.tz_localize(None)
        g.to_file(f"{stem}.gpkg", layer=stem.name, driver="GPKG")
        out.append(f"{stem}.gpkg")
    return out


def area_columns(a_all: dict, scene: dict, conf: dict | None, mode: str) -> dict:
    """Footprint area columns (km2 or pixels, as in the inference manifest)."""
    keep = {k: v for k, v in a_all.items()
            if mode in ("analytic", "all") or not k.startswith(("analytic_lo", "analytic_hi"))}
    if mode == "none":
        keep = {k: v for k, v in keep.items() if not k.startswith(("analytic", "ensemble_lo", "ensemble_hi"))}
    out = {f"area_{k}_km2": v for k, v in keep.items()}
    if conf and mode in ("conformal", "all") and "expected" in a_all and "analytic_sd" in a_all:
        groups = conf.get("groups") or {}
        g = conformal_group_key(scene, conf.get("group")) if groups else None
        q, used = (groups[g]["q_hat"], g) if g in groups else (conf["q_hat"], "pooled")
        e, sd = float(a_all["expected"]), float(a_all["analytic_sd"])
        out.update(area_conformal_lo_km2=max(0.0, e - q * sd), area_conformal_hi_km2=e + q * sd,
                   area_conformal_level=conf.get("level"), area_conformal_group=used)
    return out


def export_model(cfg: dict, model: str, layer_dir: Path, out_dir: Path, records: dict) -> dict:
    import geopandas as gpd
    from rasterio.crs import CRS

    gx = cfg.get("geo_export", {}) or {}
    rcfg, vcfg, pcfg = gx.get("raster", {}) or {}, gx.get("vectors", {}) or {}, gx.get("pixels", {}) or {}
    manifest = load_json(layer_dir / "inference_manifest.json")
    want = gx.get("layers", DEFAULT_LAYERS)
    vec_crs = CRS.from_user_input(vcfg.get("crs", "EPSG:4326"))
    mask_layer = vcfg.get("mask_layer", "pred_label")
    summary = vcfg.get("summary_layers", DEFAULT_SUMMARY)
    formats = vcfg.get("formats", ["parquet", "gpkg"])
    ev_dir = gx.get("evaluation_dir") or (cfg.get("evaluation", {}) or {}).get("output_dir")
    ev_df = None
    if ev_dir and (Path(ev_dir) / model / "per_scene_metrics.csv").exists():
        ev_df = pd.read_csv(Path(ev_dir) / model / "per_scene_metrics.csv").set_index("pair_id")

    iv_mode = str(gx.get("area_intervals", "conformal")).lower()
    conf = None
    cpath = gx.get("area_calibration") or (Path(ev_dir) / model / "area_interval_calibration.json" if ev_dir else None)
    if iv_mode in ("conformal", "all"):
        if cpath and Path(cpath).exists():
            conf = (load_json(cpath) or {}).get("class_1")
        if not conf:
            LOG.warning("%s: no area_interval_calibration.json (%s); conformal area intervals omitted", model, cpath)

    polys, foots, pix, skipped, written = [], [], [], [], []
    for scene in manifest["scenes"]:
        if scene.get("skipped"):
            continue
        pid = scene["pair_id"]
        lpaths = scene.get("layers", {})
        names = [n for n in (want or list(lpaths)) if n in lpaths]
        extra = [n for n in set(summary) | {mask_layer} if n in lpaths and n not in names]
        if not names:
            skipped.append({"pair_id": pid, "reason": "no requested layers"})
            continue
        rec = records.get(pid) or records.get(scene.get("head_outputs"))
        src, g = resolve_grid(scene, lpaths[names[0]], rec)
        if g is None:
            LOG.warning("%s: no georeferencing in layer, head outputs or context-free GeoTIFF; skipped", pid)
            skipped.append({"pair_id": pid, "reason": "no georeferencing"})
            continue
        tg = target_grid(g, rcfg.get("dst_crs"), rcfg.get("resolution"))
        rs = rcfg.get("resampling", "nearest")
        arrays = {n: warp(read_layer(lpaths[n]), g, tg, n in CATEGORICAL, rs) for n in names + extra}
        t = scene_time(scene, rec)
        meta = {"pair_id": pid, "model": model, "scene_time": t, "date": scene.get("date"),
                "classifier_split": scene.get("classifier_split"), "split_stratum": scene.get("split_stratum"),
                "product": scene.get("product")}

        rpath = out_dir / "rasters" / f"{model}_{pid}.tif"
        if rcfg.get("enabled", True):
            tags = {**{k: v for k, v in meta.items() if v is not None}, "georef_source": src,
                    "threshold": manifest.get("threshold"),
                    "selective_measure": (manifest.get("selective") or {}).get("measure"),
                    "head_outputs": scene.get("head_outputs")}
            write_scene_raster(rpath, {n: arrays[n] for n in names}, tg, tags, bool(rcfg.get("cog", True)))
            written.append(str(rpath))

        area = pixel_area_km2({"crs": tg["crs"], "transform": tg["transform"],
                               "height": tg["height"], "width": tg["width"]})
        grid_crs = CRS.from_user_input(tg["crs"])
        if vcfg.get("enabled", True) and mask_layer in arrays:
            m = arrays[mask_layer] == float(vcfg.get("positive_value", 1))
            rows = region_polygons(m, arrays, summary, tg, area, int(vcfg.get("min_pixels", 1)))
            if rows:
                gdf = gpd.GeoDataFrame([{**meta, **r} for r in rows], geometry="geometry", crs=grid_crs)
                polys.append(gdf.to_crs(vec_crs))
        if gx.get("footprints", True):
            valid = np.zeros((tg["height"], tg["width"]), bool)
            for a in arrays.values():
                valid |= np.isfinite(a)
            fp = footprint(valid, tg)
            if fp is not None:
                a_all = ((scene.get("areas") or {}).get("all_valid") or {}).get("class_1") or {}
                row = {**meta, "geometry": fp, "raster": str(rpath) if rcfg.get("enabled", True) else None,
                       "georef_source": src, "n_valid": scene.get("n_valid"),
                       "excluded_fraction": scene.get("excluded_fraction"),
                       **area_columns(a_all, scene, conf, iv_mode)}
                if ev_df is not None and pid in ev_df.index:
                    for c in ("target_macro_iou", "error_rate", "classwise_ece", "c1_ref_area_km2",
                              "selective_coverage", "selective_retained_error"):
                        if c in ev_df:
                            row[f"eval_{c}"] = ev_df.at[pid, c]
                foots.append(gpd.GeoDataFrame([row], geometry="geometry", crs=grid_crs).to_crs(vec_crs))
        if pcfg.get("enabled", False):
            pl = [n for n in (pcfg.get("layers") or names) if n in arrays]
            df = pixel_points({n: arrays[n] for n in pl}, tg, int(pcfg.get("stride", 4)))
            g2 = gpd.GeoDataFrame({**meta, **df.drop(columns=["x", "y"]).to_dict("list")},
                                  geometry=gpd.points_from_xy(df["x"], df["y"]), crs=grid_crs)
            pix.append(g2.to_crs(vec_crs))

    outs = {"rasters": written, "skipped": skipped}
    cat = lambda L: pd.concat(L, ignore_index=True) if L else None
    outs["smoke_polygons"] = _write_vectors(cat(polys), out_dir / f"{model}_smoke_polygons", formats)
    outs["scene_footprints"] = _write_vectors(cat(foots), out_dir / f"{model}_scene_footprints", formats)
    if pix:
        outs["pixels"] = _write_vectors(cat(pix), out_dir / f"{model}_pixels", ["parquet"])
    LOG.info("%s: %d rasters, %d polygons, %d footprints, %d skipped", model, len(written),
             sum(len(p) for p in polys), len(foots), len(skipped))
    return outs


def run(cfg: dict, models=None, inference_dir=None, output_dir=None) -> dict:
    gx = cfg.get("geo_export", {}) or {}
    ev = cfg.get("evaluation", {}) or {}
    inf = cfg.get("inference", {}) or {}
    out = Path(cfg["output_dir"])
    out_dir = Path(output_dir or gx.get("output_dir", out / "geo"))
    tmpl = str(inference_dir or gx.get("inference_dir_template")
               or ev.get("inference_dir_template") or out / "inference" / "{model}")
    models = models or gx.get("models") or ev.get("models") or [inf.get("model")]
    models = [m for m in models if m]
    if not models:
        raise ValueError("No models: pass --models or set geo_export.models / evaluation.models.")

    records = {}
    sf = gx.get("scene_file", inf.get("scene_file"))
    if sf and Path(sf).exists():
        for r in load_matchups(sf, require_truth=False):
            records[r["pair_id"]] = r
            if r.get("head_outputs"):
                records[r["head_outputs"]] = r

    summary = {}
    for m in models:
        layer_dir = Path(tmpl.format(model=m))
        if not (layer_dir / "inference_manifest.json").exists():
            LOG.warning("%s: no inference_manifest.json in %s; skipped", m, layer_dir)
            continue
        summary[m] = export_model(cfg, m, layer_dir, out_dir / m, records)
    save_json(out_dir / "geo_products_manifest.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_config", type=Path)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--inference-dir", default=None, help="template with {model}, or a single directory")
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args()
    run(load_yaml(a.yaml_config), a.models, a.inference_dir, a.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
