#!/usr/bin/env python3
"""
Build training matchups between context-free SIT-FUSE GeoTIFFs and the
timestamped fire-front rasters written by rasterize_fire_fronts.py (FEDS
fire lines and / or FIRMS-detection fronts), then write a classifier YAML,
an audit CSV and the encoder / classifier split table (matchup_splits.py).

Fire fronts are thin (a few hundred metres) relative to most SIT-FUSE scene
pixels, so the fine fire-front raster is aggregated onto each scene's own grid
(affine or GCP georeferencing) instead of nearest-neighbour resampling, which
would drop most front cells. Per scene pixel, with the fine cells it contains:
    front     if front fraction    >  scene_grid.front_min_fraction
    interior  else if interior fraction > scene_grid.interior_min_fraction
    negative  else if the valid (non-NoData) fraction >= scene_grid.min_valid_fraction
    NoData    otherwise
Scene pixels smaller than a fine cell take the fine cell at their centre.
One truth raster per scene is written (band 1 label, band 2 front fraction in
percent, NoData 255), on the scene grid, so the UQ pipeline reads it without
resampling.

Temporal matching (matching.policy): previous (latest front at or before the
scene, default; fronts are only known after the overpass), nearest, next.

Matchup records use the ADP generator schema plus
    truth_firms_fire_front / annotation_raster   per-scene truth raster
    fire_front_source_raster, fire_front_timestamp, fire_front_date,
    fire_front_source (feds | firms), fire_front_tag, signed_offset_seconds

Usage:
    python firms_fire_front_matchup_generator.py firms_fire_front_matchup.example.yaml
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import rasterio.errors
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import AffineTransformer, GCPTransformer

try:
    from sit_fuse.train.adp_matchup_generator import (
        collect_sit_fuse_records, load_yaml, parse_datetime, render_classifier_config, write_yaml)
    from sit_fuse.train.matchup_splits import finalize as finalize_splits
except ImportError:  # running from the scripts directory
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from adp_matchup_generator import (
        collect_sit_fuse_records, load_yaml, parse_datetime, render_classifier_config, write_yaml)
    from matchup_splits import finalize as finalize_splits

LOG = logging.getLogger("firms-fire-front-matchups")
TRUTH_FIELD = "truth_firms_fire_front"
DEFAULT_CLASSES = {"front": 1, "interior": 2, "negative": 0}
AUDIT_FIELDS = ["sit_fuse_path", "sit_fuse_timestamp", "fire_front_path", "fire_front_timestamp",
                "fire_front_source", "signed_offset_seconds", "pair_id", "status", "notes"]


# ---------------------------------------------------------------------------
# fire-front rasters
# ---------------------------------------------------------------------------

def load_front_index(fcfg: dict, start, end) -> pd.DataFrame:
    idx = pd.read_csv(fcfg["index_csv"])
    if idx.empty:
        raise ValueError(f"{fcfg['index_csv']} lists no rasters")
    idx["timestamp"] = pd.to_datetime(idx["timestamp"], utc=True).dt.tz_convert(None)
    if fcfg.get("sources"):
        idx = idx[idx["source"].isin(fcfg["sources"])]
    if fcfg.get("tags"):
        idx = idx[idx["tag"].isin(fcfg["tags"])]
    if fcfg.get("require_front", True) and "n_front" in idx:
        idx = idx[idx["n_front"] > 0]
    pad = pd.Timedelta(seconds=float(fcfg.get("search_pad_seconds", 86400)))
    idx = idx[(idx["timestamp"] >= pd.Timestamp(start) - pad) & (idx["timestamp"] <= pd.Timestamp(end) + pad)]
    return idx.sort_values(["timestamp", "path"]).reset_index(drop=True)


def choose_front(scene_ts, idx: pd.DataFrame, mcfg: dict):
    """Pick the fire-front raster for one scene time; returns (row, signed offset s)."""
    if idx.empty:
        return None, None
    off = (pd.Timestamp(scene_ts) - idx["timestamp"]).dt.total_seconds().to_numpy()  # >0: front earlier
    pol = str(mcfg.get("policy", "previous")).lower()
    tol = float(mcfg.get("max_time_difference_seconds", 43200))
    ok = {"previous": (off >= 0) & (off <= tol), "next": (off <= 0) & (-off <= tol),
          "nearest": np.abs(off) <= tol}.get(pol)
    if ok is None:
        raise ValueError("matching.policy must be previous | nearest | next")
    if not ok.any():
        return None, None
    cand = idx[ok].assign(_abs=np.abs(off[ok]), _off=off[ok])
    for src in mcfg.get("prefer_source") or []:
        if (cand["source"] == src).any():
            cand = cand[cand["source"] == src]
            break
    best = cand.sort_values(["_abs", "timestamp", "path"]).iloc[0]
    return best, float(best["_off"])


# ---------------------------------------------------------------------------
# aggregation onto the scene grid
# ---------------------------------------------------------------------------

def scene_transformer(src):
    gcps, gcrs = src.gcps
    if gcps:
        return GCPTransformer(gcps), (gcrs or CRS.from_epsg(4326)), gcps, gcrs
    if src.crs is None:
        raise ValueError(f"{src.name}: no CRS / georeferencing")
    return AffineTransformer(src.transform), src.crs, None, None


def aggregate_to_scene(fine_path, scene_path, classes, fill, gcfg):
    with rasterio.open(scene_path) as s:
        H, W = s.height, s.width
        tr, scrs, gcps, gcrs = scene_transformer(s)
        prof = {"driver": "GTiff", "height": H, "width": W, "count": 2, "dtype": "uint8",
                "nodata": fill, "compress": "deflate"}
        if not gcps:
            prof.update(crs=s.crs, transform=s.transform)
    with rasterio.open(fine_path) as f:
        A = f.read(1)
        ft = f.transform
        fcrs = f.crs or CRS.from_epsg(4326)
    fr, fc = np.indices(A.shape)
    xs, ys = ft * (fc.ravel() + 0.5, fr.ravel() + 0.5)
    xs, ys = np.asarray(xs), np.asarray(ys)
    if CRS.from_user_input(scrs) != CRS.from_user_input(fcrs):
        xs, ys = Transformer.from_crs(fcrs, scrs, always_xy=True).transform(xs, ys)
    sr, sc = tr.rowcol(xs, ys)
    sr, sc = np.asarray(sr), np.asarray(sc)
    a = A.ravel()
    inside = (sr >= 0) & (sr < H) & (sc >= 0) & (sc < W)
    pix = (sr * W + sc)[inside]
    av = a[inside]
    n = H * W
    tot = np.bincount(pix, minlength=n).astype(np.int64)
    cnt = {k: np.bincount(pix[av == v], minlength=n) for k, v in classes.items()}
    nfill = np.bincount(pix[av == fill], minlength=n)
    # scene pixels smaller than a fine cell: take the fine cell at the pixel centre
    empty = np.flatnonzero(tot == 0)
    if empty.size:
        er, ec = np.divmod(empty, W)
        ex, ey = tr.xy(er, ec)
        ex, ey = np.asarray(ex, float), np.asarray(ey, float)
        if CRS.from_user_input(scrs) != CRS.from_user_input(fcrs):
            ex, ey = Transformer.from_crs(scrs, fcrs, always_xy=True).transform(ex, ey)
        cc, rr = ~ft * (ex, ey)
        rr, cc = np.floor(rr).astype(int), np.floor(cc).astype(int)
        ok = (rr >= 0) & (rr < A.shape[0]) & (cc >= 0) & (cc < A.shape[1])
        vals = A[rr[ok], cc[ok]]
        tgt = empty[ok]
        tot[tgt] = 1
        for k, v in classes.items():
            cnt[k][tgt] += vals == v
        nfill[tgt] += vals == fill
    valid = tot - nfill
    with np.errstate(divide="ignore", invalid="ignore"):
        ffront = np.where(valid > 0, cnt["front"] / np.maximum(valid, 1), 0.0)
        fint = np.where(valid > 0, cnt["interior"] / np.maximum(valid, 1), 0.0)
        fvalid = np.where(tot > 0, valid / np.maximum(tot, 1), 0.0)
    lab = np.full(n, fill, np.uint8)
    keep = (valid > 0) & (fvalid >= float(gcfg.get("min_valid_fraction", 0.5)))
    lab[keep] = classes["negative"]
    lab[keep & (fint > float(gcfg.get("interior_min_fraction", 0.0)))] = classes["interior"]
    lab[(cnt["front"] > 0) & (ffront > float(gcfg.get("front_min_fraction", 0.0)))] = classes["front"]
    frac = np.where(valid > 0, np.round(100 * ffront), fill).astype(np.uint8)
    return lab.reshape(H, W), frac.reshape(H, W), prof, (gcps, gcrs)


def write_scene_truth(path, lab, frac, prof, gcp_info, tags):
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():  # GCP scenes: no geotransform until the GCPs are set
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        dst = rasterio.open(path, "w", **prof)
    with dst:
        dst.write(lab, 1)
        dst.write(frac, 2)
        dst.set_band_description(1, "fire_front_label")
        dst.set_band_description(2, "front_fraction_percent")
        if gcp_info[0]:
            dst.gcps = gcp_info
        dst.update_tags(**{k: str(v) for k, v in tags.items()})


# ---------------------------------------------------------------------------

def build_pair_id(scene, row, group_by):
    ft = row["timestamp"]
    return {"front_timestamp": f"fire_{ft:%Y%m%dT%H%M%SZ}_{row['tag']}",
            "front_day": f"fire_{ft:%Y%m%d}",
            "sit_fuse_timestamp": f"fire_{scene.timestamp:%Y%m%dT%H%M%SZ}",
            "sit_fuse_day": f"fire_{scene.timestamp:%Y%m%d}"}[group_by]


def validate_config(cfg):
    for k in ("time_range", "sit_fuse", "fire_fronts", "matching", "scene_grid", "classifier_config", "output"):
        if k not in cfg:
            raise ValueError(f"Missing required top-level config key: {k!r}")
    if "index_csv" not in cfg["fire_fronts"]:
        raise ValueError("Missing fire_fronts.index_csv (written by rasterize_fire_fronts.py)")
    for k in ("classifier_yaml", "audit_csv"):
        if k not in cfg["output"]:
            raise ValueError(f"Missing output.{k}")
    g = str(cfg["matching"].get("group_by", "front_timestamp"))
    if g not in ("front_timestamp", "front_day", "sit_fuse_timestamp", "sit_fuse_day"):
        raise ValueError("matching.group_by must be front_timestamp | front_day | sit_fuse_timestamp | sit_fuse_day")


def main(config_path: Path) -> None:
    cfg = load_yaml(config_path)
    validate_config(cfg)
    start, end = parse_datetime(cfg["time_range"]["start"]), parse_datetime(cfg["time_range"]["end"])
    mcfg, gcfg, fcfg = cfg["matching"], cfg["scene_grid"], cfg["fire_fronts"]
    classes = {**DEFAULT_CLASSES, **(fcfg.get("classes") or {})}
    fill = int(fcfg.get("fill_value", 255))
    truth_dir = Path(gcfg.get("truth_dir", "./fire_front_truth"))
    group_by = str(mcfg.get("group_by", "front_timestamp"))
    overwrite = bool(gcfg.get("overwrite", False))

    scenes, skipped = collect_sit_fuse_records(cfg["sit_fuse"], start, end)
    idx = load_front_index(fcfg, start, end)
    LOG.info("SIT-FUSE scenes: %d | fire-front rasters: %d", len(scenes), len(idx))

    matchups, audit = [], []
    for sc in scenes:
        base = {"sit_fuse_path": str(sc.path), "sit_fuse_timestamp": f"{sc.timestamp:%Y-%m-%dT%H:%M:%SZ}"}
        row, off = choose_front(sc.timestamp, idx, mcfg)
        if row is None:
            audit.append({**base, "status": "no_fire_front_match",
                          "notes": f"policy={mcfg.get('policy', 'previous')} "
                                   f"tol={mcfg.get('max_time_difference_seconds', 43200)}s"})
            continue
        fpath = Path(row["path"])
        out = truth_dir / f"fire_front_{row['timestamp']:%Y%m%dT%H%M%SZ}_{row['tag']}__{sc.path.name.split('.')[0]}.tif"
        if overwrite or not out.exists():
            lab, frac, prof, gcp_info = aggregate_to_scene(fpath, sc.path, classes, fill, gcfg)
            write_scene_truth(out, lab, frac, prof, gcp_info,
                              {"fire_front_source_raster": fpath, "sit_fuse_context_free": sc.path,
                               "signed_offset_seconds": off})
        else:
            with rasterio.open(out) as src:
                lab = src.read(1)
        n_front = int((lab == classes["front"]).sum())
        n_valid = int((lab != fill).sum())
        ff = {**base, "fire_front_path": str(out.resolve()),
              "fire_front_timestamp": f"{row['timestamp']:%Y-%m-%dT%H:%M:%SZ}",
              "fire_front_source": row["source"], "signed_offset_seconds": int(off)}
        if n_valid == 0:
            audit.append({**ff, "status": "no_overlap", "notes": "fire-front raster does not overlap the scene"})
            continue
        if n_front == 0 and gcfg.get("drop_scenes_without_front", False):
            audit.append({**ff, "status": "no_front_in_scene", "notes": ""})
            continue
        pid = build_pair_id(sc, row, group_by)
        matchups.append({
            "pair_id": pid, "label": "fire", "product": "fire",
            "sit_fuse_context_free": str(sc.path),
            TRUTH_FIELD: str(out.resolve()), "annotation_raster": str(out.resolve()),
            "fire_front_source_raster": str(fpath.resolve()),
            "sit_fuse_timestamp": base["sit_fuse_timestamp"],
            "fire_front_timestamp": ff["fire_front_timestamp"],
            "fire_front_date": f"{row['timestamp']:%Y-%m-%d}",
            "fire_front_source": row["source"], "fire_front_tag": row["tag"],
            "time_offset_seconds": int(abs(off)), "signed_offset_seconds": int(off),
            "match_method": f"{mcfg.get('policy', 'previous')}_timestamp",
            "scene_front_pixels": n_front, "scene_valid_pixels": n_valid})
        audit.append({**ff, "pair_id": pid, "status": "matched",
                      "notes": f"front_pixels={n_front} valid_pixels={n_valid}"})

    matchups, split_summary = finalize_splits(matchups, cfg, truth_field=TRUTH_FIELD,
                                              truth_date_field="fire_front_date", log=LOG)
    for s in skipped:
        audit.append({"sit_fuse_path": s["path"], "status": "skipped_sit_fuse", "notes": s["reason"]})

    ccfg = render_classifier_config(load_yaml(Path(cfg["classifier_config"]["base_yaml"])), matchups, cfg)
    md = ccfg.setdefault("matchup_metadata", {})
    md.update({"truth_source": "firms_fire_front_scene_grid_geotiff",
               "fire_front_classes": classes, "fire_front_fill_value": fill})
    md.setdefault("sample_schema", {})["truth_mask_path"] = TRUTH_FIELD
    if split_summary is not None:
        md["splits"] = split_summary
    write_yaml(ccfg, Path(cfg["output"]["classifier_yaml"]))
    ap = Path(cfg["output"]["audit_csv"])
    ap.parent.mkdir(parents=True, exist_ok=True)
    with ap.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=AUDIT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(audit)
    st = pd.Series([r["status"] for r in audit]).value_counts().to_dict()
    LOG.info("Audit: %s", st)
    LOG.info("Matched scenes: %d (with front pixels: %d)", len(matchups),
             sum(m["scene_front_pixels"] > 0 for m in matchups))
    LOG.info("Classifier configuration written: %s", Path(cfg["output"]["classifier_yaml"]).resolve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    p = argparse.ArgumentParser(description="SIT-FUSE / FIRMS fire-front matchups")
    p.add_argument("yaml_config", type=Path)
    a = p.parse_args()
    try:
        main(a.yaml_config)
    except Exception as exc:
        LOG.exception("Fire-front matchup generation failed: %s", exc)
        sys.exit(1)
