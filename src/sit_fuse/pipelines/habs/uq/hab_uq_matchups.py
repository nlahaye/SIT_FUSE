"""
Point (in-situ) to pixel (SIT-FUSE product) matchups for HAB UQ.

Two input modes per sensor/day:
  head_outputs  rasters written by sit_fuse/train/export_head_outputs.py
                (parent/child top-k probabilities; richest UQ features)
  label_only    the existing context-free cluster GeoTIFFs produced by the HAB
                pipeline ({PREFIX}.{YYYYMMDD}.L3m.DAY[.no_heir].tif and optional
                .proba.tif). Head bands are synthesized so the fire/smoke UQ
                code (scene_feature_frame, ContextEncoder) runs unchanged.

Matching replicates insitu_hab_to_multi_hist: every valid pixel whose centre
is within `radius_degrees` (planar degree distance) of a sample is a match;
same-day by default, optionally +/- `max_day_offset` days with a temporal
weight. Matching is vectorized with a KD-tree instead of nested iterrows.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sit_fuse.train.sf_uq_utils import (
    InvalidClusters, head_valid_mask, read_head_outputs, scene_feature_frame)
from sit_fuse.pipelines.habs.uq.hab_uq_common import (
    GEOTIFF_SCALE, NODATA, SENSORS, find_product, nominal_overpass_utc, pixel_centers)

LOG = logging.getLogger("hab-uq-matchups")


# ---------------------------------------------------------------------------
# Raster inputs
# ---------------------------------------------------------------------------

def _read_single(path):
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        prof = src.profile.copy()
        nod = src.nodata
    if nod is not None:
        arr[arr == nod] = np.nan
    return arr, prof


def label_only_bands(cluster_tif: str, num_child: int, no_heir: bool = False,
                     proba_tif: str | None = None,
                     no_heir_value: str = "parent") -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Synthesize the export_head_outputs band set from a cluster GeoTIFF.

    heir product:    joint = round(v * 1000); parent = joint // num_child
    no_heir product: v * 1000 is the parent (root) cluster when
                     no_heir_value == 'parent' -> collapsed joint = parent * num_child
                     (the SIT-FUSE convention for "no child head");
                     use 'joint' if the no_heir file already stores joint ids.
    proba_tif (optional) gives the winning-cluster probability (0-1 or 0-100);
    otherwise top-1 probabilities are 1 (degenerate but valid head features).
    """
    v, prof = _read_single(cluster_tif)
    valid = np.isfinite(v) & (v >= 0)
    lab = np.where(valid, np.rint(v * GEOTIFF_SCALE), -1).astype(np.int64)
    if no_heir and no_heir_value == "parent":
        parent = lab
        joint = np.where(valid, parent * num_child, -1)
        child = np.where(valid, 0, -1)
        has_child = np.zeros(v.shape, np.float32)
    else:
        joint = lab
        parent = np.where(valid, joint // num_child, -1)
        child = np.where(valid, joint % num_child, -1)
        has_child = valid.astype(np.float32)

    p = np.ones(v.shape)
    if proba_tif:
        pr, _ = _read_single(proba_tif)
        if pr.shape == v.shape:
            pr = np.where(np.isfinite(pr), pr, 1.0)
            p = pr / 100.0 if np.nanmax(pr) > 1.0 + 1e-6 else pr
            p = np.clip(p, 0.0, 1.0)

    f = lambda a: np.where(valid, a, NODATA).astype(np.float32)
    ent = np.where(p < 1, -(p * np.log(np.clip(p, 1e-7, 1)) + (1 - p) * np.log(np.clip(1 - p, 1e-7, 1))), 0.0)
    bands = {
        "joint_label": f(joint), "parent_label": f(parent), "child_label": f(child),
        "child_head_present": f(has_child),
        "parent_top1_id": f(parent), "parent_top1_prob": f(p),
        "parent_topk_mass": f(p), "parent_maxprob": f(p), "parent_margin": f(p),
        "parent_entropy": f(ent),
        "r1_child_top1_id": f(np.where(has_child > 0, child, -1)),
        "r1_child_top1_prob": f(np.where(has_child > 0, 1.0, 0.0)),
        "r1_child_topk_mass": f(np.where(has_child > 0, 1.0, 0.0)),
        "child_maxprob": f(np.where(has_child > 0, 1.0, 0.0)),
        "child_margin": f(np.where(has_child > 0, 1.0, 0.0)),
        "child_entropy": f(np.zeros(v.shape)), "joint_prob": f(p),
    }
    # NODATA in id bands must stay invalid for head_valid_mask
    bands["joint_label"] = np.where(valid, joint, NODATA).astype(np.float32)
    return bands, prof


def load_scene(search_dirs, sensor: str, date, uq_cfg: dict[str, Any]):
    """Return (bands, profile, source_path, mode) or None when no product exists."""
    mode = uq_cfg.get("input_mode", "auto")
    num_child = int(uq_cfg["num_child"])
    pats = uq_cfg.get("file_patterns")
    if mode in ("auto", "head_outputs"):
        p = find_product(search_dirs, "head_outputs", sensor, date, patterns=pats)
        if p:
            bands, prof, _ = read_head_outputs(p)
            return bands, prof, p, "head_outputs"
        if mode == "head_outputs":
            return None
    level = uq_cfg.get("label_level", "heir")       # heir | no_heir
    p = find_product(search_dirs, level, sensor, date, patterns=pats)
    if not p:
        return None
    pp = None
    if uq_cfg.get("use_proba", True):
        pp = find_product(search_dirs, f"{level}_proba", sensor, date, patterns=pats)
    bands, prof = label_only_bands(p, num_child, no_heir=(level == "no_heir"), proba_tif=pp,
                                   no_heir_value=uq_cfg.get("no_heir_value", "parent"))
    return bands, prof, p, "label_only"


# ---------------------------------------------------------------------------
# Spatial matching
# ---------------------------------------------------------------------------

class PixelIndex:
    """KD-tree over valid pixel centres (degrees) of one scene."""

    def __init__(self, profile, valid: np.ndarray):
        lat, lon = pixel_centers(profile)
        crs = profile.get("crs")
        if crs is not None and not getattr(crs, "is_geographic", True):
            from rasterio.warp import transform as _tr
            xs, ys = _tr(crs, "EPSG:4326", lon.ravel(), lat.ravel())
            lon, lat = np.reshape(xs, lon.shape), np.reshape(ys, lat.shape)
        self.rows, self.cols = np.nonzero(valid)
        self.lat = lat[self.rows, self.cols]
        self.lon = lon[self.rows, self.cols]
        self.tree = cKDTree(np.column_stack([self.lon, self.lat])) if self.rows.size else None

    def query(self, lon, lat, radius_deg):
        if self.tree is None:
            return [np.zeros(0, np.int64) for _ in np.atleast_1d(lon)]
        # production uses strict '<'; shrink by a hair to match
        return self.tree.query_ball_point(np.column_stack([lon, lat]), r=radius_deg * (1 - 1e-9))


def temporal_weight(day_offset, hours_offset, cfg: dict[str, Any]) -> np.ndarray:
    """
    w = exp(-|dt_days| / tau_days) * exp(-(dt_hours / sigma_hours)^2 / 2)
    tau_days ~ bloom decorrelation time (default 2 d); sigma_hours applies only
    when the sample time-of-day is known (default 6 h).
    """
    tau = float(cfg.get("tau_days", 2.0))
    sig = float(cfg.get("sigma_hours", 6.0))
    w = np.exp(-np.abs(np.asarray(day_offset, float)) / max(tau, 1e-6))
    h = np.asarray(hours_offset, float)
    w = w * np.where(np.isfinite(h), np.exp(-0.5 * (h / max(sig, 1e-6)) ** 2), 1.0)
    return w


# ---------------------------------------------------------------------------
# Matchup tables
# ---------------------------------------------------------------------------

def build_sensor_matchups(samples: pd.DataFrame, sensor: str, search_dirs,
                          uq_cfg: dict[str, Any], features_cfg: dict[str, Any],
                          invalid: InvalidClusters | None = None,
                          keep_scene_frames: bool = False):
    """
    For one sensor, match every in-situ sample to pixels of the same-day
    (or +/- k day) product.

    Returns
      rows    DataFrame: one row per (sample, pixel, scene) with fold-independent
              features (scene_feature_frame), sample metadata and weights.
      counts  DataFrame for ContextEncoder.fit: level, cluster, class, count,
              date, split_group, sample_uid, split, fold. count_mode:
                unique_per_sample (production-equivalent histogram counts),
                pixels, fractional (1/n_pixels per sample).
    """
    radius = float(uq_cfg.get("radius_degrees", 0.09))
    max_off = int(uq_cfg.get("max_day_offset", 0))
    tcfg = uq_cfg.get("temporal_weighting", {})
    num_child = int(uq_cfg["num_child"])
    count_mode = uq_cfg.get("count_mode", "unique_per_sample")

    samples = samples.copy()
    rows_out, counts_out = [], []
    scene_dates = sorted({d + pd.Timedelta(days=k) for d in samples["date"].unique()
                          for k in range(-max_off, max_off + 1)})
    for sdate in scene_dates:
        sc = load_scene(search_dirs, sensor, sdate, uq_cfg)
        if sc is None:
            continue
        bands, prof, src, mode = sc
        off = (samples["date"] - sdate).dt.days
        sub = samples[off.abs() <= max_off]
        if sub.empty:
            continue
        valid = head_valid_mask(bands, invalid)
        pix = PixelIndex(prof, valid)
        hits = pix.query(sub["Longitude"].to_numpy(), sub["Latitude"].to_numpy(), radius)
        if not any(len(h) for h in hits):
            continue
        pair_id = f"{sensor}.{pd.Timestamp(sdate).strftime('%Y%m%d')}"
        feat, _ = scene_feature_frame(bands, prof, features_cfg, pair_id, ref=None, invalid=invalid)
        rc_index = pd.Series(np.arange(len(feat)), index=pd.MultiIndex.from_arrays(
            [feat["row"].to_numpy(), feat["col"].to_numpy()]))

        t_over = nominal_overpass_utc(sensor, sdate, sub["Longitude"].to_numpy())
        for (i, srow), h, tov in zip(sub.iterrows(), hits, t_over):
            if len(h) == 0:
                continue
            h = np.asarray(h, dtype=np.int64)
            rr, cc = pix.rows[h], pix.cols[h]
            fi = rc_index.reindex(pd.MultiIndex.from_arrays([rr, cc])).to_numpy()
            fi = fi[np.isfinite(fi)].astype(np.int64)
            if fi.size == 0:
                continue
            day_off = int((srow["date"] - pd.Timestamp(sdate)).days)
            st = srow.get("sample_time_utc", pd.NaT)
            hrs = ((pd.Timestamp(st).tz_convert(None) - tov).total_seconds() / 3600.0
                   if pd.notna(st) else np.nan)
            wt = float(temporal_weight(day_off, hrs, tcfg))
            r = feat.iloc[fi].copy()
            n = len(r)
            r["sample_uid"] = srow["sample_uid"]
            r["sensor"] = sensor
            r["scene_date"] = pd.Timestamp(sdate)
            r["date"] = str(pd.Timestamp(srow["date"]).date())
            r["day_offset"] = day_off
            r["hours_offset"] = hrs
            r["ref_label"] = int(srow["ref_class"])
            for c in [k for k in srow.index if str(k).startswith("ref_soft_")]:
                r[c] = float(srow[c])
            for k in ("split", "fold", "split_group", "Location_Code", "value", "log10_value"):
                r[k] = srow[k]
            r["pix_weight"] = wt / n                     # each sample-scene sums to wt
            r["n_match_pixels"] = n
            r["input_mode"] = mode
            rows_out.append(r)

            # encoder counts (exact same-day only unless allowed)
            if day_off != 0 and not uq_cfg.get("encoder_use_offset_days", False):
                continue
            jl = r["joint_label"].to_numpy().astype(np.int64)
            pl = r["parent_label"].to_numpy().astype(np.int64)
            for level, ids in (("joint", jl), ("parent", pl)):
                if count_mode == "unique_per_sample":
                    u, cnt = np.unique(ids, return_counts=True)
                    cnt = np.ones_like(u, dtype=np.float64) * wt
                elif count_mode == "pixels":
                    u, cnt = np.unique(ids, return_counts=True)
                    cnt = cnt.astype(np.float64) * wt
                else:  # fractional
                    u, cnt = np.unique(ids, return_counts=True)
                    cnt = cnt.astype(np.float64) / n * wt
                counts_out.append(pd.DataFrame({
                    "level": level, "cluster": u, "class": int(srow["ref_class"]), "count": cnt,
                    "date": str(pd.Timestamp(srow["date"]).date()),
                    "split_group": srow["split_group"], "sample_uid": srow["sample_uid"],
                    "split": srow["split"], "fold": srow["fold"], "sensor": sensor}))
        LOG.info("%s %s: %d samples matched (%s)", sensor, pair_id, sum(len(h) > 0 for h in hits), mode)

    rows = pd.concat(rows_out, ignore_index=True) if rows_out else pd.DataFrame()
    counts = pd.concat(counts_out, ignore_index=True) if counts_out else pd.DataFrame(
        columns=["level", "cluster", "class", "count", "date", "split_group"])
    return rows, counts


def sample_level(P: np.ndarray, rows: pd.DataFrame, how: str = "mean") -> tuple[np.ndarray, pd.DataFrame]:
    """
    Aggregate pixel probabilities to one predictive distribution per
    (sample, sensor): weighted mean (linear pool) over matched pixels, the
    natural sample-level analogue of the production radius histogram.
    """
    key = rows["sample_uid"].astype(str) + "|" + rows["sensor"].astype(str)
    w = rows["pix_weight"].to_numpy(dtype=np.float64)
    codes, uniq = pd.factorize(key)
    C = P.shape[1]
    acc = np.zeros((uniq.size, C))
    ws = np.zeros(uniq.size)
    np.add.at(acc, codes, P * w[:, None])
    np.add.at(ws, codes, w)
    Ps = acc / np.maximum(ws[:, None], 1e-12)
    meta = rows.groupby(codes).agg(
        sample_uid=("sample_uid", "first"), sensor=("sensor", "first"), date=("date", "first"),
        ref_label=("ref_label", "first"), split=("split", "first"), fold=("fold", "first"),
        split_group=("split_group", "first"), Location_Code=("Location_Code", "first"),
        value=("value", "first"), log10_value=("log10_value", "first"),
        day_offset=("day_offset", "first"), hours_offset=("hours_offset", "first"),
        n_pixels=("pix_weight", "size")).reset_index(drop=True)
    meta["weight"] = ws
    soft = [c for c in rows.columns if c.startswith("ref_soft_")]
    if soft:
        meta = pd.concat([meta, rows.groupby(codes)[soft].first().reset_index(drop=True)], axis=1)
    return Ps, meta
