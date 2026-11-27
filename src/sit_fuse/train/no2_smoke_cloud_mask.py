#!/usr/bin/env python
"""
Smoke-aware cloud screening of TEMPO L3 NO2 with the SIT-FUSE GOES/TEMPO fused smoke product, and the coverage
comparison against the standard cloud screen.

    python no2_smoke_cloud_mask.py -c no2_smoke_mask.example.yaml --stage pull      # TEMPO_NO2_L3 per case
    python no2_smoke_cloud_mask.py -c no2_smoke_mask.example.yaml --stage mask -n 24
    python no2_smoke_cloud_mask.py -c no2_smoke_mask.example.yaml --stage compare
    python no2_smoke_cloud_mask.py -c no2_smoke_mask.example.yaml --stage all -n 24 [--overwrite]

Inputs
  * TEMPO_NO2_L3 scans (full files or _subset.nc from preprocessing/data_pull/pull_tempo_no2_l3.py).
    Cloud mask: support_data/eff_cloud_fraction (ECF). Quality: product/main_data_quality_flag.
    Standard screen (baseline): QA <= qa_max and ECF < cloud_threshold. cloud_threshold `auto` uses the
    version-specific recommendation (V03: 0.2, V04: 0.1; TEMPO trace gas user guide V1.x / V2.0).
  * Fused smoke rasters from fuse_multiscale_products.py (<output_dir>/<grid>/fused/rasters/fused_smoke_*.tif;
    bands p_1, ensemble_p1_q05, tempo_age_minutes, ...; tag threshold = fused decision threshold).

Method (per case bbox and NO2 scan)
  1. NO2 window read on its native 0.02 deg grid (north-up).
  2. Fused frames with scene_time in [scan start - window, scan end + window] are averaged onto the NO2 cells
     (area-weighted mean probability, the same smoke change of support as the fusion step).
  3. Time matching per cell: the TEMPO observation time of the cell is recovered from the fused rasters as
     scene_time - tempo_age_minutes (kept when it falls inside the scan); otherwise the scan midpoint is used.
     Each cell takes the fused frame closest to that time (|dt| <= max_dt_minutes, else no smoke information).
  4. Smoke-recovered cells: QA <= qa_max, ECF >= cloud_threshold, smoke variant >= its threshold and, optionally,
     ECF < recovery_max_cloud_fraction (keeps optically thick cloud screened) and cloud pressure >= a minimum.
     Updated screen = baseline OR smoke-recovered. Variants: fused p_1 (default) and the conservative fused q05
     (lower 90% bound), optionally the single-sensor bands tempo_p1 / goes_p1 with their own thresholds.
  5. Per cell class (primary variant / threshold): 0 no retrieval, 1 clear (baseline), 2 cloud-screened (no smoke),
     3 smoke-recovered, 4 smoke under cloud above the recovery cap, 5 QA-rejected.
  6. Coverage rows per scan x ECF threshold x recovery cap x variant: observed / baseline / updated area, smoke
     area and the share of it retained by each screen, added area, area-weighted NO2 statistics for added cells
     vs clear smoke cells and vs clear non-smoke cells, ECF / cloud pressure of added vs cloud-screened cells.

  7. Fire proximity (optional `fire:` section; the mask is not changed): SIT-FUSE fire products (GOES, TEMPO and/or
     fused fire geo rasters) with scene times in [scan start - window, scan end + window] are combined (union of
     decision masks, max probability) on the NO2 grid; each cell gets the distance to the nearest fire cell and
     near_fire = distance <= buffer_km (NaN where no fire product covers the scan). Coverage rows split the
     baseline / recovered areas and NO2 means into near-fire and away-from-fire cells, and the figures mark them.

Caveat: recovered cells keep the operational AMF, which treats the smoke as effective cloud (an implicit aerosol
correction); their VCDs carry larger AMF uncertainty and are flagged (class 3) so they can be analysed separately.
The fused smoke product itself is ADP-calibrated and not independent truth.

Outputs (<output_dir>)
  rasters/<case>/<no2 stem>_<case>.tif   NO2, ECF, QA, cloud pressure, smoke p per variant, dt, class, NO2 with
                                         the cloud mask only and with the smoke-aware mask; with fire: fire_detected,
                                         fire_p_max, fire_distance_km, near_fire (1 / 0, NaN without fire products)
  rows/<case>/<no2 stem>.json            coverage rows (existing rasters + rows are skipped unless --overwrite)
  no2_coverage_scenes.csv, no2_coverage_summary.csv, no2_coverage_by_hour.csv, figures/*.png
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.transform import from_origin
import rasterio.warp
import rasterio.windows
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds

import sf_uq_utils as U

LOG = logging.getLogger("no2_smoke_mask")
NO2_RE = re.compile(r"TEMPO_NO2_L3_(V\d+)_(\d{8}T\d{6}Z)_S(\d{3})(?:_subset)?\.nc$")
FUSED_RE = re.compile(r"(\d{8}T\d{6})Z")
DEFAULT_CLOUD_THRESHOLD = {"V03": 0.2, "V04": 0.1}
CLASSES = {0: "no_retrieval", 1: "clear", 2: "cloud_screened", 3: "smoke_recovered", 4: "smoke_thick_cloud",
           5: "qa_rejected"}
VARS = {"vcd": "product/vertical_column_troposphere", "vcd_unc": "product/vertical_column_troposphere_uncertainty",
        "qa": "product/main_data_quality_flag", "ecf": "support_data/eff_cloud_fraction",
        "cld_p": "support_data/amf_cloud_pressure", "sza": "geolocation/solar_zenith_angle"}


# ---------------------------------------------------------------------------
# NO2 L3

def no2_files(cfg) -> pd.DataFrame:
    c = cfg["no2"]
    rows = {}
    for f in sorted(Path(c["dir"]).glob(c.get("glob", "TEMPO_NO2_L3_*.nc"))):
        m = NO2_RE.search(f.name)
        if not m:
            continue
        key = (m.group(2), m.group(3))
        # one file per scan: the subset (if any) or the full file; preferred versions first
        pref = c.get("versions", ["V04", "V03"])
        rank = (pref.index(m.group(1)) if m.group(1) in pref else 99, 0 if "_subset" in f.name else 1)
        if key not in rows or rank < rows[key][0]:
            rows[key] = (rank, {"path": str(f), "version": m.group(1), "scan": int(m.group(3)),
                                "file_time": pd.Timestamp(dt.datetime.strptime(m.group(2), "%Y%m%dT%H%M%SZ"), tz="UTC")})
    df = pd.DataFrame([r for _, r in rows.values()])
    if df.empty:
        return df
    df = df[df.version.isin(c.get("versions", ["V04", "V03"]))]
    return df.sort_values("file_time").reset_index(drop=True)


def _var(ds, path):
    grp, _, name = path.rpartition("/")
    g = ds
    for p in filter(None, grp.split("/")):
        if p not in g.groups:
            g = None
            break
        g = g.groups[p]
    if g is not None and name in g.variables:
        return g.variables[name]
    for gg in [ds, *ds.groups.values()]:          # fall back: same name in any group
        if name in gg.variables:
            return gg.variables[name]
    return None


def read_no2(path, bbox, var_paths=VARS):
    """bbox window of an L3 file -> (arrays north-up, profile, scan start, scan end)."""
    import netCDF4
    w, e, s, n = bbox
    with netCDF4.Dataset(path) as ds:
        lat = np.asarray(ds.variables["latitude"][:], float)
        lon = np.asarray(ds.variables["longitude"][:], float)
        iy = np.where((lat >= s) & (lat <= n))[0]
        ix = np.where((lon >= w) & (lon <= e))[0]
        if iy.size < 2 or ix.size < 2:
            return None
        ys, xs = slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1)
        out = {}
        for k, p in var_paths.items():
            v = _var(ds, p)
            if v is None:
                out[k] = None
                continue
            a = v[0, ys, xs] if v.ndim == 3 else v[ys, xs]
            out[k] = np.ma.filled(np.ma.asarray(a).astype(np.float64), np.nan).astype(np.float32)
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
    la, lo = lat[ys], lon[xs]
    if la[0] < la[-1]:                       # L3 latitude ascends: flip to north-up
        out = {k: (None if a is None else a[::-1]) for k, a in out.items()}
        la = la[::-1]
    dy, dx = abs(float(la[0] - la[1])), abs(float(lo[1] - lo[0]))
    prof = {"driver": "GTiff", "height": la.size, "width": lo.size, "count": 1, "dtype": "float32",
            "crs": rasterio.crs.CRS.from_epsg(4326), "transform": from_origin(lo[0] - dx / 2, la[0] + dy / 2, dx, dy)}
    t0 = _attr_time(attrs, "time_coverage_start")
    t1 = _attr_time(attrs, "time_coverage_end")
    return out, prof, t0, t1


def _attr_time(attrs, key):
    v = attrs.get(key)
    if v is None:
        return None
    try:
        return pd.Timestamp(str(v)).tz_convert("UTC") if pd.Timestamp(str(v)).tzinfo else pd.Timestamp(str(v), tz="UTC")
    except Exception:  # noqa: BLE001
        return None


def cloud_threshold(cfg, version):
    t = cfg["no2"].get("cloud_threshold", "auto")
    return float(DEFAULT_CLOUD_THRESHOLD.get(version, 0.2)) if t in (None, "auto") else float(t)


# ---------------------------------------------------------------------------
# fused smoke

_FUSED_CACHE: dict[str, pd.DataFrame] = {}


def fused_index(cfg) -> pd.DataFrame:
    """Fused frame table (cached per process)."""
    key = json.dumps(cfg["fused"], sort_keys=True, default=str)
    if key not in _FUSED_CACHE:
        _FUSED_CACHE[key] = _fused_index(cfg)
    return _FUSED_CACHE[key]


def _fused_index(cfg) -> pd.DataFrame:
    c = cfg["fused"]
    root = Path(c["dir"]) / c.get("grid", "fine") / "fused" / "rasters"
    rows = []
    for f in sorted(root.glob(c.get("glob", "fused_smoke_*.tif"))):
        with rasterio.open(f) as s:
            t = s.tags()
            bands = list(s.descriptions)
        st = t.get("scene_time")
        if st is None:
            m = FUSED_RE.search(f.name)
            if not m:
                continue
            st = dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").isoformat()
        ts = pd.Timestamp(st)
        ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        rows.append({"path": str(f), "scene_time": ts, "threshold": float(t.get("threshold", "nan")),
                     "split": t.get("classifier_split", ""), "bands": bands})
    if not rows:
        raise FileNotFoundError(f"no fused rasters under {root}")
    df = pd.DataFrame(rows).sort_values("scene_time").reset_index(drop=True)
    splits = c.get("splits")
    if splits:
        df = df[df.split.isin(splits)].reset_index(drop=True)
    LOG.info("fused smoke: %d frames %s .. %s (%s)", len(df), df.scene_time.min(), df.scene_time.max(), root)
    return df


def warp_fused(path, bands, dst_prof, resampling=Resampling.average):
    """Selected bands (NaN nodata) of a fused / geo-product raster resampled (default: averaged) onto the NO2
    window grid."""
    H, W = dst_prof["height"], dst_prof["width"]
    out = {}
    with rasterio.open(path) as s:
        desc = list(s.descriptions)
        tr = dst_prof["transform"]
        b = (tr.c, tr.f + tr.e * H, tr.c + tr.a * W, tr.f)
        try:
            wb = from_bounds(*rasterio.warp.transform_bounds(dst_prof["crs"], s.crs, *b), transform=s.transform)
        except Exception:  # noqa: BLE001 - bounds not transformable
            return None
        c0 = max(int(np.floor(wb.col_off)) - 2, 0)
        r0 = max(int(np.floor(wb.row_off)) - 2, 0)
        c1 = min(int(np.ceil(wb.col_off + wb.width)) + 2, s.width)
        r1 = min(int(np.ceil(wb.row_off + wb.height)) + 2, s.height)
        if c1 <= c0 or r1 <= r0:
            return None                                   # no overlap
        win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        str_ = s.window_transform(win)
        for bn in bands:
            if bn not in desc:
                out[bn] = None
                continue
            a = s.read(desc.index(bn) + 1, window=win).astype(np.float32)
            if s.nodata is not None:
                a[a == s.nodata] = np.nan
            o = np.full((H, W), np.nan, np.float32)
            reproject(a, o, src_transform=str_, src_crs=s.crs, src_nodata=np.nan, dst_transform=tr,
                      dst_crs=dst_prof["crs"], dst_nodata=np.nan, resampling=resampling)
            out[bn] = o
    return out


def match_fused(fidx, t0, t1, prof, bands, fc):
    """Per-cell fused values from the frame closest to the cell's TEMPO observation time."""
    w = pd.Timedelta(minutes=float(fc.get("window_minutes", 45)))
    cand = fidx[(fidx.scene_time >= t0 - w) & (fidx.scene_time <= t1 + w)]
    H, W = prof["height"], prof["width"]
    if cand.empty:
        return None
    need = list(dict.fromkeys([*bands, "tempo_age_minutes"]))
    stacks = {b: [] for b in need}
    times, thr = [], []
    for r in cand.itertuples():
        L = warp_fused(r.path, need, prof)
        if L is None:
            continue
        for b in need:
            stacks[b].append(L[b] if L[b] is not None else np.full((H, W), np.nan, np.float32))
        times.append(r.scene_time)
        thr.append(r.threshold)
    if not times:
        return None
    S = {b: np.stack(v) for b, v in stacks.items()}
    tf = np.array([(t - t0).total_seconds() / 60.0 for t in times], np.float32)      # minutes since scan start
    dur = (t1 - t0).total_seconds() / 60.0
    tobs = tf[:, None, None] - S["tempo_age_minutes"]
    tol = float(fc.get("scan_time_tolerance_minutes", 5))
    tobs = np.where((tobs >= -tol) & (tobs <= dur + tol), tobs, np.nan)
    with np.errstate(all="ignore"):
        tstar = np.nanmedian(tobs, axis=0) if np.isfinite(tobs).any() else np.full((H, W), np.nan, np.float32)
    src = np.where(np.isfinite(tstar), 1, 2).astype(np.float32)          # 1 TEMPO-observed time, 2 scan midpoint
    tstar = np.where(np.isfinite(tstar), tstar, dur / 2.0)
    prim = S[bands[0]]
    dtm = np.abs(tf[:, None, None] - tstar[None])
    dtm = np.where(np.isfinite(prim), dtm, np.inf)
    k = np.argmin(dtm, axis=0)
    best = np.take_along_axis(dtm, k[None], 0)[0]
    ok = best <= float(fc.get("max_dt_minutes", 45))
    out = {b: np.where(ok, np.take_along_axis(S[b], k[None], 0)[0], np.nan).astype(np.float32) for b in bands}
    out["dt_minutes"] = np.where(ok, best, np.nan).astype(np.float32)
    out["time_source"] = np.where(ok, src, np.nan).astype(np.float32)
    thr = np.array(thr, np.float32)
    out["frame_threshold"] = np.where(ok, thr[k], np.nan).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# fire proximity (marks cells, never changes the mask)

_FIRE_CACHE: dict[str, pd.DataFrame] = {}


def fire_index(cfg) -> pd.DataFrame | None:
    """Fire geo-product table over all fire.sources (uq_viz.data.GeoProductIndex; cached per process)."""
    fc = cfg.get("fire") or {}
    if not fc.get("enabled", True) or not fc.get("sources"):
        return None
    key = json.dumps(fc, sort_keys=True, default=str)
    if key not in _FIRE_CACHE:
        from uq_viz.data import GeoProductIndex
        parts = []
        for src in fc["sources"]:
            model = src.get("model", "hist_gradient_boosting")
            try:
                idx = GeoProductIndex(src["geo_dir"], src.get("product", "fire"), [model], src.get("manifest"))
            except FileNotFoundError as e:
                LOG.warning("fire source %s not available (%s); skipped", src.get("name", src["geo_dir"]), e)
                continue
            d = idx.df[idx.df.model == model].copy()
            d["source"] = src.get("name", Path(src["geo_dir"]).name)
            d["mask_layer"] = src.get("mask_layer", "pred_label")
            d["p_layer"] = src.get("layer", "p_1")
            parts.append(d[["source", "path", "scene_time", "threshold", "bands", "mask_layer", "p_layer"]])
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if df.empty:
            LOG.warning("fire: no fire products found; near_fire is not computed")
        else:
            LOG.info("fire: %d rasters (%s) %s .. %s", len(df), ", ".join(sorted(df.source.unique())),
                     df.scene_time.min(), df.scene_time.max())
        _FIRE_CACHE[key] = df
    return _FIRE_CACHE[key]


def fire_layers(fidx, t0, t1, prof, fc):
    """Union of fire decisions (and max probability) over the fire scenes of the scan window on the NO2 grid,
    distance (km) to the nearest fire cell and near_fire (distance <= buffer_km). None when no fire scene."""
    if fidx is None or fidx.empty:
        return None
    w = pd.Timedelta(minutes=float(fc.get("window_minutes", 30)))
    cand = fidx[(fidx.scene_time >= t0 - w) & (fidx.scene_time <= t1 + w)]
    H, W = prof["height"], prof["width"]
    det = np.zeros((H, W), bool)
    cov = np.zeros((H, W), bool)
    pmax = np.full((H, W), np.nan, np.float32)
    used = {}
    for r in cand.itertuples():
        L = warp_fused(r.path, [r.p_layer, r.mask_layer], prof, Resampling.max)  # any fire cell inside the NO2 cell
        if L is None or L.get(r.p_layer) is None:
            continue
        p = L[r.p_layer]
        m = L.get(r.mask_layer)
        if m is None:
            thr = float(fc.get("threshold", r.threshold))
            if not np.isfinite(thr):
                continue
            m = np.where(np.isfinite(p), (p >= thr).astype(np.float32), np.nan)
        cov |= np.isfinite(p)
        det |= np.nan_to_num(m) >= 1
        pmax = np.fmax(pmax, p)
        used[r.source] = used.get(r.source, 0) + 1
    if not used:
        return None
    from scipy.ndimage import distance_transform_edt
    tr = prof["transform"]
    lat_c = tr.f + tr.e * (H / 2.0)
    dy = abs(tr.e) * 111.32
    dx = abs(tr.a) * 111.32 * np.cos(np.deg2rad(lat_c))
    cap = float(fc.get("max_distance_km", 200))
    dist = distance_transform_edt(~det, sampling=(dy, dx)).astype(np.float32) if det.any() \
        else np.full((H, W), np.inf, np.float32)
    dist = np.minimum(dist, cap)
    buf = float(fc.get("buffer_km", 10))
    near = np.where(cov | det, (dist <= buf).astype(np.float32), np.nan).astype(np.float32)
    return {"fire_detected": np.where(cov, det.astype(np.float32), np.nan).astype(np.float32),
            "fire_p_max": pmax, "fire_distance_km": np.where(cov, dist, np.nan).astype(np.float32),
            "near_fire": near, "sources": used, "buffer_km": buf}


# ---------------------------------------------------------------------------
# masks and coverage

def variants(cfg):
    v = cfg.get("smoke_variants") or [{"name": "fused_p", "band": "p_1", "threshold": "tag"},
                                      {"name": "fused_q05", "band": "ensemble_p1_q05", "threshold": "tag"}]
    return v


def smoke_threshold(F, v):
    """Per-cell threshold: fused frame tag (threshold: tag) or a fixed value; NaN without fused data."""
    if v.get("threshold", "tag") == "tag":
        return F["frame_threshold"] if F else np.float32(np.nan)
    return np.float32(v["threshold"])


def classify(N, smoke, thr_ecf, cap, qa_max, min_cld_p=None, sza_max=None):
    """Class map 0..5 (see CLASSES) for one ECF threshold / recovery cap / smoke mask."""
    observed = np.isfinite(N["vcd"]) & np.isfinite(N["qa"])
    qa_ok = observed & (N["qa"] <= qa_max)
    if sza_max is not None and N.get("sza") is not None:
        qa_ok &= np.nan_to_num(N["sza"], nan=99) <= sza_max
    ecf = N["ecf"]
    clear = qa_ok & np.isfinite(ecf) & (ecf < thr_ecf)
    cloudy = qa_ok & np.isfinite(ecf) & (ecf >= thr_ecf)
    rec = cloudy & smoke
    if min_cld_p is not None and N.get("cld_p") is not None:
        rec &= np.nan_to_num(N["cld_p"], nan=0) >= min_cld_p
    thick = np.zeros_like(rec)
    if cap is not None:
        thick = rec & (ecf >= cap)
        rec &= ecf < cap
    cls = np.zeros(ecf.shape, np.float32)
    cls[observed & ~qa_ok] = 5
    cls[cloudy] = 2
    cls[thick] = 4
    cls[rec] = 3
    cls[clear] = 1
    return cls


def _wmean(v, a, m):
    m = m & np.isfinite(v)
    return float(np.sum(v[m] * a[m]) / np.sum(a[m])) if m.any() else np.nan


def coverage_rows(N, F, prof, cfg, meta, FIRE=None):
    area = U.pixel_area_km2(prof)
    qa_max = float(cfg["no2"].get("qa_max", 0))
    r = cfg.get("recovery", {})
    caps = r.get("max_cloud_fraction", [None])
    caps = caps if isinstance(caps, list) else [caps]
    thrs = cfg["no2"].get("cloud_threshold_sweep") or [meta["cloud_threshold"]]
    thrs = sorted(set([float(x) for x in thrs] + [meta["cloud_threshold"]]))
    vcd = N["vcd"]
    rows = []
    near = (np.nan_to_num(FIRE["near_fire"]) >= 1) if FIRE else np.zeros(vcd.shape, bool)
    far = (FIRE["near_fire"] == 0) if FIRE else np.zeros(vcd.shape, bool)
    for v in variants(cfg):
        p = F.get(v["band"]) if F else None
        if p is None:
            p = np.full(vcd.shape, np.nan, np.float32)
        t_s = smoke_threshold(F, v)
        has = np.isfinite(p)
        smoke = has & (p >= t_s)
        for te in thrs:
            for cap in caps:
                cls = classify(N, smoke, te, cap, qa_max, r.get("min_cloud_pressure_hpa"), cfg["no2"].get("sza_max"))
                obs = cls > 0
                base, rec = cls == 1, cls == 3
                upd = base | rec
                A = lambda m: float(area[m].sum())  # noqa: E731
                rows.append({
                    **meta, "variant": v["name"], "ecf_threshold": te, "recovery_cap": cap if cap is not None else np.nan,
                    "observed_km2": A(obs), "fused_available_km2": A(obs & has),
                    "baseline_km2": A(base), "updated_km2": A(upd), "added_km2": A(rec),
                    "cloud_screened_km2": A(cls == 2) + A(rec) + A(cls == 4), "thick_cloud_smoke_km2": A(cls == 4),
                    "qa_rejected_km2": A(cls == 5),
                    "smoke_km2": A(obs & smoke), "smoke_baseline_km2": A(base & smoke), "smoke_updated_km2": A(upd & smoke),
                    "vcd_baseline_mean": _wmean(vcd, area, base), "vcd_added_mean": _wmean(vcd, area, rec),
                    "vcd_clear_smoke_mean": _wmean(vcd, area, base & smoke),
                    "vcd_clear_nosmoke_mean": _wmean(vcd, area, base & ~smoke & has),
                    "vcd_added_median": float(np.nanmedian(vcd[rec])) if rec.any() else np.nan,
                    "vcd_unc_added_mean": _wmean(N["vcd_unc"], area, rec) if N.get("vcd_unc") is not None else np.nan,
                    "vcd_unc_baseline_mean": _wmean(N["vcd_unc"], area, base) if N.get("vcd_unc") is not None else np.nan,
                    "ecf_added_mean": _wmean(N["ecf"], area, rec),
                    "cld_p_added_median": float(np.nanmedian(N["cld_p"][rec])) if (N.get("cld_p") is not None and rec.any()) else np.nan,
                    "cld_p_cloud_nosmoke_median": float(np.nanmedian(N["cld_p"][cls == 2])) if (N.get("cld_p") is not None and (cls == 2).any()) else np.nan,
                    "dt_minutes_median": float(np.nanmedian(F["dt_minutes"][obs])) if (F and np.isfinite(F["dt_minutes"][obs]).any()) else np.nan,
                    # fire proximity (cells within fire.buffer_km of a fire-product detection; mask unchanged)
                    "fire_available": bool(FIRE), "fire_buffer_km": FIRE["buffer_km"] if FIRE else np.nan,
                    "fire_detected_km2": A(np.nan_to_num(FIRE["fire_detected"]) >= 1) if FIRE else np.nan,
                    "near_fire_km2": A(obs & near) if FIRE else np.nan,
                    "near_fire_baseline_km2": A(base & near) if FIRE else np.nan,
                    "near_fire_updated_km2": A(upd & near) if FIRE else np.nan,
                    "near_fire_added_km2": A(rec & near) if FIRE else np.nan,
                    "near_fire_smoke_km2": A(obs & smoke & near) if FIRE else np.nan,
                    "far_fire_baseline_km2": A(base & far) if FIRE else np.nan,
                    "far_fire_added_km2": A(rec & far) if FIRE else np.nan,
                    "vcd_baseline_near_fire_mean": _wmean(vcd, area, base & near) if FIRE else np.nan,
                    "vcd_baseline_far_fire_mean": _wmean(vcd, area, base & far) if FIRE else np.nan,
                    "vcd_added_near_fire_mean": _wmean(vcd, area, rec & near) if FIRE else np.nan,
                    "vcd_added_far_fire_mean": _wmean(vcd, area, rec & far) if FIRE else np.nan,
                })
    return rows


def process(job):
    """One (case, NO2 scan): raster + coverage rows. Returns (status, rows_path)."""
    cfg, case, bbox, rec, out, overwrite = job
    stem = Path(rec["path"]).name.replace("_subset", "").replace(".nc", "")
    rpath = out / "rasters" / case / f"{stem}_{case}.tif"
    jpath = out / "rows" / case / f"{stem}.json"
    if not overwrite and rpath.exists() and jpath.exists():
        return "skipped", str(jpath)
    got = read_no2(rec["path"], bbox)
    if got is None:
        return "outside", None
    N, prof, t0, t1 = got
    if N["vcd"] is None or N["ecf"] is None or N["qa"] is None:
        raise RuntimeError(f"{rec['path']}: missing vertical_column_troposphere, eff_cloud_fraction or QA flag")
    t0 = t0 or rec["file_time"]
    t1 = t1 or (t0 + pd.Timedelta(minutes=60))
    if not np.isfinite(N["vcd"]).any():
        return "empty", None
    vs = variants(cfg)
    fidx = fused_index(cfg)
    F = match_fused(fidx, t0, t1, prof, [v["band"] for v in vs], cfg["fused"])
    if F is None:
        LOG.warning("%s %s: no fused smoke frame within the scan window; only the cloud mask applies",
                    case, Path(rec["path"]).name)
    fc = cfg.get("fire") or {}
    FIRE = fire_layers(fire_index(cfg), t0, t1, prof, fc) if fc.get("sources") and fc.get("enabled", True) else None
    if fc.get("sources") and fc.get("enabled", True) and FIRE is None:
        LOG.info("%s %s: no fire product within the scan window; near_fire not set", case, Path(rec["path"]).name)
    thr = cloud_threshold(cfg, rec["version"])
    meta = {"case": case, "no2_file": Path(rec["path"]).name, "version": rec["version"], "scan": rec["scan"],
            "scan_start": t0.isoformat(), "scan_end": t1.isoformat(), "cloud_threshold": thr}
    rows = coverage_rows(N, F, prof, cfg, meta, FIRE)
    # raster: primary variant, version threshold, first recovery cap
    v0 = vs[0]
    p0 = F.get(v0["band"]) if F else np.full(N["vcd"].shape, np.nan, np.float32)
    t_s = smoke_threshold(F, v0)
    smoke = np.isfinite(p0) & (p0 >= t_s)
    caps = cfg.get("recovery", {}).get("max_cloud_fraction", [None])
    cap0 = caps[0] if isinstance(caps, list) else caps
    cls = classify(N, smoke, thr, cap0, float(cfg["no2"].get("qa_max", 0)),
                   cfg.get("recovery", {}).get("min_cloud_pressure_hpa"), cfg["no2"].get("sza_max"))
    nanv = np.full(N["vcd"].shape, np.nan, np.float32)
    bands = {"vertical_column_troposphere": N["vcd"],
             "vertical_column_troposphere_uncertainty": N["vcd_unc"] if N.get("vcd_unc") is not None else nanv,
             "eff_cloud_fraction": N["ecf"], "main_data_quality_flag": N["qa"],
             "amf_cloud_pressure": N["cld_p"] if N.get("cld_p") is not None else nanv}
    for v in vs:
        bands[f"{v['name']}_smoke_p"] = F.get(v["band"]) if (F and F.get(v["band"]) is not None) else nanv
    bands.update({"fused_dt_minutes": F["dt_minutes"] if F else nanv,
                  "fused_time_source": F["time_source"] if F else nanv,
                  "mask_class": np.where(cls > 0, cls, np.nan).astype(np.float32),
                  "no2_cloud_mask_only": np.where(cls == 1, N["vcd"], np.nan),
                  "no2_smoke_aware_mask": np.where((cls == 1) | (cls == 3), N["vcd"], np.nan)})
    if fc.get("sources") and fc.get("enabled", True):     # fixed band set whether or not fire was found
        for b in ("fire_detected", "fire_p_max", "fire_distance_km", "near_fire"):
            bands[b] = FIRE[b] if FIRE else nanv
    tags = {"case": case, "no2_file": Path(rec["path"]).name, "scan_start": t0.isoformat(), "scan_end": t1.isoformat(),
            "cloud_threshold": thr, "qa_max": cfg["no2"].get("qa_max", 0), "smoke_variant": v0["name"],
            "recovery_max_cloud_fraction": cap0, "classes": json.dumps(CLASSES),
            "fused_dir": cfg["fused"]["dir"], "fused_grid": cfg["fused"].get("grid", "fine")}
    if FIRE:
        tags.update({"fire_sources": json.dumps(FIRE["sources"]), "fire_buffer_km": FIRE["buffer_km"],
                     "fire_window_minutes": fc.get("window_minutes", 30)})
    U.write_band_stack(rpath, {k: np.asarray(a, np.float32) for k, a in bands.items()},
                       {k: v for k, v in prof.items() if k not in ("count", "dtype")}, tags)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = jpath.with_name(jpath.name + ".part")
    tmp.write_text(json.dumps(rows, default=float))
    tmp.replace(jpath)
    return "done", str(jpath)


# ---------------------------------------------------------------------------
# stages

def case_list(cfg):
    out = []
    for name, c in cfg["cases"].items():
        out.append((name, [float(x) for x in c["bbox"]], pd.Timestamp(c["start"], tz="UTC"),
                    pd.Timestamp(c["end"], tz="UTC")))
    return out


def stage_pull(cfg):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing" / "data_pull"))
    import pull_tempo_no2_l3 as P
    pc = cfg.get("pull", {})
    for name, bbox, start, end in case_list(cfg):
        LOG.info("pull %s: %s .. %s bbox %s", name, start, end, bbox)
        P.run(start.to_pydatetime(), end.to_pydatetime(), Path(cfg["no2"]["dir"]), bbox,
              cfg["no2"].get("versions", ["V04", "V03"]), threads=int(pc.get("threads", 4)),
              access=pc.get("access", "https"), subset=bool(pc.get("subset", True)),
              delete_full=bool(pc.get("delete_full", False)), dry_run=bool(pc.get("dry_run", False)))


def stage_mask(cfg, workers, overwrite):
    out = Path(cfg["output_dir"])
    files = no2_files(cfg)
    if files.empty:
        raise FileNotFoundError(f"no TEMPO_NO2_L3 files in {cfg['no2']['dir']}")
    fused_index(cfg)                                   # fail early if the fused products are missing
    fire_index(cfg)                                    # warn early if configured fire products are missing
    jobs = []
    for name, bbox, start, end in case_list(cfg):
        sel = files[(files.file_time >= start) & (files.file_time < end)]
        if cfg["no2"].get("scans"):
            sel = sel[sel.scan.isin(cfg["no2"]["scans"])]
        LOG.info("%s: %d NO2 scan(s)", name, len(sel))
        jobs += [(cfg, name, bbox, r._asdict(), out, overwrite) for r in sel.itertuples(index=False)]
    stats = {}
    if workers <= 1:
        for j in jobs:
            s, _ = process(j)
            stats[s] = stats.get(s, 0) + 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process, j): j for j in jobs}
            for f in as_completed(futs):
                try:
                    s, _ = f.result()
                except Exception as e:  # noqa: BLE001
                    LOG.error("%s %s: %s", futs[f][1], Path(futs[f][3]["path"]).name, e)
                    s = "failed"
                stats[s] = stats.get(s, 0) + 1
    LOG.info("mask: %s", ", ".join(f"{k} {v}" for k, v in sorted(stats.items())))
    return stats


def summarize(df):
    keys = ["case", "variant", "ecf_threshold", "recovery_cap"]
    num = ["observed_km2", "fused_available_km2", "baseline_km2", "updated_km2", "added_km2", "cloud_screened_km2",
           "thick_cloud_smoke_km2", "smoke_km2", "smoke_baseline_km2", "smoke_updated_km2"]
    fire = [c for c in ("near_fire_km2", "near_fire_baseline_km2", "near_fire_updated_km2", "near_fire_added_km2",
                        "near_fire_smoke_km2", "far_fire_baseline_km2", "far_fire_added_km2", "fire_detected_km2")
            if c in df]
    num += fire
    d = df.copy()
    d["recovery_cap"] = d["recovery_cap"].fillna(-1)
    wts = {"vcd_added_mean": "added_km2", "vcd_clear_smoke_mean": "smoke_baseline_km2",
           "vcd_baseline_mean": "baseline_km2", "vcd_clear_nosmoke_mean": "baseline_km2"}
    if fire:
        wts.update({"vcd_baseline_near_fire_mean": "near_fire_baseline_km2", "vcd_added_near_fire_mean":
                    "near_fire_added_km2", "vcd_baseline_far_fire_mean": "far_fire_baseline_km2",
                    "vcd_added_far_fire_mean": "far_fire_added_km2"})
    for c, w in wts.items():
        d[c + "_w"] = d[c].fillna(0) * d[w]
        d[c + "_wa"] = np.where(d[c].notna(), d[w], 0)
    g = d.groupby(keys)
    s = g[num].sum()
    s["n_scans"] = g.size()
    s["n_scans_with_added"] = g["added_km2"].apply(lambda x: int((x > 0).sum()))
    for c in wts:
        s[c] = g[c + "_w"].sum() / g[c + "_wa"].sum().replace(0, np.nan)
    s["coverage_gain_pct"] = 100 * (s.updated_km2 / s.baseline_km2.replace(0, np.nan) - 1)
    s["baseline_frac_observed"] = s.baseline_km2 / s.observed_km2.replace(0, np.nan)
    s["updated_frac_observed"] = s.updated_km2 / s.observed_km2.replace(0, np.nan)
    s["smoke_retained_baseline_frac"] = s.smoke_baseline_km2 / s.smoke_km2.replace(0, np.nan)
    s["smoke_retained_updated_frac"] = s.smoke_updated_km2 / s.smoke_km2.replace(0, np.nan)
    s["cloud_screened_recovered_frac"] = s.added_km2 / s.cloud_screened_km2.replace(0, np.nan)
    if fire:
        s["near_fire_retained_baseline_frac"] = s.near_fire_baseline_km2 / s.near_fire_km2.replace(0, np.nan)
        s["near_fire_retained_updated_frac"] = s.near_fire_updated_km2 / s.near_fire_km2.replace(0, np.nan)
        s["added_near_fire_frac"] = s.near_fire_added_km2 / s.added_km2.replace(0, np.nan)
    s = s.reset_index()
    s["recovery_cap"] = s["recovery_cap"].replace(-1, np.nan)
    return s


def stage_compare(cfg):
    out = Path(cfg["output_dir"])
    rows = []
    for f in sorted((out / "rows").glob("*/*.json")):
        rows += json.loads(f.read_text())
    if not rows:
        raise FileNotFoundError(f"no coverage rows under {out / 'rows'} (run --stage mask)")
    df = pd.DataFrame(rows)
    df["scan_start"] = pd.to_datetime(df["scan_start"], utc=True)
    df.sort_values(["case", "scan_start", "variant", "ecf_threshold"]).to_csv(out / "no2_coverage_scenes.csv", index=False)
    s = summarize(df)
    s.to_csv(out / "no2_coverage_summary.csv", index=False)
    d = df.copy()
    d["hour_utc"] = d.scan_start.dt.hour
    h = d.groupby(["case", "variant", "ecf_threshold", "recovery_cap", "hour_utc"], dropna=False)[
        ["observed_km2", "baseline_km2", "updated_km2", "added_km2", "smoke_km2", "smoke_baseline_km2",
         "smoke_updated_km2"] + [c for c in ("near_fire_km2", "near_fire_baseline_km2", "near_fire_updated_km2",
                                             "near_fire_added_km2") if c in d]].sum(min_count=1).reset_index()
    h.to_csv(out / "no2_coverage_by_hour.csv", index=False)
    LOG.info("summary (%d rows):\n%s", len(s), s[["case", "variant", "ecf_threshold", "recovery_cap", "n_scans",
                                                  "baseline_km2", "updated_km2", "coverage_gain_pct",
                                                  "smoke_retained_baseline_frac", "smoke_retained_updated_frac"]]
             .to_string(index=False))
    if cfg.get("figures", {}).get("enabled", True):
        figures(cfg, df, s)
    return s


def figures(cfg, df, s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    out = Path(cfg["output_dir"]) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    v0 = variants(cfg)[0]["name"]
    caps = cfg.get("recovery", {}).get("max_cloud_fraction", [None])
    cap0 = caps[0] if isinstance(caps, list) else caps
    capm = (lambda x: x.isna()) if cap0 is None else (lambda x: np.isclose(x, cap0))
    has_fire = "near_fire_km2" in s and s["near_fire_km2"].fillna(0).gt(0).any()
    for case, g in s[(s.variant == v0) & capm(s.recovery_cap)].groupby("case"):
        fig, ax = plt.subplots(1, 3 if has_fire else 2, figsize=(15 if has_fire else 10, 3.6))
        x = np.arange(len(g))
        lab = [f"{t:g}" for t in g.ecf_threshold]
        ax[0].bar(x - 0.2, g.baseline_km2 / 1e3, 0.4, label="cloud mask only", color="#8c8c8c")
        ax[0].bar(x + 0.2, g.updated_km2 / 1e3, 0.4, label="cloud + SIT-FUSE smoke", color="#c4572a")
        ax[0].set(xticks=x, xticklabels=lab, xlabel="ECF threshold", ylabel="retained area (10$^3$ km$^2$)",
                  title=f"{case}: all retrievals")
        ax[1].bar(x - 0.2, 100 * g.smoke_retained_baseline_frac, 0.4, color="#8c8c8c")
        ax[1].bar(x + 0.2, 100 * g.smoke_retained_updated_frac, 0.4, color="#c4572a")
        ax[1].set(xticks=x, xticklabels=lab, xlabel="ECF threshold", ylabel="smoke area retained (%)",
                  title=f"{case}: inside fused smoke", ylim=(0, 100))
        if has_fire:
            ax[2].bar(x - 0.2, 100 * g.near_fire_retained_baseline_frac, 0.4, color="#8c8c8c")
            ax[2].bar(x + 0.2, 100 * g.near_fire_retained_updated_frac, 0.4, color="#c4572a")
            buf = df["fire_buffer_km"].dropna()
            ax[2].set(xticks=x, xticklabels=lab, xlabel="ECF threshold", ylabel="near-fire area retained (%)",
                      title=f"{case}: within {buf.iloc[0]:g} km of fire" if len(buf) else f"{case}: near fire",
                      ylim=(0, 100))
        fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(out / f"coverage_{case}.png", dpi=150)
        plt.close(fig)
    # time series, version threshold
    d = df[(df.variant == v0) & np.isclose(df.ecf_threshold, df.cloud_threshold) & capm(df.recovery_cap)]
    for case, g in d.groupby("case"):
        g = g.sort_values("scan_start")
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(g.scan_start, g.smoke_km2 / 1e3, color="k", lw=0.8, label="fused smoke area")
        ax.plot(g.scan_start, g.smoke_baseline_km2 / 1e3, color="#8c8c8c", marker=".", lw=0.8, label="retained, cloud mask only")
        ax.plot(g.scan_start, g.smoke_updated_km2 / 1e3, color="#c4572a", marker=".", lw=0.8, label="retained, cloud + smoke")
        if has_fire and g["near_fire_km2"].notna().any():         # near-fire cells (dashed; mask unchanged)
            ax.plot(g.scan_start, g.near_fire_km2 / 1e3, color="#ff8c00", lw=0.8, ls="--", label="near-fire area")
            ax.plot(g.scan_start, g.near_fire_updated_km2 / 1e3, color="#ff8c00", marker="x", lw=0.8, ls=":",
                    label="near fire, retained (cloud + smoke)")
        ax.set(ylabel="10$^3$ km$^2$", title=f"{case}: NO2 retrievals inside fused smoke"
               + (" and near fire" if has_fire else "") + " per scan")
        ax.legend(frameon=False, fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out / f"timeseries_{case}.png", dpi=150)
        plt.close(fig)
    # maps for the scans with the largest added area
    k = int(cfg.get("figures", {}).get("maps_per_case", 3))
    cmap_cls = ListedColormap(["#4d9de0", "#bdbdbd", "#e15554", "#7a306c", "#3d3d3d"])
    root = Path(cfg["output_dir"]) / "rasters"
    for case, g in d.groupby("case"):
        for r in g.nlargest(k, "added_km2").itertuples():
            if r.added_km2 <= 0:
                continue
            f = root / case / f"{r.no2_file.replace('_subset', '').replace('.nc', '')}_{case}.tif"
            if not f.exists():
                continue
            with rasterio.open(f) as src:
                desc = list(src.descriptions)
                A = {n: src.read(i + 1).astype(float) for i, n in enumerate(desc)}
                for a in A.values():
                    a[a == src.nodata] = np.nan
                ext = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
            vmax = np.nanpercentile(A["vertical_column_troposphere"], 98) if np.isfinite(A["vertical_column_troposphere"]).any() else 1
            fire_map = "near_fire" in A and np.isfinite(A["near_fire"]).any()
            fig, ax = plt.subplots(1, 6 if fire_map else 5, figsize=(22.5 if fire_map else 19, 3.8))
            panels = [("eff_cloud_fraction", "effective cloud fraction", "Blues_r", 0, 1),
                      (f"{v0}_smoke_p", "SIT-FUSE fused smoke p", "magma", 0, 1),
                      ("mask_class", "class", cmap_cls, 0.5, 5.5),
                      ("no2_cloud_mask_only", "trop. NO2, cloud mask", "viridis", 0, vmax),
                      ("no2_smoke_aware_mask", "trop. NO2, cloud + smoke", "viridis", 0, vmax)]
            if fire_map:
                panels.append(("fire_distance_km", "distance to SIT-FUSE fire (km)", "inferno_r", 0,
                               float((cfg.get("fire") or {}).get("max_distance_km", 200))))
            for a, (b, t, cm, lo, hi) in zip(ax, panels):
                im = a.imshow(A.get(b, np.full((2, 2), np.nan)), extent=ext, cmap=cm, vmin=lo, vmax=hi,
                              interpolation="nearest")
                a.set_title(t, fontsize=9)
                cb = fig.colorbar(im, ax=a, shrink=0.75)
                if b == "mask_class":
                    cb.set_ticks([1, 2, 3, 4, 5])
                    cb.set_ticklabels(["clear", "cloud", "smoke rec.", "smoke thick", "QA"])
                if fire_map and (b.startswith("no2_") or b in ("mask_class", "fire_distance_km")):
                    nf = np.nan_to_num(A["near_fire"])            # near-fire outline (mask unchanged)
                    if nf.max() >= 1:
                        ny, nx = nf.shape
                        xs = ext[0] + (np.arange(nx) + 0.5) * (ext[1] - ext[0]) / nx
                        ys = ext[3] - (np.arange(ny) + 0.5) * (ext[3] - ext[2]) / ny
                        a.contour(xs, ys, nf, levels=[0.5], colors=["#ff8c00"], linewidths=0.8)
            fire_note = ""
            if fire_map and pd.notna(getattr(r, "near_fire_added_km2", np.nan)):
                fire_note = (f"; {r.near_fire_added_km2:,.0f} km$^2$ of it within {r.fire_buffer_km:g} km of fire "
                             "(orange outline)")
            fig.suptitle(f"{case}  {r.no2_file}  added {r.added_km2:,.0f} km$^2$ (ECF >= {r.ecf_threshold:g}){fire_note}",
                         fontsize=10)
            fig.tight_layout()
            fig.savefig(out / f"map_{case}_{r.scan_start:%Y%m%dT%H%M}.png", dpi=130)
            plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--stage", choices=["pull", "mask", "compare", "all"], default="all")
    ap.add_argument("-n", "--workers", type=int, default=None, help="processes for the mask stage")
    ap.add_argument("--overwrite", action="store_true", help="redo scans whose raster and rows exist")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(Path(a.config).read_text())
    workers = a.workers or int(cfg.get("workers", 1))
    if a.stage in ("pull", "all") and cfg.get("pull", {}).get("enabled", a.stage == "pull"):
        stage_pull(cfg)
    if a.stage in ("mask", "all"):
        stage_mask(cfg, workers, a.overwrite)
    if a.stage in ("compare", "all"):
        stage_compare(cfg)


if __name__ == "__main__":
    main()
