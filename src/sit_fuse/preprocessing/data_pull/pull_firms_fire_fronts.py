#!/usr/bin/env python3
"""
Pull FIRMS fire-front products for a bbox / time range and write them as
standardized GeoParquet (and optionally GPKG) vector layers.

Sources
  feds   VIIRS Modeled Fire Perimeters (FEDS; Chen et al., 2022), the product
         FIRMS shows as fire perimeters / active fire fronts. Read keyless from
         the NASA VEDA OGC API Features endpoint, layers:
           fireline    active portion of the perimeter (the fire front)
           perimeter   modeled fire perimeter (polygon)
           newfirepix  VIIRS 375 m detections new in the 12-hour step
         Collections are chosen by date (`feds.collections: auto`):
           *_archive        2018-2021 (FEDS science archive)
           snapshot_*_nrt   2025-01 .. 2025-03 (all fires, NRT snapshot)
           lf_*_nrt         2025-03 .. present (large fires, NRT)
         FEDS time `t` is a 12-hour step label: 00:00 = AM (night) and 12:00 =
         PM (day) overpasses in approximate local solar time (fireatlas
         FireTime.t2dt / FireIO.AFP_setampm). An estimated overpass time is
         derived per feature as t_date + overpass_lst_hours[AM|PM] - lon/15 h.
  firms  FIRMS active fire detections (VIIRS 375 m / MODIS 1 km points) from
         the FIRMS Area API (MAP_KEY; *_SP science archive or *_NRT) and/or
         local FIRMS archive-download CSVs. Detections are grouped into
         overpasses (one satellite, contiguous acquisition times). Fire fronts
         are derived from these by rasterize_fire_fronts.py (FEDS-style new
         pixels), which covers years outside the FEDS collections (e.g. 2022-2024).

Outputs (EPSG:4326, in output.outdir/vectors):
  feds_fireline.parquet, feds_perimeter.parquet, feds_newfirepix.parquet
      columns: source, layer, collection, fireid, step_t, step_half, obs_time,
               + product attributes (farea, flinelen, meanfrp, ...)
  firms_detections.parquet
      columns: source, firms_source, satellite, instrument, acq_time,
               overpass_id, overpass_time, confidence, confidence_level
               (0 low, 1 nominal, 2 high), frp, scan, track, daynight
  pull_manifest.json

Usage:
  export FIRMS_MAP_KEY=...      # only for firms.sources (https://firms.modaps.eosdis.nasa.gov/api/map_key/)
  python pull_firms_fire_fronts.py firms_fire_fronts.example.yaml [--local-only]
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import yaml
from shapely.geometry import box, shape

LOG = logging.getLogger("pull-firms-fire-fronts")

FEDS_API = "https://openveda.cloud/api/features"
FIRMS_AREA_API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FEDS_LAYERS = ("fireline", "perimeter", "newfirepix")
# (collection template, first valid time, last valid time or None = open)
FEDS_AUTO = (
    ("public.eis_fire_lf_{layer}_archive", "2018-01-01", "2021-12-31T23:59:59"),
    ("public.eis_fire_snapshot_{layer}_nrt", "2025-01-01", "2025-12-31T23:59:59"),
    ("public.eis_fire_lf_{layer}_nrt", "2025-03-01", None),
)
VIIRS_CONF = {"l": 0, "low": 0, "n": 1, "nominal": 1, "h": 2, "high": 2}
CONF_LEVEL = {"low": 0, "nominal": 1, "high": 2}


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def parse_utc(v) -> pd.Timestamp:
    t = pd.Timestamp(str(v).strip().replace("Z", ""))
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for k in ("time_range", "bbox", "output"):
        if k not in cfg:
            raise ValueError(f"missing config section {k!r}")
    b = cfg["bbox"]
    cfg["_bbox"] = (float(b["min_lon"]), float(b["min_lat"]), float(b["max_lon"]), float(b["max_lat"]))
    if not (cfg["_bbox"][0] < cfg["_bbox"][2] and cfg["_bbox"][1] < cfg["_bbox"][3]):
        raise ValueError("bbox must satisfy min_lon < max_lon and min_lat < max_lat")
    cfg["_start"] = parse_utc(cfg["time_range"]["start"])
    cfg["_end"] = parse_utc(cfg["time_range"]["end"])
    if cfg["_end"] < cfg["_start"]:
        raise ValueError("time_range.end is before time_range.start")
    cfg["_outdir"] = Path(cfg["output"]["outdir"])
    return cfg


def vector_dir(cfg) -> Path:
    return Path(cfg["output"].get("vector_dir") or cfg["_outdir"] / "vectors")


def write_layer(gdf: gpd.GeoDataFrame, cfg: dict, name: str) -> list[str]:
    d = vector_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in cfg["output"].get("vector_formats", ["parquet"]):
        if fmt == "parquet":
            p = d / f"{name}.parquet"
            gdf.to_parquet(p)
        elif fmt == "gpkg":
            p = d / f"{name}.gpkg"
            g = gdf.copy()
            for c in g.columns:  # GPKG has no tz-aware datetime round trip
                if pd.api.types.is_datetime64_any_dtype(g[c]):
                    g[c] = g[c].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            g.to_file(p, layer=name, driver="GPKG")
        else:
            raise ValueError(f"unknown vector format {fmt!r} (parquet | gpkg)")
        out.append(str(p))
    return out


def http_get(url, params=None, timeout=120, retries=4):
    for k in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            if k == retries - 1:
                raise
            wait = 5 * 2 ** k
            LOG.warning("GET %s failed (%s); retry in %ds", url, exc, wait)
            time.sleep(wait)


# ---------------------------------------------------------------------------
# FEDS (VEDA OGC API Features)
# ---------------------------------------------------------------------------

def feds_collections(layer: str, fcfg: dict, start, end) -> list[str]:
    spec = fcfg.get("collections", "auto")
    if spec != "auto":
        cols = spec.get(layer) if isinstance(spec, dict) else spec
        return [c.format(layer=layer) for c in (cols or [])]
    out = []
    for tmpl, a, b in FEDS_AUTO:
        if end >= parse_utc(a) and (b is None or start <= parse_utc(b)):
            out.append(tmpl.format(layer=layer))
    return out


def ogc_items(api, collection, bbox, start, end, limit=1000):
    """All features of one collection in bbox / interval (follows 'next')."""
    url = f"{api.rstrip('/')}/collections/{collection}/items"
    params = {"bbox": ",".join(f"{v:.6f}" for v in bbox),
              "datetime": f"{start:%Y-%m-%dT%H:%M:%SZ}/{end:%Y-%m-%dT%H:%M:%SZ}",
              "limit": int(limit)}
    feats = []
    while url:
        d = http_get(url, params=params).json()
        feats.extend(d.get("features", []))
        nxt = [l["href"] for l in d.get("links", []) if l.get("rel") == "next"]
        url, params = (nxt[0], None) if nxt and d.get("features") else (None, None)
    return feats


def feds_obs_time(step_t: pd.Series, lon: np.ndarray, lst_hours: dict) -> pd.Series:
    """Estimated VIIRS overpass time (UTC) for FEDS steps (AM = 00:00, PM = 12:00)."""
    half = np.where(step_t.dt.hour >= 12, "PM", "AM")
    lst = np.where(half == "PM", float(lst_hours.get("PM", 13.5)), float(lst_hours.get("AM", 1.5)))
    off = pd.to_timedelta(lst - np.asarray(lon, float) / 15.0, unit="h")
    return step_t.dt.normalize() + off, half


def pull_feds(cfg: dict) -> dict:
    fcfg = cfg.get("feds") or {}
    api = fcfg.get("api_url", FEDS_API)
    bbox, start, end = cfg["_bbox"], cfg["_start"], cfg["_end"]
    # FEDS steps are labeled at 00/12 local-solar-ish hours; pad so steps whose
    # estimated overpass falls inside the window are kept.
    pad = pd.Timedelta(hours=float(fcfg.get("time_pad_hours", 12)))
    out = {}
    for layer in fcfg.get("layers", list(FEDS_LAYERS)):
        if layer not in FEDS_LAYERS:
            raise ValueError(f"feds.layers: unknown layer {layer!r}")
        rows = []
        for col in feds_collections(layer, fcfg, start - pad, end + pad):
            try:
                feats = ogc_items(api, col, bbox, start - pad, end + pad, fcfg.get("page_limit", 1000))
            except requests.HTTPError as exc:
                LOG.warning("FEDS %s: %s", col, exc)
                continue
            LOG.info("FEDS %s: %d features", col, len(feats))
            for f in feats:
                if not f.get("geometry"):
                    continue
                p = dict(f.get("properties") or {})
                p["collection"] = col
                p["geometry"] = shape(f["geometry"])
                rows.append(p)
        if not rows:
            LOG.warning("FEDS %s: no features in bbox/time range", layer)
            out[layer] = None
            continue
        g = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        g["fireid"] = g["fireid"].astype(str).str.replace(r"\.0$", "", regex=True)
        g["step_t"] = pd.to_datetime(g["t"], utc=True)
        lon = g.geometry.representative_point().x.to_numpy()
        g["obs_time"], g["step_half"] = feds_obs_time(g["step_t"], lon, fcfg.get("overpass_lst_hours", {}))
        g = g[(g["obs_time"] >= start) & (g["obs_time"] <= end)].copy()
        # snapshot and large-fire collections overlap in early 2025: drop exact duplicates
        g["_wkb"] = g.geometry.apply(lambda s: s.wkb_hex)
        g = g.drop_duplicates(subset=["step_t", "_wkb"]).drop(columns=["_wkb", "t"], errors="ignore")
        g.insert(0, "layer", layer)
        g.insert(0, "source", "feds")
        g = g.drop(columns=[c for c in ("wkb_geometry", "ogc_fid", "geom_counts") if c in g], errors="ignore")
        g = g.sort_values(["step_t", "fireid"]).reset_index(drop=True)
        out[layer] = g
        LOG.info("FEDS %s: %d features, %d steps kept", layer, len(g), g["step_t"].nunique())
    return out


# ---------------------------------------------------------------------------
# FIRMS active fire detections
# ---------------------------------------------------------------------------

def _norm_firms(df: pd.DataFrame, firms_source: str) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    need = {"latitude", "longitude", "acq_date", "acq_time"}
    if not need <= set(df.columns):
        raise ValueError(f"FIRMS CSV missing columns {sorted(need - set(df.columns))}")
    hhmm = df["acq_time"].astype(str).str.replace(":", "").str.zfill(4)
    df["acq_time"] = pd.to_datetime(df["acq_date"].astype(str) + " " + hhmm, format="%Y-%m-%d %H%M", utc=True)
    conf = df.get("confidence", pd.Series(["n"] * len(df))).astype(str).str.strip().str.lower()
    num = pd.to_numeric(conf, errors="coerce")  # MODIS: 0-100
    lev = conf.map(VIIRS_CONF)
    lev = lev.where(lev.notna(), np.select([num >= 80, num >= 30], [2, 1], 0))
    df["confidence"] = conf
    df["confidence_level"] = lev.astype(int)
    df["firms_source"] = firms_source
    if "instrument" not in df:
        df["instrument"] = "MODIS" if "MODIS" in firms_source.upper() else "VIIRS"
    if "satellite" not in df:
        df["satellite"] = firms_source
    for c in ("scan", "track", "frp"):
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
    default_px = np.where(df["instrument"].astype(str).str.upper().str.startswith("MODIS"), 1.0, 0.375)
    df["scan"] = df["scan"].fillna(pd.Series(default_px, index=df.index))
    df["track"] = df["track"].fillna(pd.Series(default_px, index=df.index))
    if "daynight" not in df:
        df["daynight"] = ""
    return df


def firms_area_requests(fcfg: dict, bbox, start, end):
    key = os.environ.get(fcfg.get("map_key_env", "FIRMS_MAP_KEY"), "")
    if not key:
        raise RuntimeError(f"FIRMS Area API needs a MAP_KEY in ${fcfg.get('map_key_env', 'FIRMS_MAP_KEY')} "
                           "(https://firms.modaps.eosdis.nasa.gov/api/map_key/)")
    api = fcfg.get("api_url", FIRMS_AREA_API).rstrip("/")
    day_range = min(5, max(1, int(fcfg.get("day_range", 5))))  # API maximum is 5
    area = ",".join(f"{v:.4f}" for v in bbox)
    for src in fcfg.get("sources", []):
        d = start.normalize()
        while d <= end:
            yield src, f"{api}/{key}/{src}/{area}/{day_range}/{d:%Y-%m-%d}"
            d += pd.Timedelta(days=day_range)


def assign_overpasses(df: pd.DataFrame, gap_minutes: float) -> pd.DataFrame:
    """One overpass = one satellite, acquisition times without gaps > gap_minutes."""
    df = df.sort_values(["satellite", "acq_time"]).reset_index(drop=True)
    new = (df["satellite"] != df["satellite"].shift()) | \
          (df["acq_time"].diff() > pd.Timedelta(minutes=gap_minutes))
    df["overpass_id"] = new.cumsum().astype(int)
    df["overpass_time"] = df.groupby("overpass_id")["acq_time"].transform("median").dt.round("s")
    return df


def pull_firms(cfg: dict, local_only: bool = False) -> gpd.GeoDataFrame | None:
    fcfg = cfg.get("firms") or {}
    bbox, start, end = cfg["_bbox"], cfg["_start"], cfg["_end"]
    raw_dir = cfg["_outdir"] / "firms_csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for p in fcfg.get("local_csv", []) or []:
        for f in sorted(glob.glob(str(p))):
            f = Path(f)
            frames.append(_norm_firms(pd.read_csv(f), fcfg.get("local_source_name", f.stem)))
            LOG.info("FIRMS local CSV %s: %d rows", f, len(frames[-1]))
    if fcfg.get("sources"):
        if local_only:
            for f in sorted(raw_dir.glob("*.csv")):
                src = f.stem.split("__")[0]
                frames.append(_norm_firms(pd.read_csv(f), src))
        else:
            for src, url in firms_area_requests(fcfg, bbox, start, end):
                cache = raw_dir / f"{src}__{url.rsplit('/', 1)[-1]}.csv"
                if cache.exists() and not cfg.get("overwrite", False):
                    txt = cache.read_text()
                else:
                    txt = http_get(url, timeout=180).text
                    if txt.lstrip().lower().startswith(("invalid", "<", "error")):
                        raise RuntimeError(f"FIRMS Area API error for {src}: {txt[:200]}")
                    cache.write_text(txt)
                if txt.strip():
                    df = pd.read_csv(io.StringIO(txt))
                    if len(df):
                        frames.append(_norm_firms(df, src))
                time.sleep(float(fcfg.get("request_pause_seconds", 0.5)))
    if not frames:
        LOG.warning("FIRMS: no detections (configure firms.sources or firms.local_csv)")
        return None
    df = pd.concat(frames, ignore_index=True)
    x0, y0, x1, y1 = bbox
    df = df[(df.longitude >= x0) & (df.longitude <= x1) & (df.latitude >= y0) & (df.latitude <= y1)
            & (df.acq_time >= start) & (df.acq_time <= end)]
    minc = CONF_LEVEL[str(fcfg.get("min_confidence", "nominal")).lower()]
    df = df[df.confidence_level >= minc]
    if not fcfg.get("include_night", True):
        df = df[df.daynight.astype(str).str.upper() != "N"]
    df = df.drop_duplicates(subset=["latitude", "longitude", "acq_time", "satellite"])
    if df.empty:
        LOG.warning("FIRMS: no detections left after bbox / time / confidence filters")
        return None
    df = assign_overpasses(df, float(fcfg.get("overpass_gap_minutes", 20)))
    keep = ["firms_source", "satellite", "instrument", "acq_time", "overpass_id", "overpass_time",
            "confidence", "confidence_level", "frp", "scan", "track", "daynight", "latitude", "longitude"]
    g = gpd.GeoDataFrame(df[keep].copy(), geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    g.insert(0, "layer", "detection")
    g.insert(0, "source", "firms")
    LOG.info("FIRMS: %d detections in %d overpasses", len(g), g["overpass_id"].nunique())
    return g.reset_index(drop=True)


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--local-only", action="store_true",
                    help="reuse cached FIRMS Area API CSVs instead of downloading (FEDS is always fetched)")
    ap.add_argument("--skip-feds", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(a.config)
    manifest = {"bbox": cfg["_bbox"], "start": str(cfg["_start"]), "end": str(cfg["_end"]), "layers": {}}
    if (cfg.get("feds") or {}).get("enabled", True) and not a.skip_feds:
        for layer, g in pull_feds(cfg).items():
            if g is not None and len(g):
                manifest["layers"][f"feds_{layer}"] = {
                    "files": write_layer(g, cfg, f"feds_{layer}"), "n": int(len(g)),
                    "steps": int(g["step_t"].nunique()), "collections": sorted(g["collection"].unique())}
    if (cfg.get("firms") or {}).get("enabled", False):
        g = pull_firms(cfg, local_only=a.local_only)
        if g is not None:
            manifest["layers"]["firms_detections"] = {
                "files": write_layer(g, cfg, "firms_detections"), "n": int(len(g)),
                "overpasses": int(g["overpass_id"].nunique()), "sources": sorted(g["firms_source"].unique())}
    d = vector_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    (d / "pull_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if not manifest["layers"]:
        LOG.warning("Nothing written. FEDS covers 2018-2021 and 2025-01 onward; use firms for other years.")
    LOG.info("Wrote %s", d / "pull_manifest.json")


if __name__ == "__main__":
    sys.exit(main())
