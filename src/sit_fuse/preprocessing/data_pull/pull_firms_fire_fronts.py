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
         Local FEDS vector files (fireatlas outputs or downloads, any OGR
         format: .fgb, .gpkg, .shp, .geojson, .parquet; any CRS, e.g. the
         fireatlas EPSG:9311 equal-area grid) are read from feds.local_vectors
         instead of the API (feds.source: auto | local | api | both).
         API collections are chosen by date (`feds.collections: auto`):
           *_archive        2018-2021 (FEDS science archive)
           snapshot_*_nrt   2025-01 .. 2025-03 (all fires, NRT snapshot)
           lf_*_nrt         2025-03 .. present (large fires, NRT)
         FEDS time `t` is a 12-hour step label: 00:00 = AM (night) and 12:00 =
         PM (day) overpasses in approximate local solar time (fireatlas
         FireTime.t2dt / FireIO.AFP_setampm). An estimated overpass time is
         derived per feature as t_date + overpass_lst_hours[AM|PM] - lon/15 h.
  firms  FIRMS active fire detections (VIIRS 375 m / MODIS 1 km points) from
         the FIRMS Area API (MAP_KEY; *_SP science archive or *_NRT) and/or
         local FIRMS archive-download files (firms.local_csv: .csv, .json,
         .shp or the DL_FIRE_*.zip itself; fire_archive_* = SP, fire_nrt_* =
         NRT, satellite from the M-C61 / SV-C2 / J1V-C2 / J2V-C2 code).
         Detections are grouped into
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

Relative paths in feds.local_vectors and firms.local_csv are resolved against
the config file directory. --local-only: no network (cached Area API CSVs,
local FIRMS files and local FEDS vectors only).
"""

from __future__ import annotations

import argparse
import glob
import io
import re
import zipfile
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
VIIRS_CONF = {"l": 0, "low": 0, "n": 1, "nominal": 1, "m": 1, "medium": 1, "h": 2, "high": 2}  # Landsat: L/M/H
# Nominal pixel size (km) when the CSV has no usable scan/track (Landsat scan/track are pixel indices)
DEFAULT_PIXEL_KM = {"MODIS": 1.0, "VIIRS": 0.375, "LANDSAT": 0.03}
CONF_LEVEL = {"low": 0, "nominal": 1, "high": 2}
# FIRMS archive-download file codes (DL_FIRE_<code>_<id>.zip / fire_{archive,nrt}_<code>_<id>.*)
FIRMS_ARCHIVE_CODES = {"M-C61": "MODIS", "M-C6": "MODIS", "SV-C2": "VIIRS_SNPP",
                       "J1V-C2": "VIIRS_NOAA20", "J2V-C2": "VIIRS_NOAA21", "J2C-C2": "VIIRS_NOAA21"}
FIRMS_LOCAL_EXT = (".csv", ".json", ".geojson", ".shp", ".gpkg", ".zip", ".txt")
FEDS_LOCAL_EXT = (".fgb", ".gpkg", ".shp", ".geojson", ".json", ".parquet")
# FIRMS archive `type`: 0 presumed vegetation fire, 1 active volcano, 2 other static land source, 3 offshore
FIRMS_TYPES = {0: "vegetation", 1: "volcano", 2: "static_land", 3: "offshore"}


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
    cfg["_cfgdir"] = Path(path).resolve().parent
    return cfg


def resolve_inputs(spec, base: Path, exts, what: str) -> list[Path]:
    """
    Files from a path / glob / directory or a list of them. Relative entries
    are resolved against `base` (the config directory) first, then the cwd;
    directories are searched recursively for `exts`. Warns when an entry
    matches nothing.
    """
    if spec is None:
        return []
    items = [spec] if isinstance(spec, (str, Path)) else list(spec)
    out: list[Path] = []
    for it in items:
        it = os.path.expanduser(str(it))
        cands = [it] if os.path.isabs(it) else [os.path.normpath(str(base / it)), it]
        hits: list[str] = []
        for c in cands:
            if os.path.isdir(c):
                hits = [str(p) for p in sorted(Path(c).rglob("*")) if p.suffix.lower() in exts]
            else:
                hits = sorted(glob.glob(c, recursive=True))
                hits = [h for h in hits if os.path.isfile(h)]
            if hits:
                break
        if not hits:
            LOG.warning("%s: %r matched no files (looked in %s)", what, it, " and ".join(dict.fromkeys(
                os.path.abspath(c) for c in cands)))
        out += [Path(h) for h in hits]
    return list(dict.fromkeys(out))


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


def _feds_api_layer(cfg, fcfg, layer, start, end) -> gpd.GeoDataFrame | None:
    api = fcfg.get("api_url", FEDS_API)
    rows = []
    for col in feds_collections(layer, fcfg, start, end):
        try:
            feats = ogc_items(api, col, cfg["_bbox"], start, end, fcfg.get("page_limit", 1000))
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
        return None
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


# --- local FEDS vectors ------------------------------------------------------

_FEDS_STEP_RE = re.compile(r"(\d{8})(AM|PM)", re.I)
_FEDS_FID_RE = re.compile(r"(?:^|[/\\_])F(\d+)_\d{8}(?:AM|PM)", re.I)


def _feds_layer_name(name: str) -> str | None:
    n = name.lower()
    for layer in ("newfirepix", "fireline", "perimeter"):  # newfirepix before 'fire...'
        if layer in n:
            return layer
    return None


def _list_layers(path: Path) -> list[str | None]:
    if path.suffix.lower() != ".gpkg":
        return [None]
    try:
        import pyogrio
        return [str(l[0]) for l in pyogrio.list_layers(path)]
    except Exception:  # fiona fallback
        import fiona
        return list(fiona.listlayers(path))


def _feds_time(g: gpd.GeoDataFrame, path: Path) -> pd.Series:
    """Step time: `t` (fireatlas / VEDA), else time/date columns, else YYYYMMDD(AM|PM) in the path."""
    for c in ("t", "time", "datetime", "step_t", "date"):
        if c in g.columns:
            t = pd.to_datetime(g[c], utc=True, errors="coerce")
            if t.notna().any():
                return t
    m = _FEDS_STEP_RE.findall(str(path))
    if not m:
        raise ValueError(f"{path}: FEDS step time not found (no t/time/date column, no YYYYMMDD[AM|PM] in path)")
    d, half = m[-1]
    t = pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:]}T{'12' if half.upper() == 'PM' else '00'}:00:00", tz="UTC")
    return pd.Series([t] * len(g), index=g.index)


def read_feds_local(fcfg: dict, cfg: dict) -> dict[str, list[gpd.GeoDataFrame]]:
    """
    feds.local_vectors: path / glob / directory (or a list), or a mapping
    {fireline: [...], perimeter: [...], newfirepix: [...]}. Without a mapping
    the layer comes from the file or GPKG layer name (perimeter.fgb,
    lf_fireline.fgb, F123_20240726PM/newfirepix.fgb, ...); nfplist / uptonow /
    other files are skipped. Geometries are reprojected to EPSG:4326
    (feds.local_crs for files without a CRS; fireatlas default EPSG:9311).
    """
    spec = fcfg.get("local_vectors")
    by_layer = spec if isinstance(spec, dict) else {None: spec}
    out: dict[str, list] = {l: [] for l in FEDS_LAYERS}
    bbox_geom = box(*cfg["_bbox"])
    for forced, entries in by_layer.items():
        if forced is not None and forced not in FEDS_LAYERS:
            raise ValueError(f"feds.local_vectors: unknown layer {forced!r} ({', '.join(FEDS_LAYERS)})")
        for path in resolve_inputs(entries, cfg["_cfgdir"], FEDS_LOCAL_EXT, "feds.local_vectors"):
            for lname in _list_layers(path):
                layer = forced or _feds_layer_name(lname or path.stem) or _feds_layer_name(path.parent.name)
                if layer is None:
                    LOG.debug("FEDS local: skipping %s%s", path, f" layer {lname}" if lname else "")
                    continue
                g = gpd.read_parquet(path) if path.suffix.lower() == ".parquet" else \
                    gpd.read_file(path, layer=lname) if lname else gpd.read_file(path)
                n_read = len(g)
                if g.empty:
                    continue
                if g.crs is None:
                    g = g.set_crs(fcfg.get("local_crs", "EPSG:9311"))
                g = g.to_crs("EPSG:4326")
                g = g[g.geometry.notna() & ~g.geometry.is_empty]
                g.columns = [c if c == g.geometry.name else str(c).lower() for c in g.columns]
                g = g.loc[:, ~g.columns.duplicated()]
                g["t"] = _feds_time(g, path)
                if "fireid" not in g.columns:
                    m = _FEDS_FID_RE.search(str(path))
                    g["fireid"] = m.group(1) if m else path.parent.name
                g = g[g.intersects(bbox_geom)]
                g["collection"] = f"local:{path.name}" + (f":{lname}" if lname else "")
                LOG.info("FEDS local %s%s -> %s: %d features, %d in bbox", path,
                         f" [{lname}]" if lname else "", layer, n_read, len(g))
                if len(g):
                    out[layer].append(g)
    return out


def pull_feds(cfg: dict, local_only: bool = False) -> dict:
    fcfg = cfg.get("feds") or {}
    start, end = cfg["_start"], cfg["_end"]
    # FEDS steps are labeled at 00/12 local-solar-ish hours; pad so steps whose
    # estimated overpass falls inside the window are kept.
    pad = pd.Timedelta(hours=float(fcfg.get("time_pad_hours", 12)))
    source = str(fcfg.get("source", "auto")).lower()
    if source == "auto":
        source = "local" if fcfg.get("local_vectors") else "api"
    if source not in ("local", "api", "both"):
        raise ValueError("feds.source must be auto | local | api | both")
    if local_only and source in ("api", "both"):
        LOG.info("FEDS: --local-only, VEDA API not queried")
        source = "local" if source == "both" else "none"
    local = read_feds_local(fcfg, cfg) if source in ("local", "both") else {}
    if source in ("local", "both") and not any(local.values()):
        LOG.warning("FEDS: no local features read from feds.local_vectors in the bbox")
    out = {}
    for layer in fcfg.get("layers", list(FEDS_LAYERS)):
        if layer not in FEDS_LAYERS:
            raise ValueError(f"feds.layers: unknown layer {layer!r}")
        parts = list(local.get(layer, []))
        if source in ("api", "both"):
            g_api = _feds_api_layer(cfg, fcfg, layer, start - pad, end + pad)
            if g_api is not None:
                parts.append(g_api)
        if not parts:
            LOG.warning("FEDS %s: no features in bbox/time range", layer)
            out[layer] = None
            continue
        g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs="EPSG:4326")
        g["fireid"] = g["fireid"].astype(str).str.replace(r"\.0$", "", regex=True)
        g["step_t"] = pd.to_datetime(g["t"], utc=True)
        n0 = len(g)
        g = g[g["step_t"].notna()]
        lon = g.geometry.representative_point().x.to_numpy()
        g["obs_time"], g["step_half"] = feds_obs_time(g["step_t"], lon, fcfg.get("overpass_lst_hours", {}))
        g = g[(g["obs_time"] >= start) & (g["obs_time"] <= end)].copy()
        if len(g) < n0:
            LOG.info("FEDS %s: %d of %d features in time_range %s .. %s", layer, len(g), n0, start, end)
        if g.empty:
            LOG.warning("FEDS %s: no features left in time_range", layer)
            out[layer] = None
            continue
        # snapshot / large-fire collections (and local + API copies) overlap: drop exact duplicates
        g["_wkb"] = g.geometry.apply(lambda s: s.wkb_hex)
        g = g.drop_duplicates(subset=["step_t", "_wkb"]).drop(columns=["_wkb", "t"], errors="ignore")
        g.insert(0, "layer", layer)
        g.insert(0, "source", "feds")
        g = g.drop(columns=[c for c in ("wkb_geometry", "ogc_fid", "geom_counts") if c in g], errors="ignore")
        for c in g.columns:  # mixed-type object columns from heterogeneous files break Parquet
            if c != g.geometry.name and g[c].dtype == object:
                g[c] = g[c].map(lambda v: v if v is None or isinstance(v, str) else str(v))
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
    # acq_time: HHMM (MODIS/VIIRS), "HH:MM", or "HH:MM-SS" / "HH:MM:SS" (Landsat)
    tod = df["acq_time"].astype(str).str.strip().str.split().str[-1].str.replace(r"\D", "", regex=True)
    hhmm = tod.where(tod.str.len() <= 4, tod.str[:-2]).str.zfill(4).str[-4:]
    df["acq_time"] = pd.to_datetime(df["acq_date"].astype(str) + " " + hhmm, format="%Y-%m-%d %H%M", utc=True)
    conf = df.get("confidence", pd.Series(["n"] * len(df))).astype(str).str.strip().str.lower()
    num = pd.to_numeric(conf, errors="coerce")  # MODIS: 0-100
    lev = conf.map(VIIRS_CONF)
    lev = lev.where(lev.notna(), np.select([num >= 80, num >= 30], [2, 1], 0))
    df["confidence"] = conf
    df["confidence_level"] = lev.astype(int)
    df["firms_source"] = firms_source
    src_u = str(firms_source).upper()
    if "satellite" not in df:
        df["satellite"] = firms_source
    if "instrument" not in df:
        landsat = "LANDSAT" in src_u or df["satellite"].astype(str).str.upper().str.match(r"^L[89]$").all()
        df["instrument"] = "LANDSAT" if landsat else ("MODIS" if "MODIS" in src_u else "VIIRS")
    df["instrument"] = df["instrument"].astype(str).str.upper().replace({"OLI": "LANDSAT"})
    # Keep NRT/SP level for de-duplication of the same satellite across processing levels
    df["firms_level"] = "SP" if src_u.endswith("_SP") else ("NRT" if src_u.endswith("_NRT") else "LOCAL")
    for c in ("scan", "track", "frp"):
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
    default_px = df["instrument"].map(lambda i: next((v for k, v in DEFAULT_PIXEL_KM.items() if i.startswith(k)), 0.375))
    is_ls = df["instrument"].str.startswith("LANDSAT")
    df.loc[is_ls, ["scan", "track"]] = np.nan          # Landsat scan/track are OLI line/sample indices
    df["scan"] = df["scan"].fillna(default_px)
    df["track"] = df["track"].fillna(default_px)
    if "daynight" not in df:
        df["daynight"] = ""
    return df


def infer_firms_source(name: str, default: str | None = None) -> str:
    """FIRMS source label from an archive-download file name (fire_archive_SV-C2_123 -> VIIRS_SNPP_SP)."""
    m = re.search(r"(archive|nrt)_(LS|L8|L9|LANDSAT)(?:_|$)", name, re.I)
    if m:  # Landsat downloads (fire_nrt_LS_<id>): NRT only
        return "LANDSAT_NRT"
    m = re.search(r"(archive|nrt)_([A-Z0-9]+-C\d+)", name, re.I)
    if m:
        fam = FIRMS_ARCHIVE_CODES.get(m.group(2).upper(), m.group(2).upper())
        return f"{fam}_{'SP' if m.group(1).lower() == 'archive' else 'NRT'}"
    return default or name


def _read_table(path: Path, member: str | None = None) -> pd.DataFrame:
    """FIRMS detections table from .csv / .txt / .json / .geojson / .shp / .gpkg (optionally inside a zip)."""
    ext = Path(member or path.name).suffix.lower()
    if member:
        with zipfile.ZipFile(path) as z:
            if ext in (".csv", ".txt"):
                with z.open(member) as f:
                    return pd.read_csv(f)
            if ext == ".json":
                with z.open(member) as f:
                    raw = f.read()
                try:
                    return pd.read_json(io.BytesIO(raw))
                except ValueError:
                    return pd.DataFrame(gpd.read_file(io.BytesIO(raw)).drop(columns="geometry"))
        uri = f"zip://{path.resolve()}!{member}"
        return pd.DataFrame(gpd.read_file(uri).drop(columns="geometry", errors="ignore"))
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    if ext == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            pass
    return pd.DataFrame(gpd.read_file(path).drop(columns="geometry", errors="ignore"))


def read_firms_local(fcfg: dict, cfg: dict) -> list[pd.DataFrame]:
    """
    firms.local_csv: path / glob / directory or a list (FIRMS archive-download
    .csv / .json / .shp, or the DL_FIRE_*.zip). One table per file / zip member;
    firms_source from firms.local_source_name, else the file name
    (fire_archive_* -> *_SP, fire_nrt_* -> *_NRT), so NRT rows are superseded
    by SP rows of the same satellite (prefer_standard_processing).
    """
    frames = []
    for f in resolve_inputs(fcfg.get("local_csv"), cfg["_cfgdir"], FIRMS_LOCAL_EXT, "firms.local_csv"):
        members = [None]
        if f.suffix.lower() == ".zip":
            with zipfile.ZipFile(f) as z:
                names = [n for n in z.namelist() if not n.endswith("/")]
            members = [n for n in names if n.lower().endswith((".csv", ".txt")) and "readme" not in n.lower()] or \
                      [n for n in names if n.lower().endswith(".json")] or \
                      [n for n in names if n.lower().endswith(".shp")]
            if not members:
                LOG.warning("FIRMS local %s: no .csv / .json / .shp inside", f)
                continue
        for m in members:
            df = _read_table(f, m)
            label = f"{f.name}!{m}" if m else str(f)
            if df.empty:
                LOG.warning("FIRMS local %s: no rows", label)
                continue
            src = infer_firms_source(Path(m or f.name).stem, fcfg.get("local_source_name"))
            frames.append(_norm_firms(df, src))
            d = frames[-1]
            LOG.info("FIRMS local %s: %d rows (%s, %s .. %s, lon %.2f..%.2f, lat %.2f..%.2f)", label, len(d), src,
                     d.acq_time.min(), d.acq_time.max(), d.longitude.min(), d.longitude.max(),
                     d.latitude.min(), d.latitude.max())
    return frames


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
    frames += read_firms_local(fcfg, cfg)
    use_api = bool(fcfg.get("sources"))
    if use_api and not local_only and not os.environ.get(fcfg.get("map_key_env", "FIRMS_MAP_KEY")) and frames:
        LOG.warning("FIRMS: no MAP_KEY in $%s; firms.sources (Area API) skipped, using the %d local file(s)",
                    fcfg.get("map_key_env", "FIRMS_MAP_KEY"), len(frames))
        use_api = False
    if use_api:
        if local_only:
            for f in sorted(raw_dir.glob("*.csv")):
                if f.stat().st_size == 0 or not f.read_text().strip():
                    continue
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
    LOG.info("FIRMS: %d detections read", len(df))
    x0, y0, x1, y1 = bbox

    def keep(mask, what):
        nonlocal df
        n0 = len(df)
        df = df[mask]
        LOG.info("FIRMS: %-28s %d -> %d", what, n0, len(df))
        if n0 and df.empty:
            LOG.warning("FIRMS: the %s filter removed every detection", what)

    keep((df.longitude >= x0) & (df.longitude <= x1) & (df.latitude >= y0) & (df.latitude <= y1),
         f"bbox {x0},{y0},{x1},{y1}")
    keep((df.acq_time >= start) & (df.acq_time <= end), f"time {start:%Y-%m-%d} .. {end:%Y-%m-%d}")
    minc = CONF_LEVEL[str(fcfg.get("min_confidence", "nominal")).lower()]
    keep(df.confidence_level >= minc, f"confidence >= {fcfg.get('min_confidence', 'nominal')}")
    # archive `type` (SP only): 0 vegetation fire, 1 volcano, 2 static land source, 3 offshore
    types = fcfg.get("keep_types", [0])
    if types is not None and "type" in df:
        t = pd.to_numeric(df["type"], errors="coerce")
        keep(t.isna() | t.isin([int(v) for v in types]), f"type in {list(types)}")
    if not fcfg.get("include_night", True):
        keep(df.daynight.astype(str).str.upper() != "N", "daytime only")
    df = df.drop_duplicates(subset=["latitude", "longitude", "acq_time", "satellite"])
    if fcfg.get("prefer_standard_processing", True):
        # Same satellite pulled as *_NRT and *_SP: keep SP on dates where SP has data
        fam = df["firms_source"].astype(str).str.replace(r"_(NRT|SP)$", "", regex=True)
        day = df["acq_time"].dt.floor("D")
        has_sp = (df["firms_level"] == "SP").groupby([fam, day]).transform("any")
        n0 = len(df)
        df = df[~((df["firms_level"] == "NRT") & has_sp)]
        if n0 - len(df):
            LOG.info("FIRMS: dropped %d NRT detections superseded by SP", n0 - len(df))
    if df.empty:
        LOG.warning("FIRMS: no detections left after bbox / time / confidence filters")
        return None
    df = assign_overpasses(df, float(fcfg.get("overpass_gap_minutes", 20)))
    cols = ["firms_source", "firms_level", "satellite", "instrument", "acq_time", "overpass_id", "overpass_time",
            "confidence", "confidence_level", "frp", "scan", "track", "daynight", "latitude", "longitude"]
    g = gpd.GeoDataFrame(df[cols].copy(), geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    g.insert(0, "layer", "detection")
    g.insert(0, "source", "firms")
    LOG.info("FIRMS: %d detections in %d overpasses", len(g), g["overpass_id"].nunique())
    return g.reset_index(drop=True)


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--local-only", action="store_true",
                    help="no network: cached FIRMS Area API CSVs, firms.local_csv and feds.local_vectors only")
    ap.add_argument("--skip-feds", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(a.config)
    manifest = {"bbox": cfg["_bbox"], "start": str(cfg["_start"]), "end": str(cfg["_end"]), "layers": {}}
    if (cfg.get("feds") or {}).get("enabled", True) and not a.skip_feds:
        for layer, g in pull_feds(cfg, local_only=a.local_only).items():
            if g is not None and len(g):
                manifest["layers"][f"feds_{layer}"] = {
                    "files": write_layer(g, cfg, f"feds_{layer}"), "n": int(len(g)),
                    "steps": int(g["step_t"].nunique()), "collections": sorted(g["collection"].unique())}
    fc = cfg.get("firms") or {}
    firms_on = fc.get("enabled", "auto")
    if firms_on == "auto":
        firms_on = bool(fc.get("sources") or fc.get("local_csv"))
    elif not firms_on and fc.get("local_csv"):
        LOG.warning("firms.enabled is false: firms.local_csv %s is not read", fc.get("local_csv"))
    if firms_on:
        g = pull_firms(cfg, local_only=a.local_only)
        if g is not None:
            manifest["layers"]["firms_detections"] = {
                "files": write_layer(g, cfg, "firms_detections"), "n": int(len(g)),
                "overpasses": int(g["overpass_id"].nunique()), "sources": sorted(g["firms_source"].unique())}
    d = vector_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    LOG.info("Vector layers in %s: %s", d.resolve(), ", ".join(manifest["layers"]) or "none")
    (d / "pull_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if not manifest["layers"]:
        LOG.warning("Nothing written. FEDS covers 2018-2021 and 2025-01 onward; use firms for other years.")
    LOG.info("Wrote %s", d / "pull_manifest.json")


if __name__ == "__main__":
    sys.exit(main())
