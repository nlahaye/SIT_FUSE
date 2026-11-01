#!/usr/bin/env python3
"""
Pull VIIRS Deep Blue L2 aerosol type (AERDB_L2) and write per-overpass
smoke GeoTIFFs on the same grid as pull_adp.py.

Output encoding (one band, EPSG:4326, raster.resolution_deg):
    1            smoke            (aerosol_type.smoke_values, default 1, 2, 3:
                                   smoke, high-altitude smoke, pyroCb)
    0            observed, not smoke (aerosol_type.negative_values)
    fill_value   unobserved / no retrieval / below QA / unassigned types
                 (default 255, written as the GeoTIFF nodata)

Files are named {prefix}_{YYYYMMDDTHHMMSSZ}_{PLATFORM}.tif, where the time is
the median scan time of the retrieved pixels inside the bbox, so
adp_matchup_generator.py can match them to SIT-FUSE scenes unchanged.

Workflow
    1. CMR granule search (cmr.earthdata.nasa.gov) per collection and version,
       restricted to time_range, bbox and daytime.
    2. Download with an Earthdata Login bearer token (env var named by
       auth.token_env, default EARTHDATA_TOKEN) or ~/.netrc credentials.
    3. Map Aerosol_Type_Land_Ocean (or Aerosol_Type_Land) to smoke / not
       smoke / unobserved, apply the QA threshold, rasterize each granule with
       pull_adp.rasterize_swath_points.
    4. Merge consecutive granules of one platform into one overpass raster
       (smoke where any granule saw smoke; not smoke where observed and no
       smoke; fill elsewhere).

Usage:
    export EARTHDATA_TOKEN=...   # https://urs.earthdata.nasa.gov -> Generate Token
    python pull_viirs_aerosol_type.py viirs_aerosol_type.example.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import requests
import xarray as xr

try:
    from sit_fuse.preprocessing.data_pull.pull_adp import (
        load_config, parse_datetime_utc, rasterize_swath_points, write_geotiff)
except ImportError:  # run from the data_pull directory
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pull_adp import load_config, parse_datetime_utc, rasterize_swath_points, write_geotiff


CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
URS_HOST = "urs.earthdata.nasa.gov"
TAI93 = dt.datetime(1993, 1, 1)
# leap seconds inserted after 1993-01-01 (TAI93 counts them; UTC does not)
LEAP_SECONDS = [dt.datetime(1993, 7, 1), dt.datetime(1994, 7, 1), dt.datetime(1996, 1, 1),
                dt.datetime(1997, 7, 1), dt.datetime(1999, 1, 1), dt.datetime(2006, 1, 1),
                dt.datetime(2009, 1, 1), dt.datetime(2012, 7, 1), dt.datetime(2015, 7, 1),
                dt.datetime(2017, 1, 1)]
GRANULE_RE = re.compile(r"(?P<short>AERDB_L2_VIIRS_(?P<platform>[A-Z0-9]+))\.A(?P<yd>\d{7})\.(?P<hm>\d{4})\.")

DEFAULT_COLLECTIONS = [
    {"short_name": "AERDB_L2_VIIRS_SNPP", "versions": ["2.1", "2"]},
    {"short_name": "AERDB_L2_VIIRS_NOAA20", "versions": ["2.1", "2"]},
    {"short_name": "AERDB_L2_VIIRS_NOAA21", "versions": ["2.1", "2"]},
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def validate_config(cfg: dict) -> dict:
    for key in ("time_range", "bbox", "raster", "output"):
        if key not in cfg:
            raise ValueError(f"Missing required top-level key: {key}")
    start = parse_datetime_utc(cfg["time_range"]["start"])
    end = parse_datetime_utc(cfg["time_range"]["end"])
    if end < start:
        raise ValueError("time_range.end must be >= time_range.start")
    bb = cfg["bbox"]
    bbox = (float(bb["min_lon"]), float(bb["min_lat"]), float(bb["max_lon"]), float(bb["max_lat"]))
    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise ValueError("Invalid bbox ordering")

    r = cfg["raster"]
    fill_value = float(r.get("fill_value", 255.0))
    at = cfg.get("aerosol_type", {}) or {}
    smoke_values = [int(v) for v in at.get("smoke_values", [1, 2, 3])]
    negative_values = [int(v) for v in at.get("negative_values", [0, 4, 6, 7])]
    if set(smoke_values) & set(negative_values):
        raise ValueError("aerosol_type.smoke_values and negative_values overlap")
    if fill_value in (0.0, 1.0):
        raise ValueError("raster.fill_value must differ from the labels 0 and 1")
    qa = at.get("min_qa", 1)
    ov = cfg.get("overpass", {}) or {}
    out = cfg["output"]
    return {
        "start": start, "end": end, "bbox": bbox,
        "resolution_deg": float(r.get("resolution_deg", 0.05)),
        "footprint_scale": float(r.get("footprint_scale", 0.55)),
        "fill_value": fill_value,
        "nodata": float(r.get("nodata", fill_value)),
        "dtype": str(r.get("dtype", "uint8")),
        "collections": cfg.get("collections") or DEFAULT_COLLECTIONS,
        "day_only": bool(cfg.get("day_only", True)),
        "variable": str(at.get("variable", "Aerosol_Type_Land_Ocean")),
        "smoke_values": smoke_values,
        "negative_values": negative_values,
        "min_qa": int(qa) if qa is not None else 0,
        "qa_applies_to": str(at.get("qa_applies_to", "all")).lower(),
        "max_gap_seconds": float(ov.get("max_gap_seconds", 600)),
        "min_observed_pixels": int(ov.get("min_observed_pixels", 1)),
        "outdir": Path(out["outdir"]),
        "prefix": str(out.get("prefix", "viirs_smoke")),
        "keep_nc": bool(out.get("keep_nc", True)),
        "write_granule_rasters": bool(out.get("write_granule_rasters", False)),
        "overwrite": bool(cfg.get("overwrite", False)),
        "token_env": str((cfg.get("auth", {}) or {}).get("token_env", "EARTHDATA_TOKEN")),
    }


# ---------------------------------------------------------------------------
# Search and download
# ---------------------------------------------------------------------------

def _granule_key(name: str):
    m = GRANULE_RE.search(name)
    if not m:
        return None
    t = dt.datetime.strptime(m.group("yd") + m.group("hm"), "%Y%j%H%M")
    return m.group("platform"), t


def search_granules(cfg: dict) -> list[dict]:
    """CMR search; per platform and granule time keeps the first listed version."""
    w, s, e, n = cfg["bbox"]
    temporal = f"{cfg['start']:%Y-%m-%dT%H:%M:%SZ},{cfg['end']:%Y-%m-%dT%H:%M:%SZ}"
    chosen: dict[tuple, dict] = {}
    for coll in cfg["collections"]:
        for version in coll.get("versions", [None]):
            params = {"short_name": coll["short_name"], "temporal": temporal,
                      "bounding_box": f"{w},{s},{e},{n}", "page_size": 2000, "page_num": 1}
            if version is not None:
                params["version"] = str(version)
            found = 0
            while True:
                r = requests.get(CMR_URL, params=params, timeout=120)
                r.raise_for_status()
                entries = r.json().get("feed", {}).get("entry", [])
                for g in entries:
                    if cfg["day_only"] and str(g.get("day_night_flag", "")).upper() == "NIGHT":
                        continue
                    name = g.get("producer_granule_id") or g.get("title", "")
                    key = _granule_key(name)
                    url = next((l["href"] for l in g.get("links", [])
                                if l.get("rel", "").endswith("/data#") and l["href"].startswith("https")
                                and l["href"].endswith(".nc")), None)
                    if key is None or url is None or key in chosen:
                        continue
                    chosen[key] = {"platform": key[0], "time": key[1], "name": Path(url).name,
                                   "url": url, "version": version, "short_name": coll["short_name"]}
                    found += 1
                if len(entries) < params["page_size"]:
                    break
                params["page_num"] += 1
            print(f"[cmr] {coll['short_name']} v{version}: {found} new granules")
    return sorted(chosen.values(), key=lambda g: (g["platform"], g["time"]))


class EarthdataSession(requests.Session):
    """Sends credentials only to the data host and Earthdata Login; never to
    the redirect target (e.g. a pre-signed S3 URL), as in the NASA examples."""

    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        orig = requests.utils.urlparse(response.request.url).hostname
        redir = requests.utils.urlparse(prepared_request.url).hostname
        if "Authorization" in headers and orig != redir and URS_HOST not in (orig, redir):
            del headers["Authorization"]
        if self.trust_env and redir == URS_HOST:
            auth = requests.utils.get_netrc_auth(prepared_request.url)
            if auth:
                prepared_request.prepare_auth(auth)


def make_session(token_env: str) -> EarthdataSession:
    sess = EarthdataSession()
    token = os.environ.get(token_env)
    if token:
        sess.headers["Authorization"] = f"Bearer {token}"
    else:
        print(f"[auth] {token_env} not set; using ~/.netrc credentials for {URS_HOST}")
    return sess


def download(sess: requests.Session, url: str, dest: Path, overwrite: bool = False) -> Path:
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with sess.get(url, stream=True, timeout=300) as r:
        if r.status_code in (401, 403):
            raise RuntimeError(f"HTTP {r.status_code} for {Path(url).name}: set an Earthdata Login token "
                               "in the env var named by auth.token_env, or add "
                               f"'machine {URS_HOST} login USER password PASS' to ~/.netrc")
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    with open(tmp, "rb") as f:
        magic = f.read(8)
    if not magic.startswith(b"\x89HDF") and not magic.startswith(b"CDF"):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{url} did not return netCDF (Earthdata Login failed or "
                           "LAADS application not authorized?)")
    tmp.replace(dest)
    print(f"[ok] downloaded {dest.name}")
    return dest


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def tai93_to_utc(seconds: float) -> dt.datetime:
    t = TAI93 + dt.timedelta(seconds=float(seconds))
    return t - dt.timedelta(seconds=sum(t >= ls for ls in LEAP_SECONDS))


def smoke_tristate(nc_path: Path, cfg: dict):
    """Per-pixel 1 / 0 / NaN (unobserved) plus lon, lat and scan time (TAI93)."""
    with xr.open_dataset(nc_path, engine="netcdf4", mask_and_scale=False) as ds:
        lat = ds["Latitude"].values.astype(np.float64)
        lon = ds["Longitude"].values.astype(np.float64)
        typ = ds[cfg["variable"]].values.astype(np.int32)
        qa_land = ds["Aerosol_Optical_Thickness_QA_Flag_Land"].values.astype(np.int32) \
            if "Aerosol_Optical_Thickness_QA_Flag_Land" in ds else None
        qa_ocean = ds["Aerosol_Optical_Thickness_QA_Flag_Ocean"].values.astype(np.int32) \
            if "Aerosol_Optical_Thickness_QA_Flag_Ocean" in ds else None
        scan = ds["Scan_Start_Time"].values.astype(np.float64) if "Scan_Start_Time" in ds else None
    arr = np.full(typ.shape, np.nan, dtype=np.float32)
    arr[np.isin(typ, cfg["negative_values"])] = 0.0
    arr[np.isin(typ, cfg["smoke_values"])] = 1.0          # -999, unlisted types stay NaN
    if cfg["min_qa"] > 0 and qa_land is not None:
        qa = qa_land if qa_ocean is None or cfg["variable"].endswith("_Land") \
            else np.where(qa_land > 0, qa_land, qa_ocean)
        low = qa < cfg["min_qa"]
        if cfg["qa_applies_to"] == "negatives":
            low &= arr == 0
        arr[low] = np.nan
    lon = np.where((lon < -180) | (lon > 180), np.nan, lon)
    lat = np.where((lat < -90) | (lat > 90), np.nan, lat)
    if scan is not None:
        scan = np.where(scan < 0, np.nan, scan)
    return lon, lat, arr, scan


def in_bbox(lon, lat, bbox):
    w, s, e, n = bbox
    return np.isfinite(lon) & np.isfinite(lat) & (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)


def granule_raster(nc_path: Path, cfg: dict):
    lon, lat, arr, scan = smoke_tristate(nc_path, cfg)
    inside = in_bbox(lon, lat, cfg["bbox"]) & np.isfinite(arr)
    times = scan[inside & np.isfinite(scan)] if scan is not None else np.array([])
    ras = rasterize_swath_points(lon=lon, lat=lat, values=arr, bbox=cfg["bbox"],
                                 resolution_deg=cfg["resolution_deg"], fill_value=np.nan,
                                 agg="max", footprint_scale=cfg["footprint_scale"])
    return ras, times, int(inside.sum())


def group_overpasses(granules: list[dict], max_gap_s: float) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for g in granules:
        if groups and groups[-1][-1]["platform"] == g["platform"] \
                and (g["time"] - groups[-1][-1]["time"]).total_seconds() <= max_gap_s:
            groups[-1].append(g)
        else:
            groups.append([g])
    return groups


def merge(rasters: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(rasters)
    observed = np.isfinite(stack).any(axis=0)
    out = np.where(observed, np.nanmax(np.where(observed[None], stack, -1.0), axis=0), np.nan)
    return out.astype(np.float32)


def write_overpass(ras: np.ndarray, t: dt.datetime, platform: str, cfg: dict, sub: str = "rasters") -> Path:
    out = np.where(np.isfinite(ras), ras, cfg["fill_value"])
    name = f"{cfg['prefix']}_{t:%Y%m%dT%H%M%SZ}_{platform}.tif"
    return write_geotiff(out, cfg["bbox"], cfg["resolution_deg"], cfg["outdir"] / sub / name,
                         cfg["dtype"], cfg["nodata"])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Pull VIIRS Deep Blue aerosol type and write smoke GeoTIFFs.")
    ap.add_argument("config", type=Path)
    ap.add_argument("--search-only", action="store_true", help="list granules and overpasses, no download")
    a = ap.parse_args()
    cfg = validate_config(load_config(a.config))

    granules = search_granules(cfg)
    overpasses = group_overpasses(granules, cfg["max_gap_seconds"])
    print(f"{len(granules)} granules in {len(overpasses)} overpasses")
    if a.search_only:
        for ov in overpasses:
            print(f"  {ov[0]['platform']:7s} {ov[0]['time']:%Y-%m-%d %H:%M}  "
                  + " ".join(g["name"] for g in ov))
        return
    if not granules:
        sys.exit(0)

    sess = make_session(cfg["token_env"])
    manifest = []
    for ov in overpasses:
        rasters, times, n_in = [], [], 0
        for g in ov:
            try:
                nc = download(sess, g["url"], cfg["outdir"] / "raw_nc" / f"{g['time']:%Y%m%d}" / g["name"],
                              cfg["overwrite"])
                ras, ts, n = granule_raster(nc, cfg)
                rasters.append(ras)
                times.append(ts)
                n_in += n
                if cfg["write_granule_rasters"]:
                    write_overpass(ras, g["time"], g["platform"], cfg, sub="granules")
                if not cfg["keep_nc"]:
                    nc.unlink(missing_ok=True)
            except Exception as exc:
                print(f"[err] {g['name']}: {exc}")
        if not rasters:
            continue
        ras = merge(rasters)
        n_obs = int(np.isfinite(ras).sum())
        if n_obs < cfg["min_observed_pixels"]:
            print(f"[skip] {ov[0]['platform']} {ov[0]['time']:%Y-%m-%d %H:%M}: "
                  f"{n_obs} observed grid cells in bbox")
            continue
        tt = np.concatenate(times)
        t = tai93_to_utc(np.median(tt)) if tt.size else ov[0]["time"]
        t = t.replace(microsecond=0)
        if os.path.exists(path := cfg["outdir"] / "rasters" /
                          f"{cfg['prefix']}_{t:%Y%m%dT%H%M%SZ}_{ov[0]['platform']}.tif") and not cfg["overwrite"]:
            print(f"[skip] {path.name} exists")
            continue
        path = write_overpass(ras, t, ov[0]["platform"], cfg)
        rec = {"raster": str(path), "platform": ov[0]["platform"], "time_utc": f"{t:%Y-%m-%dT%H:%M:%SZ}",
               "granules": [g["name"] for g in ov], "versions": sorted({str(g["version"]) for g in ov}),
               "smoke_cells": int((ras == 1).sum()), "not_smoke_cells": int((ras == 0).sum()),
               "unobserved_cells": int((~np.isfinite(ras)).sum()), "retrieved_pixels_in_bbox": n_in}
        manifest.append(rec)
        print(f"[ok] {path.name}: smoke={rec['smoke_cells']} not_smoke={rec['not_smoke_cells']} "
              f"unobserved={rec['unobserved_cells']}")
    cfg["outdir"].mkdir(parents=True, exist_ok=True)
    mpath = cfg["outdir"] / f"{cfg['prefix']}_manifest.json"
    old = json.loads(mpath.read_text()) if mpath.exists() else []
    keep = {r["raster"]: r for r in old}
    keep.update({r["raster"]: r for r in manifest})
    mpath.write_text(json.dumps(sorted(keep.values(), key=lambda r: r["time_utc"]), indent=2))
    print(f"\n{len(manifest)} overpass rasters written; manifest {mpath}")


if __name__ == "__main__":
    main()
