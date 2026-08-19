#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import rasterio
import requests
import xarray as xr
import yaml
from rasterio.transform import from_origin


BASE_URL = "https://www.star.nesdis.noaa.gov/pub/smcd/TEMPO_latest/ADP/"


FILENAME_RE = re.compile(
    r"(?P<name>TEMPO-ABI_ADP_L2_[^_]+_(?P<ts>\d{8}T\d{6})Z_[^\"'<> ]+\.nc)"
)


def parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def parse_datetime_utc(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def daterange(start_date: dt.date, end_date: dt.date):
    current = start_date
    while current <= end_date:
        yield current
        current += dt.timedelta(days=1)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must parse to a dictionary.")
    return cfg


def validate_config(cfg: dict) -> dict:
    required = ["time_range", "bbox", "raster", "output", "product"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing required top-level key: {key}")

    tr = cfg["time_range"]
    start = parse_date(tr["start"])
    end = parse_date(tr["end"])
    if end < start:
        raise ValueError("time_range.end must be >= time_range.start")

    bb = cfg["bbox"]
    bbox = (
        float(bb["min_lon"]),
        float(bb["min_lat"]),
        float(bb["max_lon"]),
        float(bb["max_lat"]),
    )
    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise ValueError("Invalid bbox ordering")

    raster_cfg = cfg["raster"]
    resolution_deg = float(raster_cfg.get("resolution_deg", 0.05))
    fill_value = float(raster_cfg.get("fill_value", 0.0))
    dtype = str(raster_cfg.get("dtype", "float32"))

    product = str(cfg["product"]).lower()
    valid_products = {
        "smoke_mask",
        "dust_mask",
        "saai_smoke",
        "saai_dust",
        "uv_aai",
        "deepblue_aai",
    }
    if product not in valid_products:
        raise ValueError(f"product must be one of {sorted(valid_products)}")

    quality = cfg.get("quality", {})
    smoke_conf = quality.get("smoke_confidence", None)
    dust_conf = quality.get("dust_confidence", None)
    allowed_conf = {None, "high", "medium", "low", "high_medium", "all"}
    if smoke_conf not in allowed_conf or dust_conf not in allowed_conf:
        raise ValueError("quality confidence must be one of null, high, medium, low, high_medium, all")

    combine = cfg.get("combine", None)
    if combine not in {None, "max", "sum", "count", "mean"}:
        raise ValueError("combine must be one of null, max, sum, count, mean")

    return {
        "start": start,
        "end": end,
        "bbox": bbox,
        "resolution_deg": resolution_deg,
        "fill_value": fill_value,
        "dtype": dtype,
        "product": product,
        "smoke_confidence": smoke_conf,
        "dust_confidence": dust_conf,
        "combine": combine,
        "overwrite": bool(cfg.get("overwrite", False)),
        "outdir": Path(cfg["output"]["outdir"]),
        "keep_nc": bool(cfg["output"].get("keep_nc", True)),
    }


def list_daily_files(day: dt.date) -> list[tuple[dt.datetime, str]]:
    day_url = urljoin(BASE_URL, f"{day:%Y%m%d}/")
    r = requests.get(day_url, timeout=60)
    if r.status_code == 404:
        return []
    r.raise_for_status()

    matches = FILENAME_RE.finditer(r.text)
    results = []
    seen = set()

    for m in matches:
        name = m.group("name")
        if name in seen:
            continue
        seen.add(name)
        ts = dt.datetime.strptime(m.group("ts"), "%Y%m%dT%H%M%S")
        results.append((ts, urljoin(day_url, name)))

    return sorted(results, key=lambda x: x[0])


def filter_files_in_time_range(start: dt.datetime, end: dt.datetime) -> list[tuple[dt.datetime, str]]:
    files = []
    for day in daterange(start, end):
        daily = list_daily_files(day)
        for ts, url in daily:
            if start <= ts.date() <= end:
                files.append((ts, url))
    return files


def download_file(url: str, dest: Path, overwrite: bool = False) -> Path:
    if dest.exists() and not overwrite:
        print(f"[skip] {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)

    print(f"[ok] downloaded {dest}")
    return dest


def make_grid(bbox: tuple[float, float, float, float], resolution_deg: float):
    min_lon, min_lat, max_lon, max_lat = bbox
    width = int(np.ceil((max_lon - min_lon) / resolution_deg))
    height = int(np.ceil((max_lat - min_lat) / resolution_deg))
    transform = from_origin(min_lon, max_lat, resolution_deg, resolution_deg)
    return width, height, transform


def confidence_mask_from_qc(qc: np.ndarray, kind: str, requested: str | None) -> np.ndarray:
    if requested in (None, "all"):
        return np.ones(qc.shape, dtype=bool)

    if kind == "smoke":
        vals = (qc.astype(np.int16) >> 2) & 0b11
    else:
        vals = (qc.astype(np.int16) >> 4) & 0b11

    if requested == "high":
        return vals == 0
    if requested == "medium":
        return vals == 1
    if requested == "low":
        return vals == 2
    if requested == "high_medium":
        return (vals == 0) | (vals == 1)

    return np.ones(qc.shape, dtype=bool)


def build_product_array(ds: xr.Dataset, product: str, smoke_conf: str | None, dust_conf: str | None) -> np.ndarray:
    lat = ds["/geolocation/latitude"].values
    shape = lat.shape

    if product == "smoke_mask":
        smoke = ds["/product/smoke"].values.astype(np.float32)
        qc = ds["/quality_diagnostic_flags/qc_flag"].values
        conf_mask = confidence_mask_from_qc(qc, "smoke", smoke_conf)
        arr = np.where((smoke == 1) & conf_mask, 1.0, np.nan)
        return arr.reshape(shape)

    if product == "dust_mask":
        dust = ds["/product/dust"].values.astype(np.float32)
        qc = ds["/quality_diagnostic_flags/qc_flag"].values
        pqi2 = ds["/quality_diagnostic_flags/pqi2"].values.astype(np.int16)
        conf_mask = confidence_mask_from_qc(qc, "dust", dust_conf)
        sun_glint_mask = (pqi2 & 2) == 0
        arr = np.where((dust == 1) & conf_mask & sun_glint_mask, 1.0, np.nan)
        return arr.reshape(shape)

    if product == "saai_smoke":
        saai = ds["/product/saai"].values.astype(np.float32)
        smoke = ds["/product/smoke"].values.astype(np.float32)
        qc = ds["/quality_diagnostic_flags/qc_flag"].values
        conf_mask = confidence_mask_from_qc(qc, "smoke", smoke_conf)
        arr = np.where((smoke == 1) & conf_mask & (saai > 0), saai, np.nan)
        return arr.reshape(shape)

    if product == "saai_dust":
        saai = ds["/product/saai"].values.astype(np.float32)
        dust = ds["/product/dust"].values.astype(np.float32)
        qc = ds["/quality_diagnostic_flags/qc_flag"].values
        pqi2 = ds["/quality_diagnostic_flags/pqi2"].values.astype(np.int16)
        conf_mask = confidence_mask_from_qc(qc, "dust", dust_conf)
        sun_glint_mask = (pqi2 & 2) == 0
        arr = np.where((dust == 1) & conf_mask & sun_glint_mask & (saai > 0), saai, np.nan)
        return arr.reshape(shape)

    if product == "uv_aai":
        uv_aai = ds["/product/uv_aai"].values.astype(np.float32)
        return uv_aai.reshape(shape)

    if product == "deepblue_aai":
        db_aai = ds["/product/deepblue_aai"].values.astype(np.float32)
        return db_aai.reshape(shape)

    raise ValueError(f"Unsupported product: {product}")


def rasterize_swath_points(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    bbox: tuple[float, float, float, float],
    resolution_deg: float,
    fill_value: float = 0.0,
    agg: str = "max",
) -> np.ndarray:
    min_lon, min_lat, max_lon, max_lat = bbox
    width, height, _ = make_grid(bbox, resolution_deg)

    out = np.full((height, width), np.nan, dtype=np.float32)
    count = np.zeros((height, width), dtype=np.int32)

    lonf = lon.ravel()
    latf = lat.ravel()
    valf = values.ravel()

    valid = (
        np.isfinite(lonf)
        & np.isfinite(latf)
        & np.isfinite(valf)
        & (lonf >= min_lon)
        & (lonf <= max_lon)
        & (latf >= min_lat)
        & (latf <= max_lat)
    )

    lonf = lonf[valid]
    latf = latf[valid]
    valf = valf[valid]

    if lonf.size == 0:
        return np.full((height, width), fill_value, dtype=np.float32)

    col = np.floor((lonf - min_lon) / resolution_deg).astype(int)
    row = np.floor((max_lat - latf) / resolution_deg).astype(int)

    inside = (row >= 0) & (row < height) & (col >= 0) & (col < width)
    row = row[inside]
    col = col[inside]
    valf = valf[inside]

    for r, c, v in zip(row, col, valf):
        if np.isnan(out[r, c]):
            out[r, c] = v
            count[r, c] = 1
        else:
            if agg == "sum":
                out[r, c] += v
            elif agg == "mean":
                out[r, c] += v
                count[r, c] += 1
            else:
                out[r, c] = max(out[r, c], v)

    if agg == "mean":
        mask = count > 0
        out[mask] = out[mask] / count[mask]

    out = np.where(np.isfinite(out), out, fill_value).astype(np.float32)
    return out


def write_geotiff(arr: np.ndarray, bbox: tuple[float, float, float, float], resolution_deg: float, out_path: Path, dtype: str):
    width, height, transform = make_grid(bbox, resolution_deg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
        compress="lzw",
        nodata=0,
    ) as dst:
        dst.write(arr.astype(dtype), 1)

    return out_path


def process_one_file(
    nc_path: Path,
    cfg: dict,
    timestamp: dt.datetime,
) -> Path:
    with xr.open_dataset(nc_path, engine="netcdf4", group="geolocation") as geo:
        lat = geo["latitude"].values.astype(np.float32)
        lon = geo["longitude"].values.astype(np.float32)

    with xr.open_dataset(nc_path, engine="netcdf4", group="product") as prod:
        with xr.open_dataset(nc_path, engine="netcdf4", group="quality_diagnostic_flags") as qf:
            ds = xr.Dataset(
                {
                    "/geolocation/latitude": (("y", "x"), lat),
                    "/geolocation/longitude": (("y", "x"), lon),
                    "/product/smoke": prod["smoke"],
                    "/product/dust": prod["dust"],
                    "/product/saai": prod["saai"],
                    "/product/uv_aai": prod["uv_aai"],
                    "/product/deepblue_aai": prod["deepblue_aai"],
                    "/quality_diagnostic_flags/qc_flag": qf["qc_flag"],
                    "/quality_diagnostic_flags/pqi2": qf["pqi2"],
                }
            )

            value_arr = build_product_array(
                ds,
                product=cfg["product"],
                smoke_conf=cfg["smoke_confidence"],
                dust_conf=cfg["dust_confidence"],
            )

    agg = "max" if "mask" in cfg["product"] else "mean"

    raster = rasterize_swath_points(
        lon=lon,
        lat=lat,
        values=value_arr,
        bbox=cfg["bbox"],
        resolution_deg=cfg["resolution_deg"],
        fill_value=cfg["fill_value"],
        agg=agg,
    )

    out_name = f"{cfg['product']}_{timestamp:%Y%m%dT%H%M%SZ}.tif"
    out_path = cfg["outdir"] / "rasters" / out_name
    return write_geotiff(raster, cfg["bbox"], cfg["resolution_deg"], out_path, cfg["dtype"])


def combine_rasters(paths: list[Path], out_path: Path, method: str):
    arrays = []
    meta = None

    for p in paths:
        with rasterio.open(p) as src:
            arrays.append(src.read(1).astype(np.float32))
            if meta is None:
                meta = src.meta.copy()

    stack = np.stack(arrays, axis=0)

    if method == "sum":
        combined = stack.sum(axis=0)
    elif method == "count":
        combined = (stack != 0).sum(axis=0).astype(np.float32)
    elif method == "mean":
        masked = np.where(stack == 0, np.nan, stack)
        combined = np.nanmean(masked, axis=0)
        combined = np.where(np.isfinite(combined), combined, 0).astype(np.float32)
    else:
        combined = stack.max(axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(combined.astype(np.float32), 1)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Pull TEMPO-ABI ADP data using a YAML config."
    )
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    args = parser.parse_args()

    cfg_raw = load_config(args.config)
    cfg = validate_config(cfg_raw)

    files = filter_files_in_time_range(cfg["start"], cfg["end"])
    if not files:
        print("No files found in requested time range.")
        sys.exit(0)

    raster_paths = []

    for ts, url in files:
        try:
            nc_name = Path(url).name
            local_nc = cfg["outdir"] / "raw_nc" / f"{ts:%Y%m%d}" / nc_name
            download_file(url, local_nc, overwrite=cfg["overwrite"])
            out_raster = process_one_file(local_nc, cfg, ts)
            raster_paths.append(out_raster)
            print(f"[ok] processed {ts} -> {out_raster}")

            if not cfg["keep_nc"] and local_nc.exists():
                local_nc.unlink()

        except Exception as e:
            print(f"[err] {ts} {url} -> {e}")

    print(f"\nProcessed {len(raster_paths)} granules.")

    if cfg["combine"] and raster_paths:
        combined = (
            cfg["outdir"]
            / "combined"
            / f"{cfg['product']}_{cfg['start']:%Y%m%dT%H%M%SZ}_{cfg['end']:%Y%m%dT%H%M%SZ}_{cfg['combine']}.tif"
        )
        combine_rasters(raster_paths, combined, cfg["combine"])
        print(f"[ok] combined raster -> {combined}")


if __name__ == "__main__":
    main()

