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
    start = parse_datetime_utc(tr["start"])
    end = parse_datetime_utc(tr["end"])
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

    footprint_scale = float(
        raster_cfg.get("footprint_scale", 0.55)
    )

    if footprint_scale <= 0:
        raise ValueError("raster.footprint_scale must be positive.")

    # Grid cells with no ADP observation (outside the swath, night, cloud,
    # snow/ice, bad/missing QC) get fill_value, tagged as the GeoTIFF nodata.
    # It must differ from the labels (0 = no smoke, 1 = smoke); 255 by default.
    fill_value = float(raster_cfg.get("fill_value", 255.0))
    nodata = float(raster_cfg.get("nodata", fill_value))
    dtype = str(raster_cfg.get("dtype", "float32"))
    if fill_value in (0.0, 1.0):
        print(f"[warn] raster.fill_value={fill_value:g} equals a smoke label; unobserved cells "
              "will be indistinguishable from observed 'no smoke' / 'smoke'.")
    if np.dtype(dtype).kind == "u" and fill_value < 0:
        raise ValueError(f"raster.fill_value {fill_value:g} does not fit dtype {dtype}")

    product = str(cfg["product"]).lower()
    valid_products = {
        "smoke",
        #"dust_mask",
        #"saai_smoke",
        #"saai_dust",
        #"uv_aai",
        #"deepblue_aai",
    }
    if product not in valid_products:
        raise ValueError(f"product must be one of {sorted(valid_products)}")

    quality = cfg.get("quality", {})
    smoke_conf = quality.get("smoke_confidence", None)
    dust_conf = quality.get("dust_confidence", None)
    allowed_conf = {None, "high", "medium", "low", "high_medium", "all"}
    if smoke_conf not in allowed_conf or dust_conf not in allowed_conf:
        raise ValueError("quality confidence must be one of null, high, medium, low, high_medium, all")

    # Which ADP pixels count as unobserved (written as fill_value) rather
    # than observed "no smoke". Smoke = 1 is kept wherever detected, including
    # the above-cloud UV-AAI call-back, unless excluded by smoke_confidence.
    unobserved = quality.get("unobserved", ["fill", "qc_bad", "cloud", "snowice"])
    unobserved = [str(u).lower() for u in (unobserved or [])]
    bad = set(unobserved) - {"fill", "qc_bad", "cloud", "snowice"}
    if bad:
        raise ValueError(f"quality.unobserved entries must be fill, qc_bad, cloud, snowice; got {sorted(bad)}")
    # Smoke pixels outside smoke_confidence: 'unobserved' (fill) or 'negative' (0).
    below_conf = str(quality.get("below_confidence", "unobserved")).lower()
    if below_conf not in {"unobserved", "negative"}:
        raise ValueError("quality.below_confidence must be unobserved or negative")

    combine = cfg.get("combine", None)
    if combine not in {None, "max", "sum", "count", "mean"}:
        raise ValueError("combine must be one of null, max, sum, count, mean")

    return {
        "start": start,
        "end": end,
        "bbox": bbox,
        "resolution_deg": resolution_deg,
        "fill_value": fill_value,
        "nodata": nodata,
        "dtype": dtype,
        "unobserved": unobserved,
        "below_confidence": below_conf,
        "product": product,
        "smoke_confidence": smoke_conf,
        "dust_confidence": dust_conf,
        "combine": combine,
        "overwrite": bool(cfg.get("overwrite", False)),
        "outdir": Path(cfg["output"]["outdir"]),
        "keep_nc": bool(cfg["output"].get("keep_nc", True)),
        "footprint_scale": footprint_scale,
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
            if start.date() <= ts.date() <= end.date():
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


def smoke_tristate(ds: xr.Dataset, smoke_conf: str | None, unobserved=("fill", "qc_bad", "cloud", "snowice"),
                   below_confidence: str = "unobserved") -> np.ndarray:
    """ADP smoke as 1 (smoke), 0 (observed, no smoke), NaN (unobserved).

    NaN pixels are skipped by rasterize_swath_points, so grid cells covered
    only by unobserved pixels receive raster.fill_value."""
    smoke = ds["/product/smoke"].values.astype(np.float32)
    qc = ds["/quality_diagnostic_flags/qc_flag"].values
    qc_valid = np.isfinite(qc) if np.issubdtype(np.asarray(qc).dtype, np.floating) else np.ones(qc.shape, bool)
    qc_int = np.where(qc_valid, qc, 0).astype(np.int16)
    is_smoke = smoke == 1
    unobs = np.zeros(smoke.shape, dtype=bool)
    if "fill" in unobserved:
        # decoded _FillValue (NaN) or any value outside the 0/1 flag range
        unobs |= ~np.isin(smoke, (0.0, 1.0))
    if "qc_bad" in unobserved:
        # qc_flag bits 2-3 (smoke confidence) == 11: bad/missing (smoke or not)
        unobs |= ~qc_valid | (((qc_int >> 2) & 0b11) == 3)
    for name in ("cloud", "snowice"):
        key = f"/product/{name}"
        if name in unobserved and key in ds:
            flag = ds[key].values
            unobs |= (np.nan_to_num(flag.astype(np.float32), nan=0.0) == 1) & ~is_smoke
    # smoke detected above cloud (UV-AAI call-back) stays smoke: cloud/snowice
    # mark only non-smoke pixels unobserved
    arr = np.where(unobs, np.nan, np.where(is_smoke, 1.0, 0.0)).astype(np.float32)
    if smoke_conf not in (None, "all"):
        low = is_smoke & ~confidence_mask_from_qc(qc_int, "smoke", smoke_conf)
        arr[low] = np.nan if below_confidence == "unobserved" else 0.0
    n = arr.size
    print(f"ADP smoke: smoke={np.sum(arr == 1) / n:.4f} no_smoke={np.sum(arr == 0) / n:.4f} "
          f"unobserved={np.isnan(arr).sum() / n:.4f}")
    return arr


def build_product_array(ds: xr.Dataset, product: str, smoke_conf: str | None, dust_conf: str | None,
                        unobserved=("fill", "qc_bad", "cloud", "snowice"),
                        below_confidence: str = "unobserved") -> np.ndarray:
    lat = ds["/geolocation/latitude"].values
    shape = lat.shape

    if product == "smoke":
        return smoke_tristate(ds, smoke_conf, unobserved, below_confidence).reshape(shape)

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
    footprint_scale: float = 0.55,
) -> np.ndarray:
    """
    Rasterize a geolocated swath to a regular EPSG:4326 grid using approximate
    source-pixel footprints rather than source-pixel centers.

    Why this is needed:
      Center-only binning leaves regular holes when the source swath grid and
      requested output grid are not aligned. Each valid source observation is
      therefore expanded over a small local geographic footprint.

    Parameters
    ----------
    footprint_scale:
        Multiplier applied to the local estimated source-pixel spacing.
        0.50 approximates one full native pixel width. 0.55 adds a small
        overlap buffer that avoids cracks caused by floating-point rounding
        and slightly irregular geolocation coordinates.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    width, height, _ = make_grid(bbox, resolution_deg)

    out = np.full((height, width), np.nan, dtype=np.float32)
    count = np.zeros((height, width), dtype=np.int32)

    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    values = np.asarray(values, dtype=np.float32)

    if lon.shape != lat.shape or lon.shape != values.shape:
        raise ValueError(
            "lon, lat, and values must have the same two-dimensional shape. "
            f"Got lon={lon.shape}, lat={lat.shape}, values={values.shape}"
        )

    if lon.ndim != 2:
        raise ValueError(
            f"Expected two-dimensional swath arrays; received {lon.ndim} dimensions."
        )

    valid = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & np.isfinite(values)
        & (lon >= min_lon)
        & (lon <= max_lon)
        & (lat >= min_lat)
        & (lat <= max_lat)
    )

    if not np.any(valid):
        return np.full((height, width), fill_value, dtype=np.float32)

    # Estimate local geographic cell size from center-to-center source spacing.
    # np.gradient retains the source-array dimensions and supports curvilinear
    # latitude/longitude grids.
    dlon_dy, dlon_dx = np.gradient(lon)
    dlat_dy, dlat_dx = np.gradient(lat)

    # Approximate local half-width/half-height from the largest local spacing
    # in either grid direction. Invalid estimates fall back to output spacing.
    local_lon_spacing = np.maximum(np.abs(dlon_dx), np.abs(dlon_dy))
    local_lat_spacing = np.maximum(np.abs(dlat_dx), np.abs(dlat_dy))

    fallback_lon_spacing = resolution_deg
    fallback_lat_spacing = resolution_deg

    local_lon_spacing = np.where(
        np.isfinite(local_lon_spacing) & (local_lon_spacing > 0),
        local_lon_spacing,
        fallback_lon_spacing,
    )
    local_lat_spacing = np.where(
        np.isfinite(local_lat_spacing) & (local_lat_spacing > 0),
        local_lat_spacing,
        fallback_lat_spacing,
    )

    rows, cols = np.where(valid)

    for source_row, source_col in zip(rows, cols):
        center_lon = lon[source_row, source_col]
        center_lat = lat[source_row, source_col]
        value = values[source_row, source_col]

        half_lon = max(
            footprint_scale * local_lon_spacing[source_row, source_col],
            resolution_deg / 2.0,
        )
        half_lat = max(
            footprint_scale * local_lat_spacing[source_row, source_col],
            resolution_deg / 2.0,
        )

        west = center_lon - half_lon
        east = center_lon + half_lon
        south = center_lat - half_lat
        north = center_lat + half_lat

        col_start = max(
            0,
            int(np.floor((west - min_lon) / resolution_deg)),
        )
        col_stop = min(
            width - 1,
            int(np.floor((east - min_lon) / resolution_deg)),
        )

        row_start = max(
            0,
            int(np.floor((max_lat - north) / resolution_deg)),
        )
        row_stop = min(
            height - 1,
            int(np.floor((max_lat - south) / resolution_deg)),
        )

        if row_start > row_stop or col_start > col_stop:
            continue

        target = out[
            row_start : row_stop + 1,
            col_start : col_stop + 1,
        ]
        target_count = count[
            row_start : row_stop + 1,
            col_start : col_stop + 1,
        ]

        empty = np.isnan(target)

        if agg == "max":
            target[empty] = value
            target[~empty] = np.maximum(target[~empty], value)

        elif agg == "sum":
            target[empty] = value
            target[~empty] += value

        elif agg == "mean":
            target[empty] = value
            target[~empty] += value

        else:
            raise ValueError(
                f"Unsupported aggregation method: {agg!r}. "
                "Use max, sum, or mean."
            )

        target_count += 1

    if agg == "mean":
        populated = count > 0
        out[populated] = out[populated] / count[populated]

    return np.where(
        np.isfinite(out),
        out,
        fill_value,
    ).astype(np.float32)




def write_geotiff(arr: np.ndarray, bbox: tuple[float, float, float, float], resolution_deg: float, out_path: Path,
                  dtype: str, nodata: float | None = 255.0):
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
        nodata=nodata,
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

    print("HERE Opened Dataset")



    with xr.open_dataset(nc_path, engine="netcdf4", group="product") as prod:
        with xr.open_dataset(nc_path, engine="netcdf4", group="quality_diagnostic_flags") as qf:
            ds = xr.Dataset(
                {
                    "/geolocation/latitude": (("y", "x"), lat),
                    "/geolocation/longitude": (("y", "x"), lon),
                    "/product/smoke": prod["smoke"],
                    "/product/dust": prod["dust"],
                    **{f"/product/{v}": prod[v] for v in ("cloud", "snowice") if v in prod},
                    #"/product/saai": prod["saai"],
                    #"/product/uv_aai": prod["uv_aai"],
                    #"/product/deepblue_aai": prod["deepblue_aai"],
                    "/quality_diagnostic_flags/qc_flag": qf["qc_flag"],
                    #"/quality_diagnostic_flags/pqi2": qf["pqi2"],
                }
            )

            value_arr = build_product_array(
                ds,
                product=cfg["product"],
                smoke_conf=cfg["smoke_confidence"],
                dust_conf=cfg["dust_confidence"],
                unobserved=cfg["unobserved"],
                below_confidence=cfg["below_confidence"],
            )

    agg = "max" # if "mask" in cfg["product"] else "mean"

    print("HERE extraced datasets")


    print("PRIOR TO RASTERIZE", np.nanmin(value_arr), np.nanmax(value_arr))
    raster = rasterize_swath_points(
        lon=lon,
        lat=lat,
        values=value_arr,
        bbox=cfg["bbox"],
        resolution_deg=cfg["resolution_deg"],
        fill_value=cfg["fill_value"],
        agg=agg,
        footprint_scale=cfg["footprint_scale"],
    )

    print("HERE rasterized")

    out_name = f"{cfg['product']}_{timestamp:%Y%m%dT%H%M%SZ}.tif"
    out_path = cfg["outdir"] / "rasters" / out_name
    return write_geotiff(raster, cfg["bbox"], cfg["resolution_deg"], out_path, cfg["dtype"], cfg["nodata"])


def combine_rasters(paths: list[Path], out_path: Path, method: str, nodata: float = 255.0):
    """Binary smoke composite: 1 where any granule saw smoke, 0 where at
    least one granule observed no smoke and none saw smoke, nodata where no
    granule observed the cell."""
    arrays = []
    meta = None

    for p in paths:
        with rasterio.open(p) as src:
            a = src.read(1).astype(np.float32)
            nd = src.nodata if src.nodata is not None else nodata
            arrays.append(np.where(a == nd, np.nan, a))
            if meta is None:
                meta = src.meta.copy()

    stack = np.stack(arrays, axis=0)
    observed = np.isfinite(stack).any(axis=0)
    combined = np.where(observed, (np.nan_to_num(stack, nan=0.0) > 0).any(axis=0), 0).astype(np.float32)
    combined[~observed] = nodata
    meta.update(nodata=nodata)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(combined.astype(meta["dtype"]), 1)

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
        combine_rasters(raster_paths, combined, cfg["combine"], cfg["nodata"])
        print(f"[ok] combined raster -> {combined}")


if __name__ == "__main__":
    main()

