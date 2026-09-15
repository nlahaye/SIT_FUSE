#!/usr/bin/env python3

from __future__ import annotations

import re
import argparse
import datetime as dt
import sys
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import yaml
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box


BASE_ROOT = "https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/"


DATASET_CONFIG = {
    "smoke": {
        "subdir": "Smoke_Polygons/Shapefile/",
        "filename": lambda d: f"hms_smoke{d:%Y%m%d}.zip",
    },
    "fire": {
        "subdir": "Fire_Points/Shapefile/",
        "filename": lambda d: f"hms_fire{d:%Y%m%d}.zip",
    },
}


SMOKE_DENSITY_MAP = {
    "light": 1,
    "Light": 1,
    "Medium": 2,
    "medium": 2,
    "moderate": 2,
    "Heavy": 3,
    "heavy": 3,
    "thick": 3,
}



def parse_datetime_utc(value: str) -> dt.datetime:
    """
    Parse a UTC date or timestamp from YAML.

    Accepted:
      YYYY-MM-DD
      YYYY-MM-DDTHH:MM:SSZ
      YYYY-MM-DDTHH:MM:SS
    """
    value = str(value).strip()

    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return dt.datetime.strptime(value, fmt)
        except ValueError:
            continue

    raise ValueError(
        f"Could not parse datetime '{value}'. "
        "Expected YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ."
    )


def parse_date(value: str) -> dt.date:
    return parse_datetime_utc(value).date()


def parse_hms_smoke_datetime(value, default_date: dt.date | None = None) -> pd.Timestamp:
    """
    Parse an HMS smoke Start/Stop field to UTC.

    Handles common ISO-like date/time strings and optionally accepts a
    time-only value when the archive date is supplied as default_date.
    """
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()
    parsed = pd.to_datetime(text, errors="coerce", utc=True)

    if not pd.isna(parsed):
        return parsed

    if default_date is not None:
        for fmt in ("%H:%M:%S", "%H:%M", "%H%M", "%H%M%S"):
            try:
                parsed_dt = dt.datetime.strptime(text, fmt)
                return pd.Timestamp(
                    dt.datetime.combine(default_date, parsed_dt.time()),
                    tz="UTC",
                )
            except ValueError:
                continue

    return pd.NaT


def find_column_case_insensitive(
    gdf: gpd.GeoDataFrame,
    candidates: list[str],
) -> str | None:
    lookup = {str(col).lower(): col for col in gdf.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def add_hms_smoke_intervals(
    gdf: gpd.GeoDataFrame,
    archive_day: dt.date,
) -> gpd.GeoDataFrame:
    """
    Add smoke_start_utc and smoke_stop_utc fields from HMS smoke attributes.
    """
    start_col = find_column_case_insensitive(
        gdf,
        ["Start", "START", "StartTime", "START_TIME", "Begin"],
    )
    stop_col = find_column_case_insensitive(
        gdf,
        ["Stop", "STOP", "StopTime", "STOP_TIME", "End"],
    )

    if start_col is None or stop_col is None:
        raise KeyError(
            "HMS smoke file is missing Start and/or Stop fields. "
            f"Available columns: {list(gdf.columns)}"
        )

    gdf = gdf.copy()

    gdf["smoke_start_utc"] = gdf[start_col].map(
        lambda x: parse_hms_smoke_datetime(x, default_date=archive_day)
    )
    gdf["smoke_stop_utc"] = gdf[stop_col].map(
        lambda x: parse_hms_smoke_datetime(x, default_date=archive_day)
    )

    invalid = (
        gdf["smoke_start_utc"].isna()
        | gdf["smoke_stop_utc"].isna()
        | (gdf["smoke_stop_utc"] < gdf["smoke_start_utc"])
    )

    if invalid.any():
        print(
            f"[warn] Dropping {int(invalid.sum())} smoke polygons with invalid "
            "or unparseable Start/Stop timestamps."
        )
        gdf = gdf.loc[~invalid].copy()

    return gdf


def select_smoke_time_bin(
    gdf: gpd.GeoDataFrame,
    bin_start: dt.datetime,
    bin_end: dt.datetime,
) -> gpd.GeoDataFrame:
    """
    Select polygons whose [start, stop] interval overlaps [bin_start, bin_end).

    This intentionally allows one smoke polygon to appear in multiple bins.
    """
    start_ts = pd.Timestamp(bin_start, tz="UTC")
    end_ts = pd.Timestamp(bin_end, tz="UTC")

    overlaps = (
        (gdf["smoke_start_utc"] < end_ts)
        & (gdf["smoke_stop_utc"] >= start_ts)
    )

    return gdf.loc[overlaps].copy()



def date_range(start: dt.date, end: dt.date):
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)


def build_url(dataset: str, day: dt.date) -> str:
    cfg = DATASET_CONFIG[dataset]
    return urljoin(
        BASE_ROOT,
        f"{cfg['subdir']}{day:%Y}/{day:%m}/{cfg['filename'](day)}",
    )


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("YAML config must parse to a dictionary/object.")

    return cfg


def validate_config(cfg: dict) -> dict:
    required_top = ["dataset", "time_range", "bbox", "raster", "output"]
    for key in required_top:
        if key not in cfg:
            raise ValueError(f"Missing required top-level config key: {key}")

    dataset = cfg["dataset"]
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"dataset must be one of {list(DATASET_CONFIG.keys())}")
 
    tr = cfg["time_range"]
    start_dt = parse_datetime_utc(tr["start"])
    end_dt = parse_datetime_utc(tr["end"])

    if end_dt < start_dt:
        raise ValueError("time_range.end must be >= time_range.start")

    start = start_dt.date()
    end = end_dt.date()

    temporal_resolution_minutes = cfg.get("temporal_resolution_minutes", None)
    if temporal_resolution_minutes is not None:
        temporal_resolution_minutes = int(temporal_resolution_minutes)
        if temporal_resolution_minutes <= 0:
            raise ValueError("temporal_resolution_minutes must be positive.")

 
    bb = cfg["bbox"]
    bbox = (
        float(bb["min_lon"]),
        float(bb["min_lat"]),
        float(bb["max_lon"]),
        float(bb["max_lat"]),
    )
    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise ValueError("bbox must satisfy min_lon < max_lon and min_lat < max_lat")

    raster = cfg["raster"]
    resolution_deg = float(raster.get("resolution_deg", 0.05))
    all_touched = bool(raster.get("all_touched", True))
    
    combine_method = cfg.get("combine", None)
    if combine_method not in (None, "max", "sum", "count"):
        raise ValueError("combine must be one of: null, max, sum, count")

    overwrite = bool(cfg.get("overwrite", False))
    outdir = Path(cfg["output"]["outdir"])

    return {
        "dataset": dataset,
        "start": start,
        "end": end,
        "bbox": bbox,
        "resolution_deg": resolution_deg,
        "all_touched": all_touched,
        "combine": combine_method,
        "overwrite": overwrite,
        "outdir": outdir,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "temporal_resolution_minutes": temporal_resolution_minutes,
    }


def download_file(url: str, dest: Path, overwrite: bool = False, timeout: int = 90) -> bool:
    if dest.exists() and not overwrite:
        print(f"[skip] {dest}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            if r.status_code == 404:
                print(f"[miss] {url}")
                return False
            r.raise_for_status()

            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp.replace(dest)

        print(f"[ok] downloaded {dest}")
        return True

    except requests.RequestException as e:
        print(f"[err] {url} -> {e}")
        return False


def unzip_archive(zip_path: Path, extract_dir: Path | None = None) -> Path:
    if extract_dir is None:
        extract_dir = zip_path.with_suffix("")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    return extract_dir


def find_shapefile(folder: Path) -> Path:
    shp_files = list(folder.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No shapefile found in {folder}")
    return shp_files[0]


def normalize_crs_to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Input shapefile has no CRS; cannot safely clip/rasterize.")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def clip_to_bbox(gdf: gpd.GeoDataFrame, bbox_vals: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    min_lon, min_lat, max_lon, max_lat = bbox_vals
    bbox_geom = box(min_lon, min_lat, max_lon, max_lat)
    bbox_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[bbox_geom], crs="EPSG:4326")
    clipped = gpd.clip(gdf, bbox_gdf)
    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notnull()].copy()
    return clipped


def infer_smoke_value_column(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cols = {c.lower(): c for c in gdf.columns}
    candidate_cols = ["density", "dens_cat", "smoke", "smoke_dens", "type"]

    found = None
    for c in candidate_cols:
        if c in cols:
            found = cols[c]
            break

    print(found, "FOUND")
    gdf = gdf.copy()
    if found is None:
        gdf["rast_val"] = 1
        return gdf

    def map_density(v):
        if pd.isna(v):
            return 1
        s = str(v).strip().lower()
        return SMOKE_DENSITY_MAP.get(s, 1)

    gdf["rast_val"] = gdf[found].map(map_density)
    print(gdf["rast_val"], "RAST VAL")
    return gdf


def infer_fire_value_column(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    cols = {c.lower(): c for c in gdf.columns}
    frp_candidates = ["frp", "power", "mw"]

    found = None
    for c in frp_candidates:
        if c in cols:
            found = cols[c]
            break

    if found is not None:
        vals = pd.to_numeric(gdf[found], errors="coerce").fillna(1)
        gdf["rast_val"] = np.maximum(vals, 1)
    else:
        gdf["rast_val"] = 1

    return gdf

def find_column_case_insensitive(gdf: gpd.GeoDataFrame, candidates: list[str]) -> str | None:
    """
    Return the first matching column name without depending on exact case.
    """
    lookup = {str(col).lower(): col for col in gdf.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def parse_hms_year_day(value) -> dt.date | None:
    """
    Parse HMS YearDay values.

    Expected common representations:
      2024167       -> 2024, Julian day 167
      2024-167      -> 2024, Julian day 167
      2024167.0     -> 2024, Julian day 167
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    digits = re.sub(r"[^0-9]", "", text)
    if len(digits) < 7:
        return None

    year = int(digits[:4])
    day_of_year = int(digits[4:])

    try:
        return dt.date(year, 1, 1) + dt.timedelta(days=day_of_year - 1)
    except ValueError:
        return None


def parse_hms_time(value) -> dt.time | None:
    """
    Parse HMS Time values.

    Supported examples:
      1530       -> 15:30:00
      153045     -> 15:30:45
      "15:30"    -> 15:30:00
      "15:30:45" -> 15:30:45
      930        -> 09:30:00
      "0930"     -> 09:30:00

    Invalid values return None.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return dt.datetime.strptime(text, fmt).time()
        except ValueError:
            pass

    digits = re.sub(r"[^0-9]", "", text)

    if len(digits) == 3:
        digits = digits.zfill(4)

    if len(digits) == 4:
        hour = int(digits[:2])
        minute = int(digits[2:4])
        second = 0
    elif len(digits) == 6:
        hour = int(digits[:2])
        minute = int(digits[2:4])
        second = int(digits[4:6])
    else:
        return None

    try:
        return dt.time(hour=hour, minute=minute, second=second)
    except ValueError:
        return None


def add_hms_fire_timestamp(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add fire_time_utc derived from HMS YearDay and Time fields.

    Rows with malformed or missing time metadata retain NaT.
    """
    year_day_col = find_column_case_insensitive(
        gdf,
        ["YearDay", "YEAR_DAY", "YEARDOY", "YEARDAY"],
    )
    time_col = find_column_case_insensitive(
        gdf,
        ["Time", "TIME", "HHMM", "UTC_TIME"],
    )

    if year_day_col is None or time_col is None:
        raise KeyError(
            "HMS fire file is missing required YearDay and/or Time fields. "
            f"Available columns: {list(gdf.columns)}"
        )

    gdf = gdf.copy()

    dates = gdf[year_day_col].map(parse_hms_year_day)
    times = gdf[time_col].map(parse_hms_time)

    gdf["fire_time_utc"] = [
        dt.datetime.combine(day, time)
        if day is not None and time is not None
        else pd.NaT
        for day, time in zip(dates, times)
    ]

    gdf["fire_time_utc"] = pd.to_datetime(
        gdf["fire_time_utc"],
        errors="coerce",
        utc=True,
    )

    n_valid = int(gdf["fire_time_utc"].notna().sum())
    n_missing = int(gdf["fire_time_utc"].isna().sum())

    print(
        f"[info] Parsed HMS fire times using "
        f"YearDay='{year_day_col}', Time='{time_col}': "
        f"{n_valid} valid, {n_missing} missing."
    )

    return gdf


def filter_fire_time_range(
    gdf: gpd.GeoDataFrame,
    start_utc: dt.datetime | None,
    end_utc: dt.datetime | None,
) -> gpd.GeoDataFrame:
    """
    Keep fire detections inside an optional closed UTC interval.
    """
    if start_utc is None and end_utc is None:
        return gdf

    if "fire_time_utc" not in gdf.columns:
        raise ValueError("fire_time_utc must be created before time filtering.")

    mask = gdf["fire_time_utc"].notna()

    if start_utc is not None:
        start_ts = pd.Timestamp(start_utc, tz="UTC")
        mask &= gdf["fire_time_utc"] >= start_ts

    if end_utc is not None:
        end_ts = pd.Timestamp(end_utc, tz="UTC")
        mask &= gdf["fire_time_utc"] <= end_ts

    return gdf.loc[mask].copy()


def iter_time_bins(
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    minutes: int,
):
    """
    Yield half-open UTC bins [bin_start, bin_end).

    The final bin is clipped to end_utc.
    """
    current = start_utc
    delta = dt.timedelta(minutes=minutes)

    while current < end_utc:
        next_time = min(current + delta, end_utc)
        yield current, next_time
        current = next_time


def select_fire_time_bin(
    gdf: gpd.GeoDataFrame,
    bin_start: dt.datetime,
    bin_end: dt.datetime,
) -> gpd.GeoDataFrame:
    """
    Select detections in [bin_start, bin_end).

    The final caller may optionally include the right endpoint if desired.
    """
    start_ts = pd.Timestamp(bin_start, tz="UTC")
    end_ts = pd.Timestamp(bin_end, tz="UTC")

    mask = (
        gdf["fire_time_utc"].notna()
        & (gdf["fire_time_utc"] >= start_ts)
        & (gdf["fire_time_utc"] < end_ts)
    )

    return gdf.loc[mask].copy()


def make_grid(bbox_vals: tuple[float, float, float, float], resolution_deg: float):
    min_lon, min_lat, max_lon, max_lat = bbox_vals
    width = int(np.ceil((max_lon - min_lon) / resolution_deg))
    height = int(np.ceil((max_lat - min_lat) / resolution_deg))
    transform = from_origin(min_lon, max_lat, resolution_deg, resolution_deg)
    return width, height, transform


def rasterize_gdf(
    gdf: gpd.GeoDataFrame,
    bbox_vals: tuple[float, float, float, float],
    resolution_deg: float,
    out_tif: Path,
    all_touched: bool = True,
    agg: str = "max",
    binry: bool = False,
) -> Path:
    width, height, transform = make_grid(bbox_vals, resolution_deg)
    

    if gdf.empty:
        arr = np.zeros((height, width), dtype=np.float32)

    else:
        shapes = [
            (geom, float(value))
            for geom, value in zip(gdf.geometry, gdf["rast_val"])
            if geom is not None and not geom.is_empty
        ]

        if agg == "sum":
            arr = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                fill=0,
                transform=transform,
                all_touched=all_touched,
                dtype="float32",
                merge_alg=MergeAlg.add,
            )
        else:
            # Smoke density: use maximum density of overlapping polygons.
            arr = np.zeros((height, width), dtype=np.float32)

            for geom, value in shapes:
                feature_arr = rasterize(
                    [(geom, value)],
                    out_shape=(height, width),
                    fill=0,
                    transform=transform,
                    all_touched=all_touched,
                    dtype="float32",
                )
                arr = np.maximum(arr, feature_arr)


    if binry: #TODO - setup to pass product type - set 1 for smoke and 0 for fire for binary thresh
        inds = np.where(arr > 0) 
        inds2 = np.where(arr <= 0)
        arr[inds] = 1
        arr[inds2] = 0

    with rasterio.open(
        out_tif,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
        crs="EPSG:4326",
        transform=transform,
        compress="lzw",
        nodata=-9999,
    ) as dst:
        dst.write(arr, 1)

    return out_tif


def combine_daily_rasters(raster_paths: list[Path], out_tif: Path, method: str = "max") -> Path:
    arrays = []
    meta = None

    for rp in raster_paths:
        with rasterio.open(rp) as src:
            arrays.append(src.read(1))
            if meta is None:
                meta = src.meta.copy()

    stack = np.stack(arrays, axis=0)

    if method == "sum":
        combined = stack.sum(axis=0)
    elif method == "count":
        combined = (stack > 0).sum(axis=0).astype(np.float32)
    else:
        combined = stack.max(axis=0)

    meta.update(dtype="float32", count=1, compress="lzw", nodata=-9999)
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(combined.astype(np.float32), 1)

    return out_tif


def process_one_day(
    dataset: str,
    day: dt.date,
    bbox_vals: tuple[float, float, float, float],
    resolution_deg: float,
    all_touched: bool,
    outdir: Path,
    start_dt: dt.datetime | None = None,
    end_dt: dt.datetime | None = None,
    temporal_resolution_minutes: int | None = None,
    overwrite: bool = False,
) -> list[dict]:
    """
    Download one daily HMS archive, clip features to bbox, and rasterize.

    Behavior:
      - Smoke with temporal_resolution_minutes:
          one raster per fixed bin; a smoke polygon appears in every bin
          overlapping its Start/Stop interval.

      - Fire with temporal_resolution_minutes:
          one raster per fixed bin; each fire point belongs to the bin
          containing its YearDay/Time timestamp.

      - Either dataset with temporal_resolution_minutes: null:
          one daily raster.
    """
    url = build_url(dataset, day)
    zip_name = url.split("/")[-1]

    raw_dir = outdir / "raw" / dataset / f"{day:%Y}" / f"{day:%m}"
    zip_path = raw_dir / zip_name

    if not download_file(url, zip_path, overwrite=overwrite):
        return []

    extract_dir = outdir / "extracted" / dataset / f"{day:%Y%m%d}"
    if not extract_dir.exists() or overwrite:
        unzip_archive(zip_path, extract_dir)

    shp_path = find_shapefile(extract_dir)

    gdf = gpd.read_file(shp_path)
    gdf = normalize_crs_to_wgs84(gdf)
    gdf = clip_to_bbox(gdf, bbox_vals)

    if gdf.empty:
        print(f"[warn] {day}: no {dataset} features intersect requested bbox.")
        return []

    binary = True
    if dataset == "smoke":
        gdf = infer_smoke_value_column(gdf)
        agg = "max"
        #binary = False

        if temporal_resolution_minutes is not None:
            gdf = add_hms_smoke_intervals(gdf, archive_day=day)

    elif dataset == "fire":
        gdf = add_hms_fire_timestamp(gdf)
        gdf = filter_fire_time_range(gdf, start_dt, end_dt)
        gdf = infer_fire_value_column(gdf)
        agg = "sum"

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if gdf.empty:
        print(f"[warn] {day}: no valid time-filtered {dataset} features remain.")
        return []

    results = []

    # --------------------------------------------------------------
    # Daily output: one raster for all clipped features in the archive.
    # --------------------------------------------------------------
    if temporal_resolution_minutes is None:
        vector_out = outdir / "clipped_vectors" / dataset / f"{day:%Y%m%d}_{dataset}.gpkg"
        raster_out = outdir / "rasters" / dataset / f"{day:%Y%m%d}_{dataset}.tif"

        print(vector_out.parent)

        vector_out.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(vector_out, driver="GPKG")

        if raster_out.exists() and not overwrite:
            print(f"[skip] {raster_out}")
        else:
            rasterize_gdf(
                gdf=gdf,
                bbox_vals=bbox_vals,
                resolution_deg=resolution_deg,
                out_tif=raster_out,
                all_touched=all_touched,
                agg=agg,
                binry=binary,
            )

        print(f"[ok] processed daily {dataset} raster -> {raster_out}")

        return [{
            "date": day,
            "start_time": None,
            "end_time": None,
            "vector": vector_out,
            "raster": raster_out,
            "n_features": len(gdf),
        }]

    # --------------------------------------------------------------
    # Time-resolved output: fixed bins over this day and requested range.
    # --------------------------------------------------------------
    day_start = dt.datetime.combine(day, dt.time.min)
    day_end = dt.datetime.combine(day + dt.timedelta(days=1), dt.time.min)

    if start_dt is not None:
        day_start = max(day_start, start_dt)

    if end_dt is not None:
        day_end = min(day_end, end_dt)

    if day_start >= day_end:
        return []

    for bin_start, bin_end in iter_time_bins(
        start_utc=day_start,
        end_utc=day_end,
        minutes=temporal_resolution_minutes,
    ):
        if dataset == "smoke":
            bin_gdf = select_smoke_time_bin(gdf, bin_start, bin_end)
        else:
            bin_gdf = select_fire_time_bin(gdf, bin_start, bin_end)

        start_token = bin_start.strftime("%Y%m%dT%H%M%SZ")
        end_token = bin_end.strftime("%Y%m%dT%H%M%SZ")

        stem = f"hms_{dataset}_{start_token}_{end_token}"

        vector_out = (
            outdir
            / "clipped_vectors"
            / dataset
            / f"{day:%Y%m%d}"
            / f"{stem}.gpkg"
        )

        raster_out = (
            outdir
            / "rasters"
            / dataset
            / f"{day:%Y%m%d}"
            / f"{stem}.tif"
        )

        raster_out.parent.mkdir(parents=True, exist_ok=True)
        vector_out.parent.mkdir(parents=True, exist_ok=True)
        bin_gdf.to_file(vector_out, driver="GPKG")

        if raster_out.exists() and not overwrite:
            print(f"[skip] {raster_out}")
        else:
            rasterize_gdf(
                gdf=bin_gdf,
                bbox_vals=bbox_vals,
                resolution_deg=resolution_deg,
                out_tif=raster_out,
                all_touched=all_touched,
                agg=agg,
                binry=binary,
            )

        print(
            f"[ok] {dataset} bin "
            f"{bin_start:%Y-%m-%dT%H:%M:%SZ} to "
            f"{bin_end:%Y-%m-%dT%H:%M:%SZ}: "
            f"{len(bin_gdf)} features -> {raster_out}"
        )

        results.append({
            "date": day,
            "start_time": bin_start,
            "end_time": bin_end,
            "vector": vector_out,
            "raster": raster_out,
            "n_features": len(bin_gdf),
        })

    return results



def main():
    parser = argparse.ArgumentParser(
        description="Download NOAA HMS data, clip to bbox, and rasterize using a YAML config."
    )
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    args = parser.parse_args()

    cfg_raw = load_config(args.config)
    cfg = validate_config(cfg_raw)

    results = []
    for day in date_range(cfg["start"], cfg["end"]):
        try:

           day_results = process_one_day(
               dataset=cfg["dataset"],
               day=day,
               bbox_vals=cfg["bbox"],
               resolution_deg=cfg["resolution_deg"],
               all_touched=cfg["all_touched"],
               outdir=cfg["outdir"],
               start_dt=cfg["start_dt"],
               end_dt=cfg["end_dt"],
               temporal_resolution_minutes=cfg["temporal_resolution_minutes"],
               overwrite=cfg["overwrite"],
           )

           results.extend(day_results)

        except Exception as e:
            print(f"[err] {day}: {e}")

    print(f"\nProcessed {len(results)} days successfully.")

    if cfg["combine"] and results:
        combined_out = (
            cfg["outdir"]
            / "combined"
            / (
                f"{cfg['dataset']}_"
                f"{cfg['start_dt']:%Y%m%dT%H%M%SZ}_"
                f"{cfg['end_dt']:%Y%m%dT%H%M%SZ}_"
                f"{cfg['combine']}.tif"
            )
        )
 
        all_rasters = [result["raster"] for result in results]

        combine_daily_rasters(
            all_rasters,
            combined_out,
            method=cfg["combine"],
        )

        print(f"[ok] combined raster -> {combined_out}")


if __name__ == "__main__":
    main()
