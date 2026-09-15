#!/usr/bin/env python3
"""
Download and rasterize GOES ABI-L2-FDCC Fire/Hot Spot Characterization data.

The script:
  1. Lists ABI-L2-FDCC NetCDF files in the NOAA GOES public AWS S3 bucket.
  2. Downloads files whose start time falls in a requested UTC time range.
  3. Reads the ABI FDC `Mask` variable.
  4. Selects configured fire mask values.
  5. Writes a geographic GeoTIFF on an EPSG:4326 regular grid.

Default sources:
  GOES-16: noaa-goes16
  GOES-18: noaa-goes18
  GOES-19: noaa-goes19

Usage:
  python pull_goes_abi_fdcc_fire_mask.py config.yaml

Dependencies:
  pip install boto3 numpy rasterio xarray netcdf4 pyyaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import boto3
import numpy as np
import rasterio
import xarray as xr
import yaml
from botocore import UNSIGNED
from botocore.config import Config
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject


LOG = logging.getLogger("pull-goes-abi-fdcc-fire-mask")

GOES_BUCKETS = {
    "goes16": "noaa-goes16",
    "goes17": "noaa-goes17",
    "goes18": "noaa-goes18",
    "goes19": "noaa-goes19",
}

PRODUCT = "ABI-L2-FDCC"
FILENAME_TIME_RE = re.compile(
    r"_s(?P<start>\d{4}\d{3}\d{2}\d{2}\d{2}\d)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class GoesFile:
    """One remotely available GOES FDCC file."""

    bucket: str
    key: str
    start_time: dt.datetime

    @property
    def filename(self) -> str:
        return Path(self.key).name


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if not isinstance(cfg, dict):
        raise ValueError("YAML configuration must parse to a mapping.")

    return cfg


def parse_datetime(value: str) -> dt.datetime:
    """
    Parse UTC-naive YAML time.

    Supported:
      YYYY-MM-DD
      YYYY-MM-DDTHH:MM:SS
      YYYY-MM-DDTHH:MM:SSZ
    """
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
        f"Could not parse time {value!r}. "
        "Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ."
    )


def hour_range(
    start: dt.datetime,
    end: dt.datetime,
) -> Iterator[dt.datetime]:
    """Yield UTC hourly timestamps spanning the inclusive requested range."""
    current = start.replace(minute=0, second=0, microsecond=0)
    final = end.replace(minute=0, second=0, microsecond=0)

    while current <= final:
        yield current
        current += dt.timedelta(hours=1)


def parse_goes_start_time(filename: str) -> dt.datetime | None:
    """
    Parse the GOES-R scene start timestamp from an ABI Level 2 filename.

    GOES filenames include a token such as:
      _s20242100001000
    which is YYYY + day-of-year + HHMMSS + tenth-second.
    """
    match = FILENAME_TIME_RE.search(filename)
    if match is None:
        return None

    token = match.group("start")

    try:
        return dt.datetime.strptime(token[:13], "%Y%j%H%M%S")
    except ValueError:
        return None


def get_unsigned_s3_client():
    """Create an anonymous client for NOAA’s public GOES S3 buckets."""
    return boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED),
        region_name="us-east-1",
    )


def list_fdcc_files(
    s3_client,
    bucket: str,
    start: dt.datetime,
    end: dt.datetime,
) -> Iterator[GoesFile]:
    """
    List ABI-L2-FDCC NetCDF files in the selected temporal range.

    Public NOAA GOES S3 hierarchy:
      ABI-L2-FDCC/YYYY/DDD/HH/<filename>.nc
    """
    paginator = s3_client.get_paginator("list_objects_v2")

    for hour in hour_range(start, end):
        prefix = (
            f"{PRODUCT}/"
            f"{hour:%Y}/"
            f"{hour:%j}/"
            f"{hour:%H}/"
        )

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]

                if not key.lower().endswith(".nc"):
                    continue

                scene_time = parse_goes_start_time(Path(key).name)
                if scene_time is None:
                    LOG.warning(
                        "Skipping unparseable GOES filename: %s",
                        key,
                    )
                    continue

                if start <= scene_time <= end:
                    yield GoesFile(
                        bucket=bucket,
                        key=key,
                        start_time=scene_time,
                    )


def download_goes_file(
    s3_client,
    source: GoesFile,
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Download one NetCDF source file and return its local path."""
    local_dir = (
        output_dir
        / "raw"
        / source.bucket
        / f"{source.start_time:%Y}"
        / f"{source.start_time:%j}"
    )
    local_path = local_dir / source.filename

    if local_path.exists() and not overwrite:
        LOG.info("Using existing file: %s", local_path)
        return local_path

    local_dir.mkdir(parents=True, exist_ok=True)
    partial_path = local_path.with_suffix(local_path.suffix + ".part")

    LOG.info("Downloading s3://%s/%s", source.bucket, source.key)

    try:
        s3_client.download_file(
            source.bucket,
            source.key,
            str(partial_path),
        )
        partial_path.replace(local_path)
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise

    return local_path


def geos_projection_crs(dataset: xr.Dataset) -> str:
    """
    Build a PROJ CRS string from the GOES geostationary projection metadata.
    """
    projection = dataset["goes_imager_projection"]

    perspective_height = float(
        projection.attrs["perspective_point_height"]
    )
    longitude_origin = float(
        projection.attrs["longitude_of_projection_origin"]
    )
    semi_major = float(projection.attrs["semi_major_axis"])
    semi_minor = float(projection.attrs["semi_minor_axis"])
    sweep_axis = projection.attrs.get("sweep_angle_axis", "x")

    return (
        f"+proj=geos +h={perspective_height} "
        f"+lon_0={longitude_origin} "
        f"+a={semi_major} +b={semi_minor} "
        f"+sweep={sweep_axis} +units=m +no_defs"
    )


def mask_source_transform(dataset: xr.Dataset) -> rasterio.Affine:
    """
    Create an affine transform for the ABI fixed-grid Mask image.

    x and y coordinates in GOES L2 files are scan angles in radians. Multiplying
    by perspective height converts them to the coordinate units used by the
    geostationary projection CRS.
    """
    projection = dataset["goes_imager_projection"]
    height = float(projection.attrs["perspective_point_height"])

    x = dataset["x"].values.astype(np.float64) * height
    y = dataset["y"].values.astype(np.float64) * height

    if len(x) < 2 or len(y) < 2:
        raise ValueError("GOES source x/y coordinate arrays are too short.")

    x_resolution = float(np.median(np.diff(x)))
    y_resolution = float(np.median(np.diff(y)))

    west = float(x.min() - abs(x_resolution) / 2.0)
    east = float(x.max() + abs(x_resolution) / 2.0)
    south = float(y.min() - abs(y_resolution) / 2.0)
    north = float(y.max() + abs(y_resolution) / 2.0)

    return from_bounds(
        west=west,
        south=south,
        east=east,
        north=north,
        width=len(x),
        height=len(y),
    )


def output_grid(
    bbox: tuple[float, float, float, float],
    resolution_degrees: float,
) -> tuple[int, int, rasterio.Affine]:
    """Build an EPSG:4326 destination grid for the configured bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox

    width = int(np.ceil((max_lon - min_lon) / resolution_degrees))
    height = int(np.ceil((max_lat - min_lat) / resolution_degrees))

    if width <= 0 or height <= 0:
        raise ValueError("Output bbox/resolution produces an empty raster.")

    transform = from_bounds(
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        width,
        height,
    )

    return width, height, transform


def categorize_mask(
    mask: np.ndarray,
    fire_codes: set[int],
    output_mode: str,
    fill_value: int,
) -> np.ndarray:
    """
    Convert GOES FDC Mask codes to binary or selected categorical output.

    output_mode:
      binary:
        1 where Mask matches a configured fire code; 0 otherwise.

      categorical:
        retain configured fire codes; use 0 elsewhere.

      raw:
        preserve original Mask values, except source fill values are set to
        output fill_value by the calling function.
    """
    if output_mode == "raw":
        return mask.astype(np.int16, copy=False)

    selected = np.isin(mask, list(fire_codes))

    if output_mode == "binary":
        return selected.astype(np.int16)

    if output_mode == "categorical":
        out = np.zeros(mask.shape, dtype=np.int16)
        out[selected] = mask[selected].astype(np.int16)
        return out

    raise ValueError(
        "output.mode must be one of: binary, categorical, raw"
    )


def rasterize_fdcc_mask(
    netcdf_path: Path,
    output_path: Path,
    bbox: tuple[float, float, float, float],
    resolution_degrees: float,
    fire_codes: set[int],
    output_mode: str,
    output_nodata: int,
    overwrite: bool = False,
) -> Path:
    """
    Extract and reproject the GOES FDCC Mask to a geographic GeoTIFF.

    Nearest-neighbor resampling preserves the discrete Fire Mask codes.
    """
    if output_path.exists() and not overwrite:
        LOG.info("Using existing raster: %s", output_path)
        return output_path

    with xr.open_dataset(netcdf_path) as dataset:
        if "Mask" not in dataset:
            raise KeyError(
                f"FDCC NetCDF does not contain variable 'Mask': {netcdf_path}"
            )

        source_mask = dataset["Mask"].values
        source_fill = dataset["Mask"].attrs.get(
            "_FillValue",
            dataset["Mask"].attrs.get("missing_value"),
        )

        if source_fill is not None:
            source_mask = np.where(
                source_mask == source_fill,
                output_nodata,
                source_mask,
            )

        source_mask = source_mask.astype(np.int16, copy=False)

        processed_mask = categorize_mask(
            mask=source_mask,
            fire_codes=fire_codes,
            output_mode=output_mode,
            fill_value=output_nodata,
        )

        if source_fill is not None and output_mode != "raw":
            processed_mask[source_mask == output_nodata] = output_nodata

        source_crs = geos_projection_crs(dataset)
        source_transform = mask_source_transform(dataset)

    width, height, destination_transform = output_grid(
        bbox=bbox,
        resolution_degrees=resolution_degrees,
    )

    destination = np.full(
        (height, width),
        output_nodata,
        dtype=np.int16,
    )

    reproject(
        source=processed_mask,
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=output_nodata,
        dst_transform=destination_transform,
        dst_crs="EPSG:4326",
        dst_nodata=output_nodata,
        resampling=Resampling.nearest,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "int16",
        "crs": "EPSG:4326",
        "transform": destination_transform,
        "nodata": output_nodata,
        "compress": "lzw",
        "tiled": True,
    }

    with rasterio.open(output_path, "w", **profile) as destination_file:
        destination_file.write(destination, 1)
        destination_file.update_tags(
            source_product=PRODUCT,
            source_file=netcdf_path.name,
            output_mode=output_mode,
            selected_fire_codes=",".join(
                str(code) for code in sorted(fire_codes)
            ),
        )

    return output_path


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize YAML configuration."""
    required = ["goes", "time_range", "bbox", "output"]

    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing required top-level configuration key: {key}")

    satellite = str(cfg["goes"].get("satellite", "goes19")).lower()
    if satellite not in GOES_BUCKETS:
        raise ValueError(
            f"goes.satellite must be one of {sorted(GOES_BUCKETS)}"
        )

    start = parse_datetime(cfg["time_range"]["start"])
    end = parse_datetime(cfg["time_range"]["end"])

    if end < start:
        raise ValueError("time_range.end must be on or after start.")

    bbox_cfg = cfg["bbox"]
    bbox = (
        float(bbox_cfg["min_lon"]),
        float(bbox_cfg["min_lat"]),
        float(bbox_cfg["max_lon"]),
        float(bbox_cfg["max_lat"]),
    )

    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise ValueError(
            "bbox must satisfy min_lon < max_lon and min_lat < max_lat."
        )

    raster_cfg = cfg.get("raster", {})
    resolution_degrees = float(
        raster_cfg.get("resolution_degrees", 0.02)
    )

    if resolution_degrees <= 0:
        raise ValueError("raster.resolution_degrees must be positive.")

    output_cfg = cfg["output"]
    output_dir = Path(output_cfg["directory"])

    mask_cfg = cfg.get("mask", {})
    output_mode = str(mask_cfg.get("mode", "binary")).lower()

    if output_mode not in {"binary", "categorical", "raw"}:
        raise ValueError(
            "mask.mode must be one of: binary, categorical, raw"
        )

    fire_codes = {
        int(code)
        for code in mask_cfg.get(
            "fire_codes",
            [10, 11, 12, 13, 14, 15, 30, 31, 32, 33, 34, 35],
        )
    }

    output_nodata = int(mask_cfg.get("output_nodata", -1))

    return {
        "satellite": satellite,
        "bucket": GOES_BUCKETS[satellite],
        "start": start,
        "end": end,
        "bbox": bbox,
        "resolution_degrees": resolution_degrees,
        "fire_codes": fire_codes,
        "output_mode": output_mode,
        "output_nodata": output_nodata,
        "output_dir": output_dir,
        "overwrite": bool(cfg.get("overwrite", False)),
    }


def main(config_path: Path) -> None:
    """Download FDCC files and rasterize their Mask variable."""
    config = validate_config(load_yaml(config_path))
    s3_client = get_unsigned_s3_client()

    output_dir: Path = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(
        list_fdcc_files(
            s3_client=s3_client,
            bucket=config["bucket"],
            start=config["start"],
            end=config["end"],
        )
    )

    if not files:
        LOG.warning(
            "No %s files found in %s from %s through %s.",
            PRODUCT,
            config["bucket"],
            config["start"].isoformat(),
            config["end"].isoformat(),
        )
        return

    LOG.info(
        "Found %d %s files for %s.",
        len(files),
        PRODUCT,
        config["satellite"],
    )

    processed = 0

    for source in files:
        try:
            local_netcdf = download_goes_file(
                s3_client=s3_client,
                source=source,
                output_dir=output_dir,
                overwrite=config["overwrite"],
            )

            output_path = (
                output_dir
                / "rasters"
                / config["satellite"]
                / f"{source.start_time:%Y}"
                / f"{source.start_time:%m}"
                / (
                    f"fdcc_fire_mask_"
                    f"{source.start_time:%Y%m%dT%H%M%SZ}.tif"
                )
            )

            rasterize_fdcc_mask(
                netcdf_path=local_netcdf,
                output_path=output_path,
                bbox=config["bbox"],
                resolution_degrees=config["resolution_degrees"],
                fire_codes=config["fire_codes"],
                output_mode=config["output_mode"],
                output_nodata=config["output_nodata"],
                overwrite=config["overwrite"],
            )

            LOG.info("Wrote %s", output_path)
            processed += 1

        except Exception as exc:
            LOG.exception(
                "Failed to process s3://%s/%s: %s",
                source.bucket,
                source.key,
                exc,
            )

    LOG.info("Completed %d of %d FDCC files.", processed, len(files))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Download GOES ABI-L2-FDCC files and rasterize Fire Mask data."
        )
    )
    parser.add_argument(
        "yaml_config",
        type=Path,
        help="Path to YAML configuration file.",
    )

    args = parser.parse_args()

    try:
        main(args.yaml_config)
    except Exception as exc:
        LOG.exception("GOES FDCC processing failed: %s", exc)
        sys.exit(1)



