#!/usr/bin/env python3

from __future__ import annotations

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
    "medium": 2,
    "moderate": 2,
    "heavy": 3,
    "thick": 3,
}


def parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


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
) -> Path:
    width, height, transform = make_grid(bbox_vals, resolution_deg)

    if gdf.empty:
        arr = np.zeros((height, width), dtype=np.float32)
    else:
        shapes = ((geom, float(val)) for geom, val in zip(gdf.geometry, gdf["rast_val"]))

        merge_alg = MergeAlg.add if agg == "sum" else MergeAlg.replace

        arr = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            fill=0,
            transform=transform,
            all_touched=all_touched,
            dtype="float32",
            merge_alg=merge_alg,
        )

    out_tif.parent.mkdir(parents=True, exist_ok=True)
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
        nodata=0,
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

    meta.update(dtype="float32", count=1, compress="lzw", nodata=0)
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
    overwrite: bool = False,
):
    url = build_url(dataset, day)
    zip_name = url.split("/")[-1]

    raw_dir = outdir / "raw" / dataset / f"{day:%Y}" / f"{day:%m}"
    zip_path = raw_dir / zip_name

    if not download_file(url, zip_path, overwrite=overwrite):
        return None

    extract_dir = outdir / "extracted" / dataset / f"{day:%Y%m%d}"
    if not extract_dir.exists() or overwrite:
        unzip_archive(zip_path, extract_dir)

    shp_path = find_shapefile(extract_dir)
    gdf = gpd.read_file(shp_path)
    gdf = normalize_crs_to_wgs84(gdf)
    gdf = clip_to_bbox(gdf, bbox_vals)

    if dataset == "smoke":
        gdf = infer_smoke_value_column(gdf)
        agg = "max"
    else:
        gdf = infer_fire_value_column(gdf)
        agg = "sum"

    vector_out = outdir / "clipped_vectors" / dataset / f"{day:%Y%m%d}.gpkg"
    vector_out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(vector_out, driver="GPKG")

    raster_out = outdir / "rasters" / dataset / f"{day:%Y%m%d}.tif"
    rasterize_gdf(
        gdf=gdf,
        bbox_vals=bbox_vals,
        resolution_deg=resolution_deg,
        out_tif=raster_out,
        all_touched=all_touched,
        agg=agg,
    )

    print(f"[ok] processed {day} -> {raster_out}")
    return {"date": day, "vector": vector_out, "raster": raster_out, "n_features": len(gdf)}


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
            res = process_one_day(
                dataset=cfg["dataset"],
                day=day,
                bbox_vals=cfg["bbox"],
                resolution_deg=cfg["resolution_deg"],
                all_touched=cfg["all_touched"],
                outdir=cfg["outdir"],
                overwrite=cfg["overwrite"],
            )
            if res is not None:
                results.append(res)
        except Exception as e:
            print(f"[err] {day}: {e}")

    print(f"\nProcessed {len(results)} days successfully.")

    if cfg["combine"] and results:
        combined_out = (
            cfg["outdir"]
            / "combined"
            / f"{cfg['dataset']}_{cfg['start']:%Y%m%d}_{cfg['end']:%Y%m%d}_{cfg['combine']}.tif"
        )
        combine_daily_rasters([r["raster"] for r in results], combined_out, method=cfg["combine"])
        print(f"[ok] combined raster -> {combined_out}")


if __name__ == "__main__":
    main()
