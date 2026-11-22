"""GOES-18 ABI daily clear-sky reflectance composites for the HAB pilot.

Follows the coastal-ocean use of ABI in Zheng et al. (2023, RSE 295:113660): visible/NIR
bands (C01 0.47, C02 0.64, C03 0.86 um) are composited over many daytime scans, which
removes most cloud and much of the noise of single scans. Here each day is reduced to one
SIT-FUSE input scene:

  for each scan in [--hour-start, --hour-end] UTC (every --minutes):
      ABI-L2-CMIPC C01-C03 reflectance factor -> lat/lon grid (average resampling)
      ABI-L2-ACMC clear-sky mask (BCM == 0; max resampling, so any cloudy sub-pixel masks)
      C03 > --max-nir masked: residual cloud, glint and land (vegetation / soil NIR reflectance is
      well above open and coastal water), so no separate land mask is needed for the ROI
  per pixel and band: median of the clear scans (>= --min-clear scans, else fill)

Output (gtiff reader, chan_dim 0, fill -999):
  <out>/GOES18_ABI.YYYYMMDD.L3m.DAY.RHO.tif   3 bands: C01, C02, C03 clear-sky median (model input)
  <out>/GOES18_ABI.YYYYMMDD.L3m.DAY.NCLEAR.tif 1 band: number of clear scans (diagnostic, not an input)
  <out>/goes18_geo.zarr                       (H, W, 2) = (lat, lon) pixel centres (geo subcommand),
                                              the geo_zarr of the GOES head outputs
Training regex (instruments.GOES18_ABI.no_trop.input_oc_re): '(GOES18_ABI\\.(\\d{8})\\.L3m\\.DAY\\.RHO\\.tif)'

  python -m sit_fuse.pipelines.habs.followon.goes_ocean composite --start 2024-01-01 --end 2025-12-31 \
      --out /mnt/data1/REFLECTANCES/GOES18_ABI --raw /mnt/data1/GOES18_RAW --threads 4
  python -m sit_fuse.pipelines.habs.followon.goes_ocean geo --out /mnt/data1/REFLECTANCES/GOES18_ABI

Raw files are deleted after each day unless --keep-raw. Days already written are skipped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)

BUCKET = "noaa-goes18"
BANDS = ("C01", "C02", "C03")
FILL = -999.0
DEFAULT_BBOX = (-128.0, 30.0, -116.0, 38.94)     # min_lon, min_lat, max_lon, max_lat (= roi "ca" of the OC models)


def out_name(day: dt.date, kind: str = "RHO") -> str:
    return f"GOES18_ABI.{day:%Y%m%d}.L3m.DAY.{kind}.tif"


def grid(bbox, res):
    from sit_fuse.preprocessing.data_pull.pull_goes_abi_fdcc import output_grid
    return output_grid(tuple(bbox), res)


def list_keys(s3, product: str, band: str | None, day: dt.date, h0: int, h1: int, minutes: int):
    """S3 keys of one product/band for the day's hours, thinned to one scan per `minutes`."""
    from sit_fuse.preprocessing.data_pull.pull_goes_abi_fdcc import parse_goes_start_time
    out = {}
    pag = s3.get_paginator("list_objects_v2")
    for h in range(h0, h1 + 1):
        t = dt.datetime(day.year, day.month, day.day, h)
        pre = f"{product}/{t:%Y}/{t:%j}/{t:%H}/"
        for page in pag.paginate(Bucket=BUCKET, Prefix=pre):
            for it in page.get("Contents", []):
                k = it["Key"]
                if not k.endswith(".nc") or (band and f"M6{band}_" not in k and f"M3{band}_" not in k):
                    continue
                st = parse_goes_start_time(os.path.basename(k))
                if st is None:
                    continue
                slot = (st.hour * 60 + st.minute) // minutes
                out.setdefault(slot, k)                 # first scan of each slot
    return out


def _fetch(s3, key: str, raw: Path) -> Path:
    p = raw / os.path.basename(key)
    if not p.exists():
        tmp = p.with_suffix(".part")
        s3.download_file(BUCKET, key, str(tmp))
        os.replace(tmp, p)
    return p


def _to_grid(path: Path, var: str, shape, transform, resampling, valid=None) -> np.ndarray:
    import rasterio  # noqa: F401
    import xarray as xr
    from rasterio.warp import reproject
    from sit_fuse.preprocessing.data_pull.pull_goes_abi_fdcc import geos_projection_crs, mask_source_transform
    with xr.open_dataset(path) as ds:
        a = ds[var].values.astype(np.float32)
        if valid is not None:
            a = np.where(valid(ds), a, np.nan)
        crs, tr = geos_projection_crs(ds), mask_source_transform(ds)
    dst = np.full(shape, np.nan, np.float32)
    reproject(source=a, destination=dst, src_transform=tr, src_crs=crs, src_nodata=np.nan,
              dst_transform=transform, dst_crs="EPSG:4326", dst_nodata=np.nan, resampling=resampling)
    return dst


def composite_day(day: dt.date, a, s3=None) -> str | None:
    from rasterio.enums import Resampling
    out = Path(a.out) / out_name(day)
    if out.exists() and not a.overwrite:
        return str(out)
    if s3 is None:
        from sit_fuse.preprocessing.data_pull.pull_goes_abi_fdcc import get_unsigned_s3_client
        s3 = get_unsigned_s3_client()
    W, H, T = grid(a.bbox, a.res)
    shape = (H, W)
    raw = Path(a.raw) / f"{day:%Y%m%d}"
    raw.mkdir(parents=True, exist_ok=True)
    keys = {b: list_keys(s3, "ABI-L2-CMIPC", b, day, a.hour_start, a.hour_end, a.minutes) for b in BANDS}
    acm = list_keys(s3, "ABI-L2-ACMC", None, day, a.hour_start, a.hour_end, a.minutes)
    slots = sorted(set(acm).intersection(*[set(v) for v in keys.values()]))
    if not slots:
        LOG.warning("%s: no complete scans", day)
        return None
    stack = {b: [] for b in BANDS}
    for s in slots:
        try:
            clear = _to_grid(_fetch(s3, acm[s], raw), "BCM", shape, T, Resampling.max,
                             valid=lambda ds: ds["DQF"].values == 0 if "DQF" in ds else True)
            clear = clear == 0
            fields = {b: _to_grid(_fetch(s3, keys[b][s], raw), "CMI", shape, T, Resampling.average,
                                  valid=lambda ds: (ds["DQF"].values == 0) if "DQF" in ds else True)
                      for b in BANDS}
        except Exception as e:                          # one corrupt scan must not drop the day
            LOG.warning("%s slot %s: %s", day, s, e)
            continue
        ok = clear & np.all([np.isfinite(fields[b]) for b in BANDS], axis=0)
        if a.max_nir is not None:
            ok &= fields["C03"] <= a.max_nir
        for b in BANDS:
            stack[b].append(np.where(ok, fields[b], np.nan))
    if not stack[BANDS[0]]:
        return None
    n = np.sum([np.isfinite(x) for x in stack["C01"]], axis=0).astype(np.float32)
    bands = []
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for b in BANDS:
            m = np.nanmedian(np.stack(stack[b]), axis=0)
            bands.append(np.where(n >= a.min_clear, m, FILL).astype(np.float32))
    tags = {"source": "ABI-L2-CMIPC C01-C03 + ABI-L2-ACMC", "n_scans": len(slots),
                                        "hours_utc": f"{a.hour_start}-{a.hour_end}", "minutes": a.minutes,
                                        "min_clear": a.min_clear, "max_nir": a.max_nir,
                                        "band_names": "C01,C02,C03"}
    write_tif(Path(a.out) / out_name(day, "NCLEAR"), np.where(n > 0, n, FILL)[None], T,
              {**tags, "band_names": "n_clear"})
    write_tif(out, np.stack(bands), T, tags)            # RHO last: its presence marks the day done
    if not a.keep_raw:
        shutil.rmtree(raw, ignore_errors=True)
    LOG.info("%s: %d scans -> %s", day, len(slots), out)
    return str(out)


def write_tif(path: Path, arr: np.ndarray, transform, tags: dict):
    import rasterio
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp.tif"
    prof = {"driver": "GTiff", "height": arr.shape[1], "width": arr.shape[2], "count": arr.shape[0],
            "dtype": "float32", "crs": "EPSG:4326", "transform": transform, "nodata": FILL,
            "compress": "deflate", "tiled": True}
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(arr)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
    os.replace(tmp, path)


def geo_zarr(a):
    """(H, W, 2) lat/lon of pixel centres, same layout as hab_training_utils.gen_geo_zarr_oc_daily."""
    import dask.array as da
    W, H, T = grid(a.bbox, a.res)
    lon = T.c + (np.arange(W) + 0.5) * T.a
    lat = T.f + (np.arange(H) + 0.5) * T.e
    LON, LAT = np.meshgrid(lon, lat)
    p = Path(a.out) / "goes18_geo.zarr"
    da.to_zarr(da.from_array(np.stack([LAT, LON], -1).astype(np.float32)), str(p), overwrite=True)
    with open(Path(a.out) / "goes18_grid.json", "w") as f:
        json.dump({"bbox": list(a.bbox), "res": a.res, "height": H, "width": W}, f)
    LOG.info("geo zarr %s (%d x %d)", p, H, W)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("composite", "geo"):
        q = sub.add_parser(name)
        q.add_argument("--out", required=True)
        q.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX)
        q.add_argument("--res", type=float, default=0.02, help="degrees (C01/C03 1 km, C02 0.5 km)")
        if name == "composite":
            q.add_argument("--start", required=True)
            q.add_argument("--end", required=True)
            q.add_argument("--raw", required=True)
            q.add_argument("--hour-start", type=int, default=16, help="UTC; 16-23 = 08-15 PST")
            q.add_argument("--hour-end", type=int, default=23)
            q.add_argument("--minutes", type=int, default=30)
            q.add_argument("--min-clear", type=int, default=2)
            q.add_argument("--max-nir", type=float, default=0.12, help="C03 reflectance factor cap")
            q.add_argument("--threads", type=int, default=4)
            q.add_argument("--keep-raw", action="store_true")
            q.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.cmd == "geo":
        return geo_zarr(a)
    days = [d.date() for d in __import__("pandas").date_range(a.start, a.end, freq="D")]
    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(lambda d: composite_day(d, a), days))
    LOG.info("%d / %d days written", sum(r is not None for r in res), len(days))


if __name__ == "__main__":
    main()
