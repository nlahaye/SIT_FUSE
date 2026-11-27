"""GOES-18 ABI clear-sky reflectance composites (daily and hourly) for the HAB pilot.

Follows the coastal-ocean use of ABI in Zheng et al. (2023, RSE 295:113660): the visible/NIR
bands (C01 0.47 um, 1 km; C02 0.64 um, 0.5 km; C03 0.86 um, 1 km) are composited over many
daytime scans, which removes most cloud and much of the single-scan noise.

  for each CONUS/PACUS scan in [--hour-start, --hour-end] UTC (one per --minutes; 5 = every scan):
      ABI-L2-CMIPC C01-C03 reflectance factor -> lat/lon grid (average resampling)
      ABI-L2-ACMC clear-sky mask (BCM == 0, DQF == 0; max resampling, so any cloudy sub-pixel masks)
      C03 > --max-nir masked: residual cloud, glint and land
  hourly (--hourly-out): per UTC hour, per pixel and band, median of that hour's clear scans
                         (>= --min-clear-hour scans, else fill)
  daily  (--out):        median of all the day's clear scans (>= --min-clear)

Native resolution: --res 0.01 (~1 km) is the native sampling of C01/C03 (at 34N, 120W the ABI
"1 km" pixels are ~1.1 x 1.4 km) and averages C02 2 x 2; --res 0.005 keeps C02 at its native 0.5 km
(C01/C03 then oversampled). 0.01 deg is also the PACE OCI L2 pilot grid (same bbox), so the GOES and
PACE 1 km products share one grid.

Subset reads (default): the CMIPC / ACMC netCDF4 files are chunked (250 x 250, gzip), so only the
chunks covering the bbox are fetched with S3 range requests (~8-10 % of each file for roi "ca");
nothing is written to --raw. --no-subset downloads whole files to --raw (deleted after each day
unless --keep-raw).

Outputs (gtiff reader, chan_dim 0, fill -999):
  <out>/GOES18_ABI.YYYYMMDD.L3m.DAY.RHO.tif              daily, 3 bands C01, C02, C03 (model input)
  <out>/GOES18_ABI.YYYYMMDD.L3m.DAY.NCLEAR.tif           daily clear-scan count (diagnostic)
  <hourly-out>/HHH/GOES18_ABI.YYYYMMDD.L3m.DAY.RHO.tif   hourly, HHH = H16 ... H23 (start hour, UTC)
  <hourly-out>/HHH/GOES18_ABI.YYYYMMDD.L3m.DAY.NCLEAR.tif
  <hourly-out>/_done/YYYYMMDD.json                       hours written (resume marker)
Hourly files keep the daily name inside one directory per hour, so every date-keyed SIT-FUSE stage
(training lists, heads, UQ matchups) runs unchanged on one hour slot; training on <hourly-out> walks
all hour directories.
Training regex (instruments.GOES18_ABI.no_trop.input_oc_re): '(GOES18_ABI\\.(\\d{8})\\.L3m\\.DAY\\.RHO\\.tif)'

  python -m sit_fuse.pipelines.habs.followon.goes_ocean composite --start 2025-02-01 --end 2025-05-31 \\
      --out /mnt/data1/REFLECTANCES/GOES18_ABI_DAILY_1KM --hourly-out /mnt/data1/REFLECTANCES/GOES18_ABI_HOURLY_1KM \\
      --raw /mnt/data1/GOES18_RAW --res 0.01 --minutes 5 --threads 2 --scan-threads 4
  python -m sit_fuse.pipelines.habs.followon.goes_ocean geo --out /mnt/data1/goes18_geo_1km.zarr --res 0.01

Days already written are skipped (daily RHO present and, with --hourly-out, the _done marker).
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import logging
import os
import shutil
import threading
import warnings
from collections import OrderedDict
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


def hour_dir(h: int) -> str:
    return f"H{h:02d}"


def grid(bbox, res):
    from sit_fuse.preprocessing.data_pull.pull_goes_abi_fdcc import output_grid
    return output_grid(tuple(bbox), res)


def list_keys(s3, product: str, band: str | None, day: dt.date, h0: int, h1: int, minutes: int):
    """S3 keys of one product/band for the day's hours, thinned to one scan per `minutes` (slot -> key)."""
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


# ---------------------------------------------------------------------------
# Reading: S3 range reads of the bbox window (h5py) or whole-file download
# ---------------------------------------------------------------------------

class S3RangeFile(io.RawIOBase):
    """Read-only, seekable view of one S3 object; fetches aligned blocks on demand (LRU cache)."""

    def __init__(self, s3, bucket: str, key: str, block: int = 128 * 1024, max_blocks: int = 256):
        self.s3, self.bucket, self.key = s3, bucket, key
        self.block, self.max_blocks = block, max_blocks
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.pos, self.nbytes, self.nreq = 0, 0, 0
        self.size = None
        self._get_blocks(0, 0)                          # first block; also sets self.size

    def _get_blocks(self, b0: int, b1: int):
        lo = b0 * self.block
        hi = (b1 + 1) * self.block - 1
        if self.size is not None:
            hi = min(hi, self.size - 1)
        r = self.s3.get_object(Bucket=self.bucket, Key=self.key, Range=f"bytes={lo}-{hi}")
        data = r["Body"].read()
        if self.size is None:
            self.size = int(r["ContentRange"].split("/")[-1])
        self.nbytes += len(data)
        self.nreq += 1
        for i, b in enumerate(range(b0, b1 + 1)):
            self.cache[b] = data[i * self.block:(i + 1) * self.block]
            self.cache.move_to_end(b)
        while len(self.cache) > self.max_blocks:
            self.cache.popitem(last=False)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: off, io.SEEK_CUR: self.pos + off, io.SEEK_END: self.size + off}[whence]
        return self.pos

    def readinto(self, buf):
        n = min(len(buf), max(self.size - self.pos, 0))
        if n <= 0:
            return 0
        b0, b1 = self.pos // self.block, (self.pos + n - 1) // self.block
        b = b0
        while b <= b1:                                  # coalesce runs of missing blocks into one request
            if b in self.cache:
                self.cache.move_to_end(b)
                b += 1
                continue
            e = b
            while e + 1 <= b1 and (e + 1) not in self.cache:
                e += 1
            self._get_blocks(b, e)
            b = e + 1
        data = b"".join(self.cache[i] for i in range(b0, b1 + 1))
        s = self.pos - b0 * self.block
        buf[:n] = data[s:s + n]
        self.pos += n
        return n


def _attr(v, k, default=None):
    a = v.attrs.get(k, default)
    if isinstance(a, np.ndarray):
        a = a.reshape(-1)[0]
    if isinstance(a, bytes):
        a = a.decode()
    return a


def _decode_meta(v):
    """(numpy dtype as stored, fill, scale, offset); honours the netCDF _Unsigned convention."""
    dtype = v.dtype
    if str(_attr(v, "_Unsigned", "false")).lower() == "true" and dtype.kind == "i":
        dtype = np.dtype(dtype.str.replace("i", "u"))
    fv = _attr(v, "_FillValue")
    if fv is not None:
        fv = np.array(fv, dtype=v.dtype).view(dtype).item()
    sf, ao = _attr(v, "scale_factor"), _attr(v, "add_offset")
    return dtype, fv, (None if sf is None else float(sf)), float(ao or 0.0)


def _apply_meta(a, meta):
    dtype, fv, sf, ao = meta
    a = a.view(dtype) if a.dtype.itemsize == dtype.itemsize else a.astype(dtype)
    out = a.astype(np.float64)
    if fv is not None:
        out[a == fv] = np.nan
    if sf is not None:
        out = out * sf + ao
    return out


def _scaled(v, sl=None):
    return _apply_meta(v[sl] if sl is not None else v[()], _decode_meta(v))


_WIN_CACHE: dict = {}
_WIN_LOCK = threading.Lock()


def _bbox_window(h5, bbox, margin: int = 3):
    """(r0, r1, c0, c1, transform, crs) of the fixed-grid window covering bbox (cached per grid)."""
    from rasterio.transform import from_bounds
    from rasterio.warp import transform as warp_xy
    p = h5["goes_imager_projection"]
    hpp = float(_attr(p, "perspective_point_height"))
    crs = (f"+proj=geos +h={hpp} +lon_0={float(_attr(p, 'longitude_of_projection_origin'))} "
           f"+a={float(_attr(p, 'semi_major_axis'))} +b={float(_attr(p, 'semi_minor_axis'))} "
           f"+sweep={_attr(p, 'sweep_angle_axis', 'x')} +units=m +no_defs")
    x = _scaled(h5["x"]) * hpp
    y = _scaled(h5["y"]) * hpp
    key = (crs, len(x), len(y), round(x[0], 1), round(y[0], 1), tuple(bbox))
    with _WIN_LOCK:
        if key in _WIN_CACHE:
            return _WIN_CACHE[key]
    lon = np.linspace(bbox[0], bbox[2], 41)
    lat = np.linspace(bbox[1], bbox[3], 41)
    LON, LAT = np.meshgrid(lon, lat)
    px, py = warp_xy("EPSG:4326", crs, LON.ravel().tolist(), LAT.ravel().tolist())
    px, py = np.asarray(px), np.asarray(py)
    ok = np.isfinite(px) & np.isfinite(py)
    dx, dy = x[1] - x[0], y[1] - y[0]                   # dy < 0 (rows run north -> south)
    ci = (px[ok] - x[0]) / dx
    ri = (py[ok] - y[0]) / dy
    c0 = int(max(np.floor(ci.min()) - margin, 0))
    c1 = int(min(np.ceil(ci.max()) + margin + 1, len(x)))
    r0 = int(max(np.floor(ri.min()) - margin, 0))
    r1 = int(min(np.ceil(ri.max()) + margin + 1, len(y)))
    xs, ys = x[c0:c1], y[r0:r1]
    tr = from_bounds(xs.min() - abs(dx) / 2, ys.min() - abs(dy) / 2,
                     xs.max() + abs(dx) / 2, ys.max() + abs(dy) / 2, len(xs), len(ys))
    out = (r0, r1, c0, c1, tr, crs)
    with _WIN_LOCK:
        _WIN_CACHE[key] = out
    return out


_POOL = None
_POOL_LOCK = threading.Lock()


def _pool(n: int = 8) -> ThreadPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(n)
        return _POOL


def _chunk_plan(d, r0, r1, c0, c1):
    """[(row0, col0, byte_offset, size)] of the stored chunks intersecting the window."""
    cr, cc = d.chunks
    out = []
    for ri in range(r0 // cr, (r1 - 1) // cr + 1):
        for ci in range(c0 // cc, (c1 - 1) // cc + 1):
            info = d.id.get_chunk_info_by_coord((ri * cr, ci * cc))
            if info.byte_offset is None or info.size == 0:
                continue                                 # unallocated chunk: fill
            if info.filter_mask:
                raise ValueError("chunk with skipped filters")
            out.append((ri * cr, ci * cc, int(info.byte_offset), int(info.size)))
    return out


def _fetch_ranges(s3, key, ranges, gap: int = 16 * 1024):
    """{byte_offset: bytes} for (offset, size) ranges; near-adjacent ranges share one GET."""
    ranges = sorted(set(ranges))
    groups, cur = [], []
    for off, n in ranges:
        if cur and off - (cur[-1][0] + cur[-1][1]) > gap:
            groups.append(cur)
            cur = []
        cur.append((off, n))
    if cur:
        groups.append(cur)

    def get(g):
        lo, hi = g[0][0], max(o + n for o, n in g) - 1
        body = s3.get_object(Bucket=BUCKET, Key=key, Range=f"bytes={lo}-{hi}")["Body"].read()
        return {o: body[o - lo:o - lo + n] for o, n in g}, len(body)

    out, nb = {}, 0
    for part, n in _pool().map(get, groups):
        out.update(part)
        nb += n
    return out, nb


def _decode_chunk(buf: bytes, d) -> np.ndarray:
    import zlib
    raw = zlib.decompress(buf) if d.compression == "gzip" else buf
    it = d.dtype.itemsize
    if d.shuffle and it > 1:
        raw = np.frombuffer(raw, np.uint8).reshape(it, -1).T.tobytes()
    return np.frombuffer(raw, d.dtype).reshape(d.chunks)


def _assemble(d, plan, blobs, r0, r1, c0, c1):
    fill = d.fillvalue if d.fillvalue is not None else 0
    a = np.full((r1 - r0, c1 - c0), fill, dtype=d.dtype)
    cr, cc = d.chunks
    for rr, cl, off, _ in plan:
        ch = _decode_chunk(blobs[off], d)
        y0, y1 = max(rr, r0), min(rr + cr, r1)
        x0, x1 = max(cl, c0), min(cl + cc, c1)
        a[y0 - r0:y1 - r0, x0 - c0:x1 - c0] = ch[y0 - rr:y1 - rr, x0 - cl:x1 - cl]
    return a


def read_window(s3, key: str, var: str, bbox, raw: Path | None, subset: bool):
    """var (scaled, NaN where fill or DQF != 0) over the bbox window -> (array, transform, crs, bytes read).

    subset: the HDF5 metadata is read through S3RangeFile, then only the stored chunks that intersect
    the window are fetched (coalesced, parallel range GETs) and decoded here (gzip + shuffle), so the
    data transfer runs outside the h5py lock. Falls back to h5py slicing if a chunk layout is unusual."""
    import h5py
    names = [var] + ["DQF"]
    if not subset:
        src = _fetch(s3, key, raw)
        with h5py.File(src, "r") as h5:
            r0, r1, c0, c1, tr, crs = _bbox_window(h5, bbox)
            sl = (slice(r0, r1), slice(c0, c1))
            a = _scaled(h5[var], sl)
            if "DQF" in h5:
                a[h5["DQF"][sl] != 0] = np.nan
        return a.astype(np.float32), tr, crs, os.path.getsize(src)
    src = S3RangeFile(s3, BUCKET, key, block=64 * 1024)
    with h5py.File(src, "r") as h5:
        r0, r1, c0, c1, tr, crs = _bbox_window(h5, bbox)
        names = [n for n in names if n in h5]
        try:
            plans = {n: _chunk_plan(h5[n], r0, r1, c0, c1) for n in names}
            blobs, nb = _fetch_ranges(s3, key, [(o, s) for p in plans.values() for _, _, o, s in p])
            arrs = {n: _assemble(h5[n], plans[n], blobs, r0, r1, c0, c1) for n in names}
        except (ValueError, AttributeError, TypeError) as e:     # no chunk index / skipped filters
            LOG.debug("%s: chunk reads unavailable (%s); h5py slicing", key, e)
            sl = (slice(r0, r1), slice(c0, c1))
            arrs, nb = {n: h5[n][sl] for n in names}, 0
        a = _apply_meta(arrs[var], _decode_meta(h5[var]))
    if "DQF" in arrs:
        a[arrs["DQF"] != 0] = np.nan
    return a.astype(np.float32), tr, crs, src.nbytes + nb


def _fetch(s3, key: str, raw: Path) -> Path:
    p = raw / os.path.basename(key)
    if not p.exists():
        tmp = p.with_suffix(".part")
        s3.download_file(BUCKET, key, str(tmp))
        os.replace(tmp, p)
    return p


def _reproject(a, tr, crs, shape, transform, resampling) -> np.ndarray:
    from rasterio.warp import reproject
    dst = np.full(shape, np.nan, np.float32)
    reproject(source=a, destination=dst, src_transform=tr, src_crs=crs, src_nodata=np.nan,
              dst_transform=transform, dst_crs="EPSG:4326", dst_nodata=np.nan, resampling=resampling)
    return dst


def scan_fields(s3, acm_key, band_keys, a, shape, T, raw):
    """One scan -> (clear-masked {band: grid}, bytes read)."""
    from rasterio.enums import Resampling
    m, tr, crs, nb = read_window(s3, acm_key, "BCM", a.bbox, raw, a.subset)
    clear = _reproject(m, tr, crs, shape, T, Resampling.max) == 0
    fields = {}
    for b in BANDS:
        v, tr, crs, n = read_window(s3, band_keys[b], "CMI", a.bbox, raw, a.subset)
        nb += n
        fields[b] = _reproject(v, tr, crs, shape, T, Resampling.average)
    ok = clear & np.all([np.isfinite(fields[b]) for b in BANDS], axis=0)
    if a.max_nir is not None:
        ok &= fields["C03"] <= a.max_nir
    return {b: np.where(ok, fields[b], np.nan) for b in BANDS}, nb


def _median(stack: list[np.ndarray], min_clear: int):
    n = np.sum([np.isfinite(x) for x in stack], axis=0).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmedian(np.stack(stack), axis=0)
    return np.where(n >= min_clear, m, FILL).astype(np.float32), n


def _write_composite(out_dir: Path, day, per_band: dict, min_clear: int, T, tags: dict) -> bool:
    bands, n = [], None
    for b in BANDS:
        m, n = _median(per_band[b], min_clear)
        bands.append(m)
    if not np.any(n >= min_clear):
        return False
    write_tif(out_dir / out_name(day, "NCLEAR"), np.where(n > 0, n, FILL)[None], T,
              {**tags, "band_names": "n_clear"})
    write_tif(out_dir / out_name(day), np.stack(bands), T, tags)    # RHO last: its presence marks done
    return True


def _done_marker(a, day) -> Path | None:
    return Path(a.hourly_out) / "_done" / f"{day:%Y%m%d}.json" if a.hourly_out else None


def composite_day(day: dt.date, a, s3=None) -> str | None:
    daily = Path(a.out) / out_name(day) if a.out else None
    marker = _done_marker(a, day)
    if not a.overwrite and (daily is None or daily.exists()) and (marker is None or marker.exists()):
        return str(daily or marker)
    if s3 is None:
        from sit_fuse.preprocessing.data_pull.pull_goes_abi_fdcc import get_unsigned_s3_client
        s3 = get_unsigned_s3_client()
    W, H, T = grid(a.bbox, a.res)
    shape = (H, W)
    raw = Path(a.raw) / f"{day:%Y%m%d}"
    if not a.subset:
        raw.mkdir(parents=True, exist_ok=True)
    keys = {b: list_keys(s3, "ABI-L2-CMIPC", b, day, a.hour_start, a.hour_end, a.minutes) for b in BANDS}
    acm = list_keys(s3, "ABI-L2-ACMC", None, day, a.hour_start, a.hour_end, a.minutes)
    slots = sorted(set(acm).intersection(*[set(v) for v in keys.values()]))
    if not slots:
        LOG.warning("%s: no complete scans", day)
        return None

    def one(s):
        try:
            return s, *scan_fields(s3, acm[s], {b: keys[b][s] for b in BANDS}, a, shape, T, raw)
        except Exception as e:                          # one corrupt scan must not drop the day
            LOG.warning("%s slot %s: %s", day, s, e)
            return s, None, 0

    with ThreadPoolExecutor(max(1, a.scan_threads)) as ex:
        res = [r for r in ex.map(one, slots) if r[1] is not None]
    nbytes = sum(r[2] for r in res)
    if not res:
        return None
    base = {"source": "ABI-L2-CMIPC C01-C03 + ABI-L2-ACMC", "minutes": a.minutes, "max_nir": a.max_nir,
            "res_deg": a.res, "band_names": "C01,C02,C03"}
    written = []
    if a.hourly_out:
        for h in range(a.hour_start, a.hour_end + 1):
            hs = [r[1] for r in res if (r[0] * a.minutes) // 60 == h]
            if not hs:
                continue
            ok = _write_composite(Path(a.hourly_out) / hour_dir(h), day, {b: [f[b] for f in hs] for b in BANDS},
                                  a.min_clear_hour, T, {**base, "n_scans": len(hs), "hour_utc": h,
                                                        "min_clear": a.min_clear_hour})
            if ok:
                written.append(h)
    if a.out:
        _write_composite(Path(a.out), day, {b: [r[1][b] for r in res] for b in BANDS}, a.min_clear, T,
                         {**base, "n_scans": len(res), "hours_utc": f"{a.hour_start}-{a.hour_end}",
                          "min_clear": a.min_clear})
    if marker is not None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        with open(marker, "w") as f:
            json.dump({"hours": written, "n_scans": len(res), "bytes_read": nbytes}, f)
    if not a.subset and not a.keep_raw:
        shutil.rmtree(raw, ignore_errors=True)
    LOG.info("%s: %d scans, %.0f MB read, hours %s", day, len(res), nbytes / 1e6, written)
    return str(daily or marker)


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
    """(H, W, 2) lat/lon of pixel centres, same layout as hab_training_utils.gen_geo_zarr_oc_daily.
    --out ending in .zarr is the zarr path itself; otherwise <out>/goes18_geo.zarr."""
    import dask.array as da
    W, H, T = grid(a.bbox, a.res)
    lon = T.c + (np.arange(W) + 0.5) * T.a
    lat = T.f + (np.arange(H) + 0.5) * T.e
    LON, LAT = np.meshgrid(lon, lat)
    p = Path(a.out) if str(a.out).endswith(".zarr") else Path(a.out) / "goes18_geo.zarr"
    p.parent.mkdir(parents=True, exist_ok=True)
    da.to_zarr(da.from_array(np.stack([LAT, LON], -1).astype(np.float32)), str(p), overwrite=True)
    with open(p.with_suffix(".grid.json"), "w") as f:
        json.dump({"bbox": list(a.bbox), "res": a.res, "height": H, "width": W}, f)
    LOG.info("geo zarr %s (%d x %d)", p, H, W)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("composite", "geo"):
        q = sub.add_parser(name)
        q.add_argument("--out", required=(name == "geo"), default=None,
                       help="geo: zarr path; composite: daily composite directory (optional with --hourly-out)")
        q.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX)
        q.add_argument("--res", type=float, default=0.01,
                       help="degrees; 0.01 = native C01/C03 (C02 averaged 2x2), 0.005 = native C02")
        if name == "composite":
            q.add_argument("--hourly-out", default=None, help="hourly composites root (one H?? dir per hour)")
            q.add_argument("--start", required=True)
            q.add_argument("--end", required=True)
            q.add_argument("--raw", required=True, help="whole-file downloads (only with --no-subset)")
            q.add_argument("--hour-start", type=int, default=16, help="UTC; 16-23 = 08-15 PST / 09-16 PDT")
            q.add_argument("--hour-end", type=int, default=23)
            q.add_argument("--minutes", type=int, default=5, help="scan thinning; 5 = every CONUS scan")
            q.add_argument("--min-clear", type=int, default=2, help="daily: clear scans per pixel")
            q.add_argument("--min-clear-hour", type=int, default=3, help="hourly: clear scans per pixel")
            q.add_argument("--max-nir", type=float, default=0.12, help="C03 reflectance factor cap")
            q.add_argument("--threads", type=int, default=2, help="days in parallel")
            q.add_argument("--scan-threads", type=int, default=4, help="scans in parallel within a day")
            q.add_argument("--no-subset", dest="subset", action="store_false",
                           help="download whole files instead of range-reading the bbox chunks")
            q.add_argument("--keep-raw", action="store_true")
            q.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.cmd == "geo":
        return geo_zarr(a)
    if not (a.out or a.hourly_out):
        p.error("composite needs --out and/or --hourly-out")
    days = [d.date() for d in __import__("pandas").date_range(a.start, a.end, freq="D")]
    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(lambda d: composite_day(d, a), days))
    LOG.info("%d / %d days written", sum(r is not None for r in res), len(days))


if __name__ == "__main__":
    main()
