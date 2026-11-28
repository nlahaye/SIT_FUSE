"""PACE OCI L2 AOP granules -> daily 0.01 deg (~1 km) gridded Rrs in the L3m layout SIT-FUSE reads.

Pilot for native-resolution inputs: the standard L3m Rrs is 4 km; the L2 AOP swath is ~1.2 km
at nadir. Daily granules over the ROI are flag-masked and averaged into 0.01 deg cells
(mean of the valid L2 pixels whose centre falls in the cell; no gap filling), then written as

  <out>/PACE_OCI.YYYYMMDD.L3m.DAY.AOP.V3_2.1km.nc
      Rrs (lat, lon, wavelength) int16, scale_factor/add_offset/_FillValue/valid_min/valid_max as
          the L3m AOP product; lat (descending) / lon 1-D with valid_min/valid_max; wavelength;
      n_obs (lat, lon) int16 contributing L2 pixels

so sit_fuse.utils.read_pace_oc reads it with reader_kwargs {pace_product: AOP.V3_2, pace_res: 1km}.
The 1 km geo_zarr ((H, W, 2) = lat, lon; layout of gen_geo_zarr_oc_daily) is written by `geo`;
the training pipeline skips its own geo_zarr step when the file exists.

  python -m sit_fuse.pipelines.habs.followon.pace_l2 pull --start 2024-03-01 --end 2025-12-31 \
      --raw /mnt/data1/PACE_L2_RAW --threads 4
  python -m sit_fuse.pipelines.habs.followon.pace_l2 geo --out /mnt/data1/pace_l2g_1km_geo.zarr
  python -m sit_fuse.pipelines.habs.followon.pace_l2 grid --start 2024-03-01 --end 2025-12-31 \
      --raw /mnt/data1/PACE_L2_RAW --out /mnt/data1/REFLECTANCES/PACE_OCI_L2G --threads 3 [--delete-raw]

pull needs Earthdata credentials for earthaccess (~/.netrc or EARTHDATA_USERNAME/PASSWORD).
Both steps skip days that are already complete.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

SHORT_NAME = "PACE_OCI_L2_AOP"
DEFAULT_BBOX = (-128.0, 30.0, -116.0, 38.94)     # min_lon, min_lat, max_lon, max_lat (= roi "ca")
# standard L2 masks for Rrs (l2gen / L3 binning defaults for ocean colour)
DEFAULT_FLAGS = ("ATMFAIL", "LAND", "HIGLINT", "HILT", "HISATZEN", "STRAYLIGHT", "CLDICE",
                 "COCCOLITH", "HISOLZEN", "LOWLW", "CHLFAIL", "NAVWARN", "MAXAERITER", "CHLWARN",
                 "ATMWARN", "NAVFAIL")
SCALE, OFFSET, FILLV, VMIN, VMAX = 2e-6, 0.05, -32767, -30000, 32767


def out_name(day) -> str:
    return f"PACE_OCI.{pd.Timestamp(day):%Y%m%d}.L3m.DAY.AOP.V3_2.1km.nc"


# ----------------------------------------------------------------------------- pull
def pull_day(day, a) -> list[str]:
    import earthaccess
    d = pd.Timestamp(day)
    ddir = Path(a.raw) / f"{d:%Y%m%d}"
    done = ddir / ".complete"
    if done.exists():
        return sorted(str(p) for p in ddir.glob("*.nc"))
    ddir.mkdir(parents=True, exist_ok=True)
    for p in list(ddir.glob("partial_*")) + list(ddir.glob("*.part")):   # leftovers of failed attempts
        if p.is_dir():
            import shutil
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
    kw = {"short_name": SHORT_NAME, "temporal": (f"{d:%Y-%m-%d}T00:00:00", f"{d:%Y-%m-%d}T23:59:59"),
          "bounding_box": tuple(a.bbox)}
    if a.version:
        kw["version"] = a.version
    res = earthaccess.search_data(**kw)
    if a.nrt is False:
        res = [g for g in res if ".NRT." not in str(g.data_links())]
    files = [download_granule(g, ddir, a.access) for g in res]
    done.write_text(json.dumps({"n": len(files)}))
    return [str(f) for f in files]


_S3FS = {}


def _in_region() -> bool:
    import earthaccess
    st = getattr(earthaccess, "__store__", None)
    return bool(getattr(st, "in_region", False))


def download_granule(g, ddir: Path, access: str = "auto") -> Path:
    """One granule -> ddir/<name>, written to a sibling .part file and renamed (resume-safe).

    Replaces earthaccess.download: earthaccess < 1.0 copies in-region (direct S3) files with
    s3fs.get([url], <temp file>), which recent fsspec treats as a directory target and fails with
    FileExistsError on the temp file. Here direct S3 uses get_file (a file target) and HTTPS streams
    with the authenticated Earthdata session, so the result is independent of the earthaccess version.
    """
    import earthaccess
    direct = access == "direct" or (access == "auto" and _in_region())
    links = [u for u in g.data_links(access="direct" if direct else "external") if u.endswith(".nc")]
    if not links:
        raise RuntimeError(f"no {'s3' if direct else 'https'} .nc link for {g.data_links()}")
    url = links[0]
    dst = ddir / url.rsplit("/", 1)[-1]
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    tmp = dst.with_name(dst.name + ".part")
    tmp.unlink(missing_ok=True)
    if direct:
        if "fs" not in _S3FS:          # one filesystem per process; credentials refresh via earthaccess
            _S3FS["fs"] = earthaccess.get_s3_filesystem(results=[g])
        _S3FS["fs"].get_file(url, str(tmp))
    else:
        sess = earthaccess.get_requests_https_session()
        with sess.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 << 20):
                    f.write(chunk)
    os.replace(tmp, dst)
    LOG.info("downloaded %s (%.0f MB, %s)", dst.name, dst.stat().st_size / 1e6, "s3" if direct else "https")
    return dst


# ----------------------------------------------------------------------------- grid
def flag_bits(var, names) -> int:
    masks = np.atleast_1d(var.getncattr("flag_masks")).astype(np.int64)
    meanings = str(var.getncattr("flag_meanings")).split()
    bits = 0
    for m, n in zip(masks, meanings):
        if n in names:
            bits |= int(m)
    return bits


class Grid:
    def __init__(self, bbox, res):
        self.lon0, self.lat0, self.lon1, self.lat1 = bbox
        self.res = res
        self.W = int(round((self.lon1 - self.lon0) / res))
        self.H = int(round((self.lat1 - self.lat0) / res))
        self.lat = (self.lat1 - (np.arange(self.H) + 0.5) * res).astype(np.float32)   # north first, as L3m
        self.lon = (self.lon0 + (np.arange(self.W) + 0.5) * res).astype(np.float32)

    def index(self, lat, lon):
        r = np.floor((self.lat1 - lat) / self.res).astype(np.int64)
        c = np.floor((lon - self.lon0) / self.res).astype(np.int64)
        ok = (r >= 0) & (r < self.H) & (c >= 0) & (c < self.W)
        return r, c, ok


def accumulate(path: str, g: Grid, flags, S: np.ndarray | None, N: np.ndarray, block: int = 256):
    """Add one granule's valid Rrs to the running sums S (H*W, B) and counts N (H*W)."""
    from netCDF4 import Dataset
    with Dataset(path) as f:
        nav, geo = f["navigation_data"], f["geophysical_data"]
        lat_v, lon_v = nav["latitude"], nav["longitude"]
        rrs = geo["Rrs"]
        rrs.set_auto_maskandscale(True)
        fl = geo["l2_flags"]
        bits = flag_bits(fl, set(flags))
        wl = np.asarray(f["sensor_band_parameters"]["wavelength_3d"][:], np.float32)
        n_lines = lat_v.shape[0]
        for l0 in range(0, n_lines, block):
            l1 = min(n_lines, l0 + block)
            lat = np.asarray(lat_v[l0:l1], np.float64)
            lon = np.asarray(lon_v[l0:l1], np.float64)
            r, c, ok = g.index(lat, lon)
            if not ok.any():
                continue
            ok &= (np.asarray(fl[l0:l1]).astype(np.int64) & bits) == 0
            if not ok.any():
                continue
            R = np.ma.filled(rrs[l0:l1].astype(np.float32), np.nan)      # (lines, pixels, bands)
            v = R[ok]
            good = np.all(np.isfinite(v), axis=1)
            if not good.any():
                continue
            cell = (r[ok] * g.W + c[ok])[good]
            if S is None:
                S = np.zeros((g.H * g.W, v.shape[1]), np.float64)
            np.add.at(S, cell, v[good].astype(np.float64))
            np.add.at(N, cell, 1)
    return S, wl


def write_l3m_like(path: Path, g: Grid, mean: np.ndarray, n: np.ndarray, wl, tags):
    from netCDF4 import Dataset
    tmp = str(path) + ".tmp"
    with Dataset(tmp, "w") as f:
        f.createDimension("lat", g.H)
        f.createDimension("lon", g.W)
        f.createDimension("wavelength", mean.shape[-1])
        for name, arr, lo, hi in (("lat", g.lat, -90.0, 90.0), ("lon", g.lon, -180.0, 180.0)):
            v = f.createVariable(name, "f4", (name,))
            v[:] = arr
            v.valid_min, v.valid_max = np.float32(lo), np.float32(hi)
        v = f.createVariable("wavelength", "f4", ("wavelength",))
        v[:] = wl
        raw = np.where(np.isfinite(mean), np.round((mean - OFFSET) / SCALE), FILLV)
        raw = np.clip(raw, FILLV, VMAX).astype(np.int16)
        v = f.createVariable("Rrs", "i2", ("lat", "lon", "wavelength"), fill_value=np.int16(FILLV),
                             zlib=True, complevel=4, chunksizes=(min(256, g.H), min(256, g.W), mean.shape[-1]))
        v.set_auto_maskandscale(False)
        v.scale_factor, v.add_offset = np.float32(SCALE), np.float32(OFFSET)
        v.valid_min, v.valid_max = np.int16(VMIN), np.int16(VMAX)
        v.units = "sr^-1"
        v[:] = raw
        w = f.createVariable("n_obs", "i2", ("lat", "lon"), zlib=True)
        w[:] = np.minimum(n, 32767).astype(np.int16)
        for k, val in tags.items():
            setattr(f, k, str(val))
    os.replace(tmp, path)


def grid_day(day, a) -> str | None:
    out = Path(a.out) / out_name(day)
    if out.exists() and not a.overwrite:
        return str(out)
    ddir = Path(a.raw) / f"{pd.Timestamp(day):%Y%m%d}"
    files = sorted(str(p) for p in ddir.glob("*.nc")) if ddir.exists() else []
    if not files:
        return None
    g = Grid(a.bbox, a.res)
    N = np.zeros(g.H * g.W, np.int64)
    S, wl = None, None
    for p in files:
        try:
            S, wl = accumulate(p, g, a.flags.split(","), S, N)
        except Exception as e:
            LOG.warning("%s: %s", p, e)
    if S is None or N.sum() == 0:
        LOG.info("%s: no valid pixels in ROI", day)
        return None
    mean = np.where(N[:, None] >= a.min_obs, S / np.maximum(N[:, None], 1), np.nan).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_l3m_like(out, g, mean.reshape(g.H, g.W, -1), N.reshape(g.H, g.W), wl,
                   {"source": SHORT_NAME, "granules": ",".join(os.path.basename(p) for p in files),
                    "flags_masked": a.flags, "resolution_deg": a.res, "min_obs": a.min_obs,
                    "method": "mean of flag-valid L2 pixels per cell"})
    if a.delete_raw:
        for p in files:
            os.remove(p)
    LOG.info("%s: %d granules, %d cells -> %s", day, len(files), int((N >= a.min_obs).sum()), out)
    return str(out)


def write_geo_zarr(path: str, bbox=DEFAULT_BBOX, res: float = 0.01) -> str:
    import zarr
    g = Grid(bbox, res)
    loc = np.empty((g.H, g.W, 2), np.float32)
    loc[..., 0] = g.lat[:, None]
    loc[..., 1] = g.lon[None, :]
    z = zarr.open(path, mode="w", shape=loc.shape, chunks=(512, 512, 2), dtype="f4")
    z[:] = loc
    LOG.info("geo zarr %s (%d x %d)", path, g.H, g.W)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    gq = sub.add_parser("geo")
    gq.add_argument("--out", required=True)
    gq.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX)
    gq.add_argument("--res", type=float, default=0.01)
    for name in ("pull", "grid"):
        q = sub.add_parser(name)
        q.add_argument("--start", required=True)
        q.add_argument("--end", required=True)
        q.add_argument("--raw", required=True)
        q.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX)
        q.add_argument("--threads", type=int, default=3)
        if name == "pull":
            q.add_argument("--version", default=None, help="collection version (default: latest)")
            q.add_argument("--nrt", action="store_true", help="also accept NRT granules")
            q.add_argument("--access", choices=("auto", "direct", "https"), default="auto",
                           help="direct S3 (AWS us-west-2 only), HTTPS, or auto-detect")
        else:
            q.add_argument("--out", required=True)
            q.add_argument("--res", type=float, default=0.01)
            q.add_argument("--min-obs", type=int, default=1)
            q.add_argument("--flags", default=",".join(DEFAULT_FLAGS))
            q.add_argument("--delete-raw", action="store_true")
            q.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.cmd == "geo":
        write_geo_zarr(a.out, tuple(a.bbox), a.res)
        return
    days = pd.date_range(a.start, a.end, freq="D")
    if a.cmd == "pull":
        import earthaccess
        earthaccess.login(strategy="environment" if os.environ.get("EARTHDATA_USERNAME") else "netrc")
        fn = pull_day
    else:
        fn = grid_day
    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(lambda d: fn(d, a), days))
    LOG.info("%s: %d / %d days", a.cmd, sum(bool(r) for r in res), len(days))


if __name__ == "__main__":
    main()
