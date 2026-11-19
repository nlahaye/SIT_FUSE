#!/usr/bin/env python
"""
Copyright [2022-26], by the California Institute of Technology and Chapman University.
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.

Search and download TEMPO gridded NO2 (TEMPO_NO2_L3; ASDC / LARC_CLOUD) for UTC time ranges with the earthaccess
(Earthdata Search) API, and optionally cut each scan to a bounding box (a small NetCDF with only the variables the
smoke / cloud-mask stage needs) so the ~full-field-of-regard L3 files do not have to be kept.

    python pull_tempo_no2_l3.py --start 2024-07-24 --end 2024-08-03 --bbox -125 -114 32 42.5 \
        --out-dir /data/nlahaye/remoteSensing/NO2_L3 --dry-run
    python pull_tempo_no2_l3.py --start 2024-07-24 --end 2024-08-03 --bbox -125 -114 32 42.5 \
        --out-dir /data/nlahaye/remoteSensing/NO2_L3 --threads 8 --subset --delete-full

Versions: --version V04 V03 (default) searches V04 first and uses V03 only for scans that have no V04 granule
(V03 covers 2023-08 .. 2025-09-16; V04 is the current processing). The version is kept in the filename, and the
masking stage picks its default cloud threshold from it.

Download, retries, timeouts and Earthdata authentication are those of pull_tempo_rad_l1.py (Downloader,
download_all); granules already on disk (full file or _subset.nc) are not downloaded again.
L3 names: TEMPO_NO2_L3_<V0x>_<YYYYMMDDTHHMMSSZ>_S<scan>.nc (one file per E-W scan).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_tempo_rad_l1 import download_all, granule_name, parse_utc  # noqa: E402

LOG = logging.getLogger("pull-tempo-no2-l3")
NAME_RE = re.compile(r"TEMPO_NO2_L3_(V\d+)_(\d{8}T\d{6}Z)_S(\d{3})(?:_subset)?\.nc$")

# variables written to a _subset.nc (group/name); missing ones are skipped
SUBSET_VARS = [
    "product/vertical_column_troposphere", "product/vertical_column_troposphere_uncertainty",
    "product/vertical_column_stratosphere", "product/main_data_quality_flag",
    "support_data/eff_cloud_fraction", "support_data/amf_cloud_fraction", "support_data/amf_cloud_pressure",
    "support_data/amf_troposphere", "support_data/snow_ice_fraction", "support_data/vertical_column_total",
    "geolocation/solar_zenith_angle", "geolocation/viewing_zenith_angle", "weight",
]


def scan_key(name: str):
    m = NAME_RE.search(name)
    return (m.group(2), m.group(3)) if m else None


def name_time(name: str):
    m = NAME_RE.search(name)
    return dt.datetime.strptime(m.group(2), "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc) if m else None


def subset_path(full: Path) -> Path:
    return full.with_name(full.stem + "_subset.nc")


def search(start, end, bbox=None, versions=("V04", "V03"), short_name="TEMPO_NO2_L3"):
    """[(time, name, granule)] for scans starting in [start, end); first listed version wins per scan."""
    import earthaccess
    kw = {"short_name": short_name,
          "temporal": (start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")), "count": -1}
    if bbox:
        w, e, s, n = bbox
        kw["bounding_box"] = (w, s, e, n)
    best = {}
    for v in versions:
        try:
            res = earthaccess.search_data(version=v, **kw)
        except Exception as e:  # noqa: BLE001 - e.g. no granules for this version
            LOG.info("%s %s: %s", short_name, v, e)
            continue
        for g in res:
            name = granule_name(g)
            t, k = name_time(name), scan_key(name)
            if t is None or not (start <= t < end) or k in best:
                continue
            best[k] = (t, name, g)
        LOG.info("%s %s: %d scan(s) so far", short_name, v, len(best))
    return sorted(best.values(), key=lambda x: x[0])


def _get(ds, path):
    grp, _, var = path.rpartition("/")
    g = ds
    for p in filter(None, grp.split("/")):
        if p not in g.groups:
            return None
        g = g.groups[p]
    return g.variables.get(var)


def write_subset(full: Path, bbox, out: Path | None = None, variables=SUBSET_VARS) -> Path | None:
    """Copy the bbox window of `variables` (+ lat / lon / time and global attributes) to <stem>_subset.nc."""
    import netCDF4
    out = out or subset_path(full)
    w, e, s, n = bbox
    tmp = out.with_name(out.name + ".part")
    with netCDF4.Dataset(full) as src:
        lat, lon = src.variables["latitude"][:], src.variables["longitude"][:]
        iy = np.where((lat >= s) & (lat <= n))[0]
        ix = np.where((lon >= w) & (lon <= e))[0]
        if iy.size == 0 or ix.size == 0:
            LOG.warning("%s: no cells inside bbox %s", full.name, bbox)
            return None
        ys, xs = slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1)
        with netCDF4.Dataset(tmp, "w") as dst:
            dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
            dst.setncattr("sitfuse_subset_bbox", json.dumps(list(bbox)))
            dst.setncattr("sitfuse_subset_source", full.name)
            dst.createDimension("time", 1)
            dst.createDimension("latitude", iy.size)
            dst.createDimension("longitude", ix.size)
            for nm, sl, dim in (("latitude", ys, "latitude"), ("longitude", xs, "longitude")):
                v = src.variables[nm]
                o = dst.createVariable(nm, v.dtype, (dim,))
                o.setncatts({k: v.getncattr(k) for k in v.ncattrs() if k != "_FillValue"})
                o[:] = v[sl]
            if "time" in src.variables:
                v = src.variables["time"]
                o = dst.createVariable("time", v.dtype, ("time",))
                o.setncatts({k: v.getncattr(k) for k in v.ncattrs() if k != "_FillValue"})
                o[:] = v[:1]
            for path in variables:
                v = _get(src, path)
                if v is None:
                    continue
                grp, _, nm = path.rpartition("/")
                g = dst.createGroup(grp) if grp else dst
                fill = v.getncattr("_FillValue") if "_FillValue" in v.ncattrs() else None
                o = g.createVariable(nm, v.dtype, ("time", "latitude", "longitude"), zlib=True, complevel=4,
                                     fill_value=fill)
                o.setncatts({k: v.getncattr(k) for k in v.ncattrs() if k != "_FillValue"})
                v.set_auto_maskandscale(False)
                o.set_auto_maskandscale(False)
                a = v[...]
                o[:] = a[..., ys, xs] if a.ndim == 3 else a[np.newaxis, ys, xs]
    tmp.replace(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="UTC start (inclusive), e.g. 2024-07-24")
    ap.add_argument("--end", required=True, help="UTC end (exclusive)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "E", "S", "N"),
                    help="search filter and --subset window")
    ap.add_argument("--version", nargs="+", default=["V04", "V03"], help="versions in order of preference")
    ap.add_argument("--short-name", default="TEMPO_NO2_L3")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--access", choices=["https", "s3"], default="https")
    ap.add_argument("--timeout", type=float, nargs=2, default=[30, 600], metavar=("CONNECT", "READ"))
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--subset", action="store_true", help="write <stem>_subset.nc for --bbox")
    ap.add_argument("--delete-full", action="store_true", help="with --subset: remove the full file afterwards")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--persist", action="store_true", help="store the Earthdata login in ~/.netrc")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.subset and not a.bbox:
        sys.exit("--subset needs --bbox")
    run(parse_utc(a.start), parse_utc(a.end), a.out_dir, a.bbox, a.version, a.short_name, a.threads, a.access,
        tuple(a.timeout), a.retries, a.subset, a.delete_full, a.dry_run, a.persist)


def run(start, end, out_dir: Path, bbox=None, versions=("V04", "V03"), short_name="TEMPO_NO2_L3", threads=4,
        access="https", timeout=(30, 600), retries=4, subset=False, delete_full=False, dry_run=False, persist=False):
    """Returns the local file paths (subset if written, else full) for every scan found."""
    if end <= start:
        raise ValueError("end must be after start")
    found = search(start, end, bbox, versions, short_name)
    LOG.info("%s: %d scan(s) in [%s, %s)%s", short_name, len(found), start.isoformat(), end.isoformat(),
             f" intersecting {bbox}" if bbox else "")
    if dry_run:
        for _, name, _ in found:
            print(name)
        return []
    if not found:
        return []
    import earthaccess
    auth = earthaccess.login(persist=persist)
    if not getattr(auth, "authenticated", False):
        raise RuntimeError("Earthdata login failed (set up ~/.netrc or EARTHDATA_USERNAME / EARTHDATA_PASSWORD)")
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("TEMPO_NO2_L3_*.nc.part"):
        p.unlink(missing_ok=True)

    def have(name):
        f = out_dir / name
        return (f.exists() and f.stat().st_size > 0) or (subset and subset_path(f).exists())

    todo = [g for _, name, g in found if not have(name)]
    LOG.info("downloading %d scan(s) (%d already present) to %s", len(todo), len(found) - len(todo), out_dir)
    failed = download_all(todo, out_dir, access, threads, timeout, retries) if todo else []
    files = []
    for _, name, _ in found:
        full = out_dir / name
        sub = subset_path(full)
        if subset and not sub.exists() and full.exists():
            try:
                write_subset(full, bbox, sub)
            except Exception as e:  # noqa: BLE001
                LOG.error("%s: subset failed: %s", name, e)
        if subset and sub.exists():
            if delete_full:
                full.unlink(missing_ok=True)
            files.append(sub)
        elif full.exists():
            files.append(full)
    tag = f"{start:%Y%m%dT%H%M%SZ}_{end:%Y%m%dT%H%M%SZ}"
    (out_dir / f"pull_tempo_no2_l3_{tag}.json").write_text(json.dumps(
        {"short_name": short_name, "versions": list(versions), "start": start.isoformat(), "end": end.isoformat(),
         "bbox": bbox, "scans": [n for _, n, _ in found], "files": [str(f) for f in files],
         "download_errors": failed}, indent=1))
    if failed:
        LOG.warning("%d download error(s); rerun to retry", len(failed))
    return files


if __name__ == "__main__":
    main()
