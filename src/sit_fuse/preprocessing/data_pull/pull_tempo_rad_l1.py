#!/usr/bin/env python
"""
Copyright [2022-26], by the California Institute of Technology and Chapman University.
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.

Search and download TEMPO L1 geolocated Earth radiances (TEMPO_RAD_L1, default version V04; ASDC / LARC_CLOUD)
for a UTC time range with the earthaccess (Earthdata Search) API, optionally limited to granules that intersect a
bounding box, and optionally convert the new granules to GeoTIFF with preprocessing/colocate_and_resample.py,
using config/preprocess/colocate_and_resample/tempo_clip_v04.yaml as the template (grid, resolution, readers and
resampling of the existing RAD_L1_V04 archive; only filenames / geo_filenames / output_files are replaced).

    python pull_tempo_rad_l1.py --start 2026-07-26 --end 2026-07-27T06:00:00Z \
        --out-dir /data/nlahaye/remoteSensing/RAD_L1_V04 --dry-run          # list granules (bbox: template grid)
    python pull_tempo_rad_l1.py --start 2026-07-26 --end 2026-07-27T06:00:00Z \
        --out-dir /data/nlahaye/remoteSensing/RAD_L1_V04 --threads 8 --convert
    # write the colocate_and_resample config only, run it yourself:
    python pull_tempo_rad_l1.py ... --convert --config-only
    python ../colocate_and_resample.py -y <out-dir>/tempo_clip_v04_<start>_<end>.yaml
    # + RGB / UV_RGB backgrounds (misc_scripts/tempo_rgb.py, tempo_rgb_uv.py bands) for every GeoTIFF of the range:
    python pull_tempo_rad_l1.py ... --convert --rgb
    # RGBs only, from GeoTIFFs already in --out-dir (no search, no login):
    python pull_tempo_rad_l1.py --start 2026-07-23 --end 2026-08-01T06:00:00Z --out-dir ... --rgb-only [--rgb uv]

Authentication (earthaccess.login): ~/.netrc entry for urs.earthdata.nasa.gov, or EARTHDATA_USERNAME /
EARTHDATA_PASSWORD (or EARTHDATA_TOKEN), else an interactive prompt (--persist writes ~/.netrc).
Downloads use HTTPS with an Earthdata-authenticated session (timeouts, retries, per-file progress); --access s3
uses direct S3 and is only valid on AWS us-west-2. earthaccess.download() is not used: when earthaccess detects
us-west-2 it silently switches to S3, and its HTTPS path has no timeouts or progress output.
Granules already on disk (same name, non-zero size) are not downloaded again; with --convert, granules whose
GeoTIFF already exists (next to the .nc, as in the archive) are not converted again. Without --bbox the search uses
the template's lon_bounds / lat_bounds (granules that do not touch the GeoTIFF grid are skipped).
Granule names: TEMPO_RAD_L1_V04_<YYYYMMDDTHHMMSSZ>_S<scan>G<granule>.nc; one E-W scan is split into ~9 granules,
so a bbox keeps only the granules over the region (e.g. the western G07-G09 for the Pacific Northwest).
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LOG = logging.getLogger("pull-tempo-rad-l1")
NAME_RE = re.compile(r"TEMPO_RAD_L1_(V\d+)_(\d{8}T\d{6}Z)_S(\d{3})G(\d{2})\.nc$")


def parse_utc(s: str) -> dt.datetime:
    s = s.strip().replace("Z", "+00:00")
    t = dt.datetime.fromisoformat(s) if "T" in s or ":" in s else dt.datetime.fromisoformat(s + "T00:00:00+00:00")
    return (t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)).astimezone(dt.timezone.utc)


def granule_name(g) -> str:
    try:
        return g["umm"]["GranuleUR"]
    except Exception:  # noqa: BLE001
        return Path(g.data_links()[0]).name


def granule_time(name: str):
    m = NAME_RE.search(name)
    return dt.datetime.strptime(m.group(2), "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc) if m else None


def search(short_name, version, start, end, bbox=None, scans=None):
    """earthaccess granule search; bbox [W, E, S, N]; scans: optional set of scan numbers (e.g. {7, 8})."""
    import earthaccess
    kw = {"short_name": short_name, "version": version,
          "temporal": (start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")), "count": -1}
    if bbox:
        w, e, s, n = bbox
        kw["bounding_box"] = (w, s, e, n)            # earthaccess order: (lower_left_lon, lower_left_lat, ur_lon, ur_lat)
    res = earthaccess.search_data(**kw)
    out = []
    for g in res:
        name = granule_name(g)
        t = granule_time(name)
        if t is None or not (start <= t < end):      # CMR temporal is an overlap test; keep granules starting in range
            continue
        m = NAME_RE.search(name)
        if scans and int(m.group(3)) not in scans:
            continue
        out.append((t, name, g))
    return sorted(out, key=lambda x: x[0])


def granule_links(g, access="external"):
    """.nc data links of a granule (external = https://data.asdc..., direct = s3://asdc-prod-protected/...)."""
    links = [u for u in g.data_links(access=access) if u.split("?")[0].endswith(".nc")]
    return links or list(g.data_links(access=access))


class Downloader:
    """Per-file download with timeouts, retries, a .part temp file and progress logging.

    access="https": Earthdata-authenticated requests session (one per thread), works anywhere.
    access="s3":    direct S3 (only from AWS us-west-2) with the ASDC s3credentials endpoint; the temporary
                    credentials expire after ~1 h, so they are refreshed every 45 min.
    """

    def __init__(self, access, out_dir: Path, timeout=(30, 300), retries=4, endpoint=None):
        self.access, self.out_dir, self.timeout, self.retries, self.endpoint = access, out_dir, timeout, retries, endpoint
        self._tl = threading.local()
        self._s3, self._s3_t, self._lock = None, 0.0, threading.Lock()

    def _session(self):
        import earthaccess
        if not hasattr(self._tl, "session"):
            self._tl.session = earthaccess.get_requests_https_session()
        return self._tl.session

    def _s3fs(self):
        import earthaccess
        with self._lock:
            if self._s3 is None or time.time() - self._s3_t > 45 * 60:
                LOG.info("requesting S3 credentials from %s", self.endpoint)
                self._s3 = earthaccess.get_s3_filesystem(endpoint=self.endpoint)
                self._s3_t = time.time()
            return self._s3

    def _get_once(self, url, dst: Path):
        part = dst.with_name(dst.name + ".part")
        if self.access == "s3":
            self._s3fs().get_file(url, str(part))          # single-file get (not get([url], ...))
        else:
            with self._session().get(url, stream=True, allow_redirects=True, timeout=self.timeout) as r:
                if r.status_code in (401, 403) and "eula" in (r.text or "").lower():
                    raise RuntimeError(f"EULA not accepted for {url} (accept it at urs.earthdata.nasa.gov)")
                r.raise_for_status()
                with open(part, "wb") as f:
                    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                        f.write(chunk)
        if part.stat().st_size == 0:
            raise RuntimeError(f"empty download for {url}")
        part.replace(dst)

    def get(self, url):
        dst = self.out_dir / Path(url.split("?")[0]).name
        last = None
        for k in range(self.retries):
            try:
                t0 = time.time()
                self._get_once(url, dst)
                return dst, dst.stat().st_size, time.time() - t0
            except Exception as e:  # noqa: BLE001
                last = e
                dst.with_name(dst.name + ".part").unlink(missing_ok=True)
                LOG.warning("%s: attempt %d/%d failed: %s", dst.name, k + 1, self.retries, e)
                time.sleep(min(60, 5 * 2 ** k))
        raise RuntimeError(f"{dst.name}: {last}")


def download_all(granules, out_dir: Path, access, threads, timeout, retries):
    endpoint = None
    if access == "s3":
        for u in granules[0]["umm"].get("RelatedUrls", []):
            if "/s3credentials" in u.get("URL", ""):
                endpoint = u["URL"]
    dl = Downloader(access, out_dir, timeout=timeout, retries=retries, endpoint=endpoint)
    urls = [granule_links(g, "direct" if access == "s3" else "external")[0] for g in granules]
    LOG.info("first URL: %s", urls[0])
    done, failed, nbytes, t0 = 0, [], 0, time.time()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = {ex.submit(dl.get, u): u for u in urls}
        for f in as_completed(futs):
            try:
                dst, size, sec = f.result()
                done += 1
                nbytes += size
                LOG.info("[%d/%d] %s  %.0f MB in %.0f s  (total %.1f GB, %.1f MB/s)", done + len(failed), len(urls),
                         dst.name, size / 1e6, sec, nbytes / 1e9, nbytes / 1e6 / max(time.time() - t0, 1e-6))
            except Exception as e:  # noqa: BLE001
                failed.append(str(e))
                LOG.error("[%d/%d] %s", done + len(failed), len(urls), e)
    return failed


# Band sets of misc_scripts/tempo_rgb.py and misc_scripts/tempo_rgb_uv.py (0-based indices into the
# colocate_and_resample TEMPO GeoTIFF; output suffix next to the GeoTIFF, Float32, same geotransform / projection).
RGB_PRODUCTS = {
    "rgb": {"bands": [1823, 1043, 900], "suffix": ".RGB.tif"},
    "uv": {"bands": [606, 464, 313], "suffix": ".UV_RGB.tif"},
}
TIF_RE = re.compile(r"TEMPO_RAD_L1_(V\d+)_(\d{8}T\d{6}Z)_S(\d{3})G(\d{2})\.tif$")


def rgb_path(tif: Path, product: str) -> Path:
    return tif.with_name(tif.stem + RGB_PRODUCTS[product]["suffix"])


def rgb_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    from osgeo import gdal
    gdal.UseExceptions()
    try:
        return gdal.Open(str(path)).RasterCount == 3
    except Exception:  # noqa: BLE001
        return False


def make_rgb(tif: str, product: str) -> str:
    """Same output as misc_scripts/tempo_rgb(_uv).py, but reads only the three bands (not the whole cube) and
    writes to a temporary name first, so an interrupted run never leaves a truncated RGB behind."""
    from osgeo import gdal
    gdal.UseExceptions()
    tif = Path(tif)
    out = rgb_path(tif, product)
    bands = RGB_PRODUCTS[product]["bands"]
    src = gdal.Open(str(tif))
    if src.RasterCount <= max(bands):
        raise RuntimeError(f"{tif.name}: {src.RasterCount} bands, {product} needs band index {max(bands)}")
    tmp = out.with_name(out.name.replace(".tif", ".part.tif"))
    dst = gdal.GetDriverByName("GTiff").Create(str(tmp), src.RasterXSize, src.RasterYSize, 3, gdal.GDT_Float32)
    dst.SetGeoTransform(src.GetGeoTransform())
    dst.SetProjection(src.GetProjection())
    for j, b in enumerate(bands):
        dst.GetRasterBand(j + 1).WriteArray(src.GetRasterBand(b + 1).ReadAsArray())
    dst.FlushCache()
    dst = None
    src = None
    os.replace(tmp, out)
    return str(out)


def make_rgbs(tifs: list[Path], products: list[str], procs: int, overwrite: bool):
    """Build the requested RGB products for every base GeoTIFF; existing valid outputs are kept unless overwrite."""
    for p in tifs[0].parent.glob("TEMPO_RAD_L1_*.part.tif") if tifs else []:
        p.unlink(missing_ok=True)
    jobs = [(str(t), pr) for t in tifs for pr in products if overwrite or not rgb_ok(rgb_path(t, pr))]
    LOG.info("RGB %s: %d to build, %d already present", "/".join(products), len(jobs),
             len(tifs) * len(products) - len(jobs))
    done, failed = [], []
    if not jobs:
        return done, failed
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max(1, procs)) as ex:
        futs = {ex.submit(make_rgb, t, pr): (t, pr) for t, pr in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            t, pr = futs[f]
            try:
                done.append(f.result())
                LOG.info("[%d/%d] %s", k, len(jobs), Path(done[-1]).name)
            except Exception as e:  # noqa: BLE001
                failed.append(f"{Path(t).name} {pr}: {e}")
                LOG.error("[%d/%d] %s %s: %s", k, len(jobs), Path(t).name, pr, e)
    return done, failed


def local_tifs(d: Path, start, end, scans=None) -> list[Path]:
    """Base TEMPO GeoTIFFs already in d whose start time is in [start, end) (no search, no login)."""
    out = []
    for p in d.glob("TEMPO_RAD_L1_*.tif"):
        m = TIF_RE.fullmatch(p.name)
        if not m:
            continue                                    # skips .RGB.tif / .UV_RGB.tif / .part.tif
        t = dt.datetime.strptime(m.group(2), "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
        if start <= t < end and (not scans or int(m.group(3)) in scans):
            out.append(p)
    return sorted(out)


DEFAULT_TEMPLATE = Path(__file__).resolve().parents[2] / "config" / "preprocess" / "colocate_and_resample" / \
    "tempo_clip_v04.yaml"


def read_template(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def template_bbox(tpl):
    f = tpl.get("fusion") or {}
    if "lon_bounds" in f and "lat_bounds" in f:
        return [float(f["lon_bounds"][0]), float(f["lon_bounds"][1]), float(f["lat_bounds"][0]),
                float(f["lat_bounds"][1])]
    return None


def write_convert_config(tpl, ncs: list[Path], out: Path) -> Path:
    """tempo_clip_v04.yaml with this run's granules (filenames = geo_filenames = the .nc, output_files = .tif
    next to them, as in the archive)."""
    import yaml
    cfg = copy.deepcopy(tpl)
    ncs = sorted(ncs, key=lambda p: p.name)
    cfg["low_res"]["data"]["filenames"] = [str(p) for p in ncs]
    cfg["low_res"]["data"]["geo_filenames"] = [str(p) for p in ncs]
    cfg["output_files"] = [str(p.with_suffix(".tif")) for p in ncs]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, width=200))
    return out


def run_convert(cfg_path: Path, python=sys.executable):
    """preprocessing/colocate_and_resample.py -y <config> (same entry point as the archive conversion)."""
    import subprocess
    script = Path(__file__).resolve().parents[1] / "colocate_and_resample.py"
    LOG.info("running %s -y %s", script, cfg_path)
    return subprocess.call([python, str(script), "-y", str(cfg_path)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="UTC start (inclusive), e.g. 2026-07-26 or 2026-07-26T12:00:00Z")
    ap.add_argument("--end", required=True, help="UTC end (exclusive), e.g. 2026-07-27T06:00:00Z")
    ap.add_argument("--out-dir", required=True, type=Path, help="download directory for the .nc granules")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "E", "S", "N"),
                    help="only granules intersecting this box (default: the template lon_bounds / lat_bounds)")
    ap.add_argument("--scans", nargs="+", type=int, help="only these scan numbers (S###), e.g. 7 8 9")
    ap.add_argument("--short-name", default="TEMPO_RAD_L1")
    ap.add_argument("--version", default="V04")
    ap.add_argument("--threads", type=int, default=4, help="parallel downloads")
    ap.add_argument("--access", choices=["https", "s3"], default="https",
                    help="https (default, any machine) or s3 (direct S3; only from AWS us-west-2)")
    ap.add_argument("--timeout", type=float, nargs=2, default=[30, 300], metavar=("CONNECT", "READ"),
                    help="HTTPS connect / read-stall timeouts in seconds")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="list the granules only (no login, no download)")
    ap.add_argument("--persist", action="store_true", help="store the Earthdata login in ~/.netrc")
    ap.add_argument("--convert", action="store_true",
                    help="also write GeoTIFFs with preprocessing/colocate_and_resample.py")
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                    help="colocate_and_resample template (default: config/.../tempo_clip_v04.yaml)")
    ap.add_argument("--config-only", action="store_true",
                    help="with --convert: write the colocate_and_resample config but do not run it")
    ap.add_argument("--rgb", nargs="*", choices=sorted(RGB_PRODUCTS),
                    help="also build RGB backgrounds from the GeoTIFFs: rgb (.RGB.tif, tempo_rgb.py bands) and/or uv "
                         "(.UV_RGB.tif, tempo_rgb_uv.py bands); --rgb alone = both")
    ap.add_argument("--rgb-only", action="store_true",
                    help="only build RGBs from GeoTIFFs already in --out-dir for [start, end) (no search/download)")
    ap.add_argument("--rgb-procs", type=int, default=8, help="parallel RGB processes")
    ap.add_argument("--overwrite-rgb", action="store_true", help="rebuild RGBs that already exist")
    ap.add_argument("--delete-nc", action="store_true", help="remove a .nc after its GeoTIFF was written")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    start, end = parse_utc(a.start), parse_utc(a.end)
    if end <= start:
        sys.exit("--end must be after --start")
    scans = set(a.scans) if a.scans else None
    if a.rgb_only and a.rgb is None:
        a.rgb = []
    products = (a.rgb or sorted(RGB_PRODUCTS)) if a.rgb is not None else []
    tag = f"{start:%Y%m%dT%H%M%SZ}_{end:%Y%m%dT%H%M%SZ}"
    if a.rgb_only:
        tifs = local_tifs(a.out_dir, start, end, scans)
        LOG.info("%d GeoTIFF(s) in %s for [%s, %s)", len(tifs), a.out_dir, start.isoformat(), end.isoformat())
        if a.dry_run:
            for t in tifs:
                print(t.name, " ".join(rgb_path(t, pr).name + ("" if rgb_ok(rgb_path(t, pr)) else " (missing)")
                                       for pr in products))
            return
        done, failed = make_rgbs(tifs, products, a.rgb_procs, a.overwrite_rgb)
        (a.out_dir / f"tempo_rgb_{tag}.json").write_text(json.dumps(
            {"start": start.isoformat(), "end": end.isoformat(), "products": products,
             "geotiffs": [str(t) for t in tifs], "built": done, "errors": failed}, indent=1))
        return
    tpl = read_template(a.template) if (a.convert or not a.bbox) and Path(a.template).exists() else None
    if a.convert and tpl is None:
        sys.exit(f"--convert: template {a.template} not found (set --template)")
    bbox = a.bbox or (template_bbox(tpl) if tpl else None)
    if bbox is None:
        LOG.warning("no --bbox and template %s not found: searching all granules (full TEMPO field of regard)",
                    a.template)

    found = search(a.short_name, a.version, start, end, bbox, scans)
    days = {}
    for t, name, _ in found:
        days.setdefault(t.date().isoformat(), []).append(name)
    LOG.info("%s %s: %d granule(s) starting in [%s, %s)%s", a.short_name, a.version, len(found),
             start.isoformat(), end.isoformat(), f" intersecting bbox {bbox}" if bbox else "")
    for d, names in sorted(days.items()):
        scans = sorted({NAME_RE.search(x).group(3) for x in names})
        LOG.info("  %s: %d granules, scans S%s..S%s", d, len(names), scans[0], scans[-1])
    if a.dry_run:
        for t, name, _ in found:
            print(name)
        return
    if not found:
        return

    import earthaccess
    auth = earthaccess.login(persist=a.persist)
    if not getattr(auth, "authenticated", False):
        sys.exit("Earthdata login failed (set up ~/.netrc or EARTHDATA_USERNAME / EARTHDATA_PASSWORD)")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [g for _, name, g in found
            if not ((a.out_dir / name).exists() and (a.out_dir / name).stat().st_size > 0)
            and not ((a.convert or products) and (a.out_dir / name).with_suffix(".tif").exists())]
    LOG.info("downloading %d granule(s) (%d already present) to %s", len(todo), len(found) - len(todo), a.out_dir)
    # leftovers of an interrupted run: our .part files and earthaccess.download() temp files (partial_XXXXXXXX)
    stale = list(a.out_dir.glob("TEMPO_RAD_L1_*.nc.part")) + \
        [p for p in a.out_dir.glob("partial_*") if p.is_file() and re.fullmatch(r"partial_[A-Za-z0-9_]{8}", p.name)]
    if stale:
        LOG.info("removing %d incomplete download(s) from an earlier run", len(stale))
    for p in stale:
        p.unlink(missing_ok=True)
    failed = download_all(todo, a.out_dir, a.access, a.threads, tuple(a.timeout), a.retries) if todo else []
    ncs = [a.out_dir / name for _, name, _ in found if (a.out_dir / name).exists()]
    missing = [name for _, name, _ in found if not (a.out_dir / name).exists()
               and not ((a.convert or products) and (a.out_dir / name).with_suffix(".tif").exists())]
    if missing:
        LOG.warning("%d granule(s) missing after download (rerun to retry): %s", len(missing), missing[:5])

    report = {"short_name": a.short_name, "version": a.version, "start": start.isoformat(), "end": end.isoformat(),
              "bbox": bbox, "granules": [n for _, n, _ in found], "missing": missing,
              "download_errors": failed}
    if a.convert:
        new = [p for p in ncs if not p.with_suffix(".tif").exists()]
        report["convert_config"] = None
        if new:
            cpath = write_convert_config(tpl, new, a.out_dir / f"tempo_clip_v04_{tag}.yaml")
            report["convert_config"] = str(cpath)
            LOG.info("colocate_and_resample config for %d granule(s): %s", len(new), cpath)
            if not a.config_only:
                rc = run_convert(cpath)
                if rc != 0:
                    LOG.error("colocate_and_resample.py exited with %d", rc)
        tifs = [p.with_suffix(".tif") for p in ncs if p.with_suffix(".tif").exists()]
        if a.delete_nc and not a.config_only:
            for t in tifs:
                t.with_suffix(".nc").unlink(missing_ok=True)
        report["geotiffs"] = [str(t) for t in tifs]
        LOG.info("%d of %d granule(s) have a GeoTIFF", len(tifs), len(found))
    if products:
        # every granule of the range with a base GeoTIFF, including ones converted in earlier runs (.nc may be gone)
        tifs = [(a.out_dir / n).with_suffix(".tif") for _, n, _ in found if (a.out_dir / n).with_suffix(".tif").exists()]
        no_tif = len(found) - len(tifs)
        if no_tif:
            LOG.info("%d granule(s) have no GeoTIFF yet (%s); no RGB for them", no_tif,
                     "--config-only" if a.convert else "add --convert")
        done, failed_rgb = make_rgbs(tifs, products, a.rgb_procs, a.overwrite_rgb)
        report["rgb"] = {"products": products, "built": done, "errors": failed_rgb}
    (a.out_dir / f"pull_tempo_rad_l1_{tag}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
