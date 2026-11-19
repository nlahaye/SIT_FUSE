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

Authentication (earthaccess.login): ~/.netrc entry for urs.earthdata.nasa.gov, or EARTHDATA_USERNAME /
EARTHDATA_PASSWORD (or EARTHDATA_TOKEN), else an interactive prompt (--persist writes ~/.netrc).
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
    ap.add_argument("--dry-run", action="store_true", help="list the granules only (no login, no download)")
    ap.add_argument("--persist", action="store_true", help="store the Earthdata login in ~/.netrc")
    ap.add_argument("--convert", action="store_true",
                    help="also write GeoTIFFs with preprocessing/colocate_and_resample.py")
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                    help="colocate_and_resample template (default: config/.../tempo_clip_v04.yaml)")
    ap.add_argument("--config-only", action="store_true",
                    help="with --convert: write the colocate_and_resample config but do not run it")
    ap.add_argument("--delete-nc", action="store_true", help="remove a .nc after its GeoTIFF was written")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    start, end = parse_utc(a.start), parse_utc(a.end)
    if end <= start:
        sys.exit("--end must be after --start")
    tpl = read_template(a.template) if (a.convert or not a.bbox) and Path(a.template).exists() else None
    if a.convert and tpl is None:
        sys.exit(f"--convert: template {a.template} not found (set --template)")
    bbox = a.bbox or (template_bbox(tpl) if tpl else None)

    found = search(a.short_name, a.version, start, end, bbox, set(a.scans) if a.scans else None)
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
            and not (a.convert and (a.out_dir / name).with_suffix(".tif").exists())]
    LOG.info("downloading %d granule(s) (%d already present) to %s", len(todo), len(found) - len(todo), a.out_dir)
    if todo:
        earthaccess.download(todo, local_path=str(a.out_dir), threads=a.threads)
    ncs = [a.out_dir / name for _, name, _ in found if (a.out_dir / name).exists()]
    missing = [name for _, name, _ in found if not (a.out_dir / name).exists()
               and not (a.convert and (a.out_dir / name).with_suffix(".tif").exists())]
    if missing:
        LOG.warning("%d granule(s) missing after download (rerun to retry): %s", len(missing), missing[:5])

    report = {"short_name": a.short_name, "version": a.version, "start": start.isoformat(), "end": end.isoformat(),
              "bbox": bbox, "granules": [n for _, n, _ in found], "missing": missing}
    tag = f"{start:%Y%m%dT%H%M%SZ}_{end:%Y%m%dT%H%M%SZ}"
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
    (a.out_dir / f"pull_tempo_rad_l1_{tag}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
