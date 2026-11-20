#!/usr/bin/env python
"""
Copyright [2022-26], by the California Institute of Technology and Chapman University.
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.

Find corrupt or truncated GeoTIFFs (e.g. "ZIPDecode: Decoding error at scanline 0") in geo-product directories by
decoding every band of every raster, and optionally move them aside so export_geo_products.py regenerates them.

    python check_geo_rasters.py -c fusion_config.smoke.yaml [-n 16] [--quarantine]
    python check_geo_rasters.py /path/to/geo_dir [/path/to/other_geo_dir ...] [--pattern "*.tif"]

Directories come from the positional paths and/or every `geo_dir` value found anywhere in the YAML files given
with -c (fusion, production, overlay-animation or UQ configs). Writes <out>/geo_raster_check.csv (path, ok, error)
and prints the unreadable files grouped by directory. Leftover partial files from interrupted exports
(*.part.tif / *.tmp.tif) are listed as well.

--quarantine moves unreadable rasters to <their dir>/_corrupt/ (nothing is deleted). The geo_products_manifest.json
entry then lists a missing file, so the next export_geo_products.py run for that model (skip_existing: true)
re-exports the missing scenes and reuses every intact raster.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml


def geo_dirs_from_yaml(path: Path) -> list[str]:
    """Every `geo_dir` value in a YAML (any nesting), resolved against the YAML's folder."""
    out = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "geo_dir" and isinstance(v, str):
                    out.append(v if os.path.isabs(v) else str((path.parent / v).resolve()))
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(yaml.safe_load(path.read_text()) or {})
    return out


def _root_error(e: BaseException) -> str:
    """Innermost cause of a (rasterio) exception chain, e.g. 'ZIPDecode:Decoding error at scanline 0'."""
    while e.__cause__ is not None:
        e = e.__cause__
    return str(e).splitlines()[0] if str(e) else type(e).__name__


def check(path: str):
    """(path, ok, error): open the file and decode every band in full."""
    os.environ.setdefault("GDAL_NUM_THREADS", "1")
    import rasterio
    try:
        with rasterio.open(path) as s:
            for b in range(1, s.count + 1):
                s.read(b)
        return path, True, ""
    except Exception as e:  # noqa: BLE001  (any read failure means the file cannot be used)
        return path, False, _root_error(e)[:300]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="geo-product directories (searched recursively) or GeoTIFF files")
    ap.add_argument("-c", "--config", nargs="+", default=[], help="YAML(s) whose geo_dir values are checked")
    ap.add_argument("--pattern", default="*.tif")
    ap.add_argument("-n", "--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--quarantine", action="store_true", help="move unreadable rasters to <dir>/_corrupt/")
    ap.add_argument("--out", default=".", help="folder for geo_raster_check.csv")
    a = ap.parse_args(argv)

    roots = list(a.paths)
    for c in a.config:
        roots += geo_dirs_from_yaml(Path(c).resolve())
    roots = list(dict.fromkeys(roots))
    files, partial = [], []
    for r in roots:
        if os.path.isfile(r):
            files.append(r)
            continue
        if not os.path.isdir(r):
            print(f"WARNING not found: {r}")
            continue
        for f in glob.glob(os.path.join(r, "**", a.pattern), recursive=True):
            if f"{os.sep}_corrupt{os.sep}" in f:
                continue
            (partial if f.endswith((".part.tif", ".tmp.tif")) else files).append(f)
    files = sorted(set(files))
    print(f"checking {len(files)} rasters in {len(roots)} location(s) with {a.workers} workers")
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(check, files, chunksize=8))
    df = pd.DataFrame(res, columns=["path", "ok", "error"])
    out = Path(a.out) / "geo_raster_check.csv"
    df.to_csv(out, index=False)
    bad = df[~df.ok]
    print(f"{len(bad)} unreadable of {len(df)}; report {out}")
    for d, g in bad.groupby(bad.path.map(lambda p: str(Path(p).parent))):
        print(f"\n{d}: {len(g)} unreadable")
        for _, r in g.iterrows():
            print(f"  {Path(r.path).name}  ({r.error})")
    if partial:
        print(f"\n{len(partial)} partial file(s) from interrupted exports (safe to delete), e.g. {partial[:3]}")
    if a.quarantine and len(bad):
        for p in bad.path:
            q = Path(p).parent / "_corrupt" / Path(p).name
            q.parent.mkdir(exist_ok=True)
            shutil.move(p, q)
        print(f"\nmoved {len(bad)} raster(s) to _corrupt/; rerun export_geo_products.py for those models to "
              f"regenerate them")


if __name__ == "__main__":
    main()
