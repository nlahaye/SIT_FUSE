"""
Copyright [2022-26], by the California Institute of Technology and Chapman University.
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.

Group GOES ABI per-band NetCDFs (OR_ABI-L1b-RadC-M6C01_G18_s..._e..._c....nc) into scenes and write a
colocate_and_resample.py config (low_res.data.filenames / geo_filenames = one band list per scene, output_files = one
GeoTIFF per scene, named after the first band's file).

    python build_goes_colocate_config.py -i /data/.../new_data -o fuse_goes.yaml
        [--template fuse_goes_tempo_3.yaml] [--bands 1-16] [--satellite G18] [--product L1b-RadC]
        [--start 2024-07-25] [--end 2024-07-31] [--out-dir /data/.../tifs] [--recursive] [--allow-partial]
    python colocate_and_resample.py -y fuse_goes.yaml

Scenes are keyed on (product, scan mode, satellite, start time); files whose band set is incomplete are skipped
(listed in the log) unless --allow-partial. All non-file settings (readers, fusion grid, bounds, resampling) come
from --template when given, otherwise from the defaults below (same as fuse_goes_tempo_3.yaml).
"""
import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

import yaml

LOG = logging.getLogger("build_goes_colocate_config")
PAT = re.compile(r"OR_ABI-(?P<product>[A-Za-z0-9]+-[A-Za-z0-9]+)-M(?P<mode>\d)C(?P<band>\d{2})_(?P<sat>G\d{2})"
                 r"_s(?P<start>\d{14})_e\d{14}_c\d{14}\.nc$")

DEFAULT = {
    "low_res": {"data": {
        "valid_min": -100.0, "valid_max": 100000000,
        "reader_type": "goes_netcdf", "reader_kwargs": {"no_arg": ""},
        "geo_reader_type": "goes_netcdf_geo", "geo_reader_kwargs": {"no_arg": "no_arg"},
        "filenames": [], "geo_filenames": [],
        "chan_dim": 0, "geo_coord_dim": 0, "geo_lat_index": 0, "geo_lon_index": 1}},
    "high_res": {"data": {
        "valid_min": 0.0, "valid_max": 100000000,
        "reader_type": "numpy", "reader_kwargs": {"no_arg": "no_arg"},
        "geo_reader_type": "numpy", "geo_reader_kwargs": {"no_arg": "no_arg"},
        "filenames": [], "geo_filenames": [],
        "chan_dim": 0, "geo_coord_dim": 2, "geo_lat_index": 0, "geo_lon_index": 1}},
    "fusion": {
        "projection_id": 4326, "description": "Generating GOES GeoTiffs", "area_id": "GOES",
        "projection_proj4": {"proj": "longlat", "datum": "WGS84"},
        "final_resolution": 0.018, "projection_units": "degrees",
        "resample_radius": 5000, "resample_n_neighbors": 64, "resample_n_procs": 10, "resample_epsilon": 1.6,
        "use_bilinear": True, "lon_bounds": [-125.45, -101.98], "lat_bounds": [31.31, 49.01]},
    "output_files": [],
}


def parse_bands(spec):
    out = []
    for part in str(spec).split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return sorted(set(out))


def start_time(s):
    """GOES s-field YYYYJJJHHMMSSt -> datetime (tenths dropped)."""
    return datetime.strptime(s[:13], "%Y%j%H%M%S")


def find_scenes(in_dir, bands, satellite=None, product=None, start=None, end=None, recursive=False,
                allow_partial=False):
    files = Path(in_dir).rglob("*.nc") if recursive else Path(in_dir).glob("*.nc")
    scenes = {}
    for p in files:
        m = PAT.search(p.name)
        if not m or int(m["band"]) not in bands:
            continue
        if satellite and m["sat"] != satellite or product and m["product"] != product:
            continue
        t = start_time(m["start"])
        if start and t < start or end and t >= end:
            continue
        key = (m["product"], m["mode"], m["sat"], m["start"])
        prev = scenes.setdefault(key, {}).get(int(m["band"]))
        if prev is not None:     # reprocessed duplicate: keep the latest creation time
            LOG.warning("duplicate band %s for %s: %s / %s (keeping the later)", m["band"], key, prev.name, p.name)
            p = max(prev, p, key=lambda x: x.name)
        scenes[key][int(m["band"])] = p
    out, skipped = [], 0
    for key in sorted(scenes, key=lambda k: (k[3], k[2], k[0], k[1])):
        have = scenes[key]
        missing = [b for b in bands if b not in have]
        if missing and not allow_partial:
            LOG.warning("skipping %s %s M%s s%s: missing bands %s", key[2], key[0], key[1], key[3], missing)
            skipped += 1
            continue
        out.append([str(have[b].absolute()) for b in bands if b in have])
    kinds = {k[:3] for k in scenes}
    if len(kinds) > 1:
        LOG.warning("several product/mode/satellite combinations in one config: %s "
                    "(use --satellite / --product to separate them)", sorted(kinds))
    LOG.info("%d scenes kept, %d skipped", len(out), skipped)
    return out


def _flow(lists):
    """filenames block in the style of the existing configs: [ [..16 files..], ... ]."""
    rows = [",\n".join(f'"{f}"' for f in scene) for scene in lists]
    return "[\n" + ",\n".join(f"[{r}]" for r in rows) + "\n]"


def write_config(cfg, scenes, out_dir, path):
    d = cfg["low_res"]["data"]
    outs = [str(Path(out_dir) / (Path(s[0]).stem + ".tif")) if out_dir else s[0][:-3] + ".tif" for s in scenes]
    marks = {"filenames": "__FILENAMES__", "geo_filenames": "__GEOFILENAMES__"}
    for k, v in marks.items():
        d[k] = v
    cfg["output_files"] = "__OUTPUTS__"
    txt = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, width=1000)
    txt = txt.replace("__FILENAMES__", _flow(scenes)) \
             .replace("__GEOFILENAMES__", _flow(scenes)) \
             .replace("__OUTPUTS__", "[\n" + ",\n".join(f'"{o}"' for o in outs) + "\n]")
    Path(path).write_text(txt)
    yaml.safe_load(Path(path).read_text())          # sanity: the written file parses
    return outs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input-dir", required=True, help="directory with the per-band GOES NetCDFs")
    ap.add_argument("-o", "--output", required=True, help="config YAML to write")
    ap.add_argument("--template", default=None, help="existing colocate config to copy all non-file settings from")
    ap.add_argument("--bands", default="1-16", help="ABI bands, e.g. 1-16 or 1,2,3,7,14 (default 1-16)")
    ap.add_argument("--satellite", default=None, help="e.g. G18 (default: any)")
    ap.add_argument("--product", default=None, help="e.g. L1b-RadC (default: any)")
    ap.add_argument("--start", default=None, help="keep scenes starting at/after this time (YYYY-MM-DD[THH:MM])")
    ap.add_argument("--end", default=None, help="keep scenes starting before this time")
    ap.add_argument("--out-dir", default=None, help="GeoTIFF directory (default: next to the NetCDFs)")
    ap.add_argument("--recursive", action="store_true", help="search subdirectories too")
    ap.add_argument("--allow-partial", action="store_true", help="keep scenes with missing bands (fewer channels)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(a.template).read_text()) if a.template else DEFAULT
    for k in ("low_res", "high_res", "fusion"):
        cfg.setdefault(k, DEFAULT[k])
    scenes = find_scenes(a.input_dir, parse_bands(a.bands), a.satellite, a.product,
                         datetime.fromisoformat(a.start) if a.start else None,
                         datetime.fromisoformat(a.end) if a.end else None, a.recursive, a.allow_partial)
    if not scenes:
        raise SystemExit(f"No complete GOES scenes found in {a.input_dir}")
    if a.out_dir:
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    write_config(cfg, scenes, a.out_dir, a.output)
    LOG.info("wrote %s (%d scenes)\n  python colocate_and_resample.py -y %s", a.output, len(scenes), a.output)


if __name__ == "__main__":
    main()
