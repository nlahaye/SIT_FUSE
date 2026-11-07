#!/usr/bin/env python3
"""
Pull the VIIRS 375 m active-fire swath products (VNP14IMG / VJ114IMG /
VJ214IMG, collection 2) with their I-band geolocation (VNP03IMG / VJ103IMG /
VJ203IMG) and write one observed-state raster per satellite overpass on the
fire-front grid (raster.resolution_deg over the pull bbox). These rasters
define where FIRMS "no fire" is an observation rather than an absence of
reports: rasterize_fire_fronts.py with raster.negative_zone: observed writes
negatives only on cells this script marks as observed clear land.

VNP14IMG 'fire mask' classes (VIIRS 375 m Active Fire User Guide, Table 1):
    0 not processed   1 bow-tie deletion   2 sun glint   3 water   4 cloud
    5 clear land      6 unclassified       7/8/9 low/nominal/high-confidence fire

Output (uint8, EPSG:4326, 2 bands, nodata 255):
  band 1  cell state
            0  observed, no fire    (clear land + water fraction >= min_clear_fraction,
                                     no fire; water counts as observed no fire unless
                                     water_is_negative: false)
            1  fire                 (any fire_classes pixel in the cell)
            3  water                (only with water_is_negative: false: water majority)
            4  cloud                (any cloud pixel and not observed no fire)
            6  other                (glint, unclassified, mixed)
          255  not observed in this overpass (outside the swath, or only
               ignore_classes pixels: 0 not processed, 1 bow-tie deletion)
  band 2  observed no-fire fraction (clear land + water) in percent (255 = not observed)

Files {prefix}_{YYYYMMDDTHHMMSSZ}_{PLATFORM}.tif (PLATFORM = SNPP | NOAA20 |
NOAA21; time = in-bbox pixel-weighted granule mid time) plus
{prefix}_index.csv (path, platform, time_utc, start_utc, end_utc, granules,
cell counts) in viirs_fire_mask.outdir.

Granules
  Collection 2 granules from CMR (LP DAAC fire mask, LAADS geolocation),
  downloaded with an Earthdata Login token (env var auth.token_env) or
  ~/.netrc, as in pull_viirs_aerosol_type.py. viirs_fire_mask.local_dirs reads
  already-downloaded *14IMG* / *03IMG* files instead (local_only: true = no
  CMR). Geolocation files are ~150 MB each; keep_nc: false deletes them after
  rasterizing.

  only_firms_overpasses: true (default) keeps only passes that contain a FIRMS
  VIIRS overpass of the same satellite (from the firms_detections vector layer
  in output.outdir/vectors, built from firms.local_csv when missing), so the
  download is limited to passes that produce fire-front rasters.

Usage
    export EARTHDATA_TOKEN=...
    python pull_viirs_fire_mask.py firms_fire_fronts.example.local.yaml [--search-only]
    python rasterize_fire_fronts.py firms_fire_fronts.example.local.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
import xarray as xr

try:
    from sit_fuse.preprocessing.data_pull.pull_adp import make_grid
    from sit_fuse.preprocessing.data_pull.pull_viirs_aerosol_type import CMR_URL, make_session, download
    from sit_fuse.preprocessing.data_pull.pull_firms_fire_fronts import load_config
except ImportError:  # run from the data_pull directory
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pull_adp import make_grid
    from pull_viirs_aerosol_type import CMR_URL, make_session, download
    from pull_firms_fire_fronts import load_config

LOG = logging.getLogger("pull-viirs-fire-mask")

DEFAULT_COLLECTIONS = {
    "SNPP": {"fire": {"short_name": "VNP14IMG", "versions": ["002"]},
             "geo": {"short_name": "VNP03IMG", "versions": ["2", "002"]}},
    "NOAA20": {"fire": {"short_name": "VJ114IMG", "versions": ["002"]},
               "geo": {"short_name": "VJ103IMG", "versions": ["2.1", "021", "2"]}},
    "NOAA21": {"fire": {"short_name": "VJ214IMG", "versions": ["002"]},
               "geo": {"short_name": "VJ203IMG", "versions": ["2.1", "021", "2"]}},
}
PREFIX_PLATFORM = {"VNP": "SNPP", "VJ1": "NOAA20", "VJ2": "NOAA21"}
NAME_RE = re.compile(r"(?P<pfx>VNP|VJ1|VJ2)(?P<kind>14IMG|03IMG)\.A(?P<yd>\d{7})\.(?P<hm>\d{4})\.")
GRANULE_SECONDS = 360.0
STATE = {"clear": 0, "fire": 1, "water": 3, "cloud": 4, "other": 6, "unobserved": 255}

# FIRMS satellite / firms_source labels -> VIIRS platform
_SAT_ALIASES = {
    "SNPP": ("N", "NPP", "SNPP", "SUOMINPP", "VIIRSSNPP", "SV"),
    "NOAA20": ("N20", "NOAA20", "J1", "JPSS1", "1", "VIIRSNOAA20", "J1V"),
    "NOAA21": ("N21", "NOAA21", "J2", "JPSS2", "2", "VIIRSNOAA21", "J2V"),
}


def viirs_platform(satellite=None, firms_source=None) -> str | None:
    """VIIRS platform (SNPP | NOAA20 | NOAA21) of a FIRMS overpass, None for MODIS / Landsat."""
    src = re.sub(r"[^A-Z0-9]", "", str(firms_source or "").upper())
    for plat in ("NOAA21", "NOAA20", "SNPP"):
        if plat in src:
            return plat
    sat = re.sub(r"[^A-Z0-9]", "", str(satellite or "").upper())
    for plat, names in _SAT_ALIASES.items():
        if sat in names:
            return plat
    return None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def mask_config(cfg: dict) -> dict:
    """Normalized viirs_fire_mask block (shared by this script and the rasterizer)."""
    vc = cfg.get("viirs_fire_mask", {}) or {}
    rcfg = cfg.get("raster", {}) or {}
    outdir = Path(vc.get("outdir") or Path(cfg["output"]["outdir"]) / "viirs_fire_mask")
    ranges = vc.get("time_ranges") or [cfg["time_range"]]
    tr = [(pd.Timestamp(r["start"]).tz_convert("UTC") if pd.Timestamp(r["start"]).tzinfo
           else pd.Timestamp(r["start"], tz="UTC"),
           pd.Timestamp(r["end"]).tz_convert("UTC") if pd.Timestamp(r["end"]).tzinfo
           else pd.Timestamp(r["end"], tz="UTC")) for r in ranges]
    colls = {**DEFAULT_COLLECTIONS, **(vc.get("collections") or {})}
    plats = [str(p).upper() for p in vc.get("platforms", ["SNPP", "NOAA20", "NOAA21"])]
    unknown = [p for p in plats if p not in colls]
    if unknown:
        raise ValueError(f"viirs_fire_mask.platforms {unknown} have no collections entry")
    out = {
        "outdir": outdir,
        "prefix": str(vc.get("prefix", "viirs_fire_mask")),
        "index_csv": outdir / f"{vc.get('prefix', 'viirs_fire_mask')}_index.csv",
        "platforms": plats,
        "collections": colls,
        "time_ranges": tr,
        "resolution_deg": float(vc.get("resolution_deg") or rcfg.get("resolution_deg", 0.0025)),
        "fire_classes": [int(v) for v in vc.get("fire_classes", [7, 8, 9])],
        "clear_classes": [int(v) for v in vc.get("clear_classes", [5])],
        "cloud_classes": [int(v) for v in vc.get("cloud_classes", [4])],
        "water_classes": [int(v) for v in vc.get("water_classes", [3])],
        # not observations (unprocessed, bow-tie deleted): left out of the cell totals
        "ignore_classes": [int(v) for v in vc.get("ignore_classes", [0, 1])],
        "min_clear_fraction": float(vc.get("min_clear_fraction", 0.8)),
        "water_is_negative": bool(vc.get("water_is_negative", True)),
        "hole_fill_iterations": vc.get("hole_fill_iterations", "auto"),
        "max_gap_seconds": float(vc.get("max_gap_seconds", 600)),
        "day_night": str(vc.get("day_night", "both")).lower(),
        "only_firms_overpasses": bool(vc.get("only_firms_overpasses", True)),
        "match_tolerance_minutes": float(vc.get("match_tolerance_minutes", 10)),
        "local_dirs": vc.get("local_dirs") or [],
        "local_only": bool(vc.get("local_only", False)),
        "keep_nc": bool(vc.get("keep_nc", False)),
        "download_dir": Path(vc.get("download_dir") or outdir / "raw_nc"),
        "token_env": str((cfg.get("auth", {}) or {}).get("token_env",
                         (vc.get("auth", {}) or {}).get("token_env", "EARTHDATA_TOKEN"))),
        "overwrite": bool(vc.get("overwrite", cfg.get("overwrite", False))),
    }
    if out["day_night"] not in ("both", "day", "night"):
        raise ValueError("viirs_fire_mask.day_night must be both | day | night")
    if not 0.0 < out["min_clear_fraction"] <= 1.0:
        raise ValueError("viirs_fire_mask.min_clear_fraction must be in (0, 1]")
    return out


# ---------------------------------------------------------------------------
# Granule discovery (CMR or local files)
# ---------------------------------------------------------------------------

def _parse_name(name: str):
    m = NAME_RE.search(Path(name).name)
    if not m:
        return None
    t = pd.Timestamp(dt.datetime.strptime(m.group("yd") + m.group("hm"), "%Y%j%H%M"), tz="UTC")
    return PREFIX_PLATFORM[m.group("pfx")], m.group("kind"), t


def _in_ranges(t, ranges, pad=pd.Timedelta(0)):
    return any(s - pad <= t <= e + pad for s, e in ranges)


def cmr_granules(short_name: str, versions, bbox, ranges, day_night: str) -> dict:
    """{granule start time: {url, name, day_night}} for the first version with data."""
    w, s, e, n = bbox
    found: dict = {}
    for version in versions or [None]:
        for t0, t1 in ranges:
            params = {"short_name": short_name, "bounding_box": f"{w},{s},{e},{n}",
                      "temporal": f"{t0:%Y-%m-%dT%H:%M:%SZ},{t1:%Y-%m-%dT%H:%M:%SZ}",
                      "page_size": 2000, "page_num": 1}
            if version is not None:
                params["version"] = str(version)
            while True:
                r = requests.get(CMR_URL, params=params, timeout=120)
                r.raise_for_status()
                entries = r.json().get("feed", {}).get("entry", [])
                for g in entries:
                    dn = str(g.get("day_night_flag", "")).upper()
                    if (day_night == "day" and dn == "NIGHT") or (day_night == "night" and dn == "DAY"):
                        continue
                    url = next((l["href"] for l in g.get("links", [])
                                if l.get("rel", "").endswith("/data#") and l["href"].startswith("https")
                                and l["href"].endswith(".nc")), None)
                    key = _parse_name(url or g.get("title", ""))
                    if url is None or key is None or key[2] in found:
                        continue
                    found[key[2]] = {"url": url, "name": Path(url).name, "day_night": dn}
                if len(entries) < params["page_size"]:
                    break
                params["page_num"] += 1
        if found:
            LOG.info("CMR %s v%s: %d granules", short_name, version, len(found))
            break
    if not found:
        LOG.warning("CMR %s: no granules for versions %s", short_name, versions)
    return found


def local_granules(dirs, ranges, base: Path) -> dict:
    """{(platform, kind): {time: {path, name}}} from local *14IMG* / *03IMG* netCDF files."""
    out: dict = {}
    for d in dirs:
        p = Path(d)
        p = p if p.is_absolute() else (base / p)
        files = p.rglob("*.nc") if p.is_dir() else [Path(f) for f in sorted(map(str, p.parent.glob(p.name)))]
        for f in files:
            key = _parse_name(f.name)
            if key is None or not _in_ranges(key[2], ranges, pd.Timedelta(seconds=GRANULE_SECONDS)):
                continue
            out.setdefault((key[0], key[1]), {}).setdefault(key[2], {"path": f, "name": f.name, "day_night": ""})
    return out


def group_passes(times, max_gap_s: float):
    """Consecutive granule start times of one platform -> passes (lists of times)."""
    passes = []
    for t in sorted(times):
        if passes and (t - passes[-1][-1]).total_seconds() <= max_gap_s:
            passes[-1].append(t)
        else:
            passes.append([t])
    return passes


def firms_overpasses(cfg: dict) -> pd.DataFrame | None:
    """FIRMS VIIRS overpasses (platform, overpass_time) from the firms_detections layer."""
    try:
        from sit_fuse.preprocessing.data_pull.rasterize_fire_fronts import read_layer, build_local_vectors
    except ImportError:
        from rasterize_fire_fronts import read_layer, build_local_vectors
    det = read_layer(cfg, "firms_detections")
    if det is None:
        build_local_vectors(cfg, ["firms"])
        det = read_layer(cfg, "firms_detections")
    if det is None or det.empty:
        return None
    ov = det.groupby("overpass_id").agg(overpass_time=("overpass_time", "first"),
                                        satellite=("satellite", "first"),
                                        firms_source=("firms_source", "first")).reset_index()
    ov["platform"] = [viirs_platform(s, f) for s, f in zip(ov["satellite"], ov["firms_source"])]
    return ov[ov["platform"].notna()].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def read_granule(fire_nc: Path, geo_nc: Path):
    """fire mask classes plus I-band latitude / longitude (same swath shape)."""
    with xr.open_dataset(fire_nc, mask_and_scale=False) as ds:
        var = "fire mask" if "fire mask" in ds else next(v for v in ds.data_vars if "fire" in v.lower()
                                                         and "mask" in v.lower())
        fm = ds[var].values.astype(np.uint8)
    with xr.open_dataset(geo_nc, group="geolocation_data", mask_and_scale=False) as g:
        lat = g["latitude"].values.astype(np.float32)
        lon = g["longitude"].values.astype(np.float32)
    if fm.shape != lat.shape:
        r, c = min(fm.shape[0], lat.shape[0]), min(fm.shape[1], lat.shape[1])
        LOG.warning("%s: fire mask %s vs geolocation %s; using the common %dx%d block",
                    fire_nc.name, fm.shape, lat.shape, r, c)
        fm, lat, lon = fm[:r, :c], lat[:r, :c], lon[:r, :c]
    return fm, lat, lon


def bin_counts(fm, lat, lon, bbox, res, mc):
    """Per-cell pixel counts for fire / clear / cloud / water / other on the output grid."""
    width, height, _ = make_grid(bbox, res)
    w, s, e, n = bbox
    ok = np.isfinite(lat) & np.isfinite(lon) & (lon >= w) & (lon < e) & (lat > s) & (lat <= n)
    col = np.floor((lon[ok] - w) / res).astype(np.int64)
    row = np.floor((n - lat[ok]) / res).astype(np.int64)
    keep = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    idx = row[keep] * width + col[keep]
    cls = fm[ok][keep]
    obs = ~np.isin(cls, mc["ignore_classes"])
    idx, cls = idx[obs], cls[obs]
    groups = {"fire": mc["fire_classes"], "clear": mc["clear_classes"],
              "cloud": mc["cloud_classes"], "water": mc["water_classes"]}
    out, used = {}, np.zeros(cls.shape, bool)
    for k, vals in groups.items():
        m = np.isin(cls, vals)
        used |= m
        out[k] = np.bincount(idx[m], minlength=width * height).reshape(height, width).astype(np.int32)
    out["other"] = np.bincount(idx[~used], minlength=width * height).reshape(height, width).astype(np.int32)
    return out, int(ok.sum())


def cell_state(c: dict, mc: dict):
    """band 1 state and band 2 clear percent from summed counts."""
    total = sum(c[k] for k in ("fire", "clear", "cloud", "water", "other"))
    state = np.full(total.shape, STATE["unobserved"], np.uint8)
    seen = total > 0
    frac = np.zeros(total.shape, np.float32)
    neg = c["clear"] + c["water"] if mc["water_is_negative"] else c["clear"]
    frac[seen] = neg[seen] / total[seen]
    state[seen] = STATE["other"]
    state[seen & (c["cloud"] > 0)] = STATE["cloud"]
    if not mc["water_is_negative"]:
        state[seen & (c["water"] * 2 > total)] = STATE["water"]
    state[seen & (frac >= mc["min_clear_fraction"])] = STATE["clear"]
    state[c["fire"] > 0] = STATE["fire"]
    pct = np.where(seen, np.round(frac * 100.0), 255).astype(np.uint8)
    return state, pct


def fill_holes(state, pct, iterations: int):
    """Fill unobserved cells enclosed by observed ones (grid finer than the pixel spacing)."""
    from scipy import ndimage
    for _ in range(int(iterations)):
        miss = state == STATE["unobserved"]
        if not miss.any():
            break
        obs = ~miss
        nb = ndimage.convolve(obs.astype(np.int8), np.ones((3, 3), np.int8), mode="constant")
        cand = miss & (nb >= 5)                      # mostly surrounded: a binning gap, not the swath edge
        if not cand.any():
            break
        # most conservative neighbour state: fire > cloud > other > water > clear
        rank = np.select([state == STATE["fire"], state == STATE["cloud"], state == STATE["other"],
                          state == STATE["water"], state == STATE["clear"]], [5, 4, 3, 2, 1], 0)
        best = ndimage.maximum_filter(rank, size=3, mode="constant")
        back = {5: STATE["fire"], 4: STATE["cloud"], 3: STATE["other"], 2: STATE["water"], 1: STATE["clear"]}
        p_nb = ndimage.minimum_filter(np.where(obs, pct, 255).astype(np.uint8), size=3, mode="constant")
        for r, v in back.items():
            m = cand & (best == r)
            state[m] = v
        pct[cand] = p_nb[cand]
    return state, pct


def write_state(path: Path, state, pct, bbox, res, tags: dict):
    width, height, transform = make_grid(bbox, res)
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", width=width, height=height, count=2, dtype="uint8", crs="EPSG:4326",
                transform=transform, nodata=255, compress="deflate", tiled=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(state, 1)
        dst.write(pct, 2)
        dst.set_band_description(1, "viirs_state_0clear_1fire_3water_4cloud_6other")
        dst.set_band_description(2, "observed_no_fire_percent")
        dst.update_tags(**{k: str(v) for k, v in tags.items()})


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="VIIRS 375 m fire-mask observed-state rasters per overpass.")
    ap.add_argument("config", type=Path)
    ap.add_argument("--search-only", action="store_true", help="list passes, no download")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(a.config)
    mc = mask_config(cfg)
    bbox, res = cfg["_bbox"], mc["resolution_deg"]
    holes = mc["hole_fill_iterations"]
    holes = (1 if res < 0.01 else 0) if str(holes).lower() == "auto" else int(holes)
    if res < 0.003:
        LOG.warning("resolution_deg %.4f is finer than the VIIRS I-band pixel; cells between pixel "
                    "centres stay unobserved beyond hole_fill_iterations=%d", res, holes)

    ov = firms_overpasses(cfg) if mc["only_firms_overpasses"] else None
    if mc["only_firms_overpasses"]:
        if ov is None or ov.empty:
            LOG.warning("only_firms_overpasses: no FIRMS VIIRS overpasses found; nothing to pull")
            return 0
        LOG.info("FIRMS VIIRS overpasses: %d (%s)", len(ov), ", ".join(
            f"{p}={n}" for p, n in ov["platform"].value_counts().items()))

    local = local_granules(mc["local_dirs"], mc["time_ranges"], cfg["_cfgdir"]) if mc["local_dirs"] else {}
    tol = pd.Timedelta(minutes=mc["match_tolerance_minutes"])
    jobs = []
    for plat in mc["platforms"]:
        if ov is not None and not (ov["platform"] == plat).any():
            continue
        coll = mc["collections"][plat]
        fire, geo = dict(local.get((plat, "14IMG"), {})), dict(local.get((plat, "03IMG"), {}))
        if not mc["local_only"]:
            for t, g in cmr_granules(coll["fire"]["short_name"], coll["fire"].get("versions"), bbox,
                                     mc["time_ranges"], mc["day_night"]).items():
                fire.setdefault(t, g)
            if set(fire) - set(geo):
                for t, g in cmr_granules(coll["geo"]["short_name"], coll["geo"].get("versions"), bbox,
                                         mc["time_ranges"], "both").items():
                    geo.setdefault(t, g)
        both = sorted(set(fire) & set(geo))
        if len(both) < len(fire):
            LOG.warning("%s: %d fire-mask granules without geolocation skipped", plat, len(fire) - len(both))
        for p in group_passes(both, mc["max_gap_seconds"]):
            t0, t1 = p[0], p[-1] + pd.Timedelta(seconds=GRANULE_SECONDS)
            if ov is not None:
                o = ov[ov["platform"] == plat]["overpass_time"]
                if not ((o >= t0 - tol) & (o <= t1 + tol)).any():
                    continue
            jobs.append((plat, [(t, fire[t], geo[t]) for t in p]))
    LOG.info("%d passes to rasterize", len(jobs))
    if a.search_only:
        for plat, p in jobs:
            print(f"  {plat:7s} {p[0][0]:%Y-%m-%d %H:%M}  " + " ".join(f["name"] for _, f, _ in p))
        return 0

    sess = None
    mc["outdir"].mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(mc["index_csv"]) if mc["index_csv"].exists() else pd.DataFrame()
    rows = []
    for plat, p in jobs:
        t0, t1 = p[0][0], p[-1][0] + pd.Timedelta(seconds=GRANULE_SECONDS)
        if not mc["overwrite"] and len(old) and ((old["platform"] == plat) &
                                                (old["start_utc"] == f"{t0:%Y-%m-%dT%H:%M:%SZ}")).any():
            rows.append(old[(old["platform"] == plat) & (old["start_utc"] == f"{t0:%Y-%m-%dT%H:%M:%SZ}")]
                        .iloc[0].to_dict())
            continue
        summed, mids, weights, names, dns = None, [], [], [], set()
        for t, f, g in p:
            try:
                paths = []
                for item in (f, g):
                    if "path" in item:
                        paths.append(Path(item["path"]))
                    else:
                        sess = sess or make_session(mc["token_env"])
                        paths.append(download(sess, item["url"], mc["download_dir"] / f"{t:%Y%m%d}" / item["name"],
                                              False))
                fm, lat, lon = read_granule(*paths)
                c, n_in = bin_counts(fm, lat, lon, bbox, res, mc)
                if not mc["keep_nc"]:
                    for pth, item in zip(paths, (f, g)):
                        if "url" in item:
                            pth.unlink(missing_ok=True)
            except Exception as exc:  # one bad granule must not stop the pass
                LOG.error("%s %s: %s", plat, t, exc)
                continue
            if n_in == 0:
                continue
            summed = c if summed is None else {k: summed[k] + c[k] for k in c}
            mids.append((t + pd.Timedelta(seconds=GRANULE_SECONDS / 2)).value)
            weights.append(n_in)
            names.append(f["name"])
            dns.add(f.get("day_night", ""))
        if summed is None:
            continue
        state, pct = cell_state(summed, mc)
        if holes:
            state, pct = fill_holes(state, pct, holes)
        tm = pd.Timestamp(int(np.average(mids, weights=weights)), tz="UTC").round("s")
        path = mc["outdir"] / "rasters" / f"{mc['prefix']}_{tm:%Y%m%dT%H%M%SZ}_{plat}.tif"
        write_state(path, state, pct, bbox, res, {"platform": plat, "time_utc": tm.isoformat(),
                                                  "granules": ",".join(names),
                                                  "min_clear_fraction": mc["min_clear_fraction"],
                                                  "water_is_negative": mc["water_is_negative"]})
        rec = {"path": str(path.resolve()), "platform": plat, "time_utc": f"{tm:%Y-%m-%dT%H:%M:%SZ}",
               "start_utc": f"{t0:%Y-%m-%dT%H:%M:%SZ}", "end_utc": f"{t1:%Y-%m-%dT%H:%M:%SZ}",
               "day_night": ",".join(sorted(d for d in dns if d)), "granules": ";".join(names),
               **{f"n_{k}": int((state == v).sum()) for k, v in STATE.items()}}
        rows.append(rec)
        LOG.info("%s: clear=%d fire=%d cloud=%d water=%d other=%d", path.name, rec["n_clear"], rec["n_fire"],
                 rec["n_cloud"], rec["n_water"], rec["n_other"])
    new = pd.DataFrame(rows)
    idx = pd.concat([old, new], ignore_index=True) if len(old) else new
    if len(idx):
        idx = idx.drop_duplicates(subset=["platform", "start_utc"], keep="last").sort_values("time_utc")
    idx.to_csv(mc["index_csv"], index=False)
    LOG.info("Wrote %d pass rasters; index %s", len(new), mc["index_csv"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
