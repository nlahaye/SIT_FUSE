#!/usr/bin/env python3
"""
Rasterize the vectors written by pull_firms_fire_fronts.py into timestamped
fire-front GeoTIFFs on a regular EPSG:4326 grid (raster.resolution_deg over the
pull bbox). One raster per FEDS 12-hour step and/or per FIRMS overpass.

Encoding (uint8, single band; values configurable under raster.classes):
    1    active fire front
           feds:  fire line buffered by front_buffer_m (+ new 375 m pixels)
           firms: detections that are new relative to the previous
                  lookback_hours (FEDS-style new pixels); front_mode: all makes
                  every detection front
    2    fire-affected, not front
           feds:  inside the modeled perimeter (burned / residual activity)
           firms: persistent detections (seen within lookback_hours)
    0    no fire front reported, inside raster.negative_zone
    255  NoData (outside the negative zone)

Negatives are "no front reported", not "observed clear": FEDS and FIRMS carry
no cloud or swath mask, so cloud-obscured fire is labeled 0 or 255. The
default negative zone is a ring of ring_km around the fire geometry of the step,
which keeps negatives close to observed fire activity.

Files: {prefix}_{YYYYMMDDTHHMMSSZ}_{TAG}.tif, where the time is the (estimated)
overpass time and TAG is FEDS-AM / FEDS-PM or the FIRMS satellite, plus
fire_front_index.csv (path, timestamp, source, tag, class counts) for
firms_fire_front_matchup_generator.py.

Usage:
    python rasterize_fire_fronts.py firms_fire_fronts.example.yaml [--products feds firms]

raster.products: auto (default) rasterizes every product whose vectors exist in
the pull output; a product listed without vectors, or vectors not listed, are
logged as warnings.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from shapely.geometry import box
from shapely.ops import unary_union

try:
    from sit_fuse.preprocessing.data_pull.pull_firms_fire_fronts import load_config, vector_dir
except ImportError:  # run from the data_pull directory
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pull_firms_fire_fronts import load_config, vector_dir

LOG = logging.getLogger("rasterize-fire-fronts")
DEFAULT_CLASSES = {"front": 1, "interior": 2, "negative": 0}


class Grid:
    def __init__(self, bbox, res):
        x0, y0, x1, y1 = bbox
        self.res = float(res)
        self.width = int(np.ceil((x1 - x0) / self.res))
        self.height = int(np.ceil((y1 - y0) / self.res))
        self.transform = from_origin(x0, y1, self.res, self.res)
        self.shape = (self.height, self.width)

    def burn(self, arr, geoms, value, all_touched=True):
        geoms = [g for g in geoms if g is not None and not g.is_empty]
        if geoms:
            m = rasterize(((g, 1) for g in geoms), out_shape=self.shape, transform=self.transform,
                          fill=0, all_touched=all_touched, dtype="uint8").astype(bool)
            arr[m] = value
        return arr


def read_layer(cfg, name):
    d = vector_dir(cfg)
    for ext in ("parquet", "gpkg"):
        p = d / f"{name}.{ext}"
        if p.exists():
            g = gpd.read_parquet(p) if ext == "parquet" else gpd.read_file(p)
            for c in ("step_t", "obs_time", "acq_time", "overpass_time"):
                if c in g:
                    g[c] = pd.to_datetime(g[c], utc=True)
            return g
    return None


def _layer_exists(cfg, name):
    d = vector_dir(cfg)
    return any((d / f"{name}.{ext}").exists() for ext in ("parquet", "gpkg"))


def write_tif(path, arr, grid, fill, tags):
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", height=grid.height, width=grid.width, count=1, dtype="uint8",
                crs="EPSG:4326", transform=grid.transform, nodata=fill, compress="deflate")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})


def negative_zone(fire_geoms_m, utm, bbox, rcfg):
    mode = str(rcfg.get("negative_zone", "ring")).lower()
    if mode == "bbox":
        return [box(*bbox)]
    if mode == "none":
        return []
    if mode != "ring":
        raise ValueError("raster.negative_zone must be ring | bbox | none")
    geoms = [g for g in fire_geoms_m if g is not None and not g.is_empty]
    if not geoms:
        return []
    ring = unary_union(geoms).buffer(float(rcfg.get("ring_km", 25)) * 1000.0)
    return list(gpd.GeoSeries([ring], crs=utm).to_crs("EPSG:4326"))


def counts(arr, classes, fill):
    return {f"n_{k}": int((arr == v).sum()) for k, v in classes.items()} | {"n_nodata": int((arr == fill).sum())}


# ---------------------------------------------------------------------------
# FEDS steps
# ---------------------------------------------------------------------------

def rasterize_feds(cfg, grid, utm, classes, fill, out_dir, prefix):
    rcfg = cfg.get("raster", {}) or {}
    fc = rcfg.get("feds", {}) or {}
    line, per, pix = (read_layer(cfg, f"feds_{n}") for n in ("fireline", "perimeter", "newfirepix"))
    if line is None and pix is None:
        LOG.warning("FEDS: no fireline / newfirepix layers; skipped")
        return []
    steps = set(line["step_t"]) if line is not None else set()
    if pix is not None and fc.get("include_newfirepix", True):
        steps |= set(pix["step_t"])
    steps = sorted(steps)
    buf = float(fc.get("front_buffer_m", 375))
    half_px = float(fc.get("newfirepix_size_m", 375)) / 2.0
    persist = pd.Timedelta(hours=float(fc.get("perimeter_persist_hours", 72)))
    to_m = lambda g: g.to_crs(utm) if g is not None else None
    line_m, per_m, pix_m = to_m(line), to_m(per), to_m(pix)
    rows = []
    for t in steps:
        front_m, obs = [], []
        if line_m is not None:
            sel = line_m[line_m["step_t"] == t]
            front_m += list(sel.geometry.buffer(buf))
            obs += list(line.loc[sel.index, "obs_time"])
        if pix_m is not None and fc.get("include_newfirepix", True):
            sel = pix_m[pix_m["step_t"] == t]
            front_m += list(sel.geometry.buffer(half_px, cap_style=3))
            obs += list(pix.loc[sel.index, "obs_time"])
        interior_m = []
        if per_m is not None and fc.get("interior", True):
            cur = per_m[(per_m["step_t"] <= t) & (per_m["step_t"] > t - persist)]
            if len(cur):
                last = cur.sort_values("step_t").groupby("fireid").tail(1)
                interior_m = list(last.geometry)
        if not front_m:
            continue
        to_ll = lambda gs: list(gpd.GeoSeries(gs, crs=utm).to_crs("EPSG:4326")) if gs else []
        arr = np.full(grid.shape, fill, np.uint8)
        grid.burn(arr, negative_zone(front_m + interior_m, utm, cfg["_bbox"], rcfg), classes["negative"], False)
        grid.burn(arr, to_ll(interior_m), classes["interior"], True)
        grid.burn(arr, to_ll(front_m), classes["front"], True)
        ts = pd.Series(obs).median().round("s")
        half = "PM" if t.hour >= 12 else "AM"
        path = out_dir / f"{prefix}_{ts:%Y%m%dT%H%M%SZ}_FEDS-{half}.tif"
        write_tif(path, arr, grid, fill, {"source": "feds", "step_t": t.isoformat(), "obs_time": ts.isoformat()})
        rows.append({"path": str(path.resolve()), "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "source": "feds", "tag": f"FEDS-{half}", "step_t": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "n_fires": int(line["fireid"][line["step_t"] == t].nunique()) if line is not None else 0,
                     **counts(arr, classes, fill)})
    LOG.info("FEDS: %d step rasters", len(rows))
    return rows


# ---------------------------------------------------------------------------
# FIRMS overpasses
# ---------------------------------------------------------------------------

def footprints_m(det_m):
    """Pixel footprints: scan x track km rectangles centered on the detection."""
    x, y = det_m.geometry.x.to_numpy(), det_m.geometry.y.to_numpy()
    hs, ht = det_m["scan"].to_numpy(float) * 500.0, det_m["track"].to_numpy(float) * 500.0
    return [box(a - s, b - t, a + s, b + t) for a, b, s, t in zip(x, y, hs, ht)]


def rasterize_firms(cfg, grid, utm, classes, fill, out_dir, prefix):
    rcfg = cfg.get("raster", {}) or {}
    fc = rcfg.get("firms", {}) or {}
    det = read_layer(cfg, "firms_detections")
    if det is None or det.empty:
        LOG.warning("FIRMS: no detections layer; skipped")
        return []
    lev = {"low": 0, "nominal": 1, "high": 2}[str(fc.get("min_confidence", "nominal")).lower()]
    det = det[det["confidence_level"] >= lev].reset_index(drop=True)
    det_m = det.to_crs(utm)
    xy = np.c_[det_m.geometry.x, det_m.geometry.y]
    tree = cKDTree(xy)
    times = det["acq_time"]
    mode = str(fc.get("front_mode", "new")).lower()
    look = pd.Timedelta(hours=float(fc.get("lookback_hours", 24)))
    gap = pd.Timedelta(minutes=float(fc.get("min_prior_gap_minutes", 30)))
    rad = float(fc.get("new_distance_m", 750))
    rows = []
    for oid, sel in det_m.groupby("overpass_id"):
        T = sel["overpass_time"].iloc[0]
        idx = sel.index.to_numpy()
        prior = ((times >= T - look) & (times <= T - gap)).to_numpy()
        if mode == "new":
            near = tree.query_ball_point(xy[idx], r=rad)
            is_new = np.array([not prior[n].any() for n in near])
        elif mode == "all":
            is_new = np.ones(len(idx), bool)
        else:
            raise ValueError("raster.firms.front_mode must be new | all")
        fp = footprints_m(sel)
        front_m = [g for g, n in zip(fp, is_new) if n]
        persist_m = [g for g, n in zip(fp, is_new) if not n]
        prior_m = footprints_m(det_m[prior]) if prior.any() else []
        to_ll = lambda gs: list(gpd.GeoSeries(gs, crs=utm).to_crs("EPSG:4326")) if gs else []
        arr = np.full(grid.shape, fill, np.uint8)
        grid.burn(arr, negative_zone(fp + prior_m, utm, cfg["_bbox"], rcfg), classes["negative"], False)
        grid.burn(arr, to_ll(persist_m), classes["interior"], True)
        grid.burn(arr, to_ll(front_m), classes["front"], True)
        sat = str(sel["satellite"].iloc[0]).replace(" ", "")
        path = out_dir / f"{prefix}_{T:%Y%m%dT%H%M%SZ}_{sat}.tif"
        write_tif(path, arr, grid, fill, {"source": "firms", "satellite": sat, "overpass_time": T.isoformat(),
                                          "front_mode": mode, "lookback_hours": look.total_seconds() / 3600})
        rows.append({"path": str(path.resolve()), "timestamp": T.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "source": "firms", "tag": sat, "step_t": "", "n_fires": int(len(sel)),
                     "n_new_detections": int(is_new.sum()), **counts(arr, classes, fill)})
    LOG.info("FIRMS: %d overpass rasters", len(rows))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--products", nargs="+", choices=["feds", "firms", "auto"], default=None,
                    help="default raster.products, else auto (every product with vectors)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(a.config)
    rcfg = cfg.get("raster", {}) or {}
    classes = {**DEFAULT_CLASSES, **(rcfg.get("classes") or {})}
    fill = int(rcfg.get("fill_value", 255))
    if fill in classes.values():
        raise ValueError("raster.fill_value collides with a class value")
    grid = Grid(cfg["_bbox"], rcfg.get("resolution_deg", 0.0025))
    utm = gpd.GeoSeries([box(*cfg["_bbox"])], crs="EPSG:4326").estimate_utm_crs()
    out_dir = Path(rcfg.get("outdir") or cfg["_outdir"] / "rasters")
    prefix = rcfg.get("prefix", "fire_front")
    products = a.products or rcfg.get("products", "auto")
    vdir = vector_dir(cfg)
    have = {"feds": any(_layer_exists(cfg, f"feds_{n}") for n in ("fireline", "newfirepix")),
            "firms": _layer_exists(cfg, "firms_detections")}
    LOG.info("Vector dir %s: feds=%s firms=%s", vdir.resolve(), have["feds"], have["firms"])
    if products == "auto" or products == ["auto"]:
        products = [p for p, ok in have.items() if ok]
    products = list(products)
    for p in products:
        if not have.get(p):
            LOG.warning("raster.products includes %s but %s has no %s vectors (run pull_firms_fire_fronts.py "
                        "with the same config / output.outdir)", p, vdir.resolve(),
                        "feds_fireline / feds_newfirepix" if p == "feds" else "firms_detections")
    for p, ok in have.items():
        if ok and p not in products:
            LOG.warning("%s vectors exist in %s but %s is not in raster.products %s (not rasterized)",
                        p, vdir.resolve(), p, products)
    if not products:
        LOG.warning("No vector layers to rasterize in %s", vdir.resolve())
    LOG.info("Grid %dx%d at %.4f deg, metric CRS %s", grid.width, grid.height, grid.res, utm)
    rows = []
    if "feds" in products:
        rows += rasterize_feds(cfg, grid, utm, classes, fill, out_dir, prefix)
    if "firms" in products:
        rows += rasterize_firms(cfg, grid, utm, classes, fill, out_dir, prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = out_dir / "fire_front_index.csv"
    pd.DataFrame(rows).sort_values("timestamp").to_csv(idx, index=False) if rows else \
        pd.DataFrame(columns=["path", "timestamp", "source", "tag"]).to_csv(idx, index=False)
    LOG.info("Wrote %d rasters; index %s", len(rows), idx)


if __name__ == "__main__":
    sys.exit(main())
