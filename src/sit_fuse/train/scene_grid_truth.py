#!/usr/bin/env python3
"""
Bucket (max) resampling of fine reference rasters onto SIT-FUSE scene grids.

Nearest-neighbour resampling of a fine fire mask (FDCC 0.02 deg, HMS fire,
FIRMS / FEDS fronts at ~250 m) onto a coarse scene grid (TEMPO ~10 km) keeps
only the fine cell under each scene-pixel centre, so most fire pixels are
lost. Here every fine cell centre is assigned to the scene pixel that contains
it (affine or GCP georeferencing), and each scene pixel takes the maximum of
its valid fine cells:

    positive_values given   fine cells are binarized (positive -> 1, other
                            valid -> 0); a scene pixel is 1 if ANY of its
                            fine cells is positive (max operator), 0 if it has
                            valid cells and none is positive, NoData otherwise
    positive_values null    the scene pixel takes the max of the raw valid
                            values (e.g. FIRMS front 1 / interior 2)

min_valid_fraction applies to negatives only: a pixel without positives is
NoData when fewer than that fraction of its fine cells are valid (0 = any
valid cell). A positive subpixel always makes the pixel positive. Scene pixels
smaller than a fine cell (no cell centre inside) take the fine cell at their
centre.

Output rasters are on the scene grid (same height / width / transform, or the
scene GCPs), band 1 = bucketed label, band 2 = positive fraction of the valid
fine cells in percent (255 = NoData), so the UQ pipeline reads them without
resampling (classifier_utils.rasters_are_collocated).

Used by
    matchups/build_fdcc_matchups.py, matchups/build_simple_classifier_matchups.py
        scene_grid: config block (bucket_matchups)
    sf_uq_utils.align_reference
        target.reference_resampling: max (any reference not yet on the scene grid)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio
import rasterio.errors
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import AffineTransformer, GCPTransformer
from rasterio.windows import Window

LOG = logging.getLogger("scene-grid-truth")
DEFAULT_FILL = 255


# ---------------------------------------------------------------------------
# value helpers
# ---------------------------------------------------------------------------

def value_mask(arr: np.ndarray, spec: Iterable | None) -> np.ndarray:
    """True where arr matches spec: ints and/or [lo, hi] inclusive ranges."""
    m = np.zeros(arr.shape, bool)
    for v in spec or []:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            m |= (arr >= float(v[0])) & (arr <= float(v[1]))
        else:
            m |= arr == float(v)
    return m


def _same_crs(a, b) -> bool:
    return CRS.from_user_input(a) == CRS.from_user_input(b)


# ---------------------------------------------------------------------------
# scene grids
# ---------------------------------------------------------------------------

class SceneGrid:
    """Pixel grid of a SIT-FUSE scene (affine transform or GCPs)."""

    def __init__(self, height, width, transformer, crs, transform=None, gcps=None, gcp_crs=None):
        self.height, self.width = int(height), int(width)
        self.tr, self.crs = transformer, crs
        self.transform, self.gcps, self.gcp_crs = transform, gcps, gcp_crs

    @classmethod
    def from_path(cls, path):
        with rasterio.open(path) as s:
            gcps, gcrs = s.gcps
            if gcps:
                crs = gcrs or CRS.from_epsg(4326)
                return cls(s.height, s.width, GCPTransformer(gcps), crs, gcps=gcps, gcp_crs=crs)
            if s.crs is None:
                raise ValueError(f"{path}: no CRS / georeferencing")
            return cls(s.height, s.width, AffineTransformer(s.transform), s.crs, transform=s.transform)

    @classmethod
    def from_profile(cls, prof: dict):
        if prof.get("crs") is None or prof.get("transform") is None:
            raise ValueError("bucket resampling needs a scene profile with crs and transform")
        return cls(prof["height"], prof["width"], AffineTransformer(prof["transform"]), prof["crs"],
                   transform=prof["transform"])

    def outline_xy(self, n=25):
        """Pixel-corner lattice (x, y) in the scene CRS (for source cropping)."""
        r = np.linspace(0, self.height, n)
        c = np.linspace(0, self.width, n)
        rr, cc = np.meshgrid(r, c, indexing="ij")
        xs, ys = self.tr.xy(rr.ravel(), cc.ravel(), offset="ul")
        return np.asarray(xs, float), np.asarray(ys, float)

    def profile(self, count, dtype, nodata):
        p = {"driver": "GTiff", "height": self.height, "width": self.width, "count": count,
             "dtype": dtype, "nodata": nodata, "compress": "deflate"}
        if not self.gcps:
            p.update(crs=self.crs, transform=self.transform)
        return p


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------

def _crop_window(src, grid: SceneGrid, pad_cells=2):
    """Source window covering the scene (plus a pad); None if they do not overlap."""
    xs, ys = grid.outline_xy()
    if not _same_crs(grid.crs, src.crs):
        xs, ys = Transformer.from_crs(grid.crs, src.crs, always_xy=True).transform(xs, ys)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if not ok.any():
        return None
    cols, rows = ~src.transform * (xs[ok], ys[ok])
    r0, r1 = int(np.floor(np.min(rows))) - pad_cells, int(np.ceil(np.max(rows))) + pad_cells
    c0, c1 = int(np.floor(np.min(cols))) - pad_cells, int(np.ceil(np.max(cols))) + pad_cells
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, src.height), min(c1, src.width)
    if r1 <= r0 or c1 <= c0:
        return None
    return Window(c0, r0, c1 - c0, r1 - r0)


def bucket_max(A: np.ndarray, a_transform, a_crs, grid: SceneGrid, nodata_values=(DEFAULT_FILL,),
               positive_values=None, min_valid_fraction=0.0, fill=DEFAULT_FILL):
    """
    Max-bucket A (fine grid, affine a_transform in a_crs) onto grid.

    Returns (label float32 [H, W] with fill as NoData, positive fraction percent
    uint8 [H, W] with 255 as NoData, n_fine_cells_used).
    """
    H, W = grid.height, grid.width
    n = H * W
    A = np.asarray(A, np.float64)
    invalid = ~np.isfinite(A) | value_mask(A, nodata_values)
    if positive_values is not None:
        pos = value_mask(A, positive_values) & ~invalid
        V = pos.astype(np.float64)
    else:
        V = A
        pos = (V > 0) & ~invalid
    fr, fc = np.indices(A.shape)
    fr, fc = fr.ravel(), fc.ravel()
    xs, ys = a_transform * (fc + 0.5, fr + 0.5)
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if not _same_crs(grid.crs, a_crs):
        xs, ys = Transformer.from_crs(a_crs, grid.crs, always_xy=True).transform(xs, ys)
    sr, sc = grid.tr.rowcol(xs, ys)
    sr, sc = np.asarray(sr), np.asarray(sc)
    inside = (sr >= 0) & (sr < H) & (sc >= 0) & (sc < W)
    pix = (sr * W + sc)[inside]
    a_inv = invalid.ravel()[inside]
    a_pos = pos.ravel()[inside]
    a_val = V.ravel()[inside]
    tot = np.bincount(pix, minlength=n).astype(np.int64)
    nval = np.bincount(pix[~a_inv], minlength=n).astype(np.int64)
    npos = np.bincount(pix[a_pos], minlength=n).astype(np.int64)
    vmax = np.full(n, -np.inf)
    np.maximum.at(vmax, pix[~a_inv], a_val[~a_inv])
    # scene pixels smaller than a fine cell: the fine cell at the pixel centre
    empty = np.flatnonzero(tot == 0)
    if empty.size:
        er, ec = np.divmod(empty, W)
        ex, ey = grid.tr.xy(er, ec)
        ex, ey = np.asarray(ex, float), np.asarray(ey, float)
        if not _same_crs(grid.crs, a_crs):
            ex, ey = Transformer.from_crs(grid.crs, a_crs, always_xy=True).transform(ex, ey)
        cc, rr = ~a_transform * (ex, ey)
        good = np.isfinite(rr) & np.isfinite(cc)
        rr = np.where(good, np.floor(np.where(good, rr, 0)), -1).astype(int)
        cc = np.where(good, np.floor(np.where(good, cc, 0)), -1).astype(int)
        ok = (rr >= 0) & (rr < A.shape[0]) & (cc >= 0) & (cc < A.shape[1])
        tgt, rr, cc = empty[ok], rr[ok], cc[ok]
        tot[tgt] = 1
        v_ok = ~invalid[rr, cc]
        nval[tgt] = v_ok
        npos[tgt] = pos[rr, cc]
        vmax[tgt] = np.where(v_ok, V[rr, cc], -np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        fvalid = np.where(tot > 0, nval / np.maximum(tot, 1), 0.0)
        fpos = np.where(nval > 0, npos / np.maximum(nval, 1), 0.0)
    lab = np.full(n, float(fill), np.float32)
    keep_neg = (nval > 0) & (fvalid >= float(min_valid_fraction or 0.0))
    has_pos = npos > 0
    lab[keep_neg | has_pos] = vmax[keep_neg | has_pos]
    frac = np.where(keep_neg | has_pos, np.round(100 * fpos), 255).astype(np.uint8)
    return lab.reshape(H, W), frac.reshape(H, W), int(inside.sum())


def bucket_raster(src_path, grid: SceneGrid, band=1, nodata_values=None, positive_values=None,
                  min_valid_fraction=0.0, fill=DEFAULT_FILL, use_raster_nodata=True):
    """Read src_path (cropped to the scene) and max-bucket it onto grid."""
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError(f"{src_path}: reference raster has no CRS")
        nod = list(nodata_values if nodata_values is not None else [DEFAULT_FILL, -1])
        if use_raster_nodata and src.nodata is not None and not value_mask(
                np.array([src.nodata], float), positive_values or []).any():
            nod.append(float(src.nodata))
        win = _crop_window(src, grid)
        if win is None:
            lab = np.full((grid.height, grid.width), float(fill), np.float32)
            return lab, np.full(lab.shape, 255, np.uint8), 0
        A = src.read(band, window=win)
        tr = src.window_transform(win)
        crs = src.crs
    return bucket_max(A, tr, crs, grid, nod, positive_values, min_valid_fraction, fill)


def write_scene_raster(path, grid: SceneGrid, lab, frac, fill=DEFAULT_FILL, tags=None,
                       band_names=("label_bucket_max", "positive_fraction_percent")):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lab_ok = lab[lab != fill]
    ints = lab_ok.size == 0 or np.all(np.mod(lab_ok, 1) == 0)
    u8 = ints and (lab_ok.size == 0 or (lab_ok.min() >= 0 and lab_ok.max() < 255)) and 0 <= fill <= 255
    dtype = "uint8" if u8 else "float32"
    prof = grid.profile(2, dtype, fill)
    with warnings.catch_warnings():  # GCP scenes: no geotransform until the GCPs are set
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        dst = rasterio.open(path, "w", **prof)
    with dst:
        dst.write(lab.astype(dtype), 1)
        dst.write(frac.astype(dtype), 2)
        dst.set_band_description(1, band_names[0])
        dst.set_band_description(2, band_names[1])
        if grid.gcps:
            dst.gcps = (grid.gcps, grid.gcp_crs)
        dst.update_tags(**{k: str(v) for k, v in (tags or {}).items()})
    return path


# ---------------------------------------------------------------------------
# matchup generators
# ---------------------------------------------------------------------------

def scene_grid_settings(cfg: dict | None, default_positive=None) -> dict:
    g = dict(cfg or {})
    return {
        "enabled": bool(g.get("enabled", False)),
        "truth_dir": Path(g.get("truth_dir", "./scene_grid_truth")),
        "positive_values": g.get("positive_values", default_positive),
        "nodata_values": g.get("nodata_values", [DEFAULT_FILL, -1]),
        "min_valid_fraction": float(g.get("min_valid_fraction", 0.0)),
        "fill_value": int(g.get("fill_value", DEFAULT_FILL)),
        "band": int(g.get("band", 1)),
        "products": [str(p).lower() for p in (g.get("products") or [])],  # empty = all
        "overwrite": bool(g.get("overwrite", False)),
        "drop_empty": bool(g.get("drop_empty", True)),
    }


def bucket_matchups(matchups: list[dict], truth_field: str, scene_cfg: dict | None,
                    default_positive=None, log=LOG) -> tuple[list[dict], dict]:
    """
    Replace record[truth_field] by a scene-grid (max-bucketed) truth raster.

    Per record: <truth_dir>/<truth stem>__<scene stem>.tif. The native reference
    is kept in truth_native, annotation_raster points at the bucketed file, and
    scene_positive_pixels / scene_valid_pixels are added. Records whose scene
    has no valid reference pixel are dropped when drop_empty (default).
    """
    s = scene_grid_settings(scene_cfg, default_positive)
    if not s["enabled"]:
        return matchups, {"enabled": False}
    out, n_pos, n_drop, n_skip = [], 0, 0, 0
    grids: dict[str, SceneGrid] = {}
    for r in matchups:
        prod = str(r.get("product", "")).lower()
        if s["products"] and prod not in s["products"]:
            out.append(r)
            n_skip += 1
            continue
        scene, ref = str(r["sit_fuse_context_free"]), str(r[truth_field])
        dst = s["truth_dir"] / f"{Path(ref).stem}__{Path(scene).name.split('.')[0]}.tif"
        if s["overwrite"] or not dst.exists():
            grid = grids.get(scene) or grids.setdefault(scene, SceneGrid.from_path(scene))
            lab, frac, nfine = bucket_raster(ref, grid, s["band"], s["nodata_values"], s["positive_values"],
                                             s["min_valid_fraction"], s["fill_value"])
            write_scene_raster(dst, grid, lab, frac, s["fill_value"], tags={
                "resampling": "bucket_max", "source_reference": ref, "scene": scene,
                "positive_values": s["positive_values"], "min_valid_fraction": s["min_valid_fraction"],
                "fine_cells_used": nfine})
        else:
            with rasterio.open(dst) as d:
                lab = d.read(1).astype(np.float32)
        fill = float(s["fill_value"])
        valid = int((lab != fill).sum())
        npix = int(((lab != fill) & (lab > 0)).sum())
        if valid == 0 and s["drop_empty"]:
            n_drop += 1
            log.info("scene_grid: %s has no valid reference pixels on the scene grid; dropped",
                     Path(scene).name)
            continue
        r = dict(r)
        r["truth_native"] = ref
        r[truth_field] = str(dst.resolve())
        r["annotation_raster"] = str(dst.resolve())
        r["truth_resampling"] = "bucket_max"
        r["scene_positive_pixels"] = npix
        r["scene_valid_pixels"] = valid
        n_pos += npix
        out.append(r)
    n_b = sum(1 for r in out if r.get("truth_resampling") == "bucket_max")
    summary = {"enabled": True, "records": len(out), "bucketed": n_b, "dropped_no_valid": n_drop,
               "not_bucketed_other_product": n_skip, "positive_scene_pixels": n_pos,
               "positive_records": sum(1 for r in out if r.get("scene_positive_pixels", 0) > 0),
               "truth_dir": str(s["truth_dir"].resolve())}
    log.info("scene_grid (bucket max): %d records bucketed (%d other-product records unchanged), "
             "%d with positives, %d positive scene pixels, %d dropped (no valid reference), truth in %s",
             n_b, n_skip, summary["positive_records"], n_pos, n_drop, summary["truth_dir"])
    if n_b and not summary["positive_records"]:
        log.warning("scene_grid: no bucketed record has a positive pixel; check scene_grid.positive_values "
                    "against the reference values (e.g. FDCC binary 1, raw codes [[10, 15], [30, 35]])")
    return out, summary
