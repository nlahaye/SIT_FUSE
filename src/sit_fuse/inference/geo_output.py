"""
Georeferenced (".full_geo.tif") output for generate_output.py.

Produces the same files postprocessing/generate_cluster_geotiffs.py writes for
plain cluster maps (<cluster zarr>.full_geo.tif, values = label / 1000, zero
canvas sized max(reference, output), output placed at subset_inds), so
downstream scripts that consume "*.zarr.full_geo.tif" keep working without the
extra postprocessing step.

Reference grid per scene, first match wins:
  1. output.gtiff_data[i] (output.gtiff_data_train[i] for training files);
     same meaning as data.gtiff_data in the postprocessing config.
  2. the input file(s), if GDAL opens them with a geotransform or GCPs.
  3. lat/lon from data.geo_reader_type / data.geo_reader_kwargs: regular grids
     become an EPSG:4326 geotransform, swaths become a thinned GCP grid.
  4. none -> warning, only the existing .no_geo outputs are written.

Config (all optional, under output:):
  geo_output: false          enable
  gtiff_data: [...]          reference rasters aligned with data.files_test
  gtiff_data_train: [...]    same for data.files_train (generate_train_output)
  subset_inds: [...]         [r0, r1, c0, c1] (or []) per file, as in postprocessing;
  subset_inds_train: [...]   if absent, derived from data.reader_kwargs
                             start_line/end_line/start_sample/end_sample when the
                             reference is the full scene (cases 1 and 2)
  geo_n_gcps: 32             GCP grid size per axis for swath lat/lon
  latlon_order: auto         lat_lon | lon_lat | auto, for geo reader output
"""

import os
import warnings

import numpy as np
from osgeo import gdal, osr

from sit_fuse.utils import get_read_func


def geo_output_enabled(yml_conf):
    return bool((yml_conf.get("output") or {}).get("geo_output", False))


def _first_path(fname):
    while isinstance(fname, (list, tuple)):
        if len(fname) == 0:
            return None
        fname = fname[0]
    return fname


def _wkt_4326():
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    return srs.ExportToWkt()


def gdal_reference(path, require=False):
    """GeoTransform / projection / metadata / GCPs of a GDAL-readable raster, or None."""
    if path is None or not isinstance(path, str):
        return None
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        ds = gdal.Open(path)
    except Exception:
        ds = None
    finally:
        gdal.PopErrorHandler()
    if ds is None:
        if require:
            raise ValueError("Could not open reference raster %s" % path)
        return None
    gt = ds.GetGeoTransform(can_return_null=True)
    gcp_count = ds.GetGCPCount()
    if gt is None and gcp_count == 0:
        ds = None
        if require:
            raise ValueError("Reference raster has no geotransform or GCPs: %s" % path)
        return None
    ref = {"source": path, "nx": ds.RasterXSize, "ny": ds.RasterYSize,
           "geotransform": gt, "projection": ds.GetProjection(),
           "metadata": ds.GetMetadata(), "gcps": None, "gcp_projection": None,
           "full_scene": True}
    if gcp_count > 0:
        ref["gcps"] = ds.GetGCPs()
        ref["gcp_projection"] = ds.GetGCPProjection()
    ds = None
    return ref


def _split_latlon(ll, order="auto"):
    """SIT-FUSE geo readers return (2, H, W) or (H, W, 2), usually [lat, lon]."""
    ll = np.squeeze(np.asarray(ll, dtype=np.float64))
    if ll.ndim != 3 or 2 not in (ll.shape[0], ll.shape[-1]):
        return None
    a, b = (ll[0], ll[1]) if ll.shape[0] == 2 else (ll[..., 0], ll[..., 1])
    order = str(order).lower()
    if order == "auto":  # latitude never exceeds 90 degrees in magnitude
        order = "lon_lat" if np.nanmax(np.abs(a)) > 90.0 >= np.nanmax(np.abs(b)) else "lat_lon"
    return (a, b) if order == "lat_lon" else (b, a)


def latlon_reference(lat, lon, n_gcps=32, rtol=1e-3):
    """Regular lat/lon grid -> EPSG:4326 geotransform, otherwise a thinned GCP grid."""
    H, W = lat.shape
    good = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 360)
    if not good.any():
        return None
    wkt = _wkt_4326()
    ref = {"source": "geo_reader", "nx": W, "ny": H, "geotransform": None, "projection": wkt,
           "metadata": {}, "gcps": None, "gcp_projection": None, "full_scene": False}
    if good.all() and H > 1 and W > 1:
        dlat = np.diff(lat[:, 0])
        dlon = np.diff(lon[0, :])
        regular = (np.allclose(lat, lat[:, :1], rtol=0, atol=abs(dlat).mean() * rtol)
                   and np.allclose(lon, lon[:1, :], rtol=0, atol=abs(dlon).mean() * rtol)
                   and np.allclose(dlat, dlat.mean(), rtol=rtol)
                   and np.allclose(dlon, dlon.mean(), rtol=rtol))
        if regular:
            dy, dx = float(dlat.mean()), float(dlon.mean())
            ref["geotransform"] = (float(lon[0, 0]) - dx / 2.0, dx, 0.0, float(lat[0, 0]) - dy / 2.0, 0.0, dy)
            return ref

    rr = np.unique(np.linspace(0, H - 1, min(n_gcps, H)).round().astype(int))
    cc = np.unique(np.linspace(0, W - 1, min(n_gcps, W)).round().astype(int))
    gcps = [gdal.GCP(float(lon[r, c]), float(lat[r, c]), 0.0, c + 0.5, r + 0.5)
            for r in rr for c in cc if good[r, c]]
    if len(gcps) < 3:
        return None
    ref["projection"] = ""
    ref["gcps"] = gcps
    ref["gcp_projection"] = wkt
    return ref


def _geo_reader_reference(yml_conf, fname):
    data_conf = yml_conf.get("data") or {}
    reader = data_conf.get("geo_reader_type")
    if not reader:
        return None
    out_conf = yml_conf.get("output") or {}
    try:
        ll = get_read_func(reader)(fname, **(data_conf.get("geo_reader_kwargs") or {}))
    except Exception as exc:  # reader-specific failures fall through to no-geo
        warnings.warn("geo reader %s failed for %s: %s" % (reader, fname, exc))
        return None
    if ll is None:
        return None
    split = _split_latlon(ll, out_conf.get("latlon_order", "auto"))
    if split is None:
        warnings.warn("geo reader %s returned unexpected shape %s" % (reader, str(np.shape(ll))))
        return None
    ref = latlon_reference(*split, n_gcps=int(out_conf.get("geo_n_gcps", 32)))
    if ref is not None:
        ref["source"] = "geo_reader:%s" % reader
        # Geo readers subset with geo_reader_kwargs; if they don't but the data reader does,
        # the grid is the full scene and the subset offset applies.
        gkw = data_conf.get("geo_reader_kwargs") or {}
        ref["full_scene"] = "start_line" not in gkw and "start_sample" not in gkw
    return ref


def _aligned_entry(lst, index):
    if lst is None or index is None:
        return None
    if isinstance(lst, (list, tuple)):
        return lst[index] if index < len(lst) else None
    return lst


def _reader_subset(yml_conf):
    kw = (yml_conf.get("data") or {}).get("reader_kwargs") or {}
    if all(k in kw for k in ("start_line", "end_line", "start_sample", "end_sample")):
        return [int(kw["start_line"]), int(kw["end_line"]), int(kw["start_sample"]), int(kw["end_sample"])]
    return []


def resolve_reference(yml_conf, fname, index=None, train=False):
    """
    Returns (reference dict or None, subset_inds list) for one input file.
    index is the position of fname in data.files_test (or files_train when train=True).
    """
    out_conf = yml_conf.get("output") or {}
    sfx = "_train" if train else ""
    explicit = _aligned_entry(out_conf.get("gtiff_data" + sfx), index)
    subset = _aligned_entry(out_conf.get("subset_inds" + sfx), index)

    ref = None
    if explicit:
        ref = gdal_reference(explicit, require=True)
    if ref is None:
        cands = fname if isinstance(fname, (list, tuple)) else [fname]
        for cand in cands:
            ref = gdal_reference(_first_path(cand))
            if ref is not None:
                break
    if ref is None:
        ref = _geo_reader_reference(yml_conf, fname)

    if subset is None or len(subset) == 0:
        subset = _reader_subset(yml_conf) if (ref is not None and ref.get("full_scene", True)) else []
    return ref, list(subset)


def place_on_reference(arr, ref, subset_inds=None, fill=0.0):
    """Zero canvas of max(reference, output) size with arr placed at subset_inds (clipped)."""
    arr = np.asarray(arr)
    ny = max(int(ref["ny"]), arr.shape[0])
    nx = max(int(ref["nx"]), arr.shape[1])
    out = np.full((ny, nx), fill, dtype=np.float32)
    if subset_inds is not None and len(subset_inds) > 0:
        r0, r1, c0, c1 = [int(v) for v in subset_inds[:4]]
        h = max(0, min(r1 - r0, arr.shape[0], ny - r0))
        w = max(0, min(c1 - c0, arr.shape[1], nx - c0))
        out[r0:r0 + h, c0:c0 + w] = arr[:h, :w]
    else:
        out[0:arr.shape[0], 0:arr.shape[1]] = arr
    return out


def write_geotiff(fname, arr, ref, subset_inds=None, compress=None, fill=0.0):
    out = place_on_reference(arr, ref, subset_inds, fill)
    opts = ["BIGTIFF=IF_SAFER"]
    if compress:
        opts += ["COMPRESS=%s" % compress, "TILED=YES"]
    out_ds = gdal.GetDriverByName("GTiff").Create(fname, out.shape[1], out.shape[0], 1, gdal.GDT_Float32, options=opts)
    if ref.get("metadata"):
        out_ds.SetMetadata(ref["metadata"])
    if ref.get("geotransform") is not None:
        out_ds.SetGeoTransform(ref["geotransform"])
    if ref.get("projection"):
        out_ds.SetProjection(ref["projection"])
    if ref.get("gcps"):
        out_ds.SetGCPs(ref["gcps"], ref["gcp_projection"])
    out_ds.GetRasterBand(1).WriteArray(out)
    out_ds.FlushCache()
    out_ds = None
    return fname
