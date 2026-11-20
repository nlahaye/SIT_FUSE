"""
Probabilistic products overlaid on their observations (TEMPO RGB / UV RGB, GOES ABI true colour) for multi-panel
animations (make_overlay_animations.py).

Observations
  TempoRGB    TEMPO RGB GeoTIFFs named like the TEMPO granules plus a suffix (<granule>.RGB.tif,
              <granule>.UV_RGB.tif); granules of one scan are mosaicked.
  GoesRGB     GOES ABI multi-band GeoTIFFs (colocate_and_resample.py output, bands C01..C16 as L1b radiance):
              CIMSS natural true colour (R = C02, G = 0.45 C02 + 0.1 C03 + 0.45 C01, B = C01) from reflectance
              factors (radiance x kappa0), optional solar-zenith normalisation, gamma 2.2; inverted C13 at night.
Contrast      enhance_rgb: after the fixed stretch, the luminance is histogram-equalised with one CDF pooled over frames
              sampled across the whole animation (stable, no flicker), blended with the stretched luminance
              (strength), optionally unsharp-masked, and RGB rescaled by the luminance ratio (hue preserved).
              method clahe (local, per frame, scikit-image) and linear (stretch only) are alternatives.
Products      export_geo_products.py rasters (uq_viz.data.GeoProductIndex), probability band drawn with an alpha
              ramp (transparent at low probability) and the decision mask as a contour (and points for fire).
Everything is warped onto one display grid (EPSG:4326, bbox, resolution). The frame plan holds only paths and
times, so frames render in worker processes.
"""

from __future__ import annotations

import glob
import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds

LOG = logging.getLogger("uq-overlay")

GRANULE_RE = re.compile(r"(?P<ts>\d{8}T\d{6})Z?_S(?P<scan>\d{3})G(?P<gran>\d{2})")
TS_RE = re.compile(r"(\d{8}T\d{6})Z?")
ABI_RE = re.compile(r"_s(?P<s>\d{13})\d?_e")
EPSG4326 = CRS.from_epsg(4326)


# ---------------------------------------------------------------------------
# names and times
# ---------------------------------------------------------------------------

def time_from_name(name: str) -> pd.Timestamp | None:
    """Scene start time (UTC) from a TEMPO (YYYYmmddTHHMMSS) or GOES ABI (_sYYYYJJJHHMMSS) file name."""
    name = Path(str(name)).name
    m = ABI_RE.search(name)
    if m:
        return pd.Timestamp(pd.to_datetime(m.group("s"), format="%Y%j%H%M%S"), tz="UTC")
    m = TS_RE.search(name)
    if m:
        return pd.Timestamp(pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S"), tz="UTC")
    return None


def granule_key(name: str) -> str | None:
    m = GRANULE_RE.search(Path(str(name)).name)
    return f"{m.group('ts')}_S{m.group('scan')}G{m.group('gran')}" if m else None


def tempo_scan_id(name: str, t: pd.Timestamp, day_shift_hours: float = -6.0) -> str | None:
    """One TEMPO E-W sweep: scan number on the observing day (UTC shifted so evening scans stay on their day)."""
    m = GRANULE_RE.search(Path(str(name)).name)
    if not m or t is None:
        return None
    day = (t + pd.Timedelta(hours=day_shift_hours)).strftime("%Y-%m-%d")
    return f"{day}_S{m.group('scan')}"


def utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")


# ---------------------------------------------------------------------------
# display grid and warping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Grid:
    west: float
    east: float
    south: float
    north: float
    res: float

    @property
    def width(self) -> int:
        return int(np.ceil((self.east - self.west) / self.res))

    @property
    def height(self) -> int:
        return int(np.ceil((self.north - self.south) / self.res))

    @property
    def transform(self):
        return from_origin(self.west, self.north, self.res, self.res)

    @property
    def extent(self):            # imshow / setup_map order
        return [self.west, self.west + self.width * self.res, self.north - self.height * self.res, self.north]

    def lonlat(self):
        lon = self.west + (np.arange(self.width) + 0.5) * self.res
        lat = self.north - (np.arange(self.height) + 0.5) * self.res
        return np.meshgrid(lon, lat)


def file_bounds_4326(path: str):
    """(W, S, E, N) in EPSG:4326, or None when the file has no usable georeferencing. Works for south-up grids
    (positive row step), whose rasterio bounds have bottom > top."""
    try:
        with rasterio.open(path) as s:
            if s.crs is not None and not s.transform.is_identity:
                l, b, r, t = s.bounds
                return tuple(transform_bounds(s.crs, EPSG4326, min(l, r), min(b, t), max(l, r), max(b, t),
                                              densify_pts=21))
            gcps, gcrs = s.gcps
            if gcps:
                xs, ys = [g.x for g in gcps], [g.y for g in gcps]
                return tuple(transform_bounds(gcrs or EPSG4326, EPSG4326, min(xs), min(ys), max(xs), max(ys)))
    except rasterio.errors.RasterioIOError as exc:
        LOG.warning("cannot open %s (%s)", path, exc)
    return None


def intersects(b, grid: Grid) -> bool:
    return b is not None and b[0] < grid.east and b[2] > grid.west and b[1] < grid.north and b[3] > grid.south


def _root_error(e: BaseException) -> str:
    """Innermost cause of a (rasterio) exception chain, e.g. 'ZIPDecode:Decoding error at scanline 0'."""
    while e.__cause__ is not None:
        e = e.__cause__
    return str(e).splitlines()[0] if str(e) else type(e).__name__


FILL_ABS = 1e20                          # |value| above this is fill (TEMPO RGBs are written without a nodata tag)
UNREADABLE: dict[str, str] = {}          # path -> error (corrupt / truncated files, drawn as no data)


@lru_cache(maxsize=96)
def read_to_grid(path: str, bands: tuple, grid: Grid, resampling: str = "bilinear",
                 valid_min: float | None = None, valid_max: float | None = None) -> np.ndarray:
    """Bands (1-based) of `path` warped onto `grid`; float32 (n, H, W), NaN where no data. A file that cannot be
    decoded (corrupt or truncated GeoTIFF) is logged once and returned as no data instead of stopping the run."""
    from rasterio.errors import RasterioError
    try:
        return _read_to_grid(path, bands, grid, resampling, valid_min, valid_max)
    except (RasterioError, OSError) as e:
        if path not in UNREADABLE:
            UNREADABLE[path] = _root_error(e)[:200]
            LOG.warning("unreadable raster drawn as no data (%s): %s", UNREADABLE[path], path)
        return np.full((len(bands), grid.height, grid.width), np.nan, np.float32)


def _read_to_grid(path, bands, grid, resampling, valid_min, valid_max):
    rs = getattr(Resampling, resampling)
    out = np.full((len(bands), grid.height, grid.width), np.nan, np.float32)
    with rasterio.open(path) as s:
        nd = s.nodata
        gcps, gcrs = s.gcps
        alpha = None
        ci = list(s.colorinterp)
        if 4 not in bands and s.count >= 4 and ci[3] == rasterio.enums.ColorInterp.alpha:
            alpha = 4
        win, kw = None, {}
        if s.crs is not None and not s.transform.is_identity:
            try:                                    # read only the part of the file around the display box
                b = transform_bounds(EPSG4326, s.crs, grid.west, grid.south, grid.east, grid.north, densify_pts=21)
                pad = 4 * max(abs(s.transform.a), abs(s.transform.e))
                # pixel rows/cols of the padded box corners; min/max works for north-up and south-up (positive
                # row step, e.g. latitude-ascending colocated TEMPO grids) and flipped-x transforms alike
                inv = ~s.transform
                cr = [inv * (x, y) for x in (b[0] - pad, b[2] + pad) for y in (b[1] - pad, b[3] + pad)]
                c0 = max(int(np.floor(min(c for c, _ in cr))), 0)
                r0 = max(int(np.floor(min(r for _, r in cr))), 0)
                c1 = min(int(np.ceil(max(c for c, _ in cr))), s.width)
                r1 = min(int(np.ceil(max(r for _, r in cr))), s.height)
                if c1 <= c0 or r1 <= r0:
                    return out
                win = Window(c0, r0, c1 - c0, r1 - r0)
            except Exception:  # noqa: BLE001  (degenerate window)
                return out
            kw = dict(src_transform=s.window_transform(win), src_crs=s.crs)
        elif gcps:
            kw = dict(gcps=gcps, src_crs=gcrs or EPSG4326)
        else:
            LOG.warning("%s: no georeferencing; skipped", path)
            return out
        src = s.read(list(bands), window=win).astype(np.float32)
        am = s.read(alpha, window=win) if alpha else None
    bad = ~np.isfinite(src) | (np.abs(src) > FILL_ABS)       # fill values (-1e30, 9.97e36) when nodata is unset
    if nd is not None and np.isfinite(nd):
        bad |= src == nd
    if am is not None:
        bad |= (am == 0)[None]
    if valid_min is not None:
        bad |= src < valid_min
    if valid_max is not None:
        bad |= src > valid_max
    src[bad] = np.nan
    for i in range(len(bands)):
        reproject(src[i], out[i], dst_transform=grid.transform, dst_crs=EPSG4326, resampling=rs,
                  src_nodata=np.nan, dst_nodata=np.nan, **kw)
    return out


def mosaic(paths, bands, grid, resampling="bilinear", **kw) -> np.ndarray | None:
    """First-valid mosaic (granules of one TEMPO scan do not overlap)."""
    out = None
    for p in paths:
        a = read_to_grid(str(p), tuple(bands), grid, resampling, **kw)
        out = a.copy() if out is None else np.where(np.isnan(out), a, out)
    return out


# ---------------------------------------------------------------------------
# solar geometry (NOAA general solar position, ~0.5 deg)
# ---------------------------------------------------------------------------

def cos_solar_zenith(lon, lat, t: pd.Timestamp) -> np.ndarray:
    t = utc(t)
    hour = t.hour + t.minute / 60 + t.second / 3600
    g = 2 * np.pi / 365 * (t.dayofyear - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                    - 0.040849 * np.sin(2 * g))
    ha = np.deg2rad((hour * 60 + eqt + 4 * lon) / 4 - 180)
    la = np.deg2rad(lat)
    return np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)


# ---------------------------------------------------------------------------
# observations
# ---------------------------------------------------------------------------

def _list(d, pattern, recursive=False):
    ds = d if isinstance(d, (list, tuple)) else [d]
    out = []
    for x in ds:
        out += glob.glob(str(Path(x) / ("**" if recursive else "") / pattern), recursive=recursive)
    return sorted(set(out))


def stretch_limits(paths, bands, grid, pct=(1.0, 99.0), n_sample=12, **kw):
    """Per-band display limits from a sample of files (fixed over the animation, so frames do not flicker)."""
    if not paths:
        return None
    idx = np.unique(np.linspace(0, len(paths) - 1, min(n_sample, len(paths))).round().astype(int))
    vals = [[] for _ in bands]
    for i in idx:
        a = read_to_grid(str(paths[i]), tuple(bands), grid, "bilinear", **kw)
        for b in range(len(bands)):
            v = a[b][np.isfinite(a[b])]
            if v.size:
                vals[b].append(np.random.default_rng(0).choice(v, min(v.size, 20000), replace=False))
    lims = []
    for v in vals:
        v = np.concatenate(v) if v else np.array([0.0, 1.0])
        lo, hi = np.percentile(v, pct)
        lims.append((float(lo), float(hi if hi > lo else lo + 1)))
    return lims


class TempoRGB:
    """TEMPO RGB composites (<granule><suffix>), indexed by granule and scan."""

    def __init__(self, name: str, cfg: dict, grid: Grid, t0, t1, day_shift_hours=-6.0):
        self.name, self.cfg, self.grid = name, cfg, grid
        suffix = cfg.get("suffix", ".UV_RGB.tif" if "uv" in name.lower() else ".RGB.tif")
        pattern = cfg.get("pattern", f"*{suffix}")
        rows = []
        for p in _list(cfg["dir"], pattern, cfg.get("recursive", False)):
            if not p.endswith(suffix):
                continue
            if suffix == ".RGB.tif" and p.endswith(".UV_RGB.tif"):
                continue
            t = time_from_name(p)
            if t is None or not (t0 <= t < t1):
                continue
            b = file_bounds_4326(p)
            if not intersects(b, grid):
                continue
            rows.append({"path": p, "time": t, "granule": granule_key(p),
                         "frame": tempo_scan_id(p, t, day_shift_hours)})
        self.df = pd.DataFrame(rows, columns=["path", "time", "granule", "frame"])
        self.bands = tuple(cfg.get("bands", [1, 2, 3]))
        self.gamma = float(cfg.get("gamma", 1.0))
        self.limits = None
        self.valid = (None, None)
        if len(self.df):
            with rasterio.open(self.df.path.iloc[0]) as s:
                self.dtype = s.dtypes[0]
            # float radiance RGBs carry no nodata tag: values <= 0 (fill outside a granule's swath on the colocated
            # grid) and |x| > 1e20 are no data, so the scan mosaic is not covered by one granule's fill and the
            # stretch is not pulled to 0. valid_min / valid_max: null disables.
            dflt = (None, None) if self.dtype == "uint8" else (1e-30, FILL_ABS)
            vmin, vmax = cfg.get("valid_min", dflt[0]), cfg.get("valid_max", dflt[1])
            self.valid = (None if vmin is None else float(vmin), None if vmax is None else float(vmax))
            st = cfg.get("stretch", "auto")
            if isinstance(st, (list, tuple)):
                self.limits = [tuple(st)] * len(self.bands) if np.ndim(st) == 1 else [tuple(x) for x in st]
            elif st == "percentile" or (st == "auto" and self.dtype != "uint8"):   # uint8 auto: 0..255
                self.limits = stretch_limits(list(self.df.sort_values("time").path), self.bands, grid,
                                             tuple(cfg.get("percentiles", [1.0, 99.0])),
                                             valid_min=self.valid[0], valid_max=self.valid[1])
            elif self.dtype == "uint8":
                self.limits = [(0.0, 255.0)] * len(self.bands)
        LOG.info("%s: %d granules in range and box; dtype %s, valid range %s, stretch limits %s", name, len(self.df),
                 getattr(self, "dtype", None), self.valid,
                 [tuple(float(f"{v:.4g}") for v in lh) for lh in self.limits] if self.limits else None)

    def spec(self, frame_id):
        d = self.df[self.df.frame == frame_id].sort_values("time")
        if d.empty:
            return None
        return {"kind": "tempo_rgb", "name": self.name, "paths": list(d.path), "time": d.time.median(),
                "bands": self.bands, "limits": self.limits, "gamma": self.gamma, "valid": self.valid,
                "enhance": getattr(self, "enhance", None)}


class GoesRGB:
    """GOES ABI natural true colour from the multi-band GeoTIFFs."""

    def __init__(self, name: str, cfg: dict, grid: Grid, t0, t1):
        self.name, self.cfg, self.grid = name, cfg, grid
        pattern = cfg.get("pattern", "OR_ABI-*.tif")
        excl = tuple(cfg.get("exclude", [".head_outputs.tif", "full_geo", ".clust"]))
        rows = []
        for p in _list(cfg["dir"], pattern, cfg.get("recursive", False)):
            if any(e in Path(p).name for e in excl):
                continue
            t = time_from_name(p)
            if t is None or not (t0 <= t < t1):
                continue
            rows.append({"path": p, "time": t})
        df = pd.DataFrame(rows, columns=["path", "time"]).sort_values("time")
        if len(df) and not intersects(file_bounds_4326(df.path.iloc[0]), grid):
            LOG.warning("%s: first GOES file does not cover the box", name)
        self.df = df.reset_index(drop=True)
        b = dict({"blue": 1, "red": 2, "veggie": 3, "ir": 13}, **(cfg.get("bands") or {}))
        self.bands = (b["blue"], b["red"], b["veggie"])
        self.ir_band = b.get("ir")
        self.valid = (float(cfg.get("valid_min", -5.0)), float(cfg.get("valid_max", 5000.0)))
        self.kappa_source = "config" if isinstance(cfg.get("kappa0"), dict) else None
        self.kappa = self._kappa(cfg)
        self.night = cfg.get("night", "ir")
        self.ir_limits = None
        if self.night == "ir" and self.ir_band and len(self.df):
            self.ir_limits = stretch_limits(list(self.df.path), (self.ir_band,), grid, (1.0, 99.5))[0]
        LOG.info("%s: %d GOES scenes in range; reflectance scale %s", name, len(self.df),
                 {k: round(v, 6) for k, v in self.kappa.items()} if isinstance(self.kappa, dict) else self.kappa)

    def _kappa(self, cfg):
        """kappa0 (reflectance factor per unit radiance) per band: given, read from the L1b NetCDFs, or estimated."""
        k = cfg.get("kappa0")
        if isinstance(k, dict):
            return {int(b): float(k[n]) for n, b in zip(("blue", "red", "veggie"), self.bands)}
        if cfg.get("kappa0_from") and len(self.df):
            got = _kappa_from_netcdf(cfg["kappa0_from"], self.df.path.iloc[len(self.df) // 2], self.bands,
                                     cfg.get("recursive", False))
            if got:
                self.kappa_source = "netcdf"
                return got
            LOG.warning("%s: no kappa0 in %s; using the image-based scale", self.name, cfg["kappa0_from"])
        if not len(self.df):
            return {}
        self.kappa_source = "image"
        # image-based: the 99.5th percentile radiance of the daytime sample maps to reflectance 1.0 (display only)
        day = self.df[(self.df.time.dt.hour >= 16) & (self.df.time.dt.hour <= 23)]
        lims = stretch_limits(list((day if len(day) else self.df).path), self.bands, self.grid, (1.0, 99.5),
                              valid_min=self.valid[0], valid_max=self.valid[1])
        return {int(b): 1.0 / max(hi, 1e-6) for b, (_, hi) in zip(self.bands, lims)}

    def nearest(self, t, window_minutes):
        if self.df.empty:
            return None
        dt = (self.df.time - t).abs()
        i = int(dt.values.argmin())
        if dt.iloc[i] > pd.Timedelta(minutes=window_minutes):
            return None
        return self.df.iloc[i]

    def spec_for(self, row):
        if row is None:
            return None
        return {"kind": "goes_rgb", "name": self.name, "paths": [row.path], "time": row.time, "bands": self.bands,
                "kappa": [self.kappa[int(b)] for b in self.bands], "valid": self.valid,
                "sza_correction": bool(self.cfg.get("sza_correction", self.kappa_source != "image")),
                "gamma": float(self.cfg.get("gamma", 2.2)), "night": self.night, "ir_band": self.ir_band,
                "ir_limits": self.ir_limits, "min_cos_sza": float(self.cfg.get("min_cos_sza", 0.1)),
                "enhance": getattr(self, "enhance", None)}


def _kappa_from_netcdf(nc_dir, tif_path, bands, recursive=False):
    """kappa0 from the per-band L1b NetCDFs of the same scene (or the nearest one with all three bands)."""
    try:
        from netCDF4 import Dataset
    except ImportError:
        return None
    name = Path(tif_path).name
    m = re.search(r"OR_ABI-(?P<prod>[A-Za-z0-9]+-[A-Za-z0-9]+)-M(?P<mode>\d)C\d{2}_(?P<sat>G\d{2})_s(?P<s>\d{14})", name)
    if not m:
        return None
    out = {}
    for b in bands:
        pat = f"OR_ABI-{m.group('prod')}-M{m.group('mode')}C{int(b):02d}_{m.group('sat')}_s{m.group('s')}_*.nc"
        hits = _list(nc_dir, pat, recursive) or _list(nc_dir, f"OR_ABI-{m.group('prod')}-M*C{int(b):02d}_"
                                                             f"{m.group('sat')}_s{m.group('s')[:7]}*.nc", recursive)
        if not hits:
            return None
        with Dataset(hits[0]) as f:
            out[int(b)] = float(np.asarray(f.variables["kappa0"][:]).ravel()[0])
    return out


# ---------------------------------------------------------------------------
# products
# ---------------------------------------------------------------------------

def product_sensor(name: str) -> str:
    n = name.lower()
    if "fus" in n:
        return "fused"
    return "goes" if "goes" in n or "abi" in n else "tempo"


class ProductSource:
    """One exported product (sensor x product x model) from export_geo_products.py."""

    def __init__(self, name: str, cfg: dict, t0, t1, grid: Grid, day_shift_hours=-6.0):
        from uq_viz.data import GeoProductIndex
        self.name, self.cfg = name, cfg
        self.product = cfg.get("product", "fire" if "fire" in name else "smoke")
        self.sensor = cfg.get("sensor", product_sensor(name))   # tempo | goes | fused (fused: GOES-like timing)
        self.model = cfg.get("model", "fused" if self.sensor == "fused" else "hist_gradient_boosting")
        self.style = cfg.get("style", self.product)            # DEFAULT_STYLE / style: entry used to draw it
        self.layer = cfg.get("layer", "p_1")
        self.mask_layer = cfg.get("mask_layer", "pred_label")
        idx = GeoProductIndex(cfg["geo_dir"], self.product, [self.model], cfg.get("manifest"))
        df = idx.df[idx.df.model == self.model].copy()
        df["time"] = [t if pd.notna(t) else time_from_name(p) for t, p in zip(df.scene_time, df.pair_id)]
        df = df[(df.time >= t0) & (df.time < t1)]
        keep = [intersects(file_bounds_4326(p), grid) for p in df.path]
        df = df[np.asarray(keep, bool)] if len(df) else df
        df["granule"] = [granule_key(p) for p in df.pair_id]
        df["frame"] = [tempo_scan_id(p, t, day_shift_hours) for p, t in zip(df.pair_id, df.time)] \
            if self.sensor == "tempo" else None
        self.df = df.sort_values("time").reset_index(drop=True)
        LOG.info("product %s (%s %s, %s): %d scenes in range and box", name, self.sensor, self.product,
                 self.model, len(self.df))

    def _spec(self, d):
        if d.empty:
            return None
        bands = list(d.bands.iloc[0])
        if self.layer not in bands:
            raise KeyError(f"{self.name}: band {self.layer!r} not in {bands}")
        return {"kind": "product", "name": self.name, "product": self.style, "paths": list(d.path),
                "time": d.time.median(), "band": bands.index(self.layer) + 1,
                "mask_band": bands.index(self.mask_layer) + 1 if self.mask_layer in bands else None,
                "threshold": float(d.threshold.iloc[0]) if np.isfinite(d.threshold.iloc[0]) else None}

    def for_tempo_frame(self, frame_id):
        return self._spec(self.df[self.df.frame == frame_id])

    def nearest(self, t, window_minutes, causal=False, max_age_minutes=None):
        """GOES / fused: the scene nearest t. TEMPO: the scan nearest t (causal = latest scan at or before t)."""
        if self.df.empty:
            return None
        if self.sensor == "tempo":
            ft = self.df.groupby("frame").time.median()
            dt = (t - ft) if causal else (ft - t).abs()
            dt = dt[dt >= pd.Timedelta(0)] if causal else dt
            lim = pd.Timedelta(minutes=max_age_minutes if max_age_minutes is not None else window_minutes)
            dt = dt[dt <= lim]
            return None if dt.empty else self.for_tempo_frame(dt.idxmin())
        dt = (self.df.time - t).abs()
        i = int(dt.values.argmin())
        if dt.iloc[i] > pd.Timedelta(minutes=window_minutes):
            return None
        return self._spec(self.df.iloc[[i]])


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

DEFAULT_STYLE = {
    "smoke": {"cmap": "cool", "alpha_max": 0.65, "p_min": 0.15, "p_max": 0.9, "alpha_gamma": 1.0,
              "contour": True, "contour_color": "#ff00ff", "points": False, "label": "P(smoke)"},
    "fire": {"cmap": "autumn_r", "alpha_max": 0.9, "p_min": 0.05, "p_max": 0.7, "alpha_gamma": 0.7,
             "contour": True, "contour_color": "#ff2a00", "points": True, "point_color": "#ff2a00",
             "label": "P(fire)"},
}


# ---------------------------------------------------------------------------------------------------------------
# RGB contrast enhancement (luminance only, hue preserved; fixed per animation so frames do not flicker)
# ---------------------------------------------------------------------------------------------------------------
ENHANCE_DEFAULT = {
    "tempo": {"method": "hist", "strength": 0.5, "sharpen": 0.5, "sharpen_sigma": 1.0},
    "goes": {"method": "hist", "strength": 0.3, "sharpen": 0.4, "sharpen_sigma": 1.0},
}
_CLAHE_WARNED = []


def enhance_config(sensor: str, global_cfg=None, obs_cfg=None) -> dict:
    """Default for the sensor, updated by animation-level `rgb_enhance` and observation-level `enhance`.
    method: linear (stretch only) | hist (global histogram equalisation, CDF from frames sampled across the whole
    animation) | clahe (local, per frame; needs scikit-image). strength blends equalised and stretched luminance."""
    e = dict(ENHANCE_DEFAULT.get(sensor, ENHANCE_DEFAULT["tempo"]))
    for c in (global_cfg, obs_cfg):
        if isinstance(c, str):
            c = {"method": c}
        if c is False:
            c = {"method": "linear", "sharpen": 0.0}
        e.update(c or {})
    e["method"] = str(e.get("method", "linear")).lower()
    return e


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def hist_lut(lums, n_quantiles: int = 257, max_per_frame: int = 200_000, seed: int = 0) -> list | None:
    """Quantiles of display luminance pooled over sampled frames (the equalisation CDF)."""
    rng = np.random.default_rng(seed)
    v = []
    for y in lums:
        y = y[np.isfinite(y)]
        if y.size > max_per_frame:
            y = rng.choice(y, max_per_frame, replace=False)
        v.append(y)
    v = np.concatenate(v) if v else np.empty(0)
    if v.size < 100:
        return None
    return [float(x) for x in np.quantile(v, np.linspace(0, 1, n_quantiles))]


def enhance_rgb(rgb: np.ndarray, en: dict | None) -> np.ndarray:
    """Equalise / sharpen the luminance of an (H, W, 3) 0..1 image (NaN = no data) and rescale RGB by the luminance
    ratio, capped so no channel saturates (hue preserved)."""
    if not en:
        return rgb
    m, s = en.get("method", "linear"), float(en.get("strength", 0.5))
    sharp = float(en.get("sharpen", 0.0) or 0.0)
    if m in ("linear", "none") and sharp <= 0:
        return rgb
    valid = np.all(np.isfinite(rgb), -1)
    if not valid.any():
        return rgb
    x = np.clip(np.nan_to_num(rgb), 0, 1)
    y = luminance(x)
    ye = y
    if m == "hist" and en.get("lut"):
        q = np.asarray(en["lut"], float)
        q = np.maximum.accumulate(q) + np.arange(q.size) * 1e-9       # strictly increasing for interp
        ye = np.interp(y, q, np.linspace(0, 1, q.size))
    elif m == "clahe":
        try:
            from skimage.exposure import equalize_adapthist
            k = max(8, int(min(y.shape) * float(en.get("kernel_frac", 0.125))))
            ye = equalize_adapthist(np.where(valid, y, 0.0), kernel_size=k,
                                    clip_limit=float(en.get("clip_limit", 0.01)))
        except ImportError:
            if not _CLAHE_WARNED:
                _CLAHE_WARNED.append(1)
                import logging
                logging.getLogger(__name__).warning("rgb enhance: clahe needs scikit-image; using linear stretch")
    yn = (1 - s) * y + s * ye if m in ("hist", "clahe") else y
    if sharp > 0:                                                       # unsharp mask, NaN-aware
        from scipy.ndimage import gaussian_filter
        sig = float(en.get("sharpen_sigma", 1.0))
        w = valid.astype(float)
        blur = gaussian_filter(yn * w, sig) / np.maximum(gaussian_filter(w, sig), 1e-6)
        yn = yn + sharp * (yn - blur)
    yn = np.clip(yn, 0, 1)
    scale = yn / np.maximum(y, 1e-4)
    scale = np.minimum(scale, 1.0 / np.maximum(x.max(-1), 1e-4))        # do not clip channels (keeps hue)
    out = np.where((y > 1e-4)[..., None], x * scale[..., None], yn[..., None])
    out = np.clip(out, 0, 1)
    out[~valid] = np.nan
    return out


def obs_rgb(spec: dict, grid: Grid, enhance: dict | None = "spec") -> np.ndarray | None:
    """(H, W, 4) RGBA in 0..1 for an observation spec; alpha 0 where no data. enhance: dict, None (stretch only)
    or "spec" (use spec['enhance'])."""
    if spec is None:
        return None
    en = spec.get("enhance") if isinstance(enhance, str) else enhance
    if spec["kind"] == "tempo_rgb":
        vmin, vmax = spec.get("valid") or (None, None)
        a = mosaic(spec["paths"], spec["bands"], grid, valid_min=vmin, valid_max=vmax)
        if a is None:
            return None
        lims = spec["limits"] or [(np.nanmin(x), np.nanmax(x)) for x in a]
        rgb = np.stack([(x - lo) / (hi - lo) for x, (lo, hi) in zip(a, lims)], -1)
        rgb = enhance_rgb(np.clip(rgb, 0, 1) ** (1.0 / spec["gamma"]), en)
    else:
        a = read_to_grid(spec["paths"][0], tuple(spec["bands"]), grid, "bilinear", *spec["valid"])
        refl = [x * k for x, k in zip(a, spec["kappa"])]               # blue, red, veggie reflectance factors
        lon, lat = grid.lonlat()
        mu = cos_solar_zenith(lon, lat, spec["time"])
        if spec["sza_correction"]:
            refl = [x / np.maximum(mu, spec["min_cos_sza"]) for x in refl]
        b, r, n = (np.clip(x, 0, 1) for x in refl)
        g = np.clip(0.45 * r + 0.1 * n + 0.45 * b, 0, 1)
        rgb = enhance_rgb(np.stack([r, g, b], -1) ** (1.0 / spec["gamma"]), en)   # daytime part, before IR blend
        if spec["night"] == "ir" and spec["ir_band"] and spec["ir_limits"]:
            ir = read_to_grid(spec["paths"][0], (spec["ir_band"],), grid, "bilinear")[0]
            lo, hi = spec["ir_limits"]
            gray = 1.0 - np.clip((ir - lo) / (hi - lo), 0, 1)                  # cold cloud tops bright
            w = np.clip((mu - (-0.05)) / 0.15, 0, 1)[..., None]                # day weight across the terminator
            rgb = w * rgb + (1 - w) * np.repeat(gray[..., None], 3, -1)
    valid = np.all(np.isfinite(rgb), -1)
    rgba = np.concatenate([np.nan_to_num(rgb), valid[..., None].astype(float)], -1)
    return rgba


def product_layers(spec: dict, grid: Grid):
    """Probability and decision mask of a product spec on the display grid."""
    if spec is None:
        return None, None
    p = mosaic(spec["paths"], [spec["band"]], grid, "bilinear")[0]
    m = mosaic(spec["paths"], [spec["mask_band"]], grid, "nearest")[0] if spec.get("mask_band") else None
    if m is None and spec.get("threshold") is not None:
        m = np.where(np.isfinite(p), (p >= spec["threshold"]).astype(np.float32), np.nan)
    return p, m


def overlay_rgba(p: np.ndarray, st: dict):
    """See-through overlay: colour from the colormap, alpha rising from 0 at p_min to alpha_max at p_max."""
    import matplotlib
    cmap = matplotlib.colormaps[st["cmap"]]
    lo, hi = float(st["p_min"]), float(st["p_max"])
    q = np.clip((np.nan_to_num(p, nan=0.0) - lo) / max(hi - lo, 1e-6), 0, 1)
    rgba = cmap(q)
    rgba[..., 3] = float(st["alpha_max"]) * q ** float(st.get("alpha_gamma", 1.0)) * (np.nan_to_num(p) >= lo)
    return rgba


def fmt_dt(a, b):
    if a is None or b is None:
        return ""
    m = (b - a).total_seconds() / 60
    return f"{m:+.0f} min"


def render_frame(fr: dict, static: dict, out_png: str) -> str:
    """One animation frame: panels (observation + see-through product overlays), colourbars, time stamp."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from uq_viz import plotting as P

    grid: Grid = static["grid"]
    styles = static["styles"]
    panels = fr["panels"]
    ncol = int(static.get("ncols") or len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    ext = grid.extent
    aspect = (ext[3] - ext[2]) / np.cos(np.deg2rad(0.5 * (ext[2] + ext[3]))) / (ext[1] - ext[0])
    pw = float(static.get("panel_width", 3.6))
    kinds = sorted({o["product"] for pn in panels for o in (pn or {}).get("overlays", []) if o}, key=lambda k: (k != "smoke", k != "fire", k))
    cb_h = 0.55 if kinds else 0.0
    aw = 0.86 * pw                                       # drawn map width (panel width less margins and spacing)
    row_h = aw * aspect + 0.45
    fig = plt.figure(figsize=(pw * ncol + 0.4, nrow * row_h + cb_h + 0.6))
    top = 1 - 0.6 / fig.get_figheight()
    gs = fig.add_gridspec(nrow + (1 if kinds else 0), ncol,
                          height_ratios=[row_h] * nrow + ([cb_h] if kinds else []),
                          hspace=0.14, wspace=0.24, left=0.06, right=0.99, top=top, bottom=0.04)
    lon, lat = None, None
    for j, pn in enumerate(panels):
        if not pn or pn.get("blank"):                   # layout placeholder (keeps the RGB row aligned)
            continue
        ax = fig.add_subplot(gs[j // ncol, j % ncol])
        P.setup_map(ax, ext, static.get("basemap"), ticks=static.get("ticks", True))
        if static.get("ticks", True):                    # decimals that resolve the box (setup_map shows whole degrees)
            span = min(ext[1] - ext[0], ext[3] - ext[2])
            nd = 0 if span >= 8 else 1 if span >= 2 else 2
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):.{nd}f}°{'W' if v < 0 else 'E'}"))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):.{nd}f}°{'N' if v >= 0 else 'S'}"))
            ax.tick_params(labelsize=6.5)
        ax.set_facecolor(static.get("background", "#202020"))
        obs = obs_rgb(pn["obs"], grid) if pn.get("obs") else None
        if obs is not None:
            ax.imshow(obs, extent=ext, origin="upper", interpolation=static.get("rgb_interpolation", "antialiased"),
                      zorder=2)
        notes = []
        if pn.get("obs"):                               # age of a held observation (e.g. TEMPO under GOES frames)
            age = (fr["time"] - pn["obs"]["time"]).total_seconds() / 60
            notes.append(f"obs {pn['obs']['time']:%H:%M}" + (f" (held {age:.0f} min)" if age >= 1 else
                                                             f" ({-age:+.0f} min)" if age <= -1 else ""))
        elif pn.get("obs_name"):
            notes.append("no observation")
        for o in pn["overlays"]:
            if o is None:
                continue
            st = styles[o["product"]]
            p, m = product_layers(o, grid)
            if p is None:
                continue
            ax.imshow(overlay_rgba(p, st), extent=ext, origin="upper", interpolation="nearest", zorder=3)
            if m is not None and np.nanmax(np.nan_to_num(m)) >= 1:
                x = grid.west + (np.arange(grid.width) + 0.5) * grid.res
                y = grid.north - (np.arange(grid.height) + 0.5) * grid.res
                if st.get("contour", True):
                    ax.contour(x, y, np.nan_to_num(m), levels=[0.5], colors=[st["contour_color"]],
                               linewidths=float(st.get("contour_width", 0.7)), zorder=5)
                if st.get("points"):
                    if lon is None:
                        lon, lat = grid.lonlat()
                    k = np.nan_to_num(m) >= 1
                    ax.scatter(lon[k], lat[k], s=float(st.get("point_size", 4)), c=st.get("point_color", "red"),
                               marker="^", linewidths=0, zorder=6)
            ref = pn["obs"]["time"] if pn.get("obs") else fr["time"]
            notes.append(f"{o['name']} {o['time']:%H:%M}" + (f" ({fmt_dt(ref, o['time'])})"
                                                              if abs((o["time"] - ref).total_seconds()) >= 60 else ""))
        for o in pn.get("missing", []):
            notes.append(f"{o}: none")
        ax.set_title(pn["title"], fontsize=8.5)
        ax.text(0.01, 0.01, " | ".join(notes), transform=ax.transAxes, fontsize=6, color="white", va="bottom",
                zorder=10, bbox=dict(boxstyle="square,pad=0.2", fc="black", ec="none", alpha=0.55))
    if kinds:
        holder = fig.add_subplot(gs[nrow, :])
        holder.axis("off")
        for i, k in enumerate(kinds):
            st = styles[k]
            cax = holder.inset_axes([0.06 + i * (0.9 / len(kinds)), 0.62, 0.9 / len(kinds) - 0.08, 0.28])
            sm = ScalarMappable(norm=Normalize(st["p_min"], st["p_max"]), cmap=st["cmap"])
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
            cb.set_label(f"{st['label']}  (transparent below {st['p_min']:.2f}; outline = decision mask)",
                         fontsize=7)
            cb.ax.tick_params(labelsize=6.5)
    fig.suptitle(f"{static.get('title') or 'SIT-FUSE probabilistic products'}  |  {fr['time']:%Y-%m-%d %H:%M} UTC",
                 fontsize=9)
    fig.savefig(out_png, dpi=int(static.get("dpi", 130)), facecolor="white")
    plt.close(fig)
    return out_png


def frames_table(plan: list[dict]) -> pd.DataFrame:
    rows = []
    for k, fr in enumerate(plan):
        for pn in fr["panels"]:
            if not pn or pn.get("blank"):
                continue
            r = {"frame": k, "time": fr["time"], "panel": pn["title"],
                 "obs_time": pn["obs"]["time"] if pn.get("obs") else None,
                 "obs_files": ";".join(pn["obs"]["paths"]) if pn.get("obs") else ""}
            for o in pn["overlays"]:
                if o:
                    r[f"{o['name']}_time"] = o["time"]
                    r[f"{o['name']}_files"] = ";".join(o["paths"])
            rows.append(r)
    return pd.DataFrame(rows)


def to_jsonable(x):
    return json.loads(json.dumps(x, default=str))
