"""
Data access for SIT-FUSE UQ visualization.

GeoProductIndex   export_geo_products.py outputs (manifest or <root>/<model>/rasters/*.tif),
                  indexed by model / scene / time / split from the GeoTIFF tags.
Frames            scenes grouped into display frames (TEMPO scan = one hourly
                  sweep; or fixed time bins for ABI), mosaicked across granules.
References        comparison products on the product grid:
                    hms_polygons  NOAA HMS smoke polygons (time-matched, positive-only)
                    hms_points    NOAA HMS fire points (positive-only)
                    raster        reference rasters (ADP, VIIRS, HMS daily, FDCC, FIRMS): paths from
                                  matchup / attach_references YAMLs, or a glob matched to scenes
                                  by pair id, nearest time (ADP, VIIRS) or UTC date (HMS daily)
Evaluation        evaluate_uq.py outputs (summary JSON and CSVs) per model.
Basemap           Natural Earth coastline / borders / states (downloaded once, cached).
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import rasterio
import rasterio.errors
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

LOG = logging.getLogger("uq-viz")

SCENE_RE = re.compile(r"(?P<sensor>TEMPO|OR_ABI|ABI|GOES)[^/]*?(?P<ts>\d{8}T\d{6})Z?(?:_S(?P<scan>\d{3})G(?P<gran>\d{2}))?")


# ---------------------------------------------------------------------------
# geo products
# ---------------------------------------------------------------------------

_ABI_RE = re.compile(r"_s(\d{13})\d?_e")


def _scene_time(tags: dict, name: str, m=None):
    """scene_time tag; else the TEMPO (YYYYmmddTHHMMSS) or GOES ABI (_sYYYYJJJHHMMSS) time in the pair id / file
    name (GOES rasters exported before the ABI-name fix carry scene_time=NaT)."""
    v = tags.get("scene_time")
    if v and v not in ("NaT", "None", "nan"):
        return pd.Timestamp(v)
    for s in (tags.get("pair_id", ""), name):
        a = _ABI_RE.search(str(s))
        if a:
            return pd.Timestamp(pd.to_datetime(a.group(1), format="%Y%j%H%M%S"), tz="UTC")
        b = re.search(r"(\d{8}T\d{6})", str(s))
        if b:
            return pd.Timestamp(b.group(1), tz="UTC")
    return pd.Timestamp(m.group("ts"), tz="UTC") if m else pd.NaT


class GeoProductIndex:
    """All exported rasters for one product, one row per (model, scene)."""

    def __init__(self, root: str | Path, product: str, models: Iterable[str] | None = None,
                 manifest: str | Path | None = None):
        self.root = Path(root)
        rows = []
        paths: dict[str, list[Path]] = {}
        if manifest and Path(manifest).exists():
            man = json.loads(Path(manifest).read_text())
            for model in man:
                d = self.root / model / "rasters"
                if d.is_dir():
                    paths[model] = sorted(d.glob("*.tif"))
                else:
                    LOG.warning("manifest lists %s but %s does not exist", model, d)
        else:
            for d in sorted(self.root.glob("*/rasters")):
                paths[d.parent.name] = sorted(d.glob("*.tif"))
        if models:
            paths = {m: paths[m] for m in models if m in paths}
            missing = set(models) - set(paths)
            if missing:
                LOG.warning("models without geo products: %s", ", ".join(sorted(missing)))
        for model, fs in paths.items():
            for f in fs:
                with rasterio.open(f) as s:
                    t = s.tags()
                    desc = list(s.descriptions)
                if t.get("product", product) != product:
                    continue
                m = SCENE_RE.search(f.name)
                rows.append({
                    "model": model, "path": str(f), "pair_id": t.get("pair_id", f.stem),
                    "scene_time": _scene_time(t, f.name, m),
                    "date": t.get("date"), "split": t.get("classifier_split", "unassigned"),
                    "stratum": t.get("split_stratum", ""), "threshold": float(t.get("threshold", "nan")),
                    "scan": m.group("scan") if m else None, "granule": m.group("gran") if m else None,
                    "bands": desc,
                })
        if not rows:
            raise FileNotFoundError(f"no {product} geo products under {self.root}")
        df = pd.DataFrame(rows)
        df["scene_time"] = pd.to_datetime(df["scene_time"], utc=True)
        df["event"] = df["stratum"].str.split("|").str[0]
        self.df = df.sort_values(["model", "scene_time"]).reset_index(drop=True)
        with rasterio.open(self.df.path.iloc[0]) as s:
            self.transform, self.crs, self.shape = s.transform, s.crs, (s.height, s.width)
            self.bounds = s.bounds
            self.nodata = s.nodata
        LOG.info("geo products: %d rasters, %d models, %d scenes, %s .. %s", len(df), df.model.nunique(),
                 df.pair_id.nunique(), df.scene_time.min(), df.scene_time.max())

    @property
    def models(self) -> list[str]:
        return list(dict.fromkeys(self.df.model))

    def scenes(self, model: str | None = None) -> pd.DataFrame:
        d = self.df if model is None else self.df[self.df.model == model]
        return d.drop_duplicates("pair_id")

    def extent(self):
        b = self.bounds
        return [b.left, b.right, b.bottom, b.top]


def assign_frames(scenes: pd.DataFrame, mode: str = "scan", minutes: int = 60) -> pd.DataFrame:
    """frame_id per scene: TEMPO scan (one E-W sweep, ~hourly) or fixed UTC time bins."""
    s = scenes.copy()
    if mode == "scan" and s["scan"].notna().all():
        s["frame_id"] = s["date"].astype(str) + "_S" + s["scan"].astype(str)
    else:
        s["frame_id"] = s["scene_time"].dt.floor(f"{minutes}min").dt.strftime("%Y-%m-%dT%H:%MZ")
    t = s.groupby("frame_id")["scene_time"].agg(["min", "max", "median"])
    s = s.join(t, on="frame_id")
    s = s.rename(columns={"min": "frame_start", "max": "frame_end", "median": "frame_time"})
    return s


@lru_cache(maxsize=1024)
def _read_band(path: str, band_name: str) -> np.ndarray | None:
    with rasterio.open(path) as s:
        desc = list(s.descriptions)
        if band_name not in desc:
            return None
        a = s.read(desc.index(band_name) + 1).astype(np.float32)
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
    return a


def fill_seams(a: np.ndarray) -> np.ndarray:
    """Display only: fill isolated NaN pixels whose left/right or up/down neighbours are valid."""
    out = a.copy()
    nan = np.isnan(a)
    for ax in (0, 1):
        lo, hi = np.roll(a, 1, axis=ax), np.roll(a, -1, axis=ax)
        ok = nan & np.isfinite(lo) & np.isfinite(hi)
        out[ok] = (0.5 * (lo + hi))[ok]
    return out


def mosaic(index: GeoProductIndex, model: str, pair_ids: Iterable[str], band: str,
           seams: bool = True) -> np.ndarray | None:
    """First-valid mosaic of one band over the scenes of a frame (granules do not overlap)."""
    d = index.df[(index.df.model == model) & index.df.pair_id.isin(list(pair_ids))]
    out = None
    for p in d.sort_values("scene_time").path:
        a = _read_band(p, band)
        if a is None:
            continue
        out = a.copy() if out is None else np.where(np.isnan(out), a, out)
    if out is not None and seams:
        if band in ("pred_label", "pred_label_selective", "selective_flag", "excluded_cluster_mask"):
            f = fill_seams(out)
            out = np.where(np.isnan(out), np.round(f), out)
        else:
            out = fill_seams(out)
    return out


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------

def _hms_time(s: str) -> pd.Timestamp:
    s = str(s).strip()
    return pd.to_datetime(s, format="%Y%j %H%M", utc=True)


_DENSITY_RANK = {"light": 1, "medium": 2, "heavy": 3}


class HMSPolygons:
    """NOAA HMS smoke polygons (or fire points), cached per day."""

    def __init__(self, cfg: dict, cache_dir: Path, points: bool = False):
        self.cfg, self.cache, self.points = cfg, Path(cache_dir), points
        self.cache.mkdir(parents=True, exist_ok=True)
        self._days: dict[str, Any] = {}

    def _day(self, day: pd.Timestamp):
        import geopandas as gpd
        key = day.strftime("%Y%m%d")
        if key in self._days:
            return self._days[key]
        local = self.cfg.get("local_dir")
        stem = ("hms_fire" if self.points else "hms_smoke") + key
        cands = [Path(local) / f"{stem}.zip", Path(local) / f"{stem}.shp"] if local else []
        cands.append(self.cache / f"{stem}.zip")
        f = next((c for c in cands if c.exists()), None)
        if f is None and self.cfg.get("url") and self.cfg.get("download", True):
            url = self.cfg["url"].format(Y=day.strftime("%Y"), m=day.strftime("%m"), Ymd=key)
            try:
                urllib.request.urlretrieve(url, self.cache / f"{stem}.zip")
                f = self.cache / f"{stem}.zip"
                LOG.info("HMS: downloaded %s", url)
            except Exception as exc:  # network optional
                LOG.warning("HMS: %s not available (%s)", url, exc)
        g = None
        if f is not None:
            g = gpd.read_file(f"zip://{f.resolve()}" if f.suffix == ".zip" else f)
            if g.crs is None:
                g = g.set_crs("EPSG:4326")
            g = g.to_crs("EPSG:4326")
            if "Start" in g and "End" in g:
                g["t0"] = g["Start"].map(_hms_time)
                g["t1"] = g["End"].map(_hms_time)
            if not self.points and "Density" in g:
                g["rank"] = g["Density"].astype(str).str.lower().map(_DENSITY_RANK).fillna(1)
        self._days[key] = g
        return g

    def select(self, t: pd.Timestamp) -> Any:
        """Polygons valid at t. None when no HMS analysis period covers t (e.g. night, or no file):
        an empty result then means 'not analysed', not 'smoke-free'."""
        g = self._day(t)
        if g is None or g.empty:
            return None
        if not self.points:
            g = g[g["rank"] >= _DENSITY_RANK.get(str(self.cfg.get("min_density", "light")).lower(), 1)]
        if self.cfg.get("time_match", "interval") == "interval" and "t0" in g:
            tol = pd.Timedelta(minutes=float(self.cfg.get("tolerance_minutes", 30)))
            g = g[(g.t0 - tol <= t) & (g.t1 + tol >= t)]
            if g.empty:
                return None
        return g

    def raster(self, t: pd.Timestamp, index: GeoProductIndex) -> np.ndarray | None:
        g = self.select(t)
        if g is None:
            return None
        out = np.zeros(index.shape, np.float32)
        if len(g):
            out = rasterize(((geom, 1) for geom in g.geometry if geom is not None), out_shape=index.shape,
                            transform=index.transform, fill=0, all_touched=self.points).astype(np.float32)
        return out


def _get_path(rec: dict, field: str):
    cur: Any = rec
    for part in field.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if isinstance(cur, dict):  # additional_references entries may be {path: ..., ...}
        cur = cur.get("path") or cur.get("truth") or cur.get("raster")
    return cur


def _records(obj) -> list[dict]:
    if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "pair_id" in obj[0]:
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            r = _records(v)
            if r:
                return r
    return []


_PAIR_RE = re.compile(r"\d{8}T\d{6}Z_S\d{3}G\d{2}")
_TS_RE = re.compile(r"(\d{8}T\d{6})Z?")
_RESAMPLING = {"nearest": Resampling.nearest, "max": Resampling.max, "mode": Resampling.mode,
               "average": Resampling.average}


def pair_time(pair_id: str) -> pd.Timestamp | None:
    m = _TS_RE.search(str(pair_id))
    return pd.Timestamp(m.group(1), tz="UTC") if m else None


class RasterReference:
    """Reference rasters warped onto the product grid, matched to product scenes by

    match: pair  per-scene paths from matchup / attach_references YAMLs (`matchups` + `field`) or a
                 `glob` whose file names contain the pair id (default)
           time  `glob` of timestamped rasters (e.g. smoke_20250113T202647Z.tif, viirs_smoke_..._NOAA20.tif);
                 each scene gets the temporally nearest raster within `tolerance_minutes`
                 (ADP: same granule start time; VIIRS: nearest overpass within 30 min)
           date  `glob` of daily rasters (e.g. HMS 20250113_smoke.tif); each scene gets its UTC day

    With `clip_to_scene` (default true for time/date), the reference is masked to each granule's product
    footprint before mosaicking a frame, so a domain-wide raster is only compared where TEMPO observed."""

    def __init__(self, name: str, cfg: dict):
        import yaml
        self.name, self.cfg = name, cfg
        self.match = cfg.get("match", "pair")
        self.paths: dict[str, str] = {}
        self.times: pd.Series | None = None
        self._cache: dict[tuple, np.ndarray] = {}
        for y in [cfg.get("matchups")] if isinstance(cfg.get("matchups"), str) else (cfg.get("matchups") or []):
            recs = _records(yaml.safe_load(Path(y).read_text()))
            for r in recs:
                p = _get_path(r, cfg.get("field", "annotation_raster"))
                if p:
                    self.paths[str(r["pair_id"])] = str(p)
        if cfg.get("glob"):
            import glob
            files = sorted(glob.glob(cfg["glob"], recursive=True))
            if self.match == "time":
                rx = re.compile(cfg.get("time_regex", r"(\d{8}T\d{6})Z?"))
                tt = {}
                for p in files:
                    m = rx.search(Path(p).name)
                    if m:
                        tt[p] = pd.Timestamp(m.group(1), tz="UTC")
                self.times = pd.Series(tt, dtype="datetime64[ns, UTC]").sort_values()
            elif self.match == "date":
                rx = re.compile(cfg.get("date_regex", r"(\d{8})"))
                for p in files:
                    m = rx.search(Path(p).name)
                    if m:
                        self.paths.setdefault(m.group(1), p)
            else:
                rx = re.compile(cfg.get("pair_id_regex", r"(\d{8}T\d{6}Z_S\d{3}G\d{2})"))
                for p in files:
                    m = rx.search(p)
                    if m:
                        self.paths[m.group(1)] = p
        self.clip = bool(cfg.get("clip_to_scene", self.match in ("time", "date")))
        n = len(self.times) if self.times is not None else len(self.paths)
        LOG.info("reference %s: %d rasters (match=%s)", name, n, self.match)

    def _lookup(self, pair_id: str) -> str | None:
        if self.match == "time" and self.times is not None and len(self.times):
            t = pair_time(pair_id)
            if t is None:
                return None
            dt = (self.times - t).abs()
            k = dt.idxmin()
            return k if dt[k] <= pd.Timedelta(minutes=float(self.cfg.get("tolerance_minutes", 1))) else None
        if self.match == "date":
            t = pair_time(pair_id)
            return self.paths.get(t.strftime("%Y%m%d")) if t is not None else None
        if pair_id in self.paths:
            return self.paths[pair_id]
        m = _PAIR_RE.search(pair_id)
        return self.paths.get(m.group(0)) if m else None

    def time_offset_minutes(self, pair_id: str) -> float | None:
        """|reference time - scene time| in minutes (time matching only)."""
        p, t = self._lookup(pair_id), pair_time(pair_id)
        if self.match != "time" or p is None or t is None:
            return None
        return abs((self.times[p] - t).total_seconds()) / 60.0

    def _warp(self, p: str, index: GeoProductIndex) -> np.ndarray:
        key = (p, index.shape)
        if key in self._cache:
            return self._cache[key]
        vm = {float(k): float(v) for k, v in (self.cfg.get("value_map") or {0: 0, 1: 1}).items()}
        nod = [float(v) for v in self.cfg.get("nodata_values", [255])]
        rs = _RESAMPLING.get(self.cfg.get("resampling", "nearest"), Resampling.nearest)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            with rasterio.open(p) as s:
                a = s.read(int(self.cfg.get("band", 1))).astype(np.float32)
                if s.nodata is not None:
                    nod = nod + [float(s.nodata)]
                bad = np.isin(a, nod) | ~np.isfinite(a)
                lab = np.full(a.shape, np.nan, np.float32)
                for k, v in vm.items():
                    lab[(a == k) & ~bad] = v
                gcps, gcrs = s.gcps
                src_kw = {"gcps": gcps, "src_crs": gcrs} if gcps else {"src_transform": s.transform,
                                                                       "src_crs": s.crs}
                dst = np.full(index.shape, np.nan, np.float32)
                reproject(lab, dst, dst_transform=index.transform, dst_crs=index.crs, src_nodata=np.nan,
                          dst_nodata=np.nan, resampling=rs, **src_kw)
        if len(self._cache) > int(self.cfg.get("cache_size", 256)):
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = dst
        return dst

    def raster(self, pair_id: str, index: GeoProductIndex) -> np.ndarray | None:
        p = self._lookup(pair_id)
        if not p or not Path(p).exists():
            return None
        return self._warp(p, index)


def scene_footprint(index: GeoProductIndex, model: str, pair_id: str, band: str) -> np.ndarray | None:
    """Pixels observed in one granule (finite product band)."""
    d = index.df[(index.df.model == model) & (index.df.pair_id == pair_id)]
    if d.empty:
        return None
    a = _read_band(d.path.iloc[0], band)
    return None if a is None else np.isfinite(a)


def mosaic_reference(ref: RasterReference, index: GeoProductIndex, pair_ids: Iterable[str],
                     model: str | None = None, band: str = "p_1") -> np.ndarray | None:
    """First-valid mosaic over a frame's granules; clipped to each granule footprint when ref.clip."""
    out = None
    for pid in pair_ids:
        a = ref.raster(pid, index)
        if a is None:
            continue
        if ref.clip and model:
            fp = scene_footprint(index, model, pid, band)
            if fp is not None:
                a = np.where(fp, a, np.nan)
        out = a.copy() if out is None else np.where(np.isnan(out), a, out)
    return out


# ---------------------------------------------------------------------------
# evaluation outputs
# ---------------------------------------------------------------------------

@dataclass
class ModelEval:
    model: str
    summary: dict
    per_scene: pd.DataFrame | None
    reliability: pd.DataFrame | None
    risk_coverage: pd.DataFrame | None
    unc_calibration: pd.DataFrame | None
    point: dict | None = None          # model_comparison.csv row (authoritative point estimates)

    def metric(self, k: str):
        v = (self.point or {}).get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = self.summary.get("overall", {}).get(k)
        if v is None:
            v = self.summary.get("bootstrap_ci", {}).get("metrics", {}).get(k, {}).get("mean")
        return v


def load_evaluation(eval_dir: str | Path, models: Iterable[str]) -> dict[str, ModelEval]:
    eval_dir = Path(eval_dir)
    out = {}

    def rd(p):
        return pd.read_csv(p) if p.exists() else None

    mc = {}
    if (eval_dir / "model_comparison.csv").exists():
        mc = pd.read_csv(eval_dir / "model_comparison.csv").set_index("model").to_dict("index")
    for m in models:
        d = eval_dir / m
        if not (d / "eval_summary.json").exists():
            LOG.warning("evaluation: %s has no eval_summary.json", d)
            continue
        out[m] = ModelEval(m, json.loads((d / "eval_summary.json").read_text()), rd(d / "per_scene_metrics.csv"),
                           rd(d / "reliability_class1.csv"), rd(d / "risk_coverage.csv"),
                           rd(d / "uncertainty_calibration.csv"), mc.get(m))
    return out


# ---------------------------------------------------------------------------
# basemap
# ---------------------------------------------------------------------------

NE_LAYERS = {
    "coastline": "https://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip",
    "borders": "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_boundary_lines_land.zip",
    "states": "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_1_states_provinces_lines.zip",
}


def load_basemap(cache_dir: Path, bounds, layers=("coastline", "borders", "states"), local: dict | None = None):
    import geopandas as gpd
    from shapely.geometry import box
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in layers:
        src = (local or {}).get(name)
        if src is None:
            f = cache_dir / Path(NE_LAYERS[name]).name
            if not f.exists():
                try:
                    urllib.request.urlretrieve(NE_LAYERS[name], f)
                except Exception as exc:
                    LOG.warning("basemap %s unavailable (%s)", name, exc)
                    continue
            src = f"zip://{f.resolve()}"
        g = gpd.read_file(src).to_crs("EPSG:4326")
        out[name] = g.clip(box(bounds[0] - 1, bounds[2] - 1, bounds[1] + 1, bounds[3] + 1))
    return out
