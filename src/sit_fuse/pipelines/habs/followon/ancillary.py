"""Ancillary (environmental) predictors for the HAB UQ heads.

Two column families, both opt-in through uq.features.groups:
  anc_*   environmental state at the sample (training) or pixel (inference) location and date:
            anc_sst, anc_sst_anom, anc_sst_grad   MUR SST L4 (antecedent mean over [d-w, d-1] + same day)
            anc_cuti, anc_beuti (+ _anom)         daily coastal upwelling indices (nearest 1-degree latitude)
            anc_mei, anc_pdo, anc_npgo, anc_oni   monthly climate indices, lagged lag_months
            anc_q_near, anc_q_region              log1p USGS discharge (nearest gauge within max_km, regional sum)
  phys_*  standard ocean-colour L3m products of the same platform and scene date (chlor_a, nflh,
          Kd_490, aph_443, bbp_443, adg_443), 3x3 nan-mean at the sample/pixel.
Groups: "ancillary" (all anc_*), "ancillary:<source>" (one source: sst, upwelling, indices,
discharge), "physical" (all phys_*), "physical:<product>".

Config (uq.ancillary):
  cache_dir: /mnt/data1/HAB_ANCILLARY
  bbox: [lat_min, lat_max, lon_min, lon_max]       # default 30, 38.94, -128, -116
  sst: {stride: 2, windows: [3, 7], gradient: true}
  upwelling: {windows: [3, 7, 14], climatology_years: [1988, 2022]}
  indices: {names: [mei, pdo, npgo, oni], lag_months: 1}
  discharge: {windows: [1, 7], max_km: 60, gauges: {...}}   # default: south/central CA gauges below
  physical: {dirs: {s3a: /mnt/data1/CHL/S3A_OLCI, ...}, products: {chlor_a: CHL, nflh: FLH, Kd_490: KD}}

Pull (cached, restartable; one npz per source-day / one csv per series):
  python -m sit_fuse.pipelines.habs.followon.ancillary pull -y HAB.yaml --sources sst,upwelling,indices,discharge \
      --start 2022-12-01 --end 2026-10-31 --threads 4
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

ERDDAP_MUR = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41"
ERDDAP_MUR_ANOM = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41anom1day"
ERDDAP_UPW = "https://upwell.pfeg.noaa.gov/erddap/griddap/erd{name}daily"
INDEX_URLS = {
    "mei": "https://psl.noaa.gov/enso/mei/data/meiv2.data",
    "pdo": "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat",
    "npgo": "https://www.o3d.org/npgo/data/NPGO.txt",
    "oni": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
}
USGS_DAILY = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"
DEFAULT_GAUGES = {   # site id: (lon, lat)
    "11074000": (-117.645, 33.883), "11114000": (-119.142, 34.279), "11103000": (-118.206, 33.817),
    "11023000": (-117.169, 32.765), "11118500": (-119.308, 34.352), "11046000": (-117.347, 33.311),
    "11013500": (-117.084, 32.552), "11152500": (-121.672, 36.631), "11159000": (-121.598, 36.900),
    "11141000": (-120.572, 34.976), "11120000": (-119.812, 34.425), "11162500": (-122.329, 37.261),
}
PHYS_VARS = {"chlor_a": "CHL", "nflh": "FLH", "Kd_490": "KD", "aph_443": "IOP", "bbp_443": "IOP", "adg_443": "IOP"}
SOURCES = ("sst", "upwelling", "indices", "discharge")


# ----------------------------------------------------------------------------- config / io
def anc_cfg(uq: dict[str, Any]) -> dict[str, Any]:
    a = dict(uq.get("ancillary") or {})
    a.setdefault("cache_dir", "hab_ancillary_cache")
    a.setdefault("bbox", [30.0, 38.94, -128.0, -116.0])
    return a


def enabled_sources(features_cfg: dict[str, Any]) -> tuple[set[str], set[str]]:
    """(ancillary sources, physical products) requested by features.groups."""
    g = set((features_cfg or {}).get("groups") or [])
    anc = set(SOURCES) if "ancillary" in g else {x.split(":", 1)[1] for x in g if x.startswith("ancillary:")}
    phys = {"*"} if "physical" in g else {x.split(":", 1)[1] for x in g if x.startswith("physical:")}
    return anc, phys


def column_source(c: str) -> str | None:
    if c.startswith("phys_"):
        return "physical:" + c[5:]
    if not c.startswith("anc_"):
        return None
    k = c[4:]
    if k.startswith("sst"):
        return "ancillary:sst"
    if k.startswith(("cuti", "beuti")):
        return "ancillary:upwelling"
    if k.startswith("q_"):
        return "ancillary:discharge"
    return "ancillary:indices"


def keep_column(c: str, features_cfg: dict[str, Any]) -> bool:
    """Whether an anc_/phys_ column is requested by features.groups (non-ancillary columns: True)."""
    src = column_source(c)
    if src is None:
        return True
    g = set((features_cfg or {}).get("groups") or [])
    fam = src.split(":")[0]
    return fam in g or src in g


def _get(url: str, retries: int = 4, timeout: int = 120) -> bytes:
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sit-fuse-hab/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:   # follows redirects
                return r.read()
        except Exception as e:  # pragma: no cover - network
            last = e
            time.sleep(2 ** k)
    raise RuntimeError(f"GET failed {url}: {last}")


def _atomic_save_npz(path: Path, **arrs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- SST (MUR via ERDDAP)
def _mur_url(base: str, var: str, day: pd.Timestamp, bbox, stride: int) -> str:
    t = day.strftime("%Y-%m-%dT09:00:00Z")
    q = f"{var}[({t})][({bbox[0]}):{stride}:({bbox[1]})][({bbox[2]}):{stride}:({bbox[3]})]"
    return f"{base}.csv?{urllib.parse.quote(q, safe='[]():,')}"


def _parse_griddap_csv(raw: bytes, var: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(io.BytesIO(raw), skiprows=[1])
    lat = np.sort(df["latitude"].unique())
    lon = np.sort(df["longitude"].unique())
    grid = np.full((lat.size, lon.size), np.nan, np.float32)
    grid[np.searchsorted(lat, df["latitude"]), np.searchsorted(lon, df["longitude"])] = df[var].to_numpy(np.float32)
    return lat, lon, grid


def pull_sst_day(cfg: dict[str, Any], day: pd.Timestamp) -> Path:
    out = Path(cfg["cache_dir"]) / "sst" / f"mur_{day:%Y%m%d}.npz"
    if out.exists():
        return out
    sc = cfg.get("sst") or {}
    stride = int(sc.get("stride", 2))
    lat, lon, sst = _parse_griddap_csv(_get(_mur_url(ERDDAP_MUR, "analysed_sst", day, cfg["bbox"], stride)), "analysed_sst")
    try:
        _, _, anom = _parse_griddap_csv(_get(_mur_url(ERDDAP_MUR_ANOM, "sstAnom", day, cfg["bbox"], stride)), "sstAnom")
        if anom.shape != sst.shape:
            anom = np.full_like(sst, np.nan)
    except RuntimeError:
        anom = np.full_like(sst, np.nan)
    _atomic_save_npz(out, lat=lat, lon=lon, sst=sst, anom=anom)
    return out


def _load_sst(cfg, day):
    p = Path(cfg["cache_dir"]) / "sst" / f"mur_{day:%Y%m%d}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return z["lat"], z["lon"], z["sst"], z["anom"]


def _grad_km(field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """|grad SST| in degC/km on a regular lat/lon grid."""
    dy = abs(float(np.median(np.diff(lat)))) * 111.2
    dx = abs(float(np.median(np.diff(lon)))) * 111.2 * np.cos(np.deg2rad(lat))[:, None]
    gy, gx = np.gradient(field)
    return np.hypot(gx / dx, gy / dy).astype(np.float32)


def _sample_grid(lat_g, lon_g, grid, lat, lon, half: int = 1) -> np.ndarray:
    """Nan-mean of a (2 half + 1)^2 window at the nearest grid node of each point."""
    i = np.clip(np.searchsorted(lat_g, lat), 0, lat_g.size - 1)
    j = np.clip(np.searchsorted(lon_g, lon), 0, lon_g.size - 1)
    if half <= 0:
        return grid[i, j]
    acc = np.zeros(len(lat))
    n = np.zeros(len(lat))
    for di in range(-half, half + 1):
        for dj in range(-half, half + 1):
            v = grid[np.clip(i + di, 0, grid.shape[0] - 1), np.clip(j + dj, 0, grid.shape[1] - 1)]
            ok = np.isfinite(v)
            acc[ok] += v[ok]
            n[ok] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, acc / n, np.nan)


def sst_features(cfg, day: pd.Timestamp, lat, lon) -> dict[str, np.ndarray]:
    sc = cfg.get("sst") or {}
    out = {}
    cur = _load_sst(cfg, day)
    if cur is not None:
        lg, og, sst, anom = cur
        out["anc_sst"] = _sample_grid(lg, og, sst, lat, lon)
        out["anc_sst_anom"] = _sample_grid(lg, og, anom, lat, lon)
        if sc.get("gradient", True):
            out["anc_sst_grad"] = _sample_grid(lg, og, _grad_km(sst, lg, og), lat, lon)
    for w in sc.get("windows", [3, 7]):
        vals = []
        for k in range(1, int(w) + 1):
            prev = _load_sst(cfg, day - pd.Timedelta(days=k))
            if prev is not None:
                vals.append(_sample_grid(prev[0], prev[1], prev[2], lat, lon))
        if vals:
            m = np.nanmean(np.vstack(vals), 0) if len(vals) > 1 else vals[0]
            out[f"anc_sst_mean{w}d"] = m
            if "anc_sst" in out:
                out[f"anc_sst_chg{w}d"] = out["anc_sst"] - m
    n = len(lat)
    for k in ["anc_sst", "anc_sst_anom"] + (["anc_sst_grad"] if sc.get("gradient", True) else []) + \
             [f"anc_sst_{p}{w}d" for w in sc.get("windows", [3, 7]) for p in ("mean", "chg")]:
        out.setdefault(k, np.full(n, np.nan))
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


# ----------------------------------------------------------------------------- upwelling indices
def _erddap_time_end(base: str) -> pd.Timestamp | None:
    """Last time step of an ERDDAP griddap dataset (info page), None if unavailable."""
    try:
        info = pd.read_csv(io.BytesIO(_get(base.replace("/griddap/", "/info/") + "/index.csv", retries=2)))
        row = info[(info["Variable Name"] == "NC_GLOBAL") & (info["Attribute Name"] == "time_coverage_end")]
        return pd.Timestamp(row["Value"].iloc[0]).tz_localize(None).normalize() if len(row) else None
    except Exception as e:  # pragma: no cover - network
        LOG.warning("ERDDAP info failed for %s: %s", base, e)
        return None


def pull_upwelling(cfg, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    """CUTI / BEUTI daily (31-47 N) from the climatology start through min(end, dataset end).

    ERDDAP returns 404 when a request runs past the last time step, and CUTI / BEUTI lag real time by
    weeks to months, so the request end is clamped to the datasets' time_coverage_end. Days after it get
    NaN upwelling features (Upwelling.features), and a rerun extends the cache once ERDDAP updates.
    """
    out = Path(cfg["cache_dir"]) / "upwelling" / "cuti_beuti_daily.csv"
    uc = cfg.get("upwelling") or {}
    c0 = int((uc.get("climatology_years") or [1988, 2022])[0])
    t0 = min(pd.Timestamp(f"{c0}-01-01"), start - pd.Timedelta(days=30))
    end = pd.Timestamp(end).normalize()
    names = ("CUTI", "BEUTI")
    avail = [_erddap_time_end(ERDDAP_UPW.format(name=n)) for n in names]
    t1 = min([end] + [a for a in avail if a is not None])
    if t1 < end:
        LOG.warning("upwelling: CUTI/BEUTI end %s, before requested end %s; later days will be NaN",
                    f"{t1:%Y-%m-%d}", f"{end:%Y-%m-%d}")
    if out.exists():
        have = pd.read_csv(out, parse_dates=["time"])
        if have["time"].min() <= t0 and have["time"].max() >= t1:
            return out
    frames = []
    for name in names:
        # dataset time stamps are 00:00Z; ERDDAP snaps to the nearest step
        q = f"{name}[({t0:%Y-%m-%d}T00:00:00Z):1:({t1:%Y-%m-%d}T00:00:00Z)][(31.0):1:(47.0)]"
        raw = _get(f"{ERDDAP_UPW.format(name=name)}.csv?{urllib.parse.quote(q, safe='[]():,')}")
        df = pd.read_csv(io.BytesIO(raw), skiprows=[1])
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
        frames.append(df.rename(columns={name: name.lower()}).set_index(["time", "latitude"]))
    _atomic_csv(pd.concat(frames, axis=1).reset_index(), out)
    return out


class Upwelling:
    def __init__(self, cfg):
        p = Path(cfg["cache_dir"]) / "upwelling" / "cuti_beuti_daily.csv"
        self.ok = p.exists()
        if not self.ok:
            return
        uc = cfg.get("upwelling") or {}
        self.windows = [int(w) for w in uc.get("windows", [3, 7, 14])]
        df = pd.read_csv(p, parse_dates=["time"])
        self.lats = np.sort(df["latitude"].unique())
        self.tab = {v: df.pivot(index="time", columns="latitude", values=v).sort_index() for v in ("cuti", "beuti")}
        y0, y1 = (uc.get("climatology_years") or [1988, 2022])
        self.clim = {}
        for v, t in self.tab.items():
            c = t[(t.index.year >= int(y0)) & (t.index.year <= int(y1))]
            clim = c.groupby(c.index.dayofyear).mean()
            self.clim[v] = clim.reindex(range(1, 367)).interpolate(limit_direction="both")

    def features(self, day: pd.Timestamp, lat) -> dict[str, np.ndarray]:
        n = len(lat)
        cols = [f"anc_{v}{s}" for v in ("cuti", "beuti") for s in ("", "_anom")] + \
               [f"anc_{v}_mean{w}d" for v in ("cuti", "beuti") for w in getattr(self, "windows", [3, 7, 14])]
        if not self.ok:
            return {c: np.full(n, np.nan, np.float32) for c in cols}
        li = self.lats[np.clip(np.searchsorted(self.lats, np.round(np.asarray(lat))), 0, self.lats.size - 1)]
        out = {}
        for v, t in self.tab.items():
            today = t.reindex([day]).iloc[0]
            out[f"anc_{v}"] = today.reindex(li).to_numpy()
            out[f"anc_{v}_anom"] = out[f"anc_{v}"] - self.clim[v].loc[day.dayofyear].reindex(li).to_numpy()
            for w in self.windows:
                win = t.loc[day - pd.Timedelta(days=w): day - pd.Timedelta(days=1)]
                out[f"anc_{v}_mean{w}d"] = win.mean().reindex(li).to_numpy() if len(win) else np.full(n, np.nan)
        return {k: np.asarray(v, np.float32) for k, v in out.items()}


# ----------------------------------------------------------------------------- climate indices
def _parse_index(name: str, text: str) -> pd.Series:
    rows = []
    lines = [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    if name == "npgo":
        for l in lines:
            p = l.split()
            try:
                y, m, v = float(p[0]), float(p[1]), float(p[2])
            except (IndexError, ValueError):
                continue
            if 1900 < y < 2200 and 1 <= m <= 12:
                rows.append((int(y), int(m), v))
    elif name == "oni":
        seas = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
        for l in lines:
            p = l.split()
            if len(p) >= 4 and p[0] in seas:
                rows.append((int(p[1]), seas.index(p[0]) + 1, float(p[3])))   # centre month of the season
    else:   # mei (year + 12 values, header line with year range), pdo (Year Jan..Dec)
        for l in lines:
            p = l.split()
            if len(p) == 13 and re.fullmatch(r"\d{4}", p[0]):
                for m in range(12):
                    v = float(p[m + 1])
                    if abs(v) < 90:
                        rows.append((int(p[0]), m + 1, v))
    s = pd.Series({pd.Timestamp(y, m, 1): v for y, m, v in rows}, dtype=float).sort_index()
    return s


def pull_indices(cfg, names: Iterable[str] | None = None) -> Path:
    out = Path(cfg["cache_dir"]) / "indices" / "climate_indices_monthly.csv"
    names = list(names or (cfg.get("indices") or {}).get("names", list(INDEX_URLS)))
    ser = {}
    for n in names:
        try:
            ser[n] = _parse_index(n, _get(INDEX_URLS[n]).decode("utf-8", "replace"))
        except RuntimeError as e:  # pragma: no cover - network
            LOG.warning("index %s: %s", n, e)
    df = pd.DataFrame(ser)
    df.index.name = "month"
    _atomic_csv(df.reset_index(), out)
    return out


class Indices:
    def __init__(self, cfg):
        p = Path(cfg["cache_dir"]) / "indices" / "climate_indices_monthly.csv"
        ic = cfg.get("indices") or {}
        self.names = list(ic.get("names", list(INDEX_URLS)))
        self.lag = int(ic.get("lag_months", 1))
        self.df = pd.read_csv(p, parse_dates=["month"]).set_index("month") if p.exists() else None

    def features(self, day: pd.Timestamp, n: int) -> dict[str, np.ndarray]:
        m = (pd.Timestamp(day.year, day.month, 1) - pd.DateOffset(months=self.lag))
        out = {}
        for k in self.names:
            v = np.nan
            if self.df is not None and k in self.df:
                s = self.df[k].dropna()
                s = s[s.index <= m]
                v = float(s.iloc[-1]) if len(s) and (m - s.index[-1]).days < 100 else np.nan
            out[f"anc_{k}"] = np.full(n, v, np.float32)
        return out


# ----------------------------------------------------------------------------- discharge (USGS)
def pull_discharge(cfg, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    dc = cfg.get("discharge") or {}
    gauges = dc.get("gauges") or DEFAULT_GAUGES
    t0 = start - pd.Timedelta(days=max(int(w) for w in dc.get("windows", [1, 7])) + 2)
    paths = []
    for gid in gauges:
        out = Path(cfg["cache_dir"]) / "discharge" / f"usgs_{gid}.csv"
        if out.exists():
            have = pd.read_csv(out, parse_dates=["time"])
            if len(have) and have["time"].min() <= t0 + pd.Timedelta(days=3) and have["time"].max() >= end - pd.Timedelta(days=3):
                paths.append(out)
                continue
        recs, url = [], (f"{USGS_DAILY}?monitoring_location_id=USGS-{gid}&parameter_code=00060&statistic_id=00003"
                         f"&time={t0:%Y-%m-%d}/{end:%Y-%m-%d}&f=json&limit=10000")
        while url:
            js = json.loads(_get(url))
            for f in js.get("features", []):
                p = f.get("properties", {})
                try:
                    recs.append((pd.Timestamp(p["time"]).normalize(), float(p["value"])))
                except (KeyError, TypeError, ValueError):
                    continue
            url = next((l["href"] for l in js.get("links", []) if l.get("rel") == "next"), None)
        df = pd.DataFrame(recs, columns=["time", "q_cfs"]).drop_duplicates("time").sort_values("time")
        _atomic_csv(df, out)
        paths.append(out)
    return paths


class Discharge:
    def __init__(self, cfg):
        dc = cfg.get("discharge") or {}
        self.gauges = {k: tuple(v) for k, v in (dc.get("gauges") or DEFAULT_GAUGES).items()}
        self.windows = [int(w) for w in dc.get("windows", [1, 7])]
        self.max_km = float(dc.get("max_km", 60.0))
        self.q = {}
        for gid in self.gauges:
            p = Path(cfg["cache_dir"]) / "discharge" / f"usgs_{gid}.csv"
            if p.exists():
                self.q[gid] = pd.read_csv(p, parse_dates=["time"]).set_index("time")["q_cfs"].sort_index()
        self.ids = [g for g in self.gauges if g in self.q]

    def _win(self, gid, day, w):
        s = self.q[gid].loc[day - pd.Timedelta(days=w - 1): day]   # w days ending on the sample day
        return float(s.mean()) if len(s) else np.nan

    def features(self, day: pd.Timestamp, lat, lon) -> dict[str, np.ndarray]:
        n = len(lat)
        out = {}
        if not self.ids:
            return {f"anc_q_{k}{w}d": np.full(n, np.nan, np.float32) for k in ("near", "region") for w in self.windows}
        glon = np.array([self.gauges[g][0] for g in self.ids])
        glat = np.array([self.gauges[g][1] for g in self.ids])
        la, lo = np.deg2rad(np.asarray(lat))[:, None], np.deg2rad(np.asarray(lon))[:, None]
        d = 2 * 6371 * np.arcsin(np.sqrt(np.sin((np.deg2rad(glat) - la) / 2) ** 2 +
                                         np.cos(la) * np.cos(np.deg2rad(glat)) * np.sin((np.deg2rad(glon) - lo) / 2) ** 2))
        near = d.argmin(1)
        near_ok = d[np.arange(n), near] <= self.max_km
        for w in self.windows:
            q = np.array([self._win(g, day, w) for g in self.ids])
            out[f"anc_q_near{w}d"] = np.where(near_ok, np.log1p(q[near]), 0.0)   # 0 = no gauged river nearby
            out[f"anc_q_region{w}d"] = np.full(n, np.log1p(np.nansum(q)) if np.isfinite(q).any() else np.nan)
        return {k: np.asarray(v, np.float32) for k, v in out.items()}


# ----------------------------------------------------------------------------- physical (L3m) products
class Physical:
    """Standard L3m products of the same platform and date, read from sync_l3m_rrs directories."""

    def __init__(self, cfg, wanted: set[str]):
        pc = cfg.get("physical") or {}
        self.dirs = {k: v if isinstance(v, list) else [v] for k, v in (pc.get("dirs") or {}).items()}
        prods = pc.get("products") or {"chlor_a": "CHL", "nflh": "FLH", "Kd_490": "KD"}
        self.products = {k: v for k, v in prods.items() if "*" in wanted or k in wanted}
        self._cache: dict[tuple, Any] = {}

    def _find(self, sensor: str, day: pd.Timestamp, suite: str) -> str | None:
        tok = day.strftime("%Y%m%d")
        suite_re = {"CHL": r"\.(CHL|BGC)\.", "FLH": r"\.FLH\.", "KD": r"\.KD\.", "IOP": r"\.IOP\."}[suite]
        for d in self.dirs.get(sensor, []):
            for f in sorted(glob(os.path.join(d, f"*{tok}*L3m*DAY*.nc"))):
                if re.search(suite_re, os.path.basename(f)) and ".1km." not in f:
                    return f
        return None

    def _read(self, path: str, var: str):
        key = (path, var)
        if key not in self._cache:
            import xarray as xr
            with xr.open_dataset(path, mask_and_scale=True) as ds:
                v = ds[var].squeeze()
                lat, lon = ds["lat"].to_numpy(), ds["lon"].to_numpy()
                a = v.to_numpy().astype(np.float32)
            if lat[0] > lat[-1]:
                lat, a = lat[::-1], a[::-1]
            if len(self._cache) > 16:
                self._cache.clear()
            self._cache[key] = (lat, lon, a)
        return self._cache[key]

    def features(self, sensor: str, day: pd.Timestamp, lat, lon) -> dict[str, np.ndarray]:
        out = {}
        for var, suite in self.products.items():
            f = self._find(sensor, day, suite)
            v = np.full(len(lat), np.nan, np.float32)
            if f is not None:
                try:
                    lg, og, a = self._read(f, var)
                    v = _sample_grid(lg, og, a, np.asarray(lat), np.asarray(lon)).astype(np.float32)
                except (KeyError, OSError) as e:
                    LOG.warning("physical %s %s: %s", f, var, e)
            if var in ("chlor_a", "Kd_490", "bbp_443", "aph_443", "adg_443"):
                with np.errstate(invalid="ignore", divide="ignore"):
                    v = np.log10(np.where(v > 0, v, np.nan)).astype(np.float32)
            out[f"phys_{var}"] = v
        return out


# ----------------------------------------------------------------------------- joiners
class AncillaryProvider:
    def __init__(self, uq: dict[str, Any]):
        self.cfg = anc_cfg(uq)
        self.anc, self.phys = enabled_sources(uq.get("features", {}))
        self.upw = Upwelling(self.cfg) if "upwelling" in self.anc else None
        self.idx = Indices(self.cfg) if "indices" in self.anc else None
        self.dis = Discharge(self.cfg) if "discharge" in self.anc else None
        self.phy = Physical(self.cfg, self.phys) if self.phys else None

    @property
    def active(self) -> bool:
        return bool(self.anc or self.phys)

    def at(self, sensor: str, day, lat, lon) -> dict[str, np.ndarray]:
        day = pd.Timestamp(day).normalize()
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        out: dict[str, np.ndarray] = {}
        if "sst" in self.anc:
            out.update(sst_features(self.cfg, day, lat, lon))
        if self.upw is not None:
            out.update(self.upw.features(day, lat))
        if self.idx is not None:
            out.update(self.idx.features(day, len(lat)))
        if self.dis is not None:
            out.update(self.dis.features(day, lat, lon))
        if self.phy is not None:
            out.update(self.phy.features(sensor, day, lat, lon))
        return out


def add_ancillary_columns(df: pd.DataFrame, samples: pd.DataFrame, uq: dict[str, Any]) -> pd.DataFrame:
    """Training/evaluation matchup rows: ancillary columns at the sample location.

    anc_* use the sample date (environment at the in-situ measurement); phys_* use the
    platform and scene date of each row (what the satellite saw, like the head features)."""
    prov = AncillaryProvider(uq)
    if not prov.active or df.empty:
        return df
    loc = samples.drop_duplicates("sample_uid").set_index("sample_uid")[["Latitude", "Longitude"]]
    lat = df["sample_uid"].map(loc["Latitude"]).to_numpy(float)
    lon = df["sample_uid"].map(loc["Longitude"]).to_numpy(float)
    sdate = pd.to_datetime(df["scene_date"] if "scene_date" in df else df["date"]).dt.normalize()
    mdate = pd.to_datetime(df["date"]).dt.normalize()
    new: dict[str, np.ndarray] = {}
    keyframe = pd.DataFrame({"sensor": df["sensor"].to_numpy(), "s": sdate.to_numpy(), "m": mdate.to_numpy()})
    anc_only = AncillaryProvider({**uq, "features": {"groups": [f"ancillary:{s}" for s in prov.anc]}}) if prov.anc else None
    phys_only = AncillaryProvider({**uq, "features": {"groups": [f"physical:{p}" for p in prov.phys]}}) if prov.phys else None
    if anc_only is not None:
        for m, ix in keyframe.groupby("m").indices.items():
            for k, v in anc_only.at("", m, lat[ix], lon[ix]).items():
                new.setdefault(k, np.full(len(df), np.nan, np.float32))[ix] = v
    if phys_only is not None:
        for (s, d), ix in keyframe.groupby(["sensor", "s"]).indices.items():
            for k, v in phys_only.at(s, d, lat[ix], lon[ix]).items():
                new.setdefault(k, np.full(len(df), np.nan, np.float32))[ix] = v
    LOG.info("ancillary columns: %s (non-missing %s)", sorted(new),
             {k: round(float(np.isfinite(v).mean()), 3) for k, v in new.items()})
    return df.assign(**new)


def add_pixel_ancillary(uq: dict[str, Any], feat: pd.DataFrame, prof: dict[str, Any], sensor: str, date,
                        required: Iterable[str] | None = None) -> pd.DataFrame:
    """Scene inference: ancillary columns at pixel centres (anc_* and phys_* both use the scene date)."""
    req = [c for c in (required or []) if c.startswith(("anc_", "phys_"))]
    if not req:
        return feat
    from sit_fuse.pipelines.habs.uq.hab_uq_common import pixel_centers
    LAT, LON = pixel_centers(prof)
    r, c = feat["row"].to_numpy(), feat["col"].to_numpy()
    vals = AncillaryProvider(uq).at(sensor, date, LAT[r, c], LON[r, c])
    missing = [k for k in req if k not in vals]
    if missing:
        raise ValueError(f"ancillary columns required by the model but not produced: {missing}")
    return feat.assign(**{k: vals[k] for k in req})


# ----------------------------------------------------------------------------- CLI
def _days(a, b):
    return list(pd.date_range(pd.Timestamp(a), pd.Timestamp(b), freq="D"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("pull")
    q.add_argument("-y", "--yaml", required=True)
    q.add_argument("--sources", default=",".join(SOURCES))
    q.add_argument("--start", required=True)
    q.add_argument("--end", required=True)
    q.add_argument("--threads", type=int, default=4)
    q.add_argument("--lead-days", type=int, default=None, help="extra days before start (default max SST window + 1)")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import yaml
    with open(a.yaml) as f:
        hab = yaml.safe_load(f)
    cfg = anc_cfg(hab["uq"])
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    srcs = [s.strip() for s in a.sources.split(",") if s.strip()]
    if "indices" in srcs:
        LOG.info("indices -> %s", pull_indices(cfg))
    if "upwelling" in srcs:
        LOG.info("upwelling -> %s", pull_upwelling(cfg, start, end))
    if "discharge" in srcs:
        LOG.info("discharge -> %d gauges", len(pull_discharge(cfg, start, end)))
    if "sst" in srcs:
        lead = a.lead_days if a.lead_days is not None else max((cfg.get("sst") or {}).get("windows", [3, 7])) + 1
        days = _days(start - pd.Timedelta(days=lead), end)
        fails = []

        def one(d):
            try:
                pull_sst_day(cfg, d)
            except Exception as e:  # pragma: no cover - network
                fails.append((str(d.date()), repr(e)))
        with ThreadPoolExecutor(max(1, a.threads)) as ex:
            list(ex.map(one, days))
        LOG.info("sst: %d days, %d failed", len(days), len(fails))
        if fails:
            _atomic_csv(pd.DataFrame(fails, columns=["date", "error"]), Path(cfg["cache_dir"]) / "sst" / "failed_days.csv")
            if len(fails) > 0.05 * len(days):
                raise SystemExit(f"sst: {len(fails)} of {len(days)} days failed; rerun to retry (cached days are skipped)")


if __name__ == "__main__":
    main()
