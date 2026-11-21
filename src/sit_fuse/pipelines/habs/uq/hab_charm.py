"""
C-HARM (California Harmful Algae Risk Mapping) download, cache and access.

Source: NOAA CoastWatch West Coast ERDDAP griddap, LonPM180 variants (-180..180 longitude),
3 km (0.03 deg) grid, lat 31.3-43.0, lon -127.5 to -117.0, one field per day at 12:00 UTC.
Variables (probabilities): pseudo_nitzschia  P(PN > 1e4 cells/L)
                           particulate_domoic P(pDA > 500 ng/L)
                           cellular_domoic   P(cDA > 10 pg/cell)
                           chla_filled       gap-filled chlorophyll (mg/m^3)
Forecast datasets (lead 1-3 d) are indexed by valid date, so lead L is compared on the date it
predicts.

Version rule (latest wins): each date uses the newest version whose coverage contains it; a
day missing from that version falls back to the next-newest that has it. Every day carries its
version label so results can be stratified, and overlapping dates of two versions (fetched with
--all-versions) give per-version offsets (charm_version_offsets).

  v3.1  wvcharmV3_{L}day_LonPM180        2022-11-01 ->        VIIRS inputs, ROMS T/S
  v2    charmForecast{L}dayV2_LonPM180   2021-01-01 - 2022-11-30  VIIRS
  v1    charmForecast{L}day_LonPM180     2018-06-19 - 2022-03-17  MODIS-Aqua

Cache: <cache_dir>/<dataset_id>/<dataset_id>.<YYYYMMDD>_<YYYYMMDD>.nc (one chunk of days), plus
<cache_dir>/charm_index_L<lead>.csv (date, version, dataset_id, file, time_index). Downloads are
atomic (.part -> rename) and skipped when the chunk exists, so a rerun resumes; --offline only
re-indexes the cache (cache replay, no requests).

  python -m sit_fuse.pipelines.habs.uq.hab_charm fetch -y <hab yaml> --start 2023-01-01 \
      --end 2026-09-30 --lead 0
  python -m sit_fuse.pipelines.habs.uq.hab_charm index -y <hab yaml> --lead 0

Config (uq.charm, all optional): cache_dir (<uq.out_dir>/charm_cache), erddap, bbox
[lon_min, lon_max, lat_min, lat_max], variables, chunk_days (15), versions (list overriding
VERSIONS), retries (5), pause_sec (2), radius_km (10), fallback_km (30).
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/"
VERSIONS = [
    {"version": "v3.1", "id": "wvcharmV3_{lead}day_LonPM180", "start": "2022-11-01", "end": None},
    {"version": "v2", "id": "charmForecast{lead}dayV2_LonPM180", "start": "2021-01-01", "end": "2022-11-30"},
    {"version": "v1", "id": "charmForecast{lead}day_LonPM180", "start": "2018-06-19", "end": "2022-03-17"},
]
VARIABLES = ["pseudo_nitzschia", "particulate_domoic", "cellular_domoic", "chla_filled"]
EXTRA_VARIABLES = {"v3.1": ["salinity", "water_temparture"]}   # ROMS context (sic: ERDDAP name)
TARGET_VARIABLE = {"pn": "pseudo_nitzschia", "pda": "particulate_domoic", "cda": "cellular_domoic"}
DOMAIN = [-127.5, -117.0, 31.3, 43.0]
FILL_MIN = -9000.0


def read_yaml_any(path: str) -> dict[str, Any]:
    """sit_fuse.utils.read_yaml when importable (env / include handling), else plain PyYAML."""
    try:
        from sit_fuse.utils import read_yaml
        return read_yaml(path)
    except ImportError:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)


def charm_cfg(yml: dict[str, Any]) -> dict[str, Any]:
    uq = yml.get("uq", {})
    c = dict(uq.get("charm") or {})
    c.setdefault("cache_dir", str(Path(uq.get("out_dir", ".")) / "charm_cache"))
    c.setdefault("erddap", ERDDAP)
    c.setdefault("bbox", DOMAIN)
    c.setdefault("variables", VARIABLES)
    c.setdefault("chunk_days", 15)
    c.setdefault("versions", VERSIONS)
    c.setdefault("retries", 5)
    c.setdefault("pause_sec", 2.0)
    c.setdefault("radius_km", 10.0)
    c.setdefault("fallback_km", 30.0)
    c.setdefault("coast_exclude_km", 6.0)     # coastal-edge diagnostic (2 C-HARM pixels)
    return c


def version_table(cfg: dict[str, Any], lead: int) -> list[dict[str, Any]]:
    """Versions newest first, with dataset id and coverage shifted by the lead."""
    out = []
    for v in cfg.get("versions", VERSIONS):
        s = pd.Timestamp(v["start"]) + pd.Timedelta(days=lead)
        e = None if v.get("end") is None else pd.Timestamp(v["end"]) + pd.Timedelta(days=lead)
        out.append({**v, "dataset_id": v["id"].format(lead=lead), "start": s, "end": e})
    return out


def covers(v: dict[str, Any], d: pd.Timestamp) -> bool:
    return d >= v["start"] and (v["end"] is None or d <= v["end"])


def primary_version(versions, d) -> dict[str, Any] | None:
    d = pd.Timestamp(d).normalize()
    return next((v for v in versions if covers(v, d)), None)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def griddap_url(base: str, dataset_id: str, variables, t0, t1, bbox) -> str:
    lo0, lo1, la0, la1 = bbox
    sel = (f"[({pd.Timestamp(t0):%Y-%m-%d}T12:00:00Z):1:({pd.Timestamp(t1):%Y-%m-%d}T12:00:00Z)]"
           f"[({la0}):1:({la1})][({lo0}):1:({lo1})]")
    q = ",".join(f"{v}{sel}" for v in variables)
    return f"{base.rstrip('/')}/{dataset_id}.nc?{urllib.parse.quote(q, safe=',:()')}"


def _get(url: str, dest: Path, retries: int, pause: float) -> str:
    """Download to dest atomically. Returns ok | empty (no data in range) | failed."""
    part = dest.with_suffix(dest.suffix + ".part")
    for k in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=600) as r, open(part, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            with open(part, "rb") as f:
                head = f.read(8)
            if not (head[:3] == b"CDF" or head == b"\x89HDF\r\n\x1a\n"):
                raise IOError("response is not NetCDF")
            os.replace(part, dest)
            return "ok"
        except urllib.error.HTTPError as e:
            body = e.read(400).decode("utf-8", "ignore") if hasattr(e, "read") else ""
            if e.code in (400, 404) and ("no matching results" in body.lower() or "outside" in body.lower()
                                         or e.code == 404):
                part.unlink(missing_ok=True)
                return "empty"
            LOG.warning("ERDDAP %s (attempt %d): %s", e.code, k + 1, body[:200])
        except Exception as e:  # noqa: BLE001  (timeouts, 503, truncated)
            LOG.warning("ERDDAP request failed (attempt %d): %s", k + 1, e)
        part.unlink(missing_ok=True)
        time.sleep(pause * (2 ** k))
    return "failed"


def chunk_path(cache: Path, dataset_id: str, t0, t1) -> Path:
    return cache / dataset_id / f"{dataset_id}.{pd.Timestamp(t0):%Y%m%d}_{pd.Timestamp(t1):%Y%m%d}.nc"


def _chunks(start, end, n):
    d = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    while d <= end:
        e = min(d + pd.Timedelta(days=n - 1), end)
        yield d, e
        d = e + pd.Timedelta(days=1)


def fetch(cfg: dict[str, Any], start, end, lead: int = 0, all_versions: bool = False,
          offline: bool = False) -> pd.DataFrame:
    """Fetch chunks for [start, end], primary version per date plus fallbacks for its gaps."""
    cache = Path(cfg["cache_dir"])
    versions = version_table(cfg, lead)
    log_rows = []
    for t0, t1 in _chunks(start, end, int(cfg["chunk_days"])):
        days = pd.date_range(t0, t1)
        todo = versions if all_versions else [v for v in versions if any(covers(v, d) for d in days)]
        have: set = set()
        for v in todo:
            need = [d for d in days if covers(v, d) and (all_versions or d not in have)]
            if not need:
                continue
            a, b = min(need), max(need)
            dest = chunk_path(cache, v["dataset_id"], a, b)
            status = "cached" if dest.exists() else ("missing" if offline else None)
            if status is None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                var = list(cfg["variables"]) + [x for x in EXTRA_VARIABLES.get(v["version"], [])
                                                if cfg.get("context_variables", True)]
                status = _get(griddap_url(cfg["erddap"], v["dataset_id"], var, a, b, cfg["bbox"]),
                              dest, int(cfg["retries"]), float(cfg["pause_sec"]))
                time.sleep(float(cfg["pause_sec"]))
            if dest.exists():
                have |= set(_file_days(dest))
            log_rows.append({"lead": lead, "version": v["version"], "dataset_id": v["dataset_id"],
                             "start": f"{a:%Y-%m-%d}", "end": f"{b:%Y-%m-%d}", "status": status,
                             "file": str(dest) if dest.exists() else ""})
            LOG.info("C-HARM %s L%d %s..%s: %s", v["version"], lead, f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}", status)
            if all(d in have for d in days) and not all_versions:
                break
    log = pd.DataFrame(log_rows)
    if len(log):
        cache.mkdir(parents=True, exist_ok=True)
        log.to_csv(cache / f"charm_fetch_log_L{lead}_{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%d}.csv",
                   index=False)
    write_index(cfg, lead)
    return log


# ---------------------------------------------------------------------------
# Cache index / reader
# ---------------------------------------------------------------------------

def _open(path):
    import xarray as xr
    for eng in ("netcdf4", "scipy", "h5netcdf", None):
        try:
            return xr.open_dataset(path, engine=eng) if eng else xr.open_dataset(path)
        except Exception:  # noqa: BLE001
            continue
    raise IOError(f"cannot open {path}")


def _file_days(path) -> list[pd.Timestamp]:
    with _open(path) as ds:
        return [pd.Timestamp(t).normalize() for t in ds["time"].values]


def write_index(cfg: dict[str, Any], lead: int) -> pd.DataFrame:
    cache = Path(cfg["cache_dir"])
    rows = []
    vt = version_table(cfg, lead)
    for rank, v in enumerate(vt):
        for p in sorted((cache / v["dataset_id"]).glob(f"{v['dataset_id']}.*.nc")):
            try:
                for i, d in enumerate(_file_days(p)):
                    pv = primary_version(vt, d)
                    rows.append({"date": d, "version": v["version"], "rank": rank,
                                 "dataset_id": v["dataset_id"], "file": str(p), "time_index": i,
                                 "primary": bool(pv is not None and pv["version"] == v["version"])})
            except Exception as e:  # noqa: BLE001
                LOG.warning("unreadable C-HARM chunk %s: %s", p, e)
    idx = pd.DataFrame(rows, columns=["date", "version", "rank", "dataset_id", "file", "time_index", "primary"])
    if len(idx):
        cache.mkdir(parents=True, exist_ok=True)
        idx.sort_values(["date", "rank"]).to_csv(cache / f"charm_index_L{lead}.csv", index=False)
    return idx


class CharmArchive:
    """Daily C-HARM fields from the cache (latest version that has the day)."""

    def __init__(self, cfg: dict[str, Any], lead: int = 0):
        self.cfg, self.lead = cfg, lead
        p = Path(cfg["cache_dir"]) / f"charm_index_L{lead}.csv"
        idx = pd.read_csv(p, parse_dates=["date"]) if p.exists() else write_index(cfg, lead)
        self.idx = idx.sort_values(["date", "rank"]) if len(idx) else idx
        self.best = self.idx.drop_duplicates("date", keep="first").set_index("date") if len(idx) else None
        self._grid = None
        self._open_file, self._ds = None, None

    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([]) if self.best is None else self.best.index

    def entry(self, d, version: str | None = None):
        d = pd.Timestamp(d).normalize()
        if version is not None:
            m = self.idx[(self.idx["date"] == d) & (self.idx["version"] == version)]
            return None if m.empty else m.iloc[0]
        if self.best is None or d not in self.best.index:
            return None
        return self.best.loc[d]

    def _dataset(self, path):
        if self._open_file != path:
            if self._ds is not None:
                self._ds.close()
            self._ds, self._open_file = _open(path), path
        return self._ds

    def grid(self):
        """(lat ascending 1-D, lon 1-D) of the cached fields."""
        if self._grid is None and self.best is not None and len(self.best):
            ds = self._dataset(self.best.iloc[0]["file"])
            self._grid = (np.sort(np.asarray(ds["latitude"].values, float)), np.asarray(ds["longitude"].values, float))
        return self._grid

    def read(self, d, variables=None, version: str | None = None) -> dict[str, Any] | None:
        e = self.entry(d, version)
        if e is None:
            return None
        ds = self._dataset(e["file"])
        lat = np.asarray(ds["latitude"].values, float)
        out = {"version": e["version"], "dataset_id": e["dataset_id"], "lead": self.lead,
               "lat": lat, "lon": np.asarray(ds["longitude"].values, float)}
        for v in variables or VARIABLES:
            if v not in ds:
                out[v] = None
                continue
            a = np.asarray(ds[v].isel(time=int(e["time_index"])).values, np.float32)
            a[(a <= FILL_MIN) | ~np.isfinite(a)] = np.nan
            out[v] = a
        if lat[0] > lat[-1]:          # store south -> north
            out["lat"] = lat[::-1]
            for v in variables or VARIABLES:
                if out.get(v) is not None:
                    out[v] = out[v][::-1]
        return out


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def charm_profile(lat, lon) -> dict[str, Any]:
    """North-up rasterio profile of the C-HARM grid (rows north -> south)."""
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    dlat = float(np.median(np.diff(lat)))
    dlon = float(np.median(np.diff(lon)))
    return {"driver": "GTiff", "height": len(lat), "width": len(lon), "count": 1, "dtype": "float32",
            "crs": CRS.from_epsg(4326),
            "transform": from_origin(lon.min() - dlon / 2, lat.max() + dlat / 2, dlon, dlat)}


def to_north_up(a):
    return None if a is None else a[::-1]


def regrid_to(src: np.ndarray, src_prof: dict[str, Any], dst_prof: dict[str, Any], how: str = "average"):
    """Reproject one band (NaN = nodata) onto dst_prof (average / max / nearest)."""
    from rasterio.warp import Resampling, reproject
    dst = np.full((dst_prof["height"], dst_prof["width"]), np.nan, np.float32)
    reproject(np.asarray(src, np.float32), dst, src_transform=src_prof["transform"], src_crs=src_prof["crs"],
              dst_transform=dst_prof["transform"], dst_crs=dst_prof["crs"], src_nodata=np.nan,
              dst_nodata=np.nan, resampling=getattr(Resampling, how))
    return dst


def haversine_km(lat1, lon1, lat2, lon2):
    r = np.pi / 180
    a = (np.sin((lat2 - lat1) * r / 2) ** 2
         + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin((lon2 - lon1) * r / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def point_values(field: np.ndarray, lat, lon, plat, plon, radius_km=10.0, fallback_km=30.0):
    """Mean of valid pixels within radius_km (nearest valid pixel within fallback_km otherwise)."""
    out = np.full(len(plat), np.nan)
    mode = np.zeros(len(plat), np.int8)          # 1 = radius mean, 2 = nearest fallback
    if field is None:
        return out, mode
    dlat = float(np.median(np.diff(lat)))
    win = int(np.ceil(max(radius_km, fallback_km) / (111.0 * abs(dlat)))) + 1
    for k, (a, b) in enumerate(zip(plat, plon)):
        i = int(np.argmin(np.abs(lat - a)))
        j = int(np.argmin(np.abs(lon - b)))
        i0, i1, j0, j1 = max(i - win, 0), min(i + win + 1, len(lat)), max(j - win, 0), min(j + win + 1, len(lon))
        sub = field[i0:i1, j0:j1]
        if sub.size == 0:
            continue
        LA, LO = np.meshgrid(lat[i0:i1], lon[j0:j1], indexing="ij")
        dist = haversine_km(a, b, LA, LO)
        ok = np.isfinite(sub)
        inr = ok & (dist <= radius_km)
        if inr.any():
            out[k], mode[k] = float(sub[inr].mean()), 1
        elif (ok & (dist <= fallback_km)).any():
            dd = np.where(ok, dist, np.inf)
            out[k], mode[k] = float(sub.flat[int(np.argmin(dd))]), 2
    return out, mode


def coast_distance_km(field: np.ndarray, lat, lon) -> np.ndarray:
    """Distance (km) of every valid pixel to the nearest invalid one (land / outside the C-HARM domain)."""
    from scipy.ndimage import distance_transform_edt
    ok = np.isfinite(field)
    dy = 111.0 * abs(float(np.median(np.diff(lat))))
    dx = 111.0 * abs(float(np.median(np.diff(lon)))) * np.cos(np.deg2rad(float(np.nanmean(lat))))
    d = distance_transform_edt(ok, sampling=(dy, dx))
    return np.where(ok, d, np.nan)


def point_values_ext(field: np.ndarray, lat, lon, plat, plon, radius_km=10.0, fallback_km=30.0,
                     coast_exclude_km=6.0, coast_km: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """
    point_values plus coastal-edge diagnostics:
      mean / mode     as point_values (radius mean, nearest fallback)
      nearest         nearest valid pixel within fallback_km
      offcoast        radius mean over pixels >= coast_exclude_km from land / the domain edge
                      (nearest such pixel within fallback_km otherwise)
      coast_km        coast distance of the nearest valid pixel
      edge_share      share of the radius pixels closer than coast_exclude_km to the coast
    """
    n = len(plat)
    out = {k: np.full(n, np.nan) for k in ("mean", "nearest", "offcoast", "coast_km", "edge_share")}
    out["mode"] = np.zeros(n, np.int8)
    if field is None:
        return out
    cd = coast_distance_km(field, lat, lon) if coast_km is None else coast_km
    m, mode = point_values(field, lat, lon, plat, plon, radius_km, fallback_km)
    out["mean"], out["mode"] = m, mode
    dlat = float(np.median(np.diff(lat)))
    win = int(np.ceil(max(radius_km, fallback_km) / (111.0 * abs(dlat)))) + 1
    for k, (a, b) in enumerate(zip(plat, plon)):
        i = int(np.argmin(np.abs(lat - a)))
        j = int(np.argmin(np.abs(lon - b)))
        i0, i1, j0, j1 = max(i - win, 0), min(i + win + 1, len(lat)), max(j - win, 0), min(j + win + 1, len(lon))
        sub, csub = field[i0:i1, j0:j1], cd[i0:i1, j0:j1]
        if sub.size == 0:
            continue
        LA, LO = np.meshgrid(lat[i0:i1], lon[j0:j1], indexing="ij")
        dist = haversine_km(a, b, LA, LO)
        ok = np.isfinite(sub)
        if not (ok & (dist <= fallback_km)).any():
            continue
        dd = np.where(ok, dist, np.inf)
        q = int(np.argmin(dd))
        out["nearest"][k], out["coast_km"][k] = float(sub.flat[q]), float(csub.flat[q])
        inr = ok & (dist <= radius_km)
        if inr.any():
            out["edge_share"][k] = float((csub[inr] < coast_exclude_km).mean())
        off = ok & (csub >= coast_exclude_km)
        if (off & (dist <= radius_km)).any():
            out["offcoast"][k] = float(sub[off & (dist <= radius_km)].mean())
        elif (off & (dist <= fallback_km)).any():
            out["offcoast"][k] = float(sub.flat[int(np.argmin(np.where(off, dist, np.inf)))])
    return out


def version_offsets(cfg: dict[str, Any], lead: int = 0, variables=("pseudo_nitzschia", "particulate_domoic",
                                                                    "cellular_domoic")) -> pd.DataFrame:
    """Pixelwise differences between versions on dates both cover (from --all-versions fetches)."""
    from scipy.stats import spearmanr
    arc = CharmArchive(cfg, lead)
    rows = []
    if not len(arc.idx):
        return pd.DataFrame()
    for d, g in arc.idx.groupby("date"):
        vs = list(g["version"])
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                A, B = arc.read(d, variables, vs[i]), arc.read(d, variables, vs[j])
                for v in variables:
                    a, b = A.get(v), B.get(v)
                    if a is None or b is None or a.shape != b.shape:
                        continue
                    ok = np.isfinite(a) & np.isfinite(b)
                    if ok.sum() < 10:
                        continue
                    rows.append({"date": d, "lead": lead, "newer": vs[i], "older": vs[j], "variable": v,
                                 "n": int(ok.sum()), "mean_diff": float((a[ok] - b[ok]).mean()),
                                 "spearman": float(spearmanr(a[ok], b[ok]).statistic),
                                 "agree_at_0p5": float(((a[ok] >= 0.5) == (b[ok] >= 0.5)).mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["fetch", "index", "offsets"])
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--lead", type=int, default=0)
    ap.add_argument("--all-versions", action="store_true", help="fetch every version covering a date")
    ap.add_argument("--offline", action="store_true", help="cache replay: no requests, re-index only")
    a = ap.parse_args()
    c = charm_cfg(read_yaml_any(a.yaml))
    if a.action == "fetch":
        lg = fetch(c, a.start, a.end, a.lead, a.all_versions, a.offline)
        bad = lg[lg["status"] == "failed"] if len(lg) else lg
        if len(bad):
            LOG.error("%d chunks failed; rerun to resume", len(bad))
            raise SystemExit(1)
    elif a.action == "index":
        print(write_index(c, a.lead).groupby("version").size())
    else:
        out = Path(c["cache_dir"]) / f"charm_version_offsets_L{a.lead}.csv"
        version_offsets(c, a.lead).to_csv(out, index=False)
        print(out)
