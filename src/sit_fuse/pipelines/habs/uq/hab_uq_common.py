"""
Shared definitions for HAB uncertainty quantification (UQ) on SIT-FUSE products.

Reuses the HAB pipeline constants (bins, instrument prefixes, file tokens) and
the fire/smoke UQ conventions in sit_fuse.train.sf_uq_utils:

  * joint cluster id = child + num_child * parent   (Heir_DC)
  * GeoTIFF value    = joint / 1000                 (generate_output.py)
  * class index c    = concentration bin of HAB_RANGES / DOMOIC_ACID_RANGES
                       (np.histogram semantics, as insitu_hab_to_multi_hist)

HAB targets are ordinal (6 log-spaced concentration bins), so this module also
holds the ordinal / concentration utilities used to turn calibrated bin
probabilities into exceedance probabilities, credible intervals and expected
log-concentration.
"""

from __future__ import annotations

import datetime as dt
import glob
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:  # reuse the production constants when available
    from sit_fuse.pipelines.habs.hab_post_inference_constants import (
        DOMOIC_ACID_RANGES, INSTRUMENT_PREFIX, USE_KEY_FNAME_MAP)
except Exception:  # pragma: no cover - fallback copy for standalone use
    DOMOIC_ACID_RANGES = [0, 0.1, 1, 10, 100, 1000, 10000]
    INSTRUMENT_PREFIX = {"pace": "PACE_OCI", "s3a": "S3A_OLCI_ERRNT", "s3b": "S3B_OLCI_ERRNT",
                         "jpss1": "JPSS1_VIIRS", "jpss2": "JPSS2_VIIRS", "snpp": "SNPP_VIIRS",
                         "modis": "AQUA_MODIS", "goes18": "GOES18_ABI"}
    USE_KEY_FNAME_MAP = {"Pseudo_nitzschia_seriata_group": "pseudo_nitzschia_seriata_bloom",
                         "Pseudo_nitzschia_delicatissima_group": "pseudo_nitzschia_delicatissima_bloom",
                         "Alexandrium_spp": "alexandrium_bloom",
                         "Total_Phytoplankton": "total_phytoplankton",
                         "pDA": "particulate_domoic_acid"}

EPS = 1e-7
NODATA = -9999.0
CELL_RANGES = [0, 1000, 10000, 100000, 1000000, 10000000, 100000000]
GEOTIFF_SCALE = 1000.0


# ---------------------------------------------------------------------------
# Species / target definitions
# ---------------------------------------------------------------------------

@dataclass
class SpeciesSpec:
    use_key: str
    edges: list[float]
    units: str
    fname_token: str
    detection_floor: float          # lower edge used for log-space interpolation of bin 0
    alert_thresholds: list[float] = field(default_factory=list)
    measurement_sigma_dex: float = 0.2   # default log10 measurement error of the reference

    @property
    def n_classes(self) -> int:
        return len(self.edges) - 1


DEFAULT_SPECIES = {
    # CalHABMAP alert levels: Pseudo-nitzschia >= 10,000 cells/L, pDA >= 0.5 ng/mL (500 ng/L)
    "Total_Phytoplankton": SpeciesSpec("Total_Phytoplankton", CELL_RANGES, "cells/L",
                                       USE_KEY_FNAME_MAP["Total_Phytoplankton"], 100.0,
                                       [1e5, 1e6], 0.15),
    "Pseudo_nitzschia_seriata_group": SpeciesSpec(
        "Pseudo_nitzschia_seriata_group", CELL_RANGES, "cells/L",
        USE_KEY_FNAME_MAP["Pseudo_nitzschia_seriata_group"], 100.0, [1e4, 1e5], 0.2),
    "Pseudo_nitzschia_delicatissima_group": SpeciesSpec(
        "Pseudo_nitzschia_delicatissima_group", CELL_RANGES, "cells/L",
        USE_KEY_FNAME_MAP["Pseudo_nitzschia_delicatissima_group"], 100.0, [1e4, 1e5], 0.2),
    "pDA": SpeciesSpec("pDA", DOMOIC_ACID_RANGES, "ng/mL", USE_KEY_FNAME_MAP["pDA"],
                       0.01, [0.5, 1.0], 0.15),
    "Alexandrium_spp": SpeciesSpec("Alexandrium_spp", CELL_RANGES, "cells/L",
                                   USE_KEY_FNAME_MAP["Alexandrium_spp"], 100.0, [1e3], 0.25),
}


def species_spec(use_key: str, overrides: dict[str, Any] | None = None) -> SpeciesSpec:
    base = DEFAULT_SPECIES.get(use_key)
    if base is None:
        base = SpeciesSpec(use_key, CELL_RANGES, "cells/L", use_key.lower(), 100.0)
    if overrides:
        kw = {**base.__dict__, **{k: v for k, v in overrides.items() if k in base.__dict__}}
        base = SpeciesSpec(**kw)
    return base


def concentration_to_class(values, edges) -> np.ndarray:
    """
    Bin index with np.histogram semantics (left-closed, last bin closed),
    values above the last edge clipped into the last bin (production sets
    ranges[-1] = max(ranges[-1], global_max)). NaN / negative -> -1.
    """
    v = np.asarray(values, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    out = np.searchsorted(e, v, side="right") - 1
    out = np.clip(out, 0, len(e) - 2)
    out[~np.isfinite(v) | (v < e[0])] = -1
    return out.astype(np.int64)


def _log_edges(edges, floor):
    e = np.asarray(edges, dtype=np.float64).copy()
    e[0] = max(e[0], floor) if e[0] <= 0 else e[0]
    return np.log10(e)


def measurement_soft_labels(values, edges, sigma_dex, floor) -> np.ndarray:
    """
    Reference-uncertainty-aware soft labels: P(true bin | observed value) with
    log10(true) ~ N(log10(max(x, floor)), sigma_dex). Counting statistics and
    method error (microscopy settling counts, IFCB volume, pDA ELISA/LC-MS)
    make bin membership near edges genuinely uncertain. Zeros/non-detects put
    all mass below the floor (bin 0) plus the Gaussian spill above it.
    sigma_dex may be scalar or per-sample.
    """
    from scipy.stats import norm

    v = np.asarray(values, dtype=np.float64)
    s = np.broadcast_to(np.asarray(sigma_dex, dtype=np.float64), v.shape)
    le = _log_edges(edges, floor)
    mu = np.log10(np.maximum(np.where(np.isfinite(v), v, floor), floor))
    cdf = norm.cdf((le[None, 1:-1] - mu[:, None]) / np.maximum(s[:, None], 1e-6))
    cdf = np.concatenate([np.zeros((v.size, 1)), cdf, np.ones((v.size, 1))], axis=1)
    T = np.diff(cdf, axis=1)
    T = np.clip(T, 0, None)
    return T / np.maximum(T.sum(axis=1, keepdims=True), EPS)


# ---------------------------------------------------------------------------
# Ordinal / concentration summaries of bin probabilities
# ---------------------------------------------------------------------------

def class_cdf(P: np.ndarray) -> np.ndarray:
    return np.cumsum(P, axis=-1)


def exceedance_probability(P: np.ndarray, edges, threshold: float, floor: float) -> np.ndarray:
    """
    P(X >= threshold) from bin probabilities assuming log-uniform mass inside
    each bin. Exact for thresholds on bin edges; interpolates otherwise
    (e.g. pDA 0.5 ng/mL inside the [0.1, 1) bin).
    """
    le = _log_edges(edges, floor)
    t = np.log10(max(threshold, floor))
    k = int(np.clip(np.searchsorted(le, t, side="right") - 1, 0, len(le) - 2))
    frac_above = np.clip((le[k + 1] - t) / max(le[k + 1] - le[k], EPS), 0.0, 1.0)
    return P[..., k + 1:].sum(axis=-1) + frac_above * P[..., k]


def class_quantile(P: np.ndarray, q: float) -> np.ndarray:
    """Smallest class index whose CDF >= q."""
    return np.argmax(class_cdf(P) >= q - 1e-9, axis=-1)


def log10_concentration_quantile(P: np.ndarray, edges, q: float, floor: float) -> np.ndarray:
    """Quantile of log10 concentration, log-linear interpolation within bins."""
    le = _log_edges(edges, floor)
    cdf = class_cdf(P)
    k = class_quantile(P, q)
    lo_cdf = np.where(k > 0, np.take_along_axis(cdf, np.maximum(k - 1, 0)[..., None], -1)[..., 0], 0.0)
    pk = np.take_along_axis(P, k[..., None], -1)[..., 0]
    frac = np.clip((q - lo_cdf) / np.maximum(pk, EPS), 0.0, 1.0)
    return le[k] + frac * (le[k + 1] - le[k])


def expected_log10_concentration(P: np.ndarray, edges, floor: float) -> np.ndarray:
    le = _log_edges(edges, floor)
    mid = 0.5 * (le[:-1] + le[1:])
    return (P * mid).sum(axis=-1)


def ordinal_moments(P: np.ndarray):
    c = np.arange(P.shape[-1], dtype=np.float64)
    mu = (P * c).sum(axis=-1)
    var = (P * (c - mu[..., None]) ** 2).sum(axis=-1)
    return mu, np.sqrt(np.maximum(var, 0.0))


def ranked_probability_score(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """RPS (normalized by C-1): proper score for ordinal bins."""
    C = P.shape[-1]
    Y = np.eye(C)[y]
    return ((np.cumsum(P, -1) - np.cumsum(Y, -1)) ** 2)[:, :-1].sum(axis=-1) / (C - 1)


# ---------------------------------------------------------------------------
# Sensors, overpass timing
# ---------------------------------------------------------------------------

@dataclass
class SensorSpec:
    key: str
    prefix: str
    platform: str
    # mean local solar time of the daytime node (hours); callable for drifting orbits
    node_mlt_hours: Any
    node: str                         # ascending / descending
    family: str                       # sensor family for harmonization groups

    def mlt(self, date) -> float:
        return float(self.node_mlt_hours(date) if callable(self.node_mlt_hours) else self.node_mlt_hours)


def _aqua_mlt(date) -> float:
    """
    Aqua free drift (NASA Aqua status): ~13:35 MLT after the final inclination
    maneuver (2021-03-18); drift began Jan 2022; 13:45 exceeded Feb 2023;
    ~15:30 predicted by July 2026; science end projected 2026-08-26.
    Piecewise-linear between those nodes - replace with per-pixel L2 scan
    times or TLE propagation for production use.
    """
    d = pd.Timestamp(date)
    nodes = [(pd.Timestamp("2021-03-18"), 13.583), (pd.Timestamp("2022-01-15"), 13.583),
             (pd.Timestamp("2023-02-15"), 13.75), (pd.Timestamp("2026-07-15"), 15.5)]
    xs = np.array([n[0].value for n in nodes], dtype=np.float64)
    ys = np.array([n[1] for n in nodes])
    return float(np.interp(d.value, xs, ys))


SENSORS = {
    # VIIRS: 13:30 ascending (NOAA-20 ~50 min ahead of S-NPP; NOAA-21 between them)
    "snpp": SensorSpec("snpp", INSTRUMENT_PREFIX["snpp"], "Suomi-NPP", 13.42, "ascending", "viirs"),
    "jpss1": SensorSpec("jpss1", INSTRUMENT_PREFIX["jpss1"], "NOAA-20", 13.5, "ascending", "viirs"),
    "jpss2": SensorSpec("jpss2", INSTRUMENT_PREFIX["jpss2"], "NOAA-21", 13.5, "ascending", "viirs"),
    "modis": SensorSpec("modis", INSTRUMENT_PREFIX["modis"], "Aqua", _aqua_mlt, "ascending", "modis"),
    "s3a": SensorSpec("s3a", INSTRUMENT_PREFIX["s3a"], "Sentinel-3A", 10.0, "descending", "olci"),
    "s3b": SensorSpec("s3b", INSTRUMENT_PREFIX["s3b"], "Sentinel-3B", 10.0, "descending", "olci"),
    "pace": SensorSpec("pace", INSTRUMENT_PREFIX["pace"], "PACE", 13.0, "ascending", "oci"),
}


def overpass_hours_utc(sensor: str, date, lon_deg) -> np.ndarray:
    """
    Approximate overpass time (hours UTC, any array shape) at longitude
    lon_deg: local solar time ~ node MLT (at ~33N the local crossing differs
    from the equator node by only minutes for these orbits), so
    t_utc = MLT - lon/15 h. Prefer per-pixel scan time when available.
    """
    return SENSORS[sensor].mlt(date) - np.asarray(lon_deg, dtype=np.float64) / 15.0


def nominal_overpass_utc(sensor: str, date, lon_deg) -> pd.DatetimeIndex:
    """Overpass timestamps (naive UTC) for a 1-D set of longitudes."""
    hours = np.ravel(overpass_hours_utc(sensor, date, lon_deg))
    return pd.Timestamp(date).normalize() + pd.to_timedelta(hours, unit="h")


# ---------------------------------------------------------------------------
# Product discovery (same naming as the HAB pipeline / insitu_hab_to_multi_hist)
# ---------------------------------------------------------------------------

def _date_str(date) -> str:
    return pd.Timestamp(date).strftime("%Y%m%d")


DEFAULT_PATTERNS = {
    # context-free cluster GeoTIFFs (run_context_free_geotiff_generation)
    "heir": "{prefix}.{date}.L3m.DAY.tif",
    "no_heir": "{prefix}.{date}.L3m.DAY.no_heir.tif",
    "heir_proba": "{prefix}.{date}.L3m.DAY.proba.tif",
    "no_heir_proba": "{prefix}.{date}.L3m.DAY.no_heir.proba.tif",
    # optional export_head_outputs.py rasters (richest UQ inputs)
    "head_outputs": "{prefix}.{date}.L3m.DAY.head_outputs.tif",
    # production context-assigned species product (baseline comparison)
    "production": "*{date}*_DAY.{token}.tif",
}


def find_product(search_dirs, kind: str, sensor: str, date, token: str = "",
                 patterns: dict[str, str] | None = None) -> str | None:
    pats = {**DEFAULT_PATTERNS, **(patterns or {})}
    pat = pats[kind].format(prefix=SENSORS[sensor].prefix if sensor in SENSORS
                            else INSTRUMENT_PREFIX.get(sensor, sensor),
                            date=_date_str(date), token=token)
    dirs = [search_dirs] if isinstance(search_dirs, (str, os.PathLike)) else list(search_dirs)
    for d in dirs:
        hits = sorted(glob.glob(os.path.join(str(d), "**", pat), recursive=True))
        # never let '.tif' match '.no_heir.tif' / '.proba.tif'
        if kind == "heir":
            hits = [h for h in hits if not re.search(r"\.(no_heir|proba|head_outputs)\.", os.path.basename(h))]
        if kind == "no_heir":
            hits = [h for h in hits if ".proba." not in os.path.basename(h)]
        if hits:
            return hits[0]
    return None


def instrument_dirs(hab_yml: dict[str, Any], sensor: str) -> list[str]:
    """out_dir entries for one instrument of the HAB post-inference YAML."""
    block = hab_yml.get("instruments", {}).get(sensor, {})
    out = []
    for key, val in block.items():
        if isinstance(val, dict) and "out_dir" in val:
            out.append(val["out_dir"])
    return out


def pixel_centers(profile: dict[str, Any]):
    """(lat, lon) 2-D arrays of pixel centres for a north-up EPSG:4326 raster."""
    t = profile["transform"]
    h, w = profile["height"], profile["width"]
    cols = np.arange(w) + 0.5
    rows = np.arange(h) + 0.5
    lon = t.c + cols * t.a + 0.0 * t.b
    lat = t.f + rows * t.e
    LON, LAT = np.meshgrid(lon, lat)
    if t.b != 0 or t.d != 0:  # rotated grid
        C, R = np.meshgrid(cols, rows)
        LON = t.c + C * t.a + R * t.b
        LAT = t.f + C * t.d + R * t.e
    return LAT, LON


def to_utc_date(ts) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_localize(None).normalize() if pd.Timestamp(ts).tzinfo is None \
        else pd.Timestamp(ts).tz_convert("UTC").tz_localize(None).normalize()


def utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
