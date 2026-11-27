"""
Product-level fusion of TEMPO and GOES (ABI) probabilistic smoke / fire products into multi-scale
probabilistic products.

    python fuse_multiscale_products.py -c fusion_config.yaml [--stage fit|apply|all]
    # as launcher jobs (sit_fuse.train.plan_uq_jobs): fit, N apply shards (contiguous frame blocks), merge
    python fuse_multiscale_products.py -c fusion_config.yaml --stage apply --shard i/N
    python fuse_multiscale_products.py -c fusion_config.yaml --stage merge

Inputs are the geo rasters written by export_geo_products.py for each sensor (bands p_1, optional
ensemble_p1_q05 / ensemble_p1_q95, tags scene_time, classifier_split, split_stratum, threshold).
Each source keeps its own calibration; fusion operates on calibrated probabilities.

Steps
 1. Index each source (uq_viz.data.GeoProductIndex) and group granules into frames (TEMPO: one E-W scan;
    GOES: one ABI scene / time bin). sources.<s>.every_minutes (+ offset_minutes) first keeps only the first scene
    (by scene time) of each UTC bin, e.g. 30-min fusion from a 5-min geo export.
 2. Common grids: `fine` (default: GOES grid) and `coarse` (default: TEMPO grid); outputs on either or both.
 3. Change of support, per source and grid, in logit space with a per-pixel logit-normal (m, s):
      s from the ensemble interval, s = (logit q95 - logit q05) / (2 * 1.645), else `default_logit_sd`.
      fine -> coarse  fire: noisy-OR, P(>= 1 fire sub-pixel) = 1 - exp(n_eff * mean log(1 - p)),
                      n_eff = (n sub-pixels observed) ** dependence_exponent (1 = independent, 0 = max-like);
                      or max; smoke: area-weighted mean. s is aggregated as an area mean.
      coarse -> fine  nearest (block) assignment; fire optionally split to sub-pixel probability
                      1 - (1 - p) ** (1 / n) (`fire_disaggregation: split`); s^2 += disaggregation_logit_var
                      (unresolved sub-pixel heterogeneity, Nguyen et al. 2012 style inflation).
 4. Temporal alignment for each output time t (output_times: goes | tempo | union):
      GOES: scenes within +-match_window_minutes of t, pooled as the mean logit (or max), age = |dt|.
      TEMPO: most recent scan with per-granule scene time t0 <= t (causal) or nearest (non-causal),
             carried forward up to carry_forward_max_minutes with s^2 += (carry_forward_sd_per_hour * age_h)^2.
      A source is absent where it has no valid pixel within those windows (e.g., TEMPO at night).
 5. Combination (method), fitted on calibration frames against a reference:
      log_linear  m_f = b_k + sum_i w_ki m_i, s_f^2 = sum_i w_ki^2 s_i^2, separately per availability
                  pattern k (both / tempo only / goes only); p_f = sigmoid(m_f / sqrt(1 + pi s_f^2 / 8))
                  (probit approximation of the logit-normal mean). Weights fitted by weighted NLL.
      noisy_or | max | mean | tempo | goes   fixed rules (also reported as baselines).
    Optional post-hoc recalibration (temperature | platt | isotonic, sf_uq_utils.ProbabilityCalibrator)
    and decision threshold (sf_uq_utils.select_threshold, threshold_grid) on the same calibration frames.
 6. Uncertainty: fused q05 / q95 = sigmoid(m_f -+ 1.645 s_f); predictive entropy of p_f; source ages and
    availability flags are written so users can see where the product is carried forward or single-sensor.
 7. Evaluation on holdout frames (frames whose contributing sources are all holdout) for the fused product
    and every baseline, per reference (e.g. FDCC, VIIRS / FIRMS, ADP). FDCC and ADP are ABI-derived, so
    agreement of a GOES-containing product with them is not independent validation; use observed VIIRS
    references for the independent comparison.

Production: with fusion.models_from (the fitting run's output_dir or fusion_models.pkl), apply / merge / all reuse
those fitted weights, recalibration and thresholds on new source products (e.g. classifier_split: production;
output.splits: [production]); with no references the evaluation tables are empty.

Leakage: a fused frame is `calibration` only when every contributing source frame is calibration, `holdout`
only when every one is holdout; anything else is `mixed` and excluded from fitting and evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.optimize import minimize

import sf_uq_utils as U
from uq_viz import data as D

LOG = logging.getLogger("fuse_multiscale")
EPS = 1e-6
Z90 = 1.6448536269514722
SOURCES = ("tempo", "goes")
PATTERNS = {"both": (1, 1), "tempo_only": (1, 0), "goes_only": (0, 1)}


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def logit_normal_mean(m, s):
    return sigmoid(m / np.sqrt(1.0 + np.pi * s ** 2 / 8.0))


# ---------------------------------------------------------------------------
# grids and change of support
# ---------------------------------------------------------------------------

@dataclass
class Grid:
    name: str
    transform: Any
    crs: Any
    shape: tuple[int, int]

    @property
    def profile(self):
        return {"driver": "GTiff", "height": self.shape[0], "width": self.shape[1], "count": 1,
                "crs": self.crs, "transform": self.transform, "dtype": "float32"}

    @property
    def area(self):
        return U.pixel_area_km2(self.profile)


def make_grid(name: str, spec: dict, indexes: dict[str, D.GeoProductIndex]) -> Grid:
    if spec.get("from"):
        ix = indexes[spec["from"]]
        return Grid(name, ix.transform, ix.crs, ix.shape)
    w, e, s, n = spec["bounds"]
    r = float(spec["resolution"])
    shape = (int(round((n - s) / r)), int(round((e - w) / r)))
    return Grid(name, from_origin(w, n, r, r), rasterio.crs.CRS.from_user_input(spec.get("crs", "EPSG:4326")), shape)


def _warp(a, src: Grid, dst: Grid, resampling):
    out = np.full(dst.shape, np.nan, np.float32)
    reproject(a.astype(np.float32), out, src_transform=src.transform, src_crs=src.crs, dst_transform=dst.transform,
              dst_crs=dst.crs, src_nodata=np.nan, dst_nodata=np.nan, resampling=resampling)
    return out


def _cell_ratio(src: Grid, dst: Grid) -> float:
    """Approximate number of src cells per dst cell (same CRS units assumed; geographic grids are fine)."""
    return abs(dst.transform.a * dst.transform.e) / abs(src.transform.a * src.transform.e)


def change_support(p, s, src: Grid, dst: Grid, product: str, cs: dict):
    """(p, s) on src -> (p, s) on dst; s is the logit-space sd."""
    if src.shape == dst.shape and src.transform == dst.transform and src.crs == dst.crs:
        return p, s
    ratio = _cell_ratio(src, dst)
    valid = np.isfinite(p).astype(np.float32)
    if ratio > 1.0:                                     # fine -> coarse (aggregation)
        frac = _warp(np.where(valid > 0, 1.0, np.nan), src, dst, Resampling.average)
        rule = cs.get("fire_aggregate", "noisy_or") if product == "fire" else cs.get("smoke_aggregate", "mean")
        if rule == "noisy_or":
            lq = _warp(np.log1p(-np.clip(p, 0, 1 - EPS)), src, dst, Resampling.average)
            n_obs = np.clip(np.nan_to_num(frac) * ratio, 1.0, None)
            n_eff = n_obs ** float(cs.get("dependence_exponent", 1.0))
            q = 1.0 - np.exp(n_eff * lq)
        elif rule == "max":
            q = _warp(p, src, dst, Resampling.max)
        else:
            q = _warp(p, src, dst, Resampling.average)
        sv = _warp(s ** 2, src, dst, Resampling.average)
        min_cov = float(cs.get("min_coverage", 0.5))
        bad = ~np.isfinite(frac) | (np.nan_to_num(frac) < min_cov)
        q[bad] = np.nan
        return q.astype(np.float32), np.sqrt(np.where(bad, np.nan, sv)).astype(np.float32)
    # coarse -> fine (disaggregation)
    q = _warp(p, src, dst, Resampling.nearest)
    sv = _warp(s ** 2, src, dst, Resampling.nearest)
    n = max(1.0 / ratio, 1.0)
    if product == "fire" and cs.get("fire_disaggregation", "assign") == "split":
        q = 1.0 - (1.0 - np.clip(q, 0, 1 - EPS)) ** (1.0 / n)
    sv = sv + float(cs.get("disaggregation_logit_var", 0.25 if n > 1 else 0.0))
    return q.astype(np.float32), np.sqrt(sv).astype(np.float32)


# ---------------------------------------------------------------------------
# sources and frames
# ---------------------------------------------------------------------------

class Source:
    def __init__(self, name: str, cfg: dict, product: str):
        self.name, self.cfg, self.product = name, cfg, product
        self.model = cfg.get("model", "hist_gradient_boosting")
        self.idx = D.GeoProductIndex(cfg["geo_dir"], product, [self.model], cfg.get("manifest"))
        sc = self.idx.scenes(self.model)
        every = cfg.get("every_minutes")            # e.g. 30: first scene (scene time) of each UTC bin only
        if every and len(sc) and sc["scene_time"].notna().any():
            off = pd.Timedelta(minutes=float(cfg.get("offset_minutes", 0) or 0))
            sc = sc.sort_values(["scene_time", "pair_id"])
            timed = sc[sc.scene_time.notna()]
            key = (timed.scene_time - off).dt.floor(f"{int(every)}min")
            n0 = len(sc)
            sc = pd.concat([timed[~key.duplicated()], sc[sc.scene_time.isna()]])
            LOG.info("%s: every_minutes=%s keeps %d of %d scenes", name, every, len(sc), n0)
        fr = cfg.get("frames", {})
        default_mode = "scan" if name == "tempo" else "time"
        self.frames = D.assign_frames(sc, fr.get("mode", default_mode), int(fr.get("minutes", 10)))
        self.grid = Grid(name, self.idx.transform, self.idx.crs, self.idx.shape)
        self.default_sd = float(cfg.get("default_logit_sd", 1.0))
        thr = self.idx.df.threshold.dropna()
        self.threshold = float(thr.iloc[0]) if len(thr) else 0.5
        LOG.info("%s: %d scenes in %d frames, grid %s", name, len(sc), self.frames.frame_id.nunique(), self.idx.shape)

    def frame_table(self) -> pd.DataFrame:
        g = self.frames.groupby("frame_id")
        t = pd.DataFrame({"frame_time": g.frame_time.first(), "frame_start": g.frame_start.first(),
                          "frame_end": g.frame_end.first(), "split": g.split.agg(lambda x: "|".join(sorted(set(x)))),
                          "stratum": g.stratum.first(), "date": g.date.first()})
        return t.reset_index().sort_values("frame_time")

    def read_frame(self, frame_id: str):
        """Mosaic (p, s, time_seconds) on the source grid; time is the per-pixel granule scene time."""
        d = self.frames[self.frames.frame_id == frame_id].sort_values("scene_time")
        p = s = tt = None
        rows = self.idx.df[(self.idx.df.model == self.model) & self.idx.df.pair_id.isin(d.pair_id)]
        for _, r in rows.sort_values("scene_time").iterrows():
            a = D._read_band(r.path, "p_1")
            if a is None:
                continue
            lo, hi = D._read_band(r.path, "ensemble_p1_q05"), D._read_band(r.path, "ensemble_p1_q95")
            if lo is not None and hi is not None:
                sd = (logit(hi) - logit(lo)) / (2 * Z90)
                sd = np.where(np.isfinite(sd), np.maximum(sd, float(self.cfg.get("min_logit_sd", 0.05))),
                              self.default_sd)
            else:
                sd = np.full(a.shape, self.default_sd, np.float32)
            ts = np.full(a.shape, r.scene_time.timestamp(), np.float64)
            if p is None:
                p, s, tt = a.copy(), sd.astype(np.float32), np.where(np.isfinite(a), ts, np.nan)
            else:
                new = np.isnan(p) & np.isfinite(a)
                p[new], s[new], tt[new] = a[new], sd[new], ts[new]
        return p, s, tt


@dataclass
class Layer:
    """One source resampled to an output grid at one output time."""
    m: np.ndarray       # logit mean (NaN = absent)
    s: np.ndarray       # logit sd
    age_min: np.ndarray
    split: str


def _pool_goes(items, how="mean_logit"):
    ms = np.stack([i[0] for i in items])
    ss = np.stack([i[1] for i in items])
    ages = np.stack([i[2] for i in items])
    ok = np.isfinite(ms)
    n = ok.sum(0)
    if how == "max":
        k = np.nanargmax(np.where(ok, ms, -np.inf), axis=0)
        m = np.take_along_axis(ms, k[None], 0)[0]
        s = np.take_along_axis(ss, k[None], 0)[0]
    else:
        m = np.where(n > 0, np.nansum(ms, 0) / np.maximum(n, 1), np.nan)
        # mean of n correlated looks: keep the mean member variance (no 1/n shrinkage; ABI looks are not independent)
        s = np.sqrt(np.where(n > 0, np.nansum(np.where(ok, ss ** 2, 0), 0) / np.maximum(n, 1), np.nan))
    age = np.where(n > 0, np.nanmin(np.where(ok, ages, np.inf), 0), np.nan)
    return m, s, age


class Aligner:
    def __init__(self, sources: dict[str, Source], grids: dict[str, Grid], product: str, cfg: dict):
        self.src, self.grids, self.product, self.cfg = sources, grids, product, cfg
        self.tc = cfg.get("temporal", {})
        self.cs = cfg.get("change_of_support", {})
        self.tables = {k: v.frame_table() for k, v in sources.items()}
        self._cache: dict[tuple, Any] = {}

    def output_times(self) -> pd.DataFrame:
        mode = self.tc.get("output_times", "goes")
        if mode == "union":
            t = pd.concat([self.tables[k].assign(sensor=k) for k in self.tables])
        else:
            t = self.tables[mode].assign(sensor=mode)
        t = t.drop_duplicates("frame_time").sort_values("frame_time").reset_index(drop=True)
        for k in ("dates",):
            if self.tc.get(k):
                t = t[t.date.astype(str).isin([str(x) for x in self.tc[k]])]
        return t

    def _on_grid(self, name: str, fid: str, grid: Grid):
        key = (name, fid, grid.name)
        if key not in self._cache:
            p, s, tt = self.src[name].read_frame(fid)
            if p is None:
                self._cache[key] = None
            else:
                q, sq = change_support(p, s, self.src[name].grid, grid, self.product, self.cs)
                tq = _warp(tt, self.src[name].grid, grid, Resampling.nearest)
                self._cache[key] = (q, sq, tq)
            if len(self._cache) > int(self.cfg.get("cache_frames", 64)):
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]

    def layer(self, name: str, t: pd.Timestamp, grid: Grid) -> Layer | None:
        tab = self.tables.get(name)
        if tab is None:
            return None
        ts = t.timestamp()
        if name == "goes":
            win = pd.Timedelta(minutes=float(self.tc.get("match_window_minutes", 10)))
            sel = tab[(tab.frame_time >= t - win) & (tab.frame_time <= t + win)]
            items, splits = [], set()
            for _, r in sel.iterrows():
                g = self._on_grid(name, r.frame_id, grid)
                if g is None:
                    continue
                q, sq, tq = g
                items.append((logit(q), sq, np.abs(tq - ts) / 60.0))
                splits.add(r.split)
            if not items:
                return None
            m, s, age = _pool_goes(items, self.tc.get("goes_pool", "mean_logit"))
            return Layer(m, s, age, "|".join(sorted(splits)))
        # tempo: carry forward from the most recent scan (or nearest when causal: false)
        mx = pd.Timedelta(minutes=float(self.tc.get("carry_forward_max_minutes", 120)))
        causal = bool(self.tc.get("causal", True))
        lo = t - mx
        hi = t + (pd.Timedelta(minutes=float(self.tc.get("tempo_lookahead_minutes", 5))) if causal else mx)
        sel = tab[(tab.frame_end >= lo) & (tab.frame_start <= hi)].copy()
        if sel.empty:
            return None
        sel["dt"] = (sel.frame_time - t).abs()
        rate = float(self.tc.get("carry_forward_sd_per_hour", 0.5))
        m = s = age = None
        splits = set()
        for _, r in sel.sort_values("dt").iterrows():          # nearest scan first, older scans fill gaps
            g = self._on_grid(name, r.frame_id, grid)
            if g is None:
                continue
            q, sq, tq = g
            a = (ts - tq) / 60.0
            ok = np.isfinite(q) & (np.abs(a) <= mx.total_seconds() / 60.0)
            if causal:
                ok &= a >= -float(self.tc.get("tempo_lookahead_minutes", 5))
            if not ok.any():
                continue
            splits.add(r.split)
            mm = np.where(ok, logit(q), np.nan)
            infl = np.sqrt(sq ** 2 + (rate * np.abs(a) / 60.0) ** 2)
            if m is None:
                m, s, age = mm, np.where(ok, infl, np.nan), np.where(ok, np.abs(a), np.nan)
            else:
                new = np.isnan(m) & ok
                m[new], s[new], age[new] = mm[new], infl[new], np.abs(a)[new]
        if m is None:
            return None
        return Layer(m, s, age, "|".join(sorted(splits)))


def frame_split(layers: dict[str, Layer | None]) -> str:
    """calibration / holdout / dev / production when every contributing source frame has that split, else mixed."""
    parts = set()
    for L in layers.values():
        if L is not None:
            parts |= set(L.split.split("|"))
    if len(parts) == 1:
        return next(iter(parts))
    return "mixed" if parts else "none"


# ---------------------------------------------------------------------------
# combination rules
# ---------------------------------------------------------------------------

def avail(layers):
    return {k: (np.isfinite(L.m) if L is not None else None) for k, L in layers.items()}


def pattern_masks(layers, shape):
    a = {k: (np.isfinite(layers[k].m) if layers.get(k) is not None else np.zeros(shape, bool)) for k in SOURCES}
    return {"both": a["tempo"] & a["goes"], "tempo_only": a["tempo"] & ~a["goes"],
            "goes_only": ~a["tempo"] & a["goes"]}


def source_probs(layers, shape):
    """Per-source calibrated probability on the output grid (logit-normal mean; NaN where absent)."""
    P = {}
    for k in SOURCES:
        L = layers.get(k)
        P[k] = logit_normal_mean(L.m, L.s) if L is not None else np.full(shape, np.nan)
    return P["tempo"], P["goes"]


def rule(method: str, pt, pg):
    """Fixed-rule fused probability (baselines); NaN where no source is present."""
    if method == "tempo":
        return pt
    if method == "goes":
        return pg
    st = np.stack([pt, pg])
    ok = np.isfinite(st)
    any_ = ok.any(0)
    if method == "max":
        out = np.where(ok, st, -1).max(0)
    elif method == "mean":
        out = np.where(ok, st, 0).sum(0) / np.maximum(ok.sum(0), 1)
    elif method == "noisy_or":
        out = 1.0 - np.prod(np.where(ok, 1.0 - st, 1.0), 0)
    else:
        raise ValueError(f"unknown rule {method}")
    return np.where(any_, out, np.nan)


class LogLinearPool:
    """m_f = b_k + sum_i w_ki m_i, s_f^2 = sum_i w_ki^2 s_i^2 per availability pattern k."""

    def __init__(self, l2: float = 1e-3):
        self.params: dict[str, np.ndarray] = {}
        self.l2 = l2

    @staticmethod
    def _design(k, mt, mg):
        on = PATTERNS[k]
        cols = [c for c, o in zip((mt, mg), on) if o]
        return np.stack(cols, 1)

    def fit(self, samples: dict[str, dict[str, np.ndarray]]):
        for k, d in samples.items():
            if d is None or len(d["y"]) < 20 or d["y"].sum() < 2:
                LOG.warning("pattern %s: too few calibration samples; using unit weights", k)
                self.params[k] = np.r_[0.0, np.ones(sum(PATTERNS[k]))]
                continue
            M = self._design(k, d["mt"], d["mg"])
            S = self._design(k, d["st"], d["sg"])
            y, w = d["y"].astype(float), d["w"] / d["w"].sum()

            def nll(th):
                mf = th[0] + M @ th[1:]
                sf = np.sqrt((S ** 2) @ (th[1:] ** 2))
                p = np.clip(logit_normal_mean(mf, sf), EPS, 1 - EPS)
                return -(w * (y * np.log(p) + (1 - y) * np.log1p(-p))).sum() + self.l2 * (th[1:] ** 2).sum()

            x0 = np.r_[0.0, np.ones(M.shape[1])]
            res = minimize(nll, x0, method="L-BFGS-B", bounds=[(-20, 20)] + [(0.0, 5.0)] * M.shape[1])
            self.params[k] = res.x
            LOG.info("log-linear pool %s: b=%.3f w=%s (n=%d, pos=%d)", k, res.x[0], np.round(res.x[1:], 3),
                     len(y), int(y.sum()))
        return self

    def predict(self, layers, shape):
        mt = layers["tempo"].m if layers.get("tempo") is not None else np.full(shape, np.nan)
        st = layers["tempo"].s if layers.get("tempo") is not None else np.full(shape, np.nan)
        mg = layers["goes"].m if layers.get("goes") is not None else np.full(shape, np.nan)
        sg = layers["goes"].s if layers.get("goes") is not None else np.full(shape, np.nan)
        mf = np.full(shape, np.nan)
        sf = np.full(shape, np.nan)
        for k, msk in pattern_masks(layers, shape).items():
            if not msk.any() or k not in self.params:
                continue
            th = self.params[k]
            M = self._design(k, mt[msk], mg[msk])
            S = self._design(k, st[msk], sg[msk])
            mf[msk] = th[0] + M @ th[1:]
            sf[msk] = np.sqrt((S ** 2) @ (th[1:] ** 2))
        return mf, sf

    def describe(self):
        return {k: {"bias": float(v[0]), "weights": dict(zip([s for s, o in zip(SOURCES, PATTERNS[k]) if o],
                                                               map(float, v[1:])))} for k, v in self.params.items()}


# ---------------------------------------------------------------------------
# references and sampling
# ---------------------------------------------------------------------------

def load_references(cfg: dict) -> dict[str, D.RasterReference]:
    return {n: D.RasterReference(n, {"match": "time", "tolerance_minutes": 10, **r})
            for n, r in (cfg.get("references") or {}).items()}


def ref_on_grid(ref: D.RasterReference, t: pd.Timestamp, grid: Grid):
    return ref.raster(f"fused_{t:%Y%m%dT%H%M%S}Z", grid)


def sample_pixels(y, valid, rng, max_neg, max_pos):
    pos = np.flatnonzero(valid & (y == 1))
    neg = np.flatnonzero(valid & (y == 0))
    wp = wn = 1.0
    if len(pos) > max_pos:
        wp = len(pos) / max_pos
        pos = rng.choice(pos, max_pos, replace=False)
    if len(neg) > max_neg:
        wn = len(neg) / max_neg
        neg = rng.choice(neg, max_neg, replace=False)
    idx = np.r_[pos, neg]
    return idx, np.r_[np.full(len(pos), wp), np.full(len(neg), wn)]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

BASELINES = ("tempo", "goes", "noisy_or", "max", "mean")


def build(cfg):
    product = cfg.get("product", "fire")
    sources = {k: Source(k, v, product) for k, v in cfg["sources"].items() if k in SOURCES}
    if not sources:
        raise ValueError("sources: need at least one of tempo / goes")
    idx = {k: s.idx for k, s in sources.items()}
    gspec = cfg.get("grids") or {"fine": {"from": "goes" if "goes" in sources else "tempo"},
                                 "coarse": {"from": "tempo" if "tempo" in sources else "goes"}}
    grids = {n: make_grid(n, g, idx) for n, g in gspec.items()}
    return product, sources, grids, Aligner(sources, grids, product, cfg)


def iterate(al: Aligner, grid: Grid, splits=None, select=None):
    """(time, split, layers) per output frame; select: optional set of frame positions to keep (shards)."""
    for k, (_, r) in enumerate(al.output_times().iterrows()):
        if select is not None and k not in select:
            continue
        t = r.frame_time
        layers = {k: al.layer(k, t, grid) for k in al.src}
        sp = frame_split(layers)
        if splits and sp not in splits:
            continue
        if all(L is None for L in layers.values()):
            continue
        yield t, sp, layers


def collect(al, grid, refname, refs, split, rng, fc):
    """Calibration samples per availability pattern: (mt, st, mg, sg, y, w) + baseline probabilities."""
    out = {k: [] for k in PATTERNS}
    for t, sp, layers in iterate(al, grid, {split}):
        R = ref_on_grid(refs[refname], t, grid)
        if R is None:
            continue
        shape = grid.shape
        y = np.where(np.isfinite(R), (R >= 0.5).astype(int), -1)
        for k, msk in pattern_masks(layers, shape).items():
            v = msk & (y >= 0)
            if not v.any():
                continue
            ii, ww = sample_pixels(y.ravel(), v.ravel(), rng, int(fc.get("max_negatives_per_frame", 20000)),
                                   int(fc.get("max_positives_per_frame", 20000)))
            g = lambda L, a: (getattr(L, a).ravel()[ii] if L is not None else np.full(len(ii), np.nan))
            out[k].append(dict(mt=g(layers.get("tempo"), "m"), st=g(layers.get("tempo"), "s"),
                               mg=g(layers.get("goes"), "m"), sg=g(layers.get("goes"), "s"),
                               y=y.ravel()[ii], w=ww))
    return {k: ({f: np.concatenate([d[f] for d in v]) for f in v[0]} if v else None) for k, v in out.items()}


def fit(cfg, product, sources, grids, al):
    fc = cfg.get("fusion", {})
    refs = load_references(cfg)
    refname = fc.get("fit_reference") or next(iter(refs))
    rng = np.random.default_rng(int(fc.get("seed", 0)))
    out = Path(cfg["output_dir"])
    models = {}
    for gname in fc.get("grids", list(grids)):
        grid = grids[gname]
        samples = collect(al, grid, refname, refs, fc.get("fit_split", "calibration"), rng, fc)
        n = {k: (0 if v is None else int(len(v["y"]))) for k, v in samples.items()}
        LOG.info("grid %s: calibration samples %s", gname, n)
        pool = LogLinearPool(float(fc.get("l2", 1e-3))).fit(samples)
        # recalibration + threshold on the same calibration samples, using the fused probabilities
        P, Y, W, B = [], [], [], {b: [] for b in BASELINES}
        for k, d in samples.items():
            if d is None:
                continue
            pt, pg = logit_normal_mean(d["mt"], d["st"]), logit_normal_mean(d["mg"], d["sg"])
            for b in BASELINES:
                B[b].append(rule(b, pt, pg))
            th = pool.params[k]
            M = LogLinearPool._design(k, d["mt"], d["mg"])
            S = LogLinearPool._design(k, d["st"], d["sg"])
            P.append(logit_normal_mean(th[0] + M @ th[1:], np.sqrt((S ** 2) @ (th[1:] ** 2))))
            Y.append(d["y"])
            W.append(d["w"])
        cal = None
        thr, score = 0.5, float("nan")
        if P:
            P, Y, W = np.concatenate(P), np.concatenate(Y), np.concatenate(W)
            P2 = np.column_stack([1 - P, P])
            meth = fc.get("recalibration", "none")
            if meth != "none":
                cal = U.ProbabilityCalibrator(meth).fit(P2, np.eye(2)[Y], W)
                P2 = cal.transform(P2)
            grid_t = U.threshold_grid(fc.get("threshold_grid", {"scale": "log", "min": 1e-4, "max": 0.95,
                                                                 "n": 100}))
            thr, score = U.select_threshold(P2[:, 1], Y, W, fc.get("threshold_metric", "f1"), grid_t)
            bthr = {}
            for b, v in B.items():
                v = np.concatenate(v)
                ok = np.isfinite(v)
                if ok.any() and Y[ok].sum() > 0:
                    bthr[b] = U.select_threshold(v[ok], Y[ok], W[ok], fc.get("threshold_metric", "f1"), grid_t)[0]
        else:
            bthr = {}
        models[gname] = {"pool": pool, "calibrator": cal, "threshold": thr, "baseline_thresholds": bthr}
        U.save_json(out / f"fusion_model_{gname}.json", {
            "product": product, "grid": gname, "fit_reference": refname, "samples": n,
            "log_linear_pool": pool.describe(), "recalibration": None if cal is None else cal.describe(),
            "threshold": thr, "baseline_thresholds": bthr, f"calibration_{fc.get('threshold_metric', 'f1')}": score,
            "temporal": cfg.get("temporal", {}), "change_of_support": cfg.get("change_of_support", {})})
    import pickle
    (out / "fusion_models.pkl").write_bytes(pickle.dumps(models))
    return models


def fused(model, layers, shape):
    mf, sf = model["pool"].predict(layers, shape)
    p = logit_normal_mean(mf, sf)
    lo, hi = sigmoid(mf - Z90 * sf), sigmoid(mf + Z90 * sf)
    cal = model["calibrator"]
    if cal is not None:
        ok = np.isfinite(p)
        for a in (p, lo, hi):
            v = a[ok]
            a[ok] = cal.transform(np.column_stack([1 - v, v]))[:, 1]
    return p, lo, hi, sf


def _entropy(p):
    q = np.clip(p, EPS, 1 - EPS)
    return -(q * np.log(q) + (1 - q) * np.log1p(-q)) / np.log(2)


def apply(cfg, product, sources, grids, al, models, shard=None):
    """Fused rasters for every output frame (or one shard of frames) + rows / evaluation accumulators."""
    from sit_fuse.parallel_utils import shard_indices
    select = None
    if shard is not None:
        select = set(shard_indices(len(al.output_times()), shard, "block"))   # contiguous: TEMPO scans reused
    fc = cfg.get("fusion", {})
    out = Path(cfg["output_dir"])
    refs = load_references(cfg)
    write_splits = set(cfg.get("output", {}).get("splits", ["calibration", "holdout", "dev", "mixed", "none"]))
    eval_split = fc.get("eval_split", "holdout")
    rows, acc = [], {}
    for gname, model in models.items():
        grid = grids[gname]
        area = grid.area
        for t, sp, layers in iterate(al, grid, select=select):
            shape = grid.shape
            p, lo, hi, sf = fused(model, layers, shape)
            thr = model["threshold"]
            pm = pattern_masks(layers, shape)
            nsrc = sum((np.isfinite(L.m).astype(np.int8) if L is not None else 0) for L in layers.values())
            bands = {"p_1": p, "pred_label": np.where(np.isfinite(p), (p >= thr).astype(np.float32), np.nan),
                     "ensemble_p1_q05": lo, "ensemble_p1_q95": hi, "predictive_entropy": _entropy(p),
                     "fusion_logit_sd": sf, "n_sources": np.where(np.isfinite(p), nsrc, np.nan)}
            for k in SOURCES:
                L = layers.get(k)
                bands[f"{k}_p1"] = logit_normal_mean(L.m, L.s) if L is not None else np.full(shape, np.nan)
                bands[f"{k}_age_minutes"] = L.age_min if L is not None else np.full(shape, np.nan)
            bands = {k: np.asarray(v, np.float32) for k, v in bands.items()}
            pid = f"fused_{product}_{gname}_{t:%Y%m%dT%H%M%S}Z"
            if sp in write_splits:
                prof = {k: v for k, v in grid.profile.items() if k not in ("count", "dtype")}
                tags = {"pair_id": pid, "product": product, "model": "fused_log_linear", "scene_time": t.isoformat(),
                        "date": f"{t:%Y-%m-%d}", "classifier_split": sp, "split_stratum": f"{sp}", "threshold": thr,
                        "grid": gname, "sources": ",".join(k for k, L in layers.items() if L is not None),
                        "fusion": json.dumps(models[gname]["pool"].describe())}
                U.write_band_stack(out / gname / "fused" / "rasters" / f"{pid}.tif", bands, prof, tags)
            okp = np.isfinite(p)
            rows.append({"grid": gname, "pair_id": pid, "scene_time": t, "split": sp,
                         **{f"frac_{k}": float(m[okp].mean()) if okp.any() else np.nan for k, m in pm.items()},
                         "area_expected_km2": float(np.nansum(p * area)),
                         "area_discrete_km2": float(np.nansum((p >= thr) * area)),
                         "area_q05_km2": float(np.nansum(lo * area)), "area_q95_km2": float(np.nansum(hi * area))})
            if sp != eval_split:
                continue
            for rn, ref in refs.items():
                R = ref_on_grid(ref, t, grid)
                if R is None:
                    continue
                y = np.where(np.isfinite(R), (R >= 0.5).astype(int), -1)
                common = okp & (y >= 0)
                pt, pg = source_probs(layers, shape)
                cand = {"fused": p, **{b: rule(b, pt, pg) for b in BASELINES}}
                for name, q in cand.items():
                    # all methods on the same pixels: fused valid, reference valid, method defined
                    for scope, msk in (("all", common & np.isfinite(q)),
                                       *((k, common & m & np.isfinite(q)) for k, m in pm.items())):
                        if not msk.any():
                            continue
                        a = acc.setdefault((gname, rn, name, scope), {"p": [], "y": [], "w": []})
                        a["p"].append(q[msk])
                        a["y"].append(y[msk])
                        a["w"].append(area[msk])
    return rows, acc


def reliability_table(acc, n_bins=10) -> pd.DataFrame:
    """Area-weighted reliability (mean probability vs observed frequency) per grid / reference / method / scope."""
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for (gname, rn, name, scope), a in acc.items():
        p, y, w = (np.concatenate(a[f]) for f in ("p", "y", "w"))
        if len(y) == 0:
            continue
        b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
        for k in range(n_bins):
            m = b == k
            if not m.any():
                continue
            out.append({"grid": gname, "reference": rn, "method": name, "scope": scope, "bin": k,
                        "bin_lo": edges[k], "bin_hi": edges[k + 1], "n": int(m.sum()), "weight": float(w[m].sum()),
                        "mean_p": float(np.average(p[m], weights=w[m])),
                        "obs_freq": float(np.average(y[m], weights=w[m]))})
    return pd.DataFrame(out, columns=["grid", "reference", "method", "scope", "bin", "bin_lo", "bin_hi", "n",
                                      "weight", "mean_p", "obs_freq"])


def load_models(cfg: dict):
    """Fitted fusion models: fusion.models_from (a fusion_models.pkl or the output_dir of the fitting run, e.g. to
    apply weights fitted on the evaluation periods to a production period), else <output_dir>/fusion_models.pkl."""
    import pickle
    src = (cfg.get("fusion") or {}).get("models_from")
    p = Path(src) if src else Path(cfg["output_dir"]) / "fusion_models.pkl"
    if p.is_dir():
        p = p / "fusion_models.pkl"
    if not p.exists():
        raise FileNotFoundError(f"{p}: run --stage fit first (or set fusion.models_from)")
    return pickle.loads(p.read_bytes())


def evaluate_fused(cfg, models, rows, acc):
    """fused_frames.csv and fusion_evaluation.csv from the frame rows and pooled holdout pixels."""
    out = Path(cfg["output_dir"])
    eval_split = cfg.get("fusion", {}).get("eval_split", "holdout")
    order = list(models)
    rows = sorted(rows, key=lambda r: (order.index(r["grid"]), r["scene_time"]))   # shards -> single-run order
    pd.DataFrame(rows).to_csv(out / "fused_frames.csv", index=False)
    res = []
    from sklearn.metrics import average_precision_score, roc_auc_score
    for (gname, rn, name, scope), a in acc.items():
        p, y, w = (np.concatenate(a[f]) for f in ("p", "y", "w"))
        if len(y) == 0:
            continue
        thr = models[gname]["threshold"] if name == "fused" else \
            float(models[gname].get("baseline_thresholds", {}).get(name, models[gname]["threshold"]))
        pm = U.probability_metrics(np.column_stack([1 - p, p]), y, np.eye(2)[y], w, 2, threshold=thr)
        r = {"grid": gname, "reference": rn, "method": name, "scope": scope, "n": int(len(y)),
             "prevalence": float((w * y).sum() / w.sum()), "nll": pm["nll_vs_reference"],
             "brier": pm["brier_vs_reference"], "ece": pm["classwise_ece_vs_reference"],
             "f1": pm["hard_vs_reference"]["target_macro_f1"], "iou": pm["hard_vs_reference"]["target_macro_iou"],
             "threshold": thr}
        if 0 < y.sum() < len(y):
            r["roc_auc"] = float(roc_auc_score(y, p, sample_weight=w))
            r["pr_auc"] = float(average_precision_score(y, p, sample_weight=w))
        res.append(r)
    res = pd.DataFrame(res)
    res.to_csv(out / "fusion_evaluation.csv", index=False)
    reliability_table(acc, int(cfg.get("fusion", {}).get("reliability_bins", 10))).to_csv(
        out / "fusion_reliability.csv", index=False)
    LOG.info("evaluation (%s frames):\n%s", eval_split, res.to_string(index=False) if len(res) else "none")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--stage", choices=["fit", "apply", "merge", "all"], default="all")
    ap.add_argument("--shard", default=None, metavar="i/N",
                    help="with --stage apply: only frame block i of N (0-based); writes its rasters and a partial "
                         "result to <output_dir>/fusion_shards/ for --stage merge")
    ap.add_argument("--n-shards", type=int, default=None, help="with --stage merge (default: inferred)")
    ap.add_argument("-n", "--num-threads", default=None, help="CPU thread budget (overrides compute.num_threads)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import pickle
    from sit_fuse.parallel_utils import (atomic_write_bytes, infer_n_shards, parse_shard, setup_compute,
                                         shard_files, shard_path)
    cfg = yaml.safe_load(Path(a.config).read_text())
    setup_compute(cfg, a.num_threads)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    shard, sdir = parse_shard(a.shard), out / "fusion_shards"
    if a.stage == "merge":                 # combine apply shards; same tables as a single apply
        models = load_models(cfg)
        n = a.n_shards or infer_n_shards(sdir, "apply")
        if not n:
            raise FileNotFoundError(f"No apply shards in {sdir}; run --stage apply --shard i/N first.")
        rows, acc = [], {}
        for p in shard_files(sdir, "apply", n):
            r, ac = pickle.loads(p.read_bytes())
            rows += r
            for k, v in ac.items():
                d = acc.setdefault(k, {"p": [], "y": [], "w": []})
                for f in d:
                    d[f] += v[f]
        evaluate_fused(cfg, models, rows, acc)
        return
    product, sources, grids, al = build(cfg)
    reuse = bool((cfg.get("fusion") or {}).get("models_from"))
    if a.stage == "fit" and reuse:
        raise ValueError("fusion.models_from is set: this config applies fitted models; run --stage apply / merge")
    if a.stage in ("fit", "all") and not reuse:
        models = fit(cfg, product, sources, grids, al)
    else:
        models = load_models(cfg)
    if a.stage in ("apply", "all"):
        rows, acc = apply(cfg, product, sources, grids, al, models, shard)
        if shard is not None:
            atomic_write_bytes(shard_path(sdir, "apply", shard), pickle.dumps((rows, acc)))
            LOG.info("apply shard %d/%d: %d frames", shard[0], shard[1], len(rows))
        else:
            evaluate_fused(cfg, models, rows, acc)


if __name__ == "__main__":
    main()
