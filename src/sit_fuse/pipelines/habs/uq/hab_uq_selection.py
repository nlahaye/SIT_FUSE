"""
Model selection, decision rules and interval calibration for the HAB UQ bundles.

Selection (uq.classifier.selection)
  metric: rps | rpss | exceedance_brier | qwk | composite     (default rps, the previous behaviour)
  weights: {rpss: 0.5, exceedance_brier: 0.5}                 composite only; any of the component names
  Losses are lower-is-better and scale-free where possible:
    rps               mean ranked probability score
    rpss              1 - RPSS vs the training climatology
    exceedance_brier  mean over spec.alert_thresholds of BS / BS_climatology (1 - BSS)
    qwk               1 - quadratic weighted kappa of the decision rule (hard, not proper)
  Probabilities are chosen by proper scores (rps / rpss / exceedance_brier); qwk is best
  kept for the decision rule below.

Decision rule (uq.decision)
  rule: argmax | median | expected | auto       hard class from calibrated probabilities
  auto picks among `candidates` on the calibration samples by `metric` (qwk | balanced_accuracy |
  macro_f1). median = ordinal median bin (first bin with CDF >= 0.5); expected = rounded mean bin.
  argmax of a diffuse ordinal distribution collapses to the majority bin; median / expected
  follow the predictive distribution's location instead.

Intervals (uq.intervals)
  Observations at or below spec.detection_floor are left-censored (zeros, non-detects are stored
  as log10(floor)). The log-linear bin-0 quantile always lies above log10(floor), so an
  uncensored coverage check scores every non-detect as a miss. Coverage here counts a censored
  observation as covered when the lower interval bound falls in bin 0 (the interval reaches
  the non-detect region), and uncensored observations as usual.
  Tail grid default 0.25 ... 0 (intervals from 50 % to full support).
  recalibrate: true      fit the tail mass a (interval = [q_a, q_{1-a}]) on calibration samples
                         so censored-aware coverage meets `target`; a may widen (a < 0.05) or
                         narrow (a > 0.05) the nominal 90 % interval. Optional pad_dex widening
                         when the tail grid cannot reach the target.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sit_fuse.pipelines.habs.uq.hab_uq_common import (
    EPS, class_cdf, class_quantile, exceedance_probability, log10_concentration_quantile,
    ranked_probability_score, _log_edges)

COMPONENTS = ("rps", "rpss", "exceedance_brier", "qwk")
DEFAULT_COMPOSITE = {"rpss": 0.5, "exceedance_brier": 0.5}
DECISION_RULES = ("argmax", "median", "expected")
DEFAULT_TAILS = (0.25, 0.20, 0.15, 0.125, 0.10, 0.075, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01, 0.0075, 0.005,
                 0.0025, 0.001, 0.0)


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------

def decide(P: np.ndarray, rule: str = "argmax") -> np.ndarray:
    P = np.asarray(P, float)
    if rule in (None, "argmax"):
        return P.argmax(-1)
    if rule == "median":
        return class_quantile(P, 0.5)
    if rule == "expected":
        c = np.arange(P.shape[-1], dtype=float)
        return np.clip(np.rint((P * c).sum(-1)), 0, P.shape[-1] - 1).astype(int)
    raise ValueError(f"unknown decision rule {rule!r} (argmax | median | expected | auto)")


def _qwk(y, p, C, w):
    from sklearn.metrics import cohen_kappa_score
    if len(np.unique(np.r_[y, p])) < 2:
        return 0.0
    return float(cohen_kappa_score(y, p, labels=list(range(C)), weights="quadratic", sample_weight=w))


def hard_metric(y, p, w, C, metric: str) -> float:
    from sklearn.metrics import balanced_accuracy_score, f1_score
    if metric == "qwk":
        return _qwk(y, p, C, w)
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y, p, sample_weight=w))
    if metric == "macro_f1":
        return float(f1_score(y, p, labels=list(range(C)), average="macro", sample_weight=w, zero_division=0))
    raise ValueError(f"unknown decision metric {metric!r}")


def select_decision_rule(Q, y, w, cfg: dict[str, Any] | None) -> dict[str, Any]:
    """uq.decision -> {rule, metric, scores}; rule fixed unless 'auto'."""
    cfg = cfg or {}
    rule = str(cfg.get("rule", "argmax")).lower()
    if rule != "auto":
        decide(Q[:1], rule)               # validate
        return {"rule": rule, "selected_by": "config"}
    metric = cfg.get("metric", "qwk")
    cands = cfg.get("candidates") or list(DECISION_RULES)
    C = Q.shape[1]
    scores = {r: hard_metric(y, decide(Q, r), w, C, metric) for r in cands}
    best = max(scores, key=lambda r: (scores[r], r == "argmax"))
    return {"rule": best, "selected_by": f"calibration_{metric}", "scores": scores}


# ---------------------------------------------------------------------------
# Selection losses
# ---------------------------------------------------------------------------

def exceedance_brier_ratio(Q, log10_obs, w, spec, pi_clim) -> tuple[float, dict[str, float]]:
    """Mean BS / BS_clim over alert thresholds (climatological exceedance from pi_clim)."""
    wn = np.asarray(w, float) / max(np.sum(w), EPS)
    obs = 10 ** np.asarray(log10_obs, float)
    out = {}
    for t in spec.alert_thresholds:
        pe = exceedance_probability(Q, spec.edges, t, spec.detection_floor)
        pc = float(exceedance_probability(np.asarray(pi_clim, float)[None], spec.edges, t,
                                          spec.detection_floor)[0])
        ye = (obs >= t).astype(float)
        bs, bsc = float((wn * (pe - ye) ** 2).sum()), float((wn * (pc - ye) ** 2).sum())
        out[f"{t:g}"] = bs / max(bsc, EPS)
    return (float(np.mean(list(out.values()))) if out else np.nan), out


def selection_components(Q, y, w, log10_obs, spec, pi_clim, decision_rule="argmax") -> dict[str, Any]:
    wn = np.asarray(w, float) / max(np.sum(w), EPS)
    rps = float((wn * ranked_probability_score(Q, y)).sum())
    rps_c = float((wn * ranked_probability_score(np.broadcast_to(pi_clim, Q.shape), y)).sum())
    eb, eb_t = (exceedance_brier_ratio(Q, log10_obs, w, spec, pi_clim)
                if log10_obs is not None and spec.alert_thresholds else (np.nan, {}))
    return {"rps": rps, "rpss": rps / max(rps_c, EPS),          # = 1 - RPSS
            "exceedance_brier": eb, "exceedance_brier_by_threshold": eb_t,
            "qwk": 1.0 - _qwk(y, decide(Q, decision_rule), Q.shape[1], w)}


def selection_loss(comp: dict[str, Any], cfg: dict[str, Any] | None) -> float:
    cfg = cfg or {}
    metric = str(cfg.get("metric", "rps")).lower()
    if metric in COMPONENTS:
        v = comp[metric]
        return float(v) if np.isfinite(v) else np.inf
    if metric != "composite":
        raise ValueError(f"unknown classifier.selection.metric {metric!r}")
    wts = cfg.get("weights") or DEFAULT_COMPOSITE
    tot, s = 0.0, 0.0
    for k, a in wts.items():
        if k not in COMPONENTS:
            raise ValueError(f"composite weight for unknown component {k!r}")
        v = comp[k]
        if np.isfinite(v):
            s += float(a) * float(v); tot += float(a)
    return s / tot if tot > 0 else np.inf


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------

def censored_mask(log10_obs, spec, tol: float = 1e-9) -> np.ndarray:
    return np.asarray(log10_obs, float) <= np.log10(spec.detection_floor) + tol


def interval_bounds(P, spec, tail: float = 0.05, pad_dex: float = 0.0):
    """[lo, hi] of log10 concentration for tail mass `tail` per side, widened by pad_dex."""
    le = _log_edges(spec.edges, spec.detection_floor)
    lo = log10_concentration_quantile(P, spec.edges, tail, spec.detection_floor)
    hi = log10_concentration_quantile(P, spec.edges, 1.0 - tail, spec.detection_floor)
    if pad_dex:
        lo, hi = np.maximum(lo - pad_dex, le[0]), np.minimum(hi + pad_dex, le[-1])
    return lo, hi


def interval_covered(P, log10_obs, spec, tail: float = 0.05, pad_dex: float = 0.0) -> np.ndarray:
    """Censored-aware coverage: non-detects count when the lower bound reaches bin 0."""
    lo, hi = interval_bounds(P, spec, tail, pad_dex)
    lv = np.asarray(log10_obs, float)
    cens = censored_mask(lv, spec)
    le1 = _log_edges(spec.edges, spec.detection_floor)[1]
    return np.where(cens, lo < le1, (lv >= lo) & (lv <= hi))


def interval_report(P, log10_obs, w, spec, tail=0.05, pad_dex=0.0, prefix="interval_90") -> dict[str, float]:
    lv = np.asarray(log10_obs, float)
    ok = np.isfinite(lv)
    P, lv, w = np.asarray(P)[ok], lv[ok], np.asarray(w, float)[ok]
    if not len(lv):
        return {}
    lo, hi = interval_bounds(P, spec, tail, pad_dex)
    cov = interval_covered(P, lv, spec, tail, pad_dex)
    cens = censored_mask(lv, spec)
    k_lo, k_hi = class_quantile(P, tail), class_quantile(P, 1.0 - tail)
    from sit_fuse.pipelines.habs.uq.hab_uq_common import concentration_to_class
    yb = concentration_to_class(10 ** lv, spec.edges)
    yb = np.where(cens, 0, yb)
    out = {f"{prefix}_coverage": float(np.average(cov, weights=w)),
           f"{prefix}_coverage_uncensored_only": float(np.average(cov[~cens], weights=w[~cens])) if (~cens).any() else np.nan,
           f"{prefix}_coverage_censored_only": float(np.average(cov[cens], weights=w[cens])) if cens.any() else np.nan,
           f"{prefix}_bin_coverage": float(np.average((yb >= k_lo) & (yb <= k_hi), weights=w)),
           f"{prefix}_mean_width_dex": float(np.average(hi - lo, weights=w)),
           f"{prefix}_censored_share": float(np.average(cens, weights=w)),
           f"{prefix}_tail": float(tail), f"{prefix}_pad_dex": float(pad_dex)}
    # previous (uncensored) check, kept for comparison with earlier reports
    out[f"{prefix}_coverage_legacy"] = float(np.average((lv >= lo) & (lv <= hi), weights=w))
    return out


def fit_interval_calibration(P, log10_obs, w, spec, cfg: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    uq.intervals: {recalibrate: true, target: 0.9, tails: [...], max_pad_dex: 1.0, pad_step_dex: 0.05}
    Largest tail mass (narrowest interval) whose censored-aware calibration coverage >= target;
    if none reaches it, the smallest tail plus the smallest pad_dex that does.
    """
    cfg = cfg or {}
    if not cfg.get("recalibrate", False) or P is None or len(P) == 0:
        return None
    target = float(cfg.get("target", 0.9))
    ok = np.isfinite(np.asarray(log10_obs, float))
    P, log10_obs, w = np.asarray(P)[ok], np.asarray(log10_obs, float)[ok], np.asarray(w, float)[ok]
    if not len(P):
        return None
    tails = sorted({float(t) for t in (cfg.get("tails") or DEFAULT_TAILS)}, reverse=True)
    grid = {}
    for a in tails:
        grid[a] = float(np.average(interval_covered(P, log10_obs, spec, a), weights=w))
        if grid[a] >= target:
            return {"target": target, "tail": a, "pad_dex": 0.0, "calibration_coverage": grid[a],
                    "nominal_tail": 0.05, "n": int(len(P)), "grid": grid, "reached_target": True}
    a = tails[-1]
    step, mx = float(cfg.get("pad_step_dex", 0.05)), float(cfg.get("max_pad_dex", 1.0))
    pad, cov = 0.0, grid[a]
    while cov < target and pad < mx - 1e-12:
        pad = round(pad + step, 6)
        cov = float(np.average(interval_covered(P, log10_obs, spec, a, pad), weights=w))
    return {"target": target, "tail": a, "pad_dex": pad, "calibration_coverage": cov,
            "nominal_tail": 0.05, "n": int(len(P)), "grid": grid, "reached_target": bool(cov >= target)}
