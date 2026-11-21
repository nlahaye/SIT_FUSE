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

Decision rule (uq.decision; per species: uq.species.<key>.decision)
  rule: argmax | median | expected | threshold | auto     hard class from calibrated probabilities
  auto picks among `candidates` on the calibration samples by `metric` (qwk | balanced_accuracy |
  macro_f1). median = ordinal median bin (first bin with CDF >= 0.5); expected = rounded mean bin.
  argmax of a diffuse ordinal distribution collapses to the majority bin; median / expected
  follow the predictive distribution's location instead.
  threshold = highest bin k with P(bin >= k) >= tau_k, tau tuned on the calibration samples by
  `metric` (coordinate search over quantiles of P(bin >= k)). For rare targets (pDA) argmax, median
  and expected all return bin 0 whenever P(bin >= 1) < 0.5, i.e. everywhere; tau is the only
  rule that can assign upper bins. Options: max_flag_ratio (2.0; predicted share of bin >= k at
  most this multiple of the calibration share), min_events (5; bins with fewer calibration
  samples at or above them are never assigned), n_grid (40).
  Stored as {"rule": "threshold", "thresholds": [tau_1, ..., tau_{C-1}]} (null = never assigned);
  decide() accepts the rule string or this dict.

Exceedance recalibration (uq.calibration.exceedance_recalibration; per species:
uq.species.<key>.exceedance_recalibration)
  {enabled: false | true | auto, method: platt | shift, thresholds: alert, min_events: 10,
   n_folds: 5, ridge: 1.0}
  After model selection, P(bin >= k) at every alert threshold that lies on a bin edge is mapped
  through a logistic recalibration fitted on the pooled calibration samples (platt: slope +
  intercept on logit P; shift: intercept only, used automatically below min_events). Class
  probabilities are rescaled within the groups between those edges so the class distribution
  stays valid and its shape inside each group is unchanged. auto keeps the maps only when the
  day-grouped out-of-fold Brier at the thresholds improves on the uncalibrated Brier (the in-sample
  fit always improves); true always keeps them. Applied by wrapping every calibrator
  (ExceedanceRecalibrator), so inference, evaluation and fusion use it unchanged.

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
DECISION_RULES = ("argmax", "median", "expected", "threshold")
DEFAULT_TAILS = (0.25, 0.20, 0.15, 0.125, 0.10, 0.075, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01, 0.0075, 0.005,
                 0.0025, 0.001, 0.0)


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------

def decide(P: np.ndarray, rule: Any = "argmax") -> np.ndarray:
    """rule: name, or a decision dict ({"rule": ..., "thresholds": [...]} for threshold)."""
    P = np.asarray(P, float)
    if isinstance(rule, dict):
        if str(rule.get("rule")) == "threshold":
            return threshold_decide(P, rule.get("thresholds"))
        rule = rule.get("rule", "argmax")
    if rule in (None, "argmax"):
        return P.argmax(-1)
    if rule == "median":
        return class_quantile(P, 0.5)
    if rule == "expected":
        c = np.arange(P.shape[-1], dtype=float)
        return np.clip(np.rint((P * c).sum(-1)), 0, P.shape[-1] - 1).astype(int)
    if rule == "threshold":
        raise ValueError("decision rule 'threshold' needs fitted thresholds (decision dict)")
    raise ValueError(f"unknown decision rule {rule!r} (argmax | median | expected | threshold | auto)")


def survival(P: np.ndarray) -> np.ndarray:
    """S[..., k-1] = P(bin >= k), k = 1 .. C-1."""
    P = np.asarray(P, float)
    return np.clip(1.0 - np.cumsum(P, -1)[..., :-1], 0.0, 1.0)


def threshold_decide(P: np.ndarray, thresholds) -> np.ndarray:
    """Highest k with P(bin >= k) >= tau_k (tau None / nan = never); 0 otherwise."""
    S = survival(P)
    pred = np.zeros(S.shape[:-1], dtype=int)
    for k, t in enumerate(thresholds or [], start=1):
        if t is None or not np.isfinite(float(t)) or k > S.shape[-1]:
            continue
        pred = np.where(S[..., k - 1] >= float(t), k, pred)
    return pred


def decision_spec(bundle_or_decision) -> Any:
    """Argument for decide() from a bundle / bundle['decision'] (rule name, or the dict for threshold)."""
    d = bundle_or_decision or {}
    if "decision" in d and isinstance(d.get("decision"), (dict, type(None))):
        d = d.get("decision") or {}
    if not isinstance(d, dict):
        return d or "argmax"
    rule = d.get("rule", "argmax")
    return {"rule": "threshold", "thresholds": d.get("thresholds")} if rule == "threshold" else rule


def fit_threshold_rule(Q, y, w, metric: str = "qwk", max_flag_ratio: float | None = 2.0,
                       min_events: int = 5, n_grid: int = 40, n_pass: int = 3) -> dict[str, Any]:
    """Coordinate search of tau_k on calibration samples (see module docstring)."""
    Q, y = np.asarray(Q, float), np.asarray(y, int)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    C = Q.shape[1]
    S = survival(Q)
    wn = w / max(w.sum(), EPS)
    share = [float(wn[y >= k].sum()) for k in range(1, C)]
    active = [k for k in range(1, C) if int((y >= k).sum()) >= int(min_events)]
    taus: list[float | None] = [None] * (C - 1)
    if not active:
        return {"rule": "threshold", "thresholds": taus, "metric": metric, "score": None,
                "active_bins": [], "note": "no bin with min_events calibration samples"}

    def allowed(k, t):
        if max_flag_ratio is None:
            return True
        return float(wn[S[:, k - 1] >= t].sum()) <= float(max_flag_ratio) * share[k - 1] + 1e-12

    def score(ts):
        return hard_metric(y, threshold_decide(Q, ts), w, C, metric)

    best = score(taus)
    for _ in range(int(n_pass)):
        changed = False
        for k in active:
            cand = np.unique(np.quantile(S[:, k - 1], np.linspace(0.5, 0.999, int(n_grid))))
            for t in [None] + [float(c) for c in cand]:
                if t is not None and (t <= 0 or not allowed(k, t)):
                    continue
                ts = list(taus); ts[k - 1] = t
                sc = score(ts)
                if sc > best + 1e-12:
                    best, taus, changed = sc, ts, True
        if not changed:
            break
    return {"rule": "threshold", "thresholds": taus, "metric": metric, "score": float(best),
            "active_bins": active, "max_flag_ratio": max_flag_ratio, "calibration_share": share}


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
    """uq.decision -> {rule, metric, scores[, thresholds]}; rule fixed unless 'auto'."""
    cfg = cfg or {}
    rule = str(cfg.get("rule", "argmax")).lower()
    metric = cfg.get("metric", "qwk")
    C = Q.shape[1]

    def fit_thr():
        return fit_threshold_rule(Q, y, w, metric, cfg.get("max_flag_ratio", 2.0),
                                  int(cfg.get("min_events", 5)), int(cfg.get("n_grid", 40)))
    if rule != "auto":
        if rule == "threshold":
            return {**fit_thr(), "selected_by": "config"}
        decide(Q[:1], rule)               # validate
        return {"rule": rule, "selected_by": "config"}
    cands = cfg.get("candidates") or list(DECISION_RULES)
    scores, thr = {}, None
    for r in cands:
        if r == "threshold":
            thr = fit_thr()
            scores[r] = hard_metric(y, threshold_decide(Q, thr["thresholds"]), w, C, metric)
        else:
            scores[r] = hard_metric(y, decide(Q, r), w, C, metric)
    best = max(scores, key=lambda r: (scores[r], r == "argmax"))
    out = {"rule": best, "selected_by": f"calibration_{metric}", "scores": scores}
    if best == "threshold":
        out.update({k: v for k, v in thr.items() if k not in ("rule", "score")})
    return out


# ---------------------------------------------------------------------------
# Exceedance recalibration
# ---------------------------------------------------------------------------

def edge_thresholds(spec, thresholds="alert") -> list[tuple[float, int]]:
    """[(threshold, k)] for thresholds on bin edges (P(value >= edge_k) = P(bin >= k))."""
    ts = spec.alert_thresholds if thresholds in (None, "alert") else thresholds
    e = np.asarray(spec.edges, float)
    out = []
    for t in ts or []:
        k = np.where(np.isclose(e, float(t), rtol=1e-9))[0]
        if len(k) and 0 < int(k[0]) < len(e) - 1:
            out.append((float(t), int(k[0])))
    return sorted(set(out), key=lambda x: x[1])


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logistic(x, yb, w, method="platt", ridge=1.0):
    """(a, b) of sigmoid(a * x + b) by weighted log loss; ridge pulls (a, b) toward (1, 0)."""
    from scipy.optimize import minimize
    w = np.asarray(w, float) / max(np.sum(w), EPS)

    def loss(th):
        a, b = (th[0], th[1]) if method == "platt" else (1.0, th[0])
        p = np.clip(_sigmoid(a * x + b), 1e-9, 1 - 1e-9)
        nll = -(w * (yb * np.log(p) + (1 - yb) * np.log(1 - p))).sum()
        return nll + ridge * 1e-3 * ((a - 1.0) ** 2 + b ** 2)
    th0 = [1.0, 0.0] if method == "platt" else [0.0]
    r = minimize(loss, th0, method="L-BFGS-B",
                 bounds=[(0.05, 5.0), (-10, 10)] if method == "platt" else [(-10, 10)])
    return (float(r.x[0]), float(r.x[1])) if method == "platt" else (1.0, float(r.x[0]))


def apply_exceedance_maps(P: np.ndarray, maps: list[dict[str, Any]] | None) -> np.ndarray:
    """Recalibrate P(bin >= k) at each map's edge k; rescale classes within the groups between edges."""
    if not maps:
        return P
    P = np.asarray(P, float)
    C = P.shape[-1]
    ks = [int(m["k"]) for m in maps]
    cs = np.cumsum(P, -1)
    S_new = []
    for m in maps:
        s = 1.0 - cs[..., int(m["k"]) - 1]
        S_new.append(_sigmoid(float(m["a"]) * _logit(s) + float(m["b"])))
    S_new = np.stack(S_new, -1)
    S_new = np.minimum.accumulate(S_new, axis=-1)               # P(>= k) non-increasing in k
    bounds = [0] + ks + [C]
    G_new = np.concatenate([1.0 - S_new[..., :1], S_new[..., :-1] - S_new[..., 1:], S_new[..., -1:]], -1)
    out = np.empty_like(P)
    for j in range(len(bounds) - 1):
        a, b = bounds[j], bounds[j + 1]
        blk = P[..., a:b] + 1e-12
        out[..., a:b] = blk / blk.sum(-1, keepdims=True) * G_new[..., j:j + 1]
    out = np.clip(out, 0.0, 1.0)
    return out / np.maximum(out.sum(-1, keepdims=True), EPS)


def _maps_fit(Q, y, w, edges, method, min_events, ridge):
    maps = []
    for t, k in edges:
        yb = (y >= k).astype(float)
        n_ev = int(yb.sum())
        if n_ev < 3 or n_ev > len(yb) - 3:
            continue
        m = method if n_ev >= min_events else "shift"
        x = _logit(1.0 - np.cumsum(Q, -1)[:, k - 1])
        a, b = _fit_logistic(x, yb, w, m, ridge)
        maps.append({"threshold": t, "k": k, "a": a, "b": b, "method": m, "n_events": n_ev})
    return maps


def _brier_at(Q, y, w, edges):
    wn = np.asarray(w, float) / max(np.sum(w), EPS)
    return {f"{t:g}": float((wn * ((1.0 - np.cumsum(Q, -1)[:, k - 1]) - (y >= k)) ** 2).sum()) for t, k in edges}


def fit_exceedance_recalibration(Q, y, w, spec, cfg: dict[str, Any] | None, groups=None,
                                 seed: int = 0) -> dict[str, Any] | None:
    """See module docstring. Returns {maps, enabled, oof, ...} or None when disabled."""
    cfg = cfg or {}
    mode = cfg.get("enabled", False)
    if mode in (False, None, "false", "off") or Q is None or len(Q) == 0:
        return None
    Q, y = np.asarray(Q, float), np.asarray(y, int)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    edges = edge_thresholds(spec, cfg.get("thresholds", "alert"))
    if not edges:
        return {"enabled": False, "maps": [], "note": "no alert threshold on a bin edge"}
    method, min_ev, ridge = str(cfg.get("method", "platt")), int(cfg.get("min_events", 10)), float(cfg.get("ridge", 1.0))
    maps = _maps_fit(Q, y, w, edges, method, min_ev, ridge)
    used = [(m["threshold"], m["k"]) for m in maps]
    res = {"maps": maps, "method": method, "n": int(len(y)),
           "brier_before": _brier_at(Q, y, w, used), "brier_after_in_sample": _brier_at(apply_exceedance_maps(Q, maps), y, w, used)}
    # day-grouped out-of-fold check
    g = np.asarray(groups if groups is not None else np.arange(len(y)))
    ug = np.unique(g)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(ug, rng.permutation(len(ug)) % max(int(cfg.get("n_folds", 5)), 2)))
    f = np.array([fold_of[x] for x in g])
    Qo = Q.copy()
    for k in np.unique(f):
        tr, te = f != k, f == k
        mk = _maps_fit(Q[tr], y[tr], w[tr], used, method, min_ev, ridge)
        Qo[te] = apply_exceedance_maps(Q[te], mk)
    res["brier_after_oof"] = _brier_at(Qo, y, w, used)
    b0, b1 = np.mean(list(res["brier_before"].values()) or [np.nan]), np.mean(list(res["brier_after_oof"].values()) or [np.nan])
    res["oof_improves"] = bool(np.isfinite(b1) and b1 < b0)
    res["enabled"] = bool(maps) and (res["oof_improves"] if str(mode).lower() == "auto" else True)
    res["mode"] = str(mode)
    if not res["enabled"]:
        res["maps"] = []
    return res


class ExceedanceRecalibrator:
    """Wraps a calibrator (or None): base transform, then apply_exceedance_maps."""

    def __init__(self, base, maps):
        self.base = base
        self.maps = list(maps or [])

    def transform(self, P):
        Q = self.base.transform(P) if self.base is not None else np.asarray(P, float)
        return apply_exceedance_maps(Q, self.maps)

    def transform_members(self, member_P):
        return np.stack([self.transform(p) for p in member_P])

    def describe(self):
        d = {}
        if self.base is not None:
            try:
                d = dict(self.base.describe())
            except Exception:  # pragma: no cover
                d = {"base": type(self.base).__name__}
        d["exceedance_maps"] = self.maps
        return d

    def __getattr__(self, name):
        if name in ("base", "maps") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.base, name)


def wrap_calibrators(calibrators: dict[str, Any], maps) -> dict[str, Any]:
    if not maps:
        return calibrators
    return {k: ExceedanceRecalibrator(v, maps) for k, v in calibrators.items()}


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
