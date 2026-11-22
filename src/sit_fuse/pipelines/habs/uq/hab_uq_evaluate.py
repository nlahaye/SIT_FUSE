"""
Point-based evaluation of probabilistic HAB products against held-out in-situ
samples (split == test; never used for encoder, classifier, calibration or
fusion weights).

Unit of evaluation: one (sample, sensor) pair - pixel probabilities within the
matchup radius are linear-pooled, mirroring the production radius histogram.

Metrics
  hard          accuracy, balanced accuracy, macro / weighted F1 (paper metric),
                quadratic-weighted kappa, ordinal MAE (bins), within-one-bin
  probabilistic NLL, multiclass Brier, ranked probability score (RPS) and skill
                vs climatology (RPSS), class-wise ECE (sf_uq_utils.reliability_table)
  exceedance    per alert threshold: Brier, ECE, ROC AUC, reliability table
  intervals     coverage of the q05-q95 log10 interval, mean width (dex). Left-censored
                observations (at or below the detection floor) count as covered when the
                lower bound reaches bin 0 (hab_uq_selection.interval_covered); the previous
                uncensored check is kept as interval_90_coverage_legacy. With
                uq.intervals.recalibrate, interval_90_recal_* scores the calibrated tails.
  decision      hard metrics use the bundle's decision rule (uq.decision); uq_argmax keeps
                the argmax scores when the rule differs
  selective     risk-coverage (AURC) on predictive entropy, metrics at the
                calibrated selective threshold
  baseline      the same metrics for the production-analogue hard assignment
                (histogram-mode context assignment, train-only) so UQ products
                are compared with what the HAB pipeline ships today: joint ->
                own parent -> spatial tier 2 when context_encoder.spatial_tier is
                on (production_baseline), plus the tier-1-only analogue
                (production_baseline_tier1)
  coverage      share of test pixels / samples whose hard assignment comes from
                the joint cluster, its parent, the spatial tier or none
                (assignment_coverage), and all scores per dominant tier (by_assign_tier)
CIs: grouped bootstrap over split groups (station x time block).
Stratification: sensor, station, season, year, and any extra column.
exceedance.<t>.mean_p next to base_rate: calibration-in-the-large per stratum (year effect).
presence      uq.species.<key>.presence_only (e.g. IFCB for Alexandrium): split 'presence_test' rows
              score P(>= threshold) against observed presence: Brier, skill vs training
              climatology, ROC AUC (grouped-bootstrap CI), mean_p vs base rate; overall and by
              sensor, station and year (hab_uq_evaluation_<token>.json "presence").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sit_fuse.train.sf_uq_utils import (
    EPS, full_proba, predictive_entropy_norm, reliability_table, risk_coverage)
from sit_fuse.pipelines.habs.uq.hab_uq_selection import decide, decision_spec, interval_bounds, interval_report
from sit_fuse.pipelines.habs.uq.hab_uq_common import (
    exceedance_probability, log10_concentration_quantile, ranked_probability_score, species_spec)
from sit_fuse.pipelines.habs.uq.hab_uq_matchups import sample_level
from sit_fuse.pipelines.habs.uq.hab_uq_train import add_sensor_columns
from sit_fuse.pipelines.habs.uq.hab_context_encoder import TIER_NAMES


def assignment_coverage(frame: pd.DataFrame, meta: pd.DataFrame, split: str = "test") -> dict[str, Any]:
    """Pixel and sample shares of the hard-assignment tier, overall and per sensor."""
    if "assign_tier" not in frame.columns:
        return {}
    rows = frame[frame["split"] == split]

    def block(r, m):
        t = np.clip(r["assign_tier"].to_numpy().astype(int), 0, len(TIER_NAMES) - 1)
        n = np.bincount(t, minlength=len(TIER_NAMES)).astype(float)
        w = np.bincount(t, weights=r["pix_weight"].to_numpy(float), minlength=len(TIER_NAMES))
        out = {"n_pixels": int(n.sum()),
               "pixel_share": {k: float(v) for k, v in zip(TIER_NAMES, n / max(n.sum(), 1))},
               "weighted_pixel_share": {k: float(v) for k, v in zip(TIER_NAMES, w / max(w.sum(), EPS))}}
        if "assign_tier" in m.columns:
            vc = m["assign_tier"].value_counts(normalize=True)
            out["sample_share_dominant"] = {k: float(vc.get(k, 0.0)) for k in TIER_NAMES}
        return out
    res = {"overall": block(rows, meta), "by_sensor": {}}
    for s, r in rows.groupby("sensor"):
        res["by_sensor"][str(s)] = block(r, meta[meta["sensor"] == s])
    return res


def _qwk(y, p, C, w):
    from sklearn.metrics import cohen_kappa_score
    if len(np.unique(np.r_[y, p])) < 2:
        return np.nan
    return float(cohen_kappa_score(y, p, labels=list(range(C)), weights="quadratic", sample_weight=w))


def hard_scores(pred, y, w, C) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, f1_score
    w = np.asarray(w, float)
    return {
        "accuracy": float(np.average(pred == y, weights=w)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred, sample_weight=w)),
        "macro_f1": float(f1_score(y, pred, labels=list(range(C)), average="macro", sample_weight=w, zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=list(range(C)), average="weighted", sample_weight=w, zero_division=0)),
        "qwk": _qwk(y, pred, C, w),
        "ordinal_mae_bins": float(np.average(np.abs(pred - y), weights=w)),
        "within_one_bin": float(np.average(np.abs(pred - y) <= 1, weights=w)),
    }


def prob_scores(P, y, w, spec, pi_clim, log10_obs=None, interval_cal=None) -> dict[str, Any]:
    C = P.shape[1]
    wn = np.asarray(w, float) / max(np.sum(w), EPS)
    Y = np.eye(C)[y]
    rps = ranked_probability_score(P, y)
    rps_c = ranked_probability_score(np.broadcast_to(pi_clim, P.shape), y)
    out = {
        "nll": float(-(wn * np.log(np.clip(P[np.arange(len(y)), y], EPS, 1))).sum()),
        "brier": float((wn * ((P - Y) ** 2).sum(1)).sum()),
        "rps": float((wn * rps).sum()),
        "rpss_vs_climatology": float(1 - (wn * rps).sum() / max((wn * rps_c).sum(), EPS)),
    }
    eces = []
    for c in range(C):
        if Y[:, c].sum() == 0 and P[:, c].sum() < 1e-6:
            continue
        _, e = reliability_table(P[:, c], Y[:, c], np.asarray(w, float), n_bins=10)
        eces.append(e)
    out["classwise_ece"] = float(np.mean(eces)) if eces else np.nan
    out["exceedance"] = {}
    if log10_obs is not None:
        obs = 10 ** np.asarray(log10_obs)
        for t in spec.alert_thresholds:
            pe = exceedance_probability(P, spec.edges, t, spec.detection_floor)
            ye = (obs >= t).astype(float)
            tab, ece = reliability_table(pe, ye, np.asarray(w, float), n_bins=10)
            d = {"base_rate": float(np.average(ye, weights=w)), "mean_p": float(np.average(pe, weights=w)),
                 "brier": float((wn * (pe - ye) ** 2).sum()), "ece": ece, "reliability": tab}
            if 0 < ye.sum() < len(ye):
                from sklearn.metrics import roc_auc_score
                d["roc_auc"] = float(roc_auc_score(ye, pe, sample_weight=w))
            out["exceedance"][f"{t:g}"] = d
        # nominal q05-q95, censored-aware (non-detects at the detection floor count as covered
        # when the lower bound reaches bin 0); _legacy keeps the previous uncensored check
        out.update(interval_report(P, log10_obs, w, spec, 0.05, 0.0, "interval_90"))
        if interval_cal:
            out.update(interval_report(P, log10_obs, w, spec, interval_cal["tail"],
                                       interval_cal.get("pad_dex", 0.0), "interval_90_recal"))
    return out


def eval_options(bundle) -> dict[str, Any]:
    """Decision rule and interval calibration stored in a bundle (argmax / none for older bundles)."""
    return {"decision": decision_spec(bundle),
            "interval_cal": bundle.get("interval_calibration")}


def score_block(P, meta, spec, pi_clim, baseline=None, selective_thr=None, decision="argmax",
                interval_cal=None) -> dict[str, Any]:
    C = P.shape[1]
    y = meta["ref_label"].to_numpy().astype(int)
    w = meta["weight"].to_numpy(dtype=float)
    pred = decide(P, decision)
    res = {"n_samples": int(len(y)),
           "decision_rule": decision.get("rule") if isinstance(decision, dict) else decision,
           **({"decision_thresholds": decision.get("thresholds")} if isinstance(decision, dict) else {}),
           "uq": {**hard_scores(pred, y, w, C),
                  **prob_scores(P, y, w, spec, pi_clim, meta["log10_value"].to_numpy(), interval_cal)}}
    if decision != "argmax":
        res["uq_argmax"] = hard_scores(P.argmax(1), y, w, C)      # previous decision, for comparison
    u = predictive_entropy_norm(P)
    res["uq"]["risk_coverage_entropy"] = risk_coverage(u, (pred != y), w)
    if selective_thr is not None and np.isfinite(selective_thr):
        keep = u <= selective_thr
        if keep.sum() > 1:
            res["uq_selective"] = {"coverage": float(np.average(keep, weights=w)),
                                   **hard_scores(pred[keep], y[keep], w[keep], C)}
    if baseline is not None:
        B = np.eye(C)[baseline]
        res["production_baseline"] = {**hard_scores(baseline, y, w, C),
                                      "rps": float(np.average(ranked_probability_score(B, y), weights=w))}
    return res


def _baseline_tier1(meta: pd.DataFrame, C: int) -> dict[str, float]:
    b = meta["production_bin_tier1"].to_numpy().astype(int)
    y = meta["ref_label"].to_numpy().astype(int)
    w = meta["weight"].to_numpy(dtype=float)
    return {**hard_scores(b, y, w, C),
            "rps": float(np.average(ranked_probability_score(np.eye(C)[b], y), weights=w))}


def grouped_bootstrap(P, meta, spec, pi_clim, n_boot=500, seed=0, keys=("rps", "weighted_f1", "qwk", "nll"),
                      decision="argmax"):
    rng = np.random.default_rng(seed)
    groups = meta["split_group"].to_numpy()
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        draw = rng.choice(ug, size=ug.size, replace=True)
        idx = np.concatenate([idx_by_g[g] for g in draw])
        m = meta.iloc[idx]
        r = {**hard_scores(decide(P[idx], decision), m["ref_label"].to_numpy().astype(int),
                           m["weight"].to_numpy(float), P.shape[1]),
             **prob_scores(P[idx], m["ref_label"].to_numpy().astype(int), m["weight"].to_numpy(float),
                           spec, pi_clim)}
        for k in keys:
            vals[k].append(r.get(k, np.nan))
    return {k: {"lo95": float(np.nanpercentile(v, 2.5)), "hi95": float(np.nanpercentile(v, 97.5))}
            for k, v in vals.items()}


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(p).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def presence_config(bundle) -> dict[str, Any] | None:
    sp = ((bundle.get("uq_cfg") or {}).get("species") or {}).get(bundle["species"]) or {}
    return sp.get("presence_only") or None


def presence_block(p: np.ndarray, obs: np.ndarray, p_clim: float) -> dict[str, Any]:
    obs = np.asarray(obs, float)
    bs = float(np.mean((p - obs) ** 2))
    bs_c = float(np.mean((p_clim - obs) ** 2))
    return {"n": int(obs.size), "n_present": int(obs.sum()), "base_rate": float(obs.mean()),
            "mean_p": float(np.mean(p)), "brier": bs, "brier_climatology": bs_c,
            "bss": float(1.0 - bs / bs_c) if bs_c > 0 else float("nan"), "auc": _auc(obs > 0.5, p)}


def presence_scores(P, meta, spec, pi_clim, cfg, n_boot: int = 200, seed: int = 0,
                    strata=("sensor", "Location_Code", "year")) -> dict[str, Any]:
    """P(>= cfg.threshold) vs observed presence on presence-only rows (e.g. IFCB)."""
    thr = float(cfg.get("threshold", 1.0))
    k = int(np.searchsorted(np.asarray(spec.edges[1:-1], float), thr, side="left")) + 1
    k = min(max(k, 1), P.shape[1] - 1)
    p = P[:, k:].sum(1)
    obs = (meta["value"].to_numpy(float) >= thr).astype(float)
    p_clim = float(np.asarray(pi_clim)[k:].sum())
    out = {"threshold": thr, "class_index": k, "methods": cfg.get("methods", ["ifcb"]),
           "climatology_p": p_clim, "overall": presence_block(p, obs, p_clim)}
    groups = meta["split_group"].to_numpy()
    ug = np.unique(groups)
    if ug.size > 1 and n_boot:
        rng = np.random.default_rng(seed)
        idx_by_g = {g: np.where(groups == g)[0] for g in ug}
        b = {"auc": [], "bss": []}
        for _ in range(n_boot):
            idx = np.concatenate([idx_by_g[g] for g in rng.choice(ug, size=ug.size, replace=True)])
            r = presence_block(p[idx], obs[idx], p_clim)
            b["auc"].append(r["auc"]); b["bss"].append(r["bss"])
        out["overall_ci"] = {key: {"lo95": float(np.nanpercentile(v, 2.5)), "hi95": float(np.nanpercentile(v, 97.5))}
                             for key, v in b.items()}
    m = meta.reset_index(drop=True).copy()
    if "year" not in m and "date" in m:
        m["year"] = pd.to_datetime(m["date"]).dt.year.astype(str)
    for col in strata:
        if col not in m:
            continue
        out[f"by_{col}"] = {str(g): presence_block(p[i], obs[i], p_clim)
                            for g, i in m.groupby(col).indices.items() if len(i) >= 5}
    out["_p"], out["_obs"] = p, obs
    return out


def predict_samples(bundle, frame: pd.DataFrame, split: str = "test"):
    """Calibrated sample-level probabilities + production baseline for one split."""
    C = bundle["n_classes"]
    cols = bundle["feature_columns"]
    outP, outM, outB = [], [], []
    for s in bundle["sensors"]:
        sub = frame[(frame["split"] == split) & (frame["sensor"] == s)]
        if sub.empty:
            continue
        sub = add_sensor_columns(sub, bundle["sensors"])
        for c in cols:
            if c not in sub.columns:
                sub[c] = 0.0
        key = "pooled" if bundle["model_scope"] == "pooled" else s
        P = full_proba(bundle["fits"][key]["model"], sub[cols], C)
        Ps, meta = sample_level(P, sub)
        cal = bundle["calibrators"].get(s) or bundle["calibrators"].get("_pooled")
        Q = cal.transform(Ps) if cal is not None else Ps
        # production analogue: majority of hard context-assigned bins in radius
        H = np.column_stack([sub[f"assign_hard_{c}"].to_numpy() for c in range(C)])
        Hs, _ = sample_level(H, sub)
        if f"assign_hard_t1_{C - 1}" in sub.columns:
            H1 = np.column_stack([sub[f"assign_hard_t1_{c}"].to_numpy() for c in range(C)])
            meta["production_bin_tier1"] = sample_level(H1, sub)[0].argmax(1)
        if "assign_tier" in sub.columns:
            t = np.clip(sub["assign_tier"].to_numpy().astype(int), 0, len(TIER_NAMES) - 1)
            Ts, _ = sample_level(np.eye(len(TIER_NAMES))[t], sub)
            for i, nm in enumerate(TIER_NAMES):
                meta[f"tier_share_{nm}"] = Ts[:, i]
            meta["assign_tier"] = np.asarray(TIER_NAMES, dtype=object)[Ts.argmax(1)]
        outP.append(Q); outM.append(meta); outB.append(Hs.argmax(1))
    if not outP:
        return None, None, None
    return np.vstack(outP), pd.concat(outM, ignore_index=True), np.concatenate(outB)


def prediction_table(P, meta, base, spec, eo) -> pd.DataFrame:
    """Test predictions: probabilities, decision, nominal and recalibrated interval bounds."""
    cols = {f"p_{c}": P[:, c] for c in range(P.shape[1])}
    cols["production_bin"] = base
    cols["pred_label"] = decide(P, eo["decision"])
    cols["log10_lo90"], cols["log10_hi90"] = interval_bounds(P, spec, 0.05)
    ic = eo.get("interval_cal")
    if ic:
        cols["log10_lo90_recal"], cols["log10_hi90_recal"] = interval_bounds(P, spec, ic["tail"], ic.get("pad_dex", 0.0))
    return pd.concat([meta.reset_index(drop=True), pd.DataFrame(cols)], axis=1)


def evaluate_bundle(bundle, frame: pd.DataFrame, out_dir, n_boot: int = 500,
                    strata=("sensor", "Location_Code", "season", "year")) -> dict[str, Any]:
    spec = species_spec(bundle["species"], bundle["spec"])
    P, meta, base = predict_samples(bundle, frame, "test")
    if P is None:
        return {"error": "no test matchups"}
    meta["season"] = pd.to_datetime(meta["date"]).dt.month.map(
        lambda m: {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                   6: "JJA", 7: "JJA", 8: "JJA"}.get(m, "SON"))
    meta["year"] = pd.to_datetime(meta["date"]).dt.year.astype(str)   # year effect: mean_p vs base_rate
    pi = bundle["pi_true"]
    eo = eval_options(bundle)
    report = {"species": bundle["species"], "overall": score_block(P, meta, spec, pi, base, **eo),
              "overall_ci": grouped_bootstrap(P, meta, spec, pi, n_boot=n_boot, decision=eo["decision"])}
    has_t1 = "production_bin_tier1" in meta.columns
    if has_t1:
        report["overall"]["production_baseline_tier1"] = _baseline_tier1(meta, P.shape[1])
    cov = assignment_coverage(frame, meta)
    if cov:
        report["assignment_coverage"] = cov
    uq = bundle.get("uq_cfg") or {}
    clf = uq.get("classifier") or {}
    report["run_config"] = {
        "model_name": bundle.get("model_name"),
        "model_selection_calibration_rps": bundle.get("model_selection_calibration_rps"),
        "model_trial": bundle.get("model_trial"),
        "model_selection": {k: v for k, v in (bundle.get("model_selection") or {}).items() if k != "trials"},
        "selected_class_weighting": bundle.get("class_weighting"),
        "selected_prior_correction_strength": bundle.get("prior_correction_strength"),
        "decision": bundle.get("decision"),
        "exceedance_recalibration": bundle.get("exceedance_recalibration"),
        "calibration_components_final": bundle.get("calibration_components_final"),
        # calibration-split selection components of every trial (e.g. prior strength 1 vs selected)
        "trial_components": {k: {c: v["components"].get(c) for c in ("rps", "rpss", "exceedance_brier",
                                                                       "exceedance_brier_by_threshold")}
                             for k, v in ((bundle.get("model_selection") or {}).get("trials") or {}).items()},
        "interval_calibration": {k: v for k, v in (bundle.get("interval_calibration") or {}).items() if k != "grid"}
        or None,
        "holdout_locations": ((uq.get("splits") or {}).get("holdout_locations")),
        "input_mode": sorted(frame["input_mode"].dropna().unique().tolist())
        if "input_mode" in frame.columns else None,
        "blend_lambda": (uq.get("target") or {}).get("blend_lambda"),
        "class_weighting": clf.get("class_weighting"),
        "prior_correction": (uq.get("calibration") or {}).get("prior_correction", True),
        "prior_correction_strength": (uq.get("calibration") or {}).get("prior_correction_strength", 1.0),
        "radius_km": uq.get("radius_km"), "nearest_fallback": uq.get("nearest_fallback"),
        "sample_weighting": bundle.get("sample_weighting"),
        "n_test_fallback": int(meta["match_fallback"].sum()) if "match_fallback" in meta else 0,
        "pool_sensor_group": (uq.get("context_encoder") or {}).get("pool_sensor_group", False),
        "spatial_tier": (uq.get("context_encoder") or {}).get("spatial_tier"),
    }
    if "match_fallback" in meta.columns and meta["match_fallback"].nunique() > 1:
        strata = tuple(strata) + ("match_fallback",)
    if "assign_tier" in meta.columns:
        strata = tuple(strata) + ("assign_tier",)
    for col in strata:
        report[f"by_{col}"] = {}
        for k, idx in meta.groupby(col).indices.items():
            if len(idx) < 5:
                continue
            thr = bundle["selective"].get(k, {}).get("threshold") if col == "sensor" else None
            report[f"by_{col}"][str(k)] = score_block(P[idx], meta.iloc[idx], spec, pi, base[idx], thr, **eo)
            if has_t1:
                report[f"by_{col}"][str(k)]["production_baseline_tier1"] = _baseline_tier1(
                    meta.iloc[idx], P.shape[1])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pcfg = presence_config(bundle)
    if pcfg:
        Pp, mp, _ = predict_samples(bundle, frame, "presence_test")
        if Pp is None:
            report["presence"] = {"error": "no presence_test matchups"}
        else:
            pr = presence_scores(Pp, mp, spec, pi, pcfg, n_boot=min(n_boot, 200))
            keep = [c for c in ("sample_uid", "sensor", "date", "Location_Code", "value", "split_group") if c in mp]
            mp[keep].assign(p_present=pr.pop("_p"), observed_present=pr.pop("_obs")).to_csv(
                out / f"hab_uq_presence_predictions_{spec.fname_token}.csv", index=False)
            report["presence"] = pr
    prediction_table(P, meta, base, spec, eo).to_csv(
        out / f"hab_uq_test_predictions_{spec.fname_token}.csv", index=False)
    with open(out / f"hab_uq_evaluation_{spec.fname_token}.json", "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return report
