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
  intervals     coverage of the q05-q95 log10 interval, mean width (dex)
  selective     risk-coverage (AURC) on predictive entropy, metrics at the
                calibrated selective threshold
  baseline      the same metrics for the production-analogue hard assignment
                (histogram-mode context assignment, train-only) so UQ products
                are compared with what the HAB pipeline ships today
CIs: grouped bootstrap over split groups (station x time block).
Stratification: sensor, station, season, and any extra column.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sit_fuse.train.sf_uq_utils import (
    EPS, full_proba, predictive_entropy_norm, reliability_table, risk_coverage)
from sit_fuse.pipelines.habs.uq.hab_uq_common import (
    exceedance_probability, log10_concentration_quantile, ranked_probability_score, species_spec)
from sit_fuse.pipelines.habs.uq.hab_uq_matchups import sample_level
from sit_fuse.pipelines.habs.uq.hab_uq_train import add_sensor_columns


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


def prob_scores(P, y, w, spec, pi_clim, log10_obs=None) -> dict[str, Any]:
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
            d = {"base_rate": float(np.average(ye, weights=w)),
                 "brier": float((wn * (pe - ye) ** 2).sum()), "ece": ece, "reliability": tab}
            if 0 < ye.sum() < len(ye):
                from sklearn.metrics import roc_auc_score
                d["roc_auc"] = float(roc_auc_score(ye, pe, sample_weight=w))
            out["exceedance"][f"{t:g}"] = d
        lo = log10_concentration_quantile(P, spec.edges, 0.05, spec.detection_floor)
        hi = log10_concentration_quantile(P, spec.edges, 0.95, spec.detection_floor)
        lv = np.asarray(log10_obs)
        out["interval_90_coverage"] = float(np.average((lv >= lo) & (lv <= hi), weights=w))
        out["interval_90_mean_width_dex"] = float(np.average(hi - lo, weights=w))
    return out


def score_block(P, meta, spec, pi_clim, baseline=None, selective_thr=None) -> dict[str, Any]:
    C = P.shape[1]
    y = meta["ref_label"].to_numpy().astype(int)
    w = meta["weight"].to_numpy(dtype=float)
    pred = P.argmax(1)
    res = {"n_samples": int(len(y)), "uq": {**hard_scores(pred, y, w, C),
                                             **prob_scores(P, y, w, spec, pi_clim, meta["log10_value"].to_numpy())}}
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


def grouped_bootstrap(P, meta, spec, pi_clim, n_boot=500, seed=0, keys=("rps", "weighted_f1", "qwk", "nll")):
    rng = np.random.default_rng(seed)
    groups = meta["split_group"].to_numpy()
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        draw = rng.choice(ug, size=ug.size, replace=True)
        idx = np.concatenate([idx_by_g[g] for g in draw])
        m = meta.iloc[idx]
        r = {**hard_scores(P[idx].argmax(1), m["ref_label"].to_numpy().astype(int),
                           m["weight"].to_numpy(float), P.shape[1]),
             **prob_scores(P[idx], m["ref_label"].to_numpy().astype(int), m["weight"].to_numpy(float),
                           spec, pi_clim)}
        for k in keys:
            vals[k].append(r.get(k, np.nan))
    return {k: {"lo95": float(np.nanpercentile(v, 2.5)), "hi95": float(np.nanpercentile(v, 97.5))}
            for k, v in vals.items()}


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
        outP.append(Q); outM.append(meta); outB.append(Hs.argmax(1))
    if not outP:
        return None, None, None
    return np.vstack(outP), pd.concat(outM, ignore_index=True), np.concatenate(outB)


def evaluate_bundle(bundle, frame: pd.DataFrame, out_dir, n_boot: int = 500,
                    strata=("sensor", "Location_Code", "season")) -> dict[str, Any]:
    spec = species_spec(bundle["species"], bundle["spec"])
    P, meta, base = predict_samples(bundle, frame, "test")
    if P is None:
        return {"error": "no test matchups"}
    meta["season"] = pd.to_datetime(meta["date"]).dt.month.map(
        lambda m: {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                   6: "JJA", 7: "JJA", 8: "JJA"}.get(m, "SON"))
    pi = bundle["pi_true"]
    report = {"species": bundle["species"], "overall": score_block(P, meta, spec, pi, base),
              "overall_ci": grouped_bootstrap(P, meta, spec, pi, n_boot=n_boot)}
    for col in strata:
        report[f"by_{col}"] = {}
        for k, idx in meta.groupby(col).indices.items():
            if len(idx) < 5:
                continue
            thr = bundle["selective"].get(k, {}).get("threshold") if col == "sensor" else None
            report[f"by_{col}"][str(k)] = score_block(P[idx], meta.iloc[idx], spec, pi, base[idx], thr)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([meta, pd.DataFrame(P, columns=[f"p_{c}" for c in range(P.shape[1])]),
               pd.Series(base, name="production_bin")], axis=1).to_csv(
        out / f"hab_uq_test_predictions_{spec.fname_token}.csv", index=False)
    with open(out / f"hab_uq_evaluation_{spec.fname_token}.json", "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return report
