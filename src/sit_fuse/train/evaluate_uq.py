#!/usr/bin/env python3
"""
Evaluation of the discrete, probabilistic and uncertainty products written by
classifier_inference_uq.py, against one or more references.

Scenes: classifier_split in evaluation.splits (default: holdout). Pixels are
evaluated at natural prevalence (optional per-scene subsampling with
compensating weights). Groups (split_group) are the bootstrap unit.

Discrete product (pred_label, pred_label_selective)
  - per-class precision / recall / F1 / IoU, macro target IoU / F1
  - boundary F1 at pixel tolerances (full-resolution scenes)
  - object (plume / fire-cluster) detection: POD, FAR, object F1
  - area: discrete-area bias, relative bias, MAE, correlation
  - selective product: coverage, retained / flagged error vs the target error
    set on the calibration groups
Probabilistic product (p_c, confidence, ensemble quantiles)
  - NLL, Brier with Murphy decomposition (reliability, resolution,
    uncertainty), class-wise ECE, reliability tables / diagrams
  - threshold-free: PR-AUC (average precision), ROC-AUC per target class
  - sharpness: mean confidence, probability variance, mean entropy
  - interval products: binned coverage of ensemble quantile intervals; scene
    area intervals (analytic and ensemble) covering the reference area
Uncertainty products (every uncertainty layer present)
  - error detection: AUROC, mean when correct / wrong, histograms
  - risk-coverage curves, AURC, IoU at fixed coverages
  - uncertainty calibration: observed vs model-implied error (1 - confidence)
    per uncertainty bin, UCE, bin-level Spearman monotonicity
  - scene level: Spearman of scene mean uncertainty with error rate / IoU
  - reference disagreement (aleatoric test): AUROC of each layer for pixels
    where the attached references disagree
  - distribution shift (epistemic test): AUROC and Mann-Whitney U of each
    layer between scene groups (e.g., encoder_train vs encoder_test)
Multi-reference: every metric block against each attached reference and
against the consensus of all references.
Label levels (target.label_levels): every headline metric at the primary
level (all primary-reference labels) and at the consensus level (negatives
only where the consensus references confirm them), with bootstrap CIs for
both, plus the two conflict strata (primary negative / consensus positive and
the reverse) summarized by predicted probability and uncertainty. The
consensus level is assessed only on scenes where the consensus references are
observed (all of them when require_all_observed), and is skipped with the
reason and a coverage report when none are attached, fewer than
evaluation.consensus_level.min_scenes scenes are covered, or fewer than
min_negative_pixels consensus negatives exist.
Bootstrap: percentile CIs over split_group resamples for headline metrics;
paired differences vs evaluation.reference_model on identical resamples.

Scenes without a primary reference (e.g. a new event with no ADP): when no
evaluated scene has a truth raster, a reference-free summary is written
instead (per_scene_reference_free.csv; pooled uncertainty, predicted
fraction, selective retention, shift tests).

Usage:
    python evaluate_uq.py uq_config.yaml [--models random_forest soft_assignment ...]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from sit_fuse.train.classifier_utils import load_json, load_yaml

from sf_uq_utils import (
    ABSTAIN,
    OUT_NODATA,
    LabelLevels,
    combine_consensus,
    reference_target_cfg,
    align_reference,
    decide,
    hard_metrics,
    pixel_area_km2,
    probability_metrics,
    read_head_outputs,
    reliability_table,
    risk_coverage,
    save_json,
)

LOG = logging.getLogger("evaluate-uq")

UNCERTAINTY_LAYERS = (
    "predictive_entropy", "margin_uncertainty",
    "ensemble_total_entropy", "ensemble_expected_entropy", "ensemble_mutual_information",
    "ensemble_variance", "assignment_vote_variance", "parent_entropy", "child_entropy",
)
BREAKDOWNS = ("split_stratum", "encoder_split", "classifier_split", "product", "date")
COVERAGES = (0.5, 0.7, 0.8, 0.9, 0.95, 1.0)
DEFAULT_SHIFT_TESTS = [
    {"name": "encoder_seen_vs_unseen", "column": "encoder_split",
     "in": ["encoder_train"], "out": ["encoder_test"]},
    {"name": "classifier_seen_vs_unseen", "column": "classifier_split",
     "in": ["dev"], "out": ["holdout"]},
]


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------

def _read_layer(path):
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float64)
        nd = src.nodata
    bad = ~np.isfinite(a)
    if nd is not None:
        bad |= a == nd
    bad |= a == OUT_NODATA
    a[bad] = np.nan
    return a


def _target_cfg(cfg, name=None):
    return reference_target_cfg(cfg, name)


LEVEL_KEY = "consensus_level"   # d["refs"] key: primary labels at the consensus level


def level_coverage(d, levels):
    """Consensus references observed in a pixel sample, and whether the sample
    can be assessed at the consensus level (all of them observed when
    require_all_observed, otherwise any)."""
    seen = [n for n in levels.names if n in d["refs"] and (d["refs"][n] >= 0).any()]
    ok = len(seen) == len(levels.names) if levels.require_all_observed else bool(seen)
    return ok, seen


def add_level_reference(d, levels):
    """Add primary labels at the consensus level to a pixel sample (in place).

    Scenes without the consensus references attached/observed get no
    consensus-level labels at all; otherwise their primary positives alone
    would enter the consensus level and bias it toward positives."""
    prim = d["refs"]["primary"]
    covered, seen = level_coverage(d, levels)
    d["level_covered"], d["level_refs_seen"] = covered, seen
    if not covered:
        d["refs"]["_cons"] = np.full(prim.size, -1)
        d["refs"][LEVEL_KEY] = np.full(prim.size, -1)
        return d
    labels = [d["refs"].get(n, np.full(prim.size, -1)) for n in levels.names]
    cons = combine_consensus(labels, levels.require_all_observed)
    d["refs"]["_cons"] = cons
    d["refs"][LEVEL_KEY] = levels.level_reference(prim, levels.levels(prim, cons), "consensus")
    return d


def level_guard(parts, scenes_attached, levels, ev):
    """Coverage of the consensus references over the evaluated scenes, and the
    reason to skip the consensus level (None when it can be assessed)."""
    gcfg = ev.get("consensus_level", {}) or {}
    min_scenes = int(gcfg.get("min_scenes", 1))
    min_neg = int(gcfg.get("min_negative_pixels", 100))
    covered = [p for p in parts if p.get("level_covered")]
    neg = levels.negative_class
    n_neg = int(sum((p["refs"][LEVEL_KEY] == neg).sum() for p in covered))
    attached = sorted({k for a in scenes_attached for k in a})
    cov = {"n_scenes": len(parts), "n_scenes_covered": len(covered),
           "require_all_observed": levels.require_all_observed,
           "consensus_negative_pixels": n_neg,
           "scenes_with_reference_attached": {n: int(sum(n in a for a in scenes_attached))
                                              for n in levels.names},
           "scenes_with_reference_observed": {n: int(sum(n in p.get("level_refs_seen", ()) for p in parts))
                                              for n in levels.names},
           "attached_not_configured": [k for k in attached if k not in levels.names],
           "min_scenes": min_scenes, "min_negative_pixels": min_neg}
    reason = None
    if not any(cov["scenes_with_reference_attached"].values()):
        reason = (f"no consensus reference ({', '.join(levels.names)}) is attached to any evaluated scene"
                  + (f"; attached references are {cov['attached_not_configured']}"
                     if cov["attached_not_configured"] else ""))
    elif len(covered) < min_scenes:
        reason = f"{len(covered)} scene(s) have the consensus references observed (< min_scenes={min_scenes})"
    elif n_neg < min_neg:
        reason = f"{n_neg} consensus negative pixels (< min_negative_pixels={min_neg})"
    return covered, cov, reason


def conflict_strata(d, levels, C):
    """Predicted probability / uncertainty on pixels where primary and consensus references disagree."""
    prim, cons, w = d["refs"]["primary"], d["refs"]["_cons"], d["w"]
    neg = levels.negative_class
    strata = {"primary_negative_consensus_positive": (prim == neg) & (cons >= 0) & (cons != neg),
              "primary_positive_consensus_negative": (prim >= 0) & (prim != neg) & (cons == neg),
              "consensus_negative": (prim == neg) & (cons == neg),
              "primary_negative_unconfirmed": (prim == neg) & (cons != neg)}
    out = {}
    for k, m in strata.items():
        if m.sum() == 0:
            out[k] = {"n_pixels": 0}
            continue
        r = {"n_pixels": int(m.sum()), "weighted_pixels": float(w[m].sum()),
             "predicted_positive_fraction": float(np.average(d["pred"][m] != neg, weights=w[m]))}
        for c in range(C):
            if c != neg:
                r[f"mean_p{c}"] = float(np.average(d["P"][m][:, c], weights=w[m]))
        for u, a in d["unc"].items():
            ok = m & np.isfinite(a)
            if ok.any():
                r[f"mean_{u}"] = float(np.average(a[ok], weights=w[ok]))
        out[k] = r
    return out


def load_scene(scene, prefix, layer_dir, C, cfg, with_refs=True):
    """Full-resolution arrays for one scene."""
    _, profile, _ = read_head_outputs(scene["head_outputs"])
    lay = lambda n: layer_dir / f"{prefix}_{scene['pair_id']}_{n}.tif"
    P = np.stack([_read_layer(lay(f"p_{c}")) for c in range(C)], axis=-1)
    valid = np.isfinite(P).all(axis=-1)
    sc = {"profile": profile, "P": P, "valid": valid,
          "pred": np.nan_to_num(_read_layer(lay("pred_label")), nan=-1).astype(int),
          "unc": {u: _read_layer(lay(u)) for u in UNCERTAINTY_LAYERS if lay(u).exists()},
          "quant": {}, "flag": None, "refs": {}}
    for f in layer_dir.glob(f"{prefix}_{scene['pair_id']}_ensemble_p*_q*.tif"):
        key = f.stem.split(f"{prefix}_{scene['pair_id']}_")[1]   # ensemble_p1_q05
        c, q = key.replace("ensemble_p", "").split("_q")
        sc["quant"].setdefault(int(c), {})[int(q)] = _read_layer(f)
    if lay("selective_flag").exists():
        sc["flag"] = _read_layer(lay("selective_flag"))
    sc["excluded"] = _read_layer(lay("excluded_cluster_mask")) if lay("excluded_cluster_mask").exists() else None
    if with_refs:
        if scene.get("truth"):
            sc["refs"]["primary"] = align_reference(scene["truth"], profile, _target_cfg(cfg))
        for name, path in (scene.get("additional_references") or {}).items():
            try:
                sc["refs"][name] = align_reference(path, profile, _target_cfg(cfg, name))
            except Exception as exc:
                LOG.warning("%s: reference %s not aligned (%s)", scene["pair_id"], name, exc)
    return sc


def sample_scene(sc, max_pixels, rng, C):
    """Pixel sample over valid prediction pixels (refs may be -1 there)."""
    idx = np.flatnonzero(sc["valid"].ravel())
    w_scale = 1.0
    if max_pixels and idx.size > max_pixels:
        w_scale = idx.size / max_pixels
        idx = np.sort(rng.choice(idx, size=max_pixels, replace=False))
    P = sc["P"].reshape(-1, C)[idx]
    P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
    d = {"P": P, "w": np.full(idx.size, w_scale),
         "pred": sc["pred"].ravel()[idx],
         "unc": {u: a.ravel()[idx] for u, a in sc["unc"].items()},
         "quant": {c: {q: a.ravel()[idx] for q, a in qs.items()} for c, qs in sc["quant"].items()},
         "flag": None if sc["flag"] is None else sc["flag"].ravel()[idx],
         "refs": {k: r.ravel()[idx].astype(int) for k, r in sc["refs"].items()}}
    return d


def subsample(d, n, rng):
    m = d["P"].shape[0]
    if not n or m <= n:
        return d
    idx = np.sort(rng.choice(m, size=n, replace=False))
    scale = m / n
    return _take(d, idx, scale)


def _take(d, idx, scale=1.0):
    return {"P": d["P"][idx], "w": d["w"][idx] * scale, "pred": d["pred"][idx],
            "unc": {k: v[idx] for k, v in d["unc"].items()},
            "quant": {c: {q: a[idx] for q, a in qs.items()} for c, qs in d["quant"].items()},
            "flag": None if d["flag"] is None else d["flag"][idx],
            "refs": {k: v[idx] for k, v in d["refs"].items()}}


def concat(parts):
    if not parts:
        return None
    ukeys = set.intersection(*[set(p["unc"]) for p in parts])
    rkeys = set.union(*[set(p["refs"]) for p in parts])
    qkeys = set.intersection(*[set(p["quant"]) for p in parts]) if all(p["quant"] for p in parts) else set()
    n = [p["P"].shape[0] for p in parts]
    return {
        "P": np.concatenate([p["P"] for p in parts]),
        "w": np.concatenate([p["w"] for p in parts]),
        "pred": np.concatenate([p["pred"] for p in parts]),
        "unc": {k: np.concatenate([p["unc"][k] for p in parts]) for k in ukeys},
        "quant": {c: {q: np.concatenate([p["quant"][c][q] for p in parts]) for q in parts[0]["quant"][c]}
                  for c in qkeys},
        "flag": (np.concatenate([p["flag"] for p in parts]) if all(p["flag"] is not None for p in parts) else None),
        "refs": {k: np.concatenate([p["refs"].get(k, np.full(m, -1)) for p, m in zip(parts, n)]) for k in rkeys},
    }


def consensus(refs):
    """Label where >= 2 valid references all agree; -1 otherwise."""
    R = np.stack(list(refs.values()))
    ok = R >= 0
    nvalid = ok.sum(axis=0)
    first = np.where(ok, R, -10).max(axis=0)
    agree = np.all((R == first) | ~ok, axis=0) & (nvalid >= 2)
    disagree = (nvalid >= 2) & ~agree
    return np.where(agree, first, -1), disagree, nvalid >= 2


# ---------------------------------------------------------------------------
# Metric blocks
# ---------------------------------------------------------------------------

def brier_decomposition(p, y, w, n_bins=10):
    """Murphy decomposition with equal-width bins: Brier ~ REL - RES + UNC."""
    W = w.sum()
    ybar = float((w * y).sum() / W)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rel = res = 0.0
    for k in range(n_bins):
        m = b == k
        if not m.any():
            continue
        wk = w[m].sum()
        pk = (w[m] * p[m]).sum() / wk
        ok = (w[m] * y[m]).sum() / wk
        rel += wk * (pk - ok) ** 2
        res += wk * (ok - ybar) ** 2
    return {"brier": float((w * (p - y) ** 2).sum() / W), "reliability": float(rel / W),
            "resolution": float(res / W), "uncertainty": ybar * (1 - ybar)}


def uncertainty_calibration(u, err, implied, w, n_bins=10):
    """Observed vs model-implied error in equal-weight uncertainty bins."""
    ok = np.isfinite(u)
    u, err, implied, w = u[ok], err[ok], implied[ok], w[ok]
    order = np.argsort(u, kind="stable")
    u, err, implied, w = u[order], err[order], implied[order], w[order]
    cw = np.cumsum(w)
    cuts = np.searchsorted(cw, np.linspace(0, cw[-1], n_bins + 1)[1:-1])
    rows, uce = [], 0.0
    for idx in np.split(np.arange(u.size), cuts):
        if idx.size == 0:
            continue
        ww = w[idx].sum()
        r = {"mean_uncertainty": float((w[idx] * u[idx]).sum() / ww),
             "observed_error": float((w[idx] * err[idx]).sum() / ww),
             "implied_error": float((w[idx] * implied[idx]).sum() / ww),
             "weight": float(ww / cw[-1])}
        uce += r["weight"] * abs(r["observed_error"] - r["implied_error"])
        rows.append(r)
    t = pd.DataFrame(rows)
    rho = (float(spearmanr(t["mean_uncertainty"], t["observed_error"])[0])
           if len(t) > 2 and t["mean_uncertainty"].nunique() > 1 and t["observed_error"].nunique() > 1
           else float("nan"))
    return {"uce": float(uce), "bin_spearman": rho, "table": rows}


def interval_coverage(P_c, lo, hi, y, w, n_bins=15):
    """Fraction (by weight) of equal-weight probability bins whose observed
    frequency lies inside the bin-mean ensemble interval."""
    ok = np.isfinite(lo) & np.isfinite(hi)
    P_c, lo, hi, y, w = P_c[ok], lo[ok], hi[ok], y[ok], w[ok]
    order = np.argsort(P_c)
    cw = np.cumsum(w[order])
    cuts = np.searchsorted(cw, np.linspace(0, cw[-1], n_bins + 1)[1:-1])
    cov, rows = 0.0, []
    for idx in np.split(order, cuts):
        if idx.size == 0:
            continue
        ww = w[idx].sum()
        obs = (w[idx] * y[idx]).sum() / ww
        l, h = (w[idx] * lo[idx]).sum() / ww, (w[idx] * hi[idx]).sum() / ww
        inside = bool(l <= obs <= h)
        cov += ww * inside
        rows.append({"mean_p": float((w[idx] * P_c[idx]).sum() / ww), "observed": float(obs),
                     "lo": float(l), "hi": float(h), "inside": inside})
    return {"binned_coverage": float(cov / w.sum()), "mean_width": float(np.average(hi - lo, weights=w)),
            "table": rows}


def selective_metrics(d, ref, C, target_error):
    m = ref >= 0
    if d["flag"] is None or not m.any():
        return None
    keep = (d["flag"][m] == 1)
    pred, r, w = d["pred"][m], ref[m], d["w"][m]
    err = (pred != r).astype(float)
    out = {"coverage": float(w[keep].sum() / w.sum()),
           "retained_error": float(np.average(err[keep], weights=w[keep])) if keep.any() else None,
           "flagged_error": float(np.average(err[~keep], weights=w[~keep])) if (~keep).any() else None,
           "overall_error": float(np.average(err, weights=w)), "target_error": target_error}
    if keep.any():
        out["retained_target_macro_iou"] = hard_metrics(pred[keep], r[keep], w[keep], C)["target_macro_iou"]
    if target_error is not None and out["retained_error"] is not None:
        out["meets_target"] = bool(out["retained_error"] <= float(target_error))
    return out


def uncertainty_quality(unc, err, pred, ref, w, C, conf):
    res, curves = {}, []
    for name, u in unc.items():
        ok = np.isfinite(u)
        if ok.sum() < 10:
            continue
        uu, ee, ww = u[ok], err[ok], w[ok]
        r = {"mean": float(np.average(uu, weights=ww))}
        if 0 < ee.mean() < 1:
            r["error_auroc"] = float(roc_auc_score(ee, uu, sample_weight=ww))
            r["mean_when_correct"] = float(np.average(uu[ee == 0], weights=ww[ee == 0]))
            r["mean_when_error"] = float(np.average(uu[ee == 1], weights=ww[ee == 1]))
        r.update(risk_coverage(uu, ee, ww))
        order = np.argsort(uu, kind="stable")
        cw = np.cumsum(ww[order]) / ww.sum()
        for c in COVERAGES:
            k = min(int(np.searchsorted(cw, c)) + 1, order.size)
            keep = order[:k]
            r[f"target_macro_iou_at_coverage_{int(c * 100)}"] = hard_metrics(
                pred[ok][keep], ref[ok][keep], ww[keep], C)["target_macro_iou"]
        r["calibration"] = uncertainty_calibration(uu, ee, 1.0 - conf[ok], ww)
        res[name] = r
        step = max(1, order.size // 200)
        risk = np.cumsum(ww[order] * ee[order]) / np.cumsum(ww[order])
        curves.append(pd.DataFrame({"uncertainty": name, "coverage": cw[::step], "risk": risk[::step]}))
    return res, (pd.concat(curves, ignore_index=True) if curves else pd.DataFrame())


def evaluate_block(d, ref_key, C, thr, target_error=None, full=True):
    """All pixel-level metrics against one reference (ref_key in d['refs'])."""
    ref_all = d["refs"].get(ref_key)
    if ref_all is None:
        return None, pd.DataFrame()
    m = ref_all >= 0
    if m.sum() < 10:
        return None, pd.DataFrame()
    P, ref, w = d["P"][m], ref_all[m], d["w"][m]
    pred = decide(P, thr)
    Y = np.eye(C)[ref]
    out = probability_metrics(P, ref, Y, w, C, threshold=thr)
    for k in [k for k in out if k.endswith("_vs_soft_target")]:
        out.pop(k)
    for c in out.get("reliability", {}).values():
        c.pop("ece_vs_soft_target", None)
        c.pop("table_vs_soft_target", None)
    tcls = list(range(1, C)) if C == 2 else list(range(1, C))
    out["threshold_free"], out["brier_decomposition"] = {}, {}
    for c in tcls:
        y = (ref == c).astype(int)
        if 0 < y.mean() < 1:
            out["threshold_free"][f"class_{c}"] = {
                "pr_auc": float(average_precision_score(y, P[:, c], sample_weight=w)),
                "roc_auc": float(roc_auc_score(y, P[:, c], sample_weight=w)),
                "prevalence": float(np.average(y, weights=w))}
        out["brier_decomposition"][f"class_{c}"] = brier_decomposition(P[:, c], y.astype(float), w)
    conf = P[np.arange(len(pred)), pred]
    ent = -(P * np.log(np.clip(P, 1e-12, 1))).sum(axis=1) / np.log(C)
    out["sharpness"] = {"mean_confidence": float(np.average(conf, weights=w)),
                        "mean_normalized_entropy": float(np.average(ent, weights=w)),
                        **{f"var_p{c}": float(np.average((P[:, c] - np.average(P[:, c], weights=w)) ** 2, weights=w))
                           for c in tcls}}
    out["n_pixels"] = int(m.sum())
    out["reference_class_fraction"] = (np.bincount(ref, weights=w, minlength=C) / w.sum()).tolist()
    out["predicted_class_fraction"] = (np.bincount(pred, weights=w, minlength=C) / w.sum()).tolist()
    curves = pd.DataFrame()
    if full:
        err = (pred != ref).astype(int)
        out["uncertainty"], curves = uncertainty_quality(
            {k: v[m] for k, v in d["unc"].items()}, err, pred, ref, w, C, conf)
        out["intervals"] = {}
        for c, qs in d["quant"].items():
            if len(qs) >= 2:
                lo, hi = qs[min(qs)][m], qs[max(qs)][m]
                out["intervals"][f"class_{c}"] = {"quantiles": [min(qs) / 100, max(qs) / 100],
                                                  **interval_coverage(P[:, c], lo, hi, (ref == c).astype(float), w)}
        sub = {**d, "refs": {ref_key: ref_all}}
        out["selective"] = selective_metrics(sub, ref_all, C, target_error)
    return out, curves


def headline(o):
    """Flat headline metrics from a block (used for CIs and comparisons)."""
    if o is None:
        return {}
    h = {"target_macro_iou": o["hard_vs_reference"]["target_macro_iou"],
         "target_macro_f1": o["hard_vs_reference"]["target_macro_f1"],
         "nll": o["nll_vs_reference"], "brier": o["brier_vs_reference"],
         "classwise_ece": o["classwise_ece_vs_reference"]}
    tf = o.get("threshold_free", {})
    if tf:
        h["pr_auc"] = float(np.mean([v["pr_auc"] for v in tf.values()]))
        h["roc_auc"] = float(np.mean([v["roc_auc"] for v in tf.values()]))
    for u, r in (o.get("uncertainty") or {}).items():
        if "error_auroc" in r:
            h[f"{u}_error_auroc"] = r["error_auroc"]
        h[f"{u}_aurc"] = r["aurc"]
        h[f"{u}_uce"] = r["calibration"]["uce"]
    if o.get("selective"):
        h["selective_coverage"] = o["selective"]["coverage"]
        if o["selective"]["retained_error"] is not None:
            h["selective_retained_error"] = o["selective"]["retained_error"]
    return h


# ---------------------------------------------------------------------------
# Full-resolution scene metrics: boundaries, objects, areas
# ---------------------------------------------------------------------------

def _boundary(mask, inner):
    b = mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    return b & inner


def boundary_counts(pred_c, ref_c, valid, tol):
    inner = valid & ~ndimage.binary_dilation(~valid, structure=np.ones((3, 3)))
    inner[[0, -1], :] = False
    inner[:, [0, -1]] = False
    bp, br = _boundary(pred_c & valid, inner), _boundary(ref_c & valid, inner)
    dr = ndimage.distance_transform_edt(~br) if br.any() else np.full(br.shape, np.inf)
    dp = ndimage.distance_transform_edt(~bp) if bp.any() else np.full(bp.shape, np.inf)
    return {"pred_matched": int((dr[bp] <= tol).sum()), "pred_total": int(bp.sum()),
            "ref_matched": int((dp[br] <= tol).sum()), "ref_total": int(br.sum())}


def _bf1(c):
    p = c["pred_matched"] / c["pred_total"] if c["pred_total"] else np.nan
    r = c["ref_matched"] / c["ref_total"] if c["ref_total"] else np.nan
    f = 2 * p * r / (p + r) if np.isfinite(p) and np.isfinite(r) and (p + r) > 0 else np.nan
    return {"boundary_precision": p, "boundary_recall": r, "boundary_f1": f}


def object_counts(pred_c, ref_c, valid, min_px, min_overlap):
    st = np.ones((3, 3))
    out = {}
    for kind, a, b in (("ref", ref_c & valid, pred_c & valid), ("pred", pred_c & valid, ref_c & valid)):
        lab, n = ndimage.label(a, structure=st)
        if n == 0:
            out[kind] = (0, 0)
            continue
        ids = np.arange(1, n + 1)
        size = ndimage.sum(np.ones_like(lab), lab, ids)
        frac = ndimage.mean(b.astype(float), lab, ids)
        keep = size >= min_px
        out[kind] = (int(keep.sum()), int((keep & (frac >= min_overlap)).sum()))
    return {"ref_objects": out["ref"][0], "ref_detected": out["ref"][1],
            "pred_objects": out["pred"][0], "pred_confirmed": out["pred"][1]}


def _obj(c):
    pod = c["ref_detected"] / c["ref_objects"] if c["ref_objects"] else np.nan
    far = 1 - c["pred_confirmed"] / c["pred_objects"] if c["pred_objects"] else np.nan
    f = 2 * pod * (1 - far) / (pod + 1 - far) if np.isfinite(pod) and np.isfinite(far) and (pod + 1 - far) > 0 else np.nan
    return {"object_pod": pod, "object_far": far, "object_f1": f}


def exclusion_counts(sc, C):
    """Excluded-cluster pixels vs head-valid pixels and vs reference target pixels."""
    ex, ref = sc.get("excluded"), sc["refs"].get("primary")
    if ex is None:
        return None
    head = np.isfinite(ex)
    out = {"head_pixels": int(head.sum()), "excluded_pixels": int((ex == 1).sum())}
    if ref is not None:
        for c in range(1, C):
            t = head & (ref == c)
            out[f"ref_c{c}_pixels"] = int(t.sum())
            out[f"ref_c{c}_excluded"] = int((t & (ex == 1)).sum())
    return out


def scene_full_metrics(sc, scene, C, ev):
    ref = sc["refs"].get("primary")
    if ref is None:
        return {}, {}
    valid = sc["valid"] & (ref >= 0)
    tols = ev.get("boundary", {}).get("tolerance_pixels", [1, 2])
    ob = ev.get("objects", {})
    min_px, min_ov = int(ob.get("min_object_pixels", 9)), float(ob.get("min_overlap", 0.1))
    area = pixel_area_km2(sc["profile"])
    units = "km2" if area is not None else "pixels"
    a = area if area is not None else np.ones(valid.shape)
    row, counts = {}, {}
    man_areas = (scene.get("areas") or {}).get("reference_valid", {})
    for c in (range(1, C)):
        pc, rc = sc["pred"] == c, ref == c
        for t in tols:
            bc = boundary_counts(pc, rc, valid, t)
            counts[(c, "boundary", t)] = bc
            row.update({f"c{c}_{k}_tol{t}": v for k, v in _bf1(bc).items()})
        oc = object_counts(pc, rc, valid, min_px, min_ov)
        counts[(c, "objects", 0)] = oc
        row.update({f"c{c}_{k}": v for k, v in _obj(oc).items()})
        ra = float(a[rc & valid].sum())
        row[f"c{c}_ref_area_{units}"] = ra
        md = man_areas.get(f"class_{c}")
        if md:
            row[f"c{c}_discrete_area"] = md["discrete"]
            row[f"c{c}_expected_area"] = md["expected"]
            row[f"c{c}_analytic_sd"] = md["analytic_sd"]
            row[f"c{c}_analytic_covers"] = bool(md["analytic_lo"] <= ra <= md["analytic_hi"])
            if "ensemble_lo" in md:
                row[f"c{c}_ensemble_covers"] = bool(md["ensemble_lo"] <= ra <= md["ensemble_hi"])
    row["area_units"] = units
    return row, counts


def pooled_full(per_counts, C, tols):
    out = {}
    for c in range(1, C):
        d = {}
        for t in tols:
            tot = {k: sum(pc[(c, "boundary", t)][k] for pc in per_counts if (c, "boundary", t) in pc)
                   for k in ("pred_matched", "pred_total", "ref_matched", "ref_total")}
            d[f"tol{t}"] = _bf1(tot)
        tot = {k: sum(pc[(c, "objects", 0)][k] for pc in per_counts if (c, "objects", 0) in pc)
               for k in ("ref_objects", "ref_detected", "pred_objects", "pred_confirmed")}
        d["objects"] = {**tot, **_obj(tot)}
        out[f"class_{c}"] = d
    return out


def area_summary(scene_df, C):
    out = {}
    for c in range(1, C):
        rcol = [k for k in scene_df.columns if k.startswith(f"c{c}_ref_area_")]
        if not rcol or f"c{c}_discrete_area" not in scene_df:
            continue
        r = scene_df[rcol[0]].to_numpy(float)
        d = {}
        for kind in ("discrete", "expected"):
            p = scene_df[f"c{c}_{kind}_area"].to_numpy(float)
            d[kind] = {"bias": float(np.mean(p - r)), "mae": float(np.mean(np.abs(p - r))),
                       "relative_bias": float((p.sum() - r.sum()) / r.sum()) if r.sum() > 0 else None,
                       "pearson_r": float(pearsonr(p, r)[0]) if len(r) > 2 and np.std(p) > 0 and np.std(r) > 0 else None}
        for k in ("analytic", "ensemble"):
            col = f"c{c}_{k}_covers"
            if col in scene_df:
                d[f"{k}_interval_coverage"] = float(scene_df[col].astype(float).mean())
        out[f"class_{c}"] = d
    return out


def scene_reference_areas(scene, C, cfg):
    """Reference class areas over reference-valid pixels (same units as inference)."""
    _, profile, _ = read_head_outputs(scene["head_outputs"])
    ref = align_reference(scene["truth"], profile, _target_cfg(cfg))
    a = pixel_area_km2(profile)
    a = np.ones(ref.shape) if a is None else a
    return {c: float(a[ref == c].sum()) for c in range(1, C)}


def conformal_area_intervals(manifest, scene_df, C, cfg, level=0.9):
    """
    Split-conformal scene-area intervals. Analytic (independent-pixel) and
    ensemble intervals ignore spatial error correlation and are too narrow, so
    the analytic SD is rescaled by q_hat, the conformal quantile of
    |A_ref - A_expected| / SD over calibration-split scenes, and the result is
    checked on the evaluated scenes:  A_expected +/- q_hat * SD.
    """
    cal = [s for s in manifest["scenes"] if s.get("classifier_split") == "calibration" and s.get("truth")
           and not s.get("skipped")
           and (s.get("areas") or {}).get("reference_valid")]
    out = {}
    if len(cal) < 5:
        return {"skipped": f"needs >= 5 calibration scenes with inference outputs (found {len(cal)})"}
    ref_areas = [scene_reference_areas(s, C, cfg) for s in cal]
    for c in range(1, C):
        sc, n = [], 0
        for s, ra in zip(cal, ref_areas):
            md = s["areas"]["reference_valid"].get(f"class_{c}")
            if md is None:
                continue
            sc.append(abs(ra[c] - md["expected"]) / max(md["analytic_sd"], 1e-9))
        n = len(sc)
        if n < 5:
            continue
        k = min(n, int(np.ceil((n + 1) * level)))
        q = float(np.sort(sc)[k - 1])
        d = {"level": level, "n_calibration": n, "q_hat": q}
        rc = [k_ for k_ in scene_df.columns if k_.startswith(f"c{c}_ref_area_")]
        if rc and f"c{c}_expected_area" in scene_df and f"c{c}_analytic_sd" in scene_df:
            e, sd, r = (scene_df[f"c{c}_expected_area"], scene_df[f"c{c}_analytic_sd"], scene_df[rc[0]])
            lo, hi = np.maximum(0, e - q * sd), e + q * sd
            scene_df[f"c{c}_conformal_lo"], scene_df[f"c{c}_conformal_hi"] = lo, hi
            d["evaluated_coverage"] = float(((r >= lo) & (r <= hi)).mean())
            d["mean_width"] = float((hi - lo).mean())
        out[f"class_{c}"] = d
    return out


def exclusion_summary(excl_counts, C, manifest, ev):
    """
    Pooled exclusion fractions. Excluded clusters should hold almost no
    reference smoke / fire; a large share means a listed cluster is a real
    (perhaps ambiguous) class, not an artifact, and evaluation is biased.
    """
    if not excl_counts:
        return None
    tot = {k: sum(e.get(k, 0) for e in excl_counts) for k in set().union(*excl_counts)}
    out = {"invalid_clusters": manifest.get("invalid_clusters"),
           "excluded_fraction": tot["excluded_pixels"] / max(tot["head_pixels"], 1)}
    lim = float(ev.get("max_excluded_reference_fraction", 0.02))
    for c in range(1, C):
        n = tot.get(f"ref_c{c}_pixels", 0)
        if n:
            f = tot[f"ref_c{c}_excluded"] / n
            out[f"excluded_reference_class{c}_fraction"] = f
            if f > lim:
                LOG.warning("Excluded clusters contain %.1f%% of reference class-%d pixels (> %.1f%%). "
                            "Check invalid_clusters: these may be real or ambiguous clusters.", 100 * f, c, 100 * lim)
                out[f"warning_class{c}"] = f"excluded share {f:.3f} exceeds {lim:.3f}"
    return out


# ---------------------------------------------------------------------------
# Bootstrap, reference disagreement, shift tests
# ---------------------------------------------------------------------------

def group_bootstrap(boot_parts, C, thr, target_error, B, seed, ref_key="primary"):
    """Percentile CIs over split_group resamples. Returns (replicates df, ci dict)."""
    groups = sorted(boot_parts)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(B):
        draw = rng.choice(len(groups), size=len(groups), replace=True)
        d = concat([p for i in draw for p in boot_parts[groups[i]]])
        o, _ = evaluate_block(d, ref_key, C, thr, target_error, full=True)
        reps.append(headline(o))
    df = pd.DataFrame(reps)
    return df


def ci_table(df, level=0.9):
    a = (1 - level) / 2
    return {k: {"mean": float(df[k].mean()), "lo": float(df[k].quantile(a)), "hi": float(df[k].quantile(1 - a))}
            for k in df.columns if df[k].notna().sum() > 1}


def disagreement_test(d):
    if len(d["refs"]) < 2:
        return None
    cons, dis, both = consensus(d["refs"])
    if both.sum() < 10 or dis[both].mean() in (0.0, 1.0):
        return {"n_pixels": int(both.sum()), "disagreement_rate": float(dis[both].mean()) if both.any() else None}
    out = {"n_pixels": int(both.sum()),
           "disagreement_rate": float(np.average(dis[both], weights=d["w"][both])), "layers": {}}
    for k, u in d["unc"].items():
        ok = both & np.isfinite(u)
        y = dis[ok].astype(int)
        if 0 < y.mean() < 1:
            out["layers"][k] = {
                "disagreement_auroc": float(roc_auc_score(y, u[ok], sample_weight=d["w"][ok])),
                "mean_agree": float(np.average(u[ok][y == 0], weights=d["w"][ok][y == 0])),
                "mean_disagree": float(np.average(u[ok][y == 1], weights=d["w"][ok][y == 1]))}
    return out


def reference_free_summary(manifest, scenes, prefix, layer_dir, C, cfg, ev, rng, model, thr, splits, eval_dir):
    """Evaluation for scenes without a primary reference (e.g. a new event with
    no ADP): per-scene and pooled uncertainty, predicted target fraction and
    selective retention, plus the shift tests. No reference-based metrics."""
    n_px = int(ev.get("shift_pixels_per_scene", 5000))
    rows, samples, pooled = [], [], {}
    for s in scenes:
        sc = load_scene(s, prefix, layer_dir, C, cfg, with_refs=False)
        d = subsample(sample_scene(sc, None, rng, C), n_px, rng)
        meta = {k: s.get(k) for k in BREAKDOWNS}
        samples.append((meta, d["unc"]))
        row = {"pair_id": s["pair_id"], **meta, "n_valid": int(sc["valid"].sum())}
        for c in range(1, C):
            row[f"predicted_class{c}_fraction"] = float(np.mean(d["pred"] == c)) if d["pred"].size else np.nan
            row[f"mean_p{c}"] = float(np.mean(d["P"][:, c])) if d["P"].size else np.nan
        if d["flag"] is not None and d["flag"].size:
            row["selective_retained_fraction"] = float(np.nanmean(d["flag"]))
        for u, a in d["unc"].items():
            row[f"mean_{u}"] = float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
            pooled.setdefault(u, []).append(a[np.isfinite(a)])
        rows.append(row)
    scene_df = pd.DataFrame(rows)
    scene_df.to_csv(eval_dir / "per_scene_reference_free.csv", index=False)
    qs = (0.05, 0.25, 0.5, 0.75, 0.95)
    unc = {}
    for u, arrs in pooled.items():
        a = np.concatenate(arrs) if arrs else np.array([])
        if a.size:
            unc[u] = {"mean": float(a.mean()), **{f"q{int(q * 100):02d}": float(np.quantile(a, q)) for q in qs}}
    summary = {"model": model, "threshold": thr, "splits": splits, "reference_free": True,
               "n_scenes": 0, "n_scenes_inferred": len(scenes), "overall": None,
               "skipped_reference_metrics": "no evaluated scene has a primary reference (truth) raster",
               "uncertainty": unc,
               "predicted_fraction": {k: float(scene_df[k].mean()) for k in scene_df
                                      if k.startswith("predicted_class")},
               "areas": {"scene_areas_csv": str(layer_dir / "scene_areas.csv")}}
    if "selective_retained_fraction" in scene_df:
        summary["selective_retained_fraction"] = float(scene_df["selective_retained_fraction"].mean())
    tests = ev.get("shift_tests", DEFAULT_SHIFT_TESTS)
    if tests:
        summary["shift_tests"] = shift_tests(samples, tests)
    save_json(eval_dir / "eval_summary.json", summary)
    LOG.warning("%s: no evaluated scene has a reference raster; wrote reference-free summary "
                "(uncertainty, predicted fraction, shift tests) to %s", model, eval_dir)
    return summary


def shift_tests(shift_samples, tests):
    """shift_samples: list of (scene meta, {layer: sample array})."""
    out = {}
    for t in tests:
        col = t["column"]
        ins = [s for s in shift_samples if str(s[0].get(col)) in set(map(str, t["in"]))]
        outs = [s for s in shift_samples if str(s[0].get(col)) in set(map(str, t["out"]))]
        if not ins or not outs:
            out[t["name"]] = {"skipped": f"needs scenes with {col} in {t['in']} and {t['out']}",
                              "n_in": len(ins), "n_out": len(outs)}
            continue
        res = {"n_scenes_in": len(ins), "n_scenes_out": len(outs), "layers": {}}
        layers = set.intersection(*[set(s[1]) for s in ins + outs])
        for k in sorted(layers):
            a = np.concatenate([s[1][k] for s in ins])
            b = np.concatenate([s[1][k] for s in outs])
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if a.size < 10 or b.size < 10:
                continue
            y = np.r_[np.zeros(a.size), np.ones(b.size)]
            ma = [np.nanmean(s[1][k]) for s in ins]
            mb = [np.nanmean(s[1][k]) for s in outs]
            r = {"pixel_auroc_out_vs_in": float(roc_auc_score(y, np.r_[a, b])),
                 "mean_in": float(a.mean()), "mean_out": float(b.mean()),
                 "scene_mean_in": float(np.mean(ma)), "scene_mean_out": float(np.mean(mb))}
            if len(ma) >= 2 and len(mb) >= 2:
                r["scene_mannwhitney_p"] = float(mannwhitneyu(mb, ma, alternative="greater").pvalue)
            res["layers"][k] = r
        out[t["name"]] = res
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_outputs(out_dir, rel_rows, curves, data, overall, scene_df, C):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cls = list(range(1, C))
    fig, axes = plt.subplots(1, len(cls), figsize=(4.4 * len(cls), 4.2), squeeze=False)
    for ax, c in zip(axes[0], cls):
        for grp, tab in rel_rows.get(c, {}).items():
            t = pd.DataFrame(tab)
            ax.plot(t["mean_predicted"], t["observed_frequency"], marker="o", ms=3,
                    lw=2 if grp == "overall" else 1, label=grp)
        iv = (overall.get("intervals") or {}).get(f"class_{c}")
        if iv:
            t = pd.DataFrame(iv["table"])
            ax.fill_between(t["mean_p"], t["lo"], t["hi"], color="0.6", alpha=0.25, label="ensemble interval")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency",
               title=f"Reliability, class {c}", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "reliability.png", dpi=150)
    plt.close(fig)

    if not curves.empty:
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        for name, g in curves.groupby("uncertainty"):
            ax.plot(g["coverage"], g["risk"], label=name, lw=1.2)
        sel = overall.get("selective")
        if sel and sel.get("retained_error") is not None:
            ax.scatter([sel["coverage"]], [sel["retained_error"]], marker="*", s=90, c="k", zorder=5,
                       label="selective product")
        top = curves.loc[curves["coverage"] >= 0.05, "risk"].max()
        ax.set(xlabel="Coverage (lowest-uncertainty pixels kept)", ylabel="Error rate",
               title="Risk-coverage", ylim=(0, 1.15 * top if np.isfinite(top) and top > 0 else None))
        ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout()
        fig.savefig(out_dir / "risk_coverage.png", dpi=150)
        plt.close(fig)

    unc = overall.get("uncertainty") or {}
    if unc:
        names = list(unc)
        ncol = min(3, len(names))
        nrow = int(np.ceil(len(names) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.5 * nrow), squeeze=False)
        for ax, n in zip(axes.ravel(), names):
            t = pd.DataFrame(unc[n]["calibration"]["table"])
            ax.plot(t["mean_uncertainty"], t["observed_error"], "o-", ms=3, label="observed")
            ax.plot(t["mean_uncertainty"], t["implied_error"], "s--", ms=3, label="implied (1-conf)")
            ax.set(title=f"{n}\nUCE={unc[n]['calibration']['uce']:.3f}", xlabel="uncertainty", ylabel="error")
            ax.title.set_fontsize(8)
            ax.legend(fontsize=6.5)
        for ax in axes.ravel()[len(names):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "uncertainty_calibration.png", dpi=150)
        plt.close(fig)

    names = [u for u in ("predictive_entropy", "ensemble_expected_entropy", "ensemble_mutual_information",
                         "margin_uncertainty") if u in data["unc"]]
    if names:
        err, m = data["err"], data["refs"]["primary"] >= 0
        fig, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 3.4), squeeze=False)
        for ax, n in zip(axes[0], names):
            u = data["unc"][n]
            ok = m & np.isfinite(u)
            ax.hist(u[ok & (err == 0)], bins=50, alpha=0.6, density=True, label="correct")
            ax.hist(u[ok & (err == 1)], bins=50, alpha=0.6, density=True, label="error")
            ax.set(title=n, xlabel="value", ylabel="density")
            ax.title.set_fontsize(8)
            ax.legend(fontsize=6.5)
        fig.tight_layout()
        fig.savefig(out_dir / "uncertainty_histograms.png", dpi=150)
        plt.close(fig)

    for c in cls:
        rc = [k for k in scene_df.columns if k.startswith(f"c{c}_ref_area_")]
        if not rc or f"c{c}_expected_area" not in scene_df:
            continue
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        r = scene_df[rc[0]]
        ax.scatter(r, scene_df[f"c{c}_discrete_area"], s=14, label="discrete", marker="x")
        e = scene_df[f"c{c}_expected_area"]
        if f"c{c}_conformal_lo" in scene_df:
            ax.errorbar(r, e, yerr=[e - scene_df[f"c{c}_conformal_lo"], scene_df[f"c{c}_conformal_hi"] - e],
                        fmt="o", ms=3.5, lw=0.8, capsize=2, label="expected + conformal interval")
        else:
            ax.scatter(r, e, s=14, label="expected")
        mx = float(np.nanmax([r.max(), scene_df[f"c{c}_discrete_area"].max(), scene_df[f"c{c}_expected_area"].max()]))
        ax.plot([0, mx], [0, mx], "k--", lw=0.8)
        ax.set(xlabel=f"Reference area ({scene_df['area_units'].iloc[0]})", ylabel="Predicted area",
               title=f"Scene area, class {c}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"area_class{c}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def evaluate_model(cfg, model):
    out = Path(cfg["output_dir"])
    ev = cfg.get("evaluation", {}) or {}
    layer_dir = Path(ev.get("inference_dir_template", str(out / "inference" / "{model}")).format(model=model))
    manifest = load_json(layer_dir / "inference_manifest.json")
    C = int(manifest.get("n_classes", len(cfg["target"]["classes"])))
    thr = manifest.get("threshold")
    prefix = manifest.get("layer_prefix", model)
    target_error = (manifest.get("selective") or {}).get("target_error")
    splits = ev.get("splits", ["holdout"])
    scenes = [s for s in manifest["scenes"] if (not splits or s.get("classifier_split") in set(splits))
              and not s.get("skipped")]
    if not scenes:
        raise ValueError(f"No scenes with classifier_split in {splits} in {layer_dir}.")
    seed = int(cfg.get("random_state", 42))
    rng = np.random.default_rng(seed)
    max_px = ev.get("max_pixels_per_scene")
    bcfg = ev.get("bootstrap", {}) or {}
    tols = ev.get("boundary", {}).get("tolerance_pixels", [1, 2])
    eval_dir = Path(ev.get("output_dir", out / "evaluation")) / model
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not any(s.get("truth") for s in scenes):
        return reference_free_summary(manifest, scenes, prefix, layer_dir, C, cfg, ev, rng,
                                      model, thr, splits, eval_dir), None, None

    per_scene, parts, per_counts, boot_parts, excl_counts = [], [], [], {}, []
    scenes_attached = []
    levels = LabelLevels(cfg)
    for s in scenes:
        if not s.get("truth"):
            LOG.warning("Scene %s has no truth raster; skipped.", s["pair_id"])
            continue
        sc = load_scene(s, prefix, layer_dir, C, cfg)
        d = sample_scene(sc, max_px, rng, C)
        if (d["refs"]["primary"] >= 0).sum() == 0:
            continue
        if levels:
            add_level_reference(d, levels)
            scenes_attached.append(set(s.get("additional_references") or {}))
        d["meta"] = {k: str(s.get(k, "")) for k in BREAKDOWNS + ("split_group",)}
        parts.append(d)
        boot_parts.setdefault(d["meta"]["split_group"], []).append(
            subsample(d, int(bcfg.get("max_pixels_per_scene", 5000)), rng))
        full_row, counts = scene_full_metrics(sc, s, C, ev)
        per_counts.append(counts)
        exc = exclusion_counts(sc, C)
        if exc:
            excl_counts.append(exc)
            full_row["excluded_fraction"] = exc["excluded_pixels"] / max(exc["head_pixels"], 1)
            for c in range(1, C):
                if exc.get(f"ref_c{c}_pixels"):
                    full_row[f"excluded_ref_c{c}_fraction"] = exc[f"ref_c{c}_excluded"] / exc[f"ref_c{c}_pixels"]
        m = d["refs"]["primary"] >= 0
        pred, ref, w = d["pred"][m], d["refs"]["primary"][m], d["w"][m]
        hm = hard_metrics(pred, ref, w, C)
        Y = np.eye(C)[ref]
        eces = [reliability_table(d["P"][m][:, c], Y[:, c], w)[1] for c in range(1, C)]
        row = {"pair_id": s["pair_id"], **d["meta"], "n_pixels": int(m.sum()),
               "target_macro_iou": hm["target_macro_iou"], "target_macro_f1": hm["target_macro_f1"],
               "error_rate": float(np.average(pred != ref, weights=w)),
               "classwise_ece": float(np.mean(eces)), **full_row}
        sel = selective_metrics(d, d["refs"]["primary"], C, target_error)
        if sel:
            row.update({f"selective_{k}": v for k, v in sel.items() if not isinstance(v, dict)})
        for u, a in d["unc"].items():
            row[f"mean_{u}"] = float(np.nanmean(a[m]))
        if levels:
            ml = d["refs"][LEVEL_KEY] >= 0
            row["consensus_level_covered"] = bool(d["level_covered"])
            row["consensus_refs_observed"] = ";".join(d["level_refs_seen"])
            row["consensus_level_pixels"] = int(ml.sum())
            if d["level_covered"] and ml.any():
                hl = hard_metrics(d["pred"][ml], d["refs"][LEVEL_KEY][ml], d["w"][ml], C)
                row["consensus_level_target_macro_iou"] = hl["target_macro_iou"]
                row["consensus_level_error_rate"] = float(np.average(
                    d["pred"][ml] != d["refs"][LEVEL_KEY][ml], weights=d["w"][ml]))
        per_scene.append(row)
        LOG.info("%s: IoU=%.3f ECE=%.3f n=%d", s["pair_id"], row["target_macro_iou"],
                 row["classwise_ece"], row["n_pixels"])

    if not parts:
        raise ValueError(f"{model}: no evaluated scene has valid reference pixels "
                         "(check target.value_map / nodata_values against the reference rasters).")
    scene_df = pd.DataFrame(per_scene)
    scene_df.to_csv(eval_dir / "per_scene_metrics.csv", index=False)
    allp = concat(parts)

    summary = {"model": model, "threshold": thr, "splits": splits, "n_scenes": len(parts),
               "n_groups": len(boot_parts), "selective_rule": manifest.get("selective")}
    summary["overall"], curves = evaluate_block(allp, "primary", C, thr, target_error)
    curves.to_csv(eval_dir / "risk_coverage.csv", index=False)
    summary["discrete_full_resolution"] = pooled_full(per_counts, C, tols)
    summary["excluded_clusters"] = exclusion_summary(excl_counts, C, manifest, ev)
    summary["areas"] = area_summary(scene_df, C)
    summary["areas"]["conformal"] = conformal_area_intervals(
        manifest, scene_df, C, cfg, float((ev.get("areas", {}) or {}).get("interval_level", 0.9)))
    if "skipped" not in summary["areas"]["conformal"]:
        save_json(eval_dir / "area_interval_calibration.json", summary["areas"]["conformal"])
    scene_df.to_csv(eval_dir / "per_scene_metrics.csv", index=False)

    # multi-reference and consensus
    level_ok = False
    if levels:
        covered, cov, reason = level_guard(parts, scenes_attached, levels, ev)
        o_lvl = None
        if reason is None:
            sub = concat(covered)
            o_lvl, _ = evaluate_block(sub, LEVEL_KEY, C, thr, target_error, full=True)
            if o_lvl is None:
                reason = "fewer than 10 consensus-level pixels on the covered scenes"
        if reason is not None:
            LOG.warning("Consensus level skipped: %s. Attach the consensus references with "
                        "attach_references.py --add NAME=MATCHUPS (NAME must match "
                        "target.label_levels.consensus_references).", reason)
            summary["by_label_level"] = {"skipped": reason, "definition": levels.describe(),
                                         "coverage": cov}
        else:
            level_ok = True
            neg = levels.negative_class
            prim, lv = sub["refs"]["primary"], sub["refs"][LEVEL_KEY]
            o_prim_cov, _ = evaluate_block(sub, "primary", C, thr, target_error, full=True)
            summary["by_label_level"] = {
                "definition": levels.describe(),
                "coverage": cov,
                "primary": headline(summary["overall"]),
                "primary_covered_scenes": headline(o_prim_cov),
                "consensus": headline(o_lvl),
                "primary_negative_pixels": int((prim == neg).sum()),
                "consensus_negative_pixels": int((lv == neg).sum()),
                "conflicts": conflict_strata(sub, levels, C),
            }
            summary["consensus_level_detail"] = o_lvl
            if len(covered) < len(parts):
                LOG.warning("Consensus level assessed on %d of %d scenes (consensus references "
                            "not observed on the rest); compare with primary_covered_scenes.",
                            len(covered), len(parts))
    extra = sorted(k for k in allp["refs"] if k not in ("primary", LEVEL_KEY, "_cons"))
    if extra:
        summary["by_reference"] = {}
        for k in extra:
            o, _ = evaluate_block(allp, k, C, thr, target_error, full=True)
            summary["by_reference"][k] = headline(o)
        cons, _, _ = consensus(allp["refs"])
        allp["refs"]["consensus"] = cons
        o, _ = evaluate_block(allp, "consensus", C, thr, target_error, full=True)
        summary["by_reference"]["consensus"] = headline(o)
        summary["by_reference"]["consensus_pixels"] = int((cons >= 0).sum())
        allp["refs"].pop("consensus")
        summary["reference_disagreement"] = disagreement_test(
            {**allp, "refs": {k: v for k, v in allp["refs"].items() if k not in (LEVEL_KEY, "_cons")}})

    # breakdowns
    rel_rows = {}
    for c, r in summary["overall"]["reliability"].items():
        rel_rows.setdefault(int(c.split("_")[1]), {})["overall"] = r["table_vs_reference"]
    for col in BREAKDOWNS:
        vals = sorted({p["meta"][col] for p in parts})
        if len(vals) < 2:
            continue
        summary[f"by_{col}"] = {}
        for v in vals:
            sub = concat([p for p in parts if p["meta"][col] == v])
            o, _ = evaluate_block(sub, "primary", C, thr, target_error, full=True)
            if o is None:
                continue
            if col in ("split_stratum", "encoder_split"):
                for c, r in o["reliability"].items():
                    rel_rows.setdefault(int(c.split("_")[1]), {})[f"{col}={v}"] = r["table_vs_reference"]
            summary[f"by_{col}"][v] = headline(o)
    for c, groups in rel_rows.items():
        pd.concat([pd.DataFrame(t).assign(group=g) for g, t in groups.items()]).to_csv(
            eval_dir / f"reliability_class{c}.csv", index=False)
    rows = []
    for n, r in (summary["overall"].get("uncertainty") or {}).items():
        rows += [{"uncertainty": n, **b} for b in r["calibration"]["table"]]
    pd.DataFrame(rows).to_csv(eval_dir / "uncertainty_calibration.csv", index=False)

    # scene-level reliability of uncertainty
    summary["scene_level"] = {}
    for u in UNCERTAINTY_LAYERS:
        col = f"mean_{u}"
        if col in scene_df and scene_df[col].notna().sum() >= 3:
            rho, p = spearmanr(scene_df[col], scene_df["error_rate"], nan_policy="omit")
            rho2, p2 = spearmanr(scene_df[col], scene_df["target_macro_iou"], nan_policy="omit")
            summary["scene_level"][u] = {"spearman_vs_error_rate": float(rho), "p_error": float(p),
                                         "spearman_vs_iou": float(rho2), "p_iou": float(p2)}

    # distribution-shift (epistemic) tests over all scenes in shift_splits
    tests = ev.get("shift_tests", DEFAULT_SHIFT_TESTS)
    if tests:
        shift_splits = ev.get("shift_splits")
        n_px = int(ev.get("shift_pixels_per_scene", 5000))
        samples = []
        for s in manifest["scenes"]:
            if s.get("skipped") or (shift_splits and s.get("classifier_split") not in set(shift_splits)):
                continue
            sc = load_scene(s, prefix, layer_dir, C, cfg, with_refs=False)
            dd = subsample(sample_scene(sc, None, rng, C), n_px, rng)
            samples.append(({k: s.get(k) for k in BREAKDOWNS}, dd["unc"]))
        summary["shift_tests"] = shift_tests(samples, tests)

    # bootstrap over groups
    B = int(bcfg.get("n_resamples", 200))
    reps = None
    if B > 0 and len(boot_parts) >= 2:
        reps = group_bootstrap(boot_parts, C, thr, target_error, B, seed)
        if level_ok:
            cgroups = {p["meta"]["split_group"] for p in parts if p.get("level_covered")}
            if len(cgroups) >= 2:
                lr = group_bootstrap({g: boot_parts[g] for g in cgroups}, C, thr, target_error,
                                     B, seed, ref_key=LEVEL_KEY)
                reps = pd.concat([reps, lr.add_prefix("consensus_level_")], axis=1)
            else:
                LOG.warning("Consensus-level CIs need >= 2 covered evaluation groups; skipped.")
        reps.to_csv(eval_dir / "bootstrap_replicates.csv", index=False)
        summary["bootstrap_ci"] = {"level": float(bcfg.get("ci_level", 0.9)), "n_resamples": B,
                                   "unit": "split_group",
                                   "metrics": ci_table(reps, float(bcfg.get("ci_level", 0.9)))}
    elif B > 0:
        LOG.warning("Bootstrap CIs need >= 2 evaluation groups; skipped.")

    allp["err"] = (decide(allp["P"], thr) != allp["refs"]["primary"]).astype(int)
    plot_outputs(eval_dir, rel_rows, curves, allp, summary["overall"], scene_df, C)
    save_json(eval_dir / "eval_summary.json", summary)
    o = summary["overall"]
    LOG.info("%s: IoU=%.3f ECE=%.3f NLL=%.3f -> %s", model, o["hard_vs_reference"]["target_macro_iou"],
             o["classwise_ece_vs_reference"], o["nll_vs_reference"], eval_dir)
    return summary, reps, sorted(boot_parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_config", type=Path)
    ap.add_argument("--models", nargs="+", default=None)
    a = ap.parse_args()
    cfg = load_yaml(a.yaml_config)
    ev = cfg.get("evaluation", {}) or {}
    models = a.models or ev.get("models") or [cfg.get("inference", {}).get("model")]
    out = Path(ev.get("output_dir", Path(cfg["output_dir"]) / "evaluation"))
    rows, reps, groups = [], {}, {}
    for m in models:
        s, r, g = evaluate_model(cfg, m)
        row = {"model": m, **headline(s["overall"])}
        if s.get("reference_free"):
            row.update({f"mean_{u}": v["mean"] for u, v in s["uncertainty"].items()})
            row.update(s.get("predicted_fraction", {}))
        if (s.get("by_label_level") or {}).get("consensus"):
            row.update({f"consensus_level_{k}": v for k, v in s["by_label_level"]["consensus"].items()})
        for k, v in (s.get("bootstrap_ci", {}).get("metrics", {}) or {}).items():
            row[f"{k}_lo"], row[f"{k}_hi"] = v["lo"], v["hi"]
        rows.append(row)
        reps[m], groups[m] = r, g
    if len(rows) > 1:
        pd.DataFrame(rows).to_csv(out / "model_comparison.csv", index=False)
        LOG.info("Model comparison: %s", out / "model_comparison.csv")

    ref_model = ev.get("reference_model")
    if ref_model in reps and reps[ref_model] is not None:
        level = float((ev.get("bootstrap", {}) or {}).get("ci_level", 0.9))
        a_ = (1 - level) / 2
        prow = []
        for m in models:
            if m == ref_model or reps[m] is None or groups[m] != groups[ref_model]:
                continue
            for k in sorted(set(reps[m].columns) & set(reps[ref_model].columns)):
                dlt = (reps[m][k] - reps[ref_model][k]).dropna()
                if dlt.size < 2:
                    continue
                prow.append({"model": m, "reference_model": ref_model, "metric": k,
                             "mean_difference": float(dlt.mean()), "lo": float(dlt.quantile(a_)),
                             "hi": float(dlt.quantile(1 - a_)), "fraction_positive": float((dlt > 0).mean())})
        if prow:
            pd.DataFrame(prow).to_csv(out / "paired_comparison.csv", index=False)
            LOG.info("Paired comparison vs %s: %s", ref_model, out / "paired_comparison.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
