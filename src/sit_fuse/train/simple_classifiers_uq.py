#!/usr/bin/env python3
"""
Train calibrated SIT-FUSE context classifiers with soft targets from cluster
agreement and an assignment-procedure ensemble.

Pipeline (all leakage-sensitive steps are refit inside every CV fold):
  1. Split explicit holdout dates; group all CV by date.
  2. Per fold: fit ContextEncoder (agreement fractions, best-agreement
     assignment, assignment ensemble) on training-fold dates only. Training-row
     features use inner out-of-fold encoders; validation rows use the
     training-fold encoder.
  3. Soft targets (target.mode), no ambiguity masking. Fit with the
     soft-label expansion trick.
  4. Cross-fitted (or calibration-date) predictions -> prior correction +
     temperature/Platt/isotonic calibration, decision threshold.
     calibration.label_level (option A): primary (all primary-reference
     labels) or consensus (negatives only where the consensus references
     confirm them; target.label_levels). calibration.noise_correction
     (option C): calibrate against a primary-label transition matrix
     (fixed, or estimated against the consensus references).
  5. Final fit on development dates; holdout evaluation at natural
     prevalence (population weights); bootstrap ensembles.

Baselines evaluated alongside the classifiers (no fitting beyond the encoder):
  hard_assignment, soft_assignment, assignment_ensemble

Usage:
    python simple_classifiers_uq.py uq_config.yaml
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from sit_fuse.train.classifier_utils import (
    build_cv_splitter,
    build_models as _build_models,
    holdout_split,
    load_json,
    load_yaml,
    normalize_holdout_dates,
)

from sf_uq_utils import (
    EPS,
    ContextEncoder,
    LabelLevels,
    ProbabilityCalibrator,
    bootstrap_member_probabilities,
    build_targets,
    decide,
    effective_training_prior,
    ensemble_decomposition,
    expand_soft_targets,
    forest_member_probabilities,
    full_proba,
    margin_uncertainty,
    predictive_entropy_norm,
    probability_metrics,
    read_table,
    resolve_transition,
    save_json,
    select_feature_columns,
    select_selective_threshold,
    select_threshold,
    select_threshold_multiclass,
    threshold_grid,
    uq_feature_signature,
    write_table,
)

LOG = logging.getLogger("simple-classifiers-uq")
BASELINES = ("hard_assignment", "soft_assignment", "assignment_ensemble")


# ---------------------------------------------------------------------------
# Models (classifier_utils.build_models)
# ---------------------------------------------------------------------------

def build_models(cfg):
    """
    classifier_utils.build_models, restricted to the requested classifiers.
    Oversampling is rejected: soft targets and population weights already
    define the training distribution, and prior correction assumes it.
    class_weight settings (default 'balanced') are allowed; their effect is
    removed by prior correction via effective_training_prior().
    """
    if cfg.get("oversampling", {}).get("enabled", False):
        raise ValueError("oversampling.enabled must be false for soft-target training.")
    requested = cfg.get("models", ["logistic_regression", "random_forest"])
    models = _build_models(cfg)
    unknown = set(requested) - set(models) - set(BASELINES)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    return models


def fit_soft(model, X, T, cfg, return_prior=False, base_weight=None):
    """Fit on soft targets via row expansion; optionally return effective prior."""
    Xe, ye, we = expand_soft_targets(X, T, base_weight=base_weight,
                                     min_weight=float(cfg.get("soft_target_min_weight", 1e-4)))
    model.fit(Xe, ye, clf__sample_weight=we)
    if return_prior:
        return model, effective_training_prior(model, ye, we, T.shape[1])
    return model


def baseline_proba(name, frame, enc):
    if name == "hard_assignment":
        P = enc.soft_matrix(frame, "assign_hard")
    elif name == "soft_assignment":
        P = enc.soft_matrix(frame, "sa_joint")
    elif name == "assignment_ensemble":
        P = enc.soft_matrix(frame, "assign_vote_soft")
    else:
        raise ValueError(name)
    P = np.clip(P, EPS, 1.0)
    return P / P.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Encoders (fold-dependent features)
# ---------------------------------------------------------------------------

class EncoderFactory:
    """Fits ContextEncoders on the full-scene counts of a set of groups (G column)."""

    def __init__(self, counts, n_classes, num_child, cfg, group_col="date"):
        self.counts = counts
        self.G = group_col
        self.C = n_classes
        self.num_child = num_child
        self.cfg = cfg
        self.seed = int(cfg.get("random_state", 42))

    def fit(self, dates, members=None):
        ecfg = dict(self.cfg.get("context_encoder", {}))
        if members is not None:
            ecfg["assignment_ensemble"] = dict(ecfg.get("assignment_ensemble", {}), n_members=members)
        sub = self.counts[self.counts[self.G].isin(set(dates))]
        return ContextEncoder(self.C, self.num_child, ecfg).fit(sub, seed=self.seed)

    def encode_training(self, df, dates):
        """Inner out-of-fold encoding for training rows (target-encoding hygiene)."""
        k = int(self.cfg.get("context_encoder", {}).get("inner_oof_folds", 3))
        dates = np.array(sorted(set(dates)))
        feats = self.cfg.get("features", {})
        if k < 2 or dates.size < 2:
            LOG.warning("Inner OOF encoding disabled (folds=%d, groups=%d).", k, dates.size)
            return self.fit(dates).transform(df, feats)
        k = min(k, dates.size)
        inner_members = self.cfg.get("context_encoder", {}).get("inner_oof_ensemble_members")
        parts = []
        for f, held in enumerate(np.array_split(dates, k)):
            enc = self.fit(sorted(set(dates) - set(held)), members=inner_members)
            rows = df[df[self.G].isin(set(held))]
            parts.append(enc.transform(rows, feats))
        return pd.concat(parts).loc[df.index]


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def group_column(table, cfg):
    """split.group_column; default split_group (matchup generator) when present."""
    g = cfg.get("split", {}).get("group_column")
    if g:
        return g
    return "split_group" if "split_group" in table.columns and table["split_group"].astype(str).ne("").all() else "date"


def split_holdout(df, cfg, G="date"):
    """
    Returns (development rows, holdout rows, calibration groups).

    split.mode
      matchup         classifier_split from the matchup generator
                      (holdout -> holdout, calibration -> calibration groups)
      explicit_dates  split.holdout_dates (classifier_utils.normalize_holdout_dates)
      other           classifier_utils.holdout_split (split.enabled + strategy)
    calibration.calibration_dates, if given, adds calibration groups by date.
    Groups (G) are never split across partitions.
    """
    sc = cfg.get("split", {})
    mode = str(sc.get("mode", "")).lower()
    cal_list = cfg.get("calibration", {}).get("calibration_dates") or []
    cal_dates = normalize_holdout_dates(cal_list) if cal_list else set()

    if mode == "matchup":
        if "classifier_split" not in df.columns or df["classifier_split"].eq("").all():
            raise ValueError("split.mode=matchup needs classifier_split from the matchup generator.")
        ho = df[df["classifier_split"] == "holdout"].copy()
        dev = df[df["classifier_split"].isin(["dev", "calibration"])].copy()
        cal = set(df.loc[df["classifier_split"] == "calibration", G])
    elif mode == "explicit_dates":
        hold = normalize_holdout_dates(sc.get("holdout_dates"))
        missing = hold - set(df["date"])
        if missing:
            raise ValueError(f"Holdout dates not in table: {sorted(missing)}")
        hg = set(df.loc[df["date"].isin(hold), G])
        dev, ho = df[~df[G].isin(hg)].copy(), df[df[G].isin(hg)].copy()
        cal = set()
    else:
        c2 = copy.deepcopy(cfg)
        c2.setdefault("split", {})
        c2["split"].setdefault("date_column", "date")
        c2["split"].setdefault("label_column", "ref_label")
        c2["split"]["group_column"] = G
        dev, ho = holdout_split(df, c2)
        cal = set()

    if cal_dates:
        cal |= set(dev.loc[dev["date"].isin(cal_dates), G])
        if cal_dates - set(dev["date"]):
            raise ValueError(f"Calibration dates not in development table: {sorted(cal_dates - set(dev['date']))}")
    if not ho.empty and set(dev[G]) & set(ho[G]):
        raise ValueError(f"Groups ({G}) in both development and holdout.")
    return dev, ho, cal


def grouped_folds(df, cfg, G="date"):
    """classifier_utils.build_cv_splitter, required to return grouped CV on G."""
    n_groups = df[G].nunique()
    if n_groups < 2:
        raise ValueError(f"Grouped CV needs at least two development groups ({G}).")
    c2 = copy.deepcopy(cfg)
    cv = c2.setdefault("cross_validation", {})
    n_splits = int(cv.get("n_splits", 5))
    if n_groups < n_splits:
        LOG.warning("Reducing CV folds from %d to %d (number of groups).", n_splits, n_groups)
    cv["n_splits"] = min(n_splits, n_groups)
    cv["grouped"] = True
    splitter, kind = build_cv_splitter(df[G], c2)
    if kind != "group":
        raise RuntimeError("Grouped CV is required; got a non-grouped splitter.")
    return list(splitter.split(df, groups=df[G])), cv["n_splits"]


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def calibration_targets(frame, enc, cfg, C):
    mode = str(cfg.get("calibration", {}).get("target", "training_target")).lower()
    if mode == "reference":
        return np.eye(C)[frame["ref_label"].to_numpy().astype(int)]
    return build_targets(frame, enc, cfg["target"])


def fit_calibrator(name, P, Y, w, pi_train, cfg, noise_T=None):
    method = cfg.get("calibration", {}).get("method", "temperature")
    pi_true = None
    if name in BASELINES or not cfg.get("calibration", {}).get("prior_correction", True):
        pi_train = None
    else:
        pi_true = (w[:, None] * Y).sum(axis=0) / w.sum()
    return ProbabilityCalibrator(method, pi_train, pi_true, noise_T=noise_T).fit(P, Y, w)


def level_weights(levels, frame):
    """Training weights by negative level (target.label_levels.training_weights), or None."""
    if not levels or "label_level" not in frame.columns:
        return None
    return levels.training_weights(frame["ref_label"].to_numpy(), frame["label_level"].to_numpy())


def calibration_rows(frame, levels, level):
    """Boolean mask of calibration rows at calibration.label_level (option A)."""
    ref = frame["ref_label"].to_numpy()
    if str(level).lower() == "primary":
        return ref >= 0
    if not levels or "label_level" not in frame.columns:
        raise ValueError("calibration.label_level: consensus needs target.label_levels.consensus_references "
                         "and a feature table built with them (label_level column); rebuild the table.")
    return levels.level_mask(ref, frame["label_level"].to_numpy(), level)


def uncertainty_maps(P, frame, members=None):
    u = {"predictive_entropy": predictive_entropy_norm(P),
         "margin_uncertainty": margin_uncertainty(P),
         "assignment_vote_variance": frame["assign_vote_var"].to_numpy(),
         "parent_entropy": frame["parent_entropy"].to_numpy(),
         "child_entropy": frame["child_entropy"].to_numpy()}
    if members is not None:
        _, _, _, mi, var = ensemble_decomposition(members)
        u["ensemble_mutual_information"] = mi
        u["ensemble_variance"] = var
    return u


def summarize(P, frame, T, C, threshold, members=None):
    return probability_metrics(P, frame["ref_label"].to_numpy().astype(int), T,
                               frame["pop_weight"].to_numpy(), C, threshold,
                               uncertainty_maps(P, frame, members))


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def selective_rules(name, Pc, ref, w, frame, thr, calibrator, final_models, boot_paths,
                    feats, C, cal_mode, tdir, sel_cfg):
    """
    Uncertainty thresholds for the selective discrete product, chosen on the
    calibration rows (calibration groups, or grouped OOF rows under
    cross_fit). Ensemble measures need held-out member predictions, so they
    are only available with calibration groups.
    """
    err = (decide(Pc, thr) != ref).astype(float)
    u = {"predictive_entropy": predictive_entropy_norm(Pc),
         "margin_uncertainty": margin_uncertainty(Pc)}
    if "assign_vote_var" in frame.columns:
        u["assignment_vote_variance"] = frame["assign_vote_var"].to_numpy()
    if cal_mode == "calibration_dates" and name in final_models:
        members = None
        if name == "random_forest":
            members = forest_member_probabilities(final_models[name], frame[feats], C)
        elif name in boot_paths:
            members = bootstrap_member_probabilities(tdir / f"{name}.joblib", frame[feats], boot_paths[name], C)
        if members is not None:
            _, tot, exp_, mi, var = ensemble_decomposition(calibrator.transform_members(members))
            u.update(ensemble_total_entropy=tot, ensemble_expected_entropy=exp_,
                     ensemble_mutual_information=mi, ensemble_variance=var)
    want = sel_cfg.get("measures") or list(u)
    rules = {m: select_selective_threshold(u[m], err, w, sel_cfg.get("target_error", 0.05),
                                           sel_cfg.get("target_coverage"))
             for m in want if m in u}
    primary = sel_cfg.get("measure", "predictive_entropy")
    if primary not in rules:
        LOG.warning("%s: selective measure %s unavailable; using predictive_entropy.", name, primary)
        primary = "predictive_entropy"
        rules.setdefault(primary, select_selective_threshold(u[primary], err, w,
                                                             sel_cfg.get("target_error", 0.05),
                                                             sel_cfg.get("target_coverage")))
    LOG.info("%s selective (%s): threshold=%.4f coverage=%.3f retained error=%s", name, primary,
             rules[primary]["threshold"], rules[primary]["coverage"], rules[primary]["retained_error"])
    return {"measure": primary, "target_error": sel_cfg.get("target_error", 0.05),
            "target_coverage": sel_cfg.get("target_coverage"), "rules": rules}


def run(cfg):
    out = Path(cfg["output_dir"])
    tdir = out / "training"
    tdir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(out / "build_manifest.json")
    table = read_table(manifest["pixel_features"])
    C_cfg = len(cfg["target"]["classes"])
    have = set(int(v) for v in table["ref_label"].unique())
    if any(c not in have for c in range(C_cfg)):
        raise ValueError(
            f"Feature table {manifest['pixel_features']} has reference classes {sorted(have)} but the "
            f"config defines {C_cfg}. It may be stale from an earlier run; rerun build_uq_feature_table.py "
            "and fix the error it reports before training.")
    counts = read_table(manifest["cluster_counts"])
    table["date"] = table["date"].astype(str)
    counts["date"] = counts["date"].astype(str)
    G = group_column(table, cfg)
    for df_ in (table, counts):
        if G not in df_.columns:
            raise KeyError(f"Group column {G!r} missing; rebuild the feature table.")
        df_[G] = df_[G].astype(str)
    LOG.info("Grouping unit for CV, encoders, bootstrap and splits: %s", G)

    C = len(cfg["target"]["classes"])
    num_child = int(manifest["head_settings"]["num_child_classes"])
    factory = EncoderFactory(counts, C, num_child, cfg, group_col=G)
    models = build_models(cfg)
    baselines = [b for b in cfg.get("models", []) if b in BASELINES]
    names = list(models) + baselines

    levels = LabelLevels(cfg)
    if levels and "label_level" not in table.columns:
        raise ValueError("target.label_levels.consensus_references is set but the feature table has no "
                         "label_level column; rerun build_uq_feature_table.py.")
    if levels and level_weights(levels, table) is not None:
        LOG.info("Training weights by label level: %s", levels.describe()["training_weights"])
    dev, hold, cal_groups = split_holdout(table, cfg, G)
    cal_mode = str(cfg.get("calibration", {}).get("mode", "cross_fit")).lower()
    if cal_mode == "calibration_groups":
        cal_mode = "calibration_dates"
    if cal_mode == "calibration_dates" and not cal_groups:
        mode = str(cfg.get("split", {}).get("mode", "")).lower() or "(not set)"
        in_table = (table["classifier_split"].replace("", "unassigned").value_counts().to_dict()
                    if "classifier_split" in table.columns else "column missing")
        hint = []
        if mode != "matchup" and not cfg.get("calibration", {}).get("calibration_dates"):
            hint.append("split.mode is not 'matchup', so matchup calibration labels are ignored; "
                        "set split.mode: matchup or list calibration.calibration_dates")
        elif mode == "matchup":
            sel = cfg.get("data", {}).get("include_splits")
            if sel and "calibration" not in sel:
                hint.append(f"data.include_splits={sel} drops calibration scenes from the feature table")
            hint.append("no feature-table rows have classifier_split=calibration: check the generator log "
                        "for 'No eligible groups were assigned to calibration' (calibration_fraction, "
                        "buffer_days, encoder_train_policy, strata with < 3 groups, excluded-cluster drops), "
                        "that data.matchup_file is the split-annotated file, and the build_uq_feature_table "
                        "log for skipped calibration scenes")
        raise ValueError("calibration.mode=calibration_groups needs calibration groups. "
                         f"split.mode={mode}; classifier_split rows in feature table: {in_table}. "
                         + " ".join(hint) + ". Or use calibration.mode: cross_fit.")
    if cal_mode != "calibration_dates" and cal_groups:
        LOG.warning("calibration.mode=cross_fit: %d calibration groups are used as development data.",
                    len(cal_groups))
    dev_fit = dev[~dev[G].isin(cal_groups)] if cal_mode == "calibration_dates" else dev
    dev_cal = dev[dev[G].isin(cal_groups)] if cal_mode == "calibration_dates" else None
    LOG.info("Rows: dev=%d (fit=%d), holdout=%d; groups dev=%d calibration=%d holdout=%d",
             len(dev), len(dev_fit), len(hold), dev[G].nunique(),
             len(cal_groups) if cal_mode == "calibration_dates" else 0, hold[G].nunique() if len(hold) else 0)

    # ---------------- grouped CV with fold-refit encoders ----------------
    folds, n_splits = grouped_folds(dev_fit, cfg, G)
    feats = None
    oof = {n: np.zeros((len(dev_fit), C)) for n in names}
    oof_T = np.zeros((len(dev_fit), C))
    oof_frames = []

    for k, (tr, va) in enumerate(folds, 1):
        df_tr, df_va = dev_fit.iloc[tr], dev_fit.iloc[va]
        dates_tr = sorted(df_tr[G].unique())
        LOG.info("Fold %d/%d: %d train groups, %d validation groups", k, n_splits,
                 len(dates_tr), df_va[G].nunique())
        enc = factory.fit(dates_tr)
        X_tr_frame = factory.encode_training(df_tr, dates_tr)
        T_tr = build_targets(enc.transform(df_tr, cfg.get("features", {})), enc, cfg["target"])
        X_va_frame = enc.transform(df_va, cfg.get("features", {}))
        if feats is None:
            feats = select_feature_columns(X_va_frame, cfg.get("features", {}))
            LOG.info("Using %d features", len(feats))
        oof_T[va] = calibration_targets(X_va_frame, enc, cfg, C)
        oof_frames.append(X_va_frame.assign(_pos=va))
        for name, model in models.items():
            m = fit_soft(clone(model), X_tr_frame[feats], T_tr, cfg, base_weight=level_weights(levels, X_tr_frame))
            oof[name][va] = full_proba(m, X_va_frame[feats], C)
        for b in baselines:
            oof[b][va] = baseline_proba(b, X_va_frame, enc)

    oof_frame = pd.concat(oof_frames).sort_values("_pos").drop(columns="_pos")
    w_oof = oof_frame["pop_weight"].to_numpy()

    # ---------------- final encoder + models ----------------
    dates_fit = sorted(dev_fit[G].unique())
    enc_final = factory.fit(dates_fit)
    X_fit_frame = factory.encode_training(dev_fit, dates_fit)
    T_fit = build_targets(enc_final.transform(dev_fit, cfg.get("features", {})), enc_final, cfg["target"])
    final_models, train_priors = {}, {}
    for n, m in models.items():
        final_models[n], train_priors[n] = fit_soft(clone(m), X_fit_frame[feats], T_fit, cfg, return_prior=True,
                                                    base_weight=level_weights(levels, X_fit_frame))
        joblib.dump(final_models[n], tdir / f"{n}.joblib")

    # ---------------- bootstrap ensembles (grouped by date) ----------------
    bs = cfg.get("bootstrap_ensemble", {})
    boot_paths = {}
    if bs.get("enabled", False):
        rng = np.random.default_rng(int(cfg.get("random_state", 42)))
        dates_arr = np.array(dates_fit)
        for name in bs.get("models", ["logistic_regression"]):
            if name not in models:
                continue
            paths = []
            for b in range(int(bs.get("n_models", 10))):
                draw = rng.choice(dates_arr, size=dates_arr.size, replace=True)
                idx = np.concatenate([np.where(X_fit_frame[G].to_numpy() == d)[0] for d in draw])
                mb = fit_soft(clone(models[name]), X_fit_frame.iloc[idx][feats], T_fit[idx], cfg,
                              base_weight=level_weights(levels, X_fit_frame.iloc[idx]))
                p = tdir / f"{name}_bootstrap_{b:03d}.joblib"
                joblib.dump(mb, p)
                paths.append(str(p))
            boot_paths[name] = paths

    # ---------------- calibration ----------------
    calibrators, thresholds, metrics = {}, {}, {}
    if cal_mode == "calibration_dates":
        X_cal_frame = enc_final.transform(dev_cal, cfg.get("features", {}))
        Y_cal = calibration_targets(X_cal_frame, enc_final, cfg, C)
        w_cal = X_cal_frame["pop_weight"].to_numpy()
        P_cal = {n: full_proba(final_models[n], X_cal_frame[feats], C) for n in models}
        P_cal.update({b: baseline_proba(b, X_cal_frame, enc_final) for b in baselines})
        cal_frame = X_cal_frame
    else:
        Y_cal, w_cal, P_cal, cal_frame = oof_T, w_oof, oof, oof_frame

    # option C: label-noise transition matrix, estimated on all calibration rows
    ccfg = cfg.get("calibration", {}) or {}
    nc_cfg = ccfg.get("noise_correction") or {}
    noise_T, noise_info = None, None
    if nc_cfg.get("enabled", False):
        if str(ccfg.get("target", "training_target")).lower() != "reference":
            raise ValueError("calibration.noise_correction needs calibration.target: reference (the noise "
                             "model describes the primary reference labels, not soft training targets).")
        cons = cal_frame["cons_ref_label"].to_numpy() if "cons_ref_label" in cal_frame.columns else None
        noise_T, noise_info = resolve_transition(nc_cfg, cal_frame["ref_label"].to_numpy(), cons, w_cal, C)
        LOG.info("Noise correction: transition P(primary | true) = %s (%s)",
                 np.round(noise_T, 3).tolist(), noise_info["source"])

    # option A: calibration label level
    cal_level = str(ccfg.get("label_level", "primary")).lower()
    keep = calibration_rows(cal_frame, levels, cal_level)
    if noise_T is not None and cal_level != "primary":
        LOG.warning("noise_correction with label_level=%s: the transition matrix describes primary labels "
                    "but calibration uses consensus-filtered rows.", cal_level)
    if not keep.all():
        n_before = len(keep)
        cal_frame = cal_frame[keep]
        Y_cal, w_cal = Y_cal[keep], w_cal[keep]
        P_cal = {n: P[keep] for n, P in P_cal.items()}
        LOG.info("Calibration label level %s: %d of %d calibration rows (negatives limited to consensus)",
                 cal_level, int(keep.sum()), n_before)
    present = set(cal_frame["ref_label"].astype(int).unique())
    if len(present) < C:
        raise ValueError(f"Calibration rows at label_level={cal_level} contain classes {sorted(present)} only; "
                         "use label_level: primary or add calibration groups with consensus coverage.")

    thr_metric = ccfg.get("threshold_metric", "iou")
    # Per-model override, e.g. {hard_assignment: iou}: discrete-score baselines
    # cannot match prevalence (few distinct score levels).
    thr_overrides = {str(k): str(v) for k, v in (ccfg.get("threshold_metric_overrides") or {}).items()}
    unknown = sorted(set(thr_overrides) - set(names))
    if unknown:
        LOG.warning("threshold_metric_overrides: no such model(s) %s (available: %s)", unknown, list(names))
    threshold_metrics = {}
    thr_grid = threshold_grid(ccfg.get("threshold_grid"))
    LOG.info("Threshold grid: %d values in [%.2g, %.2g]", thr_grid.size, thr_grid.min(), thr_grid.max())
    sel_cfg = cfg.get("selective", {}) or {}
    selective = {}
    for n in names:
        calibrators[n] = fit_calibrator(n, P_cal[n], Y_cal, w_cal, train_priors.get(n), cfg, noise_T)
        Pc = calibrators[n].transform(P_cal[n])
        ref_cal = cal_frame["ref_label"].to_numpy()
        m_n = threshold_metrics[n] = thr_overrides.get(n, thr_metric)
        if C == 2:
            thresholds[n], _ = select_threshold(Pc[:, 1], ref_cal, w_cal, m_n, grid=thr_grid)
        else:
            thresholds[n], _ = select_threshold_multiclass(Pc, ref_cal, w_cal, m_n, grid=thr_grid)
        LOG.info("Threshold %s: %.4g (%s)", n, thresholds[n], m_n)
        if thresholds[n] <= thr_grid.min() * (1 + 1e-9):
            LOG.warning("Threshold %s is at the grid floor (%.2g): the optimum may lie lower. Set "
                        "calibration.threshold_grid: {scale: log, min: <smaller>} (e.g. prevalence / 10).",
                        n, thr_grid.min())
        if sel_cfg.get("enabled", True):
            selective[n] = selective_rules(n, Pc, ref_cal, w_cal, cal_frame, thresholds.get(n),
                                           calibrators[n], final_models, boot_paths, feats, C,
                                           cal_mode, tdir, sel_cfg)
        metrics[n] = {
            "cv_type": f"GroupKFold(n_splits={n_splits}) by {G}",
            "selective_calibration": selective.get(n),
            "calibration_mode": cal_mode,
            "calibration_label_level": cal_level,
            "calibrator": calibrators[n].describe(),
            "threshold": thresholds.get(n),
            "threshold_grid": {"n": int(thr_grid.size), "min": float(thr_grid.min()), "max": float(thr_grid.max())},
            "oof_uncalibrated": summarize(oof[n], oof_frame, oof_T, C, 0.5),
            "oof_calibrated": summarize(calibrators[n].transform(oof[n]), oof_frame, oof_T, C, thresholds.get(n)),
        }

    # ---------------- holdout ----------------
    holdout_preds = None
    if not hold.empty:
        X_ho = enc_final.transform(hold, cfg.get("features", {}))
        T_ho = calibration_targets(X_ho, enc_final, cfg, C)
        holdout_preds = X_ho[["pair_id", "date", "row", "col", "ref_label", "pop_weight"]
                         + [c for c in ("label_level", "cons_ref_label") if c in X_ho.columns]].copy()
        for n in names:
            members = None
            if n in models:
                P = full_proba(final_models[n], X_ho[feats], C)
                if n == "random_forest":
                    members = calibrators[n].transform_members(
                        forest_member_probabilities(final_models[n], X_ho[feats], C))
                elif n in boot_paths:
                    members = calibrators[n].transform_members(bootstrap_member_probabilities(
                        tdir / f"{n}.joblib", X_ho[feats], boot_paths[n], C))
            else:
                P = baseline_proba(n, X_ho, enc_final)
            Pc = calibrators[n].transform(P)
            metrics[n]["holdout_calibrated"] = summarize(Pc, X_ho, T_ho, C, thresholds.get(n), members)
            if levels:
                km = calibration_rows(X_ho, levels, "consensus")
                if km.any() and len(set(X_ho.loc[km, "ref_label"].astype(int))) == C:
                    metrics[n]["holdout_calibrated_consensus_level"] = summarize(
                        Pc[km], X_ho[km], T_ho[km], C, thresholds.get(n),
                        None if members is None else members[:, km])
            metrics[n]["holdout_by_date"] = {
                d: summarize(Pc[(X_ho["date"] == d).to_numpy()], g, T_ho[(X_ho["date"] == d).to_numpy()],
                             C, thresholds.get(n))["hard_vs_reference"]
                for d, g in X_ho.groupby("date")}
            for col in ("split_stratum", "encoder_split"):
                if col in X_ho.columns and X_ho[col].astype(str).ne("").any():
                    metrics[n][f"holdout_by_{col}"] = {
                        str(v): summarize(Pc[(X_ho[col] == v).to_numpy()], g, T_ho[(X_ho[col] == v).to_numpy()],
                                          C, thresholds.get(n))["hard_vs_reference"]
                        for v, g in X_ho.groupby(col)}
            for c in range(C):
                holdout_preds[f"{n}_p{c}"] = Pc[:, c].astype(np.float32)

    # ---------------- save ----------------
    joblib.dump(enc_final, tdir / "context_encoder.joblib")
    joblib.dump(calibrators, tdir / "calibrators.joblib")
    oof_out = oof_frame[["pair_id", "date", "row", "col", "ref_label", "pop_weight"]
                         + [c for c in ("label_level", "cons_ref_label") if c in oof_frame.columns]].copy()
    for n in names:
        for c in range(C):
            oof_out[f"{n}_p{c}"] = oof[n][:, c].astype(np.float32)
        for c in range(C):
            oof_out[f"soft_target_p{c}"] = oof_T[:, c].astype(np.float32)
    write_table(oof_out, tdir / "oof_predictions.parquet")
    if holdout_preds is not None:
        write_table(holdout_preds, tdir / "holdout_predictions.parquet")

    save_json(tdir / "training_summary.json", {
        "metadata": {
            "feature_columns": feats,
            "feature_signature": uq_feature_signature(cfg),
            "head_settings": manifest["head_settings"],
            "classes": cfg["target"]["classes"],
            "target": cfg["target"],
            "models": list(models),
            "baselines": baselines,
            "model_paths": {n: str(tdir / f"{n}.joblib") for n in models},
            "context_encoder_path": str(tdir / "context_encoder.joblib"),
            "calibrators_path": str(tdir / "calibrators.joblib"),
            "bootstrap_manifest": boot_paths,
            "thresholds": thresholds,
            "threshold_metrics": threshold_metrics,
            "label_levels": levels.describe() if levels else None,
            "calibration_label_level": cal_level,
            "noise_correction": noise_info,
            "selective": selective,
            "effective_training_priors": train_priors,
            "development_dates": sorted(dev["date"].unique()),
            "group_column": G,
            "calibration_groups": sorted(cal_groups),
            "calibration_dates": sorted(dev_cal["date"].unique()) if dev_cal is not None else [],
            "development_groups": sorted(dev[G].unique()),
            "holdout_groups": sorted(hold[G].unique()) if len(hold) else [],
            "holdout_dates": sorted(hold["date"].unique()),
            "n_dev_rows": int(len(dev)),
            "n_holdout_rows": int(len(hold)),
        },
        "metrics": metrics,
    })
    LOG.info("Training complete. Outputs in %s", tdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_config", type=Path)
    run(load_yaml(ap.parse_args().yaml_config))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
