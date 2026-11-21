"""
Train calibrated probabilistic HAB concentration-bin models on SIT-FUSE outputs.

Per species (pDA, PN seriata, PN delicatissima, total phytoplankton) and per
sensor (or pooled across sensors), reusing sit_fuse.train.sf_uq_utils:

  1. matchups      point-pixel rows + encoder counts (hab_uq_matchups)
  2. encoder       HABContextEncoder (production histogram-mode rule, smoothed
                   agreement tables, bootstrap assignment ensemble), cross-fit
                   over split groups so classifier rows never see their own
                   samples in the cluster->bin tables. Optional production
                   dual-level matchup (hab_spatial_tier): counts pooled over a
                   sensor group's platforms (context_encoder.pool_sensor_group)
                   and a tier-2 spatial table per platform and fold
                   (context_encoder.spatial_tier)
  3. targets       build_targets modes, with the one-hot reference optionally
                   replaced by measurement-uncertainty soft labels
  4. classifier    classifier_utils.build_models (+ expand_soft_targets)
  5. members       forest trees or group-bootstrap refits (epistemic spread)
  6. calibration   ProbabilityCalibrator (prior correction + temperature, optional
                   label-noise transition) on sample-level linear-pooled
                   probabilities, per sensor with pooled fallback; raw vs
                   calibrated ECE/RPS/NLL stored in bundle["skill"][s]["calibration"].
                   Calibration/test rows can be restricted to encoder-unseen
                   dates (uq.splits.encoder_unseen_eval)
  7. selective     select_selective_threshold on calibration samples
  8. skill         per-sensor proper scores on calibration samples (used as
                   fusion weights in hab_multisensor_fusion)
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sit_fuse.train.classifier_utils import build_models
from sit_fuse.train.sf_uq_utils import (
    EPS, InvalidClusters, ProbabilityCalibrator, effective_training_prior,
    expand_soft_targets, forest_member_probabilities, full_proba,
    predictive_entropy_norm, select_feature_columns, select_selective_threshold)
from sit_fuse.pipelines.habs.uq.hab_context_encoder import default_encoder_cfg, fit_encoder
from sit_fuse.pipelines.habs.uq.hab_uq_selection import (
    decide, decision_spec, fit_exceedance_recalibration, fit_interval_calibration, selection_components,
    selection_loss, select_decision_rule, wrap_calibrators)
from sit_fuse.pipelines.habs.uq.hab_insitu import assign_splits, insitu_filter_config, load_insitu, species_frame
from sit_fuse.pipelines.habs.uq.hab_uq_common import (
    SENSORS, insitu_date_kwargs, ranked_probability_score, species_spec, instrument_dirs, utcnow_iso)
from sit_fuse.pipelines.habs.hab_sensor_groups import expand_sensors
from sit_fuse.pipelines.habs.uq.hab_uq_matchups import build_sensor_matchups, sample_level
from sit_fuse.pipelines.habs.uq.hab_spatial_tier import pool_group_counts

LOG = logging.getLogger("hab-uq-train")

# matchup bookkeeping that must never become a feature (target leakage / ids)
HAB_NON_FEATURES = {
    "value", "log10_value", "fold", "pix_weight", "n_match_pixels", "match_dist_km", "match_fallback",
    "day_offset",
    "hours_offset", "ref_label", "weight", "time_block",
    "abs_day_offset",   # added below only when features.use_day_offset is true
    "assign_tier",      # coverage diagnostic (joint / parent / spatial / none)
}
HAB_NON_FEATURE_PREFIXES = ("ref_soft_", "sensor_")


def hab_feature_columns(df: pd.DataFrame, features_cfg: dict[str, Any],
                        sensors: list[str] | None = None) -> list[str]:
    groups = features_cfg.get("groups")
    spatial_on = groups is not None and "spatial_tier" in groups
    cols = [c for c in select_feature_columns(df, features_cfg)
            if c not in HAB_NON_FEATURES and not c.startswith(HAB_NON_FEATURE_PREFIXES)
            and (spatial_on or not c.startswith("sp_tier_"))]
    if spatial_on:       # spatial_tier group: sp_tier_c, sp_tier_mass (encoder tier 2)
        cols += sorted(c for c in df.columns if c.startswith("sp_tier_") and c not in cols)
    if sensors and len(sensors) > 1:
        cols += [f"sensor_is_{s}" for s in sensors]
    if features_cfg.get("use_day_offset", False):
        cols.append("abs_day_offset")
    return list(dict.fromkeys(cols))     # unique, order kept (sklearn rejects duplicate names)


def add_sensor_columns(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    out = df.copy()
    for s in sensors:
        out[f"sensor_is_{s}"] = (out["sensor"] == s).astype(np.float32)
    out["abs_day_offset"] = out.get("day_offset", 0)
    out["abs_day_offset"] = np.abs(out["abs_day_offset"]).astype(np.float32)
    return out


def hab_targets(df: pd.DataFrame, encoder, target_cfg: dict[str, Any]) -> np.ndarray:
    """
    target.mode as sf_uq_utils.build_targets; target.reference:
      onehot       in-situ bin
      measurement  P(bin | measured value, sigma_dex)   (default)
    """
    C = encoder.C
    mode = str(target_cfg.get("mode", "blended_ensemble")).lower()
    lam = float(target_cfg.get("blend_lambda", 0.3))
    ref = df["ref_label"].to_numpy().astype(int)
    if str(target_cfg.get("reference", "measurement")).lower() == "measurement" and \
            f"ref_soft_{C - 1}" in df.columns:
        R = df[[f"ref_soft_{c}" for c in range(C)]].to_numpy(dtype=np.float64)
    else:
        R = np.eye(C)[np.clip(ref, 0, C - 1)]
    if mode == "reference":
        T = R
    elif mode == "cluster_agreement":
        T = encoder.soft_matrix(df, "sa_joint")
    elif mode == "assignment_ensemble":
        T = encoder.soft_matrix(df, "assign_vote_soft")
    elif mode == "blended":
        T = (1 - lam) * R + lam * encoder.soft_matrix(df, "sa_joint")
    elif mode == "blended_ensemble":
        T = (1 - lam) * R + lam * encoder.soft_matrix(df, "assign_vote_soft")
    else:
        raise ValueError(f"Unknown target.mode: {mode}")
    T = np.clip(T, 0.0, 1.0)
    return T / np.maximum(T.sum(axis=1, keepdims=True), EPS)


# ---------------------------------------------------------------------------
# Encoder cross-fitting
# ---------------------------------------------------------------------------

def cross_fit_encode(rows: pd.DataFrame, counts: pd.DataFrame, n_classes: int, num_child: int,
                     enc_cfg: dict[str, Any], features_cfg: dict[str, Any], seed: int = 42,
                     spatial_for=None):
    """
    Train rows: features from an encoder fit on train counts of the other
    folds. Calibration/test rows: features from the encoder fit on all train
    counts (the encoder used at inference time).
    spatial_for(fold or None) -> SpatialTable | None: tier-2 table whose source
    parent assignments exclude that fold (None = all train counts).
    """
    sp = spatial_for or (lambda k: None)
    train_counts = counts[counts["split"] == "train"]
    final = fit_encoder(train_counts, n_classes, num_child, enc_cfg, seed, spatial=sp(None))
    parts = []
    tr = rows[rows["split"] == "train"]
    for k in sorted(tr["fold"].unique()):
        sub = tr[tr["fold"] == k]
        c_k = train_counts[train_counts["fold"] != k]
        if c_k.empty:
            parts.append(final.transform(sub, features_cfg))
            continue
        enc_k = fit_encoder(c_k, n_classes, num_child, enc_cfg, seed + int(k) + 1, spatial=sp(k))
        parts.append(enc_k.transform(sub, features_cfg))
    other = rows[rows["split"] != "train"]
    if len(other):
        parts.append(final.transform(other, features_cfg))
    return pd.concat(parts).loc[rows.index], final


# ---------------------------------------------------------------------------
# Classifier + members
# ---------------------------------------------------------------------------

def class_weight_vector(ye, we, n_classes: int, cfg: dict[str, Any] | None) -> np.ndarray | None:
    """
    classifier.class_weighting: {mode: inverse | sqrt_inverse | none, max_weight: 10}

    Weights from the *weighted* class mass of the soft-expanded rows (sum of
    soft-target x sample weight per bin), unlike sklearn's class_weight
    'balanced', which counts expanded rows and is ~uniform with measurement
    soft labels. w_c = (majority mass / mass_c) ** p (p = 1 inverse, 0.5 sqrt),
    capped at max_weight (ratio to the majority bin), then rescaled so the total
    training weight is unchanged.
    """
    cfg = cfg or {}
    mode = str(cfg.get("mode", "none")).lower()
    if mode in ("none", "off", "false", ""):
        return None
    pw = {"inverse": 1.0, "sqrt_inverse": 0.5}.get(mode)
    if pw is None:
        pw = float(cfg.get("power", 1.0)) if mode == "power" else None
    if pw is None:
        raise ValueError(f"Unknown classifier.class_weighting.mode: {mode}")
    mass = np.bincount(np.asarray(ye, int), weights=np.asarray(we, float), minlength=n_classes)
    present = mass > 0
    cw = np.ones(n_classes)
    cw[present] = (mass[present].max() / mass[present]) ** pw     # majority bin = 1
    cw = np.minimum(cw, float(cfg.get("max_weight", 10.0)))
    cw *= mass.sum() / max((mass * cw).sum(), EPS)
    return cw


def _fit_one(model_proto, X, T, w, seed=None, class_weighting=None):
    m = copy.deepcopy(model_proto)
    if seed is not None:
        try:
            m.set_params(clf__random_state=seed)
        except Exception:
            pass
    Xe, ye, we = expand_soft_targets(X, T, base_weight=w)
    cw = class_weight_vector(ye, we, T.shape[1], class_weighting)
    if cw is not None:
        we = we * cw[ye]              # pi_train (prior correction) sees the reweighted mass
    m.fit(Xe, ye, clf__sample_weight=we)
    return m, ye, we


class EncoderOnlyModel:
    """
    Probabilistic context assignment without a learned classifier:
    P = sum_k a_k * M_k over encoder soft matrices (assign_vote_soft, sa_joint,
    nbr_r*_vote ...), then calibrated like any other model. This is the direct
    probabilistic analogue of the production hard assignment and the most
    robust choice for small in-situ sets.
    """

    def __init__(self, n_classes: int, mix: dict[str, float] | None = None):
        self.C = int(n_classes)
        self.mix = mix or {"assign_vote_soft": 0.5, "sa_joint": 0.5}
        self.classes_ = np.arange(self.C)

    def required_columns(self):
        return [f"{k}_{c}" for k in self.mix for c in range(self.C)]

    def fit(self, *a, **k):
        return self

    def predict_proba(self, X):
        P = np.zeros((len(X), self.C))
        tot = 0.0
        for k, a in self.mix.items():
            cols = [f"{k}_{c}" for c in range(self.C)]
            if all(c in X.columns for c in cols):
                P += float(a) * X[cols].to_numpy(dtype=np.float64)
                tot += float(a)
        P = P / max(tot, EPS) + 1e-4
        return P / P.sum(1, keepdims=True)


def fit_classifier(X, T, w, groups, clf_cfg: dict[str, Any], n_classes: int, seed: int = 42):
    name = clf_cfg.get("model", "random_forest")
    pi_target = (T * (w / max(w.sum(), EPS))[:, None]).sum(0)
    pi_target = np.clip(pi_target, EPS, None) / np.clip(pi_target, EPS, None).sum()
    if name == "encoder_only":
        m = EncoderOnlyModel(n_classes, clf_cfg.get("encoder_mix"))
        return {"model": m, "members": [], "pi_train": None, "pi_target": pi_target, "name": name}
    proto = build_models({**clf_cfg, "models": [name]})[name]
    cwc = clf_cfg.get("class_weighting")
    model, ye, we = _fit_one(proto, X, T, w, class_weighting=cwc)
    pi_train = effective_training_prior(model, ye, we, n_classes)
    members = []
    B = int(clf_cfg.get("n_bootstrap_members", 0 if name == "random_forest" else 10))
    if B > 0:
        rng = np.random.default_rng(seed)
        ug = np.unique(groups)
        for b in range(B):
            draw = rng.choice(ug, size=ug.size, replace=True)
            mult = pd.Series(draw).value_counts()
            wb = w * pd.Series(groups).map(mult).fillna(0).to_numpy()
            keep = wb > 0
            if keep.sum() < 10:
                continue
            mb, _, _ = _fit_one(proto, X[keep], T[keep], wb[keep], seed=seed + b + 1,
                                class_weighting=cwc)
            members.append(mb)
    return {"model": model, "members": members, "pi_train": pi_train, "pi_target": pi_target,
            "name": name}


def member_probabilities(fit: dict[str, Any], X, n_classes: int) -> np.ndarray | None:
    if fit["members"]:
        return np.stack([full_proba(m, X, n_classes) for m in fit["members"]])
    clf = fit["model"].named_steps["clf"] if hasattr(fit["model"], "named_steps") else fit["model"]
    if hasattr(clf, "estimators_"):
        return forest_member_probabilities(fit["model"], X, n_classes)
    return None


# ---------------------------------------------------------------------------
# Skill summaries (fusion weights)
# ---------------------------------------------------------------------------

def skill_summary(P: np.ndarray, y: np.ndarray, w: np.ndarray, P_clim: np.ndarray) -> dict[str, float]:
    w = w / max(w.sum(), EPS)
    rps = ranked_probability_score(P, y)
    rps_c = ranked_probability_score(np.broadcast_to(P_clim, P.shape), y)
    nll = -np.log(np.clip(P[np.arange(len(y)), y], EPS, 1))
    return {
        "n": int(len(y)),
        "rps": float((w * rps).sum()),
        "rps_climatology": float((w * rps_c).sum()),
        "rpss": float(1 - (w * rps).sum() / max((w * rps_c).sum(), EPS)),
        "nll": float((w * nll).sum()),
        "accuracy": float((w * (P.argmax(1) == y)).sum()),
        "within_one_bin": float((w * (np.abs(P.argmax(1) - y) <= 1)).sum()),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def read_date_roles(path) -> dict[str, str]:
    """{YYYYMMDD: train|calibration|test|excluded} from a CSV with columns date, role."""
    t = pd.read_csv(path, dtype=str)
    col = "role" if "role" in t.columns else "split"
    d = pd.to_datetime(t["date"], format="mixed").dt.strftime("%Y%m%d")
    m = {"val": "calibration"}
    return {k: m.get(v, v) for k, v in zip(d, t[col].str.strip().str.lower())}


def split_config(hab_yml: dict[str, Any]) -> dict[str, Any]:
    """uq.splits, plus the encoder training window when splits.encoder_unseen_eval."""
    cfg = dict(hab_yml.get("uq", {}).get("splits") or {})
    if cfg.pop("encoder_unseen_eval", False) and not cfg.get("encoder_train_window"):
        if hab_yml.get("train_min_date") and hab_yml.get("train_max_date"):
            cfg["encoder_train_window"] = [hab_yml["train_min_date"], hab_yml["train_max_date"]]
        else:
            LOG.warning("splits.encoder_unseen_eval set but train_min_date/train_max_date missing")
    if not cfg.get("date_roles") and cfg.get("date_roles_file"):
        # explicit day roles (e.g. written by hab_uq_pilot_plan --mode resplit); overrides scene_split
        cfg["date_roles"] = read_date_roles(cfg.pop("date_roles_file"))
    if not cfg.get("date_roles"):
        from sit_fuse.pipelines.habs.hab_scene_split import uq_date_roles
        roles = uq_date_roles(hab_yml)
        if roles:
            cfg["date_roles"] = roles
    return cfg


def make_calibrator(cal_cfg: dict[str, Any], fit: dict[str, Any], C: int) -> ProbabilityCalibrator:
    """
    As the fire/smoke ProbabilityCalibrator setup: prior correction (undo class
    balancing toward the training-target prior), then temperature / platt /
    isotonic, optionally with a label-noise transition matrix
    (calibration.noise_correction: {enabled, transition: CxC}; T[k][j] =
    P(observed bin j | true bin k), e.g. from replicate or method-comparison
    samples). 'estimate' is not available for HAB (no consensus reference).
    """
    pc = cal_cfg.get("prior_correction", True) and fit["pi_train"] is not None
    nc = cal_cfg.get("noise_correction", {}) or {}
    noise_T = None
    if nc.get("enabled", False):
        T = nc.get("transition")
        if isinstance(T, str):
            raise ValueError("calibration.noise_correction.transition must be a CxC matrix for HAB")
        noise_T = np.asarray(T, dtype=float)
        if noise_T.shape != (C, C):
            raise ValueError(f"noise_correction.transition shape {noise_T.shape} != ({C}, {C})")
        margin = float(nc.get("min_diagonal_margin", 0.2))
        off = noise_T - np.diag(np.diag(noise_T))
        if np.any(np.diag(noise_T) - off.max(1) < margin):
            raise ValueError("noise_correction.transition is near-uninformative (diagonal margin)")
    return ProbabilityCalibrator(cal_cfg.get("method", "temperature"),
                                 pi_train=fit["pi_train"] if pc else None,
                                 pi_true=fit["pi_target"] if pc else None,
                                 noise_T=noise_T,
                                 prior_strength=float(cal_cfg.get("prior_correction_strength", 1.0)))


def calibration_diagnostics(P_raw, Q, y, w, C, n_bins=10) -> dict[str, float]:
    """Raw vs calibrated class-wise ECE / RPS / NLL on the calibration rows."""
    from sit_fuse.train.sf_uq_utils import reliability_table
    Y = np.eye(C)[y]
    wn = np.asarray(w, float)

    def block(P):
        eces = [reliability_table(P[:, c], Y[:, c], wn, n_bins=n_bins)[1] for c in range(C)]
        ww = wn / max(wn.sum(), EPS)
        return {"classwise_ece": float(np.nanmean(eces)),
                "rps": float((ww * ranked_probability_score(P, y)).sum()),
                "nll": float((ww * -np.log(np.clip(P[np.arange(len(y)), y], EPS, 1))).sum())}
    return {"raw": block(P_raw), "calibrated": block(Q)}


def train_species(hab_yml: dict[str, Any], use_key: str, out_dir: str | Path,
                  samples: pd.DataFrame | None = None, matchups: dict | None = None) -> dict[str, Any]:
    uq = hab_yml["uq"]
    spec = species_spec(use_key, uq.get("species", {}).get(use_key))
    C = spec.n_classes
    num_child = int(uq["num_child"])
    seed = int(uq.get("seed", 42))
    features_cfg = uq.get("features", {"window_radii": [1, 2], "neighborhood": {"top_n_labels": 3}})
    enc_cfg = default_encoder_cfg(uq.get("context_encoder"))
    clf_cfg = uq.get("classifier", {"model": "random_forest"})
    tgt_cfg = uq.get("target", {})
    cal_cfg = uq.get("calibration", {"method": "temperature", "prior_correction": True})
    # sensor groups (olci, viirs) expand to their platforms: UQ stays per platform
    # (overpass time, band set), reading each platform's scenes from the group out_dir
    sensors = expand_sensors(uq.get("sensors") or list(hab_yml["instruments"].keys()))
    sensors = [s for s in sensors if s in SENSORS]
    scope = uq.get("model_scope", "pooled")
    invalid = InvalidClusters.from_config(uq)

    if samples is None:
        ins = load_insitu(uq.get("insitu_file", hab_yml.get("insitu_file")),
                          max_depth_m=uq.get("max_depth_m", 1.0),
                          filters=insitu_filter_config(hab_yml), **insitu_date_kwargs(uq))
        if (uq.get("sample_weighting") or {}).get("enabled") and ins.attrs.get("has_method") is False:
            raise ValueError("uq.sample_weighting needs the non-strict in-situ table (method column); "
                             f"{uq.get('insitu_file', hab_yml.get('insitu_file'))} has none")
        samples = assign_splits(species_frame(ins, spec), split_config(hab_yml))
    n_ex = int((samples["split"] == "excluded").sum())
    if n_ex:
        # buffer / unlisted days are never used; skip them before raster matching
        LOG.info("%s: %d excluded samples skipped before matching", use_key, n_ex)
        samples = samples[samples["split"] != "excluded"].reset_index(drop=True)

    sw, sw_info = source_sample_weights(samples, uq.get("sample_weighting"))
    if sw is not None:
        LOG.info("%s sample_weighting: %s", use_key, sw_info)

    rows_all, counts_all = [], {}
    enc_by_sensor = {}
    built = {}
    for s in sensors:
        if matchups is not None and s in matchups:
            rows, counts = matchups[s]
        else:
            dirs = uq.get("product_dirs", {}).get(s) or instrument_dirs(hab_yml, s)
            rows, counts = build_sensor_matchups(samples, s, dirs, uq, features_cfg, invalid)
        if rows.empty or counts.empty or (counts["split"] == "train").sum() == 0:
            LOG.warning("%s/%s: no usable matchups", use_key, s)
            continue
        if sw is not None:
            rows, counts = apply_sample_weights(rows, counts, sw)
        built[s] = (rows, counts)
    enc_counts = {s: c for s, (_, c) in built.items()}
    if enc_cfg.get("pool_sensor_group", False):
        # production context assignment of a sensor group: one histogram over its platforms
        enc_counts = pool_group_counts(enc_counts)
    spatial_by_sensor = _spatial_providers(hab_yml, uq, enc_cfg, samples, built, enc_counts,
                                           C, num_child, invalid, seed)
    for s, (rows, counts) in built.items():
        enc_rows, enc = cross_fit_encode(rows, enc_counts[s], C, num_child, enc_cfg, features_cfg, seed,
                                         spatial_for=spatial_by_sensor.get(s))
        enc_by_sensor[s] = enc
        counts_all[s] = counts
        rows_all.append(enc_rows)
    del built
    if not rows_all:
        raise RuntimeError(f"No matchups for {use_key}")
    used = list(enc_by_sensor)
    df = add_sensor_columns(pd.concat(rows_all, ignore_index=True), used)
    feat_cols = hab_feature_columns(df, features_cfg, used if scope == "pooled" else None)
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0

    train_s = df[df["split"] == "train"].drop_duplicates("sample_uid")
    w_s = None if sw is None else train_s["sample_uid"].map(sw).fillna(1.0).to_numpy(dtype=float)
    pi_true = np.bincount(train_s["ref_label"].astype(int), weights=w_s, minlength=C).astype(float) + 1.0
    pi_true /= pi_true.sum()           # climatology (skill reference, fusion prior)

    model_name = clf_cfg.get("model", "auto")
    candidates = (clf_cfg.get("candidates", ["encoder_only", "hist_gradient_boosting", "random_forest"])
                  if model_name == "auto" else [model_name])
    sel_cfg = dict(clf_cfg.get("selection") or {"metric": "rps"})
    cw_grid = list(clf_cfg.get("class_weighting_grid") or [clf_cfg.get("class_weighting")])
    g_grid = prior_strength_grid(cal_cfg)
    n_fit = len(candidates) * len(cw_grid)
    B = int(clf_cfg.get("n_bootstrap_members", 0))
    # bootstrap members do not change the selection score (main-model probabilities): skip
    # them while searching and refit the selected configuration with members afterwards
    lean = n_fit > 1 and B > 0 and not sel_cfg.get("search_members", False)
    trials, best = {}, None
    for cand in candidates:
        for cw in cw_grid:
            cfg_c = {**clf_cfg, "model": cand, "class_weighting": cw}
            fits = _fit_scopes(df, used, scope, enc_by_sensor, feat_cols, tgt_cfg,
                               {**cfg_c, "n_bootstrap_members": 0} if lean else cfg_c, C, seed)
            # encoder_only has no class weights / training prior: prior strength is a no-op
            for g in (g_grid[:1] if cand == "encoder_only" else g_grid):
                cal_g = {**cal_cfg, "prior_correction_strength": g}
                res = _calibrate(df, fits, used, scope, feat_cols, C, cal_g, uq, pi_true, spec, sel_cfg)
                name = trial_name(cand, cw, g, cand == "encoder_only" or not cal_cfg.get("prior_correction", True))
                trials[name] = {"score": res["score"], "components": res["components"], "model": cand,
                                "class_weighting": cw, "prior_correction_strength": g}
                if best is None or res["score"] < best[0]:
                    best = (res["score"], name, cfg_c, cal_g, fits, res)
    _, sel_key, cfg_sel, cal_sel, fits, res = best
    if lean and cfg_sel["model"] != "encoder_only":
        fits = _fit_scopes(df, used, scope, enc_by_sensor, feat_cols, tgt_cfg, cfg_sel, C, seed)
        res = _calibrate(df, fits, used, scope, feat_cols, C, cal_sel, uq, pi_true, spec, sel_cfg)
    # per-species overrides (uq.species.<key>.decision / .exceedance_recalibration)
    sp_cfg = (uq.get("species") or {}).get(use_key) or {}
    ex_cfg = sp_cfg.get("exceedance_recalibration", cal_cfg.get("exceedance_recalibration"))
    exrecal = fit_exceedance_recalibration(res["Q"], res["y"], res["w"], spec, ex_cfg, groups=res.get("g"),
                                           seed=seed) if res["Q"] is not None else None
    ex_maps = (exrecal or {}).get("maps") or None
    if exrecal is not None:
        LOG.info("%s exceedance recalibration: enabled=%s maps=%s brier before %s / oof %s", use_key,
                 exrecal.get("enabled"), exrecal.get("maps"), exrecal.get("brier_before"), exrecal.get("brier_after_oof"))
    if ex_maps:
        res = _calibrate(df, fits, used, scope, feat_cols, C, cal_sel, uq, pi_true, spec, sel_cfg,
                         exceed_maps=ex_maps)
    dec_cfg = sp_cfg.get("decision", uq.get("decision"))
    decision = select_decision_rule(res["Q"], res["y"], res["w"], dec_cfg) \
        if res["Q"] is not None else {"rule": "argmax", "selected_by": "no calibration samples"}
    if decision["rule"] != "argmax":
        # selective thresholds use the decision rule's errors
        res = _calibrate(df, fits, used, scope, feat_cols, C, cal_sel, uq, pi_true, spec, sel_cfg,
                         decision_rule=decision_spec(decision), exceed_maps=ex_maps)
    calibrators, selective, skill = res["calibrators"], res["selective"], res["skill"]
    interval_cal = fit_interval_calibration(res["Q"], res["lv"], res["w"], spec, uq.get("intervals"))
    model_selection = {k: float(v["score"]) for k, v in trials.items()}
    missing = [s for s in used if calibrators.get(s) is None and calibrators.get("_pooled") is None]
    if missing and cal_cfg.get("require", True):
        raise RuntimeError(f"{use_key}: no calibrator for {missing} (no calibration matchups and no "
                           "pooled fallback); set calibration.require: false to allow raw probabilities")
    LOG.info("%s model selection (calibration %s): %s -> %s", use_key, sel_cfg.get("metric", "rps"),
             model_selection, sel_key)
    LOG.info("%s decision rule: %s; interval calibration: %s", use_key, decision,
             None if interval_cal is None else {k: interval_cal[k] for k in ("tail", "pad_dex", "calibration_coverage")})

    bundle = {
        "format": "sitfuse-hab-uq-v1", "created": utcnow_iso(), "species": use_key,
        "spec": spec.__dict__, "n_classes": C, "num_child": num_child, "sensors": used,
        "model_scope": scope, "encoders": enc_by_sensor, "fits": fits,
        "feature_columns": feat_cols, "calibrators": calibrators, "selective": selective,
        "skill": skill, "pi_true": pi_true, "uq_cfg": uq,
        "model_name": cfg_sel["model"], "model_trial": sel_key,
        "model_selection_calibration_rps": {k: v["components"]["rps"] for k, v in trials.items()},
        "model_selection": {"metric": sel_cfg.get("metric", "rps"), "weights": sel_cfg.get("weights"),
                            "scores": model_selection, "trials": trials, "selected": sel_key},
        "class_weighting": cfg_sel.get("class_weighting"),
        "prior_correction_strength": cal_sel.get("prior_correction_strength"),
        "decision": decision, "interval_calibration": interval_cal,
        "exceedance_recalibration": exrecal,
        # calibration-split components of the final probabilities (after exceedance recalibration)
        "calibration_components_final": res.get("components"),
        "split_summary": samples.groupby("split")["sample_uid"].nunique().to_dict(),
        "sample_weighting": sw_info,
    }
    out = Path(out_dir) / f"hab_uq_{spec.fname_token}.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    LOG.info("saved %s", out)
    bundle["path"] = str(out)
    bundle["_frame"] = df          # in-memory only (evaluation without rematching)
    return bundle


def _spatial_providers(hab_yml, uq, enc_cfg, samples, built, enc_counts, C, num_child, invalid, seed):
    """{sensor: spatial_for(fold)} for context_encoder.spatial_tier, else {}."""
    from sit_fuse.pipelines.habs.uq.hab_spatial_tier import (
        collect_cooccurrence, parent_assignment, spatial_days, spatial_table, spatial_tier_config)
    from sit_fuse.pipelines.habs.uq.hab_uq_common import date_windows
    cfg = spatial_tier_config(enc_cfg)
    if cfg is None or not built:
        return {}
    roles = split_config(hab_yml).get("date_roles")
    days = spatial_days(samples, roles, date_windows(uq, hab_yml), cfg, seed)
    dirs = {s: uq.get("product_dirs", {}).get(s) or instrument_dirs(hab_yml, s) for s in built}
    LOG.info("spatial tier: %d training days, mode %s, sources %s", len(days), cfg["mode"], cfg["sources"])
    cooc = collect_cooccurrence(list(built), dirs, days, uq, cfg, invalid, hab_yml.get("platform_priority"))
    rule = enc_cfg.get("assignment", {})
    cache = {}

    def assign(o, k):
        if (o, k) not in cache:
            c = enc_counts[o]
            c = c[c["split"] == "train"]
            if k is not None:
                c = c[c["fold"] != k]
            cache[(o, k)] = parent_assignment(c, C, rule)
        return cache[(o, k)]

    def provider(s):
        srcs = sorted(cooc[s]["src"].unique()) if len(cooc[s]) else []
        return lambda k: spatial_table(cooc[s], {o: assign(o, k) for o in srcs if o in enc_counts},
                                       C, num_child)
    return {s: provider(s) for s in built}


def _fit_scopes(df, used, scope, enc_by_sensor, feat_cols, tgt_cfg, clf_cfg, C, seed):
    scopes = {"pooled": used} if scope == "pooled" else {s: [s] for s in used}
    fits = {}
    for key, ss in scopes.items():
        parts = [df[(df["split"] == "train") & (df["sensor"] == s)] for s in ss]
        parts = [p for p in parts if len(p)]
        if not parts:
            continue
        tr = pd.concat(parts)
        T = np.vstack([hab_targets(p, enc_by_sensor[p["sensor"].iloc[0]], tgt_cfg) for p in parts])
        w = _sample_normalized_weights(tr)
        fits[key] = fit_classifier(tr[feat_cols], T, w, tr["split_group"].to_numpy(), clf_cfg, C, seed)
    return fits


def prior_strength_grid(cal_cfg: dict[str, Any]) -> list[float]:
    """calibration.prior_correction_strength: number, or 'fit' with prior_strength_grid."""
    g = cal_cfg.get("prior_correction_strength", 1.0)
    if isinstance(g, str):
        if g.lower() != "fit":
            raise ValueError(f"calibration.prior_correction_strength must be a number or 'fit', got {g!r}")
        return [float(x) for x in cal_cfg.get("prior_strength_grid", [0.0, 0.25, 0.5, 0.75, 1.0])]
    return [float(g)]


def trial_name(model: str, cw: dict | None, g: float, no_prior: bool = False) -> str:
    cw = cw or {}
    mode = str(cw.get("mode", "none"))
    tag = mode if mode in ("none", "off") else f"{mode}{cw.get('power', '') if mode == 'power' else ''}" \
        f"_max{cw.get('max_weight', 10)}"
    return f"{model}|{tag}" + ("" if no_prior else f"|g{g:g}")


def _calibrate(df, fits, used, scope, feat_cols, C, cal_cfg, uq, pi_true, spec=None, sel_cfg=None,
               decision_rule="argmax", exceed_maps=None):
    """
    Per-sensor calibrators (pooled fallback), selective thresholds, skill, and the selection
    score (classifier.selection; default: sample-count-weighted mean calibration RPS, as before).
    Also returns the pooled calibrated calibration-sample probabilities (Q, y, w, lv, g = sample
    day) used for the decision rule, exceedance recalibration and interval calibration.
    exceed_maps: fitted exceedance maps; every calibrator is wrapped (ExceedanceRecalibrator).
    """
    calibrators, selective, skill = {}, {}, {}
    min_cal = int(cal_cfg.get("min_samples", 30))
    raw = {}
    for s in used:
        key = "pooled" if scope == "pooled" else s
        cal = df[(df["split"] == "calibration") & (df["sensor"] == s)]
        if cal.empty or key not in fits:
            continue
        Ps, meta = sample_level(full_proba(fits[key]["model"], cal[feat_cols], C), cal)
        raw[s] = (Ps, meta, key)

    def make(key):
        return make_calibrator(cal_cfg, fits[key], C)

    if raw:
        Pp = np.vstack([v[0] for v in raw.values()])
        yp = np.concatenate([v[1]["ref_label"].to_numpy().astype(int) for v in raw.values()])
        wp = np.concatenate([v[1]["weight"].to_numpy() for v in raw.values()])
        calibrators["_pooled"] = make(next(iter(raw.values()))[2]).fit(Pp, np.eye(C)[yp], wp)
        calibrators = wrap_calibrators(calibrators, exceed_maps)
    scores, wts, Qs, ys, ws, lvs, gs = [], [], [], [], [], [], []
    for s, (Ps, meta, key) in raw.items():
        y = meta["ref_label"].to_numpy().astype(int)
        w = meta["weight"].to_numpy()
        c = (wrap_calibrators({s: make(key).fit(Ps, np.eye(C)[y], w)}, exceed_maps)[s]
             if len(y) >= min_cal else calibrators.get("_pooled"))
        calibrators[s] = c
        Q = c.transform(Ps) if c is not None else Ps
        u = predictive_entropy_norm(Q)
        sel_cfg_s = uq.get("selective", {})
        err = (np.abs(decide(Q, decision_rule) - y) > int(sel_cfg_s.get("bin_tolerance", 0))).astype(float)
        selective[s] = select_selective_threshold(u, err, w, target_error=sel_cfg_s.get("target_error"),
                                                  target_coverage=sel_cfg_s.get("target_coverage", 0.8))
        skill[s] = skill_summary(Q, y, w, pi_true)
        skill[s]["calibration"] = calibration_diagnostics(Ps, Q, y, w, C)
        skill[s]["calibrator_source"] = "sensor" if len(y) >= min_cal else "pooled"
        scores.append(skill[s]["rps"] * len(y)); wts.append(len(y))
        Qs.append(Q); ys.append(y); ws.append(w)
        gs.append((pd.to_datetime(meta["date"]).dt.strftime("%Y%m%d") if "date" in meta
                   else meta["sample_uid"].astype(str)).to_numpy())
        lvs.append(meta["log10_value"].to_numpy(float) if "log10_value" in meta else np.full(len(y), np.nan))
    out = {"calibrators": calibrators, "selective": selective, "skill": skill,
           "Q": np.vstack(Qs) if Qs else None, "y": np.concatenate(ys) if ys else None,
           "w": np.concatenate(ws) if ws else None, "lv": np.concatenate(lvs) if lvs else None,
           "g": np.concatenate(gs) if gs else None}
    metric = str((sel_cfg or {}).get("metric", "rps")).lower()
    if not Qs:
        out.update(score=np.inf, components={"rps": np.inf})
    elif metric == "rps" or spec is None:
        rps = float(np.sum(scores) / max(np.sum(wts), 1))
        out.update(score=rps, components={"rps": rps})
    else:
        comp = selection_components(out["Q"], out["y"], out["w"], out["lv"], spec, pi_true, decision_rule)
        out.update(score=selection_loss(comp, sel_cfg), components=comp)
    return out


def source_sample_weights(samples: pd.DataFrame, cfg: dict[str, Any] | None):
    """
    uq.sample_weighting: down-weight high-cadence reference methods (IFCB) per
    site-day so they do not swamp weekly pier microscopy.

      {enabled: true, methods: [ifcb], period: W, block_weight: 1.0,
       splits: [train, calibration]}

    For each listed method, every (method, Location_Code, period) block carries
    total weight block_weight, shared equally over its site-days and then over
    samples within a site-day: w = block_weight / (n_site_days * n_same_day).
    period W with block_weight 1 makes a near-daily IFCB site-week count like
    one weekly pier sample; period D only splits same-day replicates. Other
    methods and splits (test) keep weight 1, so evaluation is unchanged.
    Returns ({sample_uid: weight} or None, summary dict or None).
    """
    cfg = cfg or {}
    if not cfg.get("enabled", False):
        return None, None
    if "method" not in samples.columns or samples.attrs.get("has_method") is False:
        raise ValueError("uq.sample_weighting needs a 'method' column: use the non-strict "
                         "combined_augmented.csv as uq.insitu_file")
    methods = cfg.get("methods", ["ifcb"])
    methods = None if methods in ("all", ["all"]) else {str(m) for m in methods}
    splits = set(cfg.get("splits", ["train", "calibration"]))
    period = str(cfg.get("period", "W"))
    block_w = float(cfg.get("block_weight", 1.0))
    d = samples[["sample_uid", "split", "method", "Location_Code", "date"]].copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    sel = d["split"].isin(splits) & (True if methods is None else d["method"].astype(str).isin(methods))
    w = pd.Series(1.0, index=d.index)
    if not sel.any():
        LOG.warning("sample_weighting: no %s samples with method in %s", sorted(splits), methods)
    else:
        x = d[sel].copy()
        x["_blk"] = x["date"].dt.to_period(period).astype(str)
        site_day = x.groupby(["method", "Location_Code", "date"])["sample_uid"].transform("size")
        days = x.groupby(["method", "Location_Code", "_blk"])["date"].transform("nunique")
        w[sel] = block_w / (days.to_numpy() * site_day.to_numpy())
    sw = pd.Series(w.to_numpy(), index=d["sample_uid"].astype(str).to_numpy())
    sw = sw[~sw.index.duplicated()]
    info = {"methods": sorted(methods) if methods else "all", "period": period, "block_weight": block_w,
            "splits": sorted(splits), "n_weighted": int(sel.sum())}
    for sp in sorted(splits):
        m = d["split"] == sp
        if m.any():
            is_m = sel[m].to_numpy()
            info[f"{sp}_share_before"] = float(is_m.mean())
            info[f"{sp}_share_after"] = float(w[m][is_m].sum() / max(w[m].sum(), EPS))
    return sw, info


def apply_sample_weights(rows: pd.DataFrame, counts: pd.DataFrame, sw: pd.Series):
    """Scale matched-pixel weights (classifier, calibration, fusion) and encoder counts."""
    rows = rows.copy()
    rows["pix_weight"] = rows["pix_weight"].to_numpy(dtype=np.float64) * \
        rows["sample_uid"].astype(str).map(sw).fillna(1.0).to_numpy()
    if "sample_uid" in counts.columns:
        counts = counts.copy()
        counts["count"] = counts["count"].to_numpy(dtype=np.float64) * \
            counts["sample_uid"].astype(str).map(sw).fillna(1.0).to_numpy()
    else:
        LOG.warning("sample_weighting: counts have no sample_uid (cached matchups); encoder unweighted")
    return rows, counts


def _sample_normalized_weights(rows: pd.DataFrame) -> np.ndarray:
    """Each (sample, sensor) contributes total weight = its temporal weight."""
    return rows["pix_weight"].to_numpy(dtype=np.float64) * len(rows) / max(
        rows["pix_weight"].sum(), EPS)
