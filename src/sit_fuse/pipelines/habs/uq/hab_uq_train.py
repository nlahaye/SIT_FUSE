"""
Train calibrated probabilistic HAB concentration-bin models on SIT-FUSE outputs.

Per species (pDA, PN seriata, PN delicatissima, total phytoplankton) and per
sensor (or pooled across sensors), reusing sit_fuse.train.sf_uq_utils:

  1. matchups      point-pixel rows + encoder counts (hab_uq_matchups)
  2. encoder       HABContextEncoder (production histogram-mode rule, smoothed
                   agreement tables, bootstrap assignment ensemble), cross-fit
                   over split groups so classifier rows never see their own
                   samples in the cluster->bin tables
  3. targets       build_targets modes, with the one-hot reference optionally
                   replaced by measurement-uncertainty soft labels
  4. classifier    classifier_utils.build_models (+ expand_soft_targets)
  5. members       forest trees or group-bootstrap refits (epistemic spread)
  6. calibration   ProbabilityCalibrator (prior correction + temperature) on
                   sample-level linear-pooled probabilities, per sensor
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
from sit_fuse.pipelines.habs.uq.hab_insitu import assign_splits, load_insitu, species_frame
from sit_fuse.pipelines.habs.uq.hab_uq_common import (
    SENSORS, ranked_probability_score, species_spec, instrument_dirs, utcnow_iso)
from sit_fuse.pipelines.habs.uq.hab_uq_matchups import build_sensor_matchups, sample_level

LOG = logging.getLogger("hab-uq-train")

# matchup bookkeeping that must never become a feature (target leakage / ids)
HAB_NON_FEATURES = {
    "value", "log10_value", "fold", "pix_weight", "n_match_pixels", "day_offset",
    "hours_offset", "ref_label", "weight", "time_block",
}
HAB_NON_FEATURE_PREFIXES = ("ref_soft_", "sensor_")


def hab_feature_columns(df: pd.DataFrame, features_cfg: dict[str, Any],
                        sensors: list[str] | None = None) -> list[str]:
    cols = [c for c in select_feature_columns(df, features_cfg)
            if c not in HAB_NON_FEATURES and not c.startswith(HAB_NON_FEATURE_PREFIXES)]
    if sensors and len(sensors) > 1:
        cols += [f"sensor_is_{s}" for s in sensors]
    if features_cfg.get("use_day_offset", False):
        cols.append("abs_day_offset")
    return cols


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
                     enc_cfg: dict[str, Any], features_cfg: dict[str, Any], seed: int = 42):
    """
    Train rows: features from an encoder fit on train counts of the other
    folds. Calibration/test rows: features from the encoder fit on all train
    counts (the encoder used at inference time).
    """
    train_counts = counts[counts["split"] == "train"]
    final = fit_encoder(train_counts, n_classes, num_child, enc_cfg, seed)
    parts = []
    tr = rows[rows["split"] == "train"]
    for k in sorted(tr["fold"].unique()):
        sub = tr[tr["fold"] == k]
        c_k = train_counts[train_counts["fold"] != k]
        if c_k.empty:
            parts.append(final.transform(sub, features_cfg))
            continue
        enc_k = fit_encoder(c_k, n_classes, num_child, enc_cfg, seed + int(k) + 1)
        parts.append(enc_k.transform(sub, features_cfg))
    other = rows[rows["split"] != "train"]
    if len(other):
        parts.append(final.transform(other, features_cfg))
    return pd.concat(parts).loc[rows.index], final


# ---------------------------------------------------------------------------
# Classifier + members
# ---------------------------------------------------------------------------

def _fit_one(model_proto, X, T, w, seed=None):
    m = copy.deepcopy(model_proto)
    if seed is not None:
        try:
            m.set_params(clf__random_state=seed)
        except Exception:
            pass
    Xe, ye, we = expand_soft_targets(X, T, base_weight=w)
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
    model, ye, we = _fit_one(proto, X, T, w)
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
            mb, _, _ = _fit_one(proto, X[keep], T[keep], wb[keep], seed=seed + b + 1)
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
    sensors = uq.get("sensors") or list(hab_yml["instruments"].keys())
    sensors = [s for s in sensors if s in SENSORS]
    scope = uq.get("model_scope", "pooled")
    invalid = InvalidClusters.from_config(uq)

    if samples is None:
        ins = load_insitu(uq.get("insitu_file", hab_yml.get("insitu_file")),
                          max_depth_m=uq.get("max_depth_m", 1.0),
                          start_date=uq.get("start_date"), end_date=uq.get("end_date"))
        samples = assign_splits(species_frame(ins, spec), uq.get("splits"))

    rows_all, counts_all = [], {}
    enc_by_sensor = {}
    for s in sensors:
        if matchups is not None and s in matchups:
            rows, counts = matchups[s]
        else:
            dirs = uq.get("product_dirs", {}).get(s) or instrument_dirs(hab_yml, s)
            rows, counts = build_sensor_matchups(samples, s, dirs, uq, features_cfg, invalid)
        if rows.empty or counts.empty or (counts["split"] == "train").sum() == 0:
            LOG.warning("%s/%s: no usable matchups", use_key, s)
            continue
        enc_rows, enc = cross_fit_encode(rows, counts, C, num_child, enc_cfg, features_cfg, seed)
        enc_by_sensor[s] = enc
        counts_all[s] = counts
        rows_all.append(enc_rows)
    if not rows_all:
        raise RuntimeError(f"No matchups for {use_key}")
    used = list(enc_by_sensor)
    df = add_sensor_columns(pd.concat(rows_all, ignore_index=True), used)
    feat_cols = hab_feature_columns(df, features_cfg, used if scope == "pooled" else None)
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0

    train_s = df[df["split"] == "train"].drop_duplicates("sample_uid")
    pi_true = np.bincount(train_s["ref_label"].astype(int), minlength=C).astype(float) + 1.0
    pi_true /= pi_true.sum()           # climatology (skill reference, fusion prior)

    model_name = clf_cfg.get("model", "auto")
    candidates = (clf_cfg.get("candidates", ["encoder_only", "hist_gradient_boosting", "random_forest"])
                  if model_name == "auto" else [model_name])
    trials = {}
    for cand in candidates:
        cfg_c = {**clf_cfg, "model": cand}
        fits = _fit_scopes(df, used, scope, enc_by_sensor, feat_cols, tgt_cfg, cfg_c, C, seed)
        trials[cand] = (fits, *_calibrate(df, fits, used, scope, feat_cols, C, cal_cfg, uq, pi_true))
    sel_key = min(trials, key=lambda k: trials[k][4])
    fits, calibrators, selective, skill, cal_score = trials[sel_key]
    model_selection = {k: float(v[4]) for k, v in trials.items()}
    LOG.info("%s model selection (calibration RPS): %s -> %s", use_key, model_selection, sel_key)

    bundle = {
        "format": "sitfuse-hab-uq-v1", "created": utcnow_iso(), "species": use_key,
        "spec": spec.__dict__, "n_classes": C, "num_child": num_child, "sensors": used,
        "model_scope": scope, "encoders": enc_by_sensor, "fits": fits,
        "feature_columns": feat_cols, "calibrators": calibrators, "selective": selective,
        "skill": skill, "pi_true": pi_true, "uq_cfg": uq,
        "model_name": sel_key, "model_selection_calibration_rps": model_selection,
        "split_summary": samples.groupby("split")["sample_uid"].nunique().to_dict(),
    }
    out = Path(out_dir) / f"hab_uq_{spec.fname_token}.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    LOG.info("saved %s", out)
    bundle["path"] = str(out)
    bundle["_frame"] = df          # in-memory only (evaluation without rematching)
    return bundle


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


def _calibrate(df, fits, used, scope, feat_cols, C, cal_cfg, uq, pi_true):
    """Per-sensor calibrators (pooled fallback), selective thresholds, skill, mean calibration RPS."""
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
        f = fits[key]
        pc = cal_cfg.get("prior_correction", True) and f["pi_train"] is not None
        # undo class balancing toward the (unweighted) training-target prior
        return ProbabilityCalibrator(cal_cfg.get("method", "temperature"),
                                     pi_train=f["pi_train"] if pc else None,
                                     pi_true=f["pi_target"] if pc else None)

    if raw:
        Pp = np.vstack([v[0] for v in raw.values()])
        yp = np.concatenate([v[1]["ref_label"].to_numpy().astype(int) for v in raw.values()])
        wp = np.concatenate([v[1]["weight"].to_numpy() for v in raw.values()])
        calibrators["_pooled"] = make(next(iter(raw.values()))[2]).fit(Pp, np.eye(C)[yp], wp)
    scores, wts = [], []
    for s, (Ps, meta, key) in raw.items():
        y = meta["ref_label"].to_numpy().astype(int)
        w = meta["weight"].to_numpy()
        c = make(key).fit(Ps, np.eye(C)[y], w) if len(y) >= min_cal else calibrators.get("_pooled")
        calibrators[s] = c
        Q = c.transform(Ps) if c is not None else Ps
        u = predictive_entropy_norm(Q)
        sel_cfg = uq.get("selective", {})
        err = (np.abs(Q.argmax(1) - y) > int(sel_cfg.get("bin_tolerance", 0))).astype(float)
        selective[s] = select_selective_threshold(u, err, w, target_error=sel_cfg.get("target_error"),
                                                  target_coverage=sel_cfg.get("target_coverage", 0.8))
        skill[s] = skill_summary(Q, y, w, pi_true)
        scores.append(skill[s]["rps"] * len(y)); wts.append(len(y))
    score = float(np.sum(scores) / max(np.sum(wts), 1)) if wts else np.inf
    return calibrators, selective, skill, score


def _sample_normalized_weights(rows: pd.DataFrame) -> np.ndarray:
    """Each (sample, sensor) contributes total weight = its temporal weight."""
    return rows["pix_weight"].to_numpy(dtype=np.float64) * len(rows) / max(
        rows["pix_weight"].sum(), EPS)
