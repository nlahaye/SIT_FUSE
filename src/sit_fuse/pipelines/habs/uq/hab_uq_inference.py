"""
Per-scene probabilistic + UQ HAB products from a trained bundle.

Output (one multi-band GeoTIFF per sensor/day/species, same grid as the
SIT-FUSE cluster GeoTIFF):
  {PREFIX}.{YYYYMMDD}.L3m.DAY.{token}.uq.tif

Bands
  p_{c}                       calibrated P(concentration bin c)
  pred_label                  argmax bin (comparable to production product)
  pred_label_selective        pred_label, -1 where flagged by selective rule
  selective_flag              1 retained / 0 flagged
  confidence                  P(pred_label)
  context_assignment_bin      bin of the (train-only) histogram-mode assignment
  expected_log10_conc         E[log10 X] (log-uniform within bins)
  log10_conc_q05/q50/q95      credible interval of log10 concentration
  exceed_p_{thr}              P(X >= alert threshold) (e.g. 1e4 cells/L, 0.5 ng/mL)
  ordinal_sd                  SD of bin index (ordinal spread)
  predictive_entropy          total uncertainty / log(C)
  margin_uncertainty          1 - (p1 - p2)
  ensemble_expected_entropy   aleatoric component
  ensemble_mutual_information epistemic component
  ensemble_variance           member variance
  assignment_vote_variance    structural (cluster->bin mapping) uncertainty
  overpass_hour_utc           nominal overpass time (fusion bookkeeping)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sit_fuse.train.sf_uq_utils import (
    InvalidClusters, ensemble_decomposition, full_proba, head_valid_mask,
    predictive_entropy_norm, scene_feature_frame, write_band_stack)
from sit_fuse.train.classifier_utils import margin_uncertainty
from sit_fuse.pipelines.habs.uq.hab_uq_common import (
    NODATA, SENSORS, exceedance_probability, expected_log10_concentration,
    log10_concentration_quantile, overpass_hours_utc, ordinal_moments, species_spec,
    pixel_centers, utcnow_iso)
from sit_fuse.pipelines.habs.uq.hab_uq_matchups import load_scene
from sit_fuse.pipelines.habs.uq.hab_uq_train import add_sensor_columns, member_probabilities

LOG = logging.getLogger("hab-uq-inference")

LAYER_DESCRIPTIONS = {
    "p_": "Calibrated probability of concentration bin",
    "pred_label": "Most probable concentration bin",
    "pred_label_selective": "Most probable bin; -1 where flagged as uncertain",
    "selective_flag": "1 = retained by calibrated selective rule",
    "confidence": "Probability of the most probable bin",
    "context_assignment_bin": "Bin from histogram-mode context assignment (production analogue)",
    "expected_log10_conc": "Expected log10 concentration",
    "log10_conc_q": "Quantile of log10 concentration",
    "exceed_p_": "Probability concentration >= threshold",
    "ordinal_sd": "Standard deviation of bin index",
    "predictive_entropy": "Normalized predictive entropy (total uncertainty)",
    "margin_uncertainty": "1 - (p_max - p_second)",
    "ensemble_expected_entropy": "Mean member entropy (aleatoric)",
    "ensemble_mutual_information": "Mutual information (epistemic)",
    "ensemble_variance": "Summed member-probability variance",
    "assignment_vote_variance": "Assignment-ensemble vote variance (structural)",
    "overpass_hour_utc": "Nominal overpass time, hours UTC",
}


def _thr_name(t: float) -> str:
    return f"{t:g}".replace(".", "p").replace("+", "")


def summarize_probabilities(Q: np.ndarray, members: np.ndarray | None, spec, selective_thr: float | None,
                            quantiles=(0.05, 0.5, 0.95)) -> dict[str, np.ndarray]:
    out = {}
    C = Q.shape[1]
    for c in range(C):
        out[f"p_{c}"] = Q[:, c]
    pred = Q.argmax(1)
    u = predictive_entropy_norm(Q)
    out["pred_label"] = pred.astype(np.float32)
    keep = u <= (np.inf if selective_thr is None else selective_thr)
    out["pred_label_selective"] = np.where(keep, pred, -1).astype(np.float32)
    out["selective_flag"] = keep.astype(np.float32)
    out["confidence"] = Q.max(1)
    out["expected_log10_conc"] = expected_log10_concentration(Q, spec.edges, spec.detection_floor)
    for q in quantiles:
        out[f"log10_conc_q{int(round(q * 100)):02d}"] = log10_concentration_quantile(
            Q, spec.edges, q, spec.detection_floor)
    for t in spec.alert_thresholds:
        out[f"exceed_p_{_thr_name(t)}"] = exceedance_probability(Q, spec.edges, t, spec.detection_floor)
    out["ordinal_sd"] = ordinal_moments(Q)[1]
    out["predictive_entropy"] = u
    out["margin_uncertainty"] = margin_uncertainty(Q)
    if members is not None and len(members) > 1:
        _, _, exp_e, mi, var = ensemble_decomposition(members)
        out["ensemble_expected_entropy"] = exp_e
        out["ensemble_mutual_information"] = mi
        out["ensemble_variance"] = var
    return out


# features that only exist in head_outputs.tif with MC / OOD bands (label_only
# synthesizes the standard head bands, but not these)
HEAD_ONLY_PREFIXES = ("mc_", "ood_")


def scene_calibrator(bundle: dict[str, Any], sensor: str):
    """Sensor calibrator, pooled fallback; refuse raw probabilities unless allowed."""
    cal = bundle["calibrators"].get(sensor) or bundle["calibrators"].get("_pooled")
    if cal is None and bundle["uq_cfg"].get("calibration", {}).get("require", True):
        raise RuntimeError(f"{bundle['species']}/{sensor}: bundle has no calibrator "
                           "(calibration.require: true)")
    return cal


def check_feature_coverage(bundle: dict[str, Any], columns) -> None:
    """
    A bundle trained on head-output features must not be applied to label-only
    scenes (missing columns would silently become 0 and void the calibration).
    """
    have = set(columns)
    missing = [c for c in bundle["feature_columns"] if c not in have and c.startswith(HEAD_ONLY_PREFIXES)]
    if missing and not bundle["uq_cfg"].get("allow_missing_features", False):
        raise ValueError(f"scene lacks {len(missing)} head-output features the model was trained on "
                         f"(e.g. {missing[:3]}); export head outputs or retrain with input_mode label_only")


def predict_frame(bundle: dict[str, Any], sensor: str, feat: pd.DataFrame,
                  chunk: int = 20000) -> dict[str, np.ndarray]:
    C = bundle["n_classes"]
    spec = species_spec(bundle["species"], bundle["spec"])
    enc = bundle["encoders"][sensor]
    key = "pooled" if bundle["model_scope"] == "pooled" else sensor
    fit = bundle["fits"][key]
    cal = scene_calibrator(bundle, sensor)
    sel = bundle["selective"].get(sensor, {}).get("threshold")
    fcfg = bundle["uq_cfg"].get("features", {})
    cols = bundle["feature_columns"]
    pieces = []
    for s0 in range(0, len(feat), chunk):
        f = enc.transform(feat.iloc[s0:s0 + chunk].assign(sensor=sensor), fcfg)
        f = add_sensor_columns(f, bundle["sensors"])
        if s0 == 0:
            check_feature_coverage(bundle, f.columns)
        for c in cols:
            if c not in f.columns:
                f[c] = 0.0
        X = f[cols]
        P = full_proba(fit["model"], X, C)
        M = member_probabilities(fit, X, C)
        if M is not None and bundle["uq_cfg"].get("max_members"):
            M = M[: int(bundle["uq_cfg"]["max_members"])]
        Q = cal.transform(P) if cal is not None else P
        Mq = cal.transform_members(M) if (cal is not None and M is not None) else M
        if Mq is not None and bundle["uq_cfg"].get("use_member_mean", False):
            Q = Mq.mean(0)
        d = summarize_probabilities(Q, Mq, spec, sel)
        d["context_assignment_bin"] = enc.soft_matrix(f, "assign_hard").argmax(1).astype(np.float32)
        d["assignment_vote_variance"] = f["assign_vote_var"].to_numpy()
        pieces.append(d)
    return {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}


def _describe(cal):
    if cal is None:
        return "none"
    try:
        return cal.describe()
    except Exception:  # pragma: no cover
        return type(cal).__name__


def run_scene(bundle: dict[str, Any], sensor: str, date, search_dirs, out_dir,
              invalid: InvalidClusters | None = None) -> str | None:
    uq = bundle["uq_cfg"]
    sc = load_scene(search_dirs, sensor, date, uq)
    if sc is None:
        return None
    bands, prof, src, mode = sc
    pair = f"{sensor}.{pd.Timestamp(date).strftime('%Y%m%d')}"
    feat, valid = scene_feature_frame(bands, prof, uq.get("features", {}), pair, ref=None, invalid=invalid)
    if feat.empty:
        return None
    try:
        vals = predict_frame(bundle, sensor, feat)
    except ValueError as e:
        LOG.warning("%s %s skipped: %s", sensor, date, e)
        return None
    H, W = valid.shape
    rows, cols = feat["row"].to_numpy(), feat["col"].to_numpy()
    arrays = {}
    for k, v in vals.items():
        a = np.full((H, W), NODATA, np.float32)
        a[rows, cols] = v
        arrays[k] = a
    lat, lon = pixel_centers(prof)
    hrs = overpass_hours_utc(sensor, date, lon)
    arrays["overpass_hour_utc"] = np.where(valid, hrs, NODATA).astype(np.float32)

    spec = species_spec(bundle["species"], bundle["spec"])
    name = f"{SENSORS[sensor].prefix}.{pd.Timestamp(date).strftime('%Y%m%d')}.L3m.DAY.{spec.fname_token}.uq.tif"
    path = Path(out_dir) / sensor / name
    tags = {"product": "sitfuse_hab_uq", "species": bundle["species"], "sensor": sensor,
            "date": str(pd.Timestamp(date).date()), "source_product": src, "input_mode": mode,
            "bin_edges": list(spec.edges), "units": spec.units, "created": utcnow_iso(),
            "model_bundle": bundle.get("path", ""), "selective": bundle["selective"].get(sensor),
            "calibrator": _describe(scene_calibrator(bundle, sensor)),
            "calibration_diagnostics": (bundle.get("skill", {}).get(sensor) or {}).get("calibration"),
            "layer_descriptions": {k: next((d for p, d in LAYER_DESCRIPTIONS.items() if k.startswith(p)), k)
                                   for k in arrays}}
    write_band_stack(path, arrays, prof, tags=tags, nodata=NODATA)
    return str(path)


def run_inference(bundle_path: str, dates, sensors=None, search_dirs_by_sensor=None,
                  out_dir: str = "hab_uq_products") -> list[str]:
    bundle = joblib.load(bundle_path)
    bundle["path"] = bundle_path
    invalid = InvalidClusters.from_config(bundle["uq_cfg"])
    out = []
    for s in sensors or bundle["sensors"]:
        if s not in bundle["encoders"]:
            continue
        for d in dates:
            p = run_scene(bundle, s, d, (search_dirs_by_sensor or {}).get(s, []), out_dir, invalid)
            if p:
                out.append(p)
    return out
