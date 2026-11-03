"""
Multi-sensor fusion of calibrated HAB bin probabilities.

Inputs are the per-sensor *.uq.tif products (p_c bands + overpass_hour_utc),
which are already harmonized in one sense: every sensor is calibrated to the
same in-situ bins. Fusion then combines them per pixel and day.

Weights per sensor s, pixel i, day d
    w_si = skill_s  x  exp(-|t_si - t*| / tau_h)  x  q_si
  skill_s  from calibration samples (RPSS vs climatology, clipped >= 0, or
           learned stacking weights; see fit_pool_weights)
  t_si     overpass time; t* = reference time (e.g. 10:00 local, the
           typical pier sampling time, or the product's nominal time)
  q_si     optional pixel quality: selective_flag, 1 - predictive_entropy

Pools
  linear       P = sum_s w_s P_s / sum_s w_s           (robust, never overconfident)
  log          log P = sum_s w_s log P_s / sum_s w_s   (geometric; consensus-seeking)
  poe          log P = sum_s w_s log P_s - (sum w - 1) log pi   (product of experts
               with shared-prior correction; sharpens; temper by 1/(1+rho(S-1))
               for correlated sensor errors)
Temporal
  bayes_filter forward pass of a discrete ordinal HMM over days: predict with
               transition K (lazy random walk on bins, persistence ~ bloom
               decorrelation) then update with the per-day pooled likelihood
               ratio P_fused / pi. Gives gap-filled daily products with
               propagated uncertainty and a 'days since observation' layer.
Diagnostics
  n_sensors, sensor disagreement (generalized Jensen-Shannon divergence),
  fused entropy, exceedance probabilities and intervals (via hab_uq_inference).
"""

from __future__ import annotations

import glob
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sit_fuse.train.sf_uq_utils import EPS, ProbabilityCalibrator, write_band_stack
from sit_fuse.pipelines.habs.uq.hab_uq_common import NODATA, SENSORS, species_spec


# ---------------------------------------------------------------------------
# Core array operations: P (n, S, C), mask (n, S)
# ---------------------------------------------------------------------------

def pool(P: np.ndarray, mask: np.ndarray, w: np.ndarray, method: str = "linear",
         prior: np.ndarray | None = None, rho: float = 0.0) -> np.ndarray:
    P = np.clip(np.asarray(P, float), EPS, 1.0)
    m = np.asarray(mask, bool)
    w = np.where(m, np.asarray(w, float), 0.0)
    ws = w.sum(1, keepdims=True)
    C = P.shape[-1]
    pi = np.full(C, 1.0 / C) if prior is None else np.asarray(prior, float)
    out = np.tile(pi, (P.shape[0], 1))
    has = ws[:, 0] > 0
    if method == "linear":
        out[has] = (w[has, :, None] * P[has]).sum(1) / ws[has]
    elif method in ("log", "poe"):
        L = np.log(P)
        if method == "log":
            z = (w[has, :, None] * L[has]).sum(1) / ws[has]
        else:
            n_eff = m[has].sum(1, keepdims=True).astype(float)
            temper = 1.0 / (1.0 + rho * np.maximum(n_eff - 1, 0))
            wt = w[has] * temper
            z = (wt[:, :, None] * (L[has] - np.log(pi))).sum(1) + np.log(pi)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        out[has] = e / e.sum(1, keepdims=True)
    else:
        raise ValueError(method)
    return out


def js_disagreement(P: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Generalized JS divergence among available sensors / log(C) (0 = agree)."""
    P = np.clip(P, EPS, 1)
    m = mask.astype(float)
    n = np.maximum(m.sum(1), 1)
    mean = np.clip((m[:, :, None] * P).sum(1) / n[:, None], EPS, 1)
    H = lambda Q: -(Q * np.log(Q)).sum(-1)
    out = H(mean) - (m * H(P)).sum(1) / n
    out[m.sum(1) < 2] = 0.0
    return np.maximum(out, 0) / np.log(P.shape[-1])


def ordinal_transition(C: int, stay: float = 0.7, spread: float = 1.0) -> np.ndarray:
    """K[i, j] = P(bin j tomorrow | bin i today): lazy, discretized-Gaussian random walk."""
    i = np.arange(C)
    G = np.exp(-0.5 * ((i[:, None] - i[None, :]) / max(spread, EPS)) ** 2)
    np.fill_diagonal(G, 0)
    G = G / np.maximum(G.sum(1, keepdims=True), EPS)
    return stay * np.eye(C) + (1 - stay) * G


def bayes_filter(daily_P: list[np.ndarray | None], prior: np.ndarray, K: np.ndarray,
                 obs_temper: float = 1.0):
    """
    daily_P: per day, (n, C) pooled calibrated probabilities or None/NaN rows
    for no observation. Uses the likelihood ratio P/prior so a calibrated
    posterior is not double-counting the climatological prior.
    Returns list of (posterior (n, C), days_since_obs (n,)).
    """
    pi = np.asarray(prior, float)
    n = next(p.shape[0] for p in daily_P if p is not None)
    post = np.tile(pi, (n, 1))
    since = np.full(n, np.inf)
    out = []
    for P in daily_P:
        pred = post @ K
        if P is not None:
            ok = np.isfinite(P).all(1)
            lr = np.where(ok[:, None], (np.clip(np.nan_to_num(P, nan=1.0), EPS, 1) / pi) ** obs_temper, 1.0)
            pred = pred * lr
            since = np.where(ok, 0, since + 1)
        else:
            since = since + 1
        post = pred / np.maximum(pred.sum(1, keepdims=True), EPS)
        out.append((post.copy(), since.copy()))
    return out


# ---------------------------------------------------------------------------
# Learning weights on calibration samples (stacking)
# ---------------------------------------------------------------------------

def fit_pool_weights(P: np.ndarray, mask: np.ndarray, y: np.ndarray, sample_w=None,
                     method: str = "log", prior=None, l2: float = 1e-2, time_w=None) -> np.ndarray:
    """
    Per-sensor non-negative weights minimizing pooled NLL of held-out
    calibration samples (softplus parameterization, small L2 to 1).
    """
    S = P.shape[1]
    sw = np.ones(len(y)) if sample_w is None else np.asarray(sample_w, float)
    tw = np.ones(mask.shape) if time_w is None else time_w

    def loss(theta):
        ws = np.log1p(np.exp(theta))
        Q = pool(P, mask, tw * ws[None, :], method, prior)
        nll = -np.log(np.clip(Q[np.arange(len(y)), y], EPS, 1))
        return float((sw * nll).sum() / sw.sum() + l2 * ((ws - 1) ** 2).sum())

    res = minimize(loss, x0=np.full(S, np.log(np.e - 1)), method="L-BFGS-B")
    return np.log1p(np.exp(res.x))


def skill_weights(bundle: dict[str, Any], sensors: list[str], floor: float = 0.05) -> np.ndarray:
    return np.array([max(bundle["skill"].get(s, {}).get("rpss", 0.0), floor) for s in sensors])


# ---------------------------------------------------------------------------
# Sample-level fusion (evaluation) - from hab_uq_evaluate.predict_samples output
# ---------------------------------------------------------------------------

def stack_by_sample(P: np.ndarray, meta: pd.DataFrame, sensors: list[str]):
    """(n_samples, S, C) array + mask, one row per in-situ sample."""
    uids = meta["sample_uid"].astype(str).unique()
    ui = {u: i for i, u in enumerate(uids)}
    si = {s: j for j, s in enumerate(sensors)}
    C = P.shape[1]
    A = np.full((len(uids), len(sensors), C), 1.0 / C)
    M = np.zeros((len(uids), len(sensors)), bool)
    H = np.full((len(uids), len(sensors)), np.nan)
    for k, (u, s) in enumerate(zip(meta["sample_uid"].astype(str), meta["sensor"])):
        if s in si:
            A[ui[u], si[s]] = P[k]
            M[ui[u], si[s]] = True
            H[ui[u], si[s]] = meta["hours_offset"].iloc[k] if "hours_offset" in meta else np.nan
    first = meta.drop_duplicates("sample_uid").set_index(meta.drop_duplicates("sample_uid")["sample_uid"].astype(str))
    return A, M, H, first.loc[uids].reset_index(drop=True)


def fuse_samples(bundle, P_cal, meta_cal, P_test, meta_test, method="log",
                 learn_weights=True, recalibrate=True) -> dict[str, Any]:
    sensors = bundle["sensors"]
    C = bundle["n_classes"]
    Ac, Mc, _, mc = stack_by_sample(P_cal, meta_cal, sensors)
    At, Mt, _, mt = stack_by_sample(P_test, meta_test, sensors)
    yc = mc["ref_label"].to_numpy().astype(int)
    w = (fit_pool_weights(Ac, Mc, yc, mc["weight"], method, bundle["pi_true"]) if learn_weights
         else skill_weights(bundle, sensors))
    Fc = pool(Ac, Mc, np.broadcast_to(w, Mc.shape), method, bundle["pi_true"])
    Ft = pool(At, Mt, np.broadcast_to(w, Mt.shape), method, bundle["pi_true"])
    cal = None
    if recalibrate:
        cal = ProbabilityCalibrator("temperature").fit(Fc, np.eye(C)[yc], mc["weight"].to_numpy())
        Ft = cal.transform(Ft)
    mt = mt.copy()
    mt["n_sensors"] = Mt.sum(1)
    mt["disagreement"] = js_disagreement(At, Mt)
    return {"weights": dict(zip(sensors, map(float, w))), "P_test": Ft, "meta_test": mt,
            "fusion_temperature": None if cal is None else cal.T}


# ---------------------------------------------------------------------------
# Raster driver
# ---------------------------------------------------------------------------

def _read_uq(path, ref_profile=None):
    import rasterio
    from rasterio.warp import Resampling, reproject
    with rasterio.open(path) as src:
        names = list(src.descriptions)
        data = src.read().astype(np.float64)
        prof = src.profile.copy()
    if ref_profile is not None and (prof["transform"] != ref_profile["transform"] or
                                    prof["width"] != ref_profile["width"]):
        dst = np.full((data.shape[0], ref_profile["height"], ref_profile["width"]), NODATA)
        for b in range(data.shape[0]):
            reproject(data[b], dst[b], src_transform=prof["transform"], src_crs=prof["crs"],
                      dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
                      src_nodata=NODATA, dst_nodata=NODATA, resampling=Resampling.nearest)
        data, prof = dst, ref_profile
    data[data == NODATA] = np.nan
    return {n: data[i] for i, n in enumerate(names)}, prof


def fuse_day(uq_dir: str, date, species_token: str, sensors: list[str], weights: dict[str, float],
             C: int, method: str = "log", ref_local_hour: float = 10.0, tau_hours: float = 4.0,
             prior=None, quality: str = "none"):
    """Read per-sensor *.uq.tif for one day and pool. Returns (P (H,W,C), diag, profile)."""
    from sit_fuse.pipelines.habs.uq.hab_uq_common import pixel_centers
    stacks, masks, ws, prof = [], [], [], None
    ds = pd.Timestamp(date).strftime("%Y%m%d")
    for s in sensors:
        hits = glob.glob(os.path.join(uq_dir, "**", f"{SENSORS[s].prefix}.{ds}.L3m.DAY.{species_token}.uq.tif"),
                         recursive=True)
        if not hits:
            continue
        b, prof = _read_uq(hits[0], prof)
        P = np.stack([b[f"p_{c}"] for c in range(C)], -1)
        ok = np.isfinite(P).all(-1)
        lat, lon = pixel_centers(prof)
        local = b["overpass_hour_utc"] + lon / 15.0
        tw = np.exp(-np.abs(local - ref_local_hour) / max(tau_hours, EPS))
        q = np.ones_like(tw)
        if quality == "selective" and "selective_flag" in b:
            q = np.where(b["selective_flag"] > 0, 1.0, 0.25)
        elif quality == "entropy" and "predictive_entropy" in b:
            q = 1.0 - np.nan_to_num(b["predictive_entropy"], nan=1.0)
        stacks.append(np.nan_to_num(P, nan=1.0 / C))
        masks.append(ok)
        ws.append(np.where(ok, weights.get(s, 1.0) * np.nan_to_num(tw) * q, 0.0))
    if not stacks:
        return None, None, None
    Hh, Ww = stacks[0].shape[:2]
    Pn = np.stack(stacks, 2).reshape(-1, len(stacks), C)
    Mn = np.stack(masks, -1).reshape(-1, len(stacks))
    Wn = np.stack(ws, -1).reshape(-1, len(stacks))
    F = pool(Pn, Mn, Wn, method, prior)
    diag = {"n_sensors": Mn.sum(1).reshape(Hh, Ww).astype(np.float32),
            "sensor_disagreement": js_disagreement(Pn, Mn).reshape(Hh, Ww).astype(np.float32),
            "weight_sum": Wn.sum(1).reshape(Hh, Ww).astype(np.float32)}
    F = F.reshape(Hh, Ww, C)
    F[diag["n_sensors"] == 0] = np.nan
    return F, diag, prof


def run_fusion(uq_dir, out_dir, dates, bundle: dict[str, Any], method="log", temporal_filter=True,
               ref_local_hour=10.0, tau_hours=4.0, stay=0.7, quality="selective"):
    """Daily fused products + optional ordinal Bayesian filter across days."""
    from sit_fuse.pipelines.habs.uq.hab_uq_inference import summarize_probabilities
    spec = species_spec(bundle["species"], bundle["spec"])
    C, sensors = bundle["n_classes"], bundle["sensors"]
    w = bundle.get("fusion_weights") or dict(zip(sensors, skill_weights(bundle, sensors)))
    days = [fuse_day(uq_dir, d, spec.fname_token, sensors, w, C, method, ref_local_hour,
                     tau_hours, bundle["pi_true"], quality) for d in dates]
    prof = next((x[2] for x in days if x[2] is not None), None)
    if prof is None:
        return []
    Hh, Ww = prof["height"], prof["width"]
    flat = [None if x[0] is None else x[0].reshape(-1, C) for x in days]
    filt = bayes_filter(flat, bundle["pi_true"], ordinal_transition(C, stay)) if temporal_filter else None
    paths = []
    for k, d in enumerate(dates):
        layers = {}
        if days[k][0] is not None:
            Fk = days[k][0].reshape(-1, C)
            ok = np.isfinite(Fk).all(1)
            s = summarize_probabilities(np.where(ok[:, None], Fk, 1.0 / C), None, spec, None)
            layers.update({f"fused_{n}": np.where(ok, v, NODATA).reshape(Hh, Ww) for n, v in s.items()})
            layers.update(days[k][1])
        if filt is not None:
            post, since = filt[k]
            s = summarize_probabilities(post, None, spec, None)
            layers.update({f"filtered_{n}": v.reshape(Hh, Ww) for n, v in s.items()})
            layers["days_since_observation"] = np.where(np.isfinite(since), since, NODATA).reshape(Hh, Ww)
        if not layers:
            continue
        path = os.path.join(out_dir, f"SITFUSE_MULTI.{pd.Timestamp(d).strftime('%Y%m%d')}.L3m.DAY."
                                     f"{spec.fname_token}.fused_uq.tif")
        write_band_stack(path, {k2: np.nan_to_num(v, nan=NODATA) for k2, v in layers.items()}, prof,
                         tags={"product": "sitfuse_hab_fused_uq", "method": method, "weights": w,
                               "ref_local_hour": ref_local_hour, "tau_hours": tau_hours,
                               "transition_stay": stay, "sensors": sensors})
        paths.append(path)
    return paths
