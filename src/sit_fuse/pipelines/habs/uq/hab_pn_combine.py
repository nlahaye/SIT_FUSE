"""
Combine the two Pseudo-nitzschia size-class predictions into P(total PN >= t).

C-HARM predicts one quantity, P(Pseudo-nitzschia total >= 1e4 cells/L). SIT-FUSE predicts the
seriata and delicatissima groups separately (bin probabilities over CELL_RANGES) and, as the
primary comparison target, a direct total-PN model (Pseudo_nitzschia_total). The size-class
combinations here are the checks on that primary product:

  bounds        lower = max(P(S >= t), P(D >= t))                     (either alone exceeds)
                upper = min(1, P(S >= t/2) + P(D >= t/2))              (S + D >= t needs one >= t/2)
  independence  Gaussian copula with rho = 0
  comonotonic   rho = 1 (sum of quantiles at the same rank)
  copula        Gaussian copula with rho fitted on training in-situ pairs (Kendall tau-b, by season)

Within each bin the concentration is log-uniform (the same assumption as
hab_uq_common.exceedance_probability). The copula sum is evaluated on a fixed Sobol / normal-
score sample of n points per pixel, deterministic and chunked over pixels.

Every function takes (N, C) bin probabilities and returns (N,) probabilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sit_fuse.pipelines.habs.uq.hab_uq_common import EPS, _log_edges, exceedance_probability

METHODS = ("lower_bound", "upper_bound", "independence", "comonotonic", "copula")


def exceedance_vec(P: np.ndarray, edges, thresholds, floor: float) -> np.ndarray:
    """P(X >= t_i) for a per-row threshold array (log-uniform inside bins)."""
    le = _log_edges(edges, floor)
    t = np.log10(np.maximum(np.asarray(thresholds, float), floor))
    t = np.broadcast_to(t, P.shape[:-1])
    lo, hi = le[:-1], le[1:]
    frac = np.clip((hi[None, :] - t.reshape(-1, 1)) / np.maximum(hi - lo, EPS)[None, :], 0.0, 1.0)
    return (P.reshape(-1, P.shape[-1]) * frac).sum(-1).reshape(P.shape[:-1])


def _quantiles(P: np.ndarray, le: np.ndarray, U: np.ndarray) -> np.ndarray:
    """log10 quantiles of each row's bin distribution at probabilities U (N, K)."""
    cdf = np.cumsum(P, -1)
    cdf[:, -1] = 1.0
    k = (cdf[:, None, :] < U[:, :, None]).sum(-1)
    k = np.clip(k, 0, P.shape[1] - 1)
    lo_cdf = np.where(k > 0, np.take_along_axis(cdf, np.maximum(k - 1, 0), 1), 0.0)
    pk = np.take_along_axis(P, k, 1)
    frac = np.clip((U - lo_cdf) / np.maximum(pk, EPS), 0.0, 1.0)
    return le[k] + frac * (le[k + 1] - le[k])


def _normal_pairs(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic standard-normal pairs (scrambled Sobol, fallback RNG)."""
    from scipy.stats import norm
    try:
        from scipy.stats import qmc
        u = qmc.Sobol(2, scramble=True, seed=seed).random(n)
    except Exception:  # pragma: no cover
        u = np.random.default_rng(seed).random((n, 2))
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return norm.ppf(u[:, 0]), norm.ppf(u[:, 1])


def copula_exceedance(PS: np.ndarray, PD: np.ndarray, edges, floor: float, threshold: float,
                      rho=0.0, n: int = 256, chunk: int = 20000, seed: int = 0) -> np.ndarray:
    """P(S + D >= threshold) under a Gaussian copula with correlation rho (scalar or (N,))."""
    from scipy.stats import norm
    PS = np.asarray(PS, float)
    PD = np.asarray(PD, float)
    N = PS.shape[0]
    le = _log_edges(edges, floor)
    z1, z2 = _normal_pairs(n, seed)
    r = np.broadcast_to(np.clip(np.asarray(rho, float), -0.999, 0.9999), (N,))
    out = np.empty(N)
    u1 = norm.cdf(z1)[None, :]
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        rr = r[a:b, None]
        u2 = norm.cdf(rr * z1[None, :] + np.sqrt(1 - rr ** 2) * z2[None, :])
        qs = _quantiles(PS[a:b], le, np.broadcast_to(u1, (b - a, n)))
        qd = _quantiles(PD[a:b], le, u2)
        out[a:b] = ((10 ** qs + 10 ** qd) >= threshold).mean(1)
    return out


def bounds(PS, PD, edges, floor, threshold) -> tuple[np.ndarray, np.ndarray]:
    lo = np.maximum(exceedance_probability(PS, edges, threshold, floor),
                    exceedance_probability(PD, edges, threshold, floor))
    hi = np.minimum(1.0, exceedance_probability(PS, edges, threshold / 2, floor)
                    + exceedance_probability(PD, edges, threshold / 2, floor))
    return lo, np.maximum(hi, lo)


def combine(PS, PD, edges, floor, threshold, method: str, rho=0.0, n: int = 256) -> np.ndarray:
    if method == "lower_bound":
        return bounds(PS, PD, edges, floor, threshold)[0]
    if method == "upper_bound":
        return bounds(PS, PD, edges, floor, threshold)[1]
    if method == "independence":
        return copula_exceedance(PS, PD, edges, floor, threshold, 0.0, n)
    if method == "comonotonic":
        return copula_exceedance(PS, PD, edges, floor, threshold, 0.9999, n)
    if method == "copula":
        return copula_exceedance(PS, PD, edges, floor, threshold, rho, n)
    raise ValueError(f"unknown PN combination {method!r}")


# ---------------------------------------------------------------------------
# rho from in-situ pairs
# ---------------------------------------------------------------------------

def season_of(dates) -> np.ndarray:
    m = pd.DatetimeIndex(pd.to_datetime(dates)).month
    return np.select([m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8])], ["DJF", "MAM", "JJA"], "SON")


def fit_rho(df: pd.DataFrame, s_col: str, d_col: str, by: str | None = "season",
            min_n: int = 30) -> dict[str, Any]:
    """Gaussian-copula rho = sin(pi tau_b / 2) from samples reporting both groups (ties allowed)."""
    from scipy.stats import kendalltau
    s = pd.to_numeric(df[s_col], errors="coerce")
    d = pd.to_numeric(df[d_col], errors="coerce")
    ok = np.isfinite(s) & np.isfinite(d)
    sub = df[ok].assign(_s=s[ok], _d=d[ok])

    def one(g):
        if len(g) < min_n:
            return None
        tau = kendalltau(g["_s"], g["_d"]).statistic
        return {"rho": float(np.sin(np.pi * tau / 2)), "tau_b": float(tau), "n": int(len(g))}

    out = {"all": one(sub)}
    if by == "season" and len(sub):
        for k, g in sub.groupby(season_of(sub["date"])):
            out[k] = one(g)
    return out


def rho_for(dates, rho_table: dict[str, Any]) -> np.ndarray:
    base = (rho_table.get("all") or {}).get("rho", 0.0)
    seas = season_of(dates)
    return np.array([(rho_table.get(s) or {}).get("rho", base) for s in seas], float)
