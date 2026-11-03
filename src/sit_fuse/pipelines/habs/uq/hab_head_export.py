"""
Encoder-side probabilistic outputs for the HAB pipeline.

Extends sit_fuse.train.export_head_outputs (reused unchanged for the nominal
head bands) with two optional band groups that are computed in the same pass:

  MC input noise (head_export.mc)
      M forward passes with x + N(0, sigma^2) in the model's scaled input
      space (default sigma = DBN gauss_noise_stdev, 0.01). Bands:
        mc_parent_stability    fraction of passes whose argmax parent equals
                               the nominal parent (assignment stability)
        mc_joint_stability     same for the joint (parent, child) label
                               (mc.joint: true; evaluates child heads)
        mc_parent_entropy      normalized entropy of the pass-averaged parent
                               distribution (total)
        mc_expected_entropy    mean per-pass normalized entropy (aleatoric)
        mc_mutual_information  difference of the two (sensitivity / epistemic)
        mc_parent_maxprob      max of the pass-averaged parent distribution

  Embedding-distance OOD (head_export.ood)
      Per-parent mean and diagonal variance of encoder embeddings, fit on a
      sample of the encoder's training scenes (EmbeddingStats, saved per model
      as .npz). Bands:
        ood_embed_dist_parent  mean_d (e - mu_k)^2 / var_k for the nominal
                               parent k (shrunk toward global stats for rare
                               parents); ~1 in distribution
        ood_embed_dist_global  same against global stats
        ood_embed_pctl         empirical CDF of ood_embed_dist_parent on the
                               training sample (0.99 = beyond 99% of training)

All bands go into the same {PREFIX}.{date}.L3m.DAY.head_outputs.tif, after
the standard head_band_names() bands; files without them still read.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger("hab-head-export")

MC_BANDS = ["mc_parent_stability", "mc_joint_stability", "mc_parent_entropy",
            "mc_expected_entropy", "mc_mutual_information", "mc_parent_maxprob"]
OOD_BANDS = ["ood_embed_dist_parent", "ood_embed_dist_global", "ood_embed_pctl"]

BAND_DESCRIPTIONS = {
    "mc_parent_stability": "Fraction of MC input-noise passes keeping the nominal parent cluster",
    "mc_joint_stability": "Fraction of MC input-noise passes keeping the nominal joint cluster",
    "mc_parent_entropy": "Normalized entropy of the MC-averaged parent distribution",
    "mc_expected_entropy": "Mean normalized parent entropy over MC passes",
    "mc_mutual_information": "MC mutual information (parent-assignment sensitivity to input noise)",
    "mc_parent_maxprob": "Max probability of the MC-averaged parent distribution",
    "ood_embed_dist_parent": "Diagonal Mahalanobis distance to the nominal parent's training embeddings (per dim)",
    "ood_embed_dist_global": "Diagonal Mahalanobis distance to global training embeddings (per dim)",
    "ood_embed_pctl": "Training-sample percentile of ood_embed_dist_parent",
}


def extra_band_names(ex_cfg: dict[str, Any]) -> list[str]:
    names = []
    mc = (ex_cfg or {}).get("mc", {}) or {}
    if int(mc.get("samples", 0)) > 0:
        names += [b for b in MC_BANDS if b != "mc_joint_stability" or mc.get("joint", True)]
    if ((ex_cfg or {}).get("ood", {}) or {}).get("enabled", False):
        names += OOD_BANDS
    return names


def _torch():
    import torch
    return torch


def _norm_entropy(P):
    torch = _torch()
    Pc = torch.clamp(P, min=1e-12)
    return -(Pc * torch.log(Pc)).sum(dim=1) / np.log(max(P.shape[1], 2))


def _root_probs(model, x):
    from sit_fuse.train.export_head_outputs import _as_prob_rows, _first
    return _as_prob_rows(_first(model.pretrained_model.forward(x)))


def _embed(model, x):
    from sit_fuse.train.export_head_outputs import _as_embed_rows, _first
    out = model.forward(x, return_embed=True)
    return None if out is None else _as_embed_rows(_first(out[2]))


def _joint_argmax(model, P, E):
    """Joint label child + num_child * parent, as Heir_DC.forward assigns it."""
    torch = _torch()
    from sit_fuse.train.export_head_outputs import _ensure_probs, _first
    parent = P.argmax(1)
    child = torch.zeros_like(parent)
    tree = model.clust_tree["1"]
    for pid in torch.unique(parent).tolist():
        head = tree.get(str(int(pid)))
        if head is None:
            continue
        idx = torch.where(parent == pid)[0]
        child[idx] = _ensure_probs(_first(head.forward(E[idx]))).argmax(1)
    return child + int(model.num_classes) * parent


def input_noise_sigma(ex_cfg: dict[str, Any], sf_yaml: dict[str, Any] | None, n_chan: int):
    """Per-channel sigma in the scaled input space."""
    mc = (ex_cfg or {}).get("mc", {}) or {}
    s = mc.get("input_noise")
    if s is None:
        # same noise the clustering heads were trained with
        cl = (sf_yaml or {}).get("cluster", {}) or {}
        s = (cl.get("heir", {}) or {}).get("gauss_noise_stdev") or cl.get("gauss_noise_stdev") or [0.01]
    s = np.atleast_1d(np.asarray(s, dtype=np.float32))
    if s.size == 1:
        s = np.repeat(s, n_chan)
    if s.size != n_chan:
        raise ValueError(f"head_export.mc.input_noise has {s.size} values for {n_chan} channels")
    return s


def mc_bands(model, x, P0, joint0, sigma, ex_cfg, generator=None) -> dict[str, np.ndarray]:
    """MC input-noise summaries for one batch. x: (N, C) or (B, C, h, w)."""
    torch = _torch()
    mc = ex_cfg.get("mc", {}) or {}
    M = int(mc.get("samples", 0))
    want_joint = bool(mc.get("joint", True))
    sig = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
    shape = [1, -1] + [1] * (x.ndim - 2)
    sig = sig.reshape(shape)
    parent0 = P0.argmax(1)
    Psum = torch.zeros_like(P0)
    Hsum = torch.zeros(P0.shape[0], device=P0.device)
    keep_p = torch.zeros(P0.shape[0], device=P0.device)
    keep_j = torch.zeros(P0.shape[0], device=P0.device)
    for _ in range(M):
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype) if generator \
            is not None else torch.randn_like(x)
        xm = x + noise * sig
        Pm = _root_probs(model, xm)
        Psum += Pm
        Hsum += _norm_entropy(Pm)
        keep_p += (Pm.argmax(1) == parent0).float()
        if want_joint:
            Em = _embed(model, xm)
            keep_j += (_joint_argmax(model, Pm, Em) == joint0).float()
    Pbar = Psum / M
    Htot = _norm_entropy(Pbar)
    Hexp = Hsum / M
    out = {"mc_parent_stability": keep_p / M}
    if want_joint:
        out["mc_joint_stability"] = keep_j / M
    out.update({"mc_parent_entropy": Htot, "mc_expected_entropy": Hexp,
                "mc_mutual_information": torch.clamp(Htot - Hexp, min=0.0),
                "mc_parent_maxprob": Pbar.max(1).values})
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Embedding statistics (OOD)
# ---------------------------------------------------------------------------

class EmbeddingStats:
    """
    Per-parent diagonal Gaussian over encoder embeddings.

    Accumulate with update(E, parent) over training batches, finalize(), then
    optionally calibrate_quantiles(E, parent) on a (second) training sample.
    Rare parents are shrunk toward global stats with pseudo-count `shrinkage`.
    """

    def __init__(self, n_parent: int, dim: int, shrinkage: float = 50.0, var_floor: float = 1e-6):
        self.n_parent, self.dim = int(n_parent), int(dim)
        self.shrinkage, self.var_floor = float(shrinkage), float(var_floor)
        self.n = np.zeros(self.n_parent, np.float64)
        self.s1 = np.zeros((self.n_parent, self.dim), np.float64)
        self.s2 = np.zeros((self.n_parent, self.dim), np.float64)
        self.mu = self.var = self.mu_g = self.var_g = None
        self.quantiles = None

    def update(self, E: np.ndarray, parent: np.ndarray) -> None:
        E = np.asarray(E, np.float64)
        parent = np.asarray(parent, np.int64)
        np.add.at(self.n, parent, 1.0)
        np.add.at(self.s1, parent, E)
        np.add.at(self.s2, parent, E * E)

    def finalize(self) -> "EmbeddingStats":
        N = max(self.n.sum(), 1.0)
        self.mu_g = self.s1.sum(0) / N
        self.var_g = np.maximum(self.s2.sum(0) / N - self.mu_g ** 2, self.var_floor)
        n = self.n[:, None]
        lam = self.shrinkage
        mu_k = (self.s1 + lam * self.mu_g) / (n + lam)
        ex2 = (self.s2 + lam * (self.var_g + self.mu_g ** 2)) / (n + lam)
        self.mu = mu_k
        self.var = np.maximum(ex2 - mu_k ** 2, self.var_floor)
        return self

    def distances(self, E: np.ndarray, parent: np.ndarray):
        E = np.asarray(E, np.float64)
        parent = np.clip(np.asarray(parent, np.int64), 0, self.n_parent - 1)
        d_p = (((E - self.mu[parent]) ** 2) / self.var[parent]).mean(1)
        d_g = (((E - self.mu_g) ** 2) / self.var_g).mean(1)
        return d_p.astype(np.float32), d_g.astype(np.float32)

    def calibrate_quantiles(self, E: np.ndarray, parent: np.ndarray, n_q: int = 1001) -> None:
        d_p, _ = self.distances(E, parent)
        self.quantiles = np.quantile(d_p, np.linspace(0, 1, n_q)).astype(np.float64)

    def percentile(self, d: np.ndarray) -> np.ndarray:
        if self.quantiles is None:
            return np.full(np.shape(d), np.nan, np.float32)
        q = self.quantiles
        return (np.searchsorted(q, d, side="right") / len(q)).astype(np.float32)

    def bands(self, E: np.ndarray, parent: np.ndarray) -> dict[str, np.ndarray]:
        d_p, d_g = self.distances(E, parent)
        return {"ood_embed_dist_parent": d_p, "ood_embed_dist_global": d_g,
                "ood_embed_pctl": self.percentile(d_p)}

    def save(self, path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, n=self.n, mu=self.mu, var=self.var, mu_g=self.mu_g, var_g=self.var_g,
                            quantiles=np.array([]) if self.quantiles is None else self.quantiles,
                            meta=np.array([self.n_parent, self.dim, self.shrinkage, self.var_floor]))
        return str(path)

    @classmethod
    def load(cls, path) -> "EmbeddingStats":
        z = np.load(path)
        n_parent, dim, shrink, floor = z["meta"]
        st = cls(int(n_parent), int(dim), float(shrink), float(floor))
        st.n, st.mu, st.var, st.mu_g, st.var_g = z["n"], z["mu"], z["var"], z["mu_g"], z["var_g"]
        st.quantiles = z["quantiles"] if z["quantiles"].size else None
        return st


# ---------------------------------------------------------------------------
# One batch -> all bands
# ---------------------------------------------------------------------------

def batch_bands(model, x, ex_cfg, sigma=None, emb_stats: EmbeddingStats | None = None,
                generator=None) -> np.ndarray | None:
    """
    (N_pixels, n_bands) rows: head_band_names() order, then extra_band_names().
    """
    from sit_fuse.train.export_head_outputs import head_distributions
    res = head_distributions(model, x, ex_cfg, return_aux=True)
    if res is None:
        return None
    rows, P, E = res
    names = extra_band_names(ex_cfg)
    if not names:
        return rows
    extra = {}
    if any(n.startswith("mc_") for n in names):
        torch = _torch()
        joint0 = torch.as_tensor(rows[:, 0], device=P.device).long()
        with torch.no_grad():
            extra.update(mc_bands(model, x, P, joint0, sigma, ex_cfg, generator))
    if any(n.startswith("ood_") for n in names):
        if emb_stats is None:
            raise RuntimeError("head_export.ood.enabled but no embedding statistics were fit/loaded")
        extra.update(emb_stats.bands(E.detach().cpu().numpy(), rows[:, 1].astype(np.int64)))
    cols = [extra[n] for n in names]
    return np.concatenate([rows, np.stack(cols, axis=1).astype(np.float32)], axis=1)
