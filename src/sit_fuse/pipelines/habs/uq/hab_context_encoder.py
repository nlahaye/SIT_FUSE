"""
HAB context encoder: sit_fuse.train.sf_uq_utils.ContextEncoder with the
production HAB assignment rule added.

Production (utils.insitu_hab_to_multi_hist): for every cluster, histogram the
in-situ values of samples whose radius contains the cluster (one count per
sample), take the mode bin; if the mode is bin 0 but the other bins together
have at least as many samples, take the most frequent non-zero bin.

NOTE: production computes `mx1 = np.argmax(hist[1:])`, an index into the
slice, so the fallback returns (intended bin - 1). `replicate_production_offset:
true` reproduces that exactly (for equivalence checks against existing
products); the default applies the intended +1.

rule: histogram_mode is selected with
  context_encoder:
    assignment: {rule: histogram_mode, min_samples: 1, replicate_production_offset: false}
All other rules (greedy_agreement, fraction_threshold), the smoothed agreement
tables (A1/A2), the bootstrap assignment ensemble and transform() are
inherited unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sit_fuse.train.sf_uq_utils import ContextEncoder, _Table, assign_contexts


def histogram_mode_assign(n: np.ndarray, rule_cfg: dict[str, Any], rng=None) -> np.ndarray:
    """n: (K, C) per-cluster sample counts. Returns (K,) bin index (-1: unassigned)."""
    K, C = n.shape
    out = np.full(K, -1, dtype=np.int64)
    if K == 0:
        return out
    min_n = float(rule_cfg.get("min_samples", 1))
    offset = 0 if rule_cfg.get("replicate_production_offset", False) else 1
    tot = n.sum(axis=1)
    mx = np.argmax(n, axis=1)
    if rng is not None and rule_cfg.get("random_tie_break", True):
        # break ties at random inside the ensemble (deterministic argmax otherwise)
        jitter = rng.random(n.shape) * 1e-6
        mx = np.argmax(n + jitter * (n == n.max(axis=1, keepdims=True)), axis=1)
    fallback = (mx == 0) & (n[:, 1:].sum(axis=1) >= n[:, 0]) & (n[:, 1:].sum(axis=1) > 0)
    mx = np.where(fallback, np.argmax(n[:, 1:], axis=1) + offset, mx)
    ok = (n.max(axis=1) >= 1) & (tot >= min_n)
    out[ok] = mx[ok]
    return out


class HABContextEncoder(ContextEncoder):
    """ContextEncoder whose class axis is the ordinal concentration bin."""

    def _assign(self, lev1, lev2, prior, rng=None):
        rule = str(self.rule_cfg.get("rule", "histogram_mode")).lower()
        if rule != "histogram_mode":
            return super()._assign(lev1, lev2, prior, rng=rng)
        k1, n1, _ = lev1
        k2, n2, _ = lev2
        h1 = histogram_mode_assign(n1, self.rule_cfg, rng)
        h2 = histogram_mode_assign(n2, self.rule_cfg, rng)
        # unassigned joint clusters inherit the parent (Layer-1) assignment,
        # mirroring the tiered heir / no_heir merge of the production pipeline
        par = _Table(k1, np.eye(self.C)[np.maximum(h1, 0)][:, :]).lookup(
            k2 // self.num_child, np.eye(self.C)[0]).argmax(axis=1)
        h2 = np.where(h2 >= 0, h2, par)
        return np.maximum(h1, 0), h2

    # ------------------------------------------------------------------
    def production_assignment(self, counts: pd.DataFrame, level: str = "joint",
                              geotiff_scale: float = 1000.0) -> list[list[float]]:
        """
        The `algal` output of insitu_hab_to_multi_hist: per bin, the list of
        cluster GeoTIFF values (joint / 1000) assigned to it.
        """
        from sit_fuse.train.sf_uq_utils import _aggregate_counts
        k, n = _aggregate_counts(counts, level, self.C)
        h = histogram_mode_assign(n, {**self.rule_cfg, "rule": "histogram_mode"})
        algal = [[] for _ in range(self.C)]
        for key, c in zip(k, h):
            if c >= 0:
                algal[int(c)].append(float(key) / geotiff_scale)
        return algal


def default_encoder_cfg(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    base = {
        "alpha_parent": 5.0,       # HAB counts are samples, not pixels: smaller pseudo-counts
        "alpha_joint": 5.0,
        "assignment": {"rule": "histogram_mode", "min_samples": 1,
                       "replicate_production_offset": False},
        "assignment_ensemble": {"n_members": 100, "bootstrap_unit": "split_group",
                                "bootstrap_dates": True},
    }
    if cfg:
        for k, v in cfg.items():
            base[k] = {**base[k], **v} if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def fit_encoder(counts: pd.DataFrame, n_classes: int, num_child: int,
                cfg: dict[str, Any] | None = None, seed: int = 42) -> HABContextEncoder:
    enc = HABContextEncoder(n_classes, num_child, default_encoder_cfg(cfg))
    return enc.fit(counts.astype({"cluster": np.int64, "class": np.int64}), seed=seed)
