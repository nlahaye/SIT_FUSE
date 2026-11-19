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

Tier 2 (context_encoder.spatial_tier, hab_spatial_tier): fit(..., spatial=SpatialTable)
  - A1 / A2 shrink toward the spatial class fractions (blended with the base
    target by pixel mass, spatial share capped at max_weight) instead of the
    prior / the parent's A1
  - joint clusters without in-situ counts get a spatial hard assignment
    (top fraction >= threshold), before (spatial_first, production) or after
    (parent_first) their own parent; parents without counts likewise
  - transform adds assign_tier (0 joint, 1 parent, 2 spatial, 3 none; never a
    feature), assign_hard_t1_c (tier-1-only hard assignment, production baseline
    without tier 2) and the spatial_tier feature group sp_tier_c, sp_tier_mass.
Without spatial the encoder is unchanged apart from the assign_tier column.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sit_fuse.train.sf_uq_utils import EPS, ContextEncoder, _Table, _aggregate_counts, assign_contexts

TIER_NAMES = ("joint", "parent", "spatial", "none")


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


def blend_rows(Z, mass, base, prior_pixels: float, max_weight: float = 1.0):
    """w Z + (1 - w) base per row, w = min(m / (m + prior_pixels), max_weight)."""
    m = np.asarray(mass, float)
    w = np.minimum(m / (m + max(float(prior_pixels), 1e-12)), float(max_weight))[:, None]
    return w * Z + (1.0 - w) * base


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

    # --- tier 2 ----------------------------------------------------------
    def _sp(self):
        sp = getattr(self, "spatial", None)
        return sp, (getattr(self, "spatial_cfg", None) or {})

    def _agreement(self, counts, mult=None, unit="date"):
        sp, cfg = self._sp()
        if sp is None or not cfg.get("shrinkage_target", True):
            return super()._agreement(counts, mult, unit)
        beta = float(cfg.get("prior_pixels", 200.0))
        mw = float(cfg.get("max_weight", 0.5))
        k1, n1 = _aggregate_counts(counts, "parent", self.C, mult, unit)
        k2, n2 = _aggregate_counts(counts, "joint", self.C, mult, unit)
        tot = n1.sum()
        prior = n1.sum(axis=0) / tot if tot > 0 else np.full(self.C, 1.0 / self.C)
        prior = np.clip(prior, EPS, None)
        prior = prior / prior.sum()
        T1 = blend_rows(sp.parent(k1, prior), sp.parent_mass(k1), prior, beta, mw)
        A1 = (n1 + self.alpha_parent * T1) / (n1.sum(axis=1, keepdims=True) + self.alpha_parent)
        A1_par = _Table(k1, A1).lookup(k2 // self.num_child, prior)
        T2 = blend_rows(sp.joint(k2, A1_par), sp.joint_mass(k2), A1_par, beta, mw)
        A2 = (n2 + self.alpha_joint * T2) / (n2.sum(axis=1, keepdims=True) + self.alpha_joint)
        return prior, (k1, n1, A1), (k2, n2, A2)

    def fit(self, counts: pd.DataFrame, seed: int = 42, spatial=None, spatial_cfg=None):
        self.spatial = spatial
        self.spatial_cfg = dict(spatial_cfg or {})
        super().fit(counts, seed=seed)
        self.k1_seen = self.t_A1.keys.copy()
        self.k2_seen = self.t_A2.keys.copy()
        self.sp_joint_keys = np.zeros(0, np.int64)
        self.sp_parent_keys = np.zeros(0, np.int64)
        if spatial is not None:
            self.t_H_t1, self.t_H1_t1 = self.t_H, self.t_H1
            self._extend_with_spatial()
        return self

    def _extend_with_spatial(self):
        from sit_fuse.pipelines.habs.uq.hab_spatial_tier import decisive
        sp, cfg = self._sp()
        C, eye, beta = self.C, np.eye(self.C), float(cfg.get("prior_pixels", 200.0))
        mw = float(cfg.get("max_weight", 0.5))

        def add(tab, keys, vals):
            return _Table(np.r_[tab.keys, keys], np.vstack([tab.values, vals]))

        # parents without in-situ counts
        newp = sp.keys_p[~np.isin(sp.keys_p, self.k1_seen)]
        if newp.size:
            Zp, mp = sp.parent(newp, self.prior), sp.parent_mass(newp)
            self.t_A1 = add(self.t_A1, newp, blend_rows(Zp, mp, self.prior, beta, mw))
            hp = decisive(Zp, mp, cfg)
            ok = hp >= 0
            if ok.any():
                self.t_H1 = add(self.t_H1, newp[ok], eye[hp[ok]])
                self.t_V1 = add(self.t_V1, newp[ok], eye[hp[ok]])
                self.sp_parent_keys = newp[ok]
        # joint clusters without in-situ counts
        newj = sp.keys[~np.isin(sp.keys, self.k2_seen)]
        if newj.size:
            A1p = self.t_A1.lookup(newj // self.num_child, self.prior)
            Zj, mj = sp.joint(newj, A1p), sp.joint_mass(newj)
            self.t_A2 = add(self.t_A2, newj, blend_rows(Zj, mj, A1p, beta, mw))
            hj = decisive(Zj, mj, cfg)
            ok = hj >= 0
            if cfg.get("hard_priority", "spatial_first") == "parent_first":
                ok &= ~np.isin(newj // self.num_child, self.k1_seen)
            if ok.any():
                self.t_H = add(self.t_H, newj[ok], eye[hj[ok]])
                self.t_V = add(self.t_V, newj[ok], eye[hj[ok]])
                self.sp_joint_keys = newj[ok]

    def transform(self, df: pd.DataFrame, features_cfg: dict[str, Any]) -> pd.DataFrame:
        out = super().transform(df, features_cfg)
        joint = df["joint_label"].to_numpy().astype(np.int64)
        parent = df["parent_label"].to_numpy().astype(np.int64)
        k2 = getattr(self, "k2_seen", self.t_A2.keys)
        k1 = getattr(self, "k1_seen", self.t_A1.keys)
        sj = getattr(self, "sp_joint_keys", np.zeros(0, np.int64))
        spk = getattr(self, "sp_parent_keys", np.zeros(0, np.int64))
        # mirrors hard = t_H.lookup_rows(joint, t_H1.lookup(parent)) in ContextEncoder.transform
        tier = np.where(np.isin(joint, k2), 0, np.where(np.isin(joint, sj), 2, np.where(
            np.isin(parent, k1), 1, np.where(np.isin(parent, spk), 2, 3))))
        extra = {"assign_tier": tier.astype(np.float32)}
        sp, _ = self._sp()
        if sp is not None:
            C = self.C
            cls_range = range(C) if C > 2 else [1]
            H1 = self.t_H_t1.lookup_rows(joint, self.t_H1_t1.lookup(parent, np.eye(C)[0]))
            Z = sp.joint(joint, sp.parent(parent, self.prior))
            m = np.where(sp.joint_mass(joint) > 0, sp.joint_mass(joint), sp.parent_mass(parent))
            for c in cls_range:
                extra[f"assign_hard_t1_{c}"] = H1[:, c].astype(np.float32)
                extra[f"sp_tier_{c}"] = Z[:, c].astype(np.float32)
            extra["sp_tier_mass"] = np.log1p(m).astype(np.float32)
        ex = pd.DataFrame(extra, index=out.index)
        return pd.concat([out.drop(columns=[c for c in ex.columns if c in out.columns]), ex], axis=1)

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
                cfg: dict[str, Any] | None = None, seed: int = 42,
                spatial=None) -> HABContextEncoder:
    """spatial: optional hab_spatial_tier.SpatialTable (tier 2, cfg['spatial_tier'])."""
    from sit_fuse.pipelines.habs.uq.hab_spatial_tier import spatial_tier_config
    full = default_encoder_cfg(cfg)
    enc = HABContextEncoder(n_classes, num_child, full)
    sp_cfg = spatial_tier_config(full) if spatial is not None else None
    return enc.fit(counts.astype({"cluster": np.int64, "class": np.int64}), seed=seed,
                   spatial=spatial if sp_cfg else None, spatial_cfg=sp_cfg)
