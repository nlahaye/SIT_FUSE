"""
Tier-2 (spatial) context assignment for HAB UQ: the analogue of the production
dual-level matchup (run_multi_tier_zonal_histogram + run_multi_tier_class_compare).

Production
  tier 1  in-situ histogram assignment of joint (heir) and parent (no_heir) clusters;
          context assignment of a sensor group pools every platform's scenes
  tier 2  zonal histogram of each joint cluster over the instrument's merged daily
          parent-level (no_heir) class map (group platforms composited in
          platform_priority order, first valid pixel wins); a joint cluster with
          no tier-1 assignment takes the top class if it covers >= 51 % of its
          pixels (dbf_percentage_thresh), else it stays uncertain

UQ (uq.context_encoder)
  pool_sensor_group: true    encoder counts of a platform = union over its group's
                             platforms (one count per sample and cluster, as production)
  spatial_tier:
    enabled: true
    mode: composite          composite: production merged map (the platform's group,
                             priority order, first valid wins; includes the platform itself)
                             others: first valid *other* source at the same pixel
                             (excludes the platform itself; cross-platform evidence only)
    sources: group           group: platforms of the same sensor group
                             all: every UQ sensor (other grids reprojected, nearest)
    threshold: 0.51          production dbf_percentage_thresh
    min_pixels: 25           pixel mass needed for a spatial hard assignment
    prior_pixels: 200        pseudo-pixels when blending spatial fractions into
                             shrinkage targets: w = min(m / (m + prior_pixels), max_weight)
    max_weight: 0.5          cap on the spatial share of a shrinkage target (the rest is
                             the tier-1 target: prior for A1, the parent's A1 for A2)
    shrinkage_target: true   A1 / A2 shrink toward the spatial fractions
    hard_priority: auto      spatial_first (production: unseen joints use tier 2) |
                             parent_first (unseen joints use the own parent first);
                             auto = spatial_first for composite, parent_first for others
    days: train_roles        train_roles (all train-role days of the windows) | train_samples
    max_days: null           optional random subset of days (cost cap)

Leakage: co-occurrence counts carry no labels; classes come from parent-level
assignments fit on the same training counts as the encoder (fold-excluded for
cross-fit train rows). Only training days are read.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sit_fuse.train.sf_uq_utils import _aggregate_counts, head_valid_mask
from sit_fuse.pipelines.habs.hab_sensor_groups import SENSOR_GROUPS

LOG = logging.getLogger("hab-spatial-tier")

DEFAULTS = {"enabled": False, "mode": "composite", "sources": "group", "threshold": 0.51,
            "min_pixels": 25, "prior_pixels": 200.0, "max_weight": 0.5, "shrinkage_target": True,
            "hard_priority": "auto", "days": "train_roles", "max_days": None}


def spatial_tier_config(enc_cfg: dict[str, Any] | None) -> dict[str, Any] | None:
    cfg = (enc_cfg or {}).get("spatial_tier") or {}
    if not cfg.get("enabled", False):
        return None
    out = {**DEFAULTS, **cfg}
    if out["mode"] not in ("composite", "others"):
        raise ValueError(f"spatial_tier.mode must be composite|others, got {out['mode']}")
    if out["sources"] not in ("group", "all"):
        raise ValueError(f"spatial_tier.sources must be group|all, got {out['sources']}")
    if out["hard_priority"] == "auto":
        out["hard_priority"] = "spatial_first" if out["mode"] == "composite" else "parent_first"
    return out


def platform_group(sensor: str) -> str | None:
    for g, mem in SENSOR_GROUPS.items():
        if sensor in mem:
            return g
    return None


def group_order(sensor: str, present, platform_priority: dict | None = None) -> list[str]:
    """Platforms of the sensor's group present in this run, in merge priority order."""
    g = platform_group(sensor)
    if g is None:
        return [sensor]
    mem = list(SENSOR_GROUPS[g])
    pri = (platform_priority or {}).get(g) or []
    order = [p for p in pri if p in mem] + [m for m in mem if m not in pri]
    return [p for p in order if p in present]


# ---------------------------------------------------------------------------
# Pooled group counts (production context assignment of a sensor group)
# ---------------------------------------------------------------------------

def pool_group_counts(counts_by_sensor: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Encoder counts per platform = union over its group's platforms: a sample seen
    by several platforms counts once per cluster (max over platforms), as the
    production group histogram. Non-group sensors are returned unchanged.
    """
    out = {}
    for s, c in counts_by_sensor.items():
        mem = [m for m in group_order(s, counts_by_sensor) if m in counts_by_sensor]
        if len(mem) <= 1:
            out[s] = c
            continue
        allc = pd.concat([counts_by_sensor[m] for m in mem], ignore_index=True)
        if "sample_uid" not in allc.columns:
            out[s] = allc.assign(sensor=s)
            continue
        key = ["level", "cluster", "class", "sample_uid"]
        agg = {"count": "max"}
        for k in allc.columns:
            if k not in key and k != "count":
                agg[k] = "first"
        out[s] = allc.groupby(key, as_index=False, sort=False).agg(agg).assign(sensor=s)
    return out


# ---------------------------------------------------------------------------
# Full-scene co-occurrence of joint clusters with source parent clusters
# ---------------------------------------------------------------------------

def _label_arrays(search_dirs, sensor, date, uq_cfg, invalid=None):
    """(joint, parent, profile) int64 arrays (-1 = invalid) of one scene, or None."""
    from sit_fuse.pipelines.habs.uq.hab_uq_common import find_product
    mode = uq_cfg.get("input_mode", "auto")
    if mode in ("auto", "head_outputs"):
        p = find_product(search_dirs, "head_outputs", sensor, date, patterns=uq_cfg.get("file_patterns"))
        if p:
            import rasterio
            with rasterio.open(p) as src:
                desc = list(src.descriptions)
                idx = [desc.index(b) + 1 for b in ("joint_label", "parent_label")]
                arr = src.read(idx).astype(np.float32)
                prof = src.profile.copy()
            bands = {"joint_label": arr[0], "parent_label": arr[1]}
        elif mode == "head_outputs":
            return None
        else:
            bands = None
    else:
        bands = None
    if bands is None:
        from sit_fuse.pipelines.habs.uq.hab_uq_matchups import load_scene
        sc = load_scene(search_dirs, sensor, date, uq_cfg)
        if sc is None:
            return None
        bands, prof = sc[0], sc[1]
    valid = head_valid_mask(bands, invalid)
    j = np.where(valid, bands["joint_label"], -1).astype(np.int64)
    p = np.where(valid, bands["parent_label"], -1).astype(np.int64)
    return j, p, prof


def _same_grid(a, b) -> bool:
    return (a["width"] == b["width"] and a["height"] == b["height"]
            and np.allclose(tuple(a["transform"])[:6], tuple(b["transform"])[:6])
            and str(a.get("crs")) == str(b.get("crs")))


def _align(arr, src_prof, dst_prof):
    """Source id array on the target grid (nearest neighbour, -1 outside)."""
    if _same_grid(src_prof, dst_prof):
        return arr
    from rasterio.warp import Resampling, reproject
    out = np.full((dst_prof["height"], dst_prof["width"]), -1, dtype=np.int32)
    reproject(arr.astype(np.int32), out, src_transform=src_prof["transform"], src_crs=src_prof["crs"],
              dst_transform=dst_prof["transform"], dst_crs=dst_prof["crs"],
              src_nodata=-1, dst_nodata=-1, resampling=Resampling.nearest)
    return out.astype(np.int64)


def source_order(target: str, sensors, cfg: dict[str, Any], platform_priority=None) -> list[str]:
    grp = group_order(target, sensors, platform_priority)
    if cfg["sources"] == "all":
        grp = grp + [s for s in sensors if s not in grp]
    if cfg["mode"] == "others":
        grp = [s for s in grp if s != target]
    return grp


def scene_cooccurrence(target: str, scenes: dict, order: list[str]) -> pd.DataFrame | None:
    """
    scenes: {sensor: (joint, parent, profile)} for one day. Returns joint, src,
    parent, n for the target's valid pixels covered by the first valid source.
    """
    if target not in scenes:
        return None
    jt, _, prof = scenes[target]
    src = np.full(jt.shape, -1, dtype=np.int64)
    par = np.full(jt.shape, -1, dtype=np.int64)
    for i, o in enumerate(order):
        if o not in scenes:
            continue
        po = scenes[o][1] if o == target else _align(scenes[o][1], scenes[o][2], prof)
        fill = (src < 0) & (po >= 0)
        src[fill] = i
        par[fill] = po[fill]
    ok = (jt >= 0) & (src >= 0)
    if not ok.any():
        return None
    d = pd.DataFrame({"joint": jt[ok], "src": src[ok], "parent": par[ok]})
    d = d.groupby(["joint", "src", "parent"], sort=False).size().rename("n").reset_index()
    d["src"] = np.asarray(order, dtype=object)[d["src"].to_numpy()]
    return d


def collect_cooccurrence(sensors, dirs_by_sensor: dict, days, uq_cfg: dict[str, Any],
                         cfg: dict[str, Any], invalid=None, platform_priority=None) -> dict[str, pd.DataFrame]:
    """{target sensor: DataFrame(joint, src, parent, n)} summed over the given days."""
    sensors = list(sensors)
    orders = {s: source_order(s, sensors, cfg, platform_priority) for s in sensors}
    need = sorted({o for s in sensors for o in orders[s]} | set(sensors))
    acc = {s: [] for s in sensors}
    n_days = 0
    for d in days:
        scenes = {}
        for s in need:
            r = _label_arrays(dirs_by_sensor.get(s) or [], s, d, uq_cfg, invalid)
            if r is not None:
                scenes[s] = r
        if not scenes:
            continue
        n_days += 1
        for s in sensors:
            c = scene_cooccurrence(s, scenes, orders[s])
            if c is not None:
                acc[s].append(c)
    out = {}
    for s in sensors:
        out[s] = (pd.concat(acc[s]).groupby(["joint", "src", "parent"], as_index=False)["n"].sum()
                  if acc[s] else pd.DataFrame(columns=["joint", "src", "parent", "n"]))
        LOG.info("spatial tier %s: %d days, %d joint clusters, sources %s", s, n_days,
                 out[s]["joint"].nunique(), orders[s])
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def parent_assignment(counts: pd.DataFrame, n_classes: int, rule_cfg: dict[str, Any]):
    """(parent keys, bin or -1) from the production histogram-mode rule."""
    from sit_fuse.pipelines.habs.uq.hab_context_encoder import histogram_mode_assign
    if counts is None or counts.empty:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    c = counts.astype({"cluster": np.int64, "class": np.int64})
    k, n = _aggregate_counts(c, "parent", n_classes)
    return np.asarray(k, np.int64), histogram_mode_assign(n, {**rule_cfg, "rule": "histogram_mode"})


@dataclass
class SpatialTable:
    """Per joint (and parent) cluster: class fractions Z over assigned co-occurring pixels."""
    keys: np.ndarray
    Z: np.ndarray
    mass: np.ndarray
    keys_p: np.ndarray
    Zp: np.ndarray
    mass_p: np.ndarray

    @staticmethod
    def _lookup(keys, vals, ids, default):
        ids = np.asarray(ids, np.int64)
        out = np.broadcast_to(default, ids.shape + default.shape[-1:]).copy() if default.ndim \
            else np.full(ids.shape, default, dtype=np.float64)
        if keys.size == 0:
            return out
        pos = np.clip(np.searchsorted(keys, ids), 0, keys.size - 1)
        hit = (keys[pos] == ids) & (ids >= 0)
        out[hit] = vals[pos[hit]]
        return out

    def joint(self, ids, default):
        return self._lookup(self.keys, self.Z, ids, default)

    def joint_mass(self, ids):
        return self._lookup(self.keys, self.mass, ids, np.float64(0.0))

    def parent(self, ids, default):
        return self._lookup(self.keys_p, self.Zp, ids, default)

    def parent_mass(self, ids):
        return self._lookup(self.keys_p, self.mass_p, ids, np.float64(0.0))


def _fractions(keys, cls, n, C):
    u, inv = np.unique(keys, return_inverse=True)
    M = np.zeros((u.size, C))
    np.add.at(M, (inv, cls), n)
    m = M.sum(1)
    return u.astype(np.int64), M / np.maximum(m[:, None], 1e-12), m


def spatial_table(cooc: pd.DataFrame, assignments: dict[str, tuple[np.ndarray, np.ndarray]],
                  n_classes: int, num_child: int) -> SpatialTable | None:
    """Map each source parent to its assigned bin (unassigned dropped) and normalize per joint."""
    if cooc is None or cooc.empty:
        return None
    parts = []
    for o, d in cooc.groupby("src"):
        k, h = assignments.get(o, (np.zeros(0, np.int64), np.zeros(0, np.int64)))
        if k.size == 0:
            continue
        order = np.argsort(k)
        k, h = k[order], h[order]
        pid = d["parent"].to_numpy(np.int64)
        pos = np.clip(np.searchsorted(k, pid), 0, k.size - 1)
        cls = np.where(k[pos] == pid, h[pos], -1)
        ok = cls >= 0
        parts.append((d["joint"].to_numpy(np.int64)[ok], cls[ok], d["n"].to_numpy(np.float64)[ok]))
    if not parts:
        return None
    j = np.concatenate([p[0] for p in parts])
    c = np.concatenate([p[1] for p in parts])
    n = np.concatenate([p[2] for p in parts])
    if j.size == 0:
        return None
    kj, Zj, mj = _fractions(j, c, n, n_classes)
    kp, Zp, mp = _fractions(j // num_child, c, n, n_classes)
    return SpatialTable(kj, Zj, mj, kp, Zp, mp)


def decisive(Z, mass, cfg) -> np.ndarray:
    """Production class_compare rule: top fraction >= threshold (and enough pixels); -1 otherwise."""
    if Z.size == 0:
        return np.zeros(0, np.int64)
    top = Z.max(1)
    ok = (top >= float(cfg["threshold"])) & (mass >= float(cfg["min_pixels"]))
    return np.where(ok, Z.argmax(1), -1).astype(np.int64)


def spatial_days(samples: pd.DataFrame, date_roles: dict | None, windows, cfg: dict[str, Any],
                 seed: int = 42) -> list[pd.Timestamp]:
    """Training days read for the spatial tier."""
    days = None
    if cfg["days"] == "train_roles" and date_roles:
        days = sorted({pd.Timestamp(pd.to_datetime(k, format="mixed")).normalize()
                       for k, v in date_roles.items() if str(v).lower() == "train"})
    if not days:
        tr = samples[samples["split"] == "train"]
        days = sorted(pd.to_datetime(tr["date"]).dt.normalize().unique())
    if windows:
        days = [d for d in days if any(a <= d <= b for a, b in windows)]
    if cfg.get("max_days") and len(days) > int(cfg["max_days"]):
        rng = np.random.default_rng(seed)
        days = sorted(rng.choice(np.array(days, dtype=object), int(cfg["max_days"]), replace=False))
    return [pd.Timestamp(d) for d in days]
