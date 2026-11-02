#!/usr/bin/env python3
"""
Shared utilities for SIT-FUSE hierarchical-head uncertainty quantification.

Used by:
  export_head_outputs.py        (writes the head-output rasters read here)
  build_uq_feature_table.py     (feature table + full-scene cluster counts)
  simple_classifiers_uq.py      (soft targets, assignment ensemble, CV, calibration)
  classifier_inference_uq.py    (per-scene prediction + UQ rasters)

Conventions
-----------
* Class indices 0..C-1 follow target.classes in the YAML. Index 0 is the
  background / non-target class.
* Joint (hierarchical) cluster ID follows SIT-FUSE Heir_DC:
      joint = child + num_child_classes * parent
  and joint = num_child_classes * parent when a parent has no child head.
* No ambiguity masking is applied. Only NoData pixels (no SIT-FUSE output or
  no reference coverage) are excluded.

Generic helpers come from sit_fuse.train.classifier_utils (YAML/JSON I/O,
matchup normalization, raster read + collocation, FeatureBuilder multiscale
features, feature-config normalization and signatures, entropy / margin,
forest and bootstrap member probabilities). This module only adds what the
hierarchical-head UQ workflow needs on top of them.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from scipy.optimize import minimize, minimize_scalar

# Pre-existing SIT-FUSE helpers (reused rather than re-implemented)
from sit_fuse.train.classifier_utils import (
    DEFAULT_NODATA,
    FeatureBuilder,
    _canonicalize_config_value,
    bootstrap_model_probabilities,
    canonicalize_dates,
    collocate_annotation_to_cluster_grid,
    forest_tree_probabilities,
    get_training_matchups,
    load_yaml,
    margin_uncertainty,
    normalize_feature_config,
    predictive_entropy as _predictive_entropy,
    read_raster,
    save_json as _save_json,
)

LOG = logging.getLogger("sf-uq-utils")

EPS = 1e-7
HEAD_NODATA = DEFAULT_NODATA
OUT_NODATA = DEFAULT_NODATA
FORMAT_VERSION = "sitfuse-head-outputs-v1"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _arrays_to_lists(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _arrays_to_lists(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_arrays_to_lists(v) for v in value]
    return value


def save_json(path, payload: dict[str, Any]) -> None:
    """classifier_utils.save_json after numpy/Path canonicalization."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _save_json(Path(path), _canonicalize_config_value(_arrays_to_lists(payload)))


def write_table(df: pd.DataFrame, path) -> Path:
    """Write parquet when pyarrow is available, otherwise CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except (ImportError, ValueError):
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
    return path


def read_table(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        alt = path.with_suffix(".csv") if path.suffix == ".parquet" else None
        if alt is not None and alt.exists():
            path = alt
        else:
            raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


SPLIT_META = ("split_group", "classifier_split", "encoder_split", "split_stratum")


def load_matchups(path, require_truth: bool = True, splits=None) -> list[dict[str, Any]]:
    """
    Matchup records from the HMS / ADP / FDCC matchup generators.

    Records need `head_outputs` (added by the generators when
    `head_outputs.directory` is set, or by hand). With require_truth, records
    are normalized by classifier_utils.get_training_matchups, which reads
    `truth_hms` or the generic `annotation_raster` alias the generators add
    for ADP / FDCC.

    Identity and grouping (generator split fields, when present):
      pair_id  <- scene_id  (unique raster identity; the generator pair_id is a
                             grouping key and may be shared by several scenes)
      date     <- split_date (truth date), else the normalized truth date
      split_group / classifier_split / encoder_split / split_stratum
               <- carried through; defaults: date / "" / "unknown" / ""
    splits: optional iterable of classifier_split values to keep.
    """
    doc = load_yaml(path)
    raw = doc.get("matchups", doc.get("rasters", []))
    if not raw:
        raise ValueError(f"No matchup records found in {path}")
    for rec in raw:
        if "head_outputs" not in rec:
            raise KeyError(f"Matchup record is missing 'head_outputs': {rec.get('sit_fuse_context_free')}")

    if require_truth:
        norms = get_training_matchups(doc)
        dates = [r.get("split_date") or n["hms_date"].strftime("%Y-%m-%d") for r, n in zip(raw, norms)]
        truths = [str(n["annotation_raster"]) for n in norms]
        gids = [n["pair_id"] for n in norms]
    else:
        dates = list(canonicalize_dates(pd.Series([
            r.get("split_date") or r.get("hmsdate") or r.get("hms_date") or r.get("adp_smoke_date")
            or r.get("fdcc_date") or r.get("sit_fuse_timestamp") or r.get("date")
            for r in raw]), "date"))
        truths = [r.get("annotation_raster") or r.get("truth_hms") for r in raw]
        gids = [str(r.get("pairid") or r.get("pair_id") or f"pair_{i:05d}") for i, r in enumerate(raw)]

    out = []
    for rec, date, truth, gid in zip(raw, dates, truths, gids):
        date = str(date)[:10]
        r = {
            "pair_id": str(rec.get("scene_id") or gid),
            "group_id": gid,
            "date": date,
            "product": str(rec.get("product") or "").lower(),
            "truth": truth,
            "head_outputs": rec["head_outputs"],
            "split_group": str(rec.get("split_group") or date),
            "classifier_split": str(rec.get("classifier_split") or ""),
            "encoder_split": str(rec.get("encoder_split") or "unknown"),
            "split_stratum": str(rec.get("split_stratum") or ""),
            # attach_references.py: {name: raster} from other reference products
            "additional_references": dict(rec.get("additional_references") or {}),
        }
        if splits is None or r["classifier_split"] in set(splits):
            out.append(r)

    dup = [k for k, v in pd.Series([r["pair_id"] for r in out]).value_counts().items() if v > 1]
    if dup:
        raise ValueError(
            f"Duplicate scene identities {dup[:5]}: several scenes share a pair_id. "
            "Regenerate matchups with the split section enabled (adds scene_id)."
        )
    return out


# ---------------------------------------------------------------------------
# Head-output rasters
# ---------------------------------------------------------------------------

def head_band_names(top_k_parent: int, top_k_child: int, n_parent_ranks: int) -> list[str]:
    """Band order written by export_head_outputs.py."""
    names = ["joint_label", "parent_label", "child_label", "child_head_present"]
    for i in range(1, top_k_parent + 1):
        names += [f"parent_top{i}_id", f"parent_top{i}_prob"]
    names += ["parent_topk_mass", "parent_maxprob", "parent_margin", "parent_entropy"]
    for r in range(1, n_parent_ranks + 1):
        for j in range(1, top_k_child + 1):
            names += [f"r{r}_child_top{j}_id", f"r{r}_child_top{j}_prob"]
        names += [f"r{r}_child_topk_mass"]
    names += ["child_maxprob", "child_margin", "child_entropy", "joint_prob"]
    return names


def read_head_outputs(path) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, str]]:
    """Read a head-output GeoTIFF into {band_name: 2-D float32 array}."""
    import rasterio

    with rasterio.open(path) as src:
        descriptions = list(src.descriptions)
        if any(d is None for d in descriptions):
            raise ValueError(
                f"{path}: every band needs a description. "
                "Was this file written by export_head_outputs.py?"
            )
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
        tags = src.tags()

    if tags.get("format_version") != FORMAT_VERSION:
        LOG.warning("%s: format_version=%s, expected %s",
                    path, tags.get("format_version"), FORMAT_VERSION)

    bands = {name: data[i] for i, name in enumerate(descriptions)}
    return bands, profile, tags


# ---------------------------------------------------------------------------
# Excluded ("bad") clusters
# ---------------------------------------------------------------------------

class InvalidClusters:
    """
    Clusters whose pixels are excluded everywhere downstream (features,
    agreement tables, targets, training, calibration, products, evaluation).

    Config (`invalid_clusters`, in the UQ config; the matchup generators
    accept the same block or `from_uq_config`):
      joint:  values as in SIT-FUSE cluster GeoTIFFs (joint_label / 1000,
              e.g. 73.201) - converted with round(v * geotiff_scale)
      joint_labels: integer joint labels as written by export_head_outputs.py
      parent: parent (root-head) cluster IDs; every child is excluded
      from_context_yaml: {path, index | name} - reuse a SIT-FUSE
              postprocessing context list (context.clusters[index]; `name`
              matches context.name when it is a list)
      geotiff_scale: 1000
    Only artifact / non-physical clusters belong here (fill, scan edge,
    striping, glint, bad geolocation); ambiguous clusters stay in.
    """

    def __init__(self, joint=(), parent=(), source=None):
        self.joint = frozenset(int(v) for v in joint)
        self.parent = frozenset(int(v) for v in parent)
        self.source = source or {}

    def __bool__(self):
        return bool(self.joint or self.parent)

    def mask(self, joint_label: np.ndarray, parent_label: np.ndarray | None = None) -> np.ndarray:
        """True where the pixel belongs to an excluded cluster."""
        jl = np.asarray(joint_label)
        fin = np.isfinite(jl)
        out = np.zeros(jl.shape, dtype=bool)
        if self.joint:
            out |= fin & np.isin(np.where(fin, jl, -1).astype(np.int64), list(self.joint))
        if self.parent and parent_label is not None:
            pl = np.asarray(parent_label)
            pf = np.isfinite(pl)
            out |= pf & np.isin(np.where(pf, pl, -1).astype(np.int64), list(self.parent))
        return out

    def signature(self) -> dict[str, Any]:
        return {"joint": sorted(self.joint), "parent": sorted(self.parent)}

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "InvalidClusters":
        """Resolve from a full config (top-level `invalid_clusters`) or the block itself."""
        if not cfg:
            return cls()
        block = cfg.get("invalid_clusters", cfg) if isinstance(cfg, dict) else {}
        if not block or not isinstance(block, dict):
            return cls()
        if block.get("from_uq_config"):
            return cls.from_config(load_yaml(block["from_uq_config"]))
        scale = float(block.get("geotiff_scale", 1000))
        joint = {int(round(float(v) * scale)) for v in block.get("joint") or []}
        joint |= {int(v) for v in block.get("joint_labels") or []}
        parent = {int(v) for v in block.get("parent") or []}
        src = {}
        ctx = block.get("from_context_yaml")
        if ctx:
            doc = load_yaml(ctx["path"])["context"]
            lists = doc["clusters"]
            if "name" in ctx:
                names = doc.get("name")
                names = names if isinstance(names, list) else [names]
                idx = names.index(ctx["name"])
            else:
                idx = int(ctx.get("index", 0))
            vals = lists[idx]
            joint |= {int(round(float(v) * scale)) for v in vals}
            src = {"context_yaml": str(ctx["path"]), "index": idx, "n_values": len(vals)}
        return cls(joint, parent, src)


def head_valid_mask(bands: dict[str, np.ndarray], invalid: InvalidClusters | None = None) -> np.ndarray:
    """Valid head pixels; pixels of excluded clusters are removed when `invalid` is given."""
    lab = bands["joint_label"]
    valid = np.isfinite(lab) & (lab != HEAD_NODATA) & (lab >= 0)
    if invalid:
        valid &= ~invalid.mask(lab, bands.get("parent_label"))
    return valid


def excluded_cluster_mask(bands: dict[str, np.ndarray], invalid: InvalidClusters | None) -> np.ndarray:
    """1 = head-valid pixel of an excluded cluster, 0 = head-valid and kept, NaN = no head data."""
    lab = bands["joint_label"]
    base = np.isfinite(lab) & (lab != HEAD_NODATA) & (lab >= 0)
    ex = invalid.mask(lab, bands.get("parent_label")) if invalid else np.zeros(lab.shape, bool)
    return np.where(base, ex.astype(np.float32), np.nan)


def head_settings_from_tags(tags: dict[str, str]) -> dict[str, int]:
    return {
        "num_child_classes": int(tags["num_child_classes"]),
        "num_parent_classes": int(tags["num_parent_classes"]),
        "top_k_parent": int(tags["top_k_parent"]),
        "top_k_child": int(tags["top_k_child"]),
        "n_parent_ranks": int(tags["n_parent_ranks"]),
    }


# ---------------------------------------------------------------------------
# Reference rasters
# ---------------------------------------------------------------------------

def map_reference_classes(values: np.ndarray, nodata_mask: np.ndarray,
                          target_cfg: dict[str, Any]) -> np.ndarray:
    """
    Map raw reference values to class indices; -1 marks NoData.

    target:
      classes: [0, 1]                  # class index order (0 = background)
      value_map:                       # raw reference value -> class index
        0: 0
        1: 1
      unmapped: background | nodata    # how to treat values not in value_map
    """
    value_map = {float(k): int(v) for k, v in target_cfg["value_map"].items()}
    unmapped = str(target_cfg.get("unmapped", "nodata")).lower()

    out = np.full(values.shape, -1, dtype=np.int16)
    mapped = np.zeros(values.shape, dtype=bool)
    for raw, cls in value_map.items():
        sel = values == raw
        out[sel] = cls
        mapped |= sel

    if unmapped == "background":
        out[~mapped & ~nodata_mask] = 0
    elif unmapped != "nodata":
        raise ValueError("target.unmapped must be 'background' or 'nodata'.")

    out[nodata_mask] = -1
    return out


def resolve_reference_nodata(prof: dict[str, Any], target_cfg: dict[str, Any],
                             truth_path=None) -> tuple[float, list[float]]:
    """
    NoData handling for a reference raster.

    Configured label values (target.value_map keys) always win over the
    raster's NoData tag: with target.raster_nodata: auto (default) a tag that
    equals a label (e.g., ADP masks written with NoData=0 where 0 is
    background) is ignored; `use` always applies the tag; `ignore` never does.
    Explicit config values (target.annotation_nodata, target.nodata_values)
    that equal a label are a configuration error.

    Returns (collocation_nodata, nodata_values): the value passed to
    classifier_utils.collocate_annotation_to_cluster_grid (a sentinel that is
    not a label, so out-of-extent pixels stay NoData) and the values treated
    as NoData after collocation.
    """
    labels = {float(k) for k in target_cfg["value_map"]}
    cfg_nod = [float(v) for v in target_cfg.get("nodata_values", []) or []]
    explicit = target_cfg.get("annotation_nodata")
    bad = [v for v in cfg_nod + ([float(explicit)] if explicit is not None else []) if v in labels]
    if bad:
        raise ValueError(f"target.nodata_values / target.annotation_nodata {bad} are also labels in "
                         f"target.value_map {target_cfg['value_map']}; remove them from one of the two.")
    nod = list(cfg_nod)
    if explicit is not None:
        nod.append(float(explicit))
    else:
        tag = prof.get("nodata")
        mode = str(target_cfg.get("raster_nodata", "auto")).lower()
        if tag is not None and np.isfinite(float(tag)) and mode != "ignore":
            if mode == "auto" and float(tag) in labels:
                key = ("ignored", float(tag))
                if key not in _REF_NODATA_WARNED:
                    _REF_NODATA_WARNED.add(key)
                    LOG.warning("Ignoring reference raster NoData tag %s (e.g. %s): it is a label in "
                                "target.value_map. Set target.raster_nodata: use to apply it.",
                                tag, Path(str(truth_path)).name if truth_path else "")
            else:
                nod.append(float(tag))
    # collocation sentinel: first explicit/applied NoData, else a value that is not a label
    sentinel = next((v for v in nod), None)
    if sentinel is None:
        sentinel = next(v for v in (DEFAULT_NODATA, -32768.0, -1e20) if v not in labels)
        nod.append(sentinel)
    return sentinel, nod


def align_reference(truth_path, head_profile: dict[str, Any],
                    target_cfg: dict[str, Any]) -> np.ndarray:
    """Collocate the reference to the head grid (classifier_utils, nearest)."""
    arr, prof = read_raster(Path(truth_path))
    sentinel, nod = resolve_reference_nodata(prof, target_cfg, truth_path)
    arr, _, _ = collocate_annotation_to_cluster_grid(
        arr.astype(np.float32), prof, head_profile, annotation_nodata=sentinel)
    nodata_mask = ~np.isfinite(arr)
    for v in nod:
        nodata_mask |= arr == float(v)
    return map_reference_classes(arr, nodata_mask, target_cfg)


_REF_NODATA_WARNED: set = set()


# ---------------------------------------------------------------------------
# Label levels: primary reference vs. consensus with other references
# ---------------------------------------------------------------------------

LEVEL_PRIMARY, LEVEL_CONSENSUS = 1, 2


def reference_target_cfg(cfg: dict[str, Any], name: str | None = None) -> dict[str, Any]:
    """
    target block for one reference: cfg.target, updated by per-reference
    overrides from evaluation.references[name] and then
    target.label_levels.consensus_references[name] (when given as a mapping).
    """
    base = dict(cfg["target"])
    base.pop("label_levels", None)
    if name:
        base.update((cfg.get("evaluation", {}) or {}).get("references", {}).get(name, {}) or {})
        cons = (cfg["target"].get("label_levels") or {}).get("consensus_references") or {}
        if isinstance(cons, dict):
            base.update(cons.get(name) or {})
    return base


class LabelLevels:
    """
    Two assessment levels for reference labels (default: for negatives only).

      primary    every valid pixel of the primary reference (e.g., ADP smoke:
                 positive / negative / NoData)
      consensus  primary label confirmed by every consensus reference (e.g., a
                 VIIRS aerosol-type smoke mask): all consensus references are
                 observed at the pixel (require_all_observed) and give the
                 same class

    Per pixel: cons_ref_label (class agreed by the consensus references, -1
    when unobserved or when they disagree with each other) and label_level
    (2 = consensus confirms the primary label, 1 = primary only). Negative
    levels: primary negatives (ref_label == negative_class) and consensus
    negatives (those with label_level 2). With consensus_positives: false
    (default) positives are not tiered.

    Config (target.label_levels):
      consensus_references: [name, ...] or {name: target overrides}; names are
                            attach_references.py keys in additional_references
      negative_class: 0
      require_all_observed: true
      consensus_positives: false
      training_weights: {primary_negative: 1.0, consensus_negative: 1.0,
                         positive: 1.0}
    """

    def __init__(self, cfg: dict[str, Any]):
        lv = (cfg.get("target", {}) or {}).get("label_levels") or {}
        refs = lv.get("consensus_references") or []
        self.names = list(refs) if not isinstance(refs, dict) else list(refs.keys())
        self.negative_class = int(lv.get("negative_class", 0))
        self.require_all_observed = bool(lv.get("require_all_observed", True))
        self.consensus_positives = bool(lv.get("consensus_positives", False))
        tw = lv.get("training_weights") or {}
        self.w_primary_neg = float(tw.get("primary_negative", 1.0))
        self.w_consensus_neg = float(tw.get("consensus_negative", 1.0))
        self.w_positive = float(tw.get("positive", 1.0))
        self.cfg = cfg

    def __bool__(self):
        return bool(self.names)

    def describe(self) -> dict[str, Any]:
        return {"consensus_references": self.names, "negative_class": self.negative_class,
                "require_all_observed": self.require_all_observed,
                "consensus_positives": self.consensus_positives,
                "training_weights": {"primary_negative": self.w_primary_neg,
                                     "consensus_negative": self.w_consensus_neg,
                                     "positive": self.w_positive}}

    def consensus_label(self, scene: dict[str, Any], head_profile: dict[str, Any]):
        """Class agreed by the consensus references on the head grid (-1 elsewhere), and missing names."""
        shape = (int(head_profile["height"]), int(head_profile["width"]))
        extra = scene.get("additional_references") or {}
        labels, missing = [], []
        for n in self.names:
            if n not in extra:
                missing.append(n)
                labels.append(np.full(shape, -1, dtype=np.int16))
                continue
            labels.append(align_reference(extra[n], head_profile, reference_target_cfg(self.cfg, n)))
        return combine_consensus(labels, self.require_all_observed), missing

    def levels(self, ref: np.ndarray, cons: np.ndarray) -> np.ndarray:
        """label_level per pixel: 2 where cons confirms ref, 1 where ref is valid otherwise, 0 if ref invalid."""
        ref = np.asarray(ref)
        lev = np.where(ref >= 0, LEVEL_PRIMARY, 0).astype(np.int8)
        lev[(ref >= 0) & (np.asarray(cons) == ref)] = LEVEL_CONSENSUS
        return lev

    def level_mask(self, ref, label_level, level: str) -> np.ndarray:
        """Rows usable at an assessment level ('primary' or 'consensus')."""
        ref = np.asarray(ref)
        ok = ref >= 0
        level = str(level or "primary").lower()
        if level == "primary":
            return ok
        if level != "consensus":
            raise ValueError(f"Unknown label level {level!r}; use primary or consensus.")
        confirmed = np.asarray(label_level) == LEVEL_CONSENSUS
        if self.consensus_positives:
            return ok & confirmed
        return ok & ((ref != self.negative_class) | confirmed)

    def level_reference(self, ref, label_level, level: str) -> np.ndarray:
        """Reference labels at a level, -1 where the level does not assess the pixel."""
        return np.where(self.level_mask(ref, label_level, level), ref, -1)

    def training_weights(self, ref, label_level) -> np.ndarray | None:
        ref = np.asarray(ref)
        if (self.w_primary_neg, self.w_consensus_neg, self.w_positive) == (1.0, 1.0, 1.0):
            return None
        neg = ref == self.negative_class
        cons = np.asarray(label_level) == LEVEL_CONSENSUS
        return np.where(neg & cons, self.w_consensus_neg,
                        np.where(neg, self.w_primary_neg, self.w_positive)).astype(np.float64)


def combine_consensus(labels: list[np.ndarray], require_all_observed: bool = True) -> np.ndarray:
    """Class agreed by all observed references (all must be observed if required); -1 otherwise."""
    if not labels:
        return None
    R = np.stack([np.asarray(l) for l in labels])
    ok = R >= 0
    first = np.where(ok, R, -10).max(axis=0)
    agree = np.all((R == first) | ~ok, axis=0) & ok.any(axis=0)
    if require_all_observed:
        agree &= ok.all(axis=0)
    return np.where(agree, first, -1).astype(np.int16)


# ---------------------------------------------------------------------------
# Option C: label-noise transition matrix for the primary reference
# ---------------------------------------------------------------------------

def estimate_transition(primary, consensus, w, n_classes: int, prior_counts: float = 1.0) -> np.ndarray:
    """
    T[k, j] = P(primary label = j | consensus class = k), estimated on pixels
    where both are valid, with Dirichlet smoothing (prior_counts per cell).
    Treats the consensus references as the reference of truth for the
    estimate; their own errors are absorbed into T.
    """
    primary, consensus = np.asarray(primary), np.asarray(consensus)
    w = np.ones(primary.size) if w is None else np.asarray(w, dtype=np.float64)
    ok = (primary >= 0) & (consensus >= 0)
    M = np.full((n_classes, n_classes), float(prior_counts))
    np.add.at(M, (consensus[ok].astype(int), primary[ok].astype(int)), w[ok])
    return M / M.sum(axis=1, keepdims=True)


def resolve_transition(nc_cfg: dict[str, Any], primary, consensus, w, n_classes: int):
    """Transition matrix from calibration.noise_correction (fixed or estimated) with a usability check."""
    src = nc_cfg.get("transition", "estimate")
    info = {"source": src if isinstance(src, str) else "fixed"}
    if isinstance(src, str):
        if src != "estimate":
            raise ValueError("calibration.noise_correction.transition must be 'estimate' or a CxC matrix.")
        if consensus is None or (np.asarray(consensus) >= 0).sum() == 0:
            raise ValueError("noise_correction.transition: estimate needs consensus reference labels on "
                             "the calibration rows (target.label_levels.consensus_references); give a "
                             "fixed matrix instead.")
        T = estimate_transition(primary, consensus, w, n_classes, float(nc_cfg.get("prior_counts", 1.0)))
        info["n_pixels"] = int(((np.asarray(primary) >= 0) & (np.asarray(consensus) >= 0)).sum())
    else:
        T = np.asarray(src, dtype=np.float64)
        if T.shape != (n_classes, n_classes) or (T < 0).any():
            raise ValueError(f"noise_correction.transition must be a non-negative {n_classes}x{n_classes} matrix.")
        T = T / T.sum(axis=1, keepdims=True)
    margin = float(np.min(np.diag(T) - (T.sum(axis=1) - np.diag(T))))
    info.update(matrix=T.tolist(), diagonal_margin=margin)
    if margin < float(nc_cfg.get("min_diagonal_margin", 0.2)):
        raise ValueError(f"Transition matrix {np.round(T, 3).tolist()} is too close to uninformative "
                         f"(min P(correct) - P(error) = {margin:.3f} < min_diagonal_margin); the correction "
                         "would be unstable. Check the consensus references or the fixed matrix.")
    return T, info


def _masked_mean_std(values: np.ndarray, valid: np.ndarray, window: int):
    v = np.where(valid, values, 0.0).astype(np.float64)
    m = valid.astype(np.float64)
    cnt = uniform_filter(m, size=window, mode="constant")
    s1 = uniform_filter(v, size=window, mode="constant")
    s2 = uniform_filter(v * v, size=window, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        var = np.maximum(s2 / cnt - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def neighborhood_label_topn(labels: np.ndarray, valid: np.ndarray,
                            window: int, top_n: int):
    """
    Top-n most frequent cluster labels in a window and their fractions among
    valid pixels. FeatureBuilder's hashed histogram is not invertible to
    cluster IDs, so the explicit top-n is needed for neighborhood soft
    assignment (lookup of agreement per neighbouring cluster).
    """
    h, w = labels.shape
    ids = np.full((top_n, h, w), -1, dtype=np.int64)
    fracs = np.zeros((top_n, h, w), dtype=np.float32)
    cnt = uniform_filter(valid.astype(np.float64), size=window, mode="constant")
    cnt = np.where(cnt > 0, cnt, 1.0)

    lab_int = np.where(valid, labels, -1).astype(np.int64)
    for lab in np.unique(lab_int[valid]):
        f = (uniform_filter((lab_int == lab).astype(np.float64), size=window,
                            mode="constant") / cnt).astype(np.float32)
        f[f < 1e-6] = 0.0
        cur_f, cur_id = f, np.full((h, w), lab, dtype=np.int64)
        for n in range(top_n):
            swap = cur_f > fracs[n]
            if not swap.any():
                break
            tmp_f, tmp_id = fracs[n].copy(), ids[n].copy()
            fracs[n][swap] = cur_f[swap]
            ids[n][swap] = cur_id[swap]
            cur_f = np.where(swap, tmp_f, cur_f)
            cur_id = np.where(swap, tmp_id, cur_id)
    return ids, fracs


def neighborhood_radii(features_cfg: dict[str, Any]) -> list[int]:
    """Radii parsed exactly as classifier_utils does (window_radii / scales)."""
    return [r for r in normalize_feature_config({"features": features_cfg})["window_radii"] if r > 0]


def scene_feature_frame(bands: dict[str, np.ndarray], profile: dict[str, Any],
                        features_cfg: dict[str, Any], pair_id: str,
                        ref: np.ndarray | None = None,
                        invalid: InvalidClusters | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Per-pixel fold-independent features for all valid pixels of one scene.

    1. classifier_utils.FeatureBuilder on the SIT-FUSE joint-label raster
       (frac_same_cluster_r*, n_distinct_clusters_r*, same_cluster_entropy_r*,
       n_valid_neighbors_r*, center_cluster_logfreq, optional coarse/hashed
       fractions and cross-scale deltas) - the existing multiscale features.
    2. Head-output bands (parent/child probabilities, entropy, margin, ...).
    3. Per radius: masked mean/std of head statistics and the top-n
       neighbourhood cluster labels/fractions.
    Fold-dependent features are added later by ContextEncoder.transform().

    Pixels of excluded clusters (`invalid`) are not rows, and are treated as
    nodata neighbours by every neighbourhood feature (n_valid_neighbors_r*
    therefore reflects proximity to excluded areas).
    """
    valid = head_valid_mask(bands, invalid)
    if ref is not None:
        valid &= ref >= 0

    joint = np.where(valid, bands["joint_label"], DEFAULT_NODATA).astype(np.float32)
    annotation = (np.zeros(joint.shape, np.float32) if ref is None
                  else np.where(ref >= 0, ref, np.nan).astype(np.float32))
    fb_cfg = dict(features_cfg)
    fb_cfg["cluster_nodata"] = DEFAULT_NODATA
    fb_cfg["annotation_nodata"] = None
    df = FeatureBuilder(fb_cfg).build_features_for_pair(
        joint, pair_id, profile, annotation_arr=annotation)
    rows = df["row"].to_numpy()
    cols = df["col"].to_numpy()

    extra = {name: arr[rows, cols] for name, arr in bands.items()}

    nb = features_cfg.get("neighborhood", {})
    top_n = int(nb.get("top_n_labels", 3))
    level = str(nb.get("label_level", "joint")).lower()
    stats_of = nb.get("stats_of", ["parent_entropy", "child_entropy", "parent_maxprob"])
    labels = bands["joint_label"] if level == "joint" else bands["parent_label"]
    for r in neighborhood_radii(features_cfg):
        win = 2 * r + 1
        ids, fr = neighborhood_label_topn(labels, valid, win, top_n)
        for n in range(top_n):
            extra[f"nbr_r{r}_top{n + 1}_id"] = ids[n][rows, cols].astype(np.float32)
            extra[f"nbr_r{r}_top{n + 1}_frac"] = fr[n][rows, cols]
        for name in stats_of:
            mean, std = _masked_mean_std(bands[name], valid, win)
            extra[f"nbr_r{r}_{name}_mean"] = mean[rows, cols]
            extra[f"nbr_r{r}_{name}_std"] = std[rows, cols]

    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    if ref is not None:
        df["ref_label"] = df.pop("label").astype(np.int16)
    else:
        df = df.drop(columns=["label"])
    return df, valid


def cluster_class_counts(bands, ref, valid, n_classes, meta) -> pd.DataFrame:
    """Full-scene (unsampled) cluster x class pixel counts at both levels."""
    sel = valid & (ref >= 0)
    out = []
    for level, key in (("parent", "parent_label"), ("joint", "joint_label")):
        lab = bands[key][sel].astype(np.int64)
        cls = ref[sel].astype(np.int64)
        if lab.size == 0:
            continue
        flat = lab * n_classes + cls
        uniq, cnt = np.unique(flat, return_counts=True)
        out.append(pd.DataFrame({
            "level": level,
            "cluster": uniq // n_classes,
            "class": uniq % n_classes,
            "count": cnt,
        }))
    if not out:
        return pd.DataFrame(columns=["level", "cluster", "class", "count"] + list(meta))
    df = pd.concat(out, ignore_index=True)
    for k, v in meta.items():
        df[k] = v
    return df


# ---------------------------------------------------------------------------
# Context encoder: cluster agreement, soft assignment, assignment ensemble
# ---------------------------------------------------------------------------

class _Table:
    """Sorted-key lookup table mapping integer cluster IDs to class vectors."""

    def __init__(self, keys: np.ndarray, values: np.ndarray):
        order = np.argsort(keys)
        self.keys = keys[order].astype(np.int64)
        self.values = values[order].astype(np.float64)

    def lookup(self, ids: np.ndarray, default: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        out = np.broadcast_to(default, ids.shape + (default.shape[-1],)).copy()
        if self.keys.size == 0:
            return out
        pos = np.searchsorted(self.keys, ids)
        pos = np.clip(pos, 0, self.keys.size - 1)
        hit = (self.keys[pos] == ids) & (ids >= 0)
        out[hit] = self.values[pos[hit]]
        return out

    def lookup_rows(self, ids: np.ndarray, default_rows: np.ndarray) -> np.ndarray:
        """Like lookup, but default is a per-row array (n, C)."""
        ids = np.asarray(ids, dtype=np.int64)
        out = default_rows.copy()
        if self.keys.size == 0:
            return out
        pos = np.clip(np.searchsorted(self.keys, ids), 0, self.keys.size - 1)
        hit = (self.keys[pos] == ids) & (ids >= 0)
        out[hit] = self.values[pos[hit]]
        return out


def _aggregate_counts(counts: pd.DataFrame, level: str, n_classes: int,
                      date_multiplicity: dict[str, int] | None = None, unit: str = "date"):
    sub = counts[counts["level"] == level]
    if date_multiplicity is not None:
        mult = sub[unit].map(date_multiplicity).fillna(0).to_numpy()
        sub = sub.assign(count=sub["count"].to_numpy() * mult)
        sub = sub[sub["count"] > 0]
    if sub.empty:
        return np.zeros(0, dtype=np.int64), np.zeros((0, n_classes))
    piv = (sub.groupby(["cluster", "class"])["count"].sum()
           .unstack("class", fill_value=0)
           .reindex(columns=range(n_classes), fill_value=0))
    return piv.index.to_numpy(dtype=np.int64), piv.to_numpy(dtype=np.float64)


def assign_contexts(keys, n, A, rule_cfg, rng=None, step1_onehot=None):
    """
    Best-agreement context assignment over clusters.

    keys : (K,) cluster ids
    n    : (K, C) raw pixel counts (cluster x reference class)
    A    : (K, C) smoothed agreement fractions
    rule_cfg:
      rule: greedy_agreement | fraction_threshold
      metric: iou | f1                       (greedy_agreement)
      tolerance: 0.0                         (greedy; near-optimal slack)
      threshold: 0.5                         (fraction_threshold)
      min_pixels: 50
      step1_weight: 0.0                      (two-step hierarchical blend)
    rng : if given, sample a near-optimal solution (assignment ensemble).
    Returns (K,) assigned class index (0 = background).
    """
    K, C = A.shape
    rule = str(rule_cfg.get("rule", "greedy_agreement")).lower()
    min_pixels = float(rule_cfg.get("min_pixels", 50))
    w1 = float(rule_cfg.get("step1_weight", 0.0))
    total = n.sum(axis=1)
    eligible = total >= min_pixels

    rank_score = A.copy()
    if step1_onehot is not None and w1 > 0:
        rank_score = (1.0 - w1) * A + w1 * step1_onehot

    assigned = np.zeros(K, dtype=np.int64)
    if K == 0:
        return assigned

    if rule == "fraction_threshold":
        thr = rule_cfg.get("threshold", 0.5)
        if rng is not None and isinstance(rule_cfg.get("threshold_grid"), list):
            thr = rng.choice(rule_cfg["threshold_grid"])
        thr = float(thr)
        best_c = 1 + np.argmax(rank_score[:, 1:], axis=1)
        best_v = rank_score[np.arange(K), best_c]
        sel = eligible & (best_v >= thr)
        assigned[sel] = best_c[sel]
        return assigned

    if rule != "greedy_agreement":
        raise ValueError(f"Unknown assignment rule: {rule}")

    metric = str(rule_cfg.get("metric", "iou")).lower()
    tol = float(rule_cfg.get("tolerance", 0.0))
    chosen = np.zeros((K, C), dtype=bool)
    for c in range(1, C):
        P = n[:, c].sum()
        if P <= 0:
            continue
        idx = np.where(eligible)[0]
        idx = idx[np.argsort(-rank_score[idx, c], kind="stable")]
        tp = np.cumsum(n[idx, c])
        fp = np.cumsum(total[idx] - n[idx, c])
        fn = P - tp
        if metric == "f1":
            score = 2 * tp / np.maximum(2 * tp + fp + fn, EPS)
        else:
            score = tp / np.maximum(tp + fp + fn, EPS)
        if score.size == 0:
            continue
        best = int(np.argmax(score))
        k = best
        if rng is not None and tol > 0:
            near = np.where(score >= (1.0 - tol) * score[best])[0]
            k = int(rng.choice(near))
        chosen[idx[: k + 1], c] = True

    multi = chosen.sum(axis=1) > 1
    for i in np.where(chosen.any(axis=1))[0]:
        cand = np.where(chosen[i])[0]
        assigned[i] = cand[np.argmax(rank_score[i, cand])] if multi[i] else cand[0]
    return assigned


class ContextEncoder:
    """
    Fit on full-scene cluster x class counts from a set of dates; transform a
    pixel feature frame into fold-dependent features:

      sa_parent_c       soft assignment through parent-head probabilities
      sa_joint_c        soft assignment through parent + child probabilities
      agree_joint_c     agreement fraction of the pixel's hard joint cluster
      assign_hard_c     best-agreement hard assignment (one-hot)
      assign_vote_c     assignment-ensemble vote fraction
      assign_vote_var   sum_c V(1-V)
      nbr_r{r}_sa_c     neighborhood soft assignment from label histograms
      nbr_r{r}_vote_c   neighborhood assignment-ensemble vote
    """

    def __init__(self, n_classes: int, num_child: int, cfg: dict[str, Any]):
        self.C = int(n_classes)
        self.num_child = int(num_child)
        self.cfg = copy.deepcopy(cfg)
        self.alpha_parent = float(cfg.get("alpha_parent", 20.0))
        self.alpha_joint = float(cfg.get("alpha_joint", 20.0))
        self.rule_cfg = cfg.get("assignment", {})
        self.ens_cfg = cfg.get("assignment_ensemble", {})
        self.fitted = False

    # --- fitting -----------------------------------------------------------

    def _agreement(self, counts, mult=None, unit="date"):
        k1, n1 = _aggregate_counts(counts, "parent", self.C, mult, unit)
        k2, n2 = _aggregate_counts(counts, "joint", self.C, mult, unit)
        tot = n1.sum()
        prior = n1.sum(axis=0) / tot if tot > 0 else np.full(self.C, 1.0 / self.C)
        prior = np.clip(prior, EPS, None)
        prior = prior / prior.sum()

        A1 = (n1 + self.alpha_parent * prior) / (n1.sum(axis=1, keepdims=True) + self.alpha_parent)
        t1 = _Table(k1, A1)
        parent_of_joint = k2 // self.num_child
        A1_par = t1.lookup(parent_of_joint, prior)
        A2 = (n2 + self.alpha_joint * A1_par) / (n2.sum(axis=1, keepdims=True) + self.alpha_joint)
        return prior, (k1, n1, A1), (k2, n2, A2)

    def _assign(self, lev1, lev2, prior, rng=None):
        k1, n1, A1 = lev1
        k2, n2, A2 = lev2
        h1 = assign_contexts(k1, n1, A1, self.rule_cfg, rng=rng)
        step1 = _Table(k1, np.eye(self.C)[h1]).lookup(
            k2 // self.num_child, np.eye(self.C)[0])
        h2 = assign_contexts(k2, n2, A2, self.rule_cfg, rng=rng, step1_onehot=step1)
        return h1, h2

    def fit(self, counts: pd.DataFrame, seed: int = 42) -> "ContextEncoder":
        if counts.empty:
            raise ValueError("No cluster counts available to fit ContextEncoder.")
        prior, lev1, lev2 = self._agreement(counts)
        self.prior = prior
        self.t_A1 = _Table(lev1[0], lev1[2])
        self.t_A2 = _Table(lev2[0], lev2[2])
        self.n_joint = _Table(lev2[0], lev2[1])

        h1, h2 = self._assign(lev1, lev2, prior)
        self.t_H = _Table(lev2[0], np.eye(self.C)[h2])
        self.t_H1 = _Table(lev1[0], np.eye(self.C)[h1])

        # assignment ensemble: bootstrap dates + near-optimal solutions
        B = int(self.ens_cfg.get("n_members", 50))
        rng = np.random.default_rng(seed)
        # resampling unit: split groups (matchup generator) when present, else dates
        unit = self.ens_cfg.get("bootstrap_unit") or (
            "split_group" if "split_group" in counts.columns else "date")
        dates = np.array(sorted(counts[unit].astype(str).unique()))
        counts = counts.assign(**{unit: counts[unit].astype(str)})
        keys = lev2[0]
        votes = np.zeros((keys.size, self.C))
        vote_parent = np.zeros((lev1[0].size, self.C))
        resample_dates = bool(self.ens_cfg.get("bootstrap_dates", True))
        for b in range(B):
            if resample_dates and dates.size > 1:
                draw = rng.choice(dates, size=dates.size, replace=True)
                mult = pd.Series(draw).value_counts().to_dict()
            else:
                mult = None
            pr_b, l1_b, l2_b = self._agreement(counts, mult, unit)
            h1_b, h2_b = self._assign(l1_b, l2_b, pr_b, rng=rng)
            # map bootstrap clusters back onto full key set; clusters absent
            # from the resample fall back to their parent's assignment, then
            # to background.
            par_tab = _Table(l1_b[0], np.eye(self.C)[h1_b])
            par_fallback = par_tab.lookup(keys // self.num_child, np.eye(self.C)[0])
            votes += _Table(l2_b[0], np.eye(self.C)[h2_b]).lookup_rows(keys, par_fallback)
            vote_parent += par_tab.lookup(lev1[0], np.eye(self.C)[0])
        self.t_V = _Table(keys, votes / max(B, 1))
        self.t_V1 = _Table(lev1[0], vote_parent / max(B, 1))
        self.n_members = B
        self.fitted = True
        return self

    # --- transform ---------------------------------------------------------

    def _child_mix(self, df, r, parent_ids):
        """sum_j q(child j | parent_r) A2[(parent_r, j)] + leftover * A1[parent_r]."""
        Kc = self._count_cols(df, f"r{r}_child_top", "_id")
        A1p = self.t_A1.lookup(parent_ids, self.prior)
        V1p = self.t_V1.lookup(parent_ids, np.eye(self.C)[0])
        mix_A = np.zeros_like(A1p)
        mix_V = np.zeros_like(A1p)
        mass = np.zeros(len(df))
        for j in range(1, Kc + 1):
            cid = df[f"r{r}_child_top{j}_id"].to_numpy()
            q = df[f"r{r}_child_top{j}_prob"].to_numpy(dtype=np.float64)
            q = np.where(cid >= 0, q, 0.0)
            joint = np.where(cid >= 0, parent_ids * self.num_child + cid, -1)
            mix_A += q[:, None] * self.t_A2.lookup_rows(joint, A1p)
            mix_V += q[:, None] * self.t_V.lookup_rows(joint, V1p)
            mass += q
        rest = np.clip(1.0 - mass, 0.0, 1.0)[:, None]
        # no child head for this parent: SIT-FUSE joint id = num_child * parent
        no_head = mass <= 0
        collapsed = parent_ids * self.num_child
        A_coll = self.t_A2.lookup_rows(collapsed, A1p)
        V_coll = self.t_V.lookup_rows(collapsed, V1p)
        mix_A = np.where(no_head[:, None], A_coll, mix_A + rest * A1p)
        mix_V = np.where(no_head[:, None], V_coll, mix_V + rest * V1p)
        return mix_A, mix_V

    @staticmethod
    def _count_cols(df, prefix, suffix):
        k = 0
        while f"{prefix}{k + 1}{suffix}" in df.columns:
            k += 1
        return k

    def transform(self, df: pd.DataFrame, features_cfg: dict[str, Any]) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("ContextEncoder.transform called before fit.")
        C = self.C
        out = {}
        n = len(df)

        # parent-level soft assignment over top-K parents
        Kp = self._count_cols(df, "parent_top", "_id")
        sa_p = np.zeros((n, C))
        mass = np.zeros(n)
        for i in range(1, Kp + 1):
            pid = df[f"parent_top{i}_id"].to_numpy()
            q = df[f"parent_top{i}_prob"].to_numpy(dtype=np.float64)
            sa_p += q[:, None] * self.t_A1.lookup(pid, self.prior)
            mass += q
        sa_p += np.clip(1.0 - mass, 0.0, 1.0)[:, None] * self.prior

        # joint soft assignment through child heads of the top-R parents
        R = self._count_cols(df, "r", "_child_top1_id")
        sa_j = np.zeros((n, C))
        vote_soft = np.zeros((n, C))
        pmass = np.zeros(n)
        for r in range(1, R + 1):
            pid = df[f"parent_top{r}_id"].to_numpy()
            q = df[f"parent_top{r}_prob"].to_numpy(dtype=np.float64)
            mA, mV = self._child_mix(df, r, pid)
            sa_j += q[:, None] * mA
            vote_soft += q[:, None] * mV
            pmass += q
        # parents beyond rank R: parent-level agreement only
        for i in range(R + 1, Kp + 1):
            pid = df[f"parent_top{i}_id"].to_numpy()
            q = df[f"parent_top{i}_prob"].to_numpy(dtype=np.float64)
            sa_j += q[:, None] * self.t_A1.lookup(pid, self.prior)
            vote_soft += q[:, None] * self.t_V1.lookup(pid, np.eye(C)[0])
            pmass += q
        rest = np.clip(1.0 - pmass, 0.0, 1.0)[:, None]
        sa_j += rest * self.prior
        vote_soft += rest * np.eye(C)[0]

        joint = df["joint_label"].to_numpy()
        parent = df["parent_label"].to_numpy()
        A1h = self.t_A1.lookup(parent, self.prior)
        agree = self.t_A2.lookup_rows(joint, A1h)
        hard = self.t_H.lookup_rows(joint, self.t_H1.lookup(parent, np.eye(C)[0]))
        vote = self.t_V.lookup_rows(joint, self.t_V1.lookup(parent, np.eye(C)[0]))

        cls_range = range(C) if C > 2 else [1]
        for c in cls_range:
            out[f"sa_parent_{c}"] = sa_p[:, c]
            out[f"sa_joint_{c}"] = sa_j[:, c]
            out[f"agree_joint_{c}"] = agree[:, c]
            out[f"assign_hard_{c}"] = hard[:, c]
            out[f"assign_vote_{c}"] = vote[:, c]
            out[f"assign_vote_soft_{c}"] = vote_soft[:, c]
        out["assign_vote_var"] = (vote * (1.0 - vote)).sum(axis=1)

        # neighborhood soft assignment / vote from label histograms
        level = str(features_cfg.get("neighborhood", {}).get("label_level", "joint")).lower()
        for r in neighborhood_radii(features_cfg):
            N = self._count_cols(df, f"nbr_r{r}_top", "_id")
            nA = np.zeros((n, C))
            nV = np.zeros((n, C))
            fm = np.zeros(n)
            for k in range(1, N + 1):
                lid = df[f"nbr_r{r}_top{k}_id"].to_numpy()
                f = df[f"nbr_r{r}_top{k}_frac"].to_numpy(dtype=np.float64)
                if level == "joint":
                    par = np.where(lid >= 0, lid // self.num_child, -1)
                    a = self.t_A2.lookup_rows(lid, self.t_A1.lookup(par, self.prior))
                    v = self.t_V.lookup_rows(lid, self.t_V1.lookup(par, np.eye(C)[0]))
                else:
                    a = self.t_A1.lookup(lid, self.prior)
                    v = self.t_V1.lookup(lid, np.eye(C)[0])
                nA += f[:, None] * a
                nV += f[:, None] * v
                fm += f
            rest = np.clip(1.0 - fm, 0.0, 1.0)[:, None]
            nA += rest * self.prior
            nV += rest * np.eye(C)[0]
            for c in cls_range:
                out[f"nbr_r{r}_sa_{c}"] = nA[:, c]
                out[f"nbr_r{r}_vote_{c}"] = nV[:, c]

        enc = pd.DataFrame(out, index=df.index).astype(np.float32)
        return pd.concat([df.drop(columns=[c for c in enc.columns if c in df.columns]), enc], axis=1)

    def soft_matrix(self, df: pd.DataFrame, kind: str) -> np.ndarray:
        """Full (n, C) matrix for sa_joint / assign_vote / assign_vote_soft."""
        C = self.C
        if C > 2:
            return np.column_stack([df[f"{kind}_{c}"].to_numpy() for c in range(C)])
        p1 = df[f"{kind}_1"].to_numpy()
        return np.column_stack([1.0 - p1, p1])


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

ID_COLUMNS = {
    "joint_label", "parent_label", "child_label", "row", "col",
    "pair_id", "date", "product", "ref_label", "pop_weight", "split",
    "raster_height", "raster_width", "label",
    "group_id", "split_group", "classifier_split", "encoder_split", "split_stratum",
    "cons_ref_label", "label_level",   # reference label levels: never features
}
CLUSTER_STRUCTURE_PREFIXES = (
    "frac_same_cluster_", "n_valid_neighbors_", "n_distinct_clusters_",
    "same_cluster_entropy_", "coarse_frac_", "hash_bin_frac_",
    "center_cluster_logfreq",
)


def is_id_column(name: str) -> bool:
    return (name in ID_COLUMNS or name.endswith("_id")
            or name.startswith("assign_hard_"))


def select_feature_columns(df: pd.DataFrame, features_cfg: dict[str, Any]) -> list[str]:
    """
    Feature groups (features.groups):
      cluster_structure   existing FeatureBuilder multiscale features
      representation      head maxprob / margin / entropy / topk mass / joint prob
      head_topk_probs     individual top-k parent and child probabilities
      agreement           agree_joint_c
      soft_assignment     sa_parent_c, sa_joint_c
      assignment_ensemble assign_vote_c, assign_vote_soft_c, assign_vote_var
      neighborhood        nbr_r* head-statistic means/stds, top-n label fractions
      neighborhood_context nbr_r*_sa_c, nbr_r*_vote_c
      encoder_mc          mc_* MC input-noise bands (HAB head export, optional)
      encoder_ood         ood_* embedding-distance bands (HAB head export, optional)
    Raw cluster IDs are never used as numeric features. Head files without the
    optional mc_* / ood_* bands simply contribute no columns to those groups.
    """
    groups = set(features_cfg.get("groups", [
        "cluster_structure", "representation", "head_topk_probs", "agreement",
        "soft_assignment", "assignment_ensemble", "neighborhood", "neighborhood_context",
        "encoder_mc", "encoder_ood"]))
    cols = []
    for c in df.columns:
        if is_id_column(c) or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if c.startswith(CLUSTER_STRUCTURE_PREFIXES) or "_delta_r" in c:
            g = "cluster_structure"
        elif c.startswith("mc_"):
            g = "encoder_mc"
        elif c.startswith("ood_"):
            g = "encoder_ood"
        elif c.startswith("nbr_r"):
            g = "neighborhood_context" if ("_sa_" in c or "_vote_" in c) else "neighborhood"
        elif c.startswith(("sa_parent_", "sa_joint_")):
            g = "soft_assignment"
        elif c.startswith("agree_joint_"):
            g = "agreement"
        elif c.startswith("assign_vote"):
            g = "assignment_ensemble"
        elif "_top" in c and c.endswith("_prob"):
            g = "head_topk_probs"
        else:
            g = "representation"
        if g in groups:
            cols.append(c)
    return sorted(cols)


def uq_feature_signature(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Feature-defining settings, canonicalized with classifier_utils so the
    saved signature is checked by classifier_utils.compare_feature_signatures.
    """
    sig = {
        "features": normalize_feature_config({"features": cfg.get("features", {})}),
        "context_encoder": cfg.get("context_encoder", {}),
        "target_classes": cfg.get("target", {}).get("classes"),
    }
    # resolved excluded clusters (catches edits inside a referenced context YAML);
    # omitted when empty so models trained before this option still validate
    invalid = InvalidClusters.from_config(cfg)
    if invalid:
        sig["invalid_clusters"] = invalid.signature()
    return _canonicalize_config_value(sig)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def build_targets(df: pd.DataFrame, encoder: ContextEncoder, target_cfg: dict[str, Any]) -> np.ndarray:
    """
    Soft training targets (n, C). No ambiguity masking.

    target.mode:
      reference            one-hot reference class
      cluster_agreement    sa_joint (soft context assignment)
      assignment_ensemble  assign_vote_soft (assignment-ensemble vote)
      blended              (1-lam) * onehot(ref) + lam * sa_joint
      blended_ensemble     (1-lam) * onehot(ref) + lam * assign_vote_soft
    """
    C = encoder.C
    mode = str(target_cfg.get("mode", "blended")).lower()
    lam = float(target_cfg.get("blend_lambda", 0.5))
    ref = df["ref_label"].to_numpy().astype(int)
    onehot = np.eye(C)[np.clip(ref, 0, C - 1)]

    if mode == "reference":
        T = onehot
    elif mode == "cluster_agreement":
        T = encoder.soft_matrix(df, "sa_joint")
    elif mode == "assignment_ensemble":
        T = encoder.soft_matrix(df, "assign_vote_soft")
    elif mode == "blended":
        T = (1 - lam) * onehot + lam * encoder.soft_matrix(df, "sa_joint")
    elif mode == "blended_ensemble":
        T = (1 - lam) * onehot + lam * encoder.soft_matrix(df, "assign_vote_soft")
    else:
        raise ValueError(f"Unknown target.mode: {mode}")
    T = np.clip(T, 0.0, 1.0)
    return T / np.maximum(T.sum(axis=1, keepdims=True), EPS)


def expand_soft_targets(X: pd.DataFrame, T: np.ndarray, base_weight=None, min_weight=1e-4):
    """
    Soft-label trick for estimators that only accept hard labels: one copy of
    each row per class, weighted by its soft target. Minimizing weighted
    hard-label log loss on the expanded data equals soft cross-entropy.
    """
    n, C = T.shape
    w0 = np.ones(n) if base_weight is None else np.asarray(base_weight, dtype=np.float64)
    rows, labels, weights = [], [], []
    for c in range(C):
        wc = w0 * T[:, c]
        keep = np.where(wc > min_weight)[0]
        rows.append(keep)
        labels.append(np.full(keep.size, c))
        weights.append(wc[keep])
    rows = np.concatenate(rows)
    return (X.iloc[rows].reset_index(drop=True),
            np.concatenate(labels), np.concatenate(weights))


def align_proba(P: np.ndarray, classes, n_classes: int) -> np.ndarray:
    """Align (…, n, k) probabilities over `classes` to class indices 0..C-1."""
    out = np.zeros(P.shape[:-1] + (n_classes,), dtype=np.float64)
    for j, c in enumerate(classes):
        out[..., int(c)] = P[..., j]
    return out


def full_proba(model, X, n_classes: int) -> np.ndarray:
    return align_proba(model.predict_proba(X), model.classes_, n_classes)


# ---------------------------------------------------------------------------
# Uncertainty measures (classifier_utils predictive_entropy / margin_uncertainty)
# ---------------------------------------------------------------------------

def predictive_entropy_norm(P: np.ndarray) -> np.ndarray:
    """classifier_utils.predictive_entropy normalized by log(C) to [0, 1]."""
    return _predictive_entropy(P) / np.log(P.shape[-1])


def ensemble_decomposition(member_P: np.ndarray):
    """
    member_P: (M, n, C). Returns mean probs and normalized total predictive
    entropy, expected member entropy, mutual information, and variance.
    """
    z = np.log(member_P.shape[-1])
    mean = member_P.mean(axis=0)
    total = _predictive_entropy(mean) / z
    expected = np.mean([_predictive_entropy(p) for p in member_P], axis=0) / z
    mi = np.maximum(total - expected, 0.0)
    var = member_P.var(axis=0).sum(axis=-1)
    return mean, total, expected, mi, var


def forest_member_probabilities(model, X, n_classes: int) -> np.ndarray:
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    return align_proba(forest_tree_probabilities(model, X), clf.classes_, n_classes)


def bootstrap_member_probabilities(base_model_path, X, paths, n_classes: int) -> np.ndarray:
    P, classes = bootstrap_model_probabilities(Path(base_model_path), X, list(paths))
    return align_proba(P, classes, n_classes)


def effective_training_prior(model, y_expanded, w_expanded, n_classes: int) -> np.ndarray:
    """
    Class prior the estimator effectively optimizes on the soft-expanded data,
    including any class_weight from classifier_utils.build_models
    (e.g. 'balanced'). LogisticRegression computes 'balanced' from
    sample-weighted counts; tree models and HGB use unweighted label counts.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.utils.class_weight import compute_class_weight

    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    cw = getattr(clf, "class_weight", None)
    classes = np.arange(n_classes)
    present = np.unique(y_expanded)
    if cw is None:
        cw_vec = np.ones(n_classes)
    elif isinstance(cw, dict):
        cw_vec = np.array([float(cw.get(c, cw.get(str(c), 1.0))) for c in classes])
    else:  # 'balanced' / 'balanced_subsample'
        sw = w_expanded if isinstance(clf, LogisticRegression) else None
        cw_present = compute_class_weight("balanced", classes=present, y=y_expanded, sample_weight=sw)
        cw_vec = np.ones(n_classes)
        cw_vec[present.astype(int)] = cw_present
    mass = np.array([w_expanded[y_expanded == c].sum() for c in classes]) * cw_vec
    return mass / mass.sum()


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _simplex(Q):
    """Clip negatives and renormalize rows (projection used after noise inversion)."""
    Q = np.clip(Q, 0.0, None)
    return Q / np.maximum(Q.sum(axis=1, keepdims=True), EPS)


class ProbabilityCalibrator:
    """
    Post-hoc calibration applied after the classifier.

    1. Prior correction: undo training-set class balancing,
       p'_c ∝ p_c * pi_true_c / pi_train_c.
    2. method: temperature | platt | isotonic | none
       Fit by minimizing weighted (soft) cross-entropy on calibration data.
    3. Optional label-noise correction (option C): noise_T[k, j] =
       P(observed label j | true class k). temperature / platt are fit by
       forward correction (the corrected probabilities, passed through
       noise_T, must explain the observed labels); isotonic is fit on the
       observed labels and inverted, p = solve(noise_T^T, p_obs), projected
       onto the simplex. pi_true is then the corrected (clean) prior.
       transform() returns corrected probabilities.
    """

    def __init__(self, method="temperature", pi_train=None, pi_true=None, noise_T=None):
        self.method = str(method).lower()
        self.pi_train = None if pi_train is None else np.asarray(pi_train, dtype=np.float64)
        self.pi_true = None if pi_true is None else np.asarray(pi_true, dtype=np.float64)
        self.T = 1.0
        self.a, self.b = 1.0, 0.0
        self.iso = None
        self.noise_T = None if noise_T is None else np.asarray(noise_T, dtype=np.float64)
        if self.noise_T is not None and self.pi_true is not None:
            self.pi_true = _simplex(np.linalg.solve(self.noise_T.T, self.pi_true)[None, :])[0]

    def _prior(self, P):
        if self.pi_train is None or self.pi_true is None:
            return P
        r = np.clip(self.pi_true, EPS, None) / np.clip(self.pi_train, EPS, None)
        Q = P * r[None, :]
        return Q / np.maximum(Q.sum(axis=1, keepdims=True), EPS)

    def fit(self, P, Y, w=None):
        P = self._prior(np.asarray(P, dtype=np.float64))
        Y = np.asarray(Y, dtype=np.float64)
        w = np.ones(len(P)) if w is None else np.asarray(w, dtype=np.float64)
        w = w / w.sum()
        logp = np.log(np.clip(P, EPS, 1.0))

        NT = getattr(self, "noise_T", None)
        fwd = (lambda Q: Q @ NT) if NT is not None else (lambda Q: Q)
        if self.method == "temperature":
            def nll(logT):
                Q = fwd(_softmax(logp / np.exp(logT)))
                return -(w[:, None] * Y * np.log(np.clip(Q, EPS, 1.0))).sum()
            res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
            self.T = float(np.exp(res.x))
        elif self.method == "platt":
            if P.shape[1] != 2:
                raise ValueError("platt calibration is binary only; use temperature.")
            z = np.log(np.clip(P[:, 1], EPS, 1)) - np.log(np.clip(P[:, 0], EPS, 1))
            y = Y[:, 1]

            def nll(ab):
                s = 1.0 / (1.0 + np.exp(-(ab[0] * z + ab[1])))
                if NT is not None:
                    s = (1 - s) * NT[0, 1] + s * NT[1, 1]
                s = np.clip(s, EPS, 1 - EPS)
                return -(w * (y * np.log(s) + (1 - y) * np.log(1 - s))).sum()
            res = minimize(nll, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
            self.a, self.b = map(float, res.x)
        elif self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self.iso = []
            for c in range(P.shape[1]):
                m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                m.fit(P[:, c], Y[:, c], sample_weight=w)
                self.iso.append(m)
        elif self.method != "none":
            raise ValueError(f"Unknown calibration method: {self.method}")
        return self

    def transform(self, P):
        P = self._prior(np.asarray(P, dtype=np.float64))
        if self.method == "temperature":
            return _softmax(np.log(np.clip(P, EPS, 1.0)) / self.T)
        if self.method == "platt":
            z = np.log(np.clip(P[:, 1], EPS, 1)) - np.log(np.clip(P[:, 0], EPS, 1))
            s = 1.0 / (1.0 + np.exp(-(self.a * z + self.b)))
            return np.column_stack([1 - s, s])
        if self.method == "isotonic":
            Q = np.column_stack([m.predict(P[:, c]) for c, m in enumerate(self.iso)])
            Q = Q / np.maximum(Q.sum(axis=1, keepdims=True), EPS)
            NT = getattr(self, "noise_T", None)
            if NT is not None:
                Q = _simplex(np.linalg.solve(NT.T, Q.T).T)
            return Q
        return P

    def transform_members(self, member_P):
        return np.stack([self.transform(p) for p in member_P])

    def describe(self):
        return {
            "method": self.method, "temperature": self.T, "platt_a": self.a, "platt_b": self.b,
            "pi_train": None if self.pi_train is None else self.pi_train.tolist(),
            "pi_true": None if self.pi_true is None else self.pi_true.tolist(),
            "noise_transition": None if getattr(self, "noise_T", None) is None else self.noise_T.tolist(),
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def hard_metrics(pred, ref, w, n_classes):
    from sklearn.metrics import confusion_matrix
    out = {}
    cm = confusion_matrix(ref, pred, labels=list(range(n_classes)),
                          sample_weight=np.asarray(w, dtype=np.float64))
    ious, f1s = [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
        out[f"class_{c}"] = {
            "precision": tp / (tp + fp) if tp + fp > 0 else np.nan,
            "recall": tp / (tp + fn) if tp + fn > 0 else np.nan,
            "f1": f1, "iou": iou}
        if c > 0:
            ious.append(iou)
            f1s.append(f1)
    # all-NaN when a subset has no target pixels (e.g., low-coverage selections)
    out["target_macro_f1"] = float(np.nanmean(f1s)) if np.isfinite(f1s).any() else np.nan
    out["target_macro_iou"] = float(np.nanmean(ious)) if np.isfinite(ious).any() else np.nan
    out["confusion_matrix_weighted"] = cm.tolist()
    return out


def reliability_table(p, y, w, n_bins=15):
    """Adaptive (equal-weight) bins for one class probability."""
    order = np.argsort(p)
    p, y, w = p[order], y[order], w[order]
    cw = np.cumsum(w)
    edges = np.searchsorted(cw, np.linspace(0, cw[-1], n_bins + 1)[1:-1])
    rows, ece = [], 0.0
    for idx in np.split(np.arange(p.size), edges):
        if idx.size == 0:
            continue
        ww = w[idx].sum()
        conf = float((w[idx] * p[idx]).sum() / ww)
        freq = float((w[idx] * y[idx]).sum() / ww)
        rows.append({"mean_predicted": conf, "observed_frequency": freq,
                     "weight": float(ww), "n": int(idx.size)})
        ece += ww * abs(conf - freq)
    return rows, float(ece / w.sum())


def risk_coverage(uncertainty, error, w):
    order = np.argsort(uncertainty, kind="stable")
    e, ww = error[order].astype(np.float64), w[order]
    cw = np.cumsum(ww)
    risk = np.cumsum(ww * e) / cw
    cov = cw / cw[-1]
    aurc = float(np.trapezoid(risk, cov)) if hasattr(np, "trapezoid") else float(np.trapz(risk, cov))
    at = {f"risk_at_coverage_{int(c * 100)}": float(risk[np.searchsorted(cov, c)])
          for c in (0.5, 0.8, 0.9, 1.0) if np.searchsorted(cov, c) < risk.size}
    return {"aurc": aurc, **at}


def probability_metrics(P, ref, T_soft, w, n_classes, threshold=None, uncertainty=None):
    """
    P       (n, C) calibrated probabilities
    ref     (n,)   reference class index
    T_soft  (n, C) training-style soft target
    w       (n,)   population weights (undo sampling)
    """
    w = np.asarray(w, dtype=np.float64)
    wn = w / w.sum()
    Y = np.eye(n_classes)[ref]
    logP = np.log(np.clip(P, EPS, 1.0))
    pred = decide(P, threshold)

    res = {
        "nll_vs_reference": float(-(wn[:, None] * Y * logP).sum()),
        "nll_vs_soft_target": float(-(wn[:, None] * T_soft * logP).sum()),
        "brier_vs_reference": float((wn * ((P - Y) ** 2).sum(axis=1)).sum()),
        "brier_vs_soft_target": float((wn * ((P - T_soft) ** 2).sum(axis=1)).sum()),
        "hard_vs_reference": hard_metrics(pred, ref, w, n_classes),
        "reliability": {},
    }
    eces = []
    for c in range(1 if n_classes == 2 else 0, n_classes):
        tab_r, ece_r = reliability_table(P[:, c], Y[:, c], w)
        tab_s, ece_s = reliability_table(P[:, c], T_soft[:, c], w)
        res["reliability"][f"class_{c}"] = {
            "ece_vs_reference": ece_r, "ece_vs_soft_target": ece_s,
            "table_vs_reference": tab_r, "table_vs_soft_target": tab_s}
        eces.append(ece_r)
    res["classwise_ece_vs_reference"] = float(np.mean(eces))
    if uncertainty is not None:
        res["risk_coverage"] = {k: risk_coverage(u, (pred != ref), w) for k, u in uncertainty.items()}
    return res


def decide(P, threshold=None):
    """
    Decision rule. Binary: p1 >= threshold. Multiclass: target-vs-background
    on 1 - p0, then argmax over target classes (avoids rare target classes
    losing to background under natural-prevalence argmax).
    """
    if threshold is None:
        return P.argmax(axis=1)
    if P.shape[1] == 2:
        return (P[:, 1] >= threshold).astype(int)
    tgt = 1 + P[:, 1:].argmax(axis=1)
    return np.where(1.0 - P[:, 0] >= threshold, tgt, 0)


def select_threshold_multiclass(P, ref, w, metric="iou", grid=None):
    """Threshold on 1 - p0 maximizing target-class macro IoU/F1."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else np.asarray(grid)
    best_t, best_s = 0.5, -1.0
    for t in grid:
        m = hard_metrics(decide(P, t), ref, w, P.shape[1])
        s = m["target_macro_iou"] if metric == "iou" else m["target_macro_f1"]
        if np.isfinite(s) and s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t, best_s


def select_threshold(P1, ref, w, metric="iou", grid=None):
    grid = np.linspace(0.05, 0.95, 91) if grid is None else np.asarray(grid)
    best_t, best_s = 0.5, -1.0
    for t in grid:
        pred = (P1 >= t).astype(int)
        tp = w[(pred == 1) & (ref == 1)].sum()
        fp = w[(pred == 1) & (ref == 0)].sum()
        fn = w[(pred == 0) & (ref == 1)].sum()
        s = tp / (tp + fp + fn) if metric == "iou" else 2 * tp / (2 * tp + fp + fn)
        if np.isfinite(s) and s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t, best_s


# ---------------------------------------------------------------------------
# Raster output
# ---------------------------------------------------------------------------

def write_band_stack(path, arrays: dict[str, np.ndarray], profile: dict[str, Any],
                     tags: dict[str, Any] | None = None, nodata=OUT_NODATA) -> None:
    import rasterio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = profile.copy()
    gcps = prof.pop("gcps", None)
    prof.update(driver="GTiff", count=len(arrays), dtype="float32", nodata=nodata,
                compress="deflate", BIGTIFF="IF_SAFER")
    prof.pop("blockxsize", None)
    prof.pop("blockysize", None)
    prof.pop("tiled", None)
    with rasterio.open(path, "w", **prof) as dst:
        for i, (name, arr) in enumerate(arrays.items(), start=1):
            dst.write(arr.astype(np.float32), i)
            dst.set_band_description(i, name)
        if gcps is not None:
            dst.gcps = gcps
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})


# ---------------------------------------------------------------------------
# Products: layer registry, metadata, selective thresholds, areas
# ---------------------------------------------------------------------------

PRODUCT_VERSION = "1.0"
ABSTAIN = -1.0      # pred_label_selective value for flagged pixels

# (regex, product_type, description, units, valid_range)
LAYER_REGISTRY = [
    (r"^pred_label$", "discrete", "Class label from calibrated probabilities and the calibration-set threshold", "class index", "0..C-1"),
    (r"^pred_label_selective$", "discrete", "pred_label where the selective rule retains the pixel; -1 where flagged", "class index", "-1..C-1"),
    (r"^selective_flag$", "discrete_quality", "1 = retained by the selective rule (uncertainty <= calibrated threshold), 0 = flagged", "flag", "0..1"),
    (r"^confidence$", "probabilistic", "Calibrated probability of the predicted class", "probability", "0..1"),
    (r"^p_(\d+)$", "probabilistic", "Calibrated, prior-corrected class probability", "probability", "0..1"),
    (r"^ensemble_mean_p(\d+)$", "probabilistic", "Mean calibrated member probability", "probability", "0..1"),
    (r"^ensemble_p(\d+)_q(\d+)$", "probabilistic_interval", "Quantile of calibrated member probabilities", "probability", "0..1"),
    (r"^soft_assignment_p(\d+)$", "probabilistic_baseline", "Soft context assignment from cluster agreement", "probability", "0..1"),
    (r"^assignment_vote_p(\d+)$", "probabilistic_baseline", "Assignment-ensemble vote fraction", "probability", "0..1"),
    (r"^predictive_entropy$", "uncertainty_total", "Predictive entropy of calibrated probabilities / log(C)", "normalized entropy", "0..1"),
    (r"^margin_uncertainty$", "uncertainty_total", "1 - (p_max - p_second)", "unitless", "0..1"),
    (r"^ensemble_total_entropy$", "uncertainty_total", "Entropy of the ensemble-mean probability / log(C)", "normalized entropy", "0..1"),
    (r"^ensemble_expected_entropy$", "uncertainty_aleatoric", "Mean member entropy / log(C) (aleatoric component)", "normalized entropy", "0..1"),
    (r"^ensemble_mutual_information$", "uncertainty_epistemic", "Total minus expected entropy (epistemic component)", "normalized entropy", "0..1"),
    (r"^ensemble_variance$", "uncertainty_epistemic", "Sum over classes of member-probability variance", "probability^2", "0..0.5"),
    (r"^assignment_vote_variance$", "uncertainty_structural", "Variance of assignment-ensemble votes (cluster-to-class mapping)", "unitless", "0..0.25"),
    (r"^(parent|child)_entropy$", "uncertainty_representation", "Normalized entropy of the SIT-FUSE head distribution", "normalized entropy", "0..1"),
    (r"^excluded_cluster_mask$", "quality", "1 = pixel of an excluded (invalid_clusters) cluster, 0 = kept; nodata = no head data", "flag", "0..1"),
    (r"^(parent|child)_maxprob$", "diagnostic", "Max probability of the SIT-FUSE head distribution", "probability", "0..1"),
]


def layer_info(name: str) -> dict[str, str]:
    for rx, kind, desc, units, rng in LAYER_REGISTRY:
        m = re.match(rx, name)
        if m:
            if kind in ("probabilistic", "probabilistic_baseline") and m.groups():
                desc = f"{desc} (class {m.group(1)})"
            if kind == "probabilistic_interval":
                desc = f"{desc} (class {m.group(1)}, q={int(m.group(2)) / 100:.2f})"
            return {"product_type": kind, "description": desc, "units": units, "valid_range": rng}
    return {"product_type": "other", "description": name, "units": "", "valid_range": ""}


def tag_product_layers(paths: dict[str, Path], provenance: dict[str, Any], cog: bool = False) -> None:
    """Write per-layer description/units/type and shared provenance tags; optional COG copy."""
    import rasterio
    import rasterio.shutil
    flat = {k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v))
            for k, v in provenance.items()}
    for name, path in paths.items():
        info = layer_info(name)
        with rasterio.open(path, "r+") as dst:
            dst.update_tags(**flat, layer=name, product_version=PRODUCT_VERSION)
            dst.update_tags(1, **info)
            dst.set_band_description(1, f"{name}: {info['description']}")
        if cog:
            tmp = Path(str(path) + ".tmp.tif")
            rasterio.shutil.copy(path, tmp, driver="COG", compress="DEFLATE")
            os.replace(tmp, path)


def pixel_area_km2(profile: dict[str, Any]) -> np.ndarray | None:
    """Per-pixel area (km^2) for geographic or projected grids; None without CRS."""
    from rasterio.crs import CRS
    crs, tr = profile.get("crs"), profile.get("transform")
    if crs is None or tr is None:
        return None
    crs = CRS.from_user_input(crs)
    H, W = int(profile["height"]), int(profile["width"])
    if crs.is_geographic:
        R = 6371.0088
        rows = np.arange(H + 1)
        lat_edges = np.deg2rad(tr.f + tr.e * rows)
        band = np.abs(np.sin(lat_edges[:-1]) - np.sin(lat_edges[1:]))
        a = (R ** 2) * np.deg2rad(abs(tr.a)) * band
        return np.repeat(a[:, None], W, axis=1)
    unit = crs.linear_units_factor[1] if hasattr(crs, "linear_units_factor") else 1.0
    return np.full((H, W), abs(tr.a * tr.e - tr.b * tr.d) * unit ** 2 / 1e6)


def select_selective_threshold(u, err, w, target_error=None, target_coverage=None):
    """
    Uncertainty threshold for the selective product, from calibration rows.
    target_error: largest coverage whose retained weighted error <= target.
    target_coverage: weighted quantile of uncertainty at that coverage.
    Returns dict(threshold, coverage, retained_error, flagged_error).
    """
    u, err, w = np.asarray(u, float), np.asarray(err, float), np.asarray(w, float)
    ok = np.isfinite(u)
    u, err, w = u[ok], err[ok], w[ok]
    order = np.argsort(u, kind="stable")
    uu, ee, ww = u[order], err[order], w[order]
    cw = np.cumsum(ww)
    cov = cw / cw[-1]
    risk = np.cumsum(ww * ee) / cw
    if target_coverage is not None:
        k = int(min(np.searchsorted(cov, float(target_coverage)), uu.size - 1))
    else:
        tgt = 0.05 if target_error is None else float(target_error)
        # thresholds can only sit at the end of a run of tied uncertainty values
        ends = np.r_[np.flatnonzero(np.diff(uu) > 0), uu.size - 1]
        good = ends[risk[ends] <= tgt]
        if good.size == 0:
            return {"threshold": float("-inf"), "coverage": 0.0, "retained_error": None,
                    "flagged_error": float(np.average(ee, weights=ww))}
        k = int(good[-1])
    thr = float(uu[k])
    keep = u <= thr
    return {
        "threshold": thr,
        "coverage": float(w[keep].sum() / w.sum()),
        "retained_error": float(np.average(err[keep], weights=w[keep])) if keep.any() else None,
        "flagged_error": float(np.average(err[~keep], weights=w[~keep])) if (~keep).any() else None,
    }


class AreaAccumulator:
    """
    Scene-level class areas over a pixel mask: discrete (pred == c), expected
    (sum p_c), analytic SD (independent-pixel Poisson-binomial), and ensemble
    member totals (for empirical quantiles). Areas in km^2 when pixel areas
    are known, else in pixels.
    """

    def __init__(self, n_classes: int):
        self.C = n_classes
        self.discrete = np.zeros(n_classes)
        self.expected = np.zeros(n_classes)
        self.var = np.zeros(n_classes)
        self.members = None
        self.n = 0

    def add(self, P, pred, a, members=None):
        a = np.asarray(a, float)
        for c in range(self.C):
            self.discrete[c] += a[pred == c].sum()
            self.expected[c] += (a * P[:, c]).sum()
            self.var[c] += (a ** 2 * P[:, c] * (1 - P[:, c])).sum()
        if members is not None:
            tot = np.einsum("mnc,n->mc", members, a)
            self.members = tot if self.members is None else self.members + tot
        self.n += int(P.shape[0])

    def summary(self, quantiles=(0.05, 0.95), z=1.6449):
        out = {"n_pixels": self.n}
        for c in range(1 if self.C == 2 else 0, self.C):
            sd = float(np.sqrt(self.var[c]))
            d = {"discrete": float(self.discrete[c]), "expected": float(self.expected[c]),
                 "analytic_sd": sd,
                 "analytic_lo": float(max(0.0, self.expected[c] - z * sd)),
                 "analytic_hi": float(self.expected[c] + z * sd)}
            if self.members is not None:
                q = np.quantile(self.members[:, c], quantiles)
                d.update(ensemble_lo=float(q[0]), ensemble_hi=float(q[-1]),
                         ensemble_median=float(np.median(self.members[:, c])))
            out[f"class_{c}"] = d
        return out
