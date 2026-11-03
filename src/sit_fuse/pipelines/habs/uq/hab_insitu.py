"""
In-situ HAB reference handling for UQ: loading the combined_full.csv-format
table used by insitu_hab_to_multi_hist, per-species binning, reference
uncertainty, and leakage-safe grouped splits.

Splits are assigned once per *sample group* (station x time block) and shared
by every instrument, so a pier sample used to fit the cluster->bin context
encoder for VIIRS is never used to evaluate OLCI or the fused product.
Roles:
  encoder      fit ContextEncoder (cluster -> concentration-bin agreement)
  classifier   fit the UQ classifier on out-of-fold encoder features
  calibration  fit ProbabilityCalibrator, selective thresholds, fusion skill
  test         held out for evaluate_uq / fusion evaluation only
With cross_fit: true (default) the encoder and classifier roles are merged and
encoder features for classifier rows are produced out-of-fold (K-fold over
groups), which uses all training data while keeping the features honest.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from sit_fuse.pipelines.habs.uq.hab_uq_common import (
    EPS, SpeciesSpec, concentration_to_class, measurement_soft_labels)


def load_insitu(path: str, max_depth_m: float | None = 1.0,
                start_date=None, end_date=None) -> pd.DataFrame:
    """
    Read the combined_full.csv format (or xlsx) exactly as
    insitu_hab_to_multi_hist does (Datetime = pd.to_datetime(time),
    latitude/longitude -> Latitude/Longitude) and add UQ bookkeeping columns.
    """
    df = pd.read_excel(path) if str(path).endswith((".xlsx", ".xls")) else pd.read_csv(path)
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["time"], format="mixed")
    df = df.rename(columns={"latitude": "Latitude", "longitude": "Longitude"})
    # optional full timestamp written by the ingestion package (sample_time_utc)
    if "sample_time_utc" in df.columns:
        df["sample_time_utc"] = pd.to_datetime(df["sample_time_utc"], utc=True, errors="coerce")
    else:
        df["sample_time_utc"] = pd.NaT
    df["date"] = df["Datetime"].dt.normalize()
    if max_depth_m is not None:
        dcol = "depth" if "depth" in df.columns else ("Sample Depth" if "Sample Depth" in df.columns else None)
        if dcol is not None:
            d = pd.to_numeric(df[dcol], errors="coerce")
            df = df[~(d > max_depth_m)]          # NaN depth kept (surface grab default)
    if start_date is not None:
        df = df[df["Datetime"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["Datetime"] <= pd.Timestamp(end_date)]
    if "Location_Code" not in df.columns:
        df["Location_Code"] = "UNK"
    if "source" not in df.columns:
        df["source"] = "combined_full"
    df["sample_uid"] = [
        hashlib.md5(f"{s}|{d}|{la:.4f}|{lo:.4f}|{i}".encode()).hexdigest()[:12]
        for i, (s, d, la, lo) in enumerate(zip(df["Location_Code"], df["Datetime"].astype(str),
                                               df["Latitude"], df["Longitude"]))]
    return df.reset_index(drop=True)


def species_frame(df: pd.DataFrame, spec: SpeciesSpec, sigma_col: str | None = None) -> pd.DataFrame:
    """Rows with a finite value for spec.use_key, with class index and soft labels."""
    if spec.use_key not in df.columns:
        return df.iloc[0:0].assign(value=[], ref_class=[])
    v = pd.to_numeric(df[spec.use_key], errors="coerce")
    out = df[np.isfinite(v) & (v >= 0)].copy()
    out["value"] = v[out.index].astype(np.float64)
    out["ref_class"] = concentration_to_class(out["value"].to_numpy(), spec.edges)
    out["log10_value"] = np.log10(np.maximum(out["value"].to_numpy(), spec.detection_floor))
    # per-row reference error (written by the ingestion package as {use_key}_sigma_dex,
    # e.g. IFCB vs. light microscopy); NaN -> species default
    sigma_col = sigma_col or f"{spec.use_key}_sigma_dex"
    if sigma_col in out.columns:
        sig = pd.to_numeric(out[sigma_col], errors="coerce").to_numpy(dtype=np.float64)
        sig = np.where(np.isfinite(sig) & (sig > 0), sig, spec.measurement_sigma_dex)
    else:
        sig = spec.measurement_sigma_dex
    T = measurement_soft_labels(out["value"].to_numpy(), spec.edges, sig, spec.detection_floor)
    for c in range(spec.n_classes):
        out[f"ref_soft_{c}"] = T[:, c]
    return out.reset_index(drop=True)


def _stable_unit(key: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def assign_splits(df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Grouped, deterministic split assignment.

    cfg (uq.splits):
      block: "30D"            time block length (pandas offset) for grouping
      group_by: [station, block]   or [block] for regional blocking
      fractions: {train: 0.6, calibration: 0.2, test: 0.2}
      test_years: [2024]      optional temporal holdout (overrides fractions)
      seed: 13
      n_folds: 5              cross-fit folds over train groups
      encoder_train_window: [YYYYMMDD, YYYYMMDD]   optional SIT-FUSE encoder
                              training dates (train_min_date/train_max_date).
                              Samples inside it are moved to `train`, so the
                              calibration and test splits only contain scenes
                              the encoder never saw (as the fire/smoke
                              calibration_groups mode). Matters most for
                              head-output / MC / OOD features, which are
                              in-distribution by construction on those scenes.
    Adds split_group, split, fold (and encoder_seen when a window is given).
    """
    cfg = cfg or {}
    block = cfg.get("block", "30D")
    group_by = cfg.get("group_by", ["station", "block"])
    fr = cfg.get("fractions", {"train": 0.6, "calibration": 0.2, "test": 0.2})
    seed = int(cfg.get("seed", 13))
    n_folds = int(cfg.get("n_folds", 5))
    test_years = set(int(y) for y in cfg.get("test_years", []) or [])

    out = df.copy()
    t0 = pd.Timestamp(cfg.get("block_origin", "2000-01-01"))
    days = int(pd.Timedelta(block).days) or 30
    out["time_block"] = ((out["Datetime"] - t0).dt.days // days).astype(int)
    parts = []
    for g in group_by:
        parts.append(out["Location_Code"].astype(str) if g == "station" else out["time_block"].astype(str))
    out["split_group"] = pd.Series(["|".join(p) for p in zip(*parts)], index=out.index) if parts else "all"

    edges = np.cumsum([fr.get("train", 0.6), fr.get("calibration", 0.2), fr.get("test", 0.2)])
    edges = edges / edges[-1]
    split, fold = [], []
    for g, yr in zip(out["split_group"], out["Datetime"].dt.year):
        u = _stable_unit(g, seed)
        if test_years:
            # temporal holdout; remaining groups split train/calibration by ratio
            if yr in test_years:
                s = "test"
            else:
                s = "train" if u < edges[0] / max(edges[1], EPS) else "calibration"
        else:
            s = "train" if u < edges[0] else ("calibration" if u < edges[1] else "test")
        split.append(s)
        fold.append(int(_stable_unit(g, seed + 1) * n_folds))
    out["split"] = split
    out["fold"] = fold
    win = cfg.get("encoder_train_window")
    if win:
        lo, hi = (pd.Timestamp(str(w)) for w in win)
        dt = pd.to_datetime(out["Datetime"])
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_localize(None)
        seen = (dt.dt.normalize() >= lo) & (dt.dt.normalize() <= hi)
        out["encoder_seen"] = seen.to_numpy()
        out.loc[seen & out["split"].isin(["calibration", "test"]), "split"] = "train"
    return out
