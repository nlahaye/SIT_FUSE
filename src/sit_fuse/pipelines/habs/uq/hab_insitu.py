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
    EPS, SpeciesSpec, _log_edges, concentration_to_class, in_windows, measurement_soft_labels)


# CalHABMAP / SCCOOS-CeNCOOS pier network: the original combined_full table and
# its ERDDAP extension written by pull_insitu_hab (insitu_sources.example.yaml)
SOURCE_PRESETS = {"oos_piers": ["combined_full", "calhabmap_erddap"]}


def _as_list(v):
    if v is None:
        return None
    return [v] if isinstance(v, str) else list(v)


def filter_insitu(df: pd.DataFrame, flt: dict[str, Any] | None) -> pd.DataFrame:
    """
    Subset in-situ rows by provenance (uq.insitu_filter or an evaluation subset).

    flt keys (all optional, combined with AND):
      sources / exclude_sources            `source` values, or a preset name
                                           (oos_piers = combined_full + calhabmap_erddap)
      methods / exclude_methods            `method` values (microscopy, ifcb, elisa, ...)
      location_codes / exclude_location_codes
    Source and method filters need the provenance columns written by the
    non-strict pull_insitu_hab table (combined_augmented.csv); the strict
    42-column table has none, so those filters raise instead of silently
    keeping every row. Station filters work on either table.
    """
    if not flt:
        return df
    keep = insitu_mask(df, flt)
    return df[keep].reset_index(drop=True)


def insitu_mask(df: pd.DataFrame, flt: dict[str, Any] | None) -> np.ndarray:
    """Boolean row mask for filter_insitu (also used on sample-level evaluation meta)."""
    keep = np.ones(len(df), bool)
    for col, inc, exc in (("source", "sources", "exclude_sources"),
                          ("method", "methods", "exclude_methods"),
                          ("Location_Code", "location_codes", "exclude_location_codes")):
        a, b = _as_list((flt or {}).get(inc)), _as_list((flt or {}).get(exc))
        if a is None and b is None:
            continue
        if col not in df.columns or (col != "Location_Code" and not df.attrs.get(f"has_{col}", True)):
            raise ValueError(
                f"in-situ filter on '{col}' but the table has no {col} column (strict CSV). "
                "Point uq.insitu_file (or --insitu) at the non-strict combined_augmented.csv, "
                "or filter by location_codes.")
        expand = (lambda v: [x for k in v for x in SOURCE_PRESETS.get(k, [k])]) if col == "source" else (lambda v: v)
        vals = df[col].astype(str)
        if a is not None:
            keep &= vals.isin(expand(a)).to_numpy()
        if b is not None:
            keep &= ~vals.isin(expand(b)).to_numpy()
    return keep


def load_insitu(path: str, max_depth_m: float | None = 1.0,
                start_date=None, end_date=None, filters: dict[str, Any] | None = None,
                date_windows=None) -> pd.DataFrame:
    """
    Read the combined_full.csv format (or xlsx) exactly as
    insitu_hab_to_multi_hist does (Datetime = pd.to_datetime(time),
    latitude/longitude -> Latitude/Longitude) and add UQ bookkeeping columns.
    `filters` (uq.insitu_filter) is applied after sample_uid is assigned, so
    uids match an unfiltered load of the same table. `date_windows`
    ([(start, end), ...], uq.date_windows via insitu_date_kwargs) keeps only
    samples on days inside a window; it is applied before sample_uid, so
    every stage must pass the same windows (insitu_date_kwargs does).
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
    if date_windows:
        df = df[in_windows(df["Datetime"], date_windows)]
    if "Location_Code" not in df.columns:
        df["Location_Code"] = "UNK"
    has_src, has_mth = "source" in df.columns, "method" in df.columns
    if not has_src:
        df["source"] = "combined_full"
    if not has_mth:
        df["method"] = "microscopy"
    df["source"] = df["source"].fillna("combined_full")
    df["method"] = df["method"].fillna("microscopy")
    df["sample_uid"] = [
        hashlib.md5(f"{s}|{d}|{la:.4f}|{lo:.4f}|{i}".encode()).hexdigest()[:12]
        for i, (s, d, la, lo) in enumerate(zip(df["Location_Code"], df["Datetime"].astype(str),
                                               df["Latitude"], df["Longitude"]))]
    df = df.reset_index(drop=True)
    df.attrs.update(has_source=has_src, has_method=has_mth)
    return filter_insitu(df, filters) if filters else df


def insitu_filter_config(yml: dict[str, Any]) -> dict[str, Any] | None:
    """uq.insitu_filter; a bare preset string (e.g. oos_piers) means sources: [preset]."""
    f = (yml.get("uq") or {}).get("insitu_filter")
    return {"sources": [f]} if isinstance(f, str) else (f or None)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def _sum_min1(df: pd.DataFrame, cols) -> pd.Series:
    """Row sum, NaN only when every column is NaN (a missing group counts as absent)."""
    return pd.concat([_num(df, c) for c in cols], axis=1).sum(axis=1, min_count=1)


def _component_sigma(df: pd.DataFrame, cols, default: float) -> pd.Series:
    sig = [_num(df, f"{c}_sigma_dex") for c in cols]
    return pd.concat(sig, axis=1).max(axis=1).fillna(default)


def add_derived_target(df: pd.DataFrame, spec: SpeciesSpec) -> pd.DataFrame:
    """
    Add spec.use_key (and {use_key}_sigma_dex, {use_key}_upper) for derived targets.

      sum          value = sum of spec.derived["columns"] (total Pseudo-nitzschia; IFCB and
                   microscopy groups add to the genus total)
      cellular_da  value = pDA [ng/mL = ug/L] * 1e6 / PN_total [cells/L]  (pg DA per cell),
                   only where PN_total >= min_pn_cells. A pDA non-detect (0) gives an upper
                   bound pda_lod * 1e6 / PN_total, kept (as {use_key}_upper, value NaN) when
                   the bound is below max_censored_upper, otherwise the sample is ambiguous
                   and dropped. Sigma adds the pDA and PN errors in quadrature.
    A column already present in the table (e.g. written by the ingestion package) is kept.
    """
    d = spec.derived or {}
    if not d or spec.use_key in df.columns:
        return df
    df = df.copy()
    kind = d.get("kind")
    if kind == "sum":
        cols = d.get("columns") or []
        df[spec.use_key] = _sum_min1(df, cols)
        df[f"{spec.use_key}_sigma_dex"] = _component_sigma(df, cols, spec.measurement_sigma_dex)
    elif kind == "cellular_da":
        pn = _sum_min1(df, d.get("pn") or [])
        pda = _num(df, d.get("pda", "pDA"))
        ok = np.isfinite(pn) & (pn >= float(d.get("min_pn_cells", 1000.0))) & np.isfinite(pda) & (pda >= 0)
        scale = float(d.get("scale", 1e6))
        det = ok & (pda > 0)
        nd = ok & (pda <= 0)
        val = pd.Series(np.nan, index=df.index)
        val[det] = pda[det] * scale / pn[det]
        upper = pd.Series(np.nan, index=df.index)
        ub = float(d.get("pda_lod", 0.02)) * scale / pn
        keep_nd = nd & (ub < float(d.get("max_censored_upper", 10.0)))
        if d.get("censored", "soft") == "soft":
            upper[keep_nd] = ub[keep_nd]
        df[spec.use_key] = val
        df[f"{spec.use_key}_upper"] = upper
        s_pn = _component_sigma(df, d.get("pn") or [], 0.2)
        s_da = _num(df, f"{d.get('pda', 'pDA')}_sigma_dex").fillna(0.15)
        df[f"{spec.use_key}_sigma_dex"] = np.sqrt(s_pn ** 2 + s_da ** 2)
    else:
        raise ValueError(f"{spec.use_key}: unknown derived kind {kind!r}")
    return df


def censored_soft_labels(upper, edges, floor) -> np.ndarray:
    """Soft labels for X <= upper: log-uniform mass between the detection floor and the bound."""
    le = np.asarray(_log_edges(edges, floor))
    lo = np.log10(floor)
    hi = np.log10(np.maximum(np.asarray(upper, float), floor * 1.0001))
    a = np.clip(le[None, :-1], lo, hi[:, None])
    b = np.clip(le[None, 1:], lo, hi[:, None])
    T = np.clip(b - a, 0, None)
    T[:, 0] += (hi <= lo)            # bound at / below the floor: all mass in bin 0
    return T / np.maximum(T.sum(1, keepdims=True), EPS)


def species_frame(df: pd.DataFrame, spec: SpeciesSpec, sigma_col: str | None = None) -> pd.DataFrame:
    """
    Rows with a finite value for spec.use_key, with class index and soft labels.
    Derived targets (spec.derived) are computed first. Rows with only an upper bound
    ({use_key}_upper, e.g. cDA from a pDA non-detect) are kept with censored=True: value is
    the geometric midpoint of [floor, bound] and the soft labels are log-uniform below the bound.
    """
    df = add_derived_target(df, spec)
    if spec.use_key not in df.columns:
        return df.iloc[0:0].assign(value=[], ref_class=[])
    v = pd.to_numeric(df[spec.use_key], errors="coerce")
    up = _num(df, f"{spec.use_key}_upper")
    cens = ~(np.isfinite(v) & (v >= 0)) & np.isfinite(up) & (up > 0)
    v = v.where(~cens, np.sqrt(np.maximum(up, spec.detection_floor) * spec.detection_floor))
    out = df[np.isfinite(v) & (v >= 0)].copy()
    out["value"] = v[out.index].astype(np.float64)
    out["censored"] = cens[out.index].to_numpy()
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
    c = out["censored"].to_numpy()
    if c.any():
        T[c] = censored_soft_labels(up[out.index].to_numpy()[c], spec.edges, spec.detection_floor)
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
      date_roles: {YYYYMMDD: train|calibration|test|excluded}   set from the
                              pipeline scene_split block (overrides fractions /
                              test_years for the split column)
      encoder_train_window: [YYYYMMDD, YYYYMMDD]   optional SIT-FUSE encoder
                              training dates (train_min_date/train_max_date).
                              Samples inside it are moved to `train`, so the
                              calibration and test splits only contain scenes
                              the encoder never saw (as the fire/smoke
                              calibration_groups mode). Matters most for
                              head-output / MC / OOD features, which are
                              in-distribution by construction on those scenes.
      holdout_locations: [SMP, MBF]   stations held out entirely (true out-of-sample
                              stations). Applied last: their rows go to `test` on
                              every non-excluded day and never to train/calibration;
                              rows on encoder-seen days become `excluded` unless
                              holdout_include_encoder_seen: true. Adds holdout_location.
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
    roles = cfg.get("date_roles")
    if roles:
        # Day-level scene split (hab_scene_split): held-out days -> test, val days -> calibration,
        # buffer / unused days -> excluded; train days keep the train/calibration ratio above.
        day = pd.to_datetime(out["Datetime"]).dt.strftime("%Y%m%d")
        r = day.map(lambda d: roles.get(d, "excluded"))
        tr_share = edges[0] / max(edges[1], EPS)
        u = np.array([_stable_unit(g, seed) for g in out["split_group"]])
        base = np.where(u < tr_share, "train", "calibration")
        has_val = any(v == "calibration" for v in roles.values())
        out["split"] = np.where(r == "train", "train" if has_val else base, r)
    win = cfg.get("encoder_train_window")
    if win:
        lo, hi = (pd.Timestamp(str(w)) for w in win)
        dt = pd.to_datetime(out["Datetime"])
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_localize(None)
        seen = (dt.dt.normalize() >= lo) & (dt.dt.normalize() <= hi)
        out["encoder_seen"] = seen.to_numpy()
        out.loc[seen & out["split"].isin(["calibration", "test"]), "split"] = "train"
    hold = {str(h).strip().upper() for h in (cfg.get("holdout_locations") or [])}
    if hold:
        h = out["Location_Code"].astype(str).str.strip().str.upper().isin(hold).to_numpy()
        out["holdout_location"] = h
        keep = h & (out["split"] != "excluded").to_numpy()
        out.loc[keep, "split"] = "test"
        if win and not cfg.get("holdout_include_encoder_seen", False):
            out.loc[h & out["encoder_seen"].to_numpy(), "split"] = "excluded"
        missing = hold - set(out["Location_Code"].astype(str).str.strip().str.upper())
        if missing:
            import logging
            logging.getLogger("hab-insitu").warning("holdout_locations not in the in-situ table: %s",
                                                     sorted(missing))
    return out
