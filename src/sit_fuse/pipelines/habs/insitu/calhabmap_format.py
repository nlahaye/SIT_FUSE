"""
Writer/validator for the CalHABMAP-style in-situ table consumed by
utils.insitu_hab_to_multi_hist and the HAB UQ pipeline
(config/postprocess/insitu_class_match/final/combined_full.csv).

Exact layout of the reference file: 42 columns in CALHABMAP_COLUMNS order,
CRLF line endings, `time` as M/D/YY, SampleID as YYYYMMDD, concentrations in
cells/L and pDA/tDA/dDA in ng/mL (SCCOOS ERDDAP units).

Non-strict mode appends provenance columns after `Datetime`; both
insitu_hab_to_multi_hist and hab_insitu.load_insitu ignore unknown columns:
  source            e.g. calhabmap_erddap, ifcb_dashboard, calcofi_venrick, seabass_pace_pax
  source_sample_id  native id (ERDDAP SampleID, IFCB bin pid, SeaBASS file, ...)
  sample_time_utc   full ISO timestamp (used for overpass-time weighting)
  method            microscopy | ifcb | lm_settled | hplc | fcm | elisa | lcms
  spatial_support   point | region_pooled | underway
  n_pooled          number of native samples aggregated into the row
  {use_key}_sigma_dex  per-row log10 reference error (used by hab_insitu.species_frame)
  extra_*           additional speciated / size-class concentrations
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CALHABMAP_COLUMNS = [
    "Location_Code", "latitude", "longitude", "depth", "SampleID", "time", "Temp", "Air_Temp",
    "Salinity", "Chl_Volume_Filtered", "Chl1", "Chl2", "Avg_Chloro", "Phaeo1", "Phaeo2",
    "Avg_Phaeo", "Phosphate", "Silicate", "Nitrite", "Nitrite_Nitrate", "Ammonium", "Nitrate",
    "DA_Volume_Filtered", "pDA", "tDA", "dDA", "Volume_Settled_for_Counting",
    "Akashiwo_sanguinea", "Alexandrium_spp", "Dinophysis_spp", "Lingulodinium_polyedra",
    "Prorocentrum_spp", "Pseudo_nitzschia_delicatissima_group", "Pseudo_nitzschia_seriata_group",
    "Ceratium_spp", "Cochlodinium_spp", "Gymnodinium_spp", "Other_Diatoms",
    "Other_Dinoflagellates", "Total_Phytoplankton", "Sample Depth", "Datetime",
]

TAXA_COLUMNS = CALHABMAP_COLUMNS[27:40]
PROVENANCE_COLUMNS = ["source", "source_sample_id", "sample_time_utc", "method",
                      "spatial_support", "n_pooled"]

# default log10 reference error by method (counting + method error; tune with replicates)
DEFAULT_SIGMA_DEX = {"microscopy": 0.20, "lm_settled": 0.20, "ifcb": 0.25,
                     "ifcb_width_split": 0.35, "region_pooled": 0.40, "elisa": 0.15,
                     "lcms": 0.10, "hplc": 0.10, "fcm": 0.10}


def _fmt_time(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert(None)
    return t.apply(lambda x: f"{x.month}/{x.day}/{x.strftime('%y')}" if pd.notna(x) else "")


def to_calhabmap(df: pd.DataFrame, strict: bool = False,
                 extra_cols: Iterable[str] | None = None,
                 sample_id: str = "date") -> pd.DataFrame:
    """
    Map a frame with at least Location_Code, latitude, longitude and a timestamp
    (`sample_time_utc` or `time`) onto the combined_full.csv schema.
    sample_id: "date" -> YYYYMMDD (reference convention), "native" -> keep SampleID.
    """
    df = df.copy()
    if "sample_time_utc" not in df.columns:
        df["sample_time_utc"] = pd.to_datetime(df["time"], utc=True, errors="coerce", format="mixed")
    st = pd.to_datetime(df["sample_time_utc"], utc=True, errors="coerce", format="mixed")
    out = pd.DataFrame(index=df.index)
    for c in CALHABMAP_COLUMNS:
        out[c] = df[c] if c in df.columns else np.nan
    out["time"] = _fmt_time(st)
    if sample_id == "date" or out["SampleID"].isna().all():
        out["SampleID"] = st.dt.strftime("%Y%m%d")
    out["Datetime"] = ""            # empty in the reference file; insitu code derives it from `time`
    if not strict:
        df["sample_time_utc"] = st.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        keep = list(PROVENANCE_COLUMNS)
        keep += [c for c in df.columns if c.endswith("_sigma_dex") or c.startswith("extra_")]
        keep += list(extra_cols or [])
        for c in dict.fromkeys(keep):
            if c in df.columns and c not in out.columns:
                out[c] = df[c]
    return out.reset_index(drop=True)


def write_calhabmap(df: pd.DataFrame, path: str | Path, strict: bool = False, **kw) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df if list(df.columns[:42]) == CALHABMAP_COLUMNS else to_calhabmap(df, strict=strict, **kw)
    if strict:
        out = out[CALHABMAP_COLUMNS]
    out.to_csv(path, index=False, lineterminator="\r\n", float_format="%.6g")
    return path


def validate_calhabmap(path_or_df, strict: bool = True) -> list[str]:
    """Return a list of problems (empty == conforms)."""
    df = pd.read_csv(path_or_df) if not isinstance(path_or_df, pd.DataFrame) else path_or_df
    issues = []
    cols = list(df.columns)
    if cols[:42] != CALHABMAP_COLUMNS:
        issues.append("first 42 columns differ from CALHABMAP_COLUMNS")
    if strict and len(cols) != 42:
        issues.append(f"strict: {len(cols)} columns")
    t = pd.to_datetime(df["time"], format="%m/%d/%y", errors="coerce")
    if t.isna().any():
        issues.append(f"{int(t.isna().sum())} rows with unparseable M/D/YY time")
    for c in TAXA_COLUMNS + ["pDA"]:
        v = pd.to_numeric(df[c], errors="coerce")
        if (v < 0).any():
            issues.append(f"negative values in {c}")
    lat, lon = pd.to_numeric(df["latitude"], errors="coerce"), pd.to_numeric(df["longitude"], errors="coerce")
    if (~lat.between(-90, 90)).any() or (~lon.between(-180, 180)).any():
        issues.append("lat/lon out of range or missing")
    return issues


def merge_sources(frames: list[pd.DataFrame], priority: list[str] | None = None,
                  dedup_on=("Location_Code", "_date", "method")) -> pd.DataFrame:
    """
    Concatenate formatted frames. Duplicates of the same station/date/method
    (e.g. the attached combined_full rows re-pulled from ERDDAP) keep the
    highest-priority source; different methods at the same station/date are
    retained (they share Location_Code, hence the same leakage-safe split group).
    """
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame(columns=CALHABMAP_COLUMNS)
    df = pd.concat(frames, ignore_index=True, sort=False)
    for c in ("source", "method"):
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("")
    df.loc[df["method"] == "", "method"] = "microscopy"     # legacy combined_full rows
    df["_date"] = pd.to_datetime(df["time"], format="%m/%d/%y", errors="coerce").dt.strftime("%Y%m%d")
    prio = {s: i for i, s in enumerate(priority or [])}
    df["_p"] = df["source"].map(lambda s: prio.get(s, len(prio)))
    df = df.sort_values("_p", kind="stable")
    # cross-source dedup only: rows within one source (replicates, depths, IFCB bins) are kept
    best = df.groupby(list(dedup_on), dropna=False)["_p"].transform("min")
    df = df[df["_p"] == best]
    df["_t"] = pd.to_datetime(df["time"], format="%m/%d/%y", errors="coerce")
    df = df.sort_values(["_t", "Location_Code"], kind="stable").drop(columns=["_p", "_t", "_date"])
    cols = CALHABMAP_COLUMNS + [c for c in df.columns if c not in CALHABMAP_COLUMNS]
    return df[cols].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Units and cleaning applied on ingestion
# ---------------------------------------------------------------------------

DA_COLUMNS = ["pDA", "tDA", "dDA"]
EXPECTED_UNITS = {**{c: "cells/L" for c in TAXA_COLUMNS}, **{c: "ng/mL" for c in DA_COLUMNS},
                  "depth": "m"}
# normalised unit string -> factor to the expected unit
UNIT_FACTORS = {
    "cells/L": {"cells/l": 1.0, "cellsl-1": 1.0, "cell/l": 1.0, "cells/liter": 1.0,
                "cells/ml": 1e3, "cellsml-1": 1e3, "cell/ml": 1e3, "number/ml": 1e3},
    "ng/mL": {"ng/ml": 1.0, "ngml-1": 1.0, "ug/l": 1.0, "ugl-1": 1.0, "µg/l": 1.0,
              "ng/l": 1e-3, "ngl-1": 1e-3, "pg/ml": 1e-3},
    "m": {"m": 1.0, "meters": 1.0, "metres": 1.0},
}


def _norm_unit(u) -> str:
    s = str(u).strip().lower().replace(" ", "").replace("^", "").replace("*", "")
    return s.replace("litre", "l").replace("liter", "l").replace("milli", "m").replace("per", "/")


def unit_factor(col: str, unit, source: str = "") -> float:
    """Factor converting `col` from `unit` to EXPECTED_UNITS[col]; raises on unknown/missing units."""
    want = EXPECTED_UNITS[col]
    if unit is None or (isinstance(unit, float) and np.isnan(unit)) or str(unit).strip() == "":
        raise ValueError(f"[{source}] {col}: no units given (expected {want})")
    f = UNIT_FACTORS[want].get(_norm_unit(unit))
    if f is None:
        raise ValueError(f"[{source}] {col}: unsupported units '{unit}' (expected {want})")
    return f


def harmonize_units(df: pd.DataFrame, units: dict, source: str = "") -> pd.DataFrame:
    """Check/convert every EXPECTED_UNITS column present in df using the supplied units map."""
    df = df.copy()
    for c in EXPECTED_UNITS:
        if c not in df.columns or pd.to_numeric(df[c], errors="coerce").notna().sum() == 0:
            continue
        f = unit_factor(c, units.get(c), source)
        if f != 1.0:
            df[c] = pd.to_numeric(df[c], errors="coerce") * f
            print(f"[units] {source} {c}: '{units.get(c)}' x{f:g} -> {EXPECTED_UNITS[c]}")
    return df


def negatives_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Concentrations cannot be negative: set taxa / DA / extra_* values < 0 to NaN."""
    df = df.copy()
    cols = [c for c in TAXA_COLUMNS + DA_COLUMNS if c in df.columns]
    cols += [c for c in df.columns if c.startswith("extra_") and not c.endswith("_sigma_dex")]
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        bad = v < 0
        if bad.any():
            print(f"[clean] {c}: {int(bad.sum())} negative values -> NaN")
            df[c] = v.mask(bad)
    return df


def _sample_key(df: pd.DataFrame) -> pd.Series:
    sid = df["SampleID"].astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    if "source_sample_id" in df.columns:          # IFCB bin pid / ERDDAP id: unique per sample
        sid = df["source_sample_id"].astype("string").fillna(sid)
    return sid


def collapse_repeated_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge rows that are the same physical sample (same source, Location_Code, date, method and
    sample id; SampleID 20180604 == 20180604.0). Values come from the most complete copy, with
    its gaps filled from the others, so a chemistry-only copy and a counts copy become one row. Rows without a sample id, and
    replicates with distinct ids, are kept.
    """
    if not len(df):
        return df
    d = df.copy()
    sid = _sample_key(d)
    date = pd.to_datetime(d["time"], format="%m/%d/%y", errors="coerce").dt.strftime("%Y%m%d")
    key = (d.get("source", pd.Series("", index=d.index)).astype("string").fillna("") + "|" +
           d["Location_Code"].astype("string").fillna("") + "|" + date.fillna("") + "|" +
           d.get("method", pd.Series("", index=d.index)).astype("string").fillna("") + "|" +
           sid.fillna(""))
    key = key.where(sid.notna(), "row" + pd.Series(d.index.astype(str), index=d.index))
    n_rep = int(key.duplicated().sum())
    if not n_rep:
        return df
    num = [c for c in TAXA_COLUMNS + DA_COLUMNS if c in d.columns]
    rep = key.duplicated(keep=False)
    v = d.loc[rep, num].apply(pd.to_numeric, errors="coerce").round(3)
    conflict = int((v.groupby(key[rep]).nunique() > 1).any(axis=1).sum())
    d["_k"], d["_o"] = key.values, np.arange(len(d))
    # most complete copy first (in combined_full it is the revised record), then fill its gaps
    d["_n"] = -d[[c for c in CALHABMAP_COLUMNS if c in d.columns]].notna().sum(axis=1)
    d = d.sort_values(["_n", "_o"], kind="stable").drop(columns="_n")
    out = d.groupby("_k", sort=False, dropna=False).first()
    out = out.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    print(f"[clean] collapsed {n_rep} repeated sample rows "
          f"({conflict} groups with differing values; most complete copy kept)")
    return out[[c for c in df.columns]]


def filter_depth(df: pd.DataFrame, max_depth_m: float | None, keep_missing: bool = True) -> pd.DataFrame:
    """Drop samples deeper than max_depth_m (depth, else Sample Depth). Missing depth kept by default."""
    if max_depth_m is None or not len(df):
        return df
    z = pd.to_numeric(df["depth"], errors="coerce")
    if "Sample Depth" in df.columns:
        z = z.fillna(pd.to_numeric(df["Sample Depth"], errors="coerce"))
    keep = (z <= max_depth_m) | (z.isna() & keep_missing)
    if (~keep).any():
        by = df.loc[~keep, "source"].value_counts().to_dict() if "source" in df.columns else {}
        print(f"[clean] depth > {max_depth_m} m: dropped {int((~keep).sum())} rows {by}")
    return df[keep].reset_index(drop=True)
