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
