"""
Targeted QC fixes for the combined HAB in-situ tables (combined_full, combined_augmented[_strict]).

  1. value_overrides   set listed values to NaN (default: SIO 2023-04-10 Alexandrium_spp
                       254,206 cells/L, ~15x any other California count; the rest of that sample
                       is kept, it is consistent with the April 2023 dinoflagellate bloom, Chl 52 ug/L)
  2. date_fixes        re-date rows whose SampleID encodes a different day (CPP CP241216 filed as 12/08/24)
  3. duplicates        combined_full holds two overlapping copies of the pier table (rows with and
                       without its original index). Rows with the same Location_Code, day and SampleID
                       collapse to one: the indexed copy wins, gaps are filled from the other copy.
                       Distinct SampleIDs on one day (e.g. BML replicates) are kept.
  4. ifcb_unmapped     IFCB datasets on the 51-class CalOOS classifier (Santa Cruz wharf, MBARI buoy,
                       Bodega) label Alexandrium as Alexandrium_singlet and also use Lingulodinium,
                       Ceratium and Cochlodinium names that ifcb_dashboard.CLASS_MAP does not map, so
                       those columns are structural zeros. Alexandrium and all dinoflagellate columns
                       are set to NaN (not measured) at those sites; no remapping/replay (Nov 2026).

    python -m sit_fuse.pipelines.habs.insitu.qc_fixes in.csv out.csv [--log fixes.csv]

Columns that are absent (e.g. source / method in the strict table) skip the matching fix.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

VALUE_OVERRIDES = [
    {"Location_Code": "SIO", "date": "2023-04-10", "column": "Alexandrium_spp", "min_value": 1e5,
     "reason": "implausible Alexandrium count (254,206 cells/L)"},
]
DATE_FIXES = [
    {"Location_Code": "CPP", "SampleID": "CP241216", "date": "2024-12-16",
     "reason": "SampleID encodes 2024-12-16; filed as 2024-12-08"},
]
IFCB_51CLASS = {"santa-cruz-municipal-wharf": "IFCB104", "mbari-power-buoy": "IFCB161",
                "bodega-marine-lab": "IFCB163"}
# Decision (Nov 2026): at the 51-class sites, Alexandrium and all dinoflagellate columns are "not measured"
# (Alexandrium_singlet, Lingulodinium, Ceratium, Cochlodinium etc. are unmapped; no class remapping or replay).
IFCB_UNMAPPED_COLUMNS = ["Alexandrium_spp", "Akashiwo_sanguinea", "Dinophysis_spp", "Lingulodinium_polyedra",
                         "Prorocentrum_spp", "Ceratium_spp", "Cochlodinium_spp", "Gymnodinium_spp",
                         "Other_Dinoflagellates"]
INDEX_COLS = ("Unnamed: 0", "Unnamed: 0.1")


def _day(df: pd.DataFrame) -> pd.Series:
    t = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for c, fmt in (("sample_time_utc", None), ("Datetime", None), ("time", "%m/%d/%y")):
        if c in df.columns:
            v = pd.to_datetime(df[c], format=fmt, errors="coerce", utc=True)
            t = t.fillna(v)
    return t.dt.strftime("%Y-%m-%d")


def _sid(df: pd.DataFrame) -> pd.Series:
    if "SampleID" not in df.columns:
        return pd.Series("", index=df.index)
    return df["SampleID"].astype(str).str.replace(r"\.0$", "", regex=True)


def apply_fixes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    logs = []
    day = _day(df)
    sid = _sid(df)

    for o in VALUE_OVERRIDES:
        if o["column"] not in df.columns:
            continue
        v = pd.to_numeric(df[o["column"]], errors="coerce")
        m = (df["Location_Code"] == o["Location_Code"]) & (day == o["date"]) & (v >= o["min_value"])
        for i in df.index[m]:
            logs.append({"fix": "value_override", "row": i, "Location_Code": o["Location_Code"], "date": o["date"],
                         "column": o["column"], "old": v[i], "new": np.nan, "reason": o["reason"]})
        df.loc[m, o["column"]] = np.nan

    for f in DATE_FIXES:
        m = (df["Location_Code"] == f["Location_Code"]) & (sid == f["SampleID"]) & (day != f["date"])
        for i in df.index[m]:
            logs.append({"fix": "date_fix", "row": i, "Location_Code": f["Location_Code"], "date": day[i],
                         "column": "time", "old": df.at[i, "time"] if "time" in df else None,
                         "new": f["date"], "reason": f["reason"]})
        if m.any():
            ts = pd.Timestamp(f["date"])
            if "time" in df.columns:
                df.loc[m, "time"] = ts.strftime("%m/%d/%y")
            for c in ("Datetime", "sample_time_utc"):
                mc = m & df[c].notna() if c in df.columns else None
                if mc is not None and mc.any():
                    df[c] = df[c].astype(object)
                    df.loc[mc, c] = ts.strftime("%Y-%m-%dT00:00:00Z" if c == "sample_time_utc" else "%Y-%m-%d")
            day = _day(df)

    # duplicates within the pier table (needs provenance + SampleID, else IFCB and pier rows that share a
    # Location_Code could merge)
    pier = pd.Series(False, index=df.index)
    if "SampleID" in df.columns and ("source" in df.columns or "method" in df.columns):
        pier[:] = True
        if "source" in df.columns:
            pier &= df["source"] == "combined_full"
        if "method" in df.columns:
            pier &= df["method"].astype(str) != "ifcb"
    elif "SampleID" in df.columns and "source_sample_id" not in df.columns and \
            df.columns[0].startswith("Unnamed"):
        pier[:] = True          # raw combined_full table (pier rows only)
    else:
        log.warning("duplicates: no source / method / SampleID columns; duplicate merge skipped")
    idx_col = next((c for c in INDEX_COLS if c in df.columns and c != "Unnamed: 0.1"), None)
    key = df["Location_Code"].astype(str) + "|" + day.fillna("") + "|" + sid
    drop = []
    data_cols = [c for c in df.columns if c not in INDEX_COLS]
    for k, g in df[pier].groupby(key[pier]):
        if len(g) < 2:
            continue
        order = g.index.tolist()
        if idx_col is not None:
            order = g[idx_col].notna().sort_values(ascending=False, kind="stable").index.tolist()
        keep, rest = order[0], order[1:]
        for j in rest:
            for c in data_cols:
                a, b = df.at[keep, c], df.at[j, c]
                if pd.isna(a) and not pd.isna(b):
                    df.at[keep, c] = b
                elif c != "SampleID" and not pd.isna(a) and not pd.isna(b) and str(a) != str(b):
                    logs.append({"fix": "duplicate_conflict", "row": keep, "Location_Code": k.split("|")[0],
                                 "date": k.split("|")[1], "column": c, "old": b, "new": a,
                                 "reason": "kept indexed copy"})
        logs.append({"fix": "duplicate_dropped", "row": rest, "Location_Code": k.split("|")[0],
                     "date": k.split("|")[1], "column": "", "old": len(rest), "new": "", "reason": "merged"})
        drop.extend(rest)
    df = df.drop(index=drop)

    # IFCB structural zeros (51-class classifier)
    if "source_sample_id" in df.columns:
        ssid = df["source_sample_id"].astype(str)
        m = pd.Series(False, index=df.index)
        for ds, inst in IFCB_51CLASS.items():
            m |= ssid.str.startswith(ds + ":") | ssid.str.endswith("_" + inst)
        cols = [c for c in IFCB_UNMAPPED_COLUMNS if c in df.columns]
        if m.any() and cols:
            logs.append({"fix": "ifcb_unmapped_to_nan", "row": int(m.sum()), "Location_Code":
                         ",".join(sorted(df.loc[m, "Location_Code"].astype(str).unique())), "date": "",
                         "column": ",".join(cols), "old": "0", "new": np.nan,
                         "reason": "51-class IFCB classifier: Alexandrium/dinoflagellates not measured"})
            df.loc[m, cols] = np.nan
    return df, pd.DataFrame(logs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--log", default=None, help="CSV of every change (default: <out>.qc_log.csv)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = pd.read_csv(a.inp, low_memory=False)
    out, lg = apply_fixes(df)
    out.to_csv(a.out, index=False)
    lg.to_csv(a.log or a.out.rsplit(".", 1)[0] + ".qc_log.csv", index=False)
    summ = lg.groupby("fix").size().to_dict() if len(lg) else {}
    log.info("%s: %d -> %d rows; %s", a.inp, len(df), len(out), summ)


if __name__ == "__main__":
    main()
