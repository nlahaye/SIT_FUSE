"""
Column-mapped reader for downloaded tabular cruise/monitoring data that have
no stable public API (NWFSC 2015 DA-event / Shimada HAB surveys, SBC-LTER
Plumes and Blooms, SCCWRP Bight'18 HAB, CDPH phytoplankton monitoring, HABDAC
exports). Each source is described in the ingestion YAML:

  - kind: local_table
    name: nwfsc_2015
    path: data/2015_west_coast_domoic_acid_event_seawater_data.csv
    method: lm_settled           # provenance; sets default sigma_dex
    time: {column: SAMPLE_DATE, tz: UTC}      # or {date: ..., clock: ...}
    location_code: {column: STATION, prefix: NWFSC_}
    columns:                     # native -> CalHABMAP (or extra_*) column
      LATITUDE: latitude
      LONGITUDE: longitude
      DEPTH_M: depth
      PN_LARGE_CELLS_L: Pseudo_nitzschia_seriata_group
      PN_SMALL_CELLS_L: Pseudo_nitzschia_delicatissima_group
      PDA_NG_L: pDA
    scale: {pDA: 0.001}          # ng/L -> ng/mL
    max_depth_m: 5
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sit_fuse.pipelines.habs.insitu.calhabmap_format import DEFAULT_SIGMA_DEX, to_calhabmap


def read_local_table(spec: dict[str, Any]) -> pd.DataFrame:
    p = spec["path"]
    raw = pd.read_excel(p, sheet_name=spec.get("sheet", 0)) if str(p).endswith((".xlsx", ".xls")) \
        else pd.read_csv(p, **spec.get("read_kw", {}))
    out = pd.DataFrame(index=raw.index)
    for src, dst in spec.get("columns", {}).items():
        if src in raw:
            out[dst] = pd.to_numeric(raw[src], errors="coerce") if dst not in ("Location_Code",) else raw[src]
    for c, f in (spec.get("scale") or {}).items():
        if c in out:
            out[c] = out[c] * float(f)
    t = spec.get("time", {})
    if "column" in t:
        st = pd.to_datetime(raw[t["column"]], errors="coerce", format=t.get("format", "mixed"))
    else:
        st = pd.to_datetime(raw[t["date"]].astype(str) + " " + raw[t["clock"]].astype(str),
                            errors="coerce", format="mixed")
    tz = t.get("tz", "UTC")
    st = st.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT") if st.dt.tz is None else st
    out["sample_time_utc"] = st.dt.tz_convert("UTC")
    lc = spec.get("location_code", {})
    if "column" in lc:
        out["Location_Code"] = lc.get("prefix", "") + raw[lc["column"]].astype(str).str.replace(" ", "_")
    else:
        out["Location_Code"] = lc.get("value", spec.get("name", "LOCAL"))
    if spec.get("max_depth_m") is not None and "depth" in out:
        out = out[~(out["depth"] > float(spec["max_depth_m"]))]
    method = spec.get("method", "microscopy")
    out["source"] = spec.get("name", "local_table")
    out["source_sample_id"] = out["source"] + ":" + out.index.astype(str)
    out["method"] = method
    out["spatial_support"] = spec.get("spatial_support", "point")
    out["n_pooled"] = 1
    sig = spec.get("sigma_dex", DEFAULT_SIGMA_DEX.get(method))
    if sig is not None:
        for c in ("Total_Phytoplankton", "Pseudo_nitzschia_seriata_group",
                  "Pseudo_nitzschia_delicatissima_group", "pDA"):
            if c in out:
                out[f"{c}_sigma_dex"] = sig
    out = out[out["sample_time_utc"].notna() & np.isfinite(out.get("latitude", np.nan))]
    return to_calhabmap(out)
