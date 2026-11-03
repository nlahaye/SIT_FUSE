"""
SeaBASS (.sb) readers for PACE-PAX and other NASA field campaigns
(https://seabass.gsfc.nasa.gov/experiment/PACE-PAX/, DOI 10.5067/SeaBASS/PACE-PAX/DATA001).

SeaBASS file search/download needs an Earthdata-authenticated session, so
these readers operate on local files (download via the SeaBASS file search or
`wget --user ... https://oceandata.sci.gsfc.nasa.gov/...`).

Supported:
  * HPLC pigments -> Avg_Chloro (Tchl-a) + diagnostic-pigment size classes
    (Uitz et al. 2006 coefficients), written as extra_HPLC_{micro,nano,pico}_chl (mg/m3)
    and extra_HPLC_f{micro,nano,pico} fractions.
  * IFCB per-ROI plankton files (SeaBASS plankton & particles format: one row per
    ROI with a taxonomic label and a /volume_sampled_ml header) -> cells/L using
    the same class mapping as ifcb_dashboard.
  * FCM -> extra_<field> cells/L for abundance fields (cells/mL * 1000).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sit_fuse.pipelines.habs.insitu.calhabmap_format import to_calhabmap
from sit_fuse.pipelines.habs.insitu import ifcb_dashboard as ifcb


def read_sb(path: str | Path) -> tuple[dict[str, str], pd.DataFrame]:
    """Parse a SeaBASS file -> (header dict, data frame with missing/BDL -> NaN)."""
    hdr, data, fields = {}, [], None
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("/"):
                if "=" in line:
                    k, v = line[1:].split("=", 1)
                    hdr[k.strip().lower()] = v.strip()
                if line.lower().startswith("/end_header"):
                    fields = hdr["fields"].split(",")
                continue
            if line.startswith("!") or not line.strip() or fields is None:
                continue
            data.append(re.split(r"\s*,\s*" if hdr.get("delimiter", "comma") == "comma" else r"\s+",
                                 line.strip()))
    df = pd.DataFrame(data, columns=fields)
    for k in ("missing", "below_detection_limit", "above_detection_limit"):
        if k in hdr:
            df = df.replace(hdr[k], np.nan)
    for c in df.columns:
        n = pd.to_numeric(df[c], errors="coerce")
        if n.notna().sum() >= 0.5 * df[c].notna().sum() and df[c].notna().any():
            df[c] = n
    return hdr, df


def _times(hdr: dict, df: pd.DataFrame) -> pd.Series:
    if "date" in df and "time" in df:
        return pd.to_datetime(df["date"].astype(str).str[:8] + df["time"].astype(str),
                              format="%Y%m%d%H:%M:%S", utc=True, errors="coerce")
    if "datetime" in df:
        return pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    if {"year", "month", "day"} <= set(df):
        return pd.to_datetime(df[["year", "month", "day"]].assign(hour=df.get("hour", 0)), utc=True)
    t = pd.to_datetime(hdr.get("start_date", "") + hdr.get("start_time", "").replace("[gmt]", ""),
                       format="%Y%m%d%H:%M:%S", utc=True, errors="coerce")
    return pd.Series(t, index=df.index)


def _latlon(hdr: dict, df: pd.DataFrame):
    lat = df["lat"] if "lat" in df else float(re.sub(r"\[.*\]", "", hdr.get("north_latitude", "nan")))
    lon = df["lon"] if "lon" in df else float(re.sub(r"\[.*\]", "", hdr.get("east_longitude", "nan")))
    return lat, lon


def _base(hdr, df, path, source, method):
    out = pd.DataFrame(index=df.index)
    out["sample_time_utc"] = _times(hdr, df)
    out["latitude"], out["longitude"] = _latlon(hdr, df)
    out["depth"] = df["depth"] if "depth" in df else np.nan
    cruise = hdr.get("cruise", Path(path).stem)
    stn = df["station"].astype(str) if "station" in df else hdr.get("station", "NA")
    out["Location_Code"] = (cruise + "_" + pd.Series(stn, index=df.index).astype(str)).str.replace(" ", "_")
    out["source"], out["method"], out["spatial_support"], out["n_pooled"] = source, method, "point", 1
    out["source_sample_id"] = Path(path).name + ":" + df.index.astype(str)
    return out


def hplc_to_calhabmap(path, source="seabass_pace_pax") -> pd.DataFrame:
    hdr, df = read_sb(path)
    out = _base(hdr, df, path, source, "hplc")
    g = lambda *names: next((df[n] for n in names if n in df), pd.Series(np.nan, index=df.index))
    tchla = g("tot_chl_a", "tchla", "chl_a")
    fuco, perid = g("fuco"), g("perid")
    allo, but, hex_ = g("allo"), g("but_fuco", "but-fuco"), g("hex_fuco", "hex-fuco")
    zea, tchlb = g("zea"), g("tot_chl_b", "chl_b")
    micro = 1.41 * fuco + 1.41 * perid
    nano = 0.60 * allo + 0.35 * but + 1.27 * hex_
    pico = 0.86 * zea + 1.01 * tchlb
    s = (micro + nano + pico).replace(0, np.nan)
    out["Avg_Chloro"] = tchla
    for k, v in (("micro", micro), ("nano", nano), ("pico", pico)):
        out[f"extra_HPLC_f{k}"] = v / s
        out[f"extra_HPLC_{k}_chl"] = (v / s) * tchla
    return to_calhabmap(out)


def ifcb_sb_to_calhabmap(path, source="seabass_pace_pax", label_field: str | None = None,
                         cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """One SeaBASS IFCB file = one sample (bin). Width split uses `width_field` if present."""
    cfg = {**ifcb.DEFAULTS, **(cfg or {})}
    hdr, df = read_sb(path)
    lf = label_field or next((c for c in df.columns if c.lower() in (
        "data_provider_category_automated", "data_provider_category_manual",
        "scientificname_automated", "scientificname")), None)
    if lf is None:
        raise ValueError(f"no taxonomic label field in {path}")
    vol = float(re.sub(r"\[.*\]", "", hdr.get("volume_sampled_ml", "nan")))
    labs = df[lf].astype(str).str.replace(" ", "_")
    # build one-hot "scores" so the dashboard mapping is reused unchanged
    scores = pd.get_dummies(labs).astype(float)
    scores.insert(0, "pid", [f"x_{i + 1:05d}" for i in range(len(df))])
    feats = None
    wf = next((c for c in df.columns if c.lower() in ("minoraxislength", "width", "representativewidth")), None)
    if wf is not None or "equivalent_spherical_diameter" in df:
        feats = pd.DataFrame({"roi_number": np.arange(1, len(df) + 1),
                              cfg["width_feature"]: (df[wf] if wf else np.nan) * cfg["pixels_per_um"],
                              "MajorAxisLength": df.get("majoraxislength", np.nan) * cfg["pixels_per_um"],
                              "EquivDiameter": df.get("equivalent_spherical_diameter", np.nan) * cfg["pixels_per_um"]})
    r = ifcb.bin_counts(scores, feats, cfg)
    row = _base(hdr, df.iloc[:1], path, source, "ifcb")
    for k, v in r.items():
        row[k] = v / vol * 1000.0 if np.isfinite(vol) and vol > 0 else np.nan
    return to_calhabmap(row)


def fcm_to_calhabmap(path, source="seabass_pace_pax", fields: dict[str, str] | None = None) -> pd.DataFrame:
    """fields: SeaBASS abundance field (cells/mL) -> output column (default extra_<field>)."""
    hdr, df = read_sb(path)
    out = _base(hdr, df, path, source, "fcm")
    units = dict(zip(hdr.get("fields", "").split(","), hdr.get("units", "").split(",")))
    fields = fields or {c: f"extra_{c}" for c in df.columns
                        if re.search(r"(syn|pro|euk|pico|nano|cells|abund)", c, re.I)
                        and "ml" in units.get(c, "").lower()}
    for f, c in fields.items():
        out[c] = pd.to_numeric(df[f], errors="coerce") * 1000.0
    return to_calhabmap(out)


def read_seabass_dir(root, kinds=("hplc", "ifcb", "fcm"), source="seabass_pace_pax") -> pd.DataFrame:
    """Walk a downloaded SeaBASS tree; file kind is inferred from the path/name."""
    frames = []
    for p in sorted(Path(root).rglob("*.sb")):
        name = str(p).lower()
        try:
            if "hplc" in name and "hplc" in kinds:
                frames.append(hplc_to_calhabmap(p, source))
            elif "ifcb" in name and "ifcb" in kinds:
                frames.append(ifcb_sb_to_calhabmap(p, source))
            elif ("fcm" in name or "flow_cyt" in name) and "fcm" in kinds:
                frames.append(fcm_to_calhabmap(p, source))
        except Exception as e:      # keep the walk going; report bad files
            print(f"[seabass] skip {p}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
