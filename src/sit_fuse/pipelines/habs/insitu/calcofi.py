"""
CalCOFI / CCE-LTER phytoplankton holdings on the CalCOFI ERDDAP
(https://erddap.calcofi.io, catalog https://calcofi.io/datasets/).

calcofi_phytoplankton (Venrick, 1996-2022, cells/L, 385 taxa)
  Samples are pooled across stations into four regions (NE, SE, Alley, Offshore)
  and carry only a cruise key (no per-sample time), so rows are written with
  spatial_support="region_pooled", time = cruise month midpoint and a large
  sigma_dex. Use for class-prior / distribution-shift checks and offshore
  climatology, not for daily pixel matchups. Pseudo-nitzschia is not resolved
  (Nitzschia spp. only) -> extra_Nitzschia_spp; PN columns left empty.
cce-lter_picoplankton-bacteria (FCM; number/mL, per bottle with time)
  -> extra_Prochlorococcus, extra_Synechococcus, extra_Picoeukaryotes (cells/L).
CalCOFI IFCB (calcofi-cruises-ctd / -underway) is handled by ifcb_dashboard.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
import requests

from sit_fuse.pipelines.habs.insitu.calhabmap_format import DEFAULT_SIGMA_DEX, to_calhabmap

ERDDAP = "https://erddap.calcofi.io/erddap/tabledap"

VENRICK_MAP = {   # genus regex -> CalHABMAP column
    r"^Akashiwo": "Akashiwo_sanguinea", r"^Alexandrium": "Alexandrium_spp",
    r"^Dinophysis": "Dinophysis_spp", r"^Lingulodinium|^Gonyaulax polyedra": "Lingulodinium_polyedra",
    r"^Prorocentrum": "Prorocentrum_spp", r"^Ceratium|^Tripos": "Ceratium_spp",
    r"^Cochlodinium|^Margalefidinium": "Cochlodinium_spp", r"^Gymnodinium": "Gymnodinium_spp",
}
DINO_GENERA = r"^(?:Amphidinium|Gyrodinium|Gonyaulax|Heterocapsa|Karenia|Oxytoxum|Protoperidinium|Peridinium|Scrippsiella|Pyrophacus|Podolampas|Ornithocercus|Katodinium|Torodinium|Polykrikos|Noctiluca|Warnowia|Phalacroma|Amphisolenia|Histioneis|Goniodoma|Dinoflagellat)"
DIATOM_GENERA = r"^(?:Asterionellopsis|Asteromphalus|Bacteriastrum|Cerataulina|Chaetoceros|Corethron|Coscinodiscus|Cylindrotheca|Dactyliosolen|Detonula|Ditylum|Eucampia|Fragilariopsis|Guinardia|Hemiaulus|Lauderia|Leptocylindrus|Lithodesmium|Navicula|Nitzschia|Planktoniella|Pleurosigma|Proboscia|Pseudo-?nitzschia|Rhizosolenia|Skeletonema|Stephanopyxis|Thalassionema|Thalassiosira|Thalassiothrix|Pennate|Centric|Diatom)"


def _get(dataset: str, query: str, timeout=180) -> pd.DataFrame:
    r = requests.get(f"{ERDDAP}/{dataset}.csv?{query}", timeout=timeout)
    if r.status_code == 404 and "nRows = 0" in r.text:
        return pd.DataFrame()
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), skiprows=[1])


def _cruise_time(cruise_key: str) -> pd.Timestamp:
    m = re.match(r"(\d{4})-(\d{2})", str(cruise_key))
    return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-15", tz="UTC") if m else pd.NaT


def pull_venrick(start="1996-01-01", end=None) -> pd.DataFrame:
    obs = _get("calcofi_phytoplankton",
               "sample_key,cruise_key,latitude,longitude,depth,scientific_name,measurement_value"
               "&measurement_type=%22phytoplankton_abundance%22")
    if obs.empty:
        return obs
    obs["sample_time_utc"] = obs["cruise_key"].map(_cruise_time)
    obs = obs[obs["sample_time_utc"].notna()]
    obs = obs[obs["sample_time_utc"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        obs = obs[obs["sample_time_utc"] < pd.Timestamp(end, tz="UTC")]
    name = obs["scientific_name"].fillna("")
    col = pd.Series(np.nan, index=obs.index, dtype=object)
    for rx, c in VENRICK_MAP.items():
        col[name.str.contains(rx, regex=True) & col.isna()] = c
    col[name.str.contains(r"^Nitzschia|^Pseudo-?nitzschia", regex=True) & col.isna()] = "extra_Nitzschia_spp"
    col[name.str.contains(DIATOM_GENERA, regex=True, case=False) & col.isna()] = "Other_Diatoms"
    col[name.str.contains(DINO_GENERA, regex=True, case=False) & col.isna()] = "Other_Dinoflagellates"
    obs["col"] = col.fillna("extra_Other_Phytoplankton")
    keys = ["sample_key", "cruise_key", "sample_time_utc"]
    wide = obs.pivot_table(index=keys, columns="col", values="measurement_value", aggfunc="sum").reset_index()
    tot = obs.groupby(keys)["measurement_value"].sum().rename("Total_Phytoplankton").reset_index()
    pos = obs.groupby(keys)[["latitude", "longitude", "depth"]].first().reset_index()
    wide = wide.merge(tot, on=keys).merge(pos, on=keys)
    # Nitzschia spp. also counted in Other_Diatoms, consistent with CalHABMAP lumping
    if "extra_Nitzschia_spp" in wide:
        wide["Other_Diatoms"] = wide.get("Other_Diatoms", 0).fillna(0) + wide["extra_Nitzschia_spp"].fillna(0)
    wide["Location_Code"] = "CALCOFI_" + wide["sample_key"].str.split(":").str[-1]
    wide["source"] = "calcofi_venrick"
    wide["source_sample_id"] = wide["sample_key"]
    wide["method"] = "lm_settled"
    wide["spatial_support"] = "region_pooled"
    wide["n_pooled"] = np.nan
    wide["Total_Phytoplankton_sigma_dex"] = DEFAULT_SIGMA_DEX["region_pooled"]
    return to_calhabmap(wide)


def pull_picoplankton(start="2000-01-01", end=None, max_depth_m=5.0) -> pd.DataFrame:
    q = (f"sample_key,site_key,time,latitude,longitude,depth,measurement_type,measurement_value"
         f"&depth%3C={max_depth_m}&time%3E={pd.Timestamp(start):%Y-%m-%d}")
    if end:
        q += f"&time%3C={pd.Timestamp(end):%Y-%m-%d}"
    d = _get("cce-lter_picoplankton-bacteria", q)
    if d.empty:
        return d
    names = {"prochlorococcus": "extra_Prochlorococcus", "synechococcus": "extra_Synechococcus",
             "picoeukaryotes": "extra_Picoeukaryotes", "het_bacteria": "extra_Heterotrophic_bacteria"}
    d["col"] = d["measurement_type"].map(names)
    d = d[d["col"].notna()]
    keys = ["site_key", "time", "latitude", "longitude", "depth"]
    w = d.pivot_table(index=keys, columns="col", values="measurement_value", aggfunc="mean").reset_index()
    for c in names.values():
        if c in w:
            w[c] = w[c] * 1000.0          # number/mL -> cells/L
    w["sample_time_utc"] = pd.to_datetime(w["time"], utc=True)
    w["Location_Code"] = "CALCOFI_" + w["site_key"].astype(str).str.replace(" ", "_")
    w["source"] = "cce_lter_fcm"
    w["source_sample_id"] = w["site_key"].astype(str) + "@" + w["time"].astype(str)
    w["method"] = "fcm"
    w["spatial_support"] = "point"
    w["n_pooled"] = 1
    return to_calhabmap(w.drop(columns=["time"]))
