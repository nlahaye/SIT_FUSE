"""
CalHABMAP pier/buoy time series from the SCCOOS ERDDAP (tabledap).

The ERDDAP variables are exactly the first 40 columns of combined_full.csv
(units row: cells/L, ng/mL, mg/m3, uM), so reformatting is a column copy plus
time/SampleID conversion. Full timestamps are kept in sample_time_utc.
https://erddap.sccoos.org/erddap/tabledap/index.html
"""

from __future__ import annotations

import io
import time as _time

import pandas as pd
import requests

from sit_fuse.pipelines.habs.insitu.calhabmap_format import CALHABMAP_COLUMNS, to_calhabmap

ERDDAP = "https://erddap.sccoos.org/erddap/tabledap"

# dataset id -> Location_Code (codes in the attached combined_full.csv where they exist)
CALHABMAP_DATASETS = {
    "HABs-ScrippsPier": "SIO", "HABs-SantaMonicaPier": "SMP", "HABs-NewportBeachPier": "NP",
    "HABs-StearnsWharf": "SW", "HABs-CalPolyPier": "CPP", "HABs-SantaCruzWharf": "HAB_SCW",
    "HABs-MorroBayBackBay": "MBB", "HABs-MorroBayFrontBay": "MBF", "HABs-MontereyWharf": "MW",
    "HABs-BodegaMarineLab": "BML", "HABs-BodegaMarineLabBuoy": "BMLB", "HABs-Humboldt": "HUM",
    "HABs-HumboldtSouthBay": "HSB", "HABs-InnerTomalesBay": "ITB",
    "HABs-TomalesBayMid-ChannelBuoy": "TBMB", "HABs-TomalesBayMouth": "TBM",
    "HABs-TrinidadPier": "TP",
}
SOUTHERN_CA = ["HABs-ScrippsPier", "HABs-NewportBeachPier", "HABs-SantaMonicaPier",
               "HABs-StearnsWharf", "HABs-CalPolyPier", "HABs-MorroBayBackBay",
               "HABs-MorroBayFrontBay"]


def fetch_dataset(dataset_id: str, start: str, end: str, base: str = ERDDAP,
                  timeout: int = 120, retries: int = 3) -> pd.DataFrame:
    url = (f"{base}/{dataset_id}.csv?"
           f"&time%3E={pd.Timestamp(start):%Y-%m-%dT00:00:00Z}"
           f"&time%3C={pd.Timestamp(end):%Y-%m-%dT00:00:00Z}")
    for k in range(retries):
        r = requests.get(url, timeout=timeout)
        if r.status_code == 404 and "nRows = 0" in r.text:
            return pd.DataFrame()
        if r.ok:
            return pd.read_csv(io.StringIO(r.text), skiprows=[1])     # row 1 = units
        _time.sleep(2 ** k)
    r.raise_for_status()
    return pd.DataFrame()


def pull_calhabmap(datasets=None, start="2015-01-01", end=None, sigma_dex=None) -> pd.DataFrame:
    """Pull and reformat CalHABMAP datasets (default: southern/central CA piers)."""
    end = end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    frames = []
    for ds in datasets or SOUTHERN_CA:
        df = fetch_dataset(ds, start, end)
        if df.empty:
            continue
        code = CALHABMAP_DATASETS.get(ds, ds)
        df["Location_Code"] = df["Location_Code"].fillna(code) if "Location_Code" in df else code
        df["sample_time_utc"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df["source"] = "calhabmap_erddap"
        df["source_sample_id"] = ds + ":" + df["SampleID"].astype(str)
        df["method"] = "microscopy"
        df["spatial_support"] = "point"
        df["n_pooled"] = 1
        if sigma_dex:
            for k, v in sigma_dex.items():
                df[f"{k}_sigma_dex"] = v
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=CALHABMAP_COLUMNS)
    return to_calhabmap(pd.concat(frames, ignore_index=True))
