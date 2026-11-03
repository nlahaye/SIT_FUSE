"""
California IFCB Network (https://ifcb.caloos.org, WHOI ifcbdb dashboard API)
-> CalHABMAP-format cells/L.

Per bin:
  /api/export_metadata/{dataset}?start_date=&end_date=&include_skip=false
                                      bin list, requested in monthly chunks (the
                                      unbounded export streams the full archive
                                      slowly and the server drops the connection) (pid, sample_time, ml_analyzed, lat, lon, depth, skip)
  /{dataset}/{pid}_class_scores.csv   per-ROI classifier scores
  /data/{pid}_features.csv            per-ROI morphology (RepresentativeWidth, MinorAxisLength, EquivDiameter)
Concentration = (ROIs assigned to class) / ml_analyzed * 1000  [cells/L, ROI-based].

Caveats encoded as provenance/sigma, not hidden:
  * ROI != cell for chain formers (Pseudo-nitzschia, Chaetoceros). Optional
    chain correction: cells_per_roi = max(1, MajorAxisLength_um / pn_cell_length_um).
  * PN seriata vs delicatissima group is a width split (default 3 um, CalHABMAP
    light-microscopy convention); IFCB width at ~3.4 px/um is near resolution,
    so these columns get a larger sigma_dex (ifcb_width_split).
  * IFCB under-samples < ~5 um cells, so Total_Phytoplankton is not equivalent
    to settled light-microscopy totals; method="ifcb" lets the reference model
    carry a per-method offset.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sit_fuse.pipelines.habs.insitu.calhabmap_format import DEFAULT_SIGMA_DEX, to_calhabmap

DASHBOARD = "https://ifcb.caloos.org"
USER_AGENT = "SIT-FUSE-HAB-insitu/1.0 (+https://github.com/nasa-jpl/SIT_FUSE)"
log = logging.getLogger(__name__)

# dataset -> (Location_Code shared with the co-located CalHABMAP pier, so the
# leakage-safe split groups pier microscopy and IFCB together)
IFCB_PIER_CODES = {
    "scripps-pier-ifcb-151": "SIO", "scripps-pier-ifcb-158": "SIO", "scripps-pier-ifcb-183": "SIO",
    "newport-beach-pier": "NP", "stearns-wharf": "SW", "santa-cruz-municipal-wharf": "HAB_SCW",
    "bodega-marine-lab": "BML", "trinidad-pier-ifcb-162": "TP", "del-mar-mooring": "DMM",
    "mbari-power-buoy": "MBARI", "san-francisco-pier-17": "SF17",
    "cal-poly-humboldt-hioc": "HIOC",
}
CRUISE_DATASETS = ["calcofi-cruises-ctd", "calcofi-cruises-underway", "plumes-and-blooms-cruises",
                   "cce-lter-process-cruises-ifcb-151", "monterey-bay-s-mode",
                   "san-francisco-bay-cruises", "bloofinz-io", "cph-north-wind-ifcb-162"]

PN_CLASSES = ["Pseudo-nitzschia", "pennate_Pseudo-nitzschia"]

# dashboard class -> CalHABMAP column (Tripos = Ceratium; Margalefidinium = Cochlodinium)
CLASS_MAP = {
    "Akashiwo": "Akashiwo_sanguinea", "Alexandrium_catenella": "Alexandrium_spp",
    "Dinophysis": "Dinophysis_spp", "Dinophysis_tripos": "Dinophysis_spp",
    "Lingulodinium_polyedra": "Lingulodinium_polyedra",
    "Prorocentrum": "Prorocentrum_spp", "Prorocentrum_gracile": "Prorocentrum_spp",
    "Prorocentrum_micans": "Prorocentrum_spp", "Prorocentrum_rostratum": "Prorocentrum_spp",
    "Prorocentrum_scutellum": "Prorocentrum_spp",
    "Tripos": "Ceratium_spp", "Tripos_balechii": "Ceratium_spp", "Tripos_furca": "Ceratium_spp",
    "Tripos_fusus": "Ceratium_spp", "Tripos_lineatus": "Ceratium_spp",
    "Margalefidinium": "Cochlodinium_spp", "Gymnodinium": "Gymnodinium_spp",
}
DIATOMS = {"Asterionellopsis_glacialis", "Asteromphalus", "Bacillariophyceae", "Bacteriastrum",
           "Bacteriastrum_TAG_interaction", "Cerataulina_pelagica", "Chaetoceros",
           "Chaetoceros_TAG_interaction", "Chaetoceros_didymus", "Chaetoceros_socialis",
           "Corethron_hystrix", "Coscinodiscus", "Cylindrotheca", "Cylindrotheca_morphotype1",
           "Dactyliosolen_blavyanus", "Ditylum_brightwellii", "Ephemera", "Eucampia",
           "Guinardia_flaccida", "Guinardia_striata", "Hemiaulus", "Hemiaulus_membranaceus",
           "Lauderia_annulata", "Leptocylindrus", "Leptocylindrus_mediterraneus", "Licmophora",
           "Nanoneis", "Navicula", "Pleurosigma", "Rhizosolenia", "Skeletonema", "Thalassionema",
           "Thalassiosira", "Thalassiosira_sp_aff_mala", "pennate", "pennate_Thalassionema",
           "pennate_morphotype1", "crescent"}
DINOS = {"Amphidinium", "Dinophyceae", "Fragilidinium", "Gonyaulax", "Gyrodinium",
         "Heterocapsa_rotundata", "Karenia", "Katodinium_or_Torodinium", "Oxytoxum", "Polykrikos",
         "Protoperidinium", "Scrippsiella", "Warnowia"}
NON_PHYTO = {"Acantharia", "Amoeba", "Amoeba_TAG_nano", "Bacteria_colony", "Ciliophora", "Euplotes",
             "Euplotes_morphotype1", "Eutintinnus", "Favella", "Laboea_strobila", "Mesodinium",
             "Nauplii", "Pleuronema", "Strobilidium_morphotype1", "Strombidium_capitatum",
             "Strombidium_conicum", "Strombidium_inclinatum", "Strombidium_morphotype1",
             "Strombidium_morphotype2", "Strombidium_tintinnodes", "Tiarina_fusus",
             "Tintinnidium_mucicola", "Tintinnina", "Tintinnopsis", "Tontonia_appendiculariformis",
             "bad", "bead", "bubble", "detritus", "detritus_theca_fragment", "detritus_transparent",
             "fecal_pellet", "fiber", "pollen", "pollen_TAG_nano", "zooplankton",
             "Dinophyceae_TAG_cyst", "Lingulodinium_polyedra_TAG_cyst"}

DEFAULTS = {
    "score_threshold": 0.0,        # >0: ROIs with max score below threshold -> unclassified (excluded)
    "pixels_per_um": 3.4,
    "pn_width_split_um": 3.0,
    "width_feature": "RepresentativeWidth",   # or MinorAxisLength
    "pn_cell_length_um": None,     # e.g. 70 -> chain correction
    "aggregate": "daily",          # bin | daily | window
    "window_utc_hours": [16, 24],  # aggregate only bins in this UTC window (aggregate=window)
    "min_ml_analyzed": 1.0,
    "size_classes_um": [2, 20, 200],   # extra_IFCB_nano (2-20 um ESD), extra_IFCB_micro (20-200 um)
    "keep_class_columns": False,   # write extra_ifcb_<class> for every class
    "cache_dir": None,
    "max_bins_per_day": None,      # e.g. 6 -> evenly subsample bins per site-day (each bin = 2 requests)
    "chunk_months": 1,             # metadata request span
    "request_delay_s": 0.1,        # pause between requests
    "retries": 5,                  # per request, exponential backoff on dropped connections / 429 / 5xx
    "timeout_s": 180,
    "cache_only": False,           # True -> no network; use only bins already in cache_dir
    "require_features": True,      # cache_only: skip bins whose features file was never cached
}


class IFCBDashboard:
    def __init__(self, base: str = DASHBOARD, timeout: int = 180, cache_dir: str | None = None,
                 retries: int = 5, delay: float = 0.1, chunk_months: int = 1, cache_only: bool = False):
        self.base, self.timeout, self.delay, self.chunk = base.rstrip("/"), timeout, delay, chunk_months
        self.cache = Path(cache_dir) if cache_dir else None
        self.cache_only = cache_only
        if cache_only and self.cache is None:
            raise ValueError("cache_only requires cache_dir")
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT})
        retry = Retry(total=retries, connect=retries, read=retries, status=retries, backoff_factor=2.0,
                      status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",),
                      respect_retry_after_header=True, raise_on_status=False)
        self.s.mount("https://", HTTPAdapter(max_retries=retry))
        self.s.mount("http://", HTTPAdapter(max_retries=retry))

    def _get(self, path: str, params: dict | None = None) -> requests.Response | None:
        url = f"{self.base}/{path.lstrip('/')}"
        for attempt in range(2):   # one extra fresh-connection attempt after urllib3 retries
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
                time.sleep(self.delay)
                return r
            except (requests.ConnectionError, requests.Timeout) as e:
                log.warning("IFCB request failed (%s) %s %s", e.__class__.__name__, url, params or "")
                self.s.close()
                time.sleep(10 * (attempt + 1))
        return None

    def _key(self, path: str, params: dict | None = None) -> Path | None:
        if not self.cache:
            return None
        key = path.strip("/").replace("/", "__")
        if params:
            key += "__" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
        return self.cache / key

    def is_cached(self, path: str, params: dict | None = None) -> bool:
        f = self._key(path, params)
        return f is not None and f.exists() and f.stat().st_size > 0

    def _csv(self, path: str, params: dict | None = None) -> pd.DataFrame | None:
        f = self._key(path, params)
        if f is not None and f.exists():
            if not f.stat().st_size:
                return None
            try:
                return pd.read_csv(f)
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
                log.warning("IFCB cache file unreadable (partial write?): %s", f.name)
                if self.cache_only:
                    return None
        if self.cache_only:
            return None
        r = self._get(path, params)
        if r is None:
            return None                     # transient failure: not cached, retried next run
        if not r.ok or not r.text.strip():
            if f is not None and r.status_code == 404:
                f.write_text("")            # empty query (e.g. no bins that month)
            return None
        if f is not None:
            f.write_text(r.text)
        return pd.read_csv(io.StringIO(r.text))

    def _cached_month_chunks(self, dataset: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        pre = f"api__export_metadata__{dataset}__"
        out = []
        for f in self.cache.glob(pre + "*"):
            try:
                a = pd.Timestamp(f.name.split("start_date-")[1][:10])
                b = pd.Timestamp(f.name.split("end_date-")[1][:10])
                out.append((a, b))
            except (IndexError, ValueError):
                continue
        return sorted(out)

    def datasets(self) -> list[str]:
        r = self._get("api/list_datasets")
        return r.json()["datasets"] if r is not None and r.ok else []

    def bins(self, dataset: str, start=None, end=None) -> pd.DataFrame:
        if start is None:
            start = "2016-01-01"            # California IFCB Network archive start
        t0 = pd.Timestamp(start).tz_localize(None).normalize()
        t1 = (pd.Timestamp(end).tz_localize(None) if end is not None else pd.Timestamp.now("UTC").tz_localize(None))
        parts = []
        if self.cache_only:
            for a, b in self._cached_month_chunks(dataset):
                if b < t0 or a >= t1:
                    continue
                m = self._csv(f"api/export_metadata/{dataset}",
                              {"start_date": a.strftime("%Y-%m-%d"), "end_date": b.strftime("%Y-%m-%d"),
                               "include_skip": "false"})
                if m is not None and not m.empty:
                    parts.append(m)
            a = t1
        else:
            a = t0
        while a < t1:
            b = min(a + pd.DateOffset(months=self.chunk), t1)
            # server end_date is inclusive (+1 day); trim to [start, end) below
            m = self._csv(f"api/export_metadata/{dataset}",
                          {"start_date": a.strftime("%Y-%m-%d"),
                           "end_date": (b - pd.Timedelta("1D")).strftime("%Y-%m-%d"),
                           "include_skip": "false"})
            if m is not None and not m.empty:
                parts.append(m)
            a = b
        if not parts:
            return pd.DataFrame()
        m = pd.concat(parts, ignore_index=True).drop_duplicates("pid")
        m["sample_time"] = pd.to_datetime(m["sample_time"], utc=True, errors="coerce")
        m = m[(m["sample_time"] >= t0.tz_localize("UTC")) & (m["sample_time"] < t1.tz_localize("UTC"))]
        if "skip" in m:
            m = m[pd.to_numeric(m["skip"], errors="coerce").fillna(0) == 0]
        return m.sort_values("sample_time").reset_index(drop=True)

    def class_scores(self, dataset: str, pid: str):
        return self._csv(f"{dataset}/{pid}_class_scores.csv")

    def features(self, pid: str):
        return self._csv(f"data/{pid}_features.csv")


def _subsample_per_day(meta: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep k bins per day, evenly spaced in time (cruise datasets are left intact by caller)."""
    day = meta["sample_time"].dt.floor("D")
    keep = []
    for _, g in meta.groupby(day):
        idx = np.unique(np.linspace(0, len(g) - 1, min(k, len(g))).round().astype(int))
        keep.append(g.iloc[idx])
    return pd.concat(keep) if keep else meta


def bin_counts(scores: pd.DataFrame, feats: pd.DataFrame | None, cfg: dict[str, Any]) -> dict[str, float]:
    """Per-bin ROI counts (cell-equivalents) per CalHABMAP column + extras."""
    cls_cols = [c for c in scores.columns if c != "pid"]
    S = scores[cls_cols].to_numpy(dtype=np.float64)
    lab = np.array(cls_cols, dtype=object)[S.argmax(1)]
    if cfg["score_threshold"] > 0:
        lab = np.where(S.max(1) >= cfg["score_threshold"], lab, "unclassified")
    roi = scores["pid"].str.rsplit("_", n=1).str[-1].astype(int).to_numpy()
    width = length = esd = np.full(len(roi), np.nan)
    if feats is not None and len(feats):
        f = feats.set_index("roi_number")
        ppu = cfg["pixels_per_um"]
        width = f.reindex(roi)[cfg["width_feature"]].to_numpy(np.float64) / ppu
        length = f.reindex(roi)["MajorAxisLength"].to_numpy(np.float64) / ppu
        esd = f.reindex(roi)["EquivDiameter"].to_numpy(np.float64) / ppu
    cells = np.ones(len(roi))
    is_pn = np.isin(lab, PN_CLASSES)
    if cfg.get("pn_cell_length_um"):
        cells[is_pn] = np.maximum(1.0, np.nan_to_num(length[is_pn] / cfg["pn_cell_length_um"], nan=1.0))
    out: dict[str, float] = {}
    for col in set(CLASS_MAP.values()) | {"Other_Diatoms", "Other_Dinoflagellates",
                                           "Pseudo_nitzschia_seriata_group",
                                           "Pseudo_nitzschia_delicatissima_group", "Total_Phytoplankton"}:
        out[col] = 0.0
    split = cfg["pn_width_split_um"]
    phyto = ~np.isin(lab, list(NON_PHYTO) + ["unclassified"])
    for i in range(len(lab)):
        c, n = lab[i], cells[i]
        if not phyto[i]:
            continue
        out["Total_Phytoplankton"] += n
        if c in PN_CLASSES:
            # width unknown -> seriata (conservative for alerts); counted under both in pn_total
            key = ("Pseudo_nitzschia_delicatissima_group" if np.isfinite(width[i]) and width[i] < split
                   else "Pseudo_nitzschia_seriata_group")
            out[key] += n
        elif c in CLASS_MAP:
            out[CLASS_MAP[c]] += n
        elif c in DIATOMS:
            out["Other_Diatoms"] += n
        elif c in DINOS:
            out["Other_Dinoflagellates"] += n
    out["extra_Pseudo_nitzschia_spp"] = out["Pseudo_nitzschia_seriata_group"] + out[
        "Pseudo_nitzschia_delicatissima_group"]
    lo, mid, hi = cfg["size_classes_um"]
    e = np.where(phyto, esd, np.nan)
    out["extra_IFCB_nano"] = float(np.sum((e >= lo) & (e < mid)))
    out["extra_IFCB_micro"] = float(np.sum((e >= mid) & (e < hi)))
    if cfg.get("keep_class_columns"):
        u, k = np.unique(lab, return_counts=True)
        for a, b in zip(u, k):
            out[f"extra_ifcb_{a}"] = float(b)
    return out


def pull_ifcb(datasets: list[str], start, end, cfg: dict[str, Any] | None = None,
              location_codes: dict[str, str] | None = None, max_bins: int | None = None) -> pd.DataFrame:
    cfg = {**DEFAULTS, **(cfg or {})}
    codes = {**IFCB_PIER_CODES, **(location_codes or {})}
    api = IFCBDashboard(timeout=cfg["timeout_s"], cache_dir=cfg.get("cache_dir"), retries=cfg["retries"],
                        delay=cfg["request_delay_s"], chunk_months=cfg["chunk_months"],
                        cache_only=bool(cfg.get("cache_only")))
    rows = []
    for ds in datasets:
        meta = api.bins(ds, start, end)
        if meta.empty:
            log.warning("IFCB %s: no bins in %s - %s", ds, start, end)
            continue
        meta = meta[pd.to_numeric(meta["ml_analyzed"], errors="coerce") >= cfg["min_ml_analyzed"]]
        if cfg["aggregate"] == "window":
            h0, h1 = cfg["window_utc_hours"]
            hr = meta["sample_time"].dt.hour
            meta = meta[(hr >= h0) & (hr < h1)]
        if api.cache_only:
            # restrict to bins already downloaded *before* subsampling, so a cap that differs
            # from the original run cannot select bins that were never fetched
            pids = meta["pid"].astype(str).tolist()
            ok = np.array([api.is_cached(f"{ds}/{p}_class_scores.csv") for p in pids], dtype=bool)
            if cfg.get("require_features", True):
                ok &= np.array([api.is_cached(f"data/{p}_features.csv") for p in pids], dtype=bool)
            log.info("IFCB %s (cache only): %d of %d candidate bins cached", ds, int(ok.sum()), len(ok))
            meta = meta[ok]
        if cfg.get("max_bins_per_day") and ds not in CRUISE_DATASETS:
            meta = _subsample_per_day(meta, int(cfg["max_bins_per_day"]))
        if max_bins:
            meta = meta.head(max_bins)
        log.info("IFCB %s: %d bins", ds, len(meta))
        for _, b in meta.iterrows():
            sc = api.class_scores(ds, b["pid"])
            if sc is None or sc.empty:
                continue
            r = bin_counts(sc, api.features(b["pid"]), cfg)
            r.update(dataset=ds, pid=b["pid"], sample_time_utc=b["sample_time"],
                     ml=float(b["ml_analyzed"]), latitude=b["latitude"], longitude=b["longitude"],
                     depth=b.get("depth", np.nan), cruise=b.get("cruise", np.nan))
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    count_cols = [c for c in df.columns if c in CLASS_MAP.values() or c.startswith("extra_")
                  or c in ("Other_Diatoms", "Other_Dinoflagellates", "Total_Phytoplankton",
                           "Pseudo_nitzschia_seriata_group", "Pseudo_nitzschia_delicatissima_group")]
    df[count_cols] = df[count_cols].fillna(0.0)
    cruise = df["dataset"].isin(CRUISE_DATASETS)
    if cfg["aggregate"] in ("daily", "window"):
        # pool counts and volume over the day at fixed sites (Poisson-consistent); cruises stay per-bin
        fixed = df[~cruise].copy()
        fixed["_day"] = fixed["sample_time_utc"].dt.floor("D")
        agg = {c: "sum" for c in count_cols + ["ml"]}
        agg.update(latitude="median", longitude="median", depth="median", pid="first",
                   sample_time_utc="median")
        g = fixed.groupby(["dataset", "_day"]).agg(agg)
        g["n_pooled"] = fixed.groupby(["dataset", "_day"]).size()
        df = pd.concat([g.reset_index().drop(columns="_day"), df[cruise].assign(n_pooled=1)],
                       ignore_index=True)
    else:
        df["n_pooled"] = 1
    df[count_cols] = df[count_cols].div(df["ml"], axis=0) * 1000.0      # -> cells/L
    df["Location_Code"] = [codes.get(d, f"IFCB_{d}") if d not in CRUISE_DATASETS
                           else f"IFCB_{d}_{round(la, 2)}_{round(lo, 2)}"
                           for d, la, lo in zip(df["dataset"], df["latitude"], df["longitude"])]
    df["source"] = "ifcb_dashboard"
    df["source_sample_id"] = df["dataset"] + ":" + df["pid"].astype(str)
    df["method"] = "ifcb"
    df["spatial_support"] = np.where(df["dataset"].str.contains("underway"), "underway", "point")
    # counting error ~ 1/sqrt(N) in natural log -> dex, floored at method error
    for col, base in (("Total_Phytoplankton", DEFAULT_SIGMA_DEX["ifcb"]),
                      ("Pseudo_nitzschia_seriata_group", DEFAULT_SIGMA_DEX["ifcb_width_split"]),
                      ("Pseudo_nitzschia_delicatissima_group", DEFAULT_SIGMA_DEX["ifcb_width_split"])):
        n = df[col] * df["ml"] / 1000.0
        df[f"{col}_sigma_dex"] = np.sqrt(base ** 2 + (0.434 / np.sqrt(np.maximum(n, 1.0))) ** 2)
    df = df.drop(columns=["ml"])
    return to_calhabmap(df, extra_cols=["n_pooled"])
