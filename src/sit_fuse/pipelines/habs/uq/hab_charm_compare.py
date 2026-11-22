"""
SIT-FUSE HAB UQ vs C-HARM: PN combination, in-situ scores, map agreement, chlorophyll
consistency and a merged summary. Stages (each a CLI action, parallel by target / year):

  combine      rho of the size-class Gaussian copula (Kendall tau-b by season, training split)
               and calibration-split scores of every combination method vs. the direct total-PN
               model -> charm_compare/pn_combination.json (selected method)
  points       in-situ evaluation on the test split, one target (pn | pda | cda), all leads:
               C-HARM read at the sample (mean of valid 3 km pixels within radius_km, nearest
               valid pixel within fallback_km otherwise; plus the coastal-edge variants
               charm_p_nearest and charm_p_offcoast = radius mean without pixels closer than
               coast_exclude_km to land / the domain edge, and stratum coast_bin by the
               coast distance of the nearest pixel), SIT-FUSE streams = fused, per
               instrument type (viirs, olci, modis, oci), per sensor, and for pn the direct
               total model plus the size-class combinations. Brier, BSS (vs. training base rate),
               log loss, ROC / PR AUC, ECE, reliability, POD / FAR / CSI at 0.5 and at the
               calibration-tuned cutoff; paired site-week block bootstrap of the differences;
               strata: year, season, C-HARM version, source, chlorophyll bin, match mode.
  maps         one target x year: SIT-FUSE streams averaged onto the C-HARM grid (block mean of
               probabilities), daily rank correlation, bias, MAE, kappa / IoU at 0.5, at the
               calibration-tuned point cutoffs of each stream (points must run first; skipped
               otherwise) and equal-area (ours thresholded at the quantile that flags the same
               area as C-HARM at its tuned cutoff, scale-free), coverage,
               alongshore (Hovmoller) means in a coastal strip; daily stacks for animation.
  consistency  one year, native product grid: chlorophyll biomass ceiling (PN cells <= chl /
               q_chl_min), total-vs-size-class bounds, P(pDA >= 0.5) <= P(PN >= 0.5e6 / cDA_max);
               violation rates per day and flag rasters (hatched in the C-HARM animation).
  summary      merge of all outputs (tables only; no figures of results).

  python -m sit_fuse.pipelines.habs.uq.hab_charm_compare combine -y <yaml>
  python -m sit_fuse.pipelines.habs.uq.hab_charm_compare points  -y <yaml> --target pn
  python -m sit_fuse.pipelines.habs.uq.hab_charm_compare maps    -y <yaml> --target pn --year 2024
  python -m sit_fuse.pipelines.habs.uq.hab_charm_compare consistency -y <yaml> --year 2024
  python -m sit_fuse.pipelines.habs.uq.hab_charm_compare summary -y <yaml>

Config: uq.charm (see hab_charm) plus
  leads: [0, 1, 2, 3]          map_leads: [0]       coast_exclude_km: 6
  n_bootstrap: 500             copula_samples: 256
  coastal_km: 30               hovmoller_dlat: 0.1
  chl_dirs: {snpp: /path, ...}  (L3m CHL / PACE BGC; sync_l3m_rrs --suite CHL)
  consistency: {q_chl_min_pg: 2.0, tolerance: 1.0, cda_max_pg: 100.0, eps: 0.05}
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sit_fuse.pipelines.habs.uq import hab_charm as hc
from sit_fuse.pipelines.habs.uq import hab_pn_combine as pc
from sit_fuse.pipelines.habs.uq.hab_uq_common import (
    EPS, NODATA, PN_GROUPS, SENSORS, exceedance_probability, species_spec)

LOG = logging.getLogger(__name__)

TARGETS = {
    "pn": {"species": "Pseudo_nitzschia_total", "threshold": 1e4, "groups": list(PN_GROUPS)},
    "pda": {"species": "pDA", "threshold": 0.5},
    "cda": {"species": "cDA", "threshold": 10.0},
}
FAMILIES = ("viirs", "olci", "modis", "oci")
CHL_BINS = [0, 1, 3, 10, np.inf]
CHL_LABELS = ["<1", "1-3", "3-10", ">=10"]
COAST_BINS = [0, 6, 12, 24, np.inf]
COAST_LABELS = ["<6km", "6-12km", "12-24km", ">=24km"]
CHARM_VARIANTS = ("charm_p", "charm_p_nearest", "charm_p_offcoast")


def _thr_name(t: float) -> str:
    return f"{t:g}".replace(".", "p").replace("+", "")


def cfg_of(yml):
    c = hc.charm_cfg(yml)
    c.setdefault("leads", [0, 1, 2, 3])
    c.setdefault("map_leads", [0])
    c.setdefault("n_bootstrap", 500)
    c.setdefault("copula_samples", 256)
    c.setdefault("coastal_km", 30.0)
    c.setdefault("hovmoller_dlat", 0.1)
    c.setdefault("chl_dirs", {})
    c["consistency"] = {"q_chl_min_pg": 2.0, "tolerance": 1.0, "cda_max_pg": 100.0, "eps": 0.05,
                        **(c.get("consistency") or {})}
    return c


def out_root_of(yml) -> Path:
    uq = yml["uq"]
    return Path(uq.get("out_dir", os.path.join(yml.get("final_product_dir", "."), "uq")))


def cmp_dir(yml) -> Path:
    return Path(cfg_of(yml).get("out_dir") or out_root_of(yml) / "charm_compare")


def spec_of(yml, key):
    return species_spec(key, yml["uq"].get("species", {}).get(key))


# ---------------------------------------------------------------------------
# Bundles and sample-level streams
# ---------------------------------------------------------------------------

def load_bundle(yml, key):
    import joblib
    root = out_root_of(yml)
    spec = spec_of(yml, key)
    bp = root / "models" / f"hab_uq_{spec.fname_token}.joblib"
    if not bp.exists():
        LOG.warning("%s: no model bundle %s", key, bp)
        return None, None
    b = joblib.load(bp)
    b["path"] = str(bp)
    frame = b.pop("_frame", None)
    mpq = root / "models" / f"matchups_{spec.fname_token}.parquet"
    mcsv = root / "models" / f"matchups_{spec.fname_token}.csv.gz"
    if frame is None:
        frame = pd.read_parquet(mpq) if mpq.exists() else (pd.read_csv(mcsv) if mcsv.exists() else None)
    return b, frame


def sample_streams(bundle, frame, method="log", per_sensor=True) -> dict[str, dict[str, Any]]:
    """{stream: {split: (P (n, C), meta)}} for fused, per instrument type and per sensor."""
    from sit_fuse.pipelines.habs.uq.hab_multisensor_fusion import fuse_samples
    from sit_fuse.pipelines.habs.uq.hab_uq_evaluate import predict_samples
    out: dict[str, dict[str, Any]] = {}
    Pc, mc, _ = predict_samples(bundle, frame, "calibration")
    Pt, mt, _ = predict_samples(bundle, frame, "test")
    if Pc is None or Pt is None:
        return out

    def fused(sensors):
        b2 = {**bundle, "sensors": sensors}
        kc, kt = mc["sensor"].isin(sensors).to_numpy(), mt["sensor"].isin(sensors).to_numpy()
        if not kc.any() or not kt.any():
            return None
        r = {}
        for split, (Pa, ma) in (("test", (Pt[kt], mt[kt])), ("calibration", (Pc[kc], mc[kc]))):
            fz = fuse_samples(b2, Pc[kc], mc[kc].reset_index(drop=True), Pa, ma.reset_index(drop=True), method=method)
            r[split] = (fz["P_test"], fz["meta_test"])
        return r

    f = fused(list(bundle["sensors"]))
    if f:
        out["fused"] = f
    for fam in FAMILIES:
        ss = [s for s in bundle["sensors"] if SENSORS[s].family == fam]
        if ss:
            f = fused(ss)
            if f:
                out[f"type_{fam}"] = f
    if per_sensor:
        for s in bundle["sensors"]:
            kc, kt = (mc["sensor"] == s).to_numpy(), (mt["sensor"] == s).to_numpy()
            if kt.any():
                out[f"sensor_{s}"] = {"test": (Pt[kt], mt[kt].reset_index(drop=True)),
                                      "calibration": (Pc[kc], mc[kc].reset_index(drop=True))}
    return out


def _exceed_table(streams, spec, thr, prefix="") -> dict[str, dict[str, pd.Series]]:
    """{split: {stream: Series(p, index=sample_uid)}}."""
    res: dict[str, dict[str, pd.Series]] = {"test": {}, "calibration": {}}
    for name, d in streams.items():
        for split, (P, m) in d.items():
            p = exceedance_probability(P, spec.edges, thr, spec.detection_floor)
            res[split][prefix + name] = pd.Series(p, index=m["sample_uid"].astype(str).to_numpy()).groupby(level=0).mean()
    return res


# ---------------------------------------------------------------------------
# Binary scores
# ---------------------------------------------------------------------------

def binary_scores(p, y, base_rate=None, cutoff=None) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, int)
    n, k = len(y), int(y.sum())
    out = {"n": n, "n_events": k, "event_rate": float(y.mean()) if n else np.nan}
    if n == 0:
        return out
    bs = float(((p - y) ** 2).mean())
    br = float(y.mean()) if base_rate is None else float(base_rate)
    ref = float(((br - y) ** 2).mean())
    out.update(brier=bs, bss=float(1 - bs / ref) if ref > 0 else np.nan,
               log_loss=float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
               mean_p=float(p.mean()))
    if 0 < k < n:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    bins = np.clip((p * 10).astype(int), 0, 9)
    out["ece"] = float(sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
                           for b in range(10) if (bins == b).any()))
    for tag, c in (("0p5", 0.5), ("tuned", cutoff)):
        if c is None:
            continue
        yh = p >= c
        tp, fp, fn = int((yh & (y == 1)).sum()), int((yh & (y == 0)).sum()), int((~yh & (y == 1)).sum())
        out[f"pod_{tag}"] = tp / (tp + fn) if tp + fn else np.nan
        out[f"far_{tag}"] = fp / (tp + fp) if tp + fp else np.nan
        out[f"csi_{tag}"] = tp / (tp + fp + fn) if tp + fp + fn else np.nan
        if tag == "tuned":
            out["cutoff_tuned"] = float(c)
    return out


def tune_cutoff(p, y, grid=np.linspace(0.02, 0.98, 49)) -> float | None:
    p, y = np.asarray(p, float), np.asarray(y, int)
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]
    if y.sum() == 0 or len(y) < 10:
        return None
    csi = []
    for c in grid:
        yh = p >= c
        tp, fp, fn = (yh & (y == 1)).sum(), (yh & (y == 0)).sum(), (~yh & (y == 1)).sum()
        csi.append(tp / max(tp + fp + fn, 1))
    return float(grid[int(np.argmax(csi))])


def reliability(p, y, n_bins=10) -> pd.DataFrame:
    p, y = np.asarray(p, float), np.asarray(y, int)
    b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rows = [{"bin": k, "p_lo": k / n_bins, "p_hi": (k + 1) / n_bins, "n": int((b == k).sum()),
             "mean_p": float(p[b == k].mean()), "obs_rate": float(y[b == k].mean())}
            for k in range(n_bins) if (b == k).any()]
    return pd.DataFrame(rows)


def paired_bootstrap(df, a, b, n_boot=500, seed=0) -> dict[str, float]:
    """Brier and ROC-AUC differences (a - b), site-week blocks resampled."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    blocks = (df["Location_Code"].astype(str) + "|" + pd.to_datetime(df["date"]).dt.strftime("%G-%V")).to_numpy()
    codes, uniq = pd.factorize(blocks)
    idx = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    y, pa, pb = df["y"].to_numpy(int), df[a].to_numpy(float), df[b].to_numpy(float)

    def stats(ii):
        r = {"d_brier": float(((pa[ii] - y[ii]) ** 2).mean() - ((pb[ii] - y[ii]) ** 2).mean())}
        if 0 < y[ii].sum() < len(ii):
            r["d_roc_auc"] = float(roc_auc_score(y[ii], pa[ii]) - roc_auc_score(y[ii], pb[ii]))
        return r

    pt = stats(np.arange(len(y)))
    draws = {k: [] for k in pt}
    for _ in range(int(n_boot)):
        ii = np.concatenate([idx[k] for k in rng.integers(0, len(idx), len(idx))])
        for k, v in stats(ii).items():
            draws[k].append(v)
    out = {"n": len(y), "n_blocks": len(uniq)}
    for k, v in pt.items():
        d = np.asarray(draws[k])
        out[k] = v
        if len(d):
            out[f"{k}_lo95"], out[f"{k}_hi95"] = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            out[f"{k}_p_le0"] = float((d <= 0).mean())
    return out


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------

def _insitu(yml):
    from sit_fuse.pipelines.habs.uq.hab_insitu import insitu_filter_config, load_insitu
    from sit_fuse.pipelines.habs.uq.hab_uq_common import insitu_date_kwargs
    uq = yml["uq"]
    return load_insitu(uq.get("insitu_file", yml.get("insitu_file")), max_depth_m=uq.get("max_depth_m", 1.0),
                       filters=insitu_filter_config(yml), **insitu_date_kwargs(uq))


def _splits(yml, key, ins):
    from sit_fuse.pipelines.habs.uq.hab_insitu import assign_splits, species_frame
    from sit_fuse.pipelines.habs.uq.hab_uq_train import split_config
    return assign_splits(species_frame(ins, spec_of(yml, key)), split_config(yml))


def size_class_streams(yml, method="log"):
    """Per-stream (P_seriata, P_delicatissima, dates) aligned on sample_uid, per split."""
    sd = {}
    for g in PN_GROUPS:
        b, f = load_bundle(yml, g)
        if b is None or f is None:
            return {}, None
        sd[g] = sample_streams(b, f, method, per_sensor=False)
    spec = spec_of(yml, PN_GROUPS[0])
    out = {}
    for name in set(sd[PN_GROUPS[0]]) & set(sd[PN_GROUPS[1]]):
        out[name] = {}
        for split in ("test", "calibration"):
            (PS, mS), (PD, mD) = sd[PN_GROUPS[0]][name][split], sd[PN_GROUPS[1]][name][split]
            iS = pd.Series(np.arange(len(mS)), index=mS["sample_uid"].astype(str)).groupby(level=0).first()
            iD = pd.Series(np.arange(len(mD)), index=mD["sample_uid"].astype(str)).groupby(level=0).first()
            uid = iS.index.intersection(iD.index)
            out[name][split] = (PS[iS[uid].to_numpy()], PD[iD[uid].to_numpy()],
                                pd.to_datetime(mS["date"].to_numpy()[iS[uid].to_numpy()]), uid.to_numpy())
    return out, spec


def combined_exceedance(streams, spec, thr, rho_table, methods=pc.METHODS, n=256):
    """{split: {f"combined_{method}_{stream}": Series}}."""
    res = {"test": {}, "calibration": {}}
    for name, d in streams.items():
        for split, (PS, PD, dates, uid) in d.items():
            if not len(uid):
                continue
            rho = pc.rho_for(dates, rho_table)
            for m in methods:
                p = pc.combine(PS, PD, spec.edges, spec.detection_floor, thr, m, rho, n)
                res[split][f"combined_{m}_{name}"] = pd.Series(p, index=uid)
    return res


def run_combine(yml) -> dict[str, Any]:
    c = cfg_of(yml)
    out = cmp_dir(yml)
    out.mkdir(parents=True, exist_ok=True)
    ins = _insitu(yml)
    tr = _splits(yml, "Pseudo_nitzschia_total", ins)
    tr = tr[tr["split"] == "train"]
    rho = pc.fit_rho(tr, PN_GROUPS[0], PN_GROUPS[1], by="season")
    thr = TARGETS["pn"]["threshold"]
    method = (yml["uq"].get("fusion") or {}).get("pool", "log")
    streams, spec = size_class_streams(yml, method)
    total_spec = spec_of(yml, "Pseudo_nitzschia_total")
    totvals = pd.Series(pd.to_numeric(_splits(yml, "Pseudo_nitzschia_total", ins).set_index("sample_uid")["value"]))
    rows, coverage = [], {}
    if streams and "fused" in streams:
        comb = combined_exceedance({"fused": streams["fused"]}, spec, thr, rho, n=int(c["copula_samples"]))
        bt, ft = load_bundle(yml, "Pseudo_nitzschia_total")
        direct = {}
        if bt is not None:
            direct = _exceed_table({"fused": sample_streams(bt, ft, method, False).get("fused", {})},
                                   total_spec, thr, "direct_")
        for split in ("calibration", "test"):
            cand = {**comb[split], **direct.get(split, {})}
            for name, s in cand.items():
                y = totvals.reindex(s.index)
                ok = y.notna().to_numpy()
                rows.append({"split": split, "stream": name,
                             **binary_scores(s.to_numpy()[ok], (y[ok] >= thr).to_numpy())})
            lo, hi = comb[split].get("combined_lower_bound_fused"), comb[split].get("combined_upper_bound_fused")
            d = direct.get(split, {}).get("direct_fused")
            if lo is not None and d is not None:
                u = d.index.intersection(lo.index)
                inside = (d[u] >= lo[u] - 1e-6) & (d[u] <= hi[u] + 1e-6)
                coverage[split] = {"n": int(len(u)), "direct_within_bounds": float(inside.mean()) if len(u) else None}
    sc = pd.DataFrame(rows)
    sel = "copula"
    if len(sc):
        cal = sc[(sc["split"] == "calibration") & sc["stream"].isin(
            [f"combined_{m}_fused" for m in ("independence", "comonotonic", "copula")])]
        if len(cal) and cal["brier"].notna().any():
            sel = cal.sort_values("brier").iloc[0]["stream"].split("_")[1]
        sc.to_csv(out / "pn_combination_scores.csv", index=False)
    res = {"rho": rho, "selected_method": sel, "selection": "min calibration Brier (independence, "
           "comonotonic, copula); direct total-PN model is the primary product", "bounds_coverage": coverage,
           "threshold": thr}
    (out / "pn_combination.json").write_text(json.dumps(res, indent=2, default=str))
    LOG.info("PN combination -> %s (selected %s)", out / "pn_combination.json", sel)
    return res


def _combination(yml):
    p = cmp_dir(yml) / "pn_combination.json"
    if p.exists():
        return json.loads(p.read_text())
    LOG.warning("no pn_combination.json (run combine first); copula with rho = 0.41")
    return {"rho": {"all": {"rho": 0.41}}, "selected_method": "copula"}


# ---------------------------------------------------------------------------
# points
# ---------------------------------------------------------------------------

def charm_at_samples(c, samples: pd.DataFrame, var: str, lead: int) -> pd.DataFrame:
    arc = hc.CharmArchive(c, lead)
    rows = []
    cols = ["sample_uid", "charm_p", "charm_mode", "charm_chla", "charm_version", "charm_p_nearest",
            "charm_p_offcoast", "charm_coast_km", "charm_edge_share"]
    for d, g in samples.groupby(samples["date"].dt.normalize()):
        f = arc.read(d, [var, "chla_filled"])
        if f is None:
            continue
        la, lo = g["Latitude"].to_numpy(), g["Longitude"].to_numpy()
        e = hc.point_values_ext(f[var], f["lat"], f["lon"], la, lo, c["radius_km"], c["fallback_km"],
                                float(c.get("coast_exclude_km", 6.0)))
        ch, _ = hc.point_values(f["chla_filled"], f["lat"], f["lon"], la, lo, c["radius_km"], c["fallback_km"])
        rows.append(pd.DataFrame({"sample_uid": g["sample_uid"].astype(str).to_numpy(), "charm_p": e["mean"],
                                  "charm_mode": e["mode"], "charm_chla": ch, "charm_version": f["version"],
                                  "charm_p_nearest": e["nearest"], "charm_p_offcoast": e["offcoast"],
                                  "charm_coast_km": e["coast_km"], "charm_edge_share": e["edge_share"]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)


def _season(d):
    return pc.season_of(d)


def run_points(yml, target: str) -> dict[str, Any]:
    c = cfg_of(yml)
    t = TARGETS[target]
    thr = t["threshold"]
    out = cmp_dir(yml) / "points" / target
    out.mkdir(parents=True, exist_ok=True)
    method = (yml["uq"].get("fusion") or {}).get("pool", "log")
    ins = _insitu(yml)
    sf = _splits(yml, t["species"], ins)
    spec = spec_of(yml, t["species"])
    b, f = load_bundle(yml, t["species"])
    preds = {"test": {}, "calibration": {}}
    if b is not None and f is not None:
        st = sample_streams(b, f, method)
        for k, v in _exceed_table(st, spec, thr, "direct_" if target == "pn" else "").items():
            preds[k].update(v)
    if target == "pn":
        comb = _combination(yml)
        streams, gspec = size_class_streams(yml, method)
        if streams:
            ms = ["lower_bound", "upper_bound", comb.get("selected_method", "copula")]
            ms = list(dict.fromkeys(ms + ["independence", "comonotonic", "copula"]))
            for k, v in combined_exceedance(streams, gspec, thr, comb["rho"], ms, int(c["copula_samples"])).items():
                preds[k].update(v)
    base_rate = float((sf.loc[sf["split"] == "train", "value"] >= thr).mean()) if (sf["split"] == "train").any() else None
    info = sf.set_index(sf["sample_uid"].astype(str))
    scores, paired, rel = [], [], []
    tables = []
    for lead in c["leads"]:
        test = info[info["split"] == "test"]
        cal = info[info["split"] == "calibration"]
        ch_t = charm_at_samples(c, test.reset_index(drop=True), hc.TARGET_VARIABLE[target], lead).set_index("sample_uid")
        ch_c = charm_at_samples(c, cal.reset_index(drop=True), hc.TARGET_VARIABLE[target], lead).set_index("sample_uid")
        T = test.join(ch_t, how="left")
        Ccal = cal.join(ch_c, how="left")
        T["y"] = (T["value"] >= thr).astype(int)
        Ccal["y"] = (Ccal["value"] >= thr).astype(int)
        for name, s in preds["test"].items():
            T[name] = s.reindex(T.index).to_numpy()
        for name, s in preds["calibration"].items():
            Ccal[name] = s.reindex(Ccal.index).to_numpy()
        chl = pd.to_numeric(T.get("Avg_Chloro", pd.Series(np.nan, index=T.index)), errors="coerce")
        T["chl_source"] = np.where(chl.notna(), "pier", "charm")
        T["chl_used"] = chl.fillna(T["charm_chla"])
        T["chl_bin"] = pd.cut(T["chl_used"], CHL_BINS, labels=CHL_LABELS, right=False).astype(str)
        T["year"] = T["date"].dt.year.astype(str)
        T["season"] = _season(T["date"])
        T["lead"] = lead
        T["coast_bin"] = pd.cut(T["charm_coast_km"], COAST_BINS, labels=COAST_LABELS, right=False).astype(str)
        tables.append(T.reset_index(drop=True))
        streams_here = [v for v in CHARM_VARIANTS if v in T] + list(preds["test"])
        cut = {s: tune_cutoff(Ccal[s], Ccal["y"]) if s in Ccal else None for s in streams_here}
        for s in streams_here:
            if s not in T:
                continue
            for stratum in ("all", "year", "season", "charm_version", "source", "chl_bin", "charm_mode", "coast_bin"):
                groups = [("all", T)] if stratum == "all" else list(T.groupby(stratum))
                for lvl, g in groups:
                    ok = g[s].notna() & g["charm_p"].notna()      # common sample set with C-HARM
                    if ok.sum() == 0:
                        continue
                    scores.append({"target": target, "lead": lead, "stream": s, "stratum": stratum,
                                   "level": str(lvl), **binary_scores(g.loc[ok, s], g.loc[ok, "y"], base_rate,
                                                                      cut.get(s))})
            ok = T[s].notna() & T["charm_p"].notna()
            if ok.sum():
                r = reliability(T.loc[ok, s], T.loc[ok, "y"])
                rel.append(r.assign(target=target, lead=lead, stream=s))
                if s not in CHARM_VARIANTS:
                    pb = paired_bootstrap(T[ok].reset_index(drop=True), s, "charm_p", int(c["n_bootstrap"]))
                    paired.append({"target": target, "lead": lead, "stream": s, "vs": "charm", **pb})
                    ok2 = ok & T["charm_p_offcoast"].notna() if "charm_p_offcoast" in T else None
                    if ok2 is not None and ok2.sum():
                        pb = paired_bootstrap(T[ok2].reset_index(drop=True), s, "charm_p_offcoast",
                                              int(c["n_bootstrap"]))
                        paired.append({"target": target, "lead": lead, "stream": s, "vs": "charm_offcoast", **pb})
    pd.concat(tables, ignore_index=True).to_csv(out / f"points_{target}.csv.gz", index=False)
    pd.DataFrame(scores).to_csv(out / f"scores_{target}.csv", index=False)
    pd.DataFrame(paired).to_csv(out / f"paired_{target}.csv", index=False)
    if rel:
        pd.concat(rel, ignore_index=True).to_csv(out / f"reliability_{target}.csv", index=False)
    extra = {}
    if target == "pn":
        extra["ceiling_empirical"] = ceiling_empirical(sf, c["consistency"])
    (out / f"points_{target}.json").write_text(json.dumps(
        {"target": target, "threshold": thr, "base_rate_train": base_rate, "n_test": int((sf["split"] == "test").sum()),
         "streams": list(CHARM_VARIANTS) + list(preds["test"]), **extra}, indent=2, default=str))
    LOG.info("points %s -> %s", target, out)
    return {"out": str(out)}


def ceiling_empirical(sf: pd.DataFrame, cc: dict[str, Any]) -> dict[str, Any]:
    """Share of in-situ samples whose total PN exceeds the chlorophyll ceiling (pier Avg_Chloro)."""
    if "Avg_Chloro" not in sf:
        return {}
    chl = pd.to_numeric(sf["Avg_Chloro"], errors="coerce")
    ok = chl.notna() & (chl > 0)
    nmax = chl[ok] * 1e6 / cc["q_chl_min_pg"] * cc["tolerance"]
    v = sf.loc[ok, "value"]
    return {"n": int(ok.sum()), "fraction_above_ceiling": float((v > nmax).mean()) if ok.any() else None,
            "q_chl_min_pg": cc["q_chl_min_pg"]}


# ---------------------------------------------------------------------------
# maps
# ---------------------------------------------------------------------------

def _uq_files(root: Path, kind: str, token: str, ds: str):
    pat = (f"SITFUSE_MULTI.{ds}.L3m.DAY.{token}.fused_uq.tif" if kind == "fused" else None)
    return glob.glob(str(root / "fused" / "**" / pat), recursive=True)


def _read_bands(path, names=None):
    from sit_fuse.pipelines.habs.uq.hab_multisensor_fusion import _read_uq
    b, prof = _read_uq(path)
    return (b if names is None else {k: b.get(k) for k in names}), prof


def _fused_P(root, token, ds, C):
    hits = _uq_files(root, "fused", token, ds)
    if not hits:
        return None, None
    b, prof = _read_bands(hits[0])
    if f"fused_p_{C - 1}" not in b:
        return None, prof
    return np.stack([b[f"fused_p_{k}"] for k in range(C)], -1), prof


def charm_grid_for(arc: hc.CharmArchive, prof) -> tuple[dict, np.ndarray, np.ndarray, tuple]:
    """C-HARM grid cropped to the product extent: (profile, lat, lon, (i0, i1, j0, j1))."""
    lat, lon = arc.grid()
    t = prof["transform"]
    x0, x1 = t.c, t.c + t.a * prof["width"]
    y1, y0 = t.f, t.f + t.e * prof["height"]
    i = np.flatnonzero((lat >= min(y0, y1)) & (lat <= max(y0, y1)))
    j = np.flatnonzero((lon >= min(x0, x1)) & (lon <= max(x0, x1)))
    sl = (i.min(), i.max() + 1, j.min(), j.max() + 1)
    la, lo = lat[sl[0]:sl[1]], lon[sl[2]:sl[3]]
    return hc.charm_profile(la, lo), la, lo, sl


def _crop_north_up(a, sl):
    return None if a is None else a[sl[0]:sl[1], sl[2]:sl[3]][::-1]


def coastal_strip(land_nan: np.ndarray, km: float, dlat: float) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    d = distance_transform_edt(~land_nan) * 111.0 * abs(dlat)
    return (~land_nan) & (d <= km)


def _agree(ax, by, tag) -> dict[str, float]:
    po = float((ax == by).mean())
    pe = float(ax.mean() * by.mean() + (1 - ax.mean()) * (1 - by.mean()))
    inter, union = int((ax & by).sum()), int((ax | by).sum())
    return {f"kappa_{tag}": float((po - pe) / (1 - pe)) if pe < 1 else np.nan,
            f"iou_{tag}": inter / union if union else np.nan}


def map_metrics(a, b, cut_a: float | None = None, cut_b: float | None = None) -> dict[str, float]:
    """
    a = ours, b = C-HARM on the same grid. kappa / IoU at 0.5; at the calibration-tuned point cutoffs
    (cut_a, cut_b) when given; and equal-area: b at cut_b (0.5 without it), a at the quantile that
    flags the same share of the overlap (rank agreement of the alert areas, no calibration needed).
    """
    from scipy.stats import spearmanr
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    r = {"n_overlap": n}
    if n < 10:
        return r
    x, y = a[ok], b[ok]
    ax, by = x >= 0.5, y >= 0.5
    r.update(spearman=float(spearmanr(x, y).statistic), bias=float((x - y).mean()), mae=float(np.abs(x - y).mean()),
             **_agree(ax, by, "0p5"), ours_alert_frac=float(ax.mean()), charm_alert_frac=float(by.mean()))
    if cut_a is not None and cut_b is not None:
        at, bt = x >= cut_a, y >= cut_b
        r.update(**_agree(at, bt, "tuned"), ours_alert_frac_tuned=float(at.mean()),
                 charm_alert_frac_tuned=float(bt.mean()), cut_ours=float(cut_a), cut_charm=float(cut_b))
    cb = 0.5 if cut_b is None else float(cut_b)
    be = y >= cb
    fb = float(be.mean())
    if 0 < fb < 1:
        qa = float(np.quantile(x, 1 - fb))
        r.update(**_agree(x >= qa, be, "eqarea"), eqarea_frac=fb, eqarea_cut_ours=qa)
    return r


def points_stream_name(target: str, map_stream: str, selected_method: str | None = None) -> str:
    """Map-layer name -> points stream name (scores_<target>.csv)."""
    if map_stream == "charm":
        return "charm_p"
    if target == "pn":
        if map_stream == "combined":
            return f"combined_{selected_method or 'copula'}_fused"
        return "direct_" + map_stream
    return map_stream


def tuned_cutoffs(yml, target: str, lead: int = 0) -> dict[str, float]:
    """{points stream: calibration-tuned cutoff} from points/<target>/scores_<target>.csv (stratum all)."""
    p = cmp_dir(yml) / "points" / target / f"scores_{target}.csv"
    if not p.exists() or os.path.getsize(p) < 2:
        LOG.warning("no point scores for %s (run points first): tuned map cutoffs skipped", target)
        return {}
    s = pd.read_csv(p)
    if "cutoff_tuned" not in s:
        return {}
    s = s[(s["stratum"] == "all") & (s["lead"] == lead) & s["cutoff_tuned"].notna()]
    return dict(zip(s["stream"], s["cutoff_tuned"].astype(float)))


def run_maps(yml, target: str, year: int) -> dict[str, Any]:
    from sit_fuse.pipelines.habs.uq.hab_multisensor_fusion import fuse_day
    from sit_fuse.train.sf_uq_utils import write_band_stack
    c = cfg_of(yml)
    t = TARGETS[target]
    thr = t["threshold"]
    root = out_root_of(yml)
    out = cmp_dir(yml) / "maps" / target / str(year)
    out.mkdir(parents=True, exist_ok=True)
    spec = spec_of(yml, t["species"])
    b, _ = load_bundle(yml, t["species"])
    fz = yml["uq"].get("fusion") or {}
    comb = _combination(yml) if target == "pn" else None
    gspec = spec_of(yml, PN_GROUPS[0]) if target == "pn" else None
    rows, hov = [], []
    sel_m = comb.get("selected_method", "copula") if comb else None
    for lead in c["map_leads"]:
        cuts = tuned_cutoffs(yml, target, lead)
        arc = hc.CharmArchive(c, lead)
        if not len(arc.dates()):
            LOG.warning("no C-HARM cache for lead %d", lead)
            continue
        grid = None
        for d in [x for x in arc.dates() if x.year == int(year)]:
            ds = f"{d:%Y%m%d}"
            layers, src_prof = {}, None
            P, prof = _fused_P(root, spec.fname_token, ds, spec.n_classes)
            if P is not None:
                layers["fused"] = exceedance_probability(P, spec.edges, thr, spec.detection_floor)
                src_prof = prof
            if b is not None:
                w = b.get("fusion_weights") or {}
                for fam in FAMILIES:
                    ss = [s for s in b["sensors"] if SENSORS[s].family == fam]
                    if not ss:
                        continue
                    F, _, prof2 = fuse_day(str(root / "products"), d, spec.fname_token, ss, w, spec.n_classes,
                                           fz.get("pool", "log"), fz.get("ref_local_hour", 10.0),
                                           fz.get("tau_hours", 4.0), b["pi_true"], fz.get("quality", "selective"))
                    if F is not None:
                        layers[f"type_{fam}"] = exceedance_probability(F, spec.edges, thr, spec.detection_floor)
                        src_prof = src_prof or prof2
            if target == "pn":
                PS, p1 = _fused_P(root, gspec.fname_token, ds, gspec.n_classes)
                PD, _ = _fused_P(root, spec_of(yml, PN_GROUPS[1]).fname_token, ds, gspec.n_classes)
                if PS is not None and PD is not None and PS.shape == PD.shape:
                    H, W, C = PS.shape
                    fs, fd = PS.reshape(-1, C), PD.reshape(-1, C)
                    ok = np.isfinite(fs).all(1) & np.isfinite(fd).all(1)
                    v = np.full(H * W, np.nan)
                    if ok.any():
                        rho = pc.rho_for(np.repeat(d, ok.sum()), comb["rho"])
                        v[ok] = pc.combine(fs[ok], fd[ok], gspec.edges, gspec.detection_floor, thr,
                                           comb.get("selected_method", "copula"), rho, int(c["copula_samples"]))
                    layers["combined"] = v.reshape(H, W)
                    src_prof = src_prof or p1
            if not layers:
                continue
            f = arc.read(d, [hc.TARGET_VARIABLE[target], "chla_filled"])
            if grid is None:
                grid = charm_grid_for(arc, src_prof)
                gprof, la, lo, sl = grid
                land = ~np.isfinite(_crop_north_up(f["chla_filled"], sl))
                strip = coastal_strip(land, float(c["coastal_km"]), float(np.median(np.diff(la))))
                lat2 = np.repeat(la[::-1][:, None], len(lo), 1)
                bands_lat = np.floor(lat2 / float(c["hovmoller_dlat"])) * float(c["hovmoller_dlat"])
            gprof, la, lo, sl = grid
            ch = _crop_north_up(f[hc.TARGET_VARIABLE[target]], sl)
            stack = {"charm": ch, "charm_chla": _crop_north_up(f["chla_filled"], sl)}
            for k, a in layers.items():
                a = np.where(np.isfinite(a) & (a != NODATA), a, np.nan)
                stack[k] = hc.regrid_to(a, src_prof, gprof, "average")
            write_band_stack(out / f"CHARM_CMP.{ds}.{target}.L{lead}.tif",
                             {k: np.nan_to_num(v, nan=NODATA) for k, v in stack.items()}, gprof,
                             tags={"charm_version": f["version"], "charm_dataset": f["dataset_id"], "lead": lead,
                                   "target": target, "threshold": thr}, nodata=NODATA)
            cov_ref = np.isfinite(ch) & strip
            for k in layers:
                m = map_metrics(stack[k], ch, cuts.get(points_stream_name(target, k, sel_m)), cuts.get("charm_p"))
                m["coverage_fraction"] = float((np.isfinite(stack[k]) & cov_ref).sum() / max(cov_ref.sum(), 1))
                rows.append({"target": target, "date": d, "lead": lead, "charm_version": f["version"],
                             "stream": k, **m})
            for k, a in [("charm", ch)] + [(k, stack[k]) for k in layers]:
                ok = np.isfinite(a) & strip
                if not ok.any():
                    continue
                g = pd.DataFrame({"lat_band": bands_lat[ok], "p": a[ok]}).groupby("lat_band")["p"].agg(["mean", "size"])
                hov.append(g.reset_index().assign(date=d, lead=lead, stream=k))
    pd.DataFrame(rows).to_csv(out / f"map_metrics_{target}_{year}.csv", index=False)
    if hov:
        pd.concat(hov, ignore_index=True).rename(columns={"size": "n"}).to_csv(
            out / f"hovmoller_{target}_{year}.csv.gz", index=False)
    LOG.info("maps %s %s: %d day-streams -> %s", target, year, len(rows), out)
    return {"n": len(rows), "out": str(out)}


# ---------------------------------------------------------------------------
# consistency
# ---------------------------------------------------------------------------

CHL_PATTERNS = ["{prefix}.{ds}.L3m.DAY.CHL.chlor_a.4km*.nc", "{prefix}.{ds}.L3m.DAY.BGC.V*.4km*.nc"]


def read_l3m_chl(path, prof):
    """chlor_a from an L3m CHL / BGC file, averaged onto prof (NaN outside)."""
    t = prof["transform"]
    x0, x1 = t.c, t.c + t.a * prof["width"]
    y1, y0 = t.f, t.f + t.e * prof["height"]
    with hc._open(path) as ds:
        lat, lon = ds["lat"].values, ds["lon"].values
        i = np.flatnonzero((lat >= y0 - 0.1) & (lat <= y1 + 0.1))
        j = np.flatnonzero((lon >= x0 - 0.1) & (lon <= x1 + 0.1))
        if not len(i) or not len(j):
            return None
        a = np.asarray(ds["chlor_a"].isel(lat=slice(i.min(), i.max() + 1), lon=slice(j.min(), j.max() + 1)).values,
                       np.float32)
        la, lo = lat[i.min():i.max() + 1], lon[j.min():j.max() + 1]
    a[(a <= 0) | ~np.isfinite(a)] = np.nan
    if la[0] < la[-1]:
        a, la = a[::-1], la[::-1]
    return hc.regrid_to(a, hc.charm_profile(la[::-1], lo), prof, "average")


def satellite_chl(c, d, prof, sensors):
    stacks = []
    ds = f"{pd.Timestamp(d):%Y%m%d}"
    for s in sensors:
        root = c["chl_dirs"].get(s)
        if not root:
            continue
        for pat in CHL_PATTERNS:
            hits = glob.glob(os.path.join(root, "**", pat.format(prefix=SENSORS[s].prefix, ds=ds)), recursive=True)
            if hits:
                try:
                    a = read_l3m_chl(sorted(hits)[0], prof)
                    if a is not None:
                        stacks.append(a)
                except Exception as e:  # noqa: BLE001
                    LOG.warning("chl %s: %s", hits[0], e)
                break
    if not stacks:
        return None
    with np.errstate(all="ignore"):
        return np.nanmedian(np.stack(stacks), 0)


def run_consistency(yml, year: int) -> dict[str, Any]:
    from sit_fuse.train.sf_uq_utils import write_band_stack
    c = cfg_of(yml)
    cc = c["consistency"]
    root = out_root_of(yml)
    out = cmp_dir(yml) / "consistency" / str(year)
    out.mkdir(parents=True, exist_ok=True)
    st, sS, sD = (spec_of(yml, k) for k in ("Pseudo_nitzschia_total",) + tuple(PN_GROUPS))
    sp = spec_of(yml, "pDA")
    sensors = yml["uq"].get("sensors") or list(SENSORS)
    from sit_fuse.pipelines.habs.uq.hab_uq_train import expand_sensors
    sensors = [s for s in expand_sensors(sensors) if s in SENSORS]
    arc = hc.CharmArchive(c, 0)
    days = sorted({pd.Timestamp(Path(p).name.split(".")[1]) for p in
                   glob.glob(str(root / "fused" / "**" / f"SITFUSE_MULTI.{year}*.L3m.DAY.{st.fname_token}.fused_uq.tif"),
                             recursive=True)})
    rows = []
    for d in days:
        ds = f"{d:%Y%m%d}"
        PT, prof = _fused_P(root, st.fname_token, ds, st.n_classes)
        if PT is None:
            continue
        H, W, C = PT.shape
        flags = {}
        chl = satellite_chl(c, d, prof, sensors)
        src = "satellite"
        if chl is None or not np.isfinite(chl).any():
            f = arc.read(d, ["chla_filled"])
            chl, src = (None, "none") if f is None else (hc.regrid_to(f["chla_filled"][::-1], hc.charm_profile(
                f["lat"], f["lon"]), prof, "bilinear"), "charm")
        Pf = PT.reshape(-1, C)
        okT = np.isfinite(Pf).all(1)
        r = {"date": d, "chl_source": src, "n_valid": int(okT.sum())}
        if chl is not None:
            nmax = chl.reshape(-1) * 1e6 / cc["q_chl_min_pg"] * cc["tolerance"]
            ok = okT & np.isfinite(nmax) & (nmax > 0)
            pv = np.full(H * W, np.nan)
            if ok.any():
                pv[ok] = pc.exceedance_vec(Pf[ok], st.edges, nmax[ok], st.detection_floor)
            flags["ceiling_p_violation"] = pv.reshape(H, W)
            flags["ceiling_flag"] = np.where(np.isfinite(pv), (pv > 0.5).astype(float), np.nan).reshape(H, W)
            flags["chl_used"] = chl
            r.update(ceiling_n=int(ok.sum()), ceiling_mean_p=float(np.nanmean(pv)) if ok.any() else np.nan,
                     ceiling_flag_rate=float(np.nanmean(pv[ok] > 0.5)) if ok.any() else np.nan)
        PS, _ = _fused_P(root, sS.fname_token, ds, sS.n_classes)
        PD, _ = _fused_P(root, sD.fname_token, ds, sD.n_classes)
        pT = exceedance_probability(Pf, st.edges, 1e4, st.detection_floor)
        if PS is not None and PD is not None and PS.shape == PT.shape:
            lo, hi = pc.bounds(PS.reshape(-1, C), PD.reshape(-1, C), sS.edges, sS.detection_floor, 1e4)
            ok = okT & np.isfinite(lo) & np.isfinite(hi)
            v = np.where(ok, ((pT < lo - cc["eps"]) | (pT > hi + cc["eps"])).astype(float), np.nan)
            flags["sizeclass_flag"] = v.reshape(H, W)
            r.update(sizeclass_n=int(ok.sum()), sizeclass_flag_rate=float(np.nanmean(v[ok])) if ok.any() else np.nan,
                     sizeclass_below_lower=float(np.mean(pT[ok] < lo[ok] - cc["eps"])) if ok.any() else np.nan)
        PA, _ = _fused_P(root, sp.fname_token, ds, sp.n_classes)
        if PA is not None and PA.shape[:2] == PT.shape[:2]:
            pa = exceedance_probability(PA.reshape(-1, sp.n_classes), sp.edges, 0.5, sp.detection_floor)
            pn_need = exceedance_probability(Pf, st.edges, 0.5 * 1e6 / cc["cda_max_pg"], st.detection_floor)
            ok = okT & np.isfinite(pa)
            v = np.where(ok, (pa > pn_need + cc["eps"]).astype(float), np.nan)
            flags["pda_pn_flag"] = v.reshape(H, W)
            r.update(pda_pn_n=int(ok.sum()), pda_pn_flag_rate=float(np.nanmean(v[ok])) if ok.any() else np.nan)
        if flags:
            fl = [flags[k] for k in ("ceiling_flag", "sizeclass_flag", "pda_pn_flag") if k in flags]
            with np.errstate(all="ignore"):
                import warnings
                warnings.simplefilter("ignore", RuntimeWarning)
                flags["any_flag"] = np.nanmax(np.stack(fl), 0) if fl else np.full((H, W), np.nan)
            write_band_stack(out / f"CONSISTENCY.{ds}.tif", {k: np.nan_to_num(v, nan=NODATA) for k, v in flags.items()},
                             prof, tags={"chl_source": src, **{k: str(v) for k, v in cc.items()}}, nodata=NODATA)
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / f"consistency_{year}.csv", index=False)
    LOG.info("consistency %s: %d days -> %s", year, len(rows), out)
    return {"n_days": len(rows)}


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def run_summary(yml) -> dict[str, Any]:
    base = cmp_dir(yml)
    res: dict[str, Any] = {}
    sc = [pd.read_csv(p) for p in glob.glob(str(base / "points" / "*" / "scores_*.csv"))]
    if sc:
        s = pd.concat(sc, ignore_index=True)
        s.to_csv(base / "summary_point_scores.csv", index=False)
        res["point_rows"] = len(s)
        ce = s[s["stream"].isin(CHARM_VARIANTS) & s["stratum"].isin(["all", "coast_bin"])]
        if len(ce):
            keep = [k for k in ("target", "lead", "stream", "stratum", "level", "n", "n_events", "event_rate",
                                "mean_p", "brier", "bss", "roc_auc", "cutoff_tuned") if k in ce]
            ce[keep].to_csv(base / "summary_coastal_edge.csv", index=False)
    pr = [pd.read_csv(p) for p in glob.glob(str(base / "points" / "*" / "paired_*.csv")) if os.path.getsize(p) > 1]
    if pr:
        pd.concat(pr, ignore_index=True).to_csv(base / "summary_paired.csv", index=False)
    mm = [pd.read_csv(p, parse_dates=["date"]) for p in glob.glob(str(base / "maps" / "*" / "*" / "map_metrics_*.csv"))
          if os.path.getsize(p) > 1]
    if mm:
        m = pd.concat(mm, ignore_index=True)
        m["year"] = m["date"].dt.year
        g = m.groupby(["target", "stream", "lead", "charm_version", "year"]).agg(
            days=("date", "nunique"), spearman=("spearman", "median"), bias=("bias", "median"),
            kappa_0p5=("kappa_0p5", "median"), iou_0p5=("iou_0p5", "median"),
            coverage=("coverage_fraction", "median"),
            **{c: (c, "median") for c in ("kappa_tuned", "iou_tuned", "kappa_eqarea", "iou_eqarea",
                                          "eqarea_frac", "cut_ours", "cut_charm") if c in m}).reset_index()
        g.to_csv(base / "summary_map_agreement.csv", index=False)
    cs = [pd.read_csv(p) for p in glob.glob(str(base / "consistency" / "*" / "consistency_*.csv")) if os.path.getsize(p) > 1]
    if cs:
        cdf = pd.concat(cs, ignore_index=True)
        cdf.describe().to_csv(base / "summary_consistency.csv")
    (base / "summary.json").write_text(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["combine", "points", "maps", "consistency", "summary"])
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--target", choices=list(TARGETS))
    ap.add_argument("--year", type=int)
    a = ap.parse_args()
    y = hc.read_yaml_any(a.yaml)
    if a.action == "combine":
        run_combine(y)
    elif a.action == "points":
        run_points(y, a.target)
    elif a.action == "maps":
        run_maps(y, a.target, a.year)
    elif a.action == "consistency":
        run_consistency(y, a.year)
    else:
        run_summary(y)
