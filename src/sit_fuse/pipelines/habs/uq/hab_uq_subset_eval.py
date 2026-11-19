"""
Evaluation-only subsets of the held-out test samples by in-situ provenance
(source, method, station), e.g. the OOS pier network used in the 2026 paper.
OOS = Ocean Observing System (CalHABMAP piers: combined_full + calhabmap_erddap), not
out-of-sample. These subsets reuse the same split, so pier stations also appear in
training; for stations never seen in training set uq.splits.holdout_locations and add a
{location_codes: [...]} subset. A subset that contains every test sample (e.g. oos_piers
while uq.insitu_filter keeps only pier rows) is flagged identical_to_full_test_set.

Models, calibrators, splits and fusion weights are unchanged: the trained
bundle and the cached matchup frame (models/matchups_<token>.parquet) are
reused and only the scored test samples are restricted. To also *train* on a
subset, set uq.insitu_filter instead and rerun from the train stage.

Provenance is taken from the matchup frame when it carries `source`/`method`
(runs after this change), otherwise re-attached from the in-situ table by
sample_uid, with a (station, date, value) fallback. Source filters need the
non-strict table (combined_augmented.csv); the strict CSV has no provenance
columns, so pass it with --insitu.

Config (optional; CLI flags override):
  uq:
    evaluation:
      subsets:
        oos_piers: {sources: oos_piers}            # combined_full + calhabmap_erddap
        non_pier: {exclude_sources: oos_piers}
        ifcb: {methods: [ifcb]}
        holdout_piers: {location_codes: [SMP, MBF]}  # with uq.splits.holdout_locations
      subset_insitu_file: /path/combined_augmented.csv
      subset_recalibrate_fusion: false             # true: fusion weights from subset calibration rows

CLI:
  python -m sit_fuse.pipelines.habs.uq.hab_uq_subset_eval -y run.yaml \\
      --insitu /mnt/data1/habs/insitu_augmented/combined_augmented.csv [--preset oos_piers]
Outputs <uq.out_dir>/evaluation/subsets/<name>/hab_uq_evaluation_<token>.json,
hab_uq_test_predictions_<token>.csv, and subset_summary.csv (one row per
species x product).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from sit_fuse.pipelines.habs.uq.hab_insitu import (
    SOURCE_PRESETS, insitu_mask, load_insitu, species_frame)
from sit_fuse.pipelines.habs.uq.hab_uq_common import insitu_date_kwargs, species_spec
from sit_fuse.pipelines.habs.uq.hab_uq_evaluate import (
    eval_options, grouped_bootstrap, prediction_table, predict_samples, score_block)

LOG = logging.getLogger(__name__)
PROV = ("source", "method")
SUMMARY_KEYS = ("balanced_accuracy", "weighted_f1", "macro_f1", "accuracy", "qwk", "rps",
                "rpss_vs_climatology", "classwise_ece", "interval_90_coverage", "interval_90_recal_coverage")


def default_subsets() -> dict[str, dict]:
    return {k: {"sources": [k]} for k in SOURCE_PRESETS}


def attach_provenance(frame: pd.DataFrame, yml: dict, use_key: str,
                      insitu_path: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Add source/method to matchup rows from the in-situ table (same load arguments as training)."""
    uq = yml["uq"]
    path = insitu_path or uq.get("evaluation", {}).get("subset_insitu_file") or \
        uq.get("insitu_file", yml.get("insitu_file"))
    ins = load_insitu(path, max_depth_m=uq.get("max_depth_m", 1.0),
                      **insitu_date_kwargs(uq))
    info = {"insitu_file": path, "has_source": bool(ins.attrs.get("has_source")),
            "has_method": bool(ins.attrs.get("has_method"))}
    out = frame.drop(columns=[c for c in PROV if c in frame.columns])
    m = ins.set_index("sample_uid")[list(PROV)]
    hit = out["sample_uid"].astype(str).isin(m.index)
    uids = out["sample_uid"].astype(str).unique()
    info["uid_match"] = float(pd.Index(uids).isin(m.index).mean()) if len(uids) else 0.0
    out = out.join(m, on="sample_uid")
    if not hit.all():
        # different table / row order: fall back to station + date + reported value
        spec = species_spec(use_key, uq.get("species", {}).get(use_key))
        sf = species_frame(ins, spec)
        key = lambda d: (d["Location_Code"].astype(str) + "|" + pd.to_datetime(d["date"]).dt.strftime("%Y%m%d")
                         + "|" + d["value"].astype(float).round(6).astype(str))
        km = sf.assign(_k=key(sf)).drop_duplicates("_k").set_index("_k")[list(PROV)]
        miss = ~hit
        k = key(out.loc[miss])
        for c in PROV:
            out.loc[miss, c] = k.map(km[c]).to_numpy()
    info["row_match"] = float(out["source"].notna().mean())
    if info["row_match"] < 1.0:
        LOG.warning("%s: %.1f%% of matchup rows have no provenance in %s (excluded from source subsets)",
                    use_key, 100 * (1 - info["row_match"]), path)
    return out, info


def _events(meta, spec) -> dict[str, int]:
    s = meta.drop_duplicates("sample_uid")
    return {f"{t:g}": int((s["value"] >= t).sum()) for t in spec.alert_thresholds}


def evaluate_subsets(bundle: dict, frame: pd.DataFrame, yml: dict, out_dir, subsets: dict[str, dict] | None = None,
                     insitu_path: str | None = None, n_boot: int = 200) -> dict[str, Any]:
    from sit_fuse.pipelines.habs.uq.hab_multisensor_fusion import fuse_samples
    uq = yml["uq"]
    ev = uq.get("evaluation", {})
    subsets = subsets or ev.get("subsets") or default_subsets()
    subsets = {k: ({"sources": [v]} if isinstance(v, str) else v) for k, v in subsets.items()}
    spec = species_spec(bundle["species"], bundle["spec"])
    needs_prov = any(any(k in f for k in ("sources", "exclude_sources", "methods", "exclude_methods"))
                     for f in subsets.values())
    info = {}
    # always re-read provenance for source/method filters: a frame matched from the strict
    # table carries the load_insitu default (source=combined_full) and would pass every row
    if needs_prov:
        frame, info = attach_provenance(frame, yml, bundle["species"], insitu_path)
        if not info["has_source"]:
            raise ValueError(f"{info['insitu_file']} has no source/method columns (strict CSV); "
                             "pass the non-strict combined_augmented.csv with --insitu")
    P, meta, base = predict_samples(bundle, frame, "test")
    Pc, mc, _ = predict_samples(bundle, frame, "calibration")
    if P is None:
        return {"error": "no test matchups"}
    # sample_level keeps provenance only for frames built after this change
    prov = frame.drop_duplicates("sample_uid").set_index("sample_uid")[[c for c in PROV if c in frame.columns]]
    meta = meta.drop(columns=[c for c in PROV if c in meta.columns]).join(prov, on="sample_uid")
    if mc is not None:
        mc = mc.drop(columns=[c for c in PROV if c in mc.columns]).join(prov, on="sample_uid")
    pi = bundle["pi_true"]
    eo = eval_options(bundle)
    root = Path(out_dir)
    n_all = int(meta["sample_uid"].nunique())
    reports, rows = {}, []
    for name, flt in subsets.items():
        m = insitu_mask(meta.fillna({c: "" for c in PROV if c in meta.columns}), flt)
        n = int(meta.loc[m, "sample_uid"].nunique())
        rep = {"species": bundle["species"], "subset": name, "filter": flt, "provenance": info,
               "n_test_samples": n, "n_test_rows": int(m.sum()),
               "stations": sorted(meta.loc[m, "Location_Code"].astype(str).unique().tolist()),
               "sources": meta.loc[m, "source"].value_counts().to_dict() if "source" in meta else None,
               "events_by_threshold": _events(meta[m], spec)}
        # a provenance subset equal to the whole test set is not a separate test (e.g. oos_piers
        # while uq.insitu_filter leaves only pier rows); flag it instead of reporting it as one
        rep["identical_to_full_test_set"] = bool(m.all())
        rep["share_of_test_samples"] = n / max(n_all, 1)
        if m.all():
            LOG.warning("%s/%s: subset contains every test sample (%d); it repeats the full evaluation. "
                        "Check uq.insitu_filter (training/eval provenance) against the subset filter.",
                        bundle["species"], name, n)
        if m.sum() < 5:
            rep["error"] = "fewer than 5 test rows in subset"
            reports[name] = rep
            LOG.warning("%s/%s: %s", bundle["species"], name, rep["error"])
            continue
        Pm, mm, bm = P[m], meta[m].reset_index(drop=True), base[m]
        rep["overall"] = score_block(Pm, mm, spec, pi, bm, **eo)
        rep["overall_ci"] = grouped_bootstrap(Pm, mm, spec, pi, n_boot=n_boot,
                                              keys=("rps", "weighted_f1", "balanced_accuracy", "macro_f1"),
                                              decision=eo["decision"])
        rep["by_sensor"] = {s: score_block(Pm[i], mm.iloc[i], spec, pi, bm[i], **eo)
                            for s, i in mm.groupby("sensor").indices.items() if len(i) >= 5}
        if Pc is not None and len(bundle["sensors"]) > 1:
            cm = insitu_mask(mc.fillna({c: "" for c in PROV if c in mc.columns}), flt) \
                if ev.get("subset_recalibrate_fusion", False) else np.ones(len(mc), bool)
            fz = fuse_samples(bundle, Pc[cm], mc[cm].reset_index(drop=True), Pm, mm,
                              method=uq.get("fusion", {}).get("pool", "log"))
            rep["fused"] = score_block(fz["P_test"], fz["meta_test"], spec, pi, **eo)
            rep["fused"]["weights"] = fz["weights"]
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        prediction_table(Pm, mm, bm, spec, eo).to_csv(
            d / f"hab_uq_test_predictions_{spec.fname_token}.csv", index=False)
        with open(d / f"hab_uq_evaluation_{spec.fname_token}.json", "w") as f:
            json.dump(rep, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        for prod, blk in (("production", rep["overall"].get("production_baseline")),
                          ("uq_per_sensor", rep["overall"]["uq"]), ("fused", (rep.get("fused") or {}).get("uq"))):
            if not blk:
                continue
            r = {"species": bundle["species"], "subset": name, "product": prod,
                 "identical_to_full": rep["identical_to_full_test_set"],
                 "n_samples": n, "n_rows": int(m.sum()) if prod != "fused" else int(rep["fused"]["n_samples"]),
                 **{k: blk.get(k) for k in SUMMARY_KEYS}}
            for t, e in (blk.get("exceedance") or {}).items():
                r[f"auc_{t}"] = e.get("roc_auc")
                r[f"events_{t}"] = rep["events_by_threshold"].get(t)
            rows.append(r)
        reports[name] = rep
    return {"reports": reports, "summary_rows": rows}


def run(yml: dict, species: list[str] | None = None, subsets: dict | None = None,
        insitu_path: str | None = None, n_boot: int = 200) -> pd.DataFrame:
    from sit_fuse.pipelines.habs.uq.hab_uq_pipeline import _species_run
    uq = yml["uq"]
    out_root = Path(uq.get("out_dir", os.path.join(yml["final_product_dir"], "uq")))
    rows = []
    for use_key in species or _species_run(yml):
        spec = species_spec(use_key, uq.get("species", {}).get(use_key))
        bpath = out_root / "models" / f"hab_uq_{spec.fname_token}.joblib"
        mpq = out_root / "models" / f"matchups_{spec.fname_token}.parquet"
        mcsv = out_root / "models" / f"matchups_{spec.fname_token}.csv.gz"
        if not bpath.exists() or not (mpq.exists() or mcsv.exists()):
            LOG.warning("%s: missing bundle or cached matchups under %s", use_key, out_root / "models")
            continue
        bundle = joblib.load(bpath)
        frame = pd.read_parquet(mpq) if mpq.exists() else pd.read_csv(mcsv)
        res = evaluate_subsets(bundle, frame, yml, out_root / "evaluation" / "subsets", subsets, insitu_path, n_boot)
        rows += res.get("summary_rows", [])
    df = pd.DataFrame(rows)
    out = out_root / "evaluation" / "subsets" / "subset_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and len(df):          # keep rows from earlier subset runs, replace re-run ones
        old = pd.read_csv(out)
        key = set(zip(df["species"], df["subset"]))
        old = old[[(a, b) not in key for a, b in zip(old["species"], old["subset"])]]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out, index=False)
    LOG.info("wrote %s", out)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--insitu", help="in-situ table with source/method columns (non-strict CSV)")
    ap.add_argument("--species", nargs="+")
    ap.add_argument("--name", default=None, help="subset name for the flags below")
    ap.add_argument("--preset", nargs="+", help=f"source presets: {', '.join(SOURCE_PRESETS)}")
    ap.add_argument("--sources", nargs="+")
    ap.add_argument("--exclude-sources", nargs="+")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--location-codes", nargs="+")
    ap.add_argument("--exclude-location-codes", nargs="+")
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open(a.yaml) as f:
        yml = yaml.safe_load(f)
    flt = {k: v for k, v in {"sources": (a.preset or []) + (a.sources or []) or None,
                             "exclude_sources": a.exclude_sources, "methods": a.methods,
                             "location_codes": a.location_codes,
                             "exclude_location_codes": a.exclude_location_codes}.items() if v}
    subsets = {a.name or ("_".join(a.preset) if a.preset and len(flt) == 1 else "custom"): flt} if flt else None
    df = run(yml, a.species, subsets, a.insitu, a.n_boot)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.round(3).to_string(index=False) if len(df) else "no subset results")


if __name__ == "__main__":
    main()
