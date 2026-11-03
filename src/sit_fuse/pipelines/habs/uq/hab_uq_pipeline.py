"""
HAB UQ pipeline driver. Uses the same YAML as hab_post_inference_pipeline.py
plus a `uq:` block (see config/piplines/habs/hab_uq_s_ca_multi_sensor.example.yaml).

  python -m sit_fuse.pipelines.habs.uq.hab_uq_pipeline -y <hab yaml> [--stages train,evaluate,infer,fuse]

Stages (per species in HAB_USE_KEYS[roi] or uq.species_run):
  export_heads  (optional, once) encoder head outputs + MC / OOD bands via
            hab_inference_pipeline.run_head_output_export; existing
            *.head_outputs.tif are kept
  train     matchups -> cross-fit encoder -> classifier -> calibration -> bundle
  evaluate  held-out test metrics per sensor / station / season + fused
  infer     per-sensor, per-day *.uq.tif band stacks
  fuse      multi-sensor daily fused + temporally filtered products
  ablation  (optional) retrain + test-score per input variant
            (uq.evaluation.ablation, default label_only / head_outputs /
            head_outputs_mc_ood), same samples and splits
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import copy

import joblib
import numpy as np
import pandas as pd

from sit_fuse.pipelines.habs.uq.hab_uq_common import instrument_dirs, species_spec

LOG = logging.getLogger("hab-uq-pipeline")


def _species_run(yml):
    if yml.get("uq", {}).get("species_run"):
        return yml["uq"]["species_run"]
    from sit_fuse.pipelines.habs.hab_post_inference_constants import HAB_USE_KEYS
    return HAB_USE_KEYS[yml["roi"]]


def run_hab_uq_pipeline(yml, stages=("train", "evaluate", "infer", "fuse")):
    from sit_fuse.pipelines.habs.uq.hab_uq_train import train_species
    from sit_fuse.pipelines.habs.uq.hab_uq_evaluate import evaluate_bundle, predict_samples, score_block
    from sit_fuse.pipelines.habs.uq.hab_uq_inference import run_inference
    from sit_fuse.pipelines.habs.uq.hab_multisensor_fusion import fuse_samples, run_fusion

    uq = yml["uq"]
    out_root = Path(uq.get("out_dir", os.path.join(yml["final_product_dir"], "uq")))
    summary = {}
    if "export_heads" in stages:
        from sit_fuse.pipelines.habs.hab_inference_pipeline import run_head_output_export
        run_head_output_export(yml)
    for use_key in _species_run(yml):
        spec = species_spec(use_key, uq.get("species", {}).get(use_key))
        bpath = out_root / "models" / f"hab_uq_{spec.fname_token}.joblib"
        bundle = None
        if "train" in stages:
            bundle = train_species(yml, use_key, out_root / "models")
        elif bpath.exists():
            bundle = joblib.load(bpath)
            bundle["path"] = str(bpath)
        if bundle is None:
            continue
        frame = bundle.pop("_frame", None)
        mpq = out_root / "models" / f"matchups_{spec.fname_token}.parquet"
        mcsv = out_root / "models" / f"matchups_{spec.fname_token}.csv.gz"
        if frame is None and (mpq.exists() or mcsv.exists()):
            frame = pd.read_parquet(mpq) if mpq.exists() else pd.read_csv(mcsv)
        elif frame is not None:
            frame.to_parquet(mpq) if _has_parquet() else frame.to_csv(mcsv, index=False)
        if "evaluate" in stages and frame is not None:
            rep = evaluate_bundle(bundle, frame, out_root / "evaluation",
                                  n_boot=int(uq.get("evaluation", {}).get("n_bootstrap", 500)))
            Pc, mc, _ = predict_samples(bundle, frame, "calibration")
            Pt, mt, _ = predict_samples(bundle, frame, "test")
            if Pc is not None and Pt is not None and len(bundle["sensors"]) > 1:
                fz = fuse_samples(bundle, Pc, mc, Pt, mt, method=uq.get("fusion", {}).get("pool", "log"))
                rep["fused"] = score_block(fz["P_test"], fz["meta_test"], spec, bundle["pi_true"])
                rep["fused"]["weights"] = fz["weights"]
                bundle["fusion_weights"] = fz["weights"]
                joblib.dump({k: v for k, v in bundle.items() if k != "path"}, bundle["path"])
            with open(out_root / "evaluation" / f"hab_uq_evaluation_{spec.fname_token}.json", "w") as f:
                json.dump(rep, f, indent=2, default=str)
            summary[use_key] = {k: rep.get(k) for k in ("overall", "fused")}
        if "ablation" in stages:
            summary.setdefault(use_key, {})["ablation"] = run_ablation(yml, use_key, out_root)
        dates = pd.date_range(uq.get("start_date", yml.get("start_date")),
                              uq.get("end_date", yml.get("end_date")), freq="D")
        if "infer" in stages:
            dirs = {s: uq.get("product_dirs", {}).get(s) or instrument_dirs(yml, s) for s in bundle["sensors"]}
            run_inference(bundle["path"], dates, bundle["sensors"], dirs, str(out_root / "products"))
        if "fuse" in stages:
            fcfg = uq.get("fusion", {})
            run_fusion(str(out_root / "products"), str(out_root / "fused"), dates, bundle,
                       method=fcfg.get("pool", "log"), temporal_filter=fcfg.get("temporal_filter", True),
                       ref_local_hour=fcfg.get("ref_local_hour", 10.0), tau_hours=fcfg.get("tau_hours", 4.0),
                       stay=fcfg.get("transition_stay", 0.7), quality=fcfg.get("quality", "selective"))
    return summary


BASE_FEATURE_GROUPS = ["cluster_structure", "representation", "head_topk_probs", "agreement",
                       "soft_assignment", "assignment_ensemble", "neighborhood", "neighborhood_context"]
DEFAULT_ABLATION = [
    {"name": "label_only", "input_mode": "label_only", "groups": BASE_FEATURE_GROUPS},
    {"name": "head_outputs", "input_mode": "head_outputs", "groups": BASE_FEATURE_GROUPS},
    {"name": "head_outputs_mc_ood", "input_mode": "head_outputs",
     "groups": BASE_FEATURE_GROUPS + ["encoder_mc", "encoder_ood"]},
]
ABLATION_KEYS = ("rps", "rpss_vs_climatology", "nll", "brier", "classwise_ece", "weighted_f1", "qwk",
                 "accuracy", "within_one_bin")


def run_ablation(yml, use_key, out_root) -> list[dict]:
    """
    Same species, samples and splits; only uq.input_mode / features.groups
    change. Each variant is calibrated on its own calibration split, so the
    comparison is of calibrated test scores.
    """
    from sit_fuse.pipelines.habs.uq.hab_uq_train import train_species
    from sit_fuse.pipelines.habs.uq.hab_uq_evaluate import evaluate_bundle
    variants = yml["uq"].get("evaluation", {}).get("ablation") or DEFAULT_ABLATION
    spec = species_spec(use_key, yml["uq"].get("species", {}).get(use_key))
    rows = []
    for v in variants:
        y2 = copy.deepcopy(yml)
        y2["uq"]["input_mode"] = v["input_mode"]
        y2["uq"].setdefault("features", {})["groups"] = list(v.get("groups", BASE_FEATURE_GROUPS))
        d = Path(out_root) / "ablation" / v["name"]
        try:
            b = train_species(y2, use_key, d / "models")
            frame = b.pop("_frame", None)
            rep = evaluate_bundle(b, frame, d / "evaluation", n_boot=int(v.get("n_bootstrap", 100)))
        except Exception as e:  # a variant without inputs (e.g. no head files) is reported, not fatal
            LOG.warning("ablation %s/%s failed: %s", use_key, v["name"], e)
            rows.append({"variant": v["name"], "error": str(e)})
            continue
        ov = (rep.get("overall") or {}).get("uq", {})
        cal = [sk.get("calibration") for sk in b.get("skill", {}).values() if sk.get("calibration")]
        rows.append({"variant": v["name"], "model": b.get("model_name"),
                     "n_features": len(b["feature_columns"]),
                     "n_test_samples": (rep.get("overall") or {}).get("n_samples"),
                     **{k: ov.get(k) for k in ABLATION_KEYS if k in ov},
                     "cal_split_ece_raw": float(np.mean([c["raw"]["classwise_ece"] for c in cal])) if cal else None,
                     "cal_split_ece_calibrated": float(np.mean([c["calibrated"]["classwise_ece"] for c in cal]))
                     if cal else None})
    out = Path(out_root) / "ablation" / f"ablation_{spec.fname_token}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    LOG.info("ablation -> %s", out)
    return rows


def _has_parquet():
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--stages", default="train,evaluate,infer,fuse")
    a = ap.parse_args()
    from sit_fuse.utils import read_yaml
    run_hab_uq_pipeline(read_yaml(a.yaml), tuple(a.stages.split(",")))
