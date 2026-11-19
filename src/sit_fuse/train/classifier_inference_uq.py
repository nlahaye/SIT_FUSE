#!/usr/bin/env python3
"""
Scene inference: discrete, probabilistic and uncertainty products.

For each head-output raster (export_head_outputs.py):
  1. Rebuild fold-independent features (same settings as training).
  2. Apply the saved ContextEncoder and enforce the trained feature order.
  3. Predict, apply the saved prior correction + calibrator, threshold.
  4. Write one GeoTIFF per layer (classifier_utils.write_raster_stack,
     <model>_<pair_id>_<layer>.tif), each tagged with product type,
     description, units, valid range and provenance (optional COG):
       discrete        pred_label, pred_label_selective (-1 = flagged),
                       selective_flag (1 retained / 0 flagged)
       probabilistic   p_<c> (calibrated), confidence (p of predicted class),
                       ensemble_mean_p<c>, ensemble_p<c>_q<qq> (member quantiles)
       uncertainty     predictive_entropy, margin_uncertainty,
                       ensemble_total_entropy, ensemble_expected_entropy (aleatoric),
                       ensemble_mutual_information, ensemble_variance (epistemic),
                       assignment_vote_variance (structural),
                       parent_entropy, child_entropy (representation)
       baselines       soft_assignment_p<c>, assignment_vote_p<c>
       diagnostics     parent_maxprob, child_maxprob
  5. Scene area estimates per target class (km^2 if the grid is
     georeferenced, else pixels): discrete, expected, analytic 90% interval,
     ensemble 90% interval; over all valid pixels and, when a reference is
     attached, over reference-valid pixels (used for evaluation).
     -> scene_areas.csv and inference_manifest.json

Usage:
    python classifier_inference_uq.py uq_config.yaml [--model random_forest] [-n THREADS] [--workers W]
    # as launcher jobs (sit_fuse.train.plan_uq_jobs): N shard jobs per model, then one merge job
    python classifier_inference_uq.py uq_config.yaml --model M --shard i/N   # layers + <out>/shards/
    python classifier_inference_uq.py uq_config.yaml --model M --merge       # manifest + scene_areas.csv

Resume: each finished scene is checkpointed to <out>/scene_results/<pair_id>.pkl (entry + area rows). A rerun
(serial, workers or any shard layout) reuses a scene when its checkpoint signature (inference settings, training
summary, target, features, excluded clusters) matches and all its layers exist; everything else is recomputed.
Disable with inference.resume: false or --overwrite.
Scenes are independent and written to separate files, so all modes give the same products.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sit_fuse.train.classifier_utils import (
    write_single_band_raster,
    compare_feature_signatures,
    enforce_feature_order,
    load_json,
    load_yaml,
    write_raster_stack,
)

from sf_uq_utils import (
    ABSTAIN,
    InvalidClusters,
    excluded_cluster_mask,
    OUT_NODATA,
    AreaAccumulator,
    align_reference,
    pixel_area_km2,
    tag_product_layers,
    bootstrap_member_probabilities,
    decide,
    ensemble_decomposition,
    forest_member_probabilities,
    full_proba,
    head_settings_from_tags,
    layer_info,
    load_matchups,
    margin_uncertainty,
    predictive_entropy_norm,
    read_head_outputs,
    save_json,
    scene_feature_frame,
    scene_threshold,
    uq_feature_signature,
)
from simple_classifiers_uq import BASELINES, baseline_proba

LOG = logging.getLogger("classifier-inference-uq")


def predict_chunk(frame, name, model, enc, cal, feats, C, members_src):
    if name in BASELINES:
        P = baseline_proba(name, frame, enc)
    else:
        P = full_proba(model, enforce_feature_order(frame, feats), C)
    Pc = cal.transform(P)
    members = None
    if members_src == "forest":
        members = cal.transform_members(forest_member_probabilities(model, frame[feats], C))
    elif isinstance(members_src, tuple):
        base_path, paths = members_src
        members = cal.transform_members(bootstrap_member_probabilities(base_path, frame[feats], paths, C))
    return Pc, members


def load_context(cfg, model_name=None):
    """Model, encoder, calibrator, thresholds and settings shared by every scene of one model."""
    out = Path(cfg["output_dir"])
    inf = cfg.get("inference", {})
    summary_path = inf.get("training_summary_path", out / "training" / "training_summary.json")
    summary = load_json(summary_path)
    meta = summary["metadata"]

    diffs = compare_feature_signatures(meta["feature_signature"], uq_feature_signature(cfg))
    if diffs:
        raise ValueError("Feature signature mismatch:\n  " + "\n  ".join(diffs))

    name = model_name or inf.get("model", meta["models"][0] if meta["models"] else meta["baselines"][0])
    C = len(meta["classes"])
    feats = meta["feature_columns"]
    enc = joblib.load(meta["context_encoder_path"])
    cal = joblib.load(meta["calibrators_path"])[name]
    thr = meta["thresholds"].get(name)
    # calibration.threshold_groups: per-group (event / season) thresholds
    thr_groups_cfg = meta.get("threshold_groups") or None
    thr_by_group = (meta.get("thresholds_by_group") or {}).get(name) or {}
    if thr_by_group:
        LOG.info("%s: group thresholds %s (pooled %.4g for other groups)", name,
                 {g: round(t, 5) for g, t in thr_by_group.items()}, thr)
    model = None if name in BASELINES else joblib.load(meta["model_paths"][name])
    sel = (meta.get("selective") or {}).get(name)
    sel_rule = sel["rules"][sel["measure"]] if sel else None
    quantiles = [float(q) for q in inf.get("ensemble_quantiles", [0.05, 0.95])]
    target_classes = list(range(1, C)) if C == 2 else list(range(C))

    members_src = None
    if name == "random_forest":
        members_src = "forest"
    elif meta.get("bootstrap_manifest", {}).get(name):
        members_src = (meta["model_paths"][name], meta["bootstrap_manifest"][name])

    out_dir = Path(inf.get("output_dir", out / "inference" / name))
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk = int(inf.get("chunk_size", 200000))
    features_cfg = cfg.get("features", {})
    scene_file = inf.get("scene_file", cfg["data"]["matchup_file"])
    # inference.every_minutes (default data.every_minutes; 0 = every scene): first scene of each UTC bin
    every = inf.get("every_minutes", cfg["data"].get("every_minutes"))
    off = inf.get("offset_minutes", cfg["data"].get("offset_minutes", 0))
    records = load_matchups(scene_file, require_truth=False, splits=inf.get("splits"),
                            every_minutes=every, offset_minutes=off)
    invalid = InvalidClusters.from_config(cfg)
    provenance_common = {
        "model": name, "classes": meta["classes"], "threshold": thr,
        "calibrator": cal.describe(), "selective_rule": sel or {},
        "training_summary": str(summary_path), "scene_file": str(scene_file),
        "every_minutes": every or None,
        "feature_signature_sha": _sha(meta["feature_signature"]),
        "head_settings": meta["head_settings"],
        "invalid_clusters": invalid.signature() if invalid else {},
    }
    return dict(cfg=cfg, name=name, C=C, feats=feats, enc=enc, cal=cal, thr=thr, thr_groups_cfg=thr_groups_cfg,
                thr_by_group=thr_by_group, model=model, sel=sel, sel_rule=sel_rule, quantiles=quantiles,
                target_classes=target_classes, members_src=members_src, out_dir=out_dir, chunk=chunk,
                features_cfg=features_cfg, records=records, invalid=invalid, provenance_common=provenance_common,
                meta=meta, inf=inf)


_CTX = {}


def _context(cfg, model_name, threads=None, x=None):
    """Cached load_context; inside a worker, the model's own n_jobs is capped at the worker's thread share."""
    import json
    key = (json.dumps(cfg, sort_keys=True, default=str), model_name)
    if key not in _CTX:
        _CTX.clear()
        x = x or load_context(cfg, model_name)
        m = x["model"]
        if threads and m is not None and "clf__n_jobs" in m.get_params():
            m.set_params(clf__n_jobs=threads)
        # signature of everything that determines a scene's products (per-scene checkpoints are reused only
        # when it matches): inference settings, trained model summary, target and feature definitions
        inf_sig = {k: v for k, v in (x["inf"] or {}).items() if k not in _NON_PRODUCT_KEYS}
        x["sig"] = _sha({"name": x["name"], "inf": inf_sig, "meta": x["meta"], "target": cfg.get("target"),
                         "features": x["features_cfg"], "invalid": x["invalid"].signature() if x["invalid"] else None})
        _CTX[key] = x
    return _CTX[key]


_NON_PRODUCT_KEYS = ("resume", "n_jobs", "num_threads", "num_workers", "chunk_size")


def _checkpoint(x, rec):
    """Per-scene result file: <inference out_dir>/scene_results/<pair_id>.pkl."""
    return x["out_dir"] / "scene_results" / f"{rec['pair_id']}.pkl"


def _load_checkpoint(x, rec):
    """Earlier result of this scene when its signature matches and all of its layers still exist, else None."""
    import pickle
    p = _checkpoint(x, rec)
    if not p.exists():
        return None
    try:
        d = pickle.loads(p.read_bytes())
    except Exception:  # noqa: BLE001  (truncated / foreign file -> recompute)
        return None
    if d.get("sig") != _sha([x["sig"], rec]):
        return None
    entry = d["result"][0]
    if not all(Path(f).exists() for f in (entry.get("layers") or {}).values()):
        return None
    return d["result"]


def _save_checkpoint(x, rec, result):
    import pickle
    from sit_fuse.parallel_utils import atomic_write_bytes
    atomic_write_bytes(_checkpoint(x, rec), pickle.dumps({"sig": _sha([x["sig"], rec]), "result": result}))


def infer_scene(cfg, model_name, k, threads=None):
    """
    Products for scene k of the scene file (worker-safe: the context is loaded once per process).
    Returns (manifest entry, area rows, value columns).
    """
    x = _context(cfg, model_name, threads)
    (name, C, feats, enc, cal, thr, thr_groups_cfg, thr_by_group, model, sel, sel_rule, quantiles,
     target_classes, members_src, out_dir, chunk, features_cfg, invalid, provenance_common, meta, inf) = (
        x[v] for v in ("name", "C", "feats", "enc", "cal", "thr", "thr_groups_cfg", "thr_by_group", "model",
                       "sel", "sel_rule", "quantiles", "target_classes", "members_src", "out_dir", "chunk",
                       "features_cfg", "invalid", "provenance_common", "meta", "inf"))
    rec = x["records"][k]
    resume = bool(inf.get("resume", True))
    if resume:
        done = _load_checkpoint(x, rec)
        if done is not None:
            LOG.info("%s: reusing finished scene (scene_results checkpoint)", rec["pair_id"])
            return done
    result = _infer_scene(x, rec)
    if resume:
        _save_checkpoint(x, rec, result)
    return result


def _infer_scene(x, rec):
    (name, C, feats, enc, cal, thr, thr_groups_cfg, thr_by_group, model, sel, sel_rule, quantiles,
     target_classes, members_src, out_dir, chunk, features_cfg, invalid, provenance_common, meta, inf) = (
        x[v] for v in ("name", "C", "feats", "enc", "cal", "thr", "thr_groups_cfg", "thr_by_group", "model",
                       "sel", "sel_rule", "quantiles", "target_classes", "members_src", "out_dir", "chunk",
                       "features_cfg", "invalid", "provenance_common", "meta", "inf"))
    cfg = x["cfg"]
    area_rows = []
    bands, profile, tags = read_head_outputs(rec["head_outputs"])
    if head_settings_from_tags(tags) != meta["head_settings"]:
        raise ValueError(f"{rec['head_outputs']}: head export settings differ from training.")
    base, _ = scene_feature_frame(bands, profile, features_cfg, rec["pair_id"], invalid=invalid)
    base["date"] = rec["date"]
    if base.empty:
        LOG.warning("%s: all head-valid pixels are in excluded clusters; no products written.",
                    rec["pair_id"])
        return ({"pair_id": rec["pair_id"], "date": rec["date"], "n_valid": 0,
                 "skipped": "all pixels in excluded clusters",
                 "classifier_split": rec["classifier_split"], "head_outputs": rec["head_outputs"],
                 "truth": rec["truth"]}, [], [])

    thr_s, thr_key = scene_threshold(rec, thr, thr_by_group, thr_groups_cfg)
    area_grid = pixel_area_km2(profile)
    area_units = "km2" if area_grid is not None else "pixels"
    ref = None
    if rec.get("truth") and inf.get("reference_mask_areas", True):
        try:
            ref = align_reference(rec["truth"], profile, cfg["target"])
        except Exception as exc:  # reference only defines the evaluation mask
            LOG.warning("%s: reference not aligned for area mask (%s)", rec["pair_id"], exc)
    acc_all, acc_ref = AreaAccumulator(C), AreaAccumulator(C)

    parts = []
    for s0 in range(0, len(base), chunk):
        fr = enc.transform(base.iloc[s0:s0 + chunk], features_cfg)
        Pc, members = predict_chunk(fr, name, model, enc, cal, feats, C, members_src)
        pred = decide(Pc, thr_s)
        rows, cols = fr["row"].to_numpy(), fr["col"].to_numpy()
        o = {"pair_id": fr["pair_id"].to_numpy(), "row": rows, "col": cols,
             "pred_label": pred.astype(np.float32)}
        for c in range(C):
            o[f"p_{c}"] = Pc[:, c]
        o["confidence"] = Pc[np.arange(len(pred)), pred]
        o["predictive_entropy"] = predictive_entropy_norm(Pc)
        o["margin_uncertainty"] = margin_uncertainty(Pc)
        if members is not None:
            mean, tot, exp_, mi, var = ensemble_decomposition(members)
            for c in range(C):
                o[f"ensemble_mean_p{c}"] = mean[:, c]
            for c in target_classes:
                qs = np.quantile(members[:, :, c], quantiles, axis=0)
                for q, v in zip(quantiles, qs):
                    o[f"ensemble_p{c}_q{int(round(q * 100)):02d}"] = v
            o["ensemble_total_entropy"] = tot
            o["ensemble_expected_entropy"] = exp_
            o["ensemble_mutual_information"] = mi
            o["ensemble_variance"] = var
        sa = enc.soft_matrix(fr, "sa_joint")
        vt = enc.soft_matrix(fr, "assign_vote_soft")
        for c in range(C):
            o[f"soft_assignment_p{c}"] = sa[:, c]
            o[f"assignment_vote_p{c}"] = vt[:, c]
        o["assignment_vote_variance"] = fr["assign_vote_var"].to_numpy()
        for k in ("parent_entropy", "child_entropy", "parent_maxprob", "child_maxprob"):
            o[k] = fr[k].to_numpy()
        if sel_rule is not None:
            u = np.asarray(o.get(sel["measure"], o["predictive_entropy"]), dtype=float)
            keep = u <= sel_rule["threshold"]
            o["selective_flag"] = keep.astype(np.float32)
            o["pred_label_selective"] = np.where(keep, pred, ABSTAIN).astype(np.float32)

        a = area_grid[rows, cols] if area_grid is not None else np.ones(len(pred))
        acc_all.add(Pc, pred, a, members)
        if ref is not None:
            m = ref[rows, cols] >= 0
            acc_ref.add(Pc[m], pred[m], a[m], None if members is None else members[:, m])
        parts.append(pd.DataFrame(o))

    pred_df = pd.concat(parts, ignore_index=True)
    value_cols = [c for c in pred_df.columns if c not in ("pair_id", "row", "col")]
    pair_meta = {rec["pair_id"]: {"height": profile["height"], "width": profile["width"],
                                  "transform": profile["transform"], "crs": profile["crs"]}}
    # classifier_utils.write_raster_stack: one GeoTIFF per output layer
    write_raster_stack(pred_df, value_cols, pair_meta, out_dir, prefix=name, fill_value=OUT_NODATA)
    layer_paths = {c: out_dir / f"{name}_{rec['pair_id']}_{c}.tif" for c in value_cols}
    # excluded-cluster mask over the full head grid (1 excluded, 0 kept, nodata = no head data)
    ex = excluded_cluster_mask(bands, invalid)
    ex_path = out_dir / f"{name}_{rec['pair_id']}_excluded_cluster_mask.tif"
    write_single_band_raster(ex_path, np.where(np.isfinite(ex), ex, OUT_NODATA), pair_meta[rec["pair_id"]],
                             nodata=OUT_NODATA)
    layer_paths["excluded_cluster_mask"] = ex_path
    head_ok = np.isfinite(ex)
    excl_frac = float(np.nanmean(ex)) if head_ok.any() else None
    tag_product_layers(layer_paths, {**provenance_common, "threshold": thr_s,
                                     "threshold_group": thr_key, "pair_id": rec["pair_id"],
                                     "date": rec["date"], "head_outputs": rec["head_outputs"],
                                     "classifier_split": rec["classifier_split"],
                                     "encoder_split": rec["encoder_split"]},
                       cog=bool(inf.get("cog", False)))

    areas = {"units": area_units, "all_valid": acc_all.summary(quantiles=(quantiles[0], quantiles[-1]))}
    if ref is not None:
        areas["reference_valid"] = acc_ref.summary(quantiles=(quantiles[0], quantiles[-1]))
    for scope in ("all_valid", "reference_valid"):
        for key, d in areas.get(scope, {}).items():
            if key.startswith("class_"):
                area_rows.append({"pair_id": rec["pair_id"], "date": rec["date"], "scope": scope,
                                  "class": int(key.split("_")[1]), "units": area_units, **d})
    entry = ({"pair_id": rec["pair_id"], "date": rec["date"], "n_valid": int(len(pred_df)),
                     "head_outputs": rec["head_outputs"], "truth": rec["truth"],
                     "additional_references": rec.get("additional_references", {}),
                     **{k: rec[k] for k in ("product", "split_group", "classifier_split",
                                            "encoder_split", "split_stratum")},
                     "areas": areas, "excluded_fraction": excl_frac,
                     "threshold": thr_s, "threshold_group": thr_key,
                     "layers": {c: str(p) for c, p in layer_paths.items()}})
    LOG.info("Wrote %d layers for %s", len(value_cols), rec["pair_id"])
    return entry, area_rows, value_cols


def run(cfg, model_name=None, n_workers=1, threads=1, indices=None, x=None):
    """All scenes (or scenes `indices`), serially or in worker processes; returns the per-scene results."""
    from sit_fuse.parallel_utils import cpu_map
    x = _context(cfg, model_name, x=x)            # this process (serial runs) reuses the loaded context
    idx = range(len(x["records"])) if indices is None else indices
    par = threads if n_workers > 1 else None
    res = cpu_map(infer_scene, [(cfg, model_name, k, par) for k in idx], n_workers, threads)
    return x, list(zip(idx, res))


def finalize(x, results):
    """Write scene_areas.csv and inference_manifest.json from (scene index, result) pairs."""
    (name, C, cal, thr, thr_groups_cfg, thr_by_group, sel, quantiles, out_dir, invalid) = (
        x[v] for v in ("name", "C", "cal", "thr", "thr_groups_cfg", "thr_by_group", "sel", "quantiles",
                       "out_dir", "invalid"))
    results = sorted(results, key=lambda r: r[0])
    if [k for k, _ in results] != list(range(len(x["records"]))):
        raise ValueError(f"Inference results cover {len(results)} of {len(x['records'])} scenes; the scene file "
                         "or inference.splits changed since the shards were written. Rerun the shards.")
    manifest = [r[1][0] for r in results]
    area_rows = [row for r in results for row in r[1][1]]
    value_cols = next((r[1][2] for r in reversed(results) if r[1][2]), [])
    pd.DataFrame(area_rows).to_csv(out_dir / "scene_areas.csv", index=False)
    save_json(out_dir / "inference_manifest.json", {
        "model": name, "n_classes": C, "layer_prefix": name, "threshold": thr,
        "threshold_groups": thr_groups_cfg, "thresholds_by_group": thr_by_group,
        "selective": sel, "ensemble_quantiles": quantiles, "calibrator": cal.describe(),
        "layer_info": {c: layer_info(c) for c in value_cols + ["excluded_cluster_mask"]} if manifest else {},
        "invalid_clusters": {**invalid.signature(), "source": invalid.source} if invalid else None,
        "scenes": manifest})




def _sha(obj) -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def main():
    import pickle
    from sit_fuse.parallel_utils import (add_parallel_args, atomic_write_bytes, infer_n_shards, parse_shard,
                                         setup_compute, shard_files, shard_indices, shard_path)
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_config", type=Path)
    ap.add_argument("--model", default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute every scene (ignore scene_results checkpoints; same as inference.resume: false)")
    add_parallel_args(ap)
    a = ap.parse_args()
    cfg = load_yaml(a.yaml_config)
    if a.overwrite:
        cfg.setdefault("inference", {})["resume"] = False
    n_workers, threads = setup_compute(cfg, a.num_threads, a.workers)
    shard = parse_shard(a.shard)
    if a.merge:
        x = load_context(cfg, a.model)
        sdir = x["out_dir"] / "shards"
        n = a.n_shards or infer_n_shards(sdir, "inference")
        if not n:
            raise FileNotFoundError(f"No inference shards in {sdir}; run with --shard i/N first.")
        res = []
        for p in shard_files(sdir, "inference", n):
            res += pickle.loads(p.read_bytes())["results"]
        finalize(x, res)
        return
    if shard is None:
        finalize(*run(cfg, a.model, n_workers, threads))
        return
    # one launcher job: scenes i, i+N, ... -> layers + a partial manifest for --merge
    x = _context(cfg, a.model)
    x, res = run(cfg, a.model, n_workers, threads, shard_indices(len(x["records"]), shard, "stride"), x)
    p = shard_path(x["out_dir"] / "shards", "inference", shard)
    atomic_write_bytes(p, pickle.dumps({"n_scenes": len(x["records"]), "results": res}))
    LOG.info("Shard %d/%d: %d scenes -> %s", shard[0], shard[1], len(res), p)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
