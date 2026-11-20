#!/usr/bin/env python3
"""
Build the pixel feature table and full-scene cluster counts used by
simple_classifiers_uq.py.

For each matchup (SIT-FUSE head-output raster + reference raster):
  1. Read head-output bands; align the reference to the head grid.
  2. Compute fold-independent features (head summaries, neighborhood stats).
  3. Record full-scene cluster x reference-class counts at parent and joint
     level (unsampled; used to fit agreement fractions per CV fold).
  4. Sample pixels per scene and class (classifier_utils
     sample_multiclass_by_group + cap_total_samples_stratified), recording
     population weights so calibration and evaluation use natural prevalence.

No ambiguity masking: every pixel with valid SIT-FUSE output and valid
reference coverage is eligible.

Outputs (output_dir):
  pixel_features.parquet   sampled rows (fold-independent features, ids, ref)
  cluster_counts.parquet   full-scene counts
  prevalence.json          full-scene class counts per date
  build_manifest.json

Usage:
    python build_uq_feature_table.py uq_config.yaml [-n THREADS] [--workers W]
    # as launcher jobs (sit_fuse.train.plan_uq_jobs): N shard jobs, then one merge job
    python build_uq_feature_table.py uq_config.yaml --shard i/N    # -> <output_dir>/feature_shards/
    python build_uq_feature_table.py uq_config.yaml --merge
Scenes are independent; per-scene sampling is seeded by the scene's position in the matchup list, so serial,
--workers and --shard/--merge runs write the same table.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.errors import RasterioError

from sit_fuse.train.classifier_utils import (
    cap_total_samples_stratified,
    load_yaml,
    log_class_balance,
    sample_multiclass_by_group,
)

from sf_uq_utils import (
    InvalidClusters,
    LabelLevels,
    excluded_cluster_mask,
    head_valid_mask,
    align_reference,
    cluster_class_counts,
    head_settings_from_tags,
    load_matchups,
    read_head_outputs,
    save_json,
    scene_feature_frame,
    write_table,
)

LOG = logging.getLogger("build-uq-feature-table")


def _class_counts(labels, C):
    labels = np.asarray(labels).astype(np.int64)
    labels = labels[(labels >= 0) & (labels < C)]
    return np.bincount(labels, minlength=C).tolist()


def _raw_value_summary(path, top=8):
    """dtype, nodata tag and the most frequent raw values of a reference raster."""
    from sit_fuse.train.classifier_utils import read_raster
    arr, prof = read_raster(Path(path))
    a = np.asarray(arr)
    fin = a[np.isfinite(a)] if np.issubdtype(a.dtype, np.floating) else a.ravel()
    vals, cnt = np.unique(fin, return_counts=True)
    order = np.argsort(cnt)[::-1][:top]
    return {"dtype": str(a.dtype), "shape": list(a.shape), "nodata_tag": prof.get("nodata"),
            "crs": str(prof.get("crs")), "n_nonfinite": int((~np.isfinite(a)).sum()) if np.issubdtype(a.dtype, np.floating) else 0,
            "top_values": {str(float(vals[k])): int(cnt[k]) for k in order}}


def report_missing_classes(stages, matchups, target_cfg, C, missing, out_dir):
    """Log where the missing reference classes disappear and write a diagnostic JSON."""
    tot = {k: np.sum([v[k] for v in stages.values()], axis=0).tolist()
           for k in ("reference_on_head_grid", "and_head_valid", "and_not_excluded")}
    LOG.error("Reference class counts summed over scenes (classes 0..%d):", C - 1)
    LOG.error("  on head grid after nodata/value_map : %s", tot["reference_on_head_grid"])
    LOG.error("  and valid head pixels               : %s", tot["and_head_valid"])
    LOG.error("  and outside excluded clusters       : %s", tot["and_not_excluded"])
    LOG.error("  target.raster_nodata=%s (auto ignores a raster NoData tag that equals a label)",
              target_cfg.get("raster_nodata", "auto"))
    seen, raw = set(), {}
    for m in matchups:
        t = str(m["truth"])
        if t in seen:
            continue
        seen.add(t)
        try:
            raw[t] = _raw_value_summary(t)
        except Exception as exc:
            raw[t] = {"error": str(exc)}
        if len(seen) >= 3:
            break
    for t, r in raw.items():
        LOG.error("  raw reference %s: %s", Path(t).name, r)
    c = missing[0]
    if tot["reference_on_head_grid"][c] == 0:
        LOG.error(
            "Class %d is absent right after reading/collocating the reference. Compare the raw values "
            "above with target.value_map=%s, target.unmapped=%s, target.nodata_values=%s, "
            "target.annotation_nodata=%s. If the masks store background as NoData (only the target "
            "value is burned in), there is no background to learn from: re-rasterize with fill 0 or "
            "map the fill value to class 0 in value_map and set annotation_nodata to a value not in the data.",
            c, target_cfg.get("value_map"), target_cfg.get("unmapped"), target_cfg.get("nodata_values"),
            target_cfg.get("annotation_nodata"))
    elif tot["and_head_valid"][c] == 0:
        LOG.error("Class %d exists on the head grid but never on valid head pixels: the head outputs and "
                  "reference do not overlap where the class occurs (check georeferencing / head export).", c)
    else:
        LOG.error("Class %d is removed by invalid_clusters (UQ config).", c)
    save_json(out_dir / "missing_class_diagnostic.json",
              {"missing_classes": missing, "totals": tot, "raw_reference_examples": raw,
               "target": target_cfg})


_CTX = {}


def _context(cfg: dict) -> dict:
    """Per-process settings derived from the config (rebuilt once in each worker)."""
    import json
    key = json.dumps(cfg, sort_keys=True, default=str)
    if key not in _CTX:
        target_cfg = cfg["target"]
        _CTX.clear()
        _CTX[key] = {
            "target_cfg": target_cfg, "features_cfg": cfg.get("features", {}),
            "C": len(target_cfg["classes"]), "rs": int(cfg.get("random_state", 42)),
            "caps": cfg.get("sampling", {}).get("max_samples_per_class_per_scene"),
            "invalid": InvalidClusters.from_config(cfg), "levels": LabelLevels(cfg),
        }
    return _CTX[key]


def select_matchups(cfg: dict) -> list:
    data_cfg = cfg["data"]
    product = str(data_cfg.get("product", "")).lower()
    splits = data_cfg.get("include_splits")  # e.g. [dev, calibration, holdout]
    # data.every_minutes (e.g. 60): quick first pass on the first scene of each UTC hour (see load_matchups)
    matchups = [m for m in load_matchups(data_cfg["matchup_file"], require_truth=True, splits=splits,
                                         every_minutes=data_cfg.get("every_minutes"),
                                         offset_minutes=data_cfg.get("offset_minutes", 0),
                                         missing_heads=data_cfg.get("missing_heads", "skip"))
                if not product or m["product"] == product]
    if not matchups:
        raise ValueError(f"No matchups for product='{product}'.")
    return matchups


def process_scene(cfg: dict, i: int, m: dict) -> dict:
    """
    Features, full-scene counts and per-scene sample for matchup number i (1-based position in the full
    matchup list; it seeds the per-scene sampling, so serial, worker and sharded runs give the same rows).
    """
    x = _context(cfg)
    C, invalid, levels = x["C"], x["invalid"], x["levels"]
    LOG.info("[%d] %s (%s)", i, m["pair_id"], m["date"])
    try:
        bands, profile, tags = read_head_outputs(m["head_outputs"])
    except (OSError, RasterioError) as e:    # missing (removed after listing) or unreadable / corrupt head output
        LOG.warning("%s: head_outputs unreadable (%s); scene skipped.", m["pair_id"], e)
        return {"i": i, "pair_id": m["pair_id"], "date": m["date"], "head_settings": None,
                "head_outputs": str(m["head_outputs"]), "frame": None, "counts": None, "population": None,
                "level_missing": [], "excluded": None, "stages": None, "skipped": f"head_outputs unreadable: {e}"}
    res = {"i": i, "pair_id": m["pair_id"], "date": m["date"], "head_settings": head_settings_from_tags(tags),
           "head_outputs": str(m["head_outputs"]), "frame": None, "counts": None, "population": None,
           "level_missing": [], "excluded": None}
    ref = align_reference(m["truth"], profile, x["target_cfg"])
    meta = {k: m[k] for k in ("pair_id", "date", "product", "split_group",
                              "classifier_split", "encoder_split", "split_stratum")}
    df, valid = scene_feature_frame(bands, profile, x["features_cfg"], m["pair_id"], ref=ref,
                                    invalid=invalid)
    if levels and not df.empty:
        cons, miss = levels.consensus_label(m, profile)
        res["level_missing"] = list(miss)
        r_, c_ = df["row"].to_numpy(), df["col"].to_numpy()
        df["cons_ref_label"] = cons[r_, c_].astype(np.int16)
        df["label_level"] = levels.levels(df["ref_label"].to_numpy(), df["cons_ref_label"].to_numpy())
    hv = head_valid_mask(bands)
    res["stages"] = {
        "truth": str(m["truth"]),
        "reference_on_head_grid": _class_counts(ref[ref >= 0], C),
        "reference_nodata_on_head_grid": int((ref < 0).sum()),
        "and_head_valid": _class_counts(ref[hv & (ref >= 0)], C),
        "and_not_excluded": _class_counts(df["ref_label"].to_numpy(), C) if not df.empty else [0] * C,
    }
    if invalid:
        ex = excluded_cluster_mask(bands, invalid)
        head_ok, ref_ok = np.isfinite(ex), ref >= 0
        tgt = ref_ok & (ref > 0) & head_ok
        res["excluded"] = {
            "excluded_fraction": float(np.nanmean(ex)) if head_ok.any() else None,
            "excluded_reference_target_fraction": float(ex[tgt].mean()) if tgt.any() else None,
        }
    if df.empty:
        LOG.warning("%s: no valid pixels outside excluded clusters in the matched area; scene skipped.",
                    m["pair_id"])
        return res
    res["counts"] = cluster_class_counts(bands, ref, valid, C, meta)
    pop = df["ref_label"].value_counts().reindex(range(C), fill_value=0)
    res["population"] = [int(pop[c]) for c in range(C)]
    if x["caps"]:
        df = sample_multiclass_by_group(
            df, label_column="ref_label", group_column="pair_id",
            max_samples_per_class=x["caps"], random_state=x["rs"] + i)
    for k, v in meta.items():
        df[k] = v
    res["frame"] = df
    return res


def collect_scenes(cfg: dict, matchups: list, indices=None, n_workers=1, threads=1) -> list:
    """process_scene over matchups[indices] (all by default), serially or in worker processes."""
    from sit_fuse.parallel_utils import cpu_map
    idx = range(len(matchups)) if indices is None else indices
    return cpu_map(process_scene, [(cfg, k + 1, matchups[k]) for k in idx], n_workers, threads)


def run(cfg: dict, n_workers: int = 1, threads: int = 1) -> None:
    matchups = select_matchups(cfg)
    finalize(cfg, matchups, collect_scenes(cfg, matchups, n_workers=n_workers, threads=threads))


def finalize(cfg: dict, matchups: list, results: list) -> None:
    """Combine per-scene results (in matchup order) into the feature table, counts and manifest."""
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    target_cfg = cfg["target"]
    features_cfg = cfg.get("features", {})
    samp_cfg = cfg.get("sampling", {})
    C = len(target_cfg["classes"])
    rs = int(cfg.get("random_state", 42))
    invalid = InvalidClusters.from_config(cfg)
    if invalid:
        LOG.info("Excluding %d joint and %d parent clusters (%s)", len(invalid.joint),
                 len(invalid.parent), invalid.source or "inline")
    levels = LabelLevels(cfg)
    if levels:
        LOG.info("Label levels: consensus references %s (require_all_observed=%s, consensus_positives=%s)",
                 levels.names, levels.require_all_observed, levels.consensus_positives)

    results = sorted(results, key=lambda r: r["i"])
    if [r["i"] for r in results] != list(range(1, len(matchups) + 1)):
        raise ValueError(f"Scene results cover {len(results)} of {len(matchups)} matchups; the matchup file or "
                         "data.include_splits changed since the shards were written. Rerun the shards.")
    frames, counts, prevalence, population = [], [], {}, {}
    stages, excluded, level_missing = {}, {}, {}
    head_settings = None
    skipped = {r["pair_id"]: r["skipped"] for r in results if r.get("skipped")}
    if skipped:
        LOG.warning("%d scenes skipped (unreadable head outputs); listed in build_manifest.json", len(skipped))
    for r in results:
        if r.get("skipped"):
            continue
        hs = r["head_settings"]
        if head_settings is None:
            head_settings = hs
        elif hs != head_settings:
            raise ValueError(
                f"{r['head_outputs']}: head export settings {hs} differ from "
                f"earlier scenes {head_settings}. Re-export with one setting.")
        stages[r["pair_id"]] = r["stages"]
        if r["excluded"] is not None:
            excluded[r["pair_id"]] = r["excluded"]
        for n in r["level_missing"]:
            level_missing[n] = level_missing.get(n, 0) + 1
        if r["frame"] is None:
            continue
        counts.append(r["counts"])
        d = prevalence.setdefault(r["date"], {str(c): 0 for c in range(C)})
        for c in range(C):
            d[str(c)] += r["population"][c]
            population[(r["pair_id"], c)] = r["population"][c]
        frames.append(r["frame"])

    if not frames:
        raise ValueError("No scene has valid pixels outside excluded clusters; check invalid_clusters "
                         "(joint values vs joint_labels, geotiff_scale, parent IDs).")
    table = pd.concat(frames, ignore_index=True)
    present = set(table["ref_label"].astype(int).unique())
    missing = [c for c in range(C) if c not in present]
    save_json(out_dir / "reference_stage_counts.json", stages)
    if missing:
        tot = {int(k): int(v) for k, v in table["ref_label"].value_counts().items()}
        report_missing_classes(stages, matchups, target_cfg, C, missing, out_dir)
        raise ValueError(
            f"Reference classes {missing} have no valid pixels in any scene (class counts: {tot}). "
            "Classifiers need every class. A raster NoData tag equal to a label is ignored by default "
            "(target.raster_nodata: auto), so check target.raster_nodata is not 'use' and whether the "
            "masks store background as NoData (see the per-stage counts and raw values logged above). "
            "Also check target.value_map with target.unmapped: nodata, and that invalid_clusters does "
            "not remove the class. invalid_clusters / cluster_nodata_values at the matchup step do not "
            "change reference labels.")
    table = cap_total_samples_stratified(
        table, samp_cfg.get("max_samples"), label_column="ref_label",
        group_column="pair_id", random_state=rs)

    # population weights: full-scene count / retained count per (scene, class)
    kept = table.groupby(["pair_id", "ref_label"]).size()
    keys = list(zip(table["pair_id"], table["ref_label"].astype(int)))
    table["pop_weight"] = np.array(
        [population[k] / kept[k] for k in keys], dtype=np.float32)
    log_class_balance(table, "ref_label", "Sampled")
    level_summary = None
    if levels:
        if level_missing:
            LOG.warning("Consensus references missing from matchup records (scenes): %s; those pixels are "
                        "primary-level only. Run attach_references.py for them.", level_missing)
        neg = table["ref_label"] == levels.negative_class
        w_ = table["pop_weight"]
        cons_neg = neg & (table["label_level"] == 2)
        level_summary = {
            **levels.describe(),
            "missing_reference_scenes": level_missing,
            "primary_negative_pixels": float(w_[neg].sum()),
            "consensus_negative_pixels": float(w_[cons_neg].sum()),
            "consensus_negative_fraction": float(w_[cons_neg].sum() / max(w_[neg].sum(), 1e-12)),
            "primary_negative_consensus_positive_pixels": float(
                w_[neg & (table["cons_ref_label"] >= 0) & (table["cons_ref_label"] != levels.negative_class)].sum()),
            "primary_positive_consensus_negative_pixels": float(
                w_[~neg & (table["cons_ref_label"] == levels.negative_class)].sum()),
            "consensus_unobserved_pixels": float(w_[table["cons_ref_label"] < 0].sum()),
        }
        LOG.info("Negatives (population-weighted): primary %.0f, consensus %.0f (%.1f%%); "
                 "primary-negative/consensus-positive conflicts %.0f",
                 level_summary["primary_negative_pixels"], level_summary["consensus_negative_pixels"],
                 100 * level_summary["consensus_negative_fraction"],
                 level_summary["primary_negative_consensus_positive_pixels"])

    p1 = write_table(table, out_dir / "pixel_features.parquet")
    p2 = write_table(pd.concat(counts, ignore_index=True), out_dir / "cluster_counts.parquet")
    save_json(out_dir / "prevalence.json", prevalence)
    save_json(out_dir / "build_manifest.json", {
        "pixel_features": str(p1),
        "cluster_counts": str(p2),
        "n_rows": int(len(table)),
        "skipped_scenes": skipped,              # pair_id -> reason (unreadable head outputs)
        "n_scenes": len(matchups),
        "dates": sorted(prevalence),
        "classifier_splits": {s: int(n) for s, n in
                              pd.Series([m["classifier_split"] or "unassigned" for m in matchups]).value_counts().items()},
        "head_settings": head_settings,
        "sampled_class_counts": {str(k): int(v) for k, v in
                                 table["ref_label"].value_counts().sort_index().items()},
        "features": features_cfg,
        "invalid_clusters": {**invalid.signature(), "source": invalid.source} if invalid else None,
        "excluded_by_scene": excluded,
        "label_levels": level_summary,
        "target": target_cfg,
    })
    LOG.info("Wrote %d rows from %d scenes to %s", len(table), len(frames), out_dir)


def main():
    from sit_fuse.parallel_utils import (add_parallel_args, atomic_write_bytes, infer_n_shards, parse_shard,
                                         setup_compute, shard_files, shard_indices, shard_path)
    import pickle
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_config", type=Path)
    add_parallel_args(ap)
    a = ap.parse_args()
    cfg = load_yaml(a.yaml_config)
    n_workers, threads = setup_compute(cfg, a.num_threads, a.workers)
    shard = parse_shard(a.shard)
    sdir = Path(cfg["output_dir"]) / "feature_shards"
    if shard is not None:            # one launcher job: scenes i, i+N, ... -> partial results
        matchups = select_matchups(cfg)
        idx = shard_indices(len(matchups), shard, "stride")
        res = collect_scenes(cfg, matchups, idx, n_workers, threads)
        p = shard_path(sdir, "scenes", shard)
        atomic_write_bytes(p, pickle.dumps({"n_matchups": len(matchups), "results": res},
                                           protocol=pickle.HIGHEST_PROTOCOL))
        LOG.info("Shard %d/%d: %d scenes -> %s", shard[0], shard[1], len(res), p)
    elif a.merge:                    # combine all shards; identical to a serial run
        matchups = select_matchups(cfg)
        n = a.n_shards or infer_n_shards(sdir, "scenes")
        if not n:
            raise FileNotFoundError(f"No feature shards in {sdir}; run with --shard i/N first.")
        res = []
        for p in shard_files(sdir, "scenes", n):
            res += pickle.loads(p.read_bytes())["results"]
        finalize(cfg, matchups, res)
    else:
        run(cfg, n_workers, threads)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
