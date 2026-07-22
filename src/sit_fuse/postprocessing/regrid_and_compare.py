#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
from osgeo import gdal
from pyresample import kd_tree
from pyresample.utils.rasterio import get_area_def_from_raster
from skimage.metrics import structural_similarity

from sit_fuse.utils import read_yaml


def is_finite_scalar(x):
    return np.isfinite(x)


def safe_div(num, den):
    return float(num) / float(den) if den != 0 else np.nan


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if valid.sum() == 0:
        return np.nan
    return np.sum(values[valid] * weights[valid]) / np.sum(weights[valid])


def read_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"Could not open raster: {path}")
    arr = ds.ReadAsArray()
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    metadata = ds.GetMetadata()
    gcpcount = ds.GetGCPCount()
    gcps = ds.GetGCPs() if gcpcount > 0 else None
    gcpproj = ds.GetGCPProjection() if gcpcount > 0 else None
    return {
        "ds": ds,
        "arr": np.squeeze(arr),
        "nodata": nodata,
        "gt": gt,
        "proj": proj,
        "metadata": metadata,
        "gcps": gcps,
        "gcpproj": gcpproj,
    }


def write_geotiff_like(reference_path, out_arr, out_path, dtype=gdal.GDT_Float32, nodata=None):
    ref = gdal.Open(reference_path)
    if ref is None:
        raise FileNotFoundError(f"Could not open reference raster: {reference_path}")

    ref_arr = ref.ReadAsArray()
    nx = out_arr.shape[1]
    ny = out_arr.shape[0]

    geo_transform = ref.GetGeoTransform()
    gt2 = [geo_transform[0], geo_transform[1], geo_transform[2],
           geo_transform[3], geo_transform[4], geo_transform[5]]
    gt2[1] = gt2[1] * ref_arr.shape[1] / nx
    gt2[5] = gt2[5] * ref_arr.shape[0] / ny

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_ds = gdal.GetDriverByName("GTiff").Create(out_path, nx, ny, 1, dtype)
    out_ds.SetGeoTransform(gt2)
    out_ds.SetMetadata(ref.GetMetadata())
    out_ds.SetProjection(ref.GetProjection())

    if ref.GetGCPCount() > 0:
        out_ds.SetGCPs(ref.GetGCPs(), ref.GetGCPProjection())

    band = out_ds.GetRasterBand(1)
    band.WriteArray(out_arr)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.FlushCache()
    out_ds.FlushCache()
    out_ds = None
    ref = None


def resample_to_target(src_path, src_arr, target_path, radius_of_influence=500, fill_value=np.nan):
    src_area = get_area_def_from_raster(src_path)
    tgt_area = get_area_def_from_raster(target_path)
    out = kd_tree.resample_nearest(
        src_area,
        src_arr,
        tgt_area,
        radius_of_influence=radius_of_influence,
        fill_value=fill_value,
    )
    return np.squeeze(out)


def to_binary_mask(arr, threshold=0.0, nodata=None):
    arr = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= (arr != nodata)
    mask = np.zeros(arr.shape, dtype=np.uint8)
    mask[valid & (arr > threshold)] = 1
    return mask, valid


def confusion_from_masks(truth, pred, valid_mask):
    t = truth[valid_mask].astype(np.uint8)
    p = pred[valid_mask].astype(np.uint8)

    tp = int(np.sum((t == 1) & (p == 1)))
    tn = int(np.sum((t == 0) & (p == 0)))
    fp = int(np.sum((t == 0) & (p == 1)))
    fn = int(np.sum((t == 1) & (p == 0)))
    return tp, tn, fp, fn


def metrics_from_confusion(tp, tn, fp, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    iou = safe_div(tp, tp + fp + fn)
    balanced_accuracy = np.nanmean([recall, specificity])

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_div((tp * tn - fp * fn), denom) if denom != 0 else np.nan

    total = tp + tn + fp + fn
    prevalence = safe_div(tp + fn, total)
    predicted_positive_rate = safe_div(tp + fp, total)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "iou": iou,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "prevalence": prevalence,
        "predicted_positive_rate": predicted_positive_rate,
        "n_valid_pixels": total,
    }


def dice_similarity(truth, pred, valid_mask):
    t = truth[valid_mask].astype(np.uint8)
    p = pred[valid_mask].astype(np.uint8)
    denom = (2 * np.sum((t == 1) & (p == 1)) + np.sum((t == 1) & (p == 0)) + np.sum((t == 0) & (p == 1)))
    if denom == 0:
        return np.nan
    tp = np.sum((t == 1) & (p == 1))
    fp = np.sum((t == 0) & (p == 1))
    fn = np.sum((t == 1) & (p == 0))
    return safe_div(2 * tp, 2 * tp + fp + fn)


def compute_ssim(truth, pred, valid_mask):
    if np.sum(valid_mask) == 0:
        return np.nan

    truth_f = truth.astype(np.float32).copy()
    pred_f = pred.astype(np.float32).copy()

    truth_f[~valid_mask] = 0
    pred_f[~valid_mask] = 0

    win_size = min(truth_f.shape[0], truth_f.shape[1])
    if win_size < 3:
        return np.nan
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        return np.nan

    return structural_similarity(
        truth_f,
        pred_f,
        data_range=1.0,
        gaussian_weights=False,
        win_size=win_size,
    )


def diff_map(reference_path, truth_mask, pred_mask, valid_mask, out_path):
    diff = np.full(truth_mask.shape, np.nan, dtype=np.float32)
    diff[valid_mask] = pred_mask[valid_mask].astype(np.float32) - truth_mask[valid_mask].astype(np.float32)
    write_geotiff_like(reference_path, diff, out_path, dtype=gdal.GDT_Float32, nodata=np.nan)


def summarize_metric(rows, key):
    vals = [r[key] for r in rows if is_finite_scalar(r.get(key, np.nan))]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


def weighted_metric(rows, key, weight_key="n_valid_pixels"):
    vals = [r.get(key, np.nan) for r in rows]
    wts = [r.get(weight_key, np.nan) for r in rows]
    return float(weighted_mean(vals, wts))


def aggregate_rows(rows):
    tp = sum(r["tp"] for r in rows)
    tn = sum(r["tn"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)

    agg = metrics_from_confusion(tp, tn, fp, fn)
    agg["dice_mean"] = summarize_metric(rows, "dice")
    agg["dice_weighted"] = weighted_metric(rows, "dice")
    agg["ssim_mean"] = summarize_metric(rows, "ssim")
    agg["ssim_weighted"] = weighted_metric(rows, "ssim")
    agg["n_scenes"] = len(rows)
    return agg


def compare_one_pair(sf_path, truth_path, other_path, cfg, outputs):
    radius = cfg.get("radius_of_influence", 500)
    sit_thresh = cfg.get("sit_fuse_threshold", 0.0)
    truth_thresh = cfg.get("truth_threshold", 0.0)
    other_thresh = cfg.get("other_threshold", 0.0)

    sf = read_raster(sf_path)
    truth = read_raster(truth_path)

    truth_on_sf = resample_to_target(
        truth_path,
        truth["arr"],
        sf_path,
        radius_of_influence=radius,
        fill_value=np.nan,
    )

    sf_mask, sf_valid = to_binary_mask(sf["arr"], threshold=sit_thresh, nodata=sf["nodata"])
    truth_mask, truth_valid = to_binary_mask(truth_on_sf, threshold=truth_thresh, nodata=np.nan)

    valid_eval = sf_valid & truth_valid

    row = {
        "sit_fuse_map": sf_path,
        "truth_map": truth_path,
        "other_map": other_path if other_path is not None else "",
    }

    if np.sum(valid_eval) == 0:
        row.update({
            "tp": 0, "tn": 0, "fp": 0, "fn": 0,
            "precision": np.nan, "recall": np.nan, "specificity": np.nan,
            "f1": np.nan, "iou": np.nan, "balanced_accuracy": np.nan,
            "mcc": np.nan, "prevalence": np.nan,
            "predicted_positive_rate": np.nan,
            "dice": np.nan, "ssim": np.nan,
            "n_valid_pixels": 0,
            "status": "no_valid_overlap",
        })
        return row, None

    tp, tn, fp, fn = confusion_from_masks(truth_mask, sf_mask, valid_eval)
    row.update(metrics_from_confusion(tp, tn, fp, fn))
    row["dice"] = dice_similarity(truth_mask, sf_mask, valid_eval)
    row["ssim"] = compute_ssim(truth_mask, sf_mask, valid_eval)
    row["status"] = "ok"

    if outputs.get("write_diff_rasters", True):
        diff_dir = Path(outputs["diff_raster_dir"])
        diff_dir.mkdir(parents=True, exist_ok=True)
        sf_stem = Path(sf_path).stem
        diff_map(
            sf_path,
            truth_mask,
            sf_mask,
            valid_eval,
            diff_dir / f"{sf_stem}_diff_truth_vs_sitfuse.tif",
        )

    other_row = None
    if other_path is not None and str(other_path).strip() != "":
        other = read_raster(other_path)
        other_on_sf = resample_to_target(
            other_path,
            other["arr"],
            sf_path,
            radius_of_influence=radius,
            fill_value=np.nan,
        )
        other_mask, other_valid = to_binary_mask(other_on_sf, threshold=other_thresh, nodata=np.nan)
        valid_eval_other = valid_eval & other_valid

        if np.sum(valid_eval_other) > 0:
            tp2, tn2, fp2, fn2 = confusion_from_masks(truth_mask, other_mask, valid_eval_other)
            other_row = metrics_from_confusion(tp2, tn2, fp2, fn2)
            other_row["dice"] = dice_similarity(truth_mask, other_mask, valid_eval_other)
            other_row["ssim"] = compute_ssim(truth_mask, other_mask, valid_eval_other)
            other_row["status"] = "ok"
            other_row["n_valid_pixels"] = int(np.sum(valid_eval_other))

            if outputs.get("write_diff_rasters", True):
                diff_dir = Path(outputs["diff_raster_dir"])
                sf_stem = Path(sf_path).stem
                diff_map(
                    sf_path,
                    truth_mask,
                    other_mask,
                    valid_eval_other,
                    diff_dir / f"{sf_stem}_diff_truth_vs_other.tif",
                )
        else:
            other_row = {
                "tp": 0, "tn": 0, "fp": 0, "fn": 0,
                "precision": np.nan, "recall": np.nan, "specificity": np.nan,
                "f1": np.nan, "iou": np.nan, "balanced_accuracy": np.nan,
                "mcc": np.nan, "prevalence": np.nan,
                "predicted_positive_rate": np.nan,
                "dice": np.nan, "ssim": np.nan,
                "n_valid_pixels": 0,
                "status": "no_valid_overlap",
            }

    return row, other_row


def write_rows_csv(path, rows):
    if len(rows) == 0:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(path, summary_rows):
    if len(summary_rows) == 0:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary_rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def regrid_and_compare(config):
    sit_fuse_map_fnames = config["sit_fuse_maps"]
    truth_map_fnames = config["truth_maps"]
    other_map_fnames = config.get("other_maps", None)

    if len(sit_fuse_map_fnames) != len(truth_map_fnames):
        raise ValueError("sit_fuse_maps and truth_maps must have the same length.")
    if other_map_fnames is not None and len(other_map_fnames) != len(sit_fuse_map_fnames):
        raise ValueError("other_maps must have the same length as sit_fuse_maps when provided.")

    analysis_cfg = config.get("analysis", {})
    outputs_cfg = config.get("outputs", {})
    outputs_cfg.setdefault("per_scene_csv", "comparison_per_scene.csv")
    outputs_cfg.setdefault("summary_csv", "comparison_summary.csv")
    outputs_cfg.setdefault("diff_raster_dir", "diff_rasters")
    outputs_cfg.setdefault("write_diff_rasters", True)

    per_scene_rows = []
    per_scene_other_rows = []

    for i in range(len(sit_fuse_map_fnames)):
        sf_path = sit_fuse_map_fnames[i]
        truth_path = truth_map_fnames[i]
        other_path = other_map_fnames[i] if other_map_fnames is not None else None

        print(f"[{i+1}/{len(sit_fuse_map_fnames)}] Comparing")
        print("  SIT-FUSE:", sf_path)
        print("  TRUTH   :", truth_path)
        if other_path is not None:
            print("  OTHER   :", other_path)

        row, other_row = compare_one_pair(sf_path, truth_path, other_path, analysis_cfg, outputs_cfg)
        per_scene_rows.append(row)

        if other_row is not None:
            other_full = {
                "sit_fuse_map": sf_path,
                "truth_map": truth_path,
                "other_map": other_path,
                **other_row,
            }
            per_scene_other_rows.append(other_full)

    sit_valid_rows = [r for r in per_scene_rows if r["status"] == "ok"]
    sit_summary = aggregate_rows(sit_valid_rows) if len(sit_valid_rows) > 0 else {"n_scenes": 0}
    sit_summary["comparison"] = "sit_fuse_vs_truth"

    summary_rows = [sit_summary]

    if len(per_scene_other_rows) > 0:
        other_valid_rows = [r for r in per_scene_other_rows if r["status"] == "ok"]
        other_summary = aggregate_rows(other_valid_rows) if len(other_valid_rows) > 0 else {"n_scenes": 0}
        other_summary["comparison"] = "other_vs_truth"
        summary_rows.append(other_summary)

    baseline_zero = None
    baseline_one = None
    if len(sit_valid_rows) > 0:
        total_tp = sum(r["tp"] for r in sit_valid_rows)
        total_tn = sum(r["tn"] for r in sit_valid_rows)
        total_fp = sum(r["fp"] for r in sit_valid_rows)
        total_fn = sum(r["fn"] for r in sit_valid_rows)
        total_valid = total_tp + total_tn + total_fp + total_fn
        total_pos = total_tp + total_fn
        total_neg = total_tn + total_fp

        baseline_zero = metrics_from_confusion(0, total_neg, 0, total_pos)
        baseline_zero["comparison"] = "baseline_all_zero"

        baseline_one = metrics_from_confusion(total_pos, 0, total_neg, 0)
        baseline_one["comparison"] = "baseline_all_one"

        summary_rows.append(baseline_zero)
        summary_rows.append(baseline_one)

    write_rows_csv(outputs_cfg["per_scene_csv"], per_scene_rows)
    if len(per_scene_other_rows) > 0:
        other_csv = os.path.splitext(outputs_cfg["per_scene_csv"])[0] + "_other.csv"
        write_rows_csv(other_csv, per_scene_other_rows)
    write_summary_csv(outputs_cfg["summary_csv"], summary_rows)

    print("\nSIT-FUSE vs TRUTH")
    for k, v in sit_summary.items():
        print(k, v)

    if len(per_scene_other_rows) > 0:
        print("\nOTHER vs TRUTH")
        for row in summary_rows:
            if row.get("comparison") == "other_vs_truth":
                for k, v in row.items():
                    print(k, v)

    print("\nOutputs written:")
    print("  Per-scene CSV:", outputs_cfg["per_scene_csv"])
    print("  Summary CSV  :", outputs_cfg["summary_csv"])
    if outputs_cfg.get("write_diff_rasters", True):
        print("  Diff rasters :", outputs_cfg["diff_raster_dir"])


def main(yml_fpath):
    yml_conf = read_yaml(yml_fpath)
    regrid_and_compare(yml_conf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", required=True, help="YAML config for comparison.")
    args = parser.parse_args()
    main(args.yaml)
