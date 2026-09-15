#!/usr/bin/env python3

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import rasterio

import joblib
import rasterio

from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

import yaml

from sklearn.base import BaseEstimator
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, cross_val_predict

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.pipeline import Pipeline as SklearnPipeline

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier



LOG = logging.getLogger("sit-fuse-utils")
DEFAULT_NODATA = -9999.0
EPS = 1e-8


def canonicalize_dates(values: pd.Series, column_name: str) -> pd.Series:
    """
    Convert date or timestamp values to ISO day strings: YYYY-MM-DD.

    Accepted examples:
      2024-06-15
      2024-06-15T19:00:00Z
      2024-06-15 19:00:00
    """
    parsed = pd.to_datetime(values, errors="coerce", utc=True)

    if parsed.isna().any():
        bad_examples = values.loc[parsed.isna()].astype(str).head(5).tolist()
        raise ValueError(
            f"Column '{column_name}' contains unparseable date values. "
            f"Examples: {bad_examples}"
        )

    return parsed.dt.strftime("%Y-%m-%d")


def normalize_holdout_dates(values) -> set[str]:
    """
    Normalize configured YAML date strings to YYYY-MM-DD.
    """
    if not isinstance(values, list) or not values:
        raise ValueError(
            "split.holdout_dates must be a non-empty YAML list when "
            "split.mode is 'explicit_dates'."
        )

    normalized = set()
    for value in values:
        parsed = pd.to_datetime(str(value), errors="raise", utc=True)
        normalized.add(parsed.strftime("%Y-%m-%d"))

    return normalized


def canonical_feature_signature(features_cfg: Dict[str, Any]) -> Dict[str, Any]:
    features = normalize_feature_config({"features": features_cfg})

    return {
        "window_radii": features["window_radii"],
        "cross_scale_deltas": features["cross_scale_deltas"],
        "cluster_nodata": features["cluster_nodata"],
        "annotation_nodata": features["annotation_nodata"],
        "hashed_histogram": features["hashed_histogram"],
        "coarse_context_mapping": features["coarse_context_mapping"],
    }



def normalize_feature_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize feature configuration to one canonical schema.

    Accepted scale keys, in precedence order:
      1. features.window_radii
      2. features.scales          # matchup-builder classifier YAML format
      3. features.window_radius   # legacy singular format

    `scales` are interpreted as neighborhood radii measured in pixels.
    """
    features = dict(cfg.get("features", {}))

    configured_radii = features.get("window_radii")

    if configured_radii is None:
        configured_radii = features.get("scales")

    if configured_radii is None:
        configured_radii = [features.get("window_radii", 1)]

    if isinstance(configured_radii, (int, float, str)):
        configured_radii = [configured_radii]

    try:
        window_radii = sorted({int(radius) for radius in configured_radii})
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "`features.window_radii` or `features.scales` must be a list "
            "of non-negative integer neighborhood radii."
        ) from exc

    if not window_radii:
        raise ValueError(
            "At least one neighborhood radius must be provided through "
            "`features.window_radii` or `features.scales`."
        )

    if any(radius < 0 for radius in window_radii):
        raise ValueError(
            "Neighborhood radii must be non-negative integers."
        )

    features["window_radii"] = window_radii

    # Keep the original YAML alias only for provenance/debugging. All operational
    # code below should read features['window_radii'].
    features["scales"] = window_radii

    features["cross_scale_deltas"] = bool(
        features.get("cross_scale_deltas", len(window_radii) > 1)
    )
    features["cluster_nodata"] = features.get("cluster_nodata")
    features["annotation_nodata"] = features.get("annotation_nodata")

    hashed_histogram = dict(features.get("hashed_histogram", {}))
    hashed_histogram["enabled"] = bool(
        hashed_histogram.get("enabled", False)
    )
    hashed_histogram["n_bins"] = int(
        hashed_histogram.get("n_bins", 64)
    )
    if hashed_histogram["n_bins"] <= 0:
        raise ValueError("features.hashed_histogram.n_bins must be positive.")
    features["hashed_histogram"] = hashed_histogram

    coarse_mapping = features.get("coarse_context_mapping", {}) or {}
    features["coarse_context_mapping"] = {
        str(int(context_id)): [int(cluster_id) for cluster_id in cluster_ids]
        for context_id, cluster_ids in coarse_mapping.items()
    }

    return features

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_raster(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        profile["transform"] = src.transform
        profile["crs"] = src.crs
        profile["nodata"] = src.nodata
        profile["height"] = src.height
        profile["width"] = src.width
        profile["dtype"] = str(arr.dtype)
        print(path, arr.min(), arr.max(), arr.std())
    return arr, profile

def rasters_are_collocated(
    source_profile: Dict[str, Any],
    target_profile: Dict[str, Any],
) -> bool:
    """
    Return True only if two rasters share the same pixel grid and CRS.

    Cluster rasters are the target grid; annotation rasters are source grids.
    """
    return (
        source_profile.get("height") == target_profile.get("height")
        and source_profile.get("width") == target_profile.get("width")
        and source_profile.get("transform") == target_profile.get("transform")
        and source_profile.get("crs") == target_profile.get("crs")
    )


def collocate_annotation_to_cluster_grid(
    annotation_arr: np.ndarray,
    annotation_profile: Dict[str, Any],
    cluster_profile: Dict[str, Any],
    annotation_nodata: float | int | None = None,
) -> tuple[np.ndarray, Dict[str, Any], bool]:
    """
    Reproject/resample a categorical annotation raster to the cluster-raster grid.

    Uses nearest-neighbor resampling to preserve discrete class IDs. Returns:
      collocated_annotation_array,
      output_profile matching the cluster grid,
      was_reprojected
    """
    if rasters_are_collocated(annotation_profile, cluster_profile):
        return annotation_arr, annotation_profile, False

    src_crs = annotation_profile.get("crs")
    dst_crs = cluster_profile.get("crs")

    if src_crs is None or dst_crs is None:
        raise ValueError(
            "Cannot collocate annotation raster because one or both CRS values are missing. "
            f"annotation_crs={src_crs!r}, cluster_crs={dst_crs!r}"
        )

    src_nodata = (
        annotation_nodata
        if annotation_nodata is not None
        else annotation_profile.get("nodata")
    )

    if src_nodata is None:
        src_nodata = DEFAULT_NODATA

    # Use a float destination so both integer masks and nodata values are retained
    # consistently through reproject().
    dst_nodata = float(src_nodata)
    destination = np.full(
        (int(cluster_profile["height"]), int(cluster_profile["width"])),
        dst_nodata,
        dtype=np.float32,
    )

    reproject(
        source=annotation_arr,
        destination=destination,
        src_transform=annotation_profile["transform"],
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=cluster_profile["transform"],
        dst_crs=dst_crs,
        dst_nodata=dst_nodata,
        resampling=Resampling.nearest,
    )

    output_profile = cluster_profile.copy()
    output_profile.update(
        dtype="float32",
        count=1,
        nodata=dst_nodata,
    )

    return destination, output_profile, True


def validate_collocation(cluster_profile: Dict[str, Any], annotation_profile: Dict[str, Any]) -> None:
    keys = ["height", "width", "transform"]
    mismatches = [k for k in keys if cluster_profile.get(k) != annotation_profile.get(k)]
    if mismatches:
        raise ValueError(f"Cluster and annotation rasters are not collocated; mismatch in {mismatches}")


def shift2d(arr: np.ndarray, dy: int, dx: int, fill_value: float) -> np.ndarray:
    out = np.full(arr.shape, fill_value, dtype=arr.dtype)
    src_y0 = max(0, -dy)
    src_y1 = min(arr.shape[0], arr.shape[0] - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(arr.shape[1], arr.shape[1] - dx)
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return out



class FeatureBuilder:
    def __init__(self, cfg: Dict[str, Any], require_annotations: bool = True):
        features = normalize_feature_config({"features": cfg})

        self.window_radii = features["window_radii"]
        self.coarse_mapping = features["coarse_context_mapping"]
        self.use_hashed_hist = bool(features["hashed_histogram"]["enabled"])
        self.hash_bins = int(features["hashed_histogram"]["n_bins"])
        self.add_cross_scale_deltas = bool(features["cross_scale_deltas"])
        self.annotation_nodata = features["annotation_nodata"]
        self.cluster_nodata = features["cluster_nodata"]
        self.require_annotations = require_annotations

    def map_coarse(self, cluster_arr: np.ndarray) -> np.ndarray:
        if not self.coarse_mapping:
            return np.full(cluster_arr.shape, -1, dtype=np.int32)
        out = np.full(cluster_arr.shape, -1, dtype=np.int32)
        for coarse_name, cluster_ids in self.coarse_mapping.items():
            coarse_id = int(coarse_name)
            mask = np.isin(cluster_arr, np.array(cluster_ids))
            out[mask] = coarse_id
        return out

    def _window_offsets(self, radius: int) -> List[Tuple[int, int]]:
        return [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]


    def _window_offsets(self, radius: int) -> List[Tuple[int, int]]:
        return [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]

    def _compute_scale_features(
        self,
        cluster_arr: np.ndarray,
        valid: np.ndarray,
        coarse_arr: np.ndarray,
        radius: int,
        cluster_fill: float,
    ) -> Dict[str, np.ndarray]:
        offsets = self._window_offsets(radius)
        neighbor_equal_sum = np.zeros(cluster_arr.shape, dtype=np.float32)
        valid_neighbor_sum = np.zeros(cluster_arr.shape, dtype=np.float32)
        center_cluster = cluster_arr.astype(np.int64)

        coarse_planes = {}
        coarse_ids = sorted(set(np.unique(coarse_arr)) - {-1})
        for coarse_id in coarse_ids:
            coarse_planes[coarse_id] = np.zeros(cluster_arr.shape, dtype=np.float32)

        hashed_planes = None
        if self.use_hashed_hist:
            hashed_planes = np.zeros((self.hash_bins, *cluster_arr.shape), dtype=np.float32)

        stack_values = []
        for dy, dx in offsets:
            shifted_cluster = shift2d(cluster_arr, dy, dx, cluster_fill)
            shifted_valid = shift2d(valid.astype(np.uint8), dy, dx, 0).astype(bool)
            shifted_coarse = shift2d(coarse_arr, dy, dx, -1)

            neighbor_equal_sum += ((shifted_cluster == center_cluster) & shifted_valid).astype(np.float32)
            valid_neighbor_sum += shifted_valid.astype(np.float32)

            for coarse_id in coarse_ids:
                coarse_planes[coarse_id] += ((shifted_coarse == coarse_id) & shifted_valid).astype(np.float32)

            if hashed_planes is not None:
                hashed_idx = np.mod(np.abs(shifted_cluster.astype(np.int64)), self.hash_bins)
                for b in range(self.hash_bins):
                    hashed_planes[b] += ((hashed_idx == b) & shifted_valid).astype(np.float32)

            masked_vals = np.where(shifted_valid, shifted_cluster, cluster_fill)
            stack_values.append(masked_vals)

        stack = np.stack(stack_values, axis=0)
        distinct_counts = np.sum(np.diff(np.sort(stack, axis=0), axis=0) != 0, axis=0) + 1
        distinct_counts[valid_neighbor_sum == 0] = 0

        p_same = np.divide(neighbor_equal_sum, np.maximum(valid_neighbor_sum, 1), dtype=np.float32)
        entropy = -(p_same * np.log(np.clip(p_same, EPS, 1.0)) + (1 - p_same) * np.log(np.clip(1 - p_same, EPS, 1.0)))
        entropy[valid_neighbor_sum == 0] = np.nan

        features = {
            f"frac_same_cluster_r{radius}": p_same.astype(np.float32),
            f"n_valid_neighbors_r{radius}": valid_neighbor_sum.astype(np.float32),
            f"n_distinct_clusters_r{radius}": distinct_counts.astype(np.float32),
            f"same_cluster_entropy_r{radius}": entropy.astype(np.float32),
        }

        for coarse_id, plane in coarse_planes.items():
            features[f"coarse_frac_{coarse_id}_r{radius}"] = np.divide(
                plane,
                np.maximum(valid_neighbor_sum, 1),
                dtype=np.float32,
            )

        if hashed_planes is not None:
            for b in range(self.hash_bins):
                features[f"hash_bin_frac_{b}_r{radius}"] = np.divide(
                    hashed_planes[b],
                    np.maximum(valid_neighbor_sum, 1),
                    dtype=np.float32,
                )

        return features



    def build_features_for_pair(
        self,
        cluster_arr: np.ndarray,
        pair_id: str,
        profile: Dict[str, Any],
        annotation_arr: np.ndarray | None = None,
    ) -> pd.DataFrame:
        cluster_fill = self.cluster_nodata if self.cluster_nodata is not None else DEFAULT_NODATA

        print("HERE NODATA", cluster_fill, self.annotation_nodata, self.cluster_nodata, DEFAULT_NODATA)

        valid = np.isfinite(cluster_arr)
        if self.cluster_nodata is not None:
            valid &= cluster_arr != cluster_fill
        valid &= ~np.isnan(cluster_arr)

        if self.annotation_nodata is not None:
            valid &= annotation_arr != self.annotation_nodata
        valid &= ~np.isnan(annotation_arr)

        coarse_arr = self.map_coarse(cluster_arr)
        vals, counts = np.unique(cluster_arr[valid], return_counts=True)
        freq_map = {int(v): c / max(len(cluster_arr[valid]), 1) for v, c in zip(vals, counts)}
        center_logfreq = np.vectorize(lambda x: np.log(freq_map.get(int(x), EPS)))(cluster_arr).astype(np.float32)

        data = {
            "pair_id": np.full(cluster_arr.shape, pair_id),
            "row": np.repeat(np.arange(cluster_arr.shape[0])[:, None], cluster_arr.shape[1], axis=1),
            "col": np.repeat(np.arange(cluster_arr.shape[1])[None, :], cluster_arr.shape[0], axis=0),
            "raster_height": np.full(cluster_arr.shape, profile["height"]),
            "raster_width": np.full(cluster_arr.shape, profile["width"]),
            "center_cluster_logfreq": center_logfreq,
            "label": annotation_arr,
            "is_valid": valid,
        }

        per_scale_features = {}
 
        for radius in self.window_radii:
            scale_features = self._compute_scale_features(
                cluster_arr=cluster_arr,
                valid=valid,
                coarse_arr=coarse_arr,
                radius=radius,
                cluster_fill=cluster_fill,
            )

            per_scale_features[radius] = scale_features
            data.update(scale_features)

        if self.add_cross_scale_deltas and len(self.window_radii) > 1:
            base_feature_names = [
                "frac_same_cluster",
                "n_distinct_clusters",
                "same_cluster_entropy",
            ]

            coarse_ids = sorted(set(np.unique(coarse_arr)) - {-1})
            for coarse_id in coarse_ids:
                base_feature_names.append(f"coarse_frac_{coarse_id}")

            for small_radius, large_radius in zip(
                self.window_radii[:-1],
                self.window_radii[1:],
            ):
                for base_name in base_feature_names:
                    small_key = f"{base_name}_r{small_radius}"
                    large_key = f"{base_name}_r{large_radius}"

                    if (
                        small_key not in per_scale_features[small_radius]
                        or large_key not in per_scale_features[large_radius]
                    ):
                        continue

                    data[
                        f"{base_name}_delta_r{large_radius}_minus_r{small_radius}"
                    ] = (
                        per_scale_features[large_radius][large_key]
                        - per_scale_features[small_radius][small_key]
                    )



        if annotation_arr is not None:
            data["label"] = annotation_arr

        df = pd.DataFrame({k: np.asarray(v).ravel() for k, v in data.items()})
        df = df[df["is_valid"]].drop(columns=["is_valid"]).reset_index(drop=True)



        return df

def build_oversampler(cfg: Dict):
    """
    Return an optional RandomOverSampler configured from YAML.

    Oversampling is performed only during estimator fitting when this sampler
    is placed inside an imblearn Pipeline. It never modifies CV validation
    folds or the final holdout set.

    YAML examples:

      oversampling:
        enabled: true
        sampling_strategy: 0.10

      oversampling:
        enabled: true
        sampling_strategy:
          1: 50000
    """
    over_cfg = cfg.get("oversampling", {})

    if not over_cfg.get("enabled", False):
        return None

    sampling_strategy = over_cfg.get("sampling_strategy", "minority")

    return RandomOverSampler(
        sampling_strategy=sampling_strategy,
        random_state=int(cfg.get("random_state", 42)),
    )


def build_models(cfg: Dict) -> Dict[str, Pipeline]:
    rs = int(cfg.get("random_state", 42))
    sampler = build_oversampler(cfg)


    def make_pipeline(steps):
        """
        Use imblearn Pipeline only when oversampling is enabled.
        """
        if sampler is None:
            return SklearnPipeline(steps)

        sampler_copy = RandomOverSampler(
            sampling_strategy=cfg.get("oversampling", {}).get(
                "sampling_strategy",
                "minority",
            ),
            random_state=rs,
        )

        return ImbPipeline([
            *steps[:-1],
            ("oversample", sampler_copy),
            steps[-1],
        ])


    models = {
        "logistic_regression": make_pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=rs)),
        ]),
        "decision_tree": make_pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(max_depth=cfg.get("decision_tree", {}).get("max_depth", 7),
                                           min_samples_leaf=cfg.get("decision_tree", {}).get("min_samples_leaf", 100),
                                           class_weight=cfg.get("decision_tree", {}).get("class_weight", "balanced"),
                                           random_state=rs,
                                           min_samples_split=cfg.get("decision_tree", {}).get("min_samples_split", 200), 
                                           max_features=cfg.get("decision_tree", {}).get("max_features", "sqrt"), 
                                           ccp_alpha=cfg.get("decision_tree", {}).get("ccp_alpha", 0.00001))
            ),
        ]),
        "random_forest": make_pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=cfg.get("random_forest", {}).get("n_estimators", 400),
                max_depth=cfg.get("random_forest", {}).get("max_depth", 10),
                min_samples_leaf=cfg.get("random_forest", {}).get("min_samples_leaf", 50),
                min_samples_split= cfg.get("random_forest", {}).get("min_samples_split", 100),
                class_weight=cfg.get("random_forest", {}).get("class_weight", "balanced_subsample"),
                bootstrap= cfg.get("random_forest", {}).get("bootstrap", True),
                max_features=cfg.get("random_forest", {}).get("max_features", "sqrt"),
                max_samples=cfg.get("random_forest", {}).get("max_samples", 0.7),
                n_jobs=cfg.get("random_forest", {}).get("n_jobs", -1),
                random_state=rs
            )),
        ]),
        "hist_gradient_boosting": make_pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=float(
                    cfg.get("hist_gradient_boosting", {}).get("learning_rate", 0.08)
                ),
                max_iter=int(
                    cfg.get("hist_gradient_boosting", {}).get("max_iter", 300)
                ),
                max_leaf_nodes=int(
                    cfg.get("hist_gradient_boosting", {}).get("max_leaf_nodes", 31)
                ),
                l2_regularization=float(
                    cfg.get("hist_gradient_boosting", {}).get("l2_regularization", 1.0)
                ),
                early_stopping=bool(
                    cfg.get("hist_gradient_boosting", {}).get("early_stopping", True)
                ),
                validation_fraction=float(
                    cfg.get("hist_gradient_boosting", {}).get(
                        "validation_fraction",
                        0.1,
                    )
                ),
                class_weight=cfg.get(
                    "hist_gradient_boosting",
                    {},
                    ).get("class_weight", "balanced"),
                
                n_iter_no_change=int(
                    cfg.get("hist_gradient_boosting", {}).get(
                        "n_iter_no_change",
                        20,
                    )
                ),
                tol=float(
                    cfg.get("hist_gradient_boosting", {}).get("tol", 1e-4)
                ),
                random_state=rs,
            )),
        ]),
    }
    requested = cfg.get("models", list(models.keys()))
    return {k: v for k, v in models.items() if k in requested}


def build_cv_splitter(groups: pd.Series, cfg: Dict):
    cv_cfg = cfg.get("cross_validation", {})
    n_splits = int(cv_cfg.get("n_splits", 5))
    use_grouped = bool(cv_cfg.get("grouped", True))
    rs = int(cfg.get("random_state", 42))
    if use_grouped and groups.nunique() >= n_splits:
        return GroupKFold(n_splits=n_splits), "group"
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs), "stratified"


def parse_matchup_timestamp(value: Any, field_name: str, pair_id: str) -> pd.Timestamp:
    """
    Parse the UTC timestamps emitted by build_simple_classifier_matchups.py.

    Supported generator formats:
      - YYYY-MM-DDTHHMMSSZ
      - YYYY-MM-DDTHHMMSS
      - YYYY-MM-DD
    """
    if value is None:
        raise ValueError(
            f"Matchup {pair_id!r} is missing required timestamp field {field_name!r}."
        )

    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(
            f"Matchup {pair_id!r} has an invalid {field_name!r} value: {value!r}"
        )

    return timestamp


def normalize_training_matchup(
    matchup: Dict[str, Any],
    index: int,
) -> Dict[str, Any]:
    """
    Normalize one record emitted by build_simple_classifier_matchups.py.

    Expected matchup-builder fields:
      pairid, label, product, sit_fuse_context_free, truth_hms,
      sit_fuse_timestamp, hmstimestamp, hmsdate, timeoffsetseconds.

    A legacy `rasters` record is supported as a fallback.
    """
    pair_id = matchup.get("pairid") or matchup.get("pair_id") or f"pair_{index:05d}"

    cluster_raster = (
        matchup.get("sit_fuse_context_free")
        or matchup.get("cluster_raster")
    )
    annotation_raster = (
        matchup.get("truth_hms")
        or matchup.get("annotation_raster")
    )

    if not cluster_raster:
        raise ValueError(
            f"Matchup {pair_id!r} is missing `sit_fuse_context_free` "
            "(or legacy `cluster_raster`)."
        )

    if not annotation_raster:
        raise ValueError(
            f"Matchup {pair_id!r} is missing `truth_hms` "
            "(or legacy `annotation_raster`)."
        )

    sitfuse_timestamp = parse_matchup_timestamp(
        matchup.get("sit_fuse_timestamp") or matchup.get("acquisition_datetime"),
        "sit_fuse_timestamp",
        str(pair_id),
    )

    hms_timestamp_value = matchup.get("hmstimestamp")
    hms_timestamp = (
        parse_matchup_timestamp(hms_timestamp_value, "hmstimestamp", str(pair_id))
        if hms_timestamp_value is not None
        else pd.NaT
    )

    hms_date_value = matchup.get("hmsdate")
    hms_date = (
        pd.to_datetime(hms_date_value, utc=True, errors="raise").normalize()
        if hms_date_value is not None
        else hms_timestamp.normalize()
        if pd.notna(hms_timestamp)
        else sitfuse_timestamp.normalize()
    )

    return {
        "pair_id": str(pair_id),
        "cluster_raster": Path(cluster_raster),
        "annotation_raster": Path(annotation_raster),
        "sitfuse_timestamp": sitfuse_timestamp,
        "sitfuse_date": sitfuse_timestamp.normalize(),
        "hms_timestamp": hms_timestamp,
        "hms_date": hms_date,
        "label_name": matchup.get("label"),
        "product": matchup.get("product"),
        "time_offset_seconds": matchup.get("timeoffsetseconds"),
    }


def get_training_matchups(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Read matchup-builder output (`matchups`) or legacy records (`rasters`).
    """
    raw_matchups = cfg.get("matchups")
    if raw_matchups is None:
        raw_matchups = cfg.get("rasters")

    if not isinstance(raw_matchups, list) or not raw_matchups:
        raise ValueError(
            "Expected a non-empty top-level `matchups` list from the matchup builder "
            "or a legacy `rasters` list."
        )

    return [
        normalize_training_matchup(record, index)
        for index, record in enumerate(raw_matchups)
    ]


def load_yaml(path: Path | str) -> dict[str, Any]:
    """
    Load a YAML configuration file safely.

    Parameters
    ----------
    path
        YAML file path.

    Returns
    -------
    dict
        Parsed YAML mapping.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(
            f"YAML root must be a mapping/dictionary: {path}. "
            f"Got {type(data).__name__}."
        )

    return data


def load_json(path: Path | str) -> dict[str, Any]:
    """
    Load a JSON file and require a top-level dictionary.

    Parameters
    ----------
    path
        JSON file path.

    Returns
    -------
    dict
        Parsed JSON object.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"JSON root must be an object/dictionary: {path}. "
            f"Got {type(data).__name__}."
        )

    return data


def _canonicalize_config_value(value: Any) -> Any:
    """
    Normalize nested config values for stable comparison.

    Lists retain order because feature/scales/window ordering can be
    semantically meaningful. Dictionaries are normalized recursively with
    sorted keys. Path-like objects are converted to strings.
    """
    if isinstance(value, dict):
        return {
            str(key): _canonicalize_config_value(value[key])
            for key in sorted(value, key=lambda key: str(key))
        }

    if isinstance(value, (list, tuple)):
        return [_canonicalize_config_value(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value


def compare_feature_signatures(
    expected_signature: dict[str, Any] | None,
    observed_signature: dict[str, Any] | None,
) -> list[str]:
    """
    Compare canonical feature signatures from training and inference.

    Returns a list of mismatch descriptions. An empty list means the
    signatures are equivalent.

    Examples of returned values:
      [
        "missing at inference: scales",
        "unexpected at inference: neighborhood_windows",
        "value mismatch: hash_bins "
        "(training=4096, inference=16384)",
      ]
    """
    expected = _canonicalize_config_value(expected_signature or {})
    observed = _canonicalize_config_value(observed_signature or {})

    differences: list[str] = []

    def compare_recursive(
        expected_value: Any,
        observed_value: Any,
        prefix: str = "",
    ) -> None:
        if isinstance(expected_value, dict) and isinstance(observed_value, dict):
            expected_keys = set(expected_value)
            observed_keys = set(observed_value)

            for key in sorted(expected_keys - observed_keys):
                field = f"{prefix}.{key}".lstrip(".")
                differences.append(f"missing at inference: {field}")

            for key in sorted(observed_keys - expected_keys):
                field = f"{prefix}.{key}".lstrip(".")
                differences.append(f"unexpected at inference: {field}")

            for key in sorted(expected_keys & observed_keys):
                field = f"{prefix}.{key}".lstrip(".")
                compare_recursive(
                    expected_value[key],
                    observed_value[key],
                    field,
                )
            return

        if expected_value != observed_value:
            differences.append(
                f"value mismatch: {prefix} "
                f"(training={expected_value!r}, "
                f"inference={observed_value!r})"
            )

    compare_recursive(expected, observed)
    return differences


def enforce_feature_order(
    df: pd.DataFrame,
    expected_feature_columns: list[str],
) -> pd.DataFrame:
    """
    Validate and reorder a feature DataFrame to training-time feature order.

    Parameters
    ----------
    df
        DataFrame containing candidate model features.
    expected_feature_columns
        Ordered feature names saved during training.

    Returns
    -------
    pd.DataFrame
        Copy of df restricted to expected feature columns and ordered exactly
        as during training.

    Raises
    ------
    ValueError
        If training feature names are duplicated or input columns are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "enforce_feature_order expects a pandas DataFrame, "
            f"received {type(df).__name__}."
        )

    if not expected_feature_columns:
        raise ValueError(
            "Training summary does not contain any expected feature columns."
        )

    expected = [str(column) for column in expected_feature_columns]

    duplicated_expected = sorted(
        {
            column
            for column in expected
            if expected.count(column) > 1
        }
    )

    if duplicated_expected:
        raise ValueError(
            "Training feature list contains duplicate names: "
            f"{duplicated_expected}"
        )

    observed_columns = [str(column) for column in df.columns]
    observed_set = set(observed_columns)

    missing = [column for column in expected if column not in observed_set]

    if missing:
        raise ValueError(
            "Inference features do not include all training features. "
            f"Missing {len(missing)} column(s): {missing}"
        )

    # Copy prevents callers from modifying their original metadata-bearing
    # DataFrame accidentally after feature selection.
    ordered = df.loc[:, expected].copy()

    non_numeric = [
        column
        for column in expected
        if not pd.api.types.is_numeric_dtype(ordered[column])
    ]

    if non_numeric:
        raise ValueError(
            "Non-numeric inference feature columns found: "
            f"{non_numeric}. Encode or convert them before model inference."
        )

    return ordered



def build_training_dataset(
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Build a labeled training table from matchup-builder records.

    The SIT-FUSE cluster raster defines the canonical target grid. HMS annotation
    rasters are used directly when already collocated; otherwise, they are
    resampled/reprojected to the cluster grid with nearest-neighbor resampling.

    Required matchup-builder fields:
      - pairid
      - sitfusecontextfree
      - truthhms
      - sitfusetimestamp

    Retained metadata supports split strategies based on SIT-FUSE acquisition
    time, HMS truth date, product, and pair ID.
    """
    feature_builder = FeatureBuilder(
        cfg.get("features", {}),
        require_annotations=True,
    )
    configured_annotation_nodata = cfg.get("features", {}).get(
        "annotation_nodata"
    )

    frames: List[pd.DataFrame] = []
    pair_meta: Dict[str, Dict[str, Any]] = {}

    for matchup in get_training_matchups(cfg):
        pair_id = matchup["pair_id"]

        cluster_arr, cluster_profile = read_raster(matchup["cluster_raster"])
    
        cluster_arr = (cluster_arr * 1000.0).astype(np.int32)
        cluster_profile["nodata"] = cfg.get("features").get("cluster_nodata")
        cluster_arr[np.where(cluster_arr <= 0.0)] = cluster_profile.get("nodata")

        print(cluster_profile.get("nodata"), cluster_arr.min(), cluster_arr.max())

        annotation_arr, annotation_profile = read_raster(
            matchup["annotation_raster"]
        )


        print(cluster_arr)
        annotation_profile["nodata"] = cfg.get("features").get("cluster_nodata")
        annotation_arr[np.where((annotation_arr <= 0) & (annotation_arr >= -2))] = 0
        annotation_arr[np.where(annotation_arr >= 1)] = 1
        print(annotation_arr, annotation_arr.min(), annotation_arr.max())

        (
            annotation_arr,
            aligned_annotation_profile,
            annotation_was_collocated,
        ) = collocate_annotation_to_cluster_grid(
            annotation_arr=annotation_arr,
            annotation_profile=annotation_profile,
            cluster_profile=cluster_profile,
            annotation_nodata=configured_annotation_nodata,
        )

        # This should always pass after the helper returns. Keeping this check makes
        # an unexpected grid error fail early and explicitly.
        validate_collocation(cluster_profile, aligned_annotation_profile)

        annotation_nodata = aligned_annotation_profile.get("nodata")
        if configured_annotation_nodata is not None:
            annotation_nodata = configured_annotation_nodata

        pair_meta[pair_id] = {
            "height": cluster_profile["height"],
            "width": cluster_profile["width"],
            "transform": tuple(cluster_profile["transform"]),
            "crs": (
                cluster_profile["crs"].to_string()
                if cluster_profile["crs"] is not None
                else None
            ),
            "cluster_nodata": cluster_profile.get("nodata"),
            "annotation_nodata": annotation_nodata,
            "annotation_was_collocated": annotation_was_collocated,
            "sitfuse_timestamp": matchup["sitfuse_timestamp"].isoformat(),
            "sitfuse_date": matchup["sitfuse_date"].date().isoformat(),
            "hms_timestamp": (
                matchup["hms_timestamp"].isoformat()
                if pd.notna(matchup["hms_timestamp"])
                else None
            ),
            "hms_date": matchup["hms_date"].date().isoformat(),
            "label_name": matchup["label_name"],
            "product": matchup["product"],
            "time_offset_seconds": matchup["time_offset_seconds"],
            "cluster_raster": str(matchup["cluster_raster"]),
            "annotation_raster": str(matchup["annotation_raster"]),
        }


        frame = feature_builder.build_features_for_pair(
            cluster_arr=cluster_arr,
            pair_id=pair_id,
            profile=cluster_profile,
            annotation_arr=annotation_arr,
        )

        frame["sitfuse_timestamp"] = matchup["sitfuse_timestamp"]
        frame["sitfuse_date"] = matchup["sitfuse_date"]
        frame["hms_timestamp"] = matchup["hms_timestamp"]
        frame["hms_date"] = matchup["hms_date"]
        frame["matchup_label"] = matchup["label_name"]
        frame["product"] = matchup["product"]
        frame["time_offset_seconds"] = matchup["time_offset_seconds"]
        frame["annotation_was_collocated"] = annotation_was_collocated

        frames.append(frame)

        action = "reprojected/resampled" if annotation_was_collocated else "already collocated"
        LOG.info(
            "Processed matchup %s; annotation grid was %s.",
            pair_id,
            action,
        )

    if not frames:
        raise ValueError("No matchup records were successfully processed.")

    return pd.concat(frames, ignore_index=True), pair_meta





def build_inference_dataset(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    fb = FeatureBuilder(cfg.get("features", {}), require_annotations=False)
    frames = []
    pair_meta = {}
    for i, r in enumerate(cfg["rasters"]):
        pair_id = r.get("pair_id", f"pair_{i}")
        cluster_arr, cp = read_raster(Path(r["cluster_raster"]))
        pair_meta[pair_id] = {
            "height": cp["height"],
            "width": cp["width"],
            "transform": tuple(cp["transform"]),
            "crs": cp["crs"].to_string() if cp["crs"] is not None else None,
        }
        frames.append(fb.build_features_for_pair(cluster_arr, pair_id, cp, None))
    return pd.concat(frames, ignore_index=True), pair_meta


def feature_columns(df: pd.DataFrame) -> List[str]:
    metadata_columns = {
        "pair_id",
        "row",
        "col",
        "raster_height",
        "raster_width",
        "label",
        "sitfuse_timestamp",
        "sitfuse_date",
        "hms_timestamp",
        "hms_date",
        "matchup_label",
        "product",
        "time_offset_seconds",
        "annotation_was_collocated",
    }
    return [column for column in df.columns if column not in metadata_columns]


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(np.clip(probs, EPS, 1.0)), axis=1)


def margin_uncertainty(probs: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(probs, axis=1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2] if probs.shape[1] > 1 else np.zeros_like(top1)
    return 1.0 - (top1 - top2)


def forest_tree_probabilities(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if not hasattr(clf, "estimators_"):
        raise ValueError("Model is not a random forest with estimators_")
    Xt = X.copy()
    if hasattr(model, "named_steps"):
        for step_name, step in model.named_steps.items():
            if step_name == "clf":
                break
            Xt = step.transform(Xt)
    tree_probs = []
    classes = clf.classes_
    for tree in clf.estimators_:
        p = tree.predict_proba(Xt)
        if p.shape[1] != len(classes):
            aligned = np.zeros((p.shape[0], len(classes)), dtype=np.float32)
            for j, cls in enumerate(tree.classes_):
                idx = np.where(classes == cls)[0][0]
                aligned[:, idx] = p[:, j]
            p = aligned
        tree_probs.append(p.astype(np.float32))
    return np.stack(tree_probs, axis=0)


def bootstrap_model_probabilities(base_model_path: Path, X: pd.DataFrame, bootstrap_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    import joblib
    models = [joblib.load(base_model_path)] + [joblib.load(Path(p)) for p in bootstrap_paths]
    classes = models[0].classes_
    probs = []
    for m in models:
        p = m.predict_proba(X)
        if not np.array_equal(m.classes_, classes):
            aligned = np.zeros((p.shape[0], len(classes)), dtype=np.float32)
            for j, cls in enumerate(m.classes_):
                idx = np.where(classes == cls)[0][0]
                aligned[:, idx] = p[:, j]
            p = aligned
        probs.append(p.astype(np.float32))
    return np.stack(probs, axis=0), classes


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series, cfg: Dict) -> Dict:
    splitter, split_type = build_cv_splitter(groups, cfg)
    if split_type == "group":
        probs = cross_val_predict(model, X, y, cv=splitter, groups=groups, method="predict_proba", n_jobs=1)
    else:
        probs = cross_val_predict(model, X, y, cv=splitter, method="predict_proba", n_jobs=1)

    fitted = model.fit(X, y)
    classes = fitted.classes_
    preds = classes[np.argmax(probs, axis=1)]
    labels = list(classes)
    return {
        "cv_type": split_type,
        "classification_report": classification_report(y, preds, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, preds, labels=labels).tolist(),
        "labels": [x.item() if hasattr(x, "item") else x for x in labels],
    }


def _timestamp_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise ValueError(
            f"Requested split column {column!r} is not in the dataset. "
            f"Available columns include: {sorted(df.columns)}"
        )
    timestamps = pd.to_datetime(df[column], utc=True, errors="coerce")
    if timestamps.isna().any():
        n_bad = int(timestamps.isna().sum())
        raise ValueError(
            f"Split column {column!r} contains {n_bad} unparseable timestamp values."
        )
    return timestamps


def _date_mask(
    timestamps: pd.Series,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    mask = pd.Series(True, index=timestamps.index)

    if start is not None:
        start_ts = pd.to_datetime(start, utc=True)
        mask &= timestamps >= start_ts

    if end is not None:
        end_ts = pd.to_datetime(end, utc=True)
        if len(str(end)) == 10:
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        mask &= timestamps <= end_ts

    return mask


def holdout_split(
    df: pd.DataFrame,
    cfg: Dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Supported YAML `split.strategy` values:

      grouped_random
        Randomly holds out full values of `group_column` (legacy behavior).

      grouped_date
        Randomly holds out complete UTC dates from `date_column`.

      date_cutoff
        Train before `cutoff`; test at/after `cutoff`.

      date_ranges
        Use explicit inclusive train/test windows.

    For matchup-builder output, use `sitfuse_timestamp` for temporal
    generalization. Use `hms_date` only when you deliberately want splits
    based on truth-product day rather than observation time.
    """
    split_cfg = cfg.get("split", {})

    if not split_cfg.get("enabled", False):
        return df.reset_index(drop=True), pd.DataFrame()

    strategy = str(split_cfg.get("strategy", "grouped_random")).lower()
    random_state = int(cfg.get("random_state", 42))
    label_col = split_cfg.get("label_column", cfg.get("label_column", "label"))
    group_col = split_cfg.get("group_column", cfg.get("group_column", "pair_id"))
    date_col = split_cfg.get("date_column", "sitfuse_timestamp")

    if strategy == "grouped_random":
        if group_col not in df.columns:
            raise ValueError(f"Group column {group_col!r} does not exist.")

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=float(split_cfg.get("test_size", 0.2)),
            random_state=random_state,
        )
        train_idx, test_idx = next(
            splitter.split(df, df[label_col], groups=df[group_col])
        )
        return (
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True),
        )

    timestamps = _timestamp_series(df, date_col)

    if strategy == "grouped_date":
        day_groups = timestamps.dt.normalize()
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=float(split_cfg.get("test_size", 0.2)),
            random_state=random_state,
        )
        train_idx, test_idx = next(
            splitter.split(df, df[label_col], groups=day_groups)
        )
        return (
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True),
        )

    if strategy == "date_cutoff":
        cutoff = split_cfg.get("cutoff")
        if cutoff is None:
            raise ValueError("`split.cutoff` is required for strategy: date_cutoff.")

        cutoff_ts = pd.to_datetime(cutoff, utc=True)
        train_mask = timestamps < cutoff_ts
        test_mask = timestamps >= cutoff_ts

    elif strategy == "date_ranges":
        train_cfg = split_cfg.get("train", {})
        test_cfg = split_cfg.get("test", {})

        if not train_cfg or not test_cfg:
            raise ValueError(
                "strategy: date_ranges requires both `split.train` and `split.test`."
            )

        train_mask = _date_mask(
            timestamps,
            start=train_cfg.get("start"),
            end=train_cfg.get("end"),
        )
        test_mask = _date_mask(
            timestamps,
            start=test_cfg.get("start"),
            end=test_cfg.get("end"),
        )

    else:
        raise ValueError(
            f"Unsupported split strategy {strategy!r}. "
            "Use grouped_random, grouped_date, date_cutoff, or date_ranges."
        )

    if (train_mask & test_mask).any():
        raise ValueError("The requested date split has overlapping train/test records.")

    train_df = df.loc[train_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)

    if train_df.empty:
        raise ValueError("Date split produced no training records.")
    if test_df.empty:
        raise ValueError("Date split produced no holdout records.")

    return train_df, test_df



def train_bootstrap_ensemble(
    base_name: str,
    base_model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    output_dir: Path,
    cfg: Dict,
) -> list[str]:
    bs_cfg = cfg.get("bootstrap_ensemble", {})
    if base_name != "logistic_regression" or not bs_cfg.get("enabled", False):
        return []

    n_models = int(bs_cfg.get("n_models", 10))
    rs = np.random.RandomState(int(cfg.get("random_state", 42)))
    group_values = groups.astype(str).to_numpy()
    unique_groups = np.unique(group_values)
    paths = []

    for i in range(n_models):
        sampled_groups = rs.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = []
        for g in sampled_groups:
            sampled_indices.extend(np.where(group_values == g)[0].tolist())
        sampled_indices = np.array(sampled_indices, dtype=int)

        model_i = clone(base_model)
        model_i.fit(X.iloc[sampled_indices], y.iloc[sampled_indices])

        out_path = output_dir / f"{base_name}_bootstrap_{i:03d}.joblib"
        joblib.dump(model_i, out_path)
        paths.append(str(out_path))

    return paths

def sample_no_fire_by_group(
    df: pd.DataFrame,
    label_column: str = "label",
    group_column: str = "hms_date",
    positive_label=1,
    negative_label=0,
    negative_per_positive: int = 50,
    random_state: int = 42,
    keep_all_negative_only_groups: bool = False,
) -> pd.DataFrame:
    """
    Retain all positive samples and randomly downsample negative samples
    independently within each group.

    Intended for sparse binary fire labels:
      positive_label = fire
      negative_label = no-fire

    Groups without positives are omitted by default because they can dominate
    the final training table. Set keep_all_negative_only_groups=True to retain
    them unchanged.
    """
    if negative_per_positive <= 0:
        raise ValueError("negative_per_positive must be > 0.")

    required_columns = {label_column, group_column}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(
            f"Cannot downsample no-fire samples; missing columns: "
            f"{sorted(missing)}"
        )

    sampled_parts = []
    rng = np.random.RandomState(random_state)

    for group_value, group_df in df.groupby(group_column, dropna=False):
        positives = group_df.loc[
            group_df[label_column] == positive_label
        ]
        negatives = group_df.loc[
            group_df[label_column] == negative_label
        ]

        other_labels = group_df.loc[
            ~group_df[label_column].isin([positive_label, negative_label])
        ]

        if positives.empty:
            if keep_all_negative_only_groups:
                sampled_parts.append(group_df)
            continue

        n_negative_target = min(
            len(negatives),
            negative_per_positive * len(positives),
        )

        if n_negative_target > 0:
            group_seed = int(rng.randint(0, np.iinfo(np.int32).max))

            sampled_negatives = negatives.sample(
                n=n_negative_target,
                replace=False,
                random_state=group_seed,
            )
        else:
            sampled_negatives = negatives.iloc[0:0]

        sampled_parts.extend([
            positives,
            sampled_negatives,
            other_labels,
        ])

        LOG.debug(
            "Group=%s: positives=%d, negatives=%d, retained_negatives=%d",
            group_value,
            len(positives),
            len(negatives),
            len(sampled_negatives),
        )

    if not sampled_parts:
        raise ValueError(
            "No samples remain after no-fire downsampling. "
            "Check positive_label, negative_label, and group_column."
        )

    sampled_df = pd.concat(sampled_parts, axis=0, ignore_index=True)

    sampled_df = sampled_df.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)

    n_before = len(df)
    n_after = len(sampled_df)

    LOG.info(
        "No-fire sampling reduced training table from %d to %d rows "
        "(%.2f%% retained).",
        n_before,
        n_after,
        100.0 * n_after / n_before,
    )

    return sampled_df



def sample_multiclass_by_group(
    df: pd.DataFrame,
    label_column: str = "label",
    group_column: str = "hms_date",
    max_samples_per_class: dict | None = None,
    random_state: int = 42,
    keep_unspecified_classes: bool = True,
) -> pd.DataFrame:
    """
    Randomly downsample each class independently within each group.

    max_samples_per_class maps class label -> maximum retained samples
    per group. Classes with fewer samples are preserved fully.

    Example:
        {
            0: 50000,
            1: 25000,
            2: 25000,
            3: 25000,
        }

    This performs no synthetic oversampling. It only reduces overrepresented
    classes while retaining all available samples below the specified cap.
    """
    if not max_samples_per_class:
        raise ValueError(
            "max_samples_per_class must be a non-empty mapping."
        )

    required_columns = {label_column, group_column}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(
            f"Cannot sample multiclass data; missing columns: "
            f"{sorted(missing)}"
        )

    normalized_caps = {
        str(label): int(max_count)
        for label, max_count in max_samples_per_class.items()
    }

    if any(cap <= 0 for cap in normalized_caps.values()):
        raise ValueError(
            "Every max_samples_per_class value must be positive."
        )

    rng = np.random.RandomState(random_state)
    sampled_parts = []

    for group_value, group_df in df.groupby(group_column, dropna=False):
        print("HERE GROUP", group_value)
        for label_value, class_df in group_df.groupby(label_column, dropna=False):
            print("HERE LABEL", label_value)
            label_key = str(int(label_value))

            print(label_key, normalized_caps.keys(), label_key not in normalized_caps)

            if label_key not in normalized_caps:
                if keep_unspecified_classes:
                    sampled_parts.append(class_df)
                continue

            max_count = normalized_caps[label_key]

            if len(class_df) <= max_count:
                sampled_parts.append(class_df)
                continue

            class_seed = int(rng.randint(0, np.iinfo(np.int32).max))

            sampled_parts.append(
                class_df.sample(
                    n=max_count,
                    replace=False,
                    random_state=class_seed,
                )
            )

            LOG.debug(
                "Group=%s, class=%s: retained %d of %d samples.",
                group_value,
                label_value,
                max_count,
                len(class_df),
            )

    if not sampled_parts:
        raise ValueError(
            "No samples remain after multiclass sampling."
        )

    sampled_df = pd.concat(sampled_parts, axis=0, ignore_index=True)

    return sampled_df.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)


def sample_training_data(
    train_df: pd.DataFrame,
    cfg: Dict,
) -> pd.DataFrame:
    """
    Optionally apply binary fire sampling or multi-class sampling.

    YAML:
      sampling:
        enabled: true
        mode: binary_negative_sampling

    or:

      sampling:
        enabled: true
        mode: multiclass_cap
    """
    sampling_cfg = cfg.get("sampling", {})

    if not sampling_cfg.get("enabled", False):
        return train_df

    mode = str(
        sampling_cfg.get("mode", "binary_negative_sampling")
    ).lower()

    label_column = sampling_cfg.get(
        "label_column",
        cfg.get("label_column", "label"),
    )
    group_column = sampling_cfg.get(
        "group_column",
        cfg.get("group_column", "pair_id"),
    )
    random_state = int(cfg.get("random_state", 42))

    if mode == "binary_negative_sampling":
        return sample_no_fire_by_group(
            df=train_df,
            label_column=label_column,
            group_column=group_column,
            positive_label=sampling_cfg.get("positive_label", 1),
            negative_label=sampling_cfg.get("negative_label", 0),
            negative_per_positive=int(
                sampling_cfg.get("negative_per_positive", 50)
            ),
            random_state=random_state,
            keep_all_negative_only_groups=bool(
                sampling_cfg.get(
                    "keep_all_negative_only_groups",
                    False,
                )
            ),
        )

    if mode == "multiclass_cap":
        return sample_multiclass_by_group(
            df=train_df,
            label_column=label_column,
            group_column=group_column,
            max_samples_per_class=sampling_cfg[
                "max_samples_per_class"
            ],
            random_state=random_state,
            keep_unspecified_classes=bool(
                sampling_cfg.get(
                    "keep_unspecified_classes",
                    True,
                )
            ),
        )

    raise ValueError(
        f"Unsupported sampling.mode='{mode}'. "
        "Supported values are: binary_negative_sampling, multiclass_cap."
    )




def cap_total_samples_stratified(
    df: pd.DataFrame,
    total_max_samples: int | None,
    label_column: str,
    group_column: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cap total rows while approximately preserving the joint distribution
    across (group, class). Every non-empty stratum retains at least one row.
    """
    if total_max_samples is None:
        return df

    total_max_samples = int(total_max_samples)

    if total_max_samples <= 0:
        raise ValueError("total_max_samples must be positive.")

    if len(df) <= total_max_samples:
        return df.reset_index(drop=True)

    strata_sizes = (
        df.groupby([group_column, label_column], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )

    n_strata = len(strata_sizes)

    if total_max_samples < n_strata:
        raise ValueError(
            f"total_max_samples={total_max_samples} is smaller than the "
            f"number of non-empty (group, label) strata={n_strata}. "
            "Increase total_max_samples or reduce the grouping resolution."
        )

    strata_sizes["raw_target"] = (
        total_max_samples * strata_sizes["n"] / strata_sizes["n"].sum()
    )

    strata_sizes["target_n"] = np.floor(
        strata_sizes["raw_target"]
    ).astype(int)

    # Preserve at least one example from every active group/class stratum.
    strata_sizes["target_n"] = strata_sizes["target_n"].clip(lower=1)

    remaining = total_max_samples - int(strata_sizes["target_n"].sum())

    if remaining > 0:
        strata_sizes["remainder"] = (
            strata_sizes["raw_target"] - strata_sizes["target_n"]
        )

        for idx in strata_sizes.sort_values(
            "remainder",
            ascending=False,
        ).index[:remaining]:
            strata_sizes.loc[idx, "target_n"] += 1

    elif remaining < 0:
        excess = abs(remaining)

        removable = strata_sizes.sort_values(
            "target_n",
            ascending=False,
        )

        for idx in removable.index:
            if excess <= 0:
                break

            current = int(strata_sizes.loc[idx, "target_n"])

            if current > 1:
                strata_sizes.loc[idx, "target_n"] -= 1
                excess -= 1

    targets = {
        (row[group_column], row[label_column]): int(row["target_n"])
        for _, row in strata_sizes.iterrows()
    }

    rng = np.random.RandomState(random_state)
    sampled_parts = []

    for (group_value, label_value), subset in df.groupby(
        [group_column, label_column],
        dropna=False,
    ):
        target_n = targets[(group_value, label_value)]

        if len(subset) <= target_n:
            sampled_parts.append(subset)
        else:
            sampled_parts.append(
                subset.sample(
                    n=target_n,
                    replace=False,
                    random_state=int(
                        rng.randint(0, np.iinfo(np.int32).max)
                    ),
                )
            )

    out = pd.concat(sampled_parts, ignore_index=True)

    # The rounding routine can leave a small difference. Trim only from
    # strata that have more than one retained row.
    if len(out) > total_max_samples:
        out = out.sample(
            n=total_max_samples,
            replace=False,
            random_state=random_state,
        )

    return out.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)







def train_and_save(df: pd.DataFrame, cfg: Dict, output_dir: Path) -> None:
    feats = feature_columns(df)
    label_col = cfg.get("label_column", "label")
    group_col = cfg.get("group_column", "pair_id")

    source_cfg = cfg.get("data_source", {})
    holdout_csv_path = source_cfg.get("holdout_csv_path")

    if holdout_csv_path:
        holdout_csv_path = Path(holdout_csv_path)

        if not holdout_csv_path.exists():
            raise FileNotFoundError(
                f"Configured holdout CSV does not exist: {holdout_csv_path}"
            )

        train_df = df.reset_index(drop=True)
        test_df = pd.read_csv(holdout_csv_path).reset_index(drop=True)

        LOG.info(
            "Using explicit CSV split: %d train rows, %d holdout rows.",
            len(train_df),
            len(test_df),
        )
    else:
        train_df, test_df = holdout_split(df, cfg)


    pre_sample_count = len(train_df)

    # Downsample only after the date/group holdout split. The holdout dataset must
    # remain unchanged so its metrics represent natural fire/no-fire prevalence.
    train_df = sample_training_data(train_df, cfg)

    sampling_cfg = cfg.get("sampling", {})

    train_df = cap_total_samples_stratified(
        df=train_df,
        total_max_samples=sampling_cfg.get("total_max_samples"),
        label_column=sampling_cfg.get(
            "label_column",
            cfg.get("label_column", "label"),
        ),
        group_column=sampling_cfg.get(
            "group_column",
            cfg.get("group_column", "pair_id"),
        ),
        random_state=int(cfg.get("random_state", 42)),
    )

 
    log_class_balance(train_df, label_col, "Post-sampling training")

    if not test_df.empty:
        log_class_balance(test_df, label_col, "Holdout")
 
    X_train = train_df[feats]
    y_train = train_df[label_col]
    groups_train = train_df[group_col]

    models = build_models(cfg)


    metrics = {}
    bootstrap_manifest = {}

    for name, model in models.items():
        LOG.info("Training %s", name)
        cv_metrics = evaluate_model(model, X_train, y_train, groups_train, cfg)

        model.fit(X_train, y_train)
        joblib.dump(model, output_dir / f"{name}.joblib")

        bootstrap_paths = train_bootstrap_ensemble(name, model, X_train, y_train, groups_train, output_dir, cfg)
        if bootstrap_paths:
            bootstrap_manifest[name] = bootstrap_paths

        model_metrics = {"cross_validation": cv_metrics}
        if not test_df.empty:
            X_test, y_test = test_df[feats], test_df[label_col]
            probs = model.predict_proba(X_test)
            classes = model.classes_
            preds = classes[np.argmax(probs, axis=1)]
            labels = list(classes)
            model_metrics["holdout"] = {
                "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
                "confusion_matrix": confusion_matrix(y_test, preds, labels=labels).tolist(),
                "labels": [x.item() if hasattr(x, "item") else x for x in labels],
            }
        metrics[name] = model_metrics

    train_df.to_csv(output_dir / "training_features.csv", index=False)
    if not test_df.empty:
        test_df.to_csv(output_dir / "holdout_features.csv", index=False)

    save_json(output_dir / "training_summary.json", {
        "metadata": {
            "feature_columns": feats,
            "feature_signature": canonical_feature_signature(cfg.get("features", {})),
            "n_train": len(train_df),
            "n_holdout": len(test_df),
            "models": list(models.keys()),
            "bootstrap_manifest": bootstrap_manifest,
            "label_column": label_col,
            "group_column": group_col,
            "negative_sampling": cfg.get("negative_sampling", {}),
        },
        "metrics": metrics,
    })


def canonical_feature_signature(features_cfg: dict[str, Any]) -> dict[str, Any]:
    return _canonicalize_config_value(features_cfg or {})


def log_class_balance(
    df: pd.DataFrame,
    label_column: str,
    label: str,
) -> None:
    counts = df[label_column].value_counts(dropna=False).to_dict()

    LOG.info(
        "%s class counts: %s",
        label,
        counts,
    )



def reconstruct_arrays(
    df: pd.DataFrame,
    value_cols: List[str],
    pair_meta: Dict[str, Dict[str, Any]],
    fill_value: float = DEFAULT_NODATA,
) -> Dict[str, Dict[str, np.ndarray]]:
    outputs = {pair_id: {} for pair_id in pair_meta}
    for pair_id, group in df.groupby("pair_id"):
        h = int(pair_meta[pair_id]["height"])
        w = int(pair_meta[pair_id]["width"])
        rows = group["row"].to_numpy(dtype=int)
        cols = group["col"].to_numpy(dtype=int)
        for col in value_cols:
            arr = np.full((h, w), fill_value, dtype=np.float32)
            arr[rows, cols] = group[col].to_numpy(dtype=np.float32)
            outputs[pair_id][col] = arr
    return outputs


def write_single_band_raster(path: Path, arr: np.ndarray, meta: Dict[str, Any], nodata: float = DEFAULT_NODATA) -> None:
    profile = {
        "driver": "GTiff",
        "height": int(meta["height"]),
        "width": int(meta["width"]),
        "count": 1,
        "dtype": "float32",
        "nodata": nodata,
        "transform": Affine(*meta["transform"]),
    }
    if meta.get("crs") is not None:
        profile["crs"] = meta["crs"]
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def write_raster_stack(
    df: pd.DataFrame,
    value_cols: List[str],
    pair_meta: Dict[str, Dict[str, Any]],
    output_dir: Path,
    prefix: str = "",
    fill_value: float = DEFAULT_NODATA,
) -> None:
    raster_dict = reconstruct_arrays(df[["pair_id", "row", "col"] + value_cols], value_cols, pair_meta, fill_value)
    for pair_id, col_arrays in raster_dict.items():
        meta = pair_meta[pair_id]
        for col, arr in col_arrays.items():
            stem = f"{prefix}_{pair_id}_{col}" if prefix else f"{pair_id}_{col}"
            write_single_band_raster(output_dir / f"{stem}.tif", arr, meta, nodata=fill_value)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)



