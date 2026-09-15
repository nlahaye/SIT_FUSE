#!/usr/bin/env python3
"""
Run inference for models trained by simple_classifiers.py.

Supports the same high-level configuration conventions as simple_classifiers:

    output_dir
    data_source:
      mode: build | csv
      csv_path: ...

    features:
      ...

    inference:
      model_path: ...
      training_summary_path: ...
      bootstrap_model_paths: [...]
      write_rasters: true | false

When data_source.mode == "build", this script calls build_inference_dataset(cfg)
from sit_fuse_utils_multiscale_final.

When data_source.mode == "csv", this script reads a precomputed feature CSV.
Raster output is only possible when spatial metadata is available through the
normal build_inference_dataset pathway or a compatible pair metadata source.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from classifier_utils import (
    DEFAULT_NODATA,
    EPS,
    bootstrap_model_probabilities,
    build_inference_dataset,
    canonical_feature_signature,
    compare_feature_signatures,
    enforce_feature_order,
    feature_columns,
    forest_tree_probabilities,
    load_json,
    load_yaml,
    margin_uncertainty,
    predictive_entropy,
    save_json,
    write_raster_stack,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOG = logging.getLogger("sit-fuse-classifier-inference")


def resolve_path(
    cfg: dict[str, Any],
    key: str,
    legacy_key: str,
) -> Path:
    """
    Resolve paths first from inference.<key>, then from old top-level keys.

    Example:
      inference:
        model_path: /models/random_forest.joblib

    Legacy:
      model_path: /models/random_forest.joblib
    """
    inference_cfg = cfg.get("inference", {})

    value = inference_cfg.get(key, cfg.get(legacy_key))
    if not value:
        raise ValueError(
            f"Missing inference.{key} or legacy top-level {legacy_key}."
        )

    return Path(value)


def get_bootstrap_paths(cfg: dict[str, Any]) -> list[str]:
    """
    Resolve bootstrap ensemble paths from newer or legacy config locations.
    """
    inference_cfg = cfg.get("inference", {})

    return list(
        inference_cfg.get(
            "bootstrap_model_paths",
            cfg.get("bootstrap_model_paths", []),
        )
    )


def get_write_rasters(cfg: dict[str, Any]) -> bool:
    """
    Default to raster output for build mode, unless explicitly disabled.
    """
    inference_cfg = cfg.get("inference", {})
    return bool(inference_cfg.get("write_rasters", True))


def get_raster_prefix(cfg: dict[str, Any]) -> str:
    return str(
        cfg.get("inference", {}).get(
            "raster_prefix",
            "inference",
        )
    )


def load_or_build_inference_dataframe(
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, Any]:
    """
    Build inference features from rasters or load a precomputed CSV.

    Returns:
      df: Feature table.
      pair_meta: Spatial metadata needed by write_raster_stack, or None
                 when reading an arbitrary feature CSV.
    """
    source_cfg = cfg.get("data_source", {})
    mode = str(source_cfg.get("mode", "build")).strip().lower()

    if mode == "build":
        LOG.info("Building inference dataset from configured raster inputs.")
        df, pair_meta = build_inference_dataset(cfg)
        return df, pair_meta

    if mode == "csv":
        csv_path_value = source_cfg.get("csv_path")
        if not csv_path_value:
            raise ValueError(
                "data_source.csv_path is required when data_source.mode='csv'."
            )

        csv_path = Path(csv_path_value)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Configured inference feature CSV does not exist: {csv_path}"
            )

        LOG.info("Loading precomputed inference features from %s", csv_path)
        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError(f"Inference feature CSV is empty: {csv_path}")

        LOG.info(
            "Loaded inference CSV with %d rows and %d columns.",
            len(df),
            len(df.columns),
        )

        return df, None

    raise ValueError(
        f"Unsupported data_source.mode='{mode}'. "
        "Supported values: build, csv."
    )


def validate_features(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    training_summary: dict[str, Any],
) -> pd.DataFrame:
    """
    Confirm that inference features match training features exactly and
    return them in the training-time column order.
    """
    expected_features = training_summary["metadata"]["feature_columns"]
    expected_signature = training_summary["metadata"].get(
        "feature_signature",
        {},
    )

    observed_features = feature_columns(df)

    missing_features = sorted(
        set(expected_features) - set(observed_features)
    )
    extra_features = sorted(
        set(observed_features) - set(expected_features)
    )

    if missing_features:
        raise ValueError(
            "Inference table is missing trained feature columns: "
            f"{missing_features}"
        )

    if extra_features:
        LOG.warning(
            "Inference table has extra feature columns not used by model: %s",
            extra_features,
        )

    expected_feature_cfg = expected_signature
    observed_feature_cfg = canonical_feature_signature(
        cfg.get("features", {})
    )

    if expected_feature_cfg:
        differences = compare_feature_signatures(
            expected_feature_cfg,
            observed_feature_cfg,
        )

        if differences:
            raise ValueError(
                "Inference feature configuration does not match training "
                f"configuration. Mismatched keys: {differences}"
            )

    return enforce_feature_order(
        df[expected_features],
        expected_features,
    )


def add_predictions_and_uncertainty(
    df: pd.DataFrame,
    X: pd.DataFrame,
    model,
    model_path: Path,
    bootstrap_paths: list[str],
) -> tuple[pd.DataFrame, list[Any], list[str]]:
    """
    Add predicted labels, per-class probabilities, and uncertainty proxies.

    Epistemic uncertainty is estimated either:
      1. Across random-forest trees, or
      2. Across bootstrap logistic-regression models.
    """
    output = df.copy()

    probabilities = model.predict_proba(X)
    classes = model.classes_
    predictions = classes[np.argmax(probabilities, axis=1)]

    output["predicted_label"] = predictions

    probability_columns = []
    for class_index, class_value in enumerate(classes):
        column_name = f"prob_{class_value}"
        output[column_name] = probabilities[:, class_index].astype(np.float32)
        probability_columns.append(column_name)

    output["aleatoric_entropy"] = predictive_entropy(
        probabilities
    ).astype(np.float32)

    output["confidence_margin_uncertainty"] = margin_uncertainty(
        probabilities
    ).astype(np.float32)

    epistemic_columns = []

    classifier = (
        model.named_steps["clf"]
        if hasattr(model, "named_steps")
        else model
    )

    if hasattr(classifier, "estimators_"):
        tree_probabilities = forest_tree_probabilities(model, X)
        mean_tree_probabilities = np.mean(tree_probabilities, axis=0)

        output["epistemic_variance"] = np.sum(
            np.var(tree_probabilities, axis=0),
            axis=1,
        ).astype(np.float32)

        tree_entropies = -np.sum(
            tree_probabilities * np.log(
                np.clip(tree_probabilities, EPS, 1.0)
            ),
            axis=2,
        )

        output["epistemic_mutual_information"] = (
            predictive_entropy(mean_tree_probabilities)
            - np.mean(tree_entropies, axis=0)
        ).astype(np.float32)

        epistemic_columns = [
            "epistemic_variance",
            "epistemic_mutual_information",
        ]

    elif bootstrap_paths:
        ensemble_probabilities, ensemble_classes = (
            bootstrap_model_probabilities(
                model_path,
                X,
                bootstrap_paths,
            )
        )

        if not np.array_equal(ensemble_classes, classes):
            raise ValueError(
                "Bootstrap ensemble classes do not match base-model classes."
            )

        mean_ensemble_probabilities = np.mean(
            ensemble_probabilities,
            axis=0,
        )

        output["epistemic_variance"] = np.sum(
            np.var(ensemble_probabilities, axis=0),
            axis=1,
        ).astype(np.float32)

        ensemble_entropies = -np.sum(
            ensemble_probabilities * np.log(
                np.clip(ensemble_probabilities, EPS, 1.0)
            ),
            axis=2,
        )

        output["epistemic_mutual_information"] = (
            predictive_entropy(mean_ensemble_probabilities)
            - np.mean(ensemble_entropies, axis=0)
        ).astype(np.float32)

        epistemic_columns = [
            "epistemic_variance",
            "epistemic_mutual_information",
        ]

    else:
        LOG.warning(
            "No random-forest tree ensemble or bootstrap ensemble was found; "
            "only entropy and confidence-margin uncertainty will be written."
        )

    value_columns = [
        "predicted_label",
        "aleatoric_entropy",
        "confidence_margin_uncertainty",
        *probability_columns,
        *epistemic_columns,
    ]

    return output, list(classes), value_columns


def run_inference(cfg: dict[str, Any]) -> None:
    output_dir = Path(cfg.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_path(
        cfg,
        key="model_path",
        legacy_key="model_path",
    )
    training_summary_path = resolve_path(
        cfg,
        key="training_summary_path",
        legacy_key="training_summary_path",
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    if not training_summary_path.exists():
        raise FileNotFoundError(
            f"Training summary does not exist: {training_summary_path}"
        )

    training_summary = load_json(training_summary_path)

    df, pair_meta = load_or_build_inference_dataframe(cfg)

    X = validate_features(
        df=df,
        cfg=cfg,
        training_summary=training_summary,
    )

    LOG.info(
        "Running inference for %d samples with %d validated features.",
        len(X),
        X.shape[1],
    )

    model = joblib.load(model_path)

    predicted_df, classes, value_columns = add_predictions_and_uncertainty(
        df=df,
        X=X,
        model=model,
        model_path=model_path,
        bootstrap_paths=get_bootstrap_paths(cfg),
    )

    prediction_csv = output_dir / "inference_predictions.csv"
    predicted_df.to_csv(prediction_csv, index=False)

    write_rasters = get_write_rasters(cfg)

    if write_rasters:
        if pair_meta is None:
            raise ValueError(
                "Raster writing was requested, but data_source.mode='csv' "
                "does not provide pair metadata. Either use "
                "data_source.mode='build' or set inference.write_rasters=false."
            )

        write_raster_stack(
            predicted_df,
            value_columns,
            pair_meta,
            output_dir,
            prefix=get_raster_prefix(cfg),
            fill_value=DEFAULT_NODATA,
        )

    save_json(
        output_dir / "inference_summary.json",
        {
            "model_path": str(model_path),
            "training_summary_path": str(training_summary_path),
            "classes": [
                value.item() if hasattr(value, "item") else value
                for value in classes
            ],
            "output_columns": value_columns,
            "n_samples": int(len(predicted_df)),
            "n_pairs": int(len(pair_meta)) if pair_meta is not None else None,
            "feature_columns": training_summary["metadata"][
                "feature_columns"
            ],
            "feature_signature": training_summary["metadata"].get(
                "feature_signature",
                {},
            ),
            "data_source_mode": cfg.get(
                "data_source",
                {},
            ).get("mode", "build"),
            "raster_output_written": write_rasters,
        },
    )

    LOG.info("Inference complete. Outputs written to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference using a model produced by simple_classifiers.py."
        )
    )
    parser.add_argument(
        "yaml_config",
        type=Path,
        help="Path to inference YAML configuration.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.yaml_config)
    run_inference(cfg)


if __name__ == "__main__":
    main()
