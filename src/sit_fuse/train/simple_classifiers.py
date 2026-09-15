#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


from sklearn.base import BaseEstimator
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, cross_val_predict


from sit_fuse.train.classifier_utils import (
    build_training_dataset,
    train_and_save,
    build_cv_splitter,
    build_models,
    evaluate_model,
    canonicalize_dates,
    canonical_feature_signature,
    normalize_holdout_dates,
    feature_columns,
    load_yaml,
    save_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger("sit-fuse-train-multiscale-final")



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


def load_or_build_training_dataframe(cfg: Dict) -> pd.DataFrame:
    """
    Load a precomputed feature table when configured; otherwise construct one
    through build_training_dataset().

    Config modes:
      data_source:
        mode: build

      data_source:
        mode: csv
        csv_path: /path/to/features.csv
    """
    source_cfg = cfg.get("data_source", {})
    mode = str(source_cfg.get("mode", "build")).strip().lower()

    if mode == "build":
        LOG.info("Building training dataset from configured raster inputs.")
        df, _ = build_training_dataset(cfg)
        return df

    if mode == "csv":
        csv_path = source_cfg.get("csv_path")
        if not csv_path:
            raise ValueError(
                "data_source.csv_path is required when data_source.mode is 'csv'."
            )

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Configured feature CSV does not exist: {csv_path}"
            )

        LOG.info("Loading precomputed training dataset from %s", csv_path)
        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError(f"Precomputed feature CSV is empty: {csv_path}")

        label_col = cfg.get("label_column", "label")
        group_col = cfg.get("group_column", "pair_id")

        required_columns = {label_col, group_col}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise KeyError(
                f"CSV is missing required columns: {sorted(missing_columns)}. "
                f"Available columns: {list(df.columns)}"
            )

        features = feature_columns(df)
        if not features:
            raise ValueError(
                "No feature columns were identified in the precomputed CSV. "
                "Confirm that its columns match the expectations of "
                "feature_columns()."
            )

        LOG.info(
            "Loaded %d rows, %d features, %d labels, and %d groups.",
            len(df),
            len(features),
            df[label_col].nunique(),
            df[group_col].nunique(),
        )

        return df

    raise ValueError(
        f"Unsupported data_source.mode='{mode}'. "
        "Supported values are: build, csv."
    )



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train final multi-scale simple classifiers and grouped-bootstrap logistic ensembles from SIT-FUSE rasters."
    )
    parser.add_argument("yaml_config", type=str)
    args = parser.parse_args()

    cfg = load_yaml(Path(args.yaml_config))
    output_dir = Path(cfg.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_or_build_training_dataframe(cfg)
    train_and_save(df, cfg, output_dir)
 
    LOG.info("Done. Outputs written to %s", output_dir)


if __name__ == "__main__":
    main()


