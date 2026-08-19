#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
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




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train final multi-scale simple classifiers and grouped-bootstrap logistic ensembles from SIT-FUSE rasters."
    )
    parser.add_argument("yaml_config", type=str)
    args = parser.parse_args()

    cfg = load_yaml(Path(args.yaml_config))
    output_dir = Path(cfg.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    df, _ = build_training_dataset(cfg)
    train_and_save(df, cfg, output_dir)
    LOG.info("Done. Outputs written to %s", output_dir)


if __name__ == "__main__":
    main()


