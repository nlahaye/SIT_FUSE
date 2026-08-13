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

from sit_fuse_utils_multiscale_final import (
    build_training_dataset,
    canonical_feature_signature,
    feature_columns,
    load_yaml,
    save_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger("sit-fuse-train-multiscale-final")


def build_models(cfg: Dict) -> Dict[str, Pipeline]:
    rs = int(cfg.get("random_state", 42))
    models = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=rs)),
        ]),
        "decision_tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=rs)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=rs
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


def maybe_holdout_split(df: pd.DataFrame, cfg: Dict):
    split_cfg = cfg.get("split", {})
    if not split_cfg.get("enabled", False):
        return df, pd.DataFrame()
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=float(split_cfg.get("test_size", 0.2)),
        random_state=int(cfg.get("random_state", 42))
    )
    idx_train, idx_test = next(
        gss.split(
            df,
            df[split_cfg.get("label_column", "label")],
            groups=df[split_cfg.get("group_column", "pair_id")]
        )
    )
    return df.iloc[idx_train].reset_index(drop=True), df.iloc[idx_test].reset_index(drop=True)


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


def train_and_save(df: pd.DataFrame, cfg: Dict, output_dir: Path) -> None:
    feats = feature_columns(df)
    label_col = cfg.get("label_column", "label")
    group_col = cfg.get("group_column", "pair_id")
    train_df, test_df = maybe_holdout_split(df, cfg)
    X_train, y_train, groups_train = train_df[feats], train_df[label_col], train_df[group_col]
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
        },
        "metrics": metrics,
    })


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


