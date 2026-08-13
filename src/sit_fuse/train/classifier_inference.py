#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import joblib
import numpy as np

from sit_fuse_utils_multiscale_final import (
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger("sit-fuse-infer-multiscale-final")


def run_inference(cfg):
    output_dir = Path(cfg.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    df, pair_meta = build_inference_dataset(cfg)
    observed_feats = feature_columns(df)

    training_summary_path = Path(cfg["training_summary_path"])
    training_summary = load_json(training_summary_path)
    expected_feature_columns = training_summary["metadata"]["feature_columns"]
    expected_signature = training_summary["metadata"].get("feature_signature", {})
    observed_signature = canonical_feature_signature(cfg.get("features", {}))
    diffs = compare_feature_signatures(expected_signature, observed_signature)
    if diffs:
        raise ValueError(f"Inference feature config does not match training config; mismatched keys: {diffs}")

    X = enforce_feature_order(df[observed_feats], expected_feature_columns)
    model_path = Path(cfg["model_path"])
    model = joblib.load(model_path)

    probs = model.predict_proba(X)
    classes = model.classes_
    preds = classes[np.argmax(probs, axis=1)]

    df["predicted_label"] = preds.astype(np.float32)
    for i, cls in enumerate(classes):
        df[f"prob_{cls}"] = probs[:, i].astype(np.float32)
    df["aleatoric_entropy"] = predictive_entropy(probs).astype(np.float32)
    df["confidence_margin_uncertainty"] = margin_uncertainty(probs).astype(np.float32)

    epistemic_cols = []
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if hasattr(clf, "estimators_"):
        tree_probs = forest_tree_probabilities(model, X)
        mean_tree_probs = np.mean(tree_probs, axis=0)
        df["epistemic_variance"] = np.sum(np.var(tree_probs, axis=0), axis=1).astype(np.float32)
        df["epistemic_mutual_information"] = (
            predictive_entropy(mean_tree_probs)
            - np.mean(-np.sum(tree_probs * np.log(np.clip(tree_probs, EPS, 1.0)), axis=2), axis=0)
        ).astype(np.float32)
        epistemic_cols = ["epistemic_variance", "epistemic_mutual_information"]
    elif cfg.get("bootstrap_model_paths"):
        ens_probs, ens_classes = bootstrap_model_probabilities(model_path, X, cfg["bootstrap_model_paths"])
        if not np.array_equal(ens_classes, classes):
            raise ValueError("Bootstrap ensemble classes do not match the base model classes")
        mean_probs = np.mean(ens_probs, axis=0)
        df["epistemic_variance"] = np.sum(np.var(ens_probs, axis=0), axis=1).astype(np.float32)
        df["epistemic_mutual_information"] = (
            predictive_entropy(mean_probs)
            - np.mean(-np.sum(ens_probs * np.log(np.clip(ens_probs, EPS, 1.0)), axis=2), axis=0)
        ).astype(np.float32)
        epistemic_cols = ["epistemic_variance", "epistemic_mutual_information"]
    else:
        LOG.warning("No ensemble source found for epistemic uncertainty. Only aleatoric proxies will be written.")

    df.to_csv(output_dir / "inference_predictions.csv", index=False)
    value_cols = ["predicted_label", "aleatoric_entropy", "confidence_margin_uncertainty"] + [f"prob_{cls}" for cls in classes] + epistemic_cols
    write_raster_stack(df, value_cols, pair_meta, output_dir, prefix="inference", fill_value=DEFAULT_NODATA)

    save_json(output_dir / "inference_summary.json", {
        "model_path": str(model_path),
        "training_summary_path": str(training_summary_path),
        "classes": [c.item() if hasattr(c, "item") else c for c in classes],
        "output_columns": value_cols,
        "n_samples": int(len(df)),
        "n_pairs": int(len(pair_meta)),
        "validated_feature_columns": expected_feature_columns,
        "validated_feature_signature": expected_signature,
    })
    LOG.info("Inference complete. Outputs written to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run final SIT-FUSE multi-scale inference with validation and uncertainty proxies."
    )
    parser.add_argument("yaml_config", type=str)
    args = parser.parse_args()

    cfg = load_yaml(Path(args.yaml_config))
    run_inference(cfg)


if __name__ == "__main__":
    main()


