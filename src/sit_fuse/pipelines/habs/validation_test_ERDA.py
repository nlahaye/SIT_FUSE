import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Canonical severity/bin class labels, in display order, matching the class index
# derived from each row tuple's third value (severity * 1000).
XLABEL_SEVERITY = ["0", "1", "2", "3", "4", "5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create macro-F1 box-and-whisker plots from HAB validation pickle output.")
    parser.add_argument("pickle_file", default=None, nargs="?")
    parser.add_argument("--output-dir", default=None, help="Directory for output plots. Defaults to the pickle file's parent directory.")
    return parser.parse_args()


def load_validation_payload(pickle_path: Path) -> Dict[str, Any]:
    with pickle_path.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict) or "validation" not in payload:
        raise ValueError("The pickle file does not contain a top-level 'validation' dictionary.")

    return payload


def iter_validation_rows(context_data: Dict[str, Any]):
    """Yield the raw (row_vector, bin_edges, severity*1000) tuples for one species/instrument/context.

    context_data['validation'] looks like [[tuple, tuple, tuple]] - a list containing one
    list of row-tuples - so each entry is unwrapped one level to get at the tuples.
    """
    validation_entries = context_data.get("validation", [])
    if isinstance(validation_entries, dict):
        validation_entries = [validation_entries]
    if not isinstance(validation_entries, (list, tuple)):
        validation_entries = [validation_entries]

    for entry in validation_entries:
        if isinstance(entry, (list, tuple)):
            for item in entry:
                yield item
        else:
            yield entry


def build_confusion_matrix(rows: List[Any]) -> Optional[np.ndarray]:
    """Stack per-row vectors into a full confusion matrix.

    Each row tuple is (row_vector, bin_edges/other, severity_class * 1000). The severity
    value (divided by 1000) gives the true-class index; row_vector is that class's row
    of predicted-class counts. Rows for classes with no matchups are simply absent, so
    missing rows are filled with zeros. If the same class index appears in more than one
    tuple, the row vectors are summed (they represent counts).
    """
    row_vectors: Dict[int, np.ndarray] = {}

    for entry in rows:
        if not isinstance(entry, tuple) or len(entry) < 3:
            continue

        raw_class = entry[2]
        if not isinstance(raw_class, (int, float, np.integer, np.floating)):
            continue
        class_index = int(round(float(raw_class) / 1000.0))

        vector = np.asarray(entry[0], dtype=float).ravel()
        if vector.size == 0:
            continue

        if class_index in row_vectors and row_vectors[class_index].size == vector.size:
            row_vectors[class_index] = row_vectors[class_index] + vector
        else:
            row_vectors[class_index] = vector

    if not row_vectors:
        return None

    num_classes = max(max(row_vectors) + 1, max(v.size for v in row_vectors.values()))
    matrix = np.zeros((num_classes, num_classes), dtype=float)
    for class_index, vector in row_vectors.items():
        matrix[class_index, : vector.size] = vector

    return matrix


def macro_f1_from_confusion_matrix(matrix: np.ndarray) -> Optional[float]:
    """Macro-average F1 across classes, skipping classes with zero true and predicted count."""
    f1_scores = []
    for i in range(matrix.shape[0]):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        if denom <= 0:
            continue
        f1_scores.append((2 * tp) / denom)

    if not f1_scores:
        return None

    return float(np.mean(f1_scores))


def per_class_f1_from_confusion_matrix(matrix: np.ndarray) -> Dict[int, float]:
    """F1 score for each individual class/severity row, keyed by class index.

    Classes with zero true and predicted count (denom <= 0) are omitted rather than
    scored, since F1 is undefined for a class that never appears either way.
    """
    scores: Dict[int, float] = {}
    for i in range(matrix.shape[0]):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        if denom <= 0:
            continue
        scores[i] = (2 * tp) / denom
    return scores


def collect_grouped_series(payload: Dict[str, Any]) -> Dict[str, Dict[str, List[float]]]:
    """Macro-F1 per species/instrument group, plus per-class F1 bucketed by severity level.

    species/instrument: one macro-F1 value per species/instrument/context combination,
    filed under that combination's species name and instrument name respectively.

    severity: every species/instrument/context combination contributes up to six values
    (one per severity class present in its confusion matrix), filed under the matching
    severity label in XLABEL_SEVERITY - so this groups across all combinations by which
    severity/bin class the F1 score belongs to, not by context name.
    """
    grouped: Dict[str, Dict[str, List[float]]] = {
        "species": {},
        "instrument": {},
        "severity": {label: [] for label in XLABEL_SEVERITY},
    }

    validation_payload = payload["validation"]
    for species_name, species_data in validation_payload.items():
        if not isinstance(species_data, dict):
            continue

        for instrument_name, instrument_data in species_data.items():
            if not isinstance(instrument_data, dict):
                continue

            for context_name, context_data in instrument_data.items():
                if not isinstance(context_data, dict):
                    continue

                rows = list(iter_validation_rows(context_data))
                confusion_matrix = build_confusion_matrix(rows)
                if confusion_matrix is None:
                    continue

                macro_score = macro_f1_from_confusion_matrix(confusion_matrix)
                if macro_score is not None:
                    grouped["species"].setdefault(species_name, []).append(macro_score)
                    grouped["instrument"].setdefault(instrument_name, []).append(macro_score)

                for class_index, class_score in per_class_f1_from_confusion_matrix(confusion_matrix).items():
                    if 0 <= class_index < len(XLABEL_SEVERITY):
                        label = XLABEL_SEVERITY[class_index]
                    else:
                        # Class index outside the expected 0-5 range; still keep the data
                        # rather than silently dropping it, filed under its own label.
                        label = str(class_index)
                        grouped["severity"].setdefault(label, [])
                    grouped["severity"][label].append(class_score)

    return grouped


def make_boxplot(data_by_label: Dict[str, List[float]], title: str, ylabel: str, output_path: Path) -> None:
    labels = list(data_by_label.keys())
    values = [data_by_label[label] for label in labels]

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(labels)), 6))
    ax.boxplot(values)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Group")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pickle_path = Path(args.pickle_file).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else pickle_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_validation_payload(pickle_path)
    grouped_series = collect_grouped_series(payload)

    for group_name, data_by_label in grouped_series.items():
        if not data_by_label:
            continue

        if group_name == "severity":
            title = "F1 score by severity level"
            ylabel = "F1 score (per severity class)"
        else:
            title = f"Macro F1 score by {group_name}"
            ylabel = "Macro F1 score"

        output_path = output_dir / f"validation_boxplot_{group_name}.png"
        make_boxplot(data_by_label, title, ylabel, output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()