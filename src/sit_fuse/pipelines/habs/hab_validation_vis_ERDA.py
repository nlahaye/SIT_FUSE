import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create box-and-whisker plots from HAB validation pickle output.")
    #parser.add_argument("pickle_file", nargs="?", default=r"D:\Users\charlotte.rhoads\Downloads\output_vals.pkl")
    parser.add_argument("pickle_file", default=None, nargs="?")
    parser.add_argument("--output-dir", default=None, help="Directory for output plots. Defaults to the pickle file's parent directory.")
    parser.add_argument(
        "--metric",
        choices=["severity", "count", "sum"],
        default="severity",
        help="Which value to plot from each validation tuple. 'severity' uses the third tuple value (scaled by 1000), 'count' uses the sum of the first array, and 'sum' is an alias for count.",
    )
    return parser.parse_args()


def load_validation_payload(pickle_path: Path) -> Dict[str, Any]:
    with pickle_path.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict) or "validation" not in payload:
        raise ValueError("The pickle file does not contain a top-level 'validation' dictionary.")

    return payload


def extract_metric(entry: Any, metric: str) -> Optional[float]:
    """Extract a numeric value from a validation row tuple.

    The validation entries are structured like:
    (array_of_counts, bin_edges, severity_class_scaled_by_1000)
    """

    if isinstance(entry, tuple):
        if metric == "severity" and len(entry) >= 3:
            severity = entry[2]
            if isinstance(severity, (int, float, np.integer, np.floating)):
                return float(severity) / 1000.0

        if metric in {"count", "sum"} and len(entry) >= 1:
            counts = np.asarray(entry[0], dtype=float)
            if counts.size:
                return float(counts.sum())

        for item in entry:
            value = extract_metric(item, metric)
            if value is not None:
                return value

    if isinstance(entry, np.ndarray):
        values = np.asarray(entry, dtype=float)
        if values.size:
            if metric in {"count", "sum"}:
                return float(values.sum())
            return float(values[-1])

    if isinstance(entry, list):
        values = [extract_metric(item, metric) for item in entry]
        values = [value for value in values if value is not None]
        if values:
            return float(np.mean(values))

    if isinstance(entry, (int, float, np.integer, np.floating)):
        return float(entry)

    return None


def iter_validation_rows(context_data: Dict[str, Any]):
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


def collect_grouped_series(payload: Dict[str, Any], metric: str) -> Dict[str, Dict[str, List[float]]]:
    grouped: Dict[str, Dict[str, List[float]]] = {"species": {}, "instrument": {}, "context": {}}

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

                values = [extract_metric(row, metric) for row in iter_validation_rows(context_data)]
                values = [value for value in values if value is not None]

                if not values:
                    continue

                grouped["species"].setdefault(species_name, []).extend(values)
                grouped["instrument"].setdefault(instrument_name, []).extend(values)
                grouped["context"].setdefault(context_name, []).extend(values)

    return grouped


def make_boxplot(data_by_label: Dict[str, List[float]], title: str, ylabel: str, output_path: Path) -> None:
    labels = list(data_by_label.keys())
    values = [data_by_label[label] for label in labels]

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(labels)), 6))
    ax.boxplot(values, label=labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Group")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pickle_path = Path(args.pickle_file).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else pickle_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_validation_payload(pickle_path)
    grouped_series = collect_grouped_series(payload, args.metric)

    for group_name, data_by_label in grouped_series.items():
        if not data_by_label:
            continue

        title = f"Validation values by {group_name}"
        ylabel = f"{args.metric.capitalize()} value"
        output_path = output_dir / f"validation_boxplot_{group_name}.png"
        make_boxplot(data_by_label, title, ylabel, output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
