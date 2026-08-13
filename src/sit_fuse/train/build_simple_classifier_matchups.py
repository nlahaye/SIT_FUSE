#!/usr/bin/env python3
"""
Build training matchups between context-free SIT-FUSE GeoTIFFs and daily
NOAA HMS smoke/fire raster masks, then write a classifier YAML config and
a CSV audit table.

Usage:
    python build_simple_classifier_matchups.py config.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


LOG = logging.getLogger("build-simple-classifier-matchups")
GEOTIFF_SUFFIXES = {".tif", ".tiff"}


@dataclass(frozen=True)
class RasterRecord:
    path: Path
    product: str
    timestamp: dt.datetime
    date: dt.date
    source: str


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Configuration must parse to a YAML mapping.")

    return cfg


def write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def parse_datetime(value: str) -> dt.datetime:
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return dt.datetime.strptime(value, fmt)
        except ValueError:
            continue

    raise ValueError(
        f"Could not parse '{value}'. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ."
    )


def daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)


def find_geotiffs(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in GEOTIFF_SUFFIXES:
            yield path


def infer_product(
    filename: str,
    smoke_patterns: list[str],
    fire_patterns: list[str],
) -> str | None:
    text = filename.lower()

    for pattern in smoke_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return "smoke"

    for pattern in fire_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return "fire"

    return None


def parse_filename_time(
    filename: str,
    timestamp_patterns: list[str],
    daily_date_patterns: list[str],
    daily_time_policy: str,
) -> dt.datetime | None:
    """
    Return UTC-naive datetime. All matching in this pipeline assumes UTC.

    Timestamp patterns must contain one capture group holding:
      YYYYMMDDTHHMMSSZ or YYYYMMDDTHHMMSS.

    Daily date patterns must contain one capture group holding:
      YYYYMMDD.
    """
    for pattern in timestamp_patterns:
        match = re.search(pattern, filename, flags=re.IGNORECASE)
        if not match:
            continue

        token = match.group(1)
        for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S"):
            try:
                return dt.datetime.strptime(token, fmt)
            except ValueError:
                continue

    for pattern in daily_date_patterns:
        match = re.search(pattern, filename, flags=re.IGNORECASE)
        if not match:
            continue

        try:
            day = dt.datetime.strptime(match.group(1), "%Y%m%d")
        except ValueError:
            continue

        if daily_time_policy == "midday":
            return day.replace(hour=12)
        if daily_time_policy == "end_of_day":
            return day.replace(hour=23, minute=59, second=59)

        return day

    return None


def collect_records(
    *,
    root: Path,
    source: str,
    smoke_patterns: list[str],
    fire_patterns: list[str],
    timestamp_patterns: list[str],
    daily_date_patterns: list[str],
    daily_time_policy: str,
    start: dt.datetime,
    end: dt.datetime,
    include_if_same_day: bool,
) -> tuple[list[RasterRecord], list[dict[str, str]]]:
    records: list[RasterRecord] = []
    skipped: list[dict[str, str]] = []

    for path in find_geotiffs(root):
        product = infer_product(path.name, smoke_patterns, fire_patterns)
        if product is None:
            skipped.append(
                {
                    "path": str(path.resolve()),
                    "reason": "could_not_infer_smoke_or_fire_product",
                }
            )
            continue

        timestamp = parse_filename_time(
            filename=path.name,
            timestamp_patterns=timestamp_patterns,
            daily_date_patterns=daily_date_patterns,
            daily_time_policy=daily_time_policy,
        )
        if timestamp is None:
            skipped.append(
                {
                    "path": str(path.resolve()),
                    "reason": "could_not_parse_timestamp",
                }
            )
            continue

        in_range = start <= timestamp <= end
        same_day_in_range = start.date() <= timestamp.date() <= end.date()

        if not in_range and not (include_if_same_day and same_day_in_range):
            continue

        records.append(
            RasterRecord(
                path=path.resolve(),
                product=product,
                timestamp=timestamp,
                date=timestamp.date(),
                source=source,
            )
        )

    records.sort(key=lambda x: (x.product, x.timestamp, str(x.path)))
    return records, skipped


def is_context_free(path: Path, include_patterns: list[str], exclude_patterns: list[str]) -> bool:
    text = path.name.lower()

    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in exclude_patterns):
        return False

    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in include_patterns
    )


def collect_context_free_sit_fuse_records(
    cfg: dict[str, Any],
    start: dt.datetime,
    end: dt.datetime,
) -> tuple[list[RasterRecord], list[dict[str, str]]]:
    root = Path(cfg["directory"])
    filters = cfg["context_free_filters"]

    candidates, skipped = collect_records(
        root=root,
        source="sit_fuse_context_free",
        smoke_patterns=cfg["patterns"]["smoke"],
        fire_patterns=cfg["patterns"]["fire"],
        timestamp_patterns=cfg["patterns"]["timestamp_patterns"],
        daily_date_patterns=cfg["patterns"].get("daily_date_patterns", []),
        daily_time_policy=cfg.get("daily_time_policy", "start_of_day"),
        start=start,
        end=end,
        include_if_same_day=False,
    )

    records = []
    for record in candidates:
        if is_context_free(
            path=record.path,
            include_patterns=filters["include_patterns"],
            exclude_patterns=filters.get("exclude_patterns", []),
        ):
            records.append(record)
        else:
            skipped.append(
                {
                    "path": str(record.path),
                    "reason": "not_context_free_output",
                }
            )

    return records, skipped


def make_hms_lookup(records: list[RasterRecord]) -> dict[tuple[dt.date, str], list[RasterRecord]]:
    lookup: dict[tuple[dt.date, str], list[RasterRecord]] = {}

    for record in records:
        lookup.setdefault((record.date, record.product), []).append(record)

    for key in lookup:
        lookup[key].sort(key=lambda x: (x.timestamp, str(x.path)))

    return lookup


def seconds_offset(a: dt.datetime, b: dt.datetime) -> int:
    return int(abs((a - b).total_seconds()))


def choose_hms_match(
    sit_fuse_record: RasterRecord,
    hms_lookup: dict[tuple[dt.date, str], list[RasterRecord]],
    max_offset_seconds: int,
    allow_reuse: bool,
    used_hms_paths: set[str],
) -> RasterRecord | None:
    candidates = hms_lookup.get((sit_fuse_record.date, sit_fuse_record.product), [])

    if not allow_reuse:
        candidates = [x for x in candidates if str(x.path) not in used_hms_paths]

    if not candidates:
        return None

    chosen = min(
        candidates,
        key=lambda x: (seconds_offset(sit_fuse_record.timestamp, x.timestamp), str(x.path)),
    )

    if seconds_offset(sit_fuse_record.timestamp, chosen.timestamp) > max_offset_seconds:
        return None

    return chosen


def build_pair_id(
    sit_fuse: RasterRecord,
    hms: RasterRecord,
    group_by: str,
) -> str:
    if group_by == "hms_day":
        return f"{hms.product}_{hms.date:%Y%m%d}"

    if group_by == "sit_fuse_day":
        return f"{sit_fuse.product}_{sit_fuse.date:%Y%m%d}"

    if group_by == "product":
        return sit_fuse.product

    return (
        f"{sit_fuse.product}_"
        f"{sit_fuse.timestamp:%Y%m%dT%H%M%SZ}_"
        f"{hms.date:%Y%m%d}"
    )


def build_matchups(
    sit_fuse_records: list[RasterRecord],
    hms_records: list[RasterRecord],
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hms_lookup = make_hms_lookup(hms_records)
    used_hms_paths: set[str] = set()

    matching_cfg = cfg["matching"]
    max_offset_seconds = int(matching_cfg.get("max_time_difference_seconds", 86400))
    allow_hms_reuse = bool(matching_cfg.get("allow_hms_reuse", True))
    group_by = matching_cfg.get("group_by", "hms_day")

    matchups: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    for sf in sit_fuse_records:
        hms = choose_hms_match(
            sit_fuse_record=sf,
            hms_lookup=hms_lookup,
            max_offset_seconds=max_offset_seconds,
            allow_reuse=allow_hms_reuse,
            used_hms_paths=used_hms_paths,
        )

        if hms is None:
            audit.append(
                {
                    "sit_fuse_path": str(sf.path),
                    "sit_fuse_product": sf.product,
                    "sit_fuse_timestamp": sf.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "hms_path": "",
                    "hms_product": "",
                    "hms_timestamp": "",
                    "time_offset_seconds": "",
                    "pair_id": "",
                    "status": "no_hms_match",
                    "notes": "No HMS truth raster found for this product and date.",
                }
            )
            continue

        if not allow_hms_reuse:
            used_hms_paths.add(str(hms.path))

        pair_id = build_pair_id(sf, hms, group_by)

        matchup = {
            "pair_id": pair_id,
            "label": sf.product,
            "product": sf.product,
            "sit_fuse_context_free": str(sf.path),
            "truth_hms": str(hms.path),
            "sit_fuse_timestamp": sf.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hms_timestamp": hms.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hms_date": hms.date.isoformat(),
            "time_offset_seconds": seconds_offset(sf.timestamp, hms.timestamp),
        }
        matchups.append(matchup)

        audit.append(
            {
                "sit_fuse_path": str(sf.path),
                "sit_fuse_product": sf.product,
                "sit_fuse_timestamp": sf.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "hms_path": str(hms.path),
                "hms_product": hms.product,
                "hms_timestamp": hms.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "time_offset_seconds": seconds_offset(sf.timestamp, hms.timestamp),
                "pair_id": pair_id,
                "status": "matched",
                "notes": "",
            }
        )

    return matchups, audit


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "sit_fuse_path",
        "sit_fuse_product",
        "sit_fuse_timestamp",
        "hms_path",
        "hms_product",
        "hms_timestamp",
        "time_offset_seconds",
        "pair_id",
        "status",
        "notes",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def set_nested_value(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """
    Assign value to a nested dictionary using a dotted path.

    Example:
        set_nested_value(cfg, "training.matchups", records)
    produces:
        cfg["training"]["matchups"] = records
    """
    keys = dotted_key.split(".")
    current = data

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        if not isinstance(current[key], dict):
            raise ValueError(
                f"Cannot create nested config key '{dotted_key}': "
                f"'{key}' is not a dictionary."
            )
        current = current[key]

    current[keys[-1]] = value


def render_classifier_config(
    base_classifier_cfg: dict[str, Any],
    matchups: list[dict[str, Any]],
    generator_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Write generic matchup records to the config passed to simple_classifiers.py.

    Adapt this function if build_training_dataset() expects a different schema.
    """
    cfg = copy.deepcopy(base_classifier_cfg)

    out_cfg = generator_cfg["output"]
    matchup_key = out_cfg.get("matchup_key", "matchups")

    set_nested_value(cfg, matchup_key, matchups)

    cfg.setdefault("label_column", "label")
    cfg.setdefault("group_column", "pair_id")

    cfg.setdefault("matchup_metadata", {})
    cfg["matchup_metadata"].update(
        {
            "sit_fuse_source": "context_free_geotiff",
            "truth_source": "hms",
            "n_matchups": len(matchups),
            "sample_schema": {
                "sit_fuse_feature_path": "sit_fuse_context_free",
                "truth_mask_path": "truth_hms",
                "label_column": "label",
                "group_column": "pair_id",
            },
        }
    )

    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required = [
        "time_range",
        "sit_fuse",
        "hms",
        "matching",
        "classifier_config",
        "output",
    ]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing required top-level config key: '{key}'.")

    for key in ("start", "end"):
        if key not in cfg["time_range"]:
            raise ValueError(f"Missing time_range.{key}")

    sit_fuse_required = ["directory", "context_free_filters", "patterns"]
    for key in sit_fuse_required:
        if key not in cfg["sit_fuse"]:
            raise ValueError(f"Missing sit_fuse.{key}")

    hms_required = ["directory", "patterns"]
    for key in hms_required:
        if key not in cfg["hms"]:
            raise ValueError(f"Missing hms.{key}")

    output_required = ["classifier_yaml", "audit_csv"]
    for key in output_required:
        if key not in cfg["output"]:
            raise ValueError(f"Missing output.{key}")


def main(config_path: str) -> None:
    cfg = load_yaml(Path(config_path))
    validate_config(cfg)

    start = parse_datetime(cfg["time_range"]["start"])
    end = parse_datetime(cfg["time_range"]["end"])
    if end < start:
        raise ValueError("time_range.end must be on or after time_range.start.")

    LOG.info("Finding context-free SIT-FUSE GeoTIFFs.")
    sit_fuse_records, sit_fuse_skipped = collect_context_free_sit_fuse_records(
        cfg=cfg["sit_fuse"],
        start=start,
        end=end,
    )

    LOG.info("Finding HMS smoke/fire truth GeoTIFFs.")
    hms_cfg = cfg["hms"]
    hms_records, hms_skipped = collect_records(
        root=Path(hms_cfg["directory"]),
        source="hms",
        smoke_patterns=hms_cfg["patterns"]["smoke"],
        fire_patterns=hms_cfg["patterns"]["fire"],
        timestamp_patterns=hms_cfg["patterns"].get("timestamp_patterns", []),
        daily_date_patterns=hms_cfg["patterns"]["daily_date_patterns"],
        daily_time_policy=hms_cfg.get("daily_time_policy", "midday"),
        start=start,
        end=end,
        include_if_same_day=True,
    )

    LOG.info("SIT-FUSE context-free rasters found: %d", len(sit_fuse_records))
    LOG.info("HMS rasters found: %d", len(hms_records))

    matchups, audit = build_matchups(
        sit_fuse_records=sit_fuse_records,
        hms_records=hms_records,
        cfg=cfg,
    )

    for item in sit_fuse_skipped:
        audit.append(
            {
                "sit_fuse_path": item["path"],
                "sit_fuse_product": "",
                "sit_fuse_timestamp": "",
                "hms_path": "",
                "hms_product": "",
                "hms_timestamp": "",
                "time_offset_seconds": "",
                "pair_id": "",
                "status": "skipped_sit_fuse",
                "notes": item["reason"],
            }
        )

    for item in hms_skipped:
        audit.append(
            {
                "sit_fuse_path": "",
                "sit_fuse_product": "",
                "sit_fuse_timestamp": "",
                "hms_path": item["path"],
                "hms_product": "",
                "hms_timestamp": "",
                "time_offset_seconds": "",
                "pair_id": "",
                "status": "skipped_hms",
                "notes": item["reason"],
            }
        )

    base_classifier_config_path = Path(cfg["classifier_config"]["base_yaml"])
    base_classifier_cfg = load_yaml(base_classifier_config_path)

    classifier_cfg = render_classifier_config(
        base_classifier_cfg=base_classifier_cfg,
        matchups=matchups,
        generator_cfg=cfg,
    )

    classifier_yaml_path = Path(cfg["output"]["classifier_yaml"])
    audit_csv_path = Path(cfg["output"]["audit_csv"])

    write_yaml(classifier_cfg, classifier_yaml_path)
    write_csv(audit, audit_csv_path)

    n_matched = sum(row["status"] == "matched" for row in audit)
    n_unmatched = sum(row["status"] == "no_hms_match" for row in audit)

    LOG.info("Matched SIT-FUSE/HMS pairs: %d", n_matched)
    LOG.info("Unmatched SIT-FUSE files: %d", n_unmatched)
    LOG.info("Classifier config: %s", classifier_yaml_path.resolve())
    LOG.info("Matchup audit CSV: %s", audit_csv_path.resolve())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Generate context-free SIT-FUSE/HMS matchups and a training "
            "configuration for simple_classifiers.py."
        )
    )
    parser.add_argument(
        "yaml_config",
        type=str,
        help="Path to matchup-generator YAML configuration.",
    )
    args = parser.parse_args()

    try:
        main(args.yaml_config)
    except Exception as exc:
        LOG.exception("Matchup generation failed: %s", exc)
        sys.exit(1)
