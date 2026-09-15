#!/usr/bin/env python3
"""
Build training matchups between context-free SIT-FUSE GeoTIFFs and timestamped
ADP smoke GeoTIFFs, then write a classifier YAML configuration and a CSV audit.

Expected ADP smoke raster filename format:
    smoke_YYYYMMDDTHHMMSSZ.tif

Example:
    smoke_20240728T000501Z.tif

Usage:
    python build_sit_fuse_adp_smoke_matchups.py config.yaml
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


LOG = logging.getLogger("build-sit-fuse-adp-smoke-matchups")
GEOTIFF_SUFFIXES = {".tif", ".tiff"}


@dataclass(frozen=True)
class RasterRecord:
    """A timestamped raster used as either SIT-FUSE input or ADP truth."""

    path: Path
    timestamp: dt.datetime
    source: str

    @property
    def date(self) -> dt.date:
        return self.timestamp.date()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Configuration must parse to a YAML mapping.")

    return config


def write_yaml(data: dict[str, Any], path: Path) -> None:
    """Write a YAML mapping, creating parent directories when necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            data,
            handle,
            sort_keys=False,
            default_flow_style=False,
        )


def parse_datetime(value: str) -> dt.datetime:
    """
    Parse a UTC-naive datetime from YAML.

    Accepted examples:
      - 2024-07-28
      - 2024-07-28T00:05:01
      - 2024-07-28T00:05:01Z
    """
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
        f"Could not parse {value!r}. "
        "Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ."
    )


def find_geotiffs(root: Path) -> Iterable[Path]:
    """Yield GeoTIFFs recursively below root."""
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in GEOTIFF_SUFFIXES:
            yield path


def parse_timestamp_from_filename(
    filename: str,
    patterns: list[str],
    source_name: str,
) -> dt.datetime | None:
    """
    Parse a timestamp from a filename using configured regex patterns.

    Each regular expression must contain one capture group with either:
      YYYYMMDDTHHMMSSZ
    or:
      YYYYMMDDTHHMMSS

    Example ADP pattern:
      (?i)^smoke_(\\d{8}T\\d{6}Z)\\.tiff?$
    """
    for pattern in patterns:
        match = re.search(pattern, filename, flags=re.IGNORECASE)

        if match is None:
            continue

        if not match.groups():
            raise ValueError(
                f"{source_name} timestamp pattern must have one capture group. "
                f"Invalid pattern: {pattern!r}"
            )

        token = match.group(1)

        for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S"):
            try:
                return dt.datetime.strptime(token, fmt)
            except ValueError:
                continue

        raise ValueError(
            f"Could not parse timestamp token {token!r} in filename {filename!r}."
        )

    return None


def is_context_free(
    path: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> bool:
    """
    Return True if the SIT-FUSE output name matches an included context-free
    pattern and does not match an excluded pattern.
    """
    filename = path.name

    if any(
        re.search(pattern, filename, flags=re.IGNORECASE)
        for pattern in exclude_patterns
    ):
        return False

    return any(
        re.search(pattern, filename, flags=re.IGNORECASE)
        for pattern in include_patterns
    )


def collect_sit_fuse_records(
    config: dict[str, Any],
    start: dt.datetime,
    end: dt.datetime,
) -> tuple[list[RasterRecord], list[dict[str, str]]]:
    """Find valid, timestamped context-free SIT-FUSE GeoTIFFs."""
    root = Path(config["directory"])
    timestamp_patterns = config["patterns"]["timestamp_patterns"]
    filters = config["context_free_filters"]

    records: list[RasterRecord] = []
    skipped: list[dict[str, str]] = []

    for path in find_geotiffs(root):
        if not is_context_free(
            path=path,
            include_patterns=filters["include_patterns"],
            exclude_patterns=filters.get("exclude_patterns", []),
        ):
            skipped.append(
                {
                    "path": str(path.resolve()),
                    "reason": "not_context_free_output",
                }
            )
            continue

        timestamp = parse_timestamp_from_filename(
            filename=path.name,
            patterns=timestamp_patterns,
            source_name="SIT-FUSE",
        )

        if timestamp is None:
            skipped.append(
                {
                    "path": str(path.resolve()),
                    "reason": "could_not_parse_sit_fuse_timestamp",
                }
            )
            continue

        if start <= timestamp <= end:
            records.append(
                RasterRecord(
                    path=path.resolve(),
                    timestamp=timestamp,
                    source="sit_fuse_context_free",
                )
            )

    records.sort(key=lambda record: (record.timestamp, str(record.path)))
    return records, skipped


def collect_adp_smoke_records(
    config: dict[str, Any],
    start: dt.datetime,
    end: dt.datetime,
) -> tuple[list[RasterRecord], list[dict[str, str]]]:
    """
    Find timestamped ADP smoke GeoTIFFs.

    Expected default filenames:
      smoke_YYYYMMDDTHHMMSSZ.tif
    """
    root = Path(config["directory"])
    patterns = config["patterns"]

    include_patterns = patterns.get(
        "include_patterns",
        [r"(?i)^smoke_"],
    )

    timestamp_patterns = patterns.get(
        "timestamp_patterns",
        [r"(?i)^smoke_(\d{8}T\d{6}Z)\.tiff?$"],
    )

    records: list[RasterRecord] = []
    skipped: list[dict[str, str]] = []

    for path in find_geotiffs(root):
        if not any(
            re.search(pattern, path.name, flags=re.IGNORECASE)
            for pattern in include_patterns
        ):
            skipped.append(
                {
                    "path": str(path.resolve()),
                    "reason": "not_adp_smoke_raster",
                }
            )
            continue

        timestamp = parse_timestamp_from_filename(
            filename=path.name,
            patterns=timestamp_patterns,
            source_name="ADP smoke",
        )

        if timestamp is None:
            skipped.append(
                {
                    "path": str(path.resolve()),
                    "reason": "could_not_parse_adp_timestamp",
                }
            )
            continue

        if start <= timestamp <= end:
            records.append(
                RasterRecord(
                    path=path.resolve(),
                    timestamp=timestamp,
                    source="adp_smoke",
                )
            )

    records.sort(key=lambda record: (record.timestamp, str(record.path)))
    return records, skipped


def seconds_offset(
    first: dt.datetime,
    second: dt.datetime,
) -> int:
    """Return absolute difference between two timestamps in seconds."""
    return int(abs((first - second).total_seconds()))


def choose_adp_match(
    sit_fuse_record: RasterRecord,
    adp_records: list[RasterRecord],
    max_time_difference_seconds: int,
    allow_adp_reuse: bool,
    used_adp_paths: set[str],
) -> tuple[RasterRecord | None, int | None]:
    """
    Select the temporally nearest ADP smoke raster within tolerance.

    Ties are resolved deterministically by:
      1. smallest absolute offset;
      2. earlier ADP timestamp;
      3. lexical file path.
    """
    candidates = [
        record
        for record in adp_records
        if allow_adp_reuse or str(record.path) not in used_adp_paths
    ]

    if not candidates:
        return None, None

    candidates.sort(
        key=lambda record: (
            seconds_offset(sit_fuse_record.timestamp, record.timestamp),
            record.timestamp,
            str(record.path),
        )
    )

    chosen = candidates[0]
    offset_seconds = seconds_offset(
        sit_fuse_record.timestamp,
        chosen.timestamp,
    )

    if offset_seconds > max_time_difference_seconds:
        return None, None

    return chosen, offset_seconds


def build_pair_id(
    sit_fuse: RasterRecord,
    adp: RasterRecord,
    group_by: str,
) -> str:
    """
    Construct a group identifier for holdout splitting and grouped CV.

    `adp_timestamp` is recommended because all SIT-FUSE scenes associated with
    the same ADP truth raster are kept in the same validation partition.
    """
    if group_by == "adp_timestamp":
        return f"smoke_{adp.timestamp:%Y%m%dT%H%M%SZ}"

    if group_by == "adp_day":
        return f"smoke_{adp.timestamp:%Y%m%d}"

    if group_by == "sit_fuse_timestamp":
        return f"smoke_{sit_fuse.timestamp:%Y%m%dT%H%M%SZ}"

    if group_by == "sit_fuse_day":
        return f"smoke_{sit_fuse.timestamp:%Y%m%d}"

    raise ValueError(
        "matching.group_by must be one of: "
        "adp_timestamp, adp_day, sit_fuse_timestamp, sit_fuse_day"
    )


def build_matchups(
    sit_fuse_records: list[RasterRecord],
    adp_records: list[RasterRecord],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Create classifier-ready SIT-FUSE/ADP matchups and a CSV-ready audit table.
    """
    matching = config["matching"]
    max_time_difference_seconds = int(
        matching.get("max_time_difference_seconds", 900)
    )
    allow_adp_reuse = bool(matching.get("allow_adp_reuse", True))
    group_by = str(
        matching.get("group_by", "adp_timestamp")
    ).lower()

    used_adp_paths: set[str] = set()
    matchups: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    for sit_fuse in sit_fuse_records:
        adp, offset_seconds = choose_adp_match(
            sit_fuse_record=sit_fuse,
            adp_records=adp_records,
            max_time_difference_seconds=max_time_difference_seconds,
            allow_adp_reuse=allow_adp_reuse,
            used_adp_paths=used_adp_paths,
        )

        if adp is None:
            audit.append(
                {
                    "sit_fuse_path": str(sit_fuse.path),
                    "sit_fuse_timestamp": sit_fuse.timestamp.strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "adp_smoke_path": "",
                    "adp_smoke_timestamp": "",
                    "time_offset_seconds": "",
                    "pair_id": "",
                    "status": "no_adp_match",
                    "notes": (
                        "No ADP smoke raster was found within "
                        f"{max_time_difference_seconds} seconds."
                    ),
                }
            )
            continue

        if not allow_adp_reuse:
            used_adp_paths.add(str(adp.path))

        pair_id = build_pair_id(
            sit_fuse=sit_fuse,
            adp=adp,
            group_by=group_by,
        )

        matchup = {
            "pair_id": pair_id,
            "label": "smoke",
            "product": "smoke",
            "sit_fuse_context_free": str(sit_fuse.path),
            "truth_adp_smoke": str(adp.path),
            "sit_fuse_timestamp": sit_fuse.timestamp.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "adp_smoke_timestamp": adp.timestamp.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "adp_smoke_date": adp.date.isoformat(),
            "time_offset_seconds": offset_seconds,
            "match_method": "nearest_timestamp",
        }

        matchups.append(matchup)

        audit.append(
            {
                "sit_fuse_path": str(sit_fuse.path),
                "sit_fuse_timestamp": matchup["sit_fuse_timestamp"],
                "adp_smoke_path": str(adp.path),
                "adp_smoke_timestamp": matchup["adp_smoke_timestamp"],
                "time_offset_seconds": offset_seconds,
                "pair_id": pair_id,
                "status": "matched",
                "notes": "",
            }
        )

    return matchups, audit


def write_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write a complete match/skip audit file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sit_fuse_path",
        "sit_fuse_timestamp",
        "adp_smoke_path",
        "adp_smoke_timestamp",
        "time_offset_seconds",
        "pair_id",
        "status",
        "notes",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def set_nested_value(
    data: dict[str, Any],
    dotted_key: str,
    value: Any,
) -> None:
    """
    Assign a nested YAML value by a dotted path.

    Example:
      set_nested_value(config, "training.matchups", records)
    """
    keys = dotted_key.split(".")
    current = data

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}

        if not isinstance(current[key], dict):
            raise ValueError(
                f"Cannot write {dotted_key!r}: {key!r} is not a mapping."
            )

        current = current[key]

    current[keys[-1]] = value


def render_classifier_config(
    base_classifier_config: dict[str, Any],
    matchups: list[dict[str, Any]],
    generator_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Copy the base classifier configuration and inject ADP matchup records.

    The current classifier utility should use:
      sit_fuse_context_free  -> SIT-FUSE cluster raster
      truth_adp_smoke        -> ADP label raster
      sit_fuse_timestamp     -> observation time
      pair_id                -> grouping key
    """
    classifier_config = copy.deepcopy(base_classifier_config)

    matchup_key = generator_config["output"].get(
        "matchup_key",
        "matchups",
    )
    set_nested_value(classifier_config, matchup_key, matchups)

    classifier_config.setdefault("label_column", "label")
    classifier_config.setdefault("group_column", "pair_id")

    classifier_config.setdefault("matchup_metadata", {})
    classifier_config["matchup_metadata"].update(
        {
            "sit_fuse_source": "context_free_geotiff",
            "truth_source": "adp_smoke_timestamped_geotiff",
            "n_matchups": len(matchups),
            "sample_schema": {
                "sit_fuse_feature_path": "sit_fuse_context_free",
                "truth_mask_path": "truth_adp_smoke",
                "label_column": "label",
                "group_column": "pair_id",
            },
        }
    )

    return classifier_config


def validate_config(config: dict[str, Any]) -> None:
    """Validate required top-level YAML keys before scanning files."""
    required_top_level = [
        "time_range",
        "sit_fuse",
        "adp_smoke",
        "matching",
        "classifier_config",
        "output",
    ]

    for key in required_top_level:
        if key not in config:
            raise ValueError(f"Missing required top-level config key: {key!r}")

    for key in ("start", "end"):
        if key not in config["time_range"]:
            raise ValueError(f"Missing time_range.{key}")

    if "directory" not in config["sit_fuse"]:
        raise ValueError("Missing sit_fuse.directory")
    if "patterns" not in config["sit_fuse"]:
        raise ValueError("Missing sit_fuse.patterns")
    if "context_free_filters" not in config["sit_fuse"]:
        raise ValueError("Missing sit_fuse.context_free_filters")

    if "directory" not in config["adp_smoke"]:
        raise ValueError("Missing adp_smoke.directory")
    if "patterns" not in config["adp_smoke"]:
        raise ValueError("Missing adp_smoke.patterns")

    if "base_yaml" not in config["classifier_config"]:
        raise ValueError("Missing classifier_config.base_yaml")

    for key in ("classifier_yaml", "audit_csv"):
        if key not in config["output"]:
            raise ValueError(f"Missing output.{key}")


def main(config_path: Path) -> None:
    """Run discovery, matching, classifier-config construction, and audit output."""
    config = load_yaml(config_path)
    validate_config(config)

    start = parse_datetime(config["time_range"]["start"])
    end = parse_datetime(config["time_range"]["end"])

    if end < start:
        raise ValueError(
            "time_range.end must be on or after time_range.start."
        )

    sit_fuse_records, sit_fuse_skipped = collect_sit_fuse_records(
        config=config["sit_fuse"],
        start=start,
        end=end,
    )

    adp_records, adp_skipped = collect_adp_smoke_records(
        config=config["adp_smoke"],
        start=start,
        end=end,
    )

    LOG.info(
        "Context-free SIT-FUSE rasters found: %d",
        len(sit_fuse_records),
    )
    LOG.info(
        "Timestamped ADP smoke rasters found: %d",
        len(adp_records),
    )

    matchups, audit = build_matchups(
        sit_fuse_records=sit_fuse_records,
        adp_records=adp_records,
        config=config,
    )

    for skipped in sit_fuse_skipped:
        audit.append(
            {
                "sit_fuse_path": skipped["path"],
                "sit_fuse_timestamp": "",
                "adp_smoke_path": "",
                "adp_smoke_timestamp": "",
                "time_offset_seconds": "",
                "pair_id": "",
                "status": "skipped_sit_fuse",
                "notes": skipped["reason"],
            }
        )

    for skipped in adp_skipped:
        audit.append(
            {
                "sit_fuse_path": "",
                "sit_fuse_timestamp": "",
                "adp_smoke_path": skipped["path"],
                "adp_smoke_timestamp": "",
                "time_offset_seconds": "",
                "pair_id": "",
                "status": "skipped_adp_smoke",
                "notes": skipped["reason"],
            }
        )

    base_classifier_config = load_yaml(
        Path(config["classifier_config"]["base_yaml"])
    )
    classifier_config = render_classifier_config(
        base_classifier_config=base_classifier_config,
        matchups=matchups,
        generator_config=config,
    )

    classifier_yaml_path = Path(config["output"]["classifier_yaml"])
    audit_csv_path = Path(config["output"]["audit_csv"])

    write_yaml(classifier_config, classifier_yaml_path)
    write_csv(audit, audit_csv_path)

    n_matched = sum(row["status"] == "matched" for row in audit)
    n_unmatched = sum(row["status"] == "no_adp_match" for row in audit)

    LOG.info("Matched SIT-FUSE/ADP smoke pairs: %d", n_matched)
    LOG.info("Unmatched SIT-FUSE rasters: %d", n_unmatched)
    LOG.info(
        "Classifier configuration written: %s",
        classifier_yaml_path.resolve(),
    )
    LOG.info("Audit CSV written: %s", audit_csv_path.resolve())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Generate context-free SIT-FUSE/ADP smoke matchups and a "
            "classifier-ready YAML configuration."
        )
    )
    parser.add_argument(
        "yaml_config",
        type=Path,
        help="Path to matchup-generator YAML configuration.",
    )

    arguments = parser.parse_args()

    try:
        main(arguments.yaml_config)
    except Exception as exc:
        LOG.exception("ADP matchup generation failed: %s", exc)
        sys.exit(1)



