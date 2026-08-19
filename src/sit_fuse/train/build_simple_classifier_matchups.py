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
    timestamp: dt.datetime | None
    date: dt.date
    source: str

    # For SIT-FUSE rasters, these are normally None.
    # For time-sliced HMS truth rasters, both are populated.
    interval_start: dt.datetime | None = None
    interval_end: dt.datetime | None = None

    @property
    def interval_midpoint(self) -> dt.datetime | None:
        if self.interval_start is None or self.interval_end is None:
            return None
        return self.interval_start + (
            self.interval_end - self.interval_start
        ) / 2




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

def parse_hms_interval_from_filename(
    filename: str,
    interval_patterns: list[str],
) -> tuple[dt.datetime, dt.datetime] | None:
    """
    Parse HMS time-sliced raster intervals from filename patterns.

    Each pattern must contain two capture groups:
      1. start timestamp: YYYYMMDDTHHMMSSZ or YYYYMMDDTHHMMSS
      2. end timestamp:   YYYYMMDDTHHMMSSZ or YYYYMMDDTHHMMSS
    """
    for pattern in interval_patterns:
        match = re.search(pattern, filename, flags=re.IGNORECASE)
        if match is None:
            continue

        if len(match.groups()) < 2:
            raise ValueError(
                "Each hms interval_timestamp_patterns entry must contain "
                "two capture groups: start timestamp and end timestamp. "
                f"Pattern: {pattern!r}"
            )

        start_token, end_token = match.group(1), match.group(2)

        for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S"):
            try:
                start = dt.datetime.strptime(start_token, fmt)
                break
            except ValueError:
                start = None
        else:
            start = None

        for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S"):
            try:
                end = dt.datetime.strptime(end_token, fmt)
                break
            except ValueError:
                end = None
        else:
            end = None

        if start is None or end is None:
            raise ValueError(
                f"Could not parse interval tokens {start_token!r}, {end_token!r} "
                f"from {filename!r}."
            )

        if end < start:
            raise ValueError(
                f"Invalid HMS interval in {filename!r}: "
                f"end {end} precedes start {start}."
            )

        return start, end

    return None

def collect_hms_truth_records(
    cfg: dict[str, Any],
    start: dt.datetime,
    end: dt.datetime,
) -> tuple[list[RasterRecord], list[dict[str, str]]]:
    """
    Collect time-sliced HMS truth rasters.

    Time-sliced filenames are preferred. Legacy daily HMS rasters remain
    supported via the existing parse_filename_time() fallback.
    """
    root = Path(cfg["directory"])
    patterns = cfg["patterns"]
    interval_patterns = patterns.get("interval_timestamp_patterns", [])
    daily_time_policy = cfg.get("daily_time_policy", "midday")

    records: list[RasterRecord] = []
    skipped: list[dict[str, str]] = []

    for path in find_geotiffs(root):
        product = infer_product(
            path.name,
            patterns["smoke"],
            patterns["fire"],
        )
        if product is None:
            skipped.append({
                "path": str(path),
                "reason": "could_not_infer_smoke_or_fire_product",
            })
            continue

        interval = parse_hms_interval_from_filename(
            path.name,
            interval_patterns,
        )

        if interval is not None:
            interval_start, interval_end = interval

            # Preserve intervals that overlap the requested temporal range.
            if interval_end < start or interval_start > end:
                continue

            records.append(
                RasterRecord(
                    path=path.resolve(),
                    product=product,
                    timestamp=interval_start,
                    date=interval_start.date(),
                    source="hms_interval",
                    interval_start=interval_start,
                    interval_end=interval_end,
                )
            )
            continue

        # Backward-compatible daily-raster fallback.
        timestamp = parse_filename_time(
            filename=path.name,
            timestamp_patterns=patterns.get("timestamp_patterns", []),
            daily_date_patterns=patterns.get("daily_date_patterns", []),
            daily_time_policy=daily_time_policy,
        )

        if timestamp is None:
            skipped.append({
                "path": str(path),
                "reason": "could_not_parse_truth_interval_or_timestamp",
            })
            continue

        if not (start.date() <= timestamp.date() <= end.date()):
            continue

        records.append(
            RasterRecord(
                path=path.resolve(),
                product=product,
                timestamp=timestamp,
                date=timestamp.date(),
                source="hms_daily",
                interval_start=None,
                interval_end=None,
            )
        )

    records.sort(
        key=lambda record: (
            record.product,
            record.interval_start or record.timestamp,
            record.interval_end or record.timestamp,
            str(record.path),
        )
    )
    return records, skipped



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


def seconds_to_interval(
    timestamp: dt.datetime,
    interval_start: dt.datetime,
    interval_end: dt.datetime,
) -> int:
    """
    Distance from a timestamp to a closed interval in seconds.

    Returns zero if the timestamp is inside the interval.
    """
    if interval_start <= timestamp <= interval_end:
        return 0
    if timestamp < interval_start:
        return int((interval_start - timestamp).total_seconds())
    return int((timestamp - interval_end).total_seconds())


def interval_duration_seconds(record: RasterRecord) -> int:
    if record.interval_start is None or record.interval_end is None:
        return 24 * 60 * 60
    return int((record.interval_end - record.interval_start).total_seconds())


def choose_hms_interval_match(
    sitfuse_record: RasterRecord,
    hms_records: list[RasterRecord],
    max_offset_seconds: int,
    allow_hms_reuse: bool,
    used_hms_paths: set[str],
    allow_nearest_interval: bool = True,
) -> tuple[RasterRecord | None, int | None, str | None]:
    """
    Match a SIT-FUSE timestamp to an HMS truth interval.

    Priority:
      1. Same-product truth intervals containing the SIT-FUSE timestamp.
      2. If configured, nearest same-product truth interval within tolerance.
      3. Legacy daily rasters are interpreted as a daily surrogate interval.
    """
    candidates = [
        record
        for record in hms_records
        if record.product == sitfuse_record.product
        and (allow_hms_reuse or str(record.path) not in used_hms_paths)
    ]

    if not candidates:
        return None, None, None

    def record_interval(record: RasterRecord) -> tuple[dt.datetime, dt.datetime]:
        if record.interval_start is not None and record.interval_end is not None:
            return record.interval_start, record.interval_end

        # Legacy daily rasters: treat as a one-day validity interval.
        day_start = dt.datetime.combine(record.date, dt.time.min)
        day_end = dt.datetime.combine(record.date, dt.time.max)
        return day_start, day_end

    containing = []
    for record in candidates:
        start_time, end_time = record_interval(record)
        if start_time <= sitfuse_record.timestamp <= end_time:
            midpoint = start_time + (end_time - start_time) / 2
            midpoint_distance = abs(
                (sitfuse_record.timestamp - midpoint).total_seconds()
            )
            containing.append(
                (
                    midpoint_distance,
                    interval_duration_seconds(record),
                    str(record.path),
                    record,
                )
            )

    if containing:
        containing.sort(key=lambda item: item[:3])
        chosen = containing[0][3]
        return chosen, 0, "contains_timestamp"

    if not allow_nearest_interval:
        return None, None, None

    nearest = []
    for record in candidates:
        start_time, end_time = record_interval(record)
        distance = seconds_to_interval(
            sitfuse_record.timestamp,
            start_time,
            end_time,
        )
        nearest.append(
            (
                distance,
                interval_duration_seconds(record),
                str(record.path),
                record,
            )
        )

    nearest.sort(key=lambda item: item[:3])
    distance, _, _, chosen = nearest[0]

    if distance > max_offset_seconds:
        return None, None, None

    return chosen, distance, "nearest_interval"




def seconds_offset(a: dt.datetime, b: dt.datetime) -> int:
    return int(abs((a - b).total_seconds()))


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
    sitfuse_records: list[RasterRecord],
    hms_records: list[RasterRecord],
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matching_cfg = cfg["matching"]
    max_offset_seconds = int(
        matching_cfg.get("max_time_difference_seconds", 3600)
    )
    allow_hms_reuse = bool(matching_cfg.get("allow_hms_reuse", True))
    allow_nearest_interval = bool(
        matching_cfg.get("allow_nearest_interval", True)
    )
    group_by = matching_cfg.get("group_by", "hms_interval")

    used_hms_paths: set[str] = set()
    matchups: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    for sf in sitfuse_records:
        hms, offset_seconds, match_method = choose_hms_interval_match(
            sitfuse_record=sf,
            hms_records=hms_records,
            max_offset_seconds=max_offset_seconds,
            allow_hms_reuse=allow_hms_reuse,
            used_hms_paths=used_hms_paths,
            allow_nearest_interval=allow_nearest_interval,
        )

        if hms is None:
            audit.append({
                "sit_fuse_path": str(sf.path),
                "sit_fuse_product": sf.product,
                "sit_fuse_timestamp": sf.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "hms_path": "",
                "hms_product": "",
                "hms_interval_start": "",
                "hms_timestamp": "",
                "hms_interval_end": "",
                "time_offset_seconds": "",
                "match_method": "",
                "pair_id": "",
                "status": "no_hms_interval_match",
                "notes": "No same-product HMS interval contained or was close enough to the SIT-FUSE timestamp.",
            })
            continue

        if not allow_hms_reuse:
            used_hms_paths.add(str(hms.path))

        hms_start = hms.interval_start or dt.datetime.combine(
            hms.date, dt.time.min
        )
        hms_end = hms.interval_end or dt.datetime.combine(
            hms.date, dt.time.max
        )

        if group_by == "hms_interval":
            pair_id = (
                f"{hms.product}_"
                f"{hms_start:%Y%m%dT%H%M%SZ}_"
                f"{hms_end:%Y%m%dT%H%M%SZ}"
            )
        elif group_by == "sit_fuse_timestamp":
            pair_id = f"{sf.product}_{sf.timestamp:%Y%m%dT%H%M%SZ}"
        elif group_by == "hms_day":
            pair_id = f"{hms.product}_{hms_start:%Y%m%d}"
        elif group_by == "product":
            pair_id = hms.product
        else:
            raise ValueError(
                "matching.group_by must be one of: hms_interval, "
                "sit_fuse_timestamp, hms_day, product"
            )

        matchup = {
            "pair_id": pair_id,
            "label": sf.product,
            "product": sf.product,
            "sit_fuse_context_free": str(sf.path),
            "truth_hms": str(hms.path),
            "sit_fuse_timestamp": sf.timestamp.strftime("%Y-%m-%dT%H%M%SZ"),
            "hms_date": hms_start.date().isoformat(),
            "hms_interval_start": hms_start.strftime("%Y-%m-%dT%H%M%SZ"),
            "hms_interval_end": hms_end.strftime("%Y-%m-%dT%H%M%SZ"),
            "hms_timestamp": hms_start.strftime("%Y-%m-%dT%H%M%SZ"),
            "time_offset_seconds": offset_seconds,
            "match_method": match_method,
        }
        matchups.append(matchup)

        audit.append({
            "sit_fuse_path": str(sf.path),
            "sit_fuse_product": sf.product,
            "sit_fuse_timestamp": matchup["sit_fuse_timestamp"],
            "hms_path": str(hms.path),
            "hms_product": hms.product,
            "hms_interval_start": matchup["hms_interval_start"],
            "hms_interval_end": matchup["hms_interval_end"],
            "hms_timestamp": matchup["hms_interval_start"],
            "time_offset_seconds": offset_seconds,
            "match_method": match_method,
            "pair_id": pair_id,
            "status": "matched",
            "notes": "",
        })

    return matchups, audit

def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "sit_fuse_path",
        "sit_fuse_product",
        "sit_fuse_timestamp",
        "hms_path",
        "hms_product",
        "hms_interval_start",
        "hms_interval_end",
        "hms_timestamp",
        "time_offset_seconds",
        "match_method",
        "pair_id",
        "status",
        "notes",
    ]


    print(rows[0].keys())

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

    hms_records, hms_skipped = collect_hms_truth_records(
        cfg=cfg["hms"],
        start=start,
        end=end,
    )


    LOG.info("SIT-FUSE context-free rasters found: %d", len(sit_fuse_records))
    LOG.info("HMS rasters found: %d", len(hms_records))

    matchups, audit = build_matchups(
        sit_fuse_records,
        hms_records,
        cfg,
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
