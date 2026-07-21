#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


GEOTIFF_EXTS = {".tif", ".tiff"}


@dataclass
class RasterRecord:
    path: Path
    product: str            # smoke | fire
    timestamp: dt.datetime
    source_family: str      # sit_fuse | hms | tempo | other
    source_group: str       # truth | other | sit_fuse


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must parse to a dictionary.")
    return cfg


def write_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def parse_time(value: str) -> dt.datetime:
    value = value.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"Could not parse time value: {value}")


def iter_rasters(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in GEOTIFF_EXTS:
            yield p


def infer_product(name: str, smoke_patterns: list[str], fire_patterns: list[str]) -> str | None:
    lname = name.lower()
    for pat in smoke_patterns:
        if re.search(pat, lname):
            return "smoke"
    for pat in fire_patterns:
        if re.search(pat, lname):
            return "fire"
    return None


def parse_timestamp(
    name: str,
    timestamp_patterns: list[str],
    daily_date_patterns: list[str],
    daily_time_policy: str = "start_of_day",
) -> dt.datetime | None:
    for pat in timestamp_patterns:
        m = re.search(pat, name)
        if m:
            txt = m.group(1)
            for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S"):
                try:
                    return dt.datetime.strptime(txt, fmt)
                except ValueError:
                    pass

    for pat in daily_date_patterns:
        m = re.search(pat, name)
        if m:
            txt = m.group(1)
            try:
                day = dt.datetime.strptime(txt, "%Y%m%d")
            except ValueError:
                continue

            if daily_time_policy == "midday":
                return day.replace(hour=12, minute=0, second=0)
            if daily_time_policy == "end_of_day":
                return day.replace(hour=23, minute=59, second=59)
            return day.replace(hour=0, minute=0, second=0)

    return None


def build_records(
    root: Path,
    source_family: str,
    source_group: str,
    smoke_patterns: list[str],
    fire_patterns: list[str],
    timestamp_patterns: list[str],
    daily_date_patterns: list[str],
    start: dt.datetime,
    end: dt.datetime,
    daily_time_policy: str = "start_of_day",
) -> list[RasterRecord]:
    records: list[RasterRecord] = []

    for path in iter_rasters(root):
        product = infer_product(path.name, smoke_patterns, fire_patterns)
        if product is None:
            continue

        ts = parse_timestamp(
            path.name,
            timestamp_patterns=timestamp_patterns,
            daily_date_patterns=daily_date_patterns,
            daily_time_policy=daily_time_policy,
        )
        if ts is None:
            continue

        if not (start <= ts <= end):
            continue

        records.append(
            RasterRecord(
                path=path.resolve(),
                product=product,
                timestamp=ts,
                source_family=source_family,
                source_group=source_group,
            )
        )

    records.sort(key=lambda r: (r.product, r.timestamp, str(r.path)))
    return records


def abs_seconds(a: dt.datetime, b: dt.datetime) -> float:
    return abs((a - b).total_seconds())


def best_match(
    target: RasterRecord,
    candidates: list[RasterRecord],
    tolerance_seconds: int,
) -> RasterRecord | None:
    if not candidates:
        return None

    same_product = [c for c in candidates if c.product == target.product]
    if not same_product:
        return None

    chosen = min(
        same_product,
        key=lambda c: (abs_seconds(target.timestamp, c.timestamp), str(c.path)),
    )
    if abs_seconds(target.timestamp, chosen.timestamp) <= tolerance_seconds:
        return chosen
    return None


def choose_truth_match(
    sf: RasterRecord,
    truth_records_by_family: dict[str, list[RasterRecord]],
    family_priority: list[str],
    family_tolerances: dict[str, int],
    used_truth: set[str],
) -> RasterRecord | None:
    family_best: list[tuple[float, int, RasterRecord]] = []

    for priority_idx, family in enumerate(family_priority):
        candidates = [
            rec for rec in truth_records_by_family.get(family, [])
            if rec.product == sf.product and str(rec.path) not in used_truth
        ]
        match = best_match(sf, candidates, family_tolerances[family])
        if match is not None:
            family_best.append((abs_seconds(sf.timestamp, match.timestamp), priority_idx, match))

    if not family_best:
        return None

    family_best.sort(key=lambda x: (x[0], x[1], str(x[2].path)))
    return family_best[0][2]


def choose_other_match(
    sf: RasterRecord,
    other_records: list[RasterRecord] | None,
    tolerance_seconds: int,
    used_other: set[str],
) -> RasterRecord | None:
    if other_records is None:
        return None

    candidates = [
        rec for rec in other_records
        if rec.product == sf.product and str(rec.path) not in used_other
    ]
    return best_match(sf, candidates, tolerance_seconds)


def build_matchups(
    sit_fuse_records: list[RasterRecord],
    truth_records_by_family: dict[str, list[RasterRecord]],
    truth_family_priority: list[str],
    truth_family_tolerances: dict[str, int],
    other_records: list[RasterRecord] | None,
    other_tolerance_seconds: int,
    require_other_match: bool,
) -> tuple[list[str], list[str], list[str] | None]:
    sit_fuse_maps: list[str] = []
    truth_maps: list[str] = []
    other_maps: list[str] | None = [] if other_records is not None else None

    used_truth: set[str] = set()
    used_other: set[str] = set()

    for sf in sit_fuse_records:
        truth_match = choose_truth_match(
            sf=sf,
            truth_records_by_family=truth_records_by_family,
            family_priority=truth_family_priority,
            family_tolerances=truth_family_tolerances,
            used_truth=used_truth,
        )
        if truth_match is None:
            continue

        other_match = choose_other_match(
            sf=sf,
            other_records=other_records,
            tolerance_seconds=other_tolerance_seconds,
            used_other=used_other,
        )
        if require_other_match and other_records is not None and other_match is None:
            continue

        sit_fuse_maps.append(str(sf.path))
        truth_maps.append(str(truth_match.path))
        used_truth.add(str(truth_match.path))

        if other_maps is not None:
            if other_match is not None:
                other_maps.append(str(other_match.path))
                used_other.add(str(other_match.path))
            else:
                other_maps.append("")

    return sit_fuse_maps, truth_maps, other_maps


def main():
    parser = argparse.ArgumentParser(
        description="Generate matchup YAML for SIT-FUSE vs HMS/TEMPO truth rasters."
    )
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    start = parse_time(cfg["time_range"]["start"])
    end = parse_time(cfg["time_range"]["end"])
    if end < start:
        raise ValueError("time_range.end must be >= time_range.start")

    sit_cfg = cfg["sit_fuse"]
    sit_fuse_dir = Path(sit_cfg["directory"])
    sit_fuse_records = build_records(
        root=sit_fuse_dir,
        source_family="sit_fuse",
        source_group="sit_fuse",
        smoke_patterns=sit_cfg["patterns"]["smoke"],
        fire_patterns=sit_cfg["patterns"]["fire"],
        timestamp_patterns=sit_cfg["patterns"]["timestamp_patterns"],
        daily_date_patterns=sit_cfg["patterns"].get("daily_date_patterns", []),
        daily_time_policy=sit_cfg.get("daily_time_policy", "start_of_day"),
        start=start,
        end=end,
    )

    truth_records_by_family: dict[str, list[RasterRecord]] = {}
    truth_family_priority: list[str] = []
    truth_family_tolerances: dict[str, int] = {}

    for fam_cfg in cfg["truth_families"]:
        family = fam_cfg["name"]
        truth_family_priority.append(family)
        truth_family_tolerances[family] = int(fam_cfg["max_time_difference_seconds"])

        truth_records_by_family[family] = build_records(
            root=Path(fam_cfg["directory"]),
            source_family=family,
            source_group="truth",
            smoke_patterns=fam_cfg["patterns"]["smoke"],
            fire_patterns=fam_cfg["patterns"]["fire"],
            timestamp_patterns=fam_cfg["patterns"].get("timestamp_patterns", []),
            daily_date_patterns=fam_cfg["patterns"].get("daily_date_patterns", []),
            daily_time_policy=fam_cfg.get("daily_time_policy", "start_of_day"),
            start=start,
            end=end,
        )

    other_records = None
    require_other_match = False
    other_tolerance_seconds = 0
    if "other" in cfg and cfg["other"] is not None:
        other_cfg = cfg["other"]
        require_other_match = bool(other_cfg.get("require_match", False))
        other_tolerance_seconds = int(other_cfg["max_time_difference_seconds"])
        other_records = build_records(
            root=Path(other_cfg["directory"]),
            source_family="other",
            source_group="other",
            smoke_patterns=other_cfg["patterns"]["smoke"],
            fire_patterns=other_cfg["patterns"]["fire"],
            timestamp_patterns=other_cfg["patterns"].get("timestamp_patterns", []),
            daily_date_patterns=other_cfg["patterns"].get("daily_date_patterns", []),
            daily_time_policy=other_cfg.get("daily_time_policy", "start_of_day"),
            start=start,
            end=end,
        )

    print(f"SIT-FUSE rasters found: {len(sit_fuse_records)}")
    for family in truth_family_priority:
        print(f"Truth rasters found for {family}: {len(truth_records_by_family[family])}")
    if other_records is not None:
        print(f"Other rasters found: {len(other_records)}")

    sit_fuse_maps, truth_maps, other_maps = build_matchups(
        sit_fuse_records=sit_fuse_records,
        truth_records_by_family=truth_records_by_family,
        truth_family_priority=truth_family_priority,
        truth_family_tolerances=truth_family_tolerances,
        other_records=other_records,
        other_tolerance_seconds=other_tolerance_seconds,
        require_other_match=require_other_match,
    )

    output = {
        "sit_fuse_maps": sit_fuse_maps,
        "truth_maps": truth_maps,
    }
    if other_maps is not None:
        output["other_maps"] = other_maps

    out_yaml = Path(cfg["output"]["matchup_yaml"])
    write_yaml(output, out_yaml)

    print(f"Matched pairs written: {len(sit_fuse_maps)}")
    print(f"Output YAML: {out_yaml.resolve()}")


if __name__ == "__main__":
    main()
