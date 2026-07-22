#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import importlib
import sys
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Pipeline config must parse to a dictionary.")
    return data


def write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def import_callable(module_name: str, func_name: str):
    module = importlib.import_module(module_name)
    func = getattr(module, func_name, None)
    if func is None:
        raise AttributeError(f"Function '{func_name}' not found in module '{module_name}'")
    return func


def build_matchup_config_from_pipeline_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    matchup_cfg = cfg["matchup"]
    return copy.deepcopy(matchup_cfg)


def build_compare_config_from_outputs(
    matchup_output_cfg: dict[str, Any],
    pipeline_cfg: dict[str, Any],
) -> dict[str, Any]:
    matchup_yaml_path = Path(matchup_output_cfg["output"]["matchup_yaml"])
    with open(matchup_yaml_path, "r", encoding="utf-8") as f:
        matchup_yaml = yaml.safe_load(f)

    if not isinstance(matchup_yaml, dict):
        raise ValueError("Generated matchup YAML must parse to a dictionary.")

    compare_cfg = {
        "sit_fuse_maps": matchup_yaml["sit_fuse_maps"],
        "truth_maps": matchup_yaml["truth_maps"],
    }

    if "other_maps" in matchup_yaml:
        compare_cfg["other_maps"] = matchup_yaml["other_maps"]

    compare_overrides = copy.deepcopy(pipeline_cfg.get("compare", {}))
    compare_cfg = deep_update(compare_cfg, compare_overrides)
    return compare_cfg


def run_matchup_stage(cfg: dict[str, Any]) -> dict[str, Any]:
    runtime = cfg.get("runtime", {})
    module_name = runtime.get("matchup_module", "build_matchups_v2")
    func_name = runtime.get("matchup_function", "main")

    matchup_cfg = build_matchup_config_from_pipeline_cfg(cfg)

    callable_obj = import_callable(module_name, func_name)

    accepts_dict = runtime.get("matchup_accepts_dict", True)
    if accepts_dict:
        callable_obj(matchup_cfg)
    else:
        temp_cfg_path = Path(matchup_cfg["output"].get("temp_matchup_config_yaml", "tmp_matchup_config.yaml"))
        write_yaml(matchup_cfg, temp_cfg_path)
        callable_obj(str(temp_cfg_path))

    return matchup_cfg


def run_compare_stage(cfg: dict[str, Any], matchup_cfg: dict[str, Any]) -> dict[str, Any]:
    runtime = cfg.get("runtime", {})
    module_name = runtime.get("compare_module", "regrid_and_compare_v2")
    func_name = runtime.get("compare_function", "regrid_and_compare")

    compare_cfg = build_compare_config_from_outputs(matchup_cfg, cfg)
    callable_obj = import_callable(module_name, func_name)

    accepts_dict = runtime.get("compare_accepts_dict", True)
    if accepts_dict:
        callable_obj(compare_cfg)
    else:
        temp_compare_yaml = Path(
            cfg["compare"].get("outputs", {}).get("temp_compare_yaml", "tmp_compare_config.yaml")
        )
        write_yaml(compare_cfg, temp_compare_yaml)
        callable_obj(str(temp_compare_yaml))

    return compare_cfg


def write_pipeline_manifest(cfg: dict[str, Any], matchup_cfg: dict[str, Any], compare_cfg: dict[str, Any]) -> None:
    manifest_path = Path(cfg["pipeline"]["output_manifest_yaml"])
    manifest = {
        "pipeline_name": cfg["pipeline"].get("name", "sit_fuse_validation_pipeline"),
        "matchup_yaml": matchup_cfg["output"]["matchup_yaml"],
        "matchup_audit_csv": matchup_cfg["output"].get("audit_csv", ""),
        "compare_per_scene_csv": compare_cfg.get("outputs", {}).get("per_scene_csv", ""),
        "compare_summary_csv": compare_cfg.get("outputs", {}).get("summary_csv", ""),
        "compare_diff_raster_dir": compare_cfg.get("outputs", {}).get("diff_raster_dir", ""),
    }
    write_yaml(manifest, manifest_path)


def validate_pipeline_cfg(cfg: dict[str, Any]) -> None:
    required = ["pipeline", "matchup", "compare"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Missing required top-level key: {k}")

    if "output_manifest_yaml" not in cfg["pipeline"]:
        raise ValueError("pipeline.output_manifest_yaml is required")

    matchup_required = ["time_range", "sit_fuse", "truth_families", "output"]
    for k in matchup_required:
        if k not in cfg["matchup"]:
            raise ValueError(f"Missing required matchup config key: {k}")

    if "matchup_yaml" not in cfg["matchup"]["output"]:
        raise ValueError("matchup.output.matchup_yaml is required")


def main(config_path: str):
    cfg = load_yaml(Path(config_path))
    validate_pipeline_cfg(cfg)

    print("Running matchup stage...")
    matchup_cfg = run_matchup_stage(cfg)

    print("Running comparison stage...")
    compare_cfg = run_compare_stage(cfg, matchup_cfg)

    write_pipeline_manifest(cfg, matchup_cfg, compare_cfg)
    print("Pipeline complete.")
    print("Manifest:", cfg["pipeline"]["output_manifest_yaml"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified SIT-FUSE matchup + comparison pipeline."
    )
    parser.add_argument("config", help="Path to pipeline YAML config")
    args = parser.parse_args()

    try:
        main(args.config)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)



