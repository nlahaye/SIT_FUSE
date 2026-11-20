#!/usr/bin/env python3
"""
List matchup records whose head_outputs raster is missing (or unreadable), and optionally write an
export_head_outputs.py config that exports only those inputs.

    # from a UQ config (data.matchup_file) or matchup YAMLs directly
    python check_head_outputs.py -c goes18/g18_uq_config_fire.yaml
    python check_head_outputs.py goes18/g18_fdcc_fire_classifier_matchups.yaml --splits dev calibration holdout
    # also open every head file (catches truncated / corrupt rasters; slower)
    python check_head_outputs.py -c goes18/g18_uq_config_fire.yaml --read
    # export config for the missing inputs only (copy of the given export config with files: set)
    python check_head_outputs.py -c goes18/g18_uq_config_fire.yaml -e goes18/g18_export_fire.yaml \
        --write-export goes18/g18_export_fire_missing.yaml
    python export_head_outputs.py goes18/g18_export_fire_missing.yaml

The UQ scripts skip records without a head output (load_matchups missing_heads: skip), so a run continues with
the available scenes either way. Inputs that export_head_outputs.py skips with "no valid samples" never get a
head file; they stay listed here after a re-export and are skipped by the UQ scripts.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml


def _read(p):
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _records(path, splits=None):
    doc = _read(path)
    for r in doc.get("matchups", doc.get("rasters", [])) or []:
        if splits and str(r.get("classifier_split") or "") not in splits:
            continue
        yield r


def _readable(p):
    try:
        import rasterio
        with rasterio.open(p) as s:
            s.read(1, window=((0, min(8, s.height)), (0, min(8, s.width))))
            s.read(s.count, window=((s.height - 1, s.height), (s.width - 1, s.width)))
        return None
    except Exception as e:  # noqa: BLE001  (any GDAL / IO failure)
        return str(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("matchups", nargs="*", type=Path, help="matchup YAMLs (default: data.matchup_file of -c)")
    ap.add_argument("-c", "--uq-config", type=Path, help="UQ config; uses data.matchup_file and data.include_splits")
    ap.add_argument("-e", "--export-config", type=Path, help="export_head_outputs.py config to copy for --write-export")
    ap.add_argument("--write-export", type=Path, help="write an export config with files: = inputs with missing heads")
    ap.add_argument("--splits", nargs="+", help="only these classifier_split values")
    ap.add_argument("--read", action="store_true", help="also open each existing head file (unreadable = missing)")
    ap.add_argument("--out", type=Path, help="CSV of the missing records")
    a = ap.parse_args()

    files, splits = list(a.matchups), set(a.splits or [])
    if a.uq_config:
        d = _read(a.uq_config).get("data", {})
        files = files or [Path(d["matchup_file"])]
        splits = splits or set(d.get("include_splits") or [])
    if not files:
        ap.error("give matchup YAMLs or -c <uq_config>")

    rows, seen, n = [], set(), 0
    for f in files:
        for r in _records(f, splits):
            h = str(r.get("head_outputs") or "")
            if not h or h in seen:
                continue
            seen.add(h)
            n += 1
            reason = None if Path(h).exists() else "missing"
            if reason is None and a.read:
                err = _readable(h)
                reason = f"unreadable: {err}" if err else None
            if reason:
                src = r.get("sit_fuse_input")
                src = src[0] if isinstance(src, list) and len(src) == 1 else src
                rows.append({"scene_id": r.get("scene_id") or r.get("pair_id"), "split": r.get("classifier_split"),
                             "head_outputs": h, "reason": reason, "sit_fuse_input": src,
                             "input_exists": all(Path(str(x)).exists() for x in (src if isinstance(src, list)
                                                                               else [src])) if src else False})
    by = {}
    for r in rows:
        by[r["split"]] = by.get(r["split"], 0) + 1
    print(f"{len(rows)} of {n} head outputs missing/unreadable" + (f" (by split: {by})" if rows else ""))
    for r in rows[:15]:
        print(f"  {r['reason'][:40]:40s} {Path(r['head_outputs']).name}  input_exists={r['input_exists']}")
    if len(rows) > 15:
        print(f"  ... {len(rows) - 15} more" + (f" (see {a.out})" if a.out else " (use --out list.csv)"))
    if a.out and rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    if a.write_export:
        if not a.export_config:
            ap.error("--write-export needs -e <export config>")
        todo = [r["sit_fuse_input"] for r in rows if r["sit_fuse_input"] and r["input_exists"]]
        no_in = sum(1 for r in rows if not r["input_exists"])
        cfg = _read(a.export_config)
        cfg.pop("matchup_file", None)
        cfg.pop("matchup_splits", None)
        cfg["files"] = todo
        if (cfg.get("georef") or {}).get("reference_geotiffs") or (cfg.get("georef") or {}).get("subset_inds"):
            print("note: georef.reference_geotiffs / subset_inds are per-file lists; cleared for the subset")
            cfg["georef"]["reference_geotiffs"] = None
            cfg["georef"]["subset_inds"] = None
        a.write_export.parent.mkdir(parents=True, exist_ok=True)
        a.write_export.write_text(yaml.safe_dump(cfg, sort_keys=False, width=200))
        print(f"wrote {a.write_export}: {len(todo)} inputs to export"
              + (f"; {no_in} have no input on disk (cannot be exported)" if no_in else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
