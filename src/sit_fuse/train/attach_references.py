#!/usr/bin/env python3
"""
Attach additional reference products to a primary matchup file, for
multi-reference evaluation and reference-disagreement (aleatoric) tests.

Records are joined on the SIT-FUSE output (sit_fuse_context_free, compared by
basename). For each additional matchup file (HMS / ADP / FDCC generator
output), the joined record gains

    additional_references: {<name>: <annotation_raster or truth path>}

The primary record's split fields are left unchanged, so the attached
references never alter dev / calibration / holdout membership.

Usage:
    python attach_references.py primary_matchups.yaml out.yaml \
        --add adp_smoke=adp_smoke_classifier_matchups.yaml \
        [--add hms_smoke=hms_smoke_classifier_matchups.yaml]
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml

from sit_fuse.train.classifier_utils import load_yaml

LOG = logging.getLogger("attach-references")
TRUTH_KEYS = ("annotation_raster", "truth_hms", "truth_adp_smoke", "truth_fdcc_fire")


def _truth(rec):
    for k in TRUTH_KEYS:
        if rec.get(k):
            return str(rec[k])
    return None


def attach(primary: dict, others: dict[str, dict], product_filter: bool = True) -> dict:
    recs = primary.get("matchups", [])
    stats = {}
    for name, doc in others.items():
        index = {}
        for r in doc.get("matchups", []):
            key = (os.path.basename(str(r["sit_fuse_context_free"])),
                   str(r.get("product", "")).lower() if product_filter else "")
            index.setdefault(key, _truth(r))
        hit = 0
        for r in recs:
            key = (os.path.basename(str(r["sit_fuse_context_free"])),
                   str(r.get("product", "")).lower() if product_filter else "")
            t = index.get(key)
            if t is None and product_filter:
                t = index.get((key[0], ""))
            if t:
                r.setdefault("additional_references", {})[name] = t
                hit += 1
        stats[name] = {"attached": hit, "of": len(recs)}
        LOG.info("%s: attached to %d / %d records", name, hit, len(recs))
    primary.setdefault("matchup_metadata", {})["additional_references"] = stats
    return primary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("primary", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--add", action="append", required=True, metavar="NAME=MATCHUP_YAML")
    ap.add_argument("--ignore-product", action="store_true",
                    help="join on the SIT-FUSE output only (e.g., FDCC fire onto HMS fire)")
    a = ap.parse_args()
    others = {}
    for item in a.add:
        print(item)
        name, path = item.split("=", 1)
        others[name] = load_yaml(Path(path))
    doc = attach(load_yaml(a.primary), others, product_filter=not a.ignore_product)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    LOG.info("Wrote %s", a.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
