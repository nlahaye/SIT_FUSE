#!/usr/bin/env python
"""
Copyright [2022-26], by the California Institute of Technology and Chapman University.
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.

Re-pin the classifier partitions (dev / calibration / holdout) of an existing classifier matchup YAML by split group,
without regenerating the matchups, and check that every event has calibration and holdout scenes with head outputs.
Used to give a second sensor the same held-out days as a reference run (e.g. GOES-18 fire aligned with the TEMPO fire
holdout days), while the matchups, attached references and head outputs stay as they are.

    python pin_matchup_splits.py goes18/g18_fire_ts_splits.yaml            # write the pinned YAML + check
    python pin_matchup_splits.py goes18/g18_fire_ts_splits.yaml --check-only  # inventory / check of `input` only

YAML keys
  input:        classifier matchup YAML (e.g. the attach_references.py output read by the UQ config)
  output:       pinned copy (point data.matchup_file of the new UQ config here)
  split_csv:    per-record split table (matchup_splits.SPLIT_FIELDS); default <output stem>_splits.csv
  summary_json: inventory + check results; default <output stem>_splits_summary.json
  assign_groups: {holdout: [YYYY-MM-DD, ...], calibration: [...], dev: [...]}
                split_group values (observing days); a merged link_shared_truth group ("a+b") matches any member day.
                Calibration / holdout only for eligible groups: every record encoder_test (as in matchup_splits).
  calibration_auto: {per_event: 1, min_gap_days: 1, exclude: [...]}
                for events with no listed calibration group: pick the eligible, unlisted group(s) with the most
                scenes that have head outputs, more than min_gap_days from every holdout day of that event.
  others:       dev (default) | keep. Groups not listed: dev, or keep their current partition.
  check:        {min_events: 3, min_holdout_scenes: 1, min_calibration_scenes: 1, require_head_outputs: true,
                 events: [names]}  -> exit 1 when an event (split_stratum prefix) lacks calibration or holdout
                scenes with head outputs on disk, or fewer than min_events events have records.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from matchup_splits import (CAL, DEV, ENCODER_TEST, HOLDOUT, load_yaml_any, write_split_csv)

LOG = logging.getLogger("pin-splits")


def _records(doc):
    """(key, list) of the matchup records: the top-level list whose entries carry classifier_split."""
    if isinstance(doc, list):
        return None, doc
    for k, v in doc.items():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "classifier_split" in v[0]:
            return k, v
    raise ValueError("no matchup list with classifier_split found (run the matchup generator with classifier_split)")


def _event(m):
    return str(m.get("split_stratum") or "none").split("|")[0]


def _days(group):
    out = []
    for p in str(group).split("+"):
        try:
            out.append(dt.date.fromisoformat(p[:10]))
        except ValueError:
            pass
    return out


def _has_heads(m):
    h = m.get("head_outputs")
    return bool(h) and Path(h).exists()


def inventory(recs, heads=True):
    """{event: {partition: {groups: [...], n: int, n_heads: int}}}."""
    inv = defaultdict(lambda: defaultdict(lambda: {"groups": set(), "n": 0, "n_heads": 0}))
    for m in recs:
        e = inv[_event(m)][m["classifier_split"]]
        e["groups"].add(str(m["split_group"]))
        e["n"] += 1
        e["n_heads"] += int(_has_heads(m)) if heads else 1
    return {ev: {p: {"groups": sorted(v["groups"]), "n": v["n"], "n_heads": v["n_heads"]} for p, v in ps.items()}
            for ev, ps in inv.items()}


def pin(recs, cfg):
    groups = defaultdict(list)
    for m in recs:
        groups[str(m["split_group"])].append(m)
    eligible = {g: all(m.get("encoder_split") == ENCODER_TEST for m in ms) for g, ms in groups.items()}
    ev_of = {g: Counter(_event(m) for m in ms).most_common(1)[0][0] for g, ms in groups.items()}
    others = str(cfg.get("others", DEV)).lower()
    assign = {g: (ms[0]["classifier_split"] if others == "keep" else DEV) for g, ms in groups.items()}
    listed = {}
    for part, vals in (cfg.get("assign_groups") or {}).items():
        part = str(part).lower()
        if part not in (DEV, CAL, HOLDOUT):
            raise ValueError(f"assign_groups: unknown partition {part!r} (use {DEV}, {CAL} or {HOLDOUT})")
        for v in vals or []:
            v = str(v)
            hits = [g for g in groups if v == g or v in g.split("+")]
            if not hits:
                LOG.warning("assign_groups.%s: no split_group matches %s (not in the matchups)", part, v)
            for g in hits:
                if g in listed and listed[g] != part:
                    raise ValueError(f"assign_groups: group {g} listed for both {listed[g]} and {part}")
                if part != DEV and not eligible[g]:
                    LOG.warning("assign_groups.%s: %s has encoder_train / unknown records; left as dev", part, g)
                    listed[g] = DEV
                    continue
                listed[g] = part
    assign.update(listed)
    ca = cfg.get("calibration_auto")
    if ca:
        per, gap = int(ca.get("per_event", 1)), float(ca.get("min_gap_days", 1))
        excl = {str(x) for x in ca.get("exclude") or []}
        for ev in sorted(set(ev_of.values())):
            if any(assign[g] == CAL for g in groups if ev_of[g] == ev):
                continue
            hold = [d for g in groups if ev_of[g] == ev and assign[g] == HOLDOUT for d in _days(g)]
            cand = []
            for g, ms in groups.items():
                if ev_of[g] != ev or g in listed or not eligible[g] or set(g.split("+")) & excl:
                    continue
                if any(abs((d - h).days) <= gap for d in _days(g) for h in hold):
                    continue
                pos = sum(_has_heads(m) and float(m.get("truth_positive_fraction") or 0) > 0 for m in ms)
                cand.append((-pos, -sum(_has_heads(m) for m in ms), g))   # most positive scenes, then most scenes
            for *_, g in sorted(cand)[:per]:
                LOG.info("calibration_auto: %s -> calibration for %s (%d scenes with heads)", g, ev,
                         sum(_has_heads(m) for m in groups[g]))
                assign[g] = CAL
            if not cand:
                LOG.warning("calibration_auto: no eligible group for %s (more than %g days from its holdout days)",
                            ev, gap)
    out, rows, moved = [], [], Counter()
    for m in recs:
        m = dict(m)
        new = assign[str(m["split_group"])]
        if new != m["classifier_split"]:
            moved[(m["classifier_split"], new)] += 1
        m["classifier_split"] = new
        out.append(m)
        rows.append({**m, "status": "kept", "notes": "pinned" if str(m["split_group"]) in listed else others})
    for (a, b), n in sorted(moved.items()):
        LOG.info("moved %5d records %s -> %s", n, a, b)
    return out, rows


def check(inv, ck):
    errs = []
    evs = [e for e in inv if e != "none"]
    if len(evs) < int(ck.get("min_events", 1)):
        errs.append(f"{len(evs)} event(s) with records ({sorted(evs)}), expected >= {ck.get('min_events')}; "
                    "an event missing here was dropped upstream (encoder_split inference_yamls / matchups)")
    key = "n_heads" if ck.get("require_head_outputs", True) else "n"
    for ev in ck.get("events") or evs:
        for part, mk in ((HOLDOUT, "min_holdout_scenes"), (CAL, "min_calibration_scenes")):
            n = (inv.get(ev, {}).get(part) or {}).get(key, 0)
            if n < int(ck.get(mk, 1)):
                have = inv.get(ev, {}).get(part) or {}
                errs.append(f"{ev}: {n} {part} scene(s) with {'head outputs' if key == 'n_heads' else 'records'} "
                            f"(records {have.get('n', 0)}, groups {have.get('groups', [])})")
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaml_config", type=Path)
    ap.add_argument("--check-only", action="store_true", help="inventory / check of the input, write nothing")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_yaml_any(a.yaml_config)              # relative paths: from the working dir (src/sit_fuse/train)
    doc = load_yaml_any(cfg["input"])
    key, recs = _records(doc)
    ck = cfg.get("check") or {}
    if a.check_only:
        inv = inventory(recs, ck.get("require_head_outputs", True))
        print(json.dumps(inv, indent=1))
        errs = check(inv, ck)
        for e in errs:
            LOG.error(e)
        sys.exit(1 if errs else 0)
    out, rows = pin(recs, cfg)
    inv = inventory(out, ck.get("require_head_outputs", True))
    for ev in sorted(inv):
        LOG.info("%s: %s", ev, {p: f"{v['n']} scenes ({v['n_heads']} with heads), groups {v['groups']}"
                                for p, v in sorted(inv[ev].items())})
    errs = check(inv, ck)
    op = Path(cfg["output"])
    if key is None:
        doc = out
    else:
        doc[key] = out
    op.parent.mkdir(parents=True, exist_ok=True)
    tmp = op.with_suffix(op.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(doc, sort_keys=False, width=200))
    tmp.replace(op)
    write_split_csv(rows, cfg.get("split_csv") or op.with_name(op.stem + "_splits.csv"))
    sj = Path(cfg.get("summary_json") or op.with_name(op.stem + "_splits_summary.json"))
    sj.write_text(json.dumps({"input": str(cfg["input"]), "output": str(op), "inventory": inv, "errors": errs},
                             indent=1))
    LOG.info("wrote %s (%d records), summary %s", op, len(out), sj)
    for e in errs:
        LOG.error(e)
    if errs:
        LOG.error("split check failed: fix the matchups / head outputs (or the check block) before training")
        sys.exit(1)


if __name__ == "__main__":
    main()
