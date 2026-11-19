#!/usr/bin/env python
"""Show how build_production_run.py maps inputs / RGB stand-ins to head outputs, for a production YAML.

    python check_stand_in_heads.py production_run.yaml [-n 5]

For every pipeline: the heads folder the builder uses (<output_root>/heads/<sensor>_<n>), how many selected scenes
have a head output there, and for the first N scenes the stand-in file, the original input name, the expected head
output and whether it exists. Missing heads are also searched for (by scene stem) under output_root and the
pipeline's UQ output_dir, so heads written somewhere else are reported with their actual location.
"""
import argparse
import json
from pathlib import Path

import build_production_run as B
from uq_viz.overlay import utc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaml_config", type=Path)
    ap.add_argument("-n", type=int, default=5, help="examples per pipeline")
    a = ap.parse_args()
    cfg = B._read(a.yaml_config)
    base = a.yaml_config.resolve().parent
    root = Path(B._abs(base, cfg["output_root"]))
    t0, t1 = utc(cfg["start"]), utc(cfg["end"])
    bbox = cfg.get("bbox")
    print(f"output_root {root}\n[start, end) {t0} .. {t1}   bbox {bbox}")

    inputs = {}
    for s, sc in cfg["sensors"].items():
        inputs[s], cnt = B.list_inputs(sc, t0, t1, bbox, base, s)
        print(f"\nsensor {s}: inputs {sc['inputs']}  pattern {sc.get('pattern', '*.tif')}  "
              f"stand_in_suffix {sc.get('stand_in_suffix')}  -> {len(inputs[s])} scene(s)")
        print("  counts " + json.dumps({k: v for k, v in cnt.items() if v and k != 'first_times'}))

    exports = {}
    for name, p in cfg["pipelines"].items():
        p = dict(p, _name=name, _uq_path=B._abs(base, p["uq_config"]))
        uq = B._read(p["_uq_path"])
        s = p["sensor"]
        try:
            hs = B.head_settings(p, uq, base)
        except Exception as ex:                       # report and continue with a placeholder key
            print(f"\npipeline {name}: head_settings failed: {ex}")
            hs = {"error": str(ex)}
        key = json.dumps([s, hs], sort_keys=True)
        e = exports.setdefault(key, {"name": f"{s}_{len([x for x in exports.values() if x['sensor'] == s])}",
                                     "sensor": s})
        hdir = root / "heads" / e["name"]
        items = inputs[s]
        have = [f for f, _ in items if B.head_path(hdir, f).exists()]
        n_in_dir = len(list(hdir.glob("*.head_outputs.tif"))) if hdir.is_dir() else 0
        print(f"\npipeline {name}: heads folder {hdir} ({'exists' if hdir.is_dir() else 'MISSING'}, "
              f"{n_in_dir} head output files)")
        print(f"  selected scenes with a head output: {len(have)} / {len(items)}")
        if n_in_dir:
            print(f"  e.g. existing: {sorted(x.name for x in hdir.glob('*.head_outputs.tif'))[0]}")
        uq_out = Path(B._abs(Path(p["_uq_path"]).parent, uq.get("output_dir", "."))) if uq.get("output_dir") else None
        for f, t in items[:a.n]:
            h = B.head_path(hdir, f)
            print(f"  - time       {t:%Y-%m-%dT%H:%M:%SZ}\n"
                  f"    stand-in   {B.STAND_IN.get(f, '(none, original input used)')}\n"
                  f"    original   {f}  ({'exists' if Path(f).exists() else 'not on disk'})\n"
                  f"    head       {h}  ({'FOUND' if h.exists() else 'not found'})")
            if not h.exists():
                stem = Path(f).name[:-4]
                for d in [root] + ([uq_out] if uq_out else []):
                    hits = sorted(d.glob(f"**/{stem}*.head_outputs.tif"))[:3] if d and d.is_dir() else []
                    for x in hits:
                        print(f"    found elsewhere: {x}")


if __name__ == "__main__":
    main()
