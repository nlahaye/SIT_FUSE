"""
Observing days and classifier splits of each fusion source, to pick fusion fit / evaluation dates.

    python list_fusion_splits.py configs/fusion_config.smoke.yaml

A fused frame is used for fitting only when every contributing source frame is `calibration`
(fuse_multiscale_products.frame_split); days where the sensors disagree are `mixed` and skipped.
"""
import sys

import pandas as pd
import yaml

from uq_viz import data as D

cfg = yaml.safe_load(open(sys.argv[1]))
cols = {}
for name, s in cfg["sources"].items():
    sc = D.GeoProductIndex(s["geo_dir"], cfg["product"], [s["model"]], s.get("manifest")).scenes(s["model"])
    g = sc.groupby(sc.date.astype(str))
    cols[name] = g.split.agg(lambda x: "|".join(sorted(set(x))))
    cols[f"{name}_scenes"] = g.size()
d = pd.DataFrame(cols)
src = list(cfg["sources"])
for k in src:
    d[k] = d[k].fillna("-")
    d[f"{k}_scenes"] = d[f"{k}_scenes"].fillna(0).astype(int)
d["fused"] = [next(iter(v)) if len(v := set(r[src]) - {"-"}) == 1 else "mixed" for _, r in d.iterrows()]
d["both_sensors"] = [all(r[k] != "-" for k in src) for _, r in d.iterrows()]
pd.set_option("display.width", 200)
print(d.to_string())
for sp in ("calibration", "holdout"):
    both = d[(d.fused == sp) & d.both_sensors].index.tolist()
    print(f"{sp} days with both sensors: {both}")
