"""
Pull, reformat and merge in-situ HAB references into the combined_full.csv
format used by insitu_hab_to_multi_hist and the HAB UQ pipeline.

  python -m sit_fuse.pipelines.habs.insitu.pull_insitu_hab \
      -y src/sit_fuse/config/postprocess/insitu_class_match/insitu_sources.example.yaml

Output:
  {out_dir}/combined_augmented.csv        42 reference columns + provenance/extra columns
  {out_dir}/combined_augmented_strict.csv exact 42-column layout (drop-in for legacy code)
  {out_dir}/by_source/{source}.csv        per-source tables
  {out_dir}/inventory.csv                 rows / stations / date range / non-null counts per source
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from sit_fuse.pipelines.habs.insitu.calhabmap_format import (
    CALHABMAP_COLUMNS, merge_sources, validate_calhabmap, write_calhabmap)

TARGETS = ["Total_Phytoplankton", "Pseudo_nitzschia_seriata_group",
           "Pseudo_nitzschia_delicatissima_group", "pDA"]


def _run_source(s: dict, start, end) -> pd.DataFrame:
    kind = s["kind"]
    st, en = s.get("start", start), s.get("end", end)
    if kind == "existing":
        df = pd.read_csv(s["path"])
        df["source"] = s.get("name", "combined_full")
        df["method"] = df.get("method", "microscopy")
        return df
    if kind == "calhabmap_erddap":
        from sit_fuse.pipelines.habs.insitu.erddap_habmap import pull_calhabmap
        return pull_calhabmap(s.get("datasets"), st, en, s.get("sigma_dex"))
    if kind == "ifcb_dashboard":
        from sit_fuse.pipelines.habs.insitu.ifcb_dashboard import pull_ifcb
        return pull_ifcb(s["datasets"], st, en, s.get("options"), s.get("location_codes"),
                         s.get("max_bins"))
    if kind == "calcofi_venrick":
        from sit_fuse.pipelines.habs.insitu.calcofi import pull_venrick
        return pull_venrick(st, en)
    if kind == "calcofi_picoplankton":
        from sit_fuse.pipelines.habs.insitu.calcofi import pull_picoplankton
        return pull_picoplankton(st, en, s.get("max_depth_m", 5.0))
    if kind == "seabass_dir":
        from sit_fuse.pipelines.habs.insitu.seabass import read_seabass_dir
        return read_seabass_dir(s["path"], tuple(s.get("kinds", ("hplc", "ifcb", "fcm"))),
                                s.get("name", "seabass"))
    if kind == "local_table":
        from sit_fuse.pipelines.habs.insitu.local_table import read_local_table
        return read_local_table(s)
    raise ValueError(f"unknown source kind {kind}")


def inventory(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(df["time"], format="%m/%d/%y", errors="coerce")
    g = df.assign(_t=t).groupby(["source", "method"], dropna=False)
    inv = g.agg(rows=("Location_Code", "size"), stations=("Location_Code", "nunique"),
                start=("_t", "min"), end=("_t", "max"))
    for c in TARGETS:
        inv[f"n_{c}"] = g[c].apply(lambda x: pd.to_numeric(x, errors="coerce").notna().sum())
    return inv.reset_index()


def main(cfg: dict) -> pd.DataFrame:
    out = Path(cfg.get("out_dir", "insitu_augmented"))
    (out / "by_source").mkdir(parents=True, exist_ok=True)
    start, end = cfg.get("start", "2015-01-01"), cfg.get("end")
    frames = []
    for s in cfg["sources"]:
        if not s.get("enabled", True):
            continue
        name = s.get("name", s["kind"])
        try:
            df = _run_source(s, start, end)
        except Exception as e:
            print(f"[pull_insitu_hab] {name} failed: {e}")
            continue
        print(f"[pull_insitu_hab] {name}: {len(df)} rows")
        if len(df):
            write_calhabmap(df, out / "by_source" / f"{name}.csv")
            frames.append(df)
    merged = merge_sources(frames, priority=cfg.get("priority"))
    bbox = cfg.get("bbox")          # [lon_min, lat_min, lon_max, lat_max]
    if bbox is not None and len(merged):
        lo, la = pd.to_numeric(merged["longitude"]), pd.to_numeric(merged["latitude"])
        merged = merged[lo.between(bbox[0], bbox[2]) & la.between(bbox[1], bbox[3])]
    write_calhabmap(merged, out / "combined_augmented.csv")
    write_calhabmap(merged[CALHABMAP_COLUMNS], out / "combined_augmented_strict.csv", strict=True)
    inventory(merged).to_csv(out / "inventory.csv", index=False)
    issues = validate_calhabmap(out / "combined_augmented_strict.csv", strict=True)
    print("[pull_insitu_hab] validation:", issues or "ok")
    return merged


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yaml", required=True)
    a = ap.parse_args()
    with open(a.yaml) as f:
        main(yaml.safe_load(f))
