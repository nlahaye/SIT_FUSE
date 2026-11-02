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
  {out_dir}/coverage/                     per-product coverage report (see coverage_report.py)

Restart with previously merged non-IFCB output and cache-only IFCB:

  python -m sit_fuse.pipelines.habs.insitu.pull_insitu_hab -y sources.yaml \
      --only ifcb_dashboard --ifcb-cache-only --existing /data/habs/other_sources.csv

  --only NAME ...        run only these source names (others in the YAML are skipped)
  --ifcb-cache-only      ifcb_dashboard sources make no network requests; cached bins only
  --existing CSV ...     add previously produced tables (keeps their source/method columns)
  --out-dir DIR          override out_dir
  --no-coverage          skip the coverage report
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
        df = pd.read_csv(s["path"], low_memory=False)
        # keep per-row provenance when the table is a previous pull_insitu_hab output
        name = s.get("name", "combined_full")
        df["source"] = df["source"].fillna(name) if "source" in df.columns else name
        df["method"] = df["method"].fillna("microscopy") if "method" in df.columns else "microscopy"
        drop = set(s.get("exclude_sources") or [])      # sources re-pulled in this run
        if drop:
            df = df[~df["source"].isin(drop)]
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
        if len(df) and s["kind"] != "existing":
            write_calhabmap(df, out / "by_source" / f"{name}.csv")
        if len(df):
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
    if cfg.get("coverage", True) and len(merged):
        from sit_fuse.pipelines.habs.insitu.coverage_report import coverage_report
        cov = cfg.get("coverage_options") or {}
        coverage_report(merged, out / "coverage", start=cov.get("start", start), end=cov.get("end", end),
                        **{k: v for k, v in cov.items() if k not in ("start", "end")})
        print(f"[pull_insitu_hab] coverage report: {out / 'coverage' / 'coverage_report.md'}")
    return merged


def apply_cli(cfg: dict, a) -> dict:
    cfg = dict(cfg)
    srcs = [dict(s) for s in cfg["sources"]]
    if a.only:
        for s in srcs:
            s["enabled"] = s.get("name", s["kind"]) in a.only
    if a.ifcb_cache_only:
        for s in srcs:
            if s["kind"] == "ifcb_dashboard":
                s["options"] = {**(s.get("options") or {}), "cache_only": True}
    rerun = [s.get("name", s["kind"]) for s in srcs if s.get("enabled", True) and s["kind"] != "existing"]
    for i, path in enumerate(a.existing or []):
        srcs.insert(i, {"kind": "existing", "name": Path(path).stem, "path": path, "exclude_sources": rerun})
    cfg["sources"] = srcs
    if a.out_dir:
        cfg["out_dir"] = a.out_dir
    if a.no_coverage:
        cfg["coverage"] = False
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--only", nargs="+", help="source names to run")
    ap.add_argument("--ifcb-cache-only", action="store_true", help="IFCB: cached bins only, no requests")
    ap.add_argument("--existing", nargs="+", help="previously produced CSV(s) to merge")
    ap.add_argument("--out-dir")
    ap.add_argument("--no-coverage", action="store_true")
    a = ap.parse_args()
    with open(a.yaml) as f:
        main(apply_cli(yaml.safe_load(f), a))
