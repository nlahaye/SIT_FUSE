"""
Input / product coverage audit per platform x year: finds where a platform drops out of the chain
(Rrs on disk -> inference scene list -> head outputs -> per-sensor UQ products -> fused days).

    python -m sit_fuse.pipelines.habs.hab_input_audit -y <hab yaml> --window 2023-01-01:2023-12-31 \
        [--instruments olci,modis] [--species pDA,...] [--out audit.csv] [--require s3a,s3b,modis]

Per (instrument, key) of the HAB YAML, read the model YAML exactly as the heads stage does
(hab_inference_pipeline.model_config_path / apply_data_overrides / scene_entries with the window and
inference.extend_input_lists), then per platform (file-family prefix) and year count

  disk_days    days with Rrs files on disk in the directories of the listed entries, matched by
               the listed file family (basename before the date)
  scene_days   days in the inference scene list (listed + extend_input_lists additions)
  head_days    days with {PREFIX}.{YYYYMMDD}*.head_outputs.tif in the instrument out_dirs
  uq_days      days with a per-sensor *.uq.tif, one column per species (uq.out_dir/products/<sensor>)
  fused_days   days with a fused product per species (uq.out_dir/fused), in a second table

and a diagnosis: no_rrs_on_disk (sync), naming_mismatch (files on disk but not in the scene list:
the listed naming template does not match, e.g. NRT vs standard or a new processing version),
heads_missing (heads not run / failed), uq_missing (infer not run for that window).
--require <sensors>: exit 1 when any listed platform has zero head_days (--require-stage heads) or
zero head_days / uq_days (--require-stage uq) in a year of the window (launcher gates).
"""

from __future__ import annotations

import argparse
import copy
import glob
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("hab-input-audit")


def _years(windows):
    out = []
    for a, b in windows:
        out += list(range(int(str(a)[:4]), int(str(b)[:4]) + 1))
    return sorted(set(out))


def _in_windows(day: str, windows) -> bool:
    return any(str(a).replace("-", "")[:8] <= day <= str(b).replace("-", "")[:8] for a, b in windows)


def _sensor_of(base: str, prefixes: dict[str, str]) -> str | None:
    for s, p in sorted(prefixes.items(), key=lambda kv: -len(kv[1])):
        if p and base.startswith(p):
            return s
    for s, p in prefixes.items():       # file families of the Rrs inputs (e.g. S3A_OLCI_ERRNT / S3A_OLCI_EFRNT)
        stem = p.split("_")[0] if p else None
        if stem and base.startswith(stem + "_"):
            return s
    return None


def _days(paths) -> set[str]:
    out = set()
    for p in paths:
        m = re.search(r"\.(\d{8})[\._]", os.path.basename(p)) or re.search(r"(\d{8})", os.path.basename(p))
        if m:
            out.add(m.group(1))
    return out


def audit_instrument(yml, instrument, key, windows, prefixes) -> list[dict]:
    from sit_fuse.pipelines.habs import hab_inference_pipeline as hip
    inf = copy.deepcopy(yml.get("inference", {}) or {})
    inf["date_windows"] = [list(w) for w in windows]
    cfg_path = hip.model_config_path(yml, instrument, key)
    sf_yaml = hip._read_yaml(cfg_path)
    hip.apply_data_overrides(sf_yaml, inf)
    out_dir = hip.instrument_out_dir(yml, instrument, key, sf_yaml)
    entries = hip.scene_entries(sf_yaml, inf)
    lists = inf.get("input_lists") or ["files_test", "files_train"]
    listed = [e for k in lists for e in (sf_yaml["data"].get(k) or [])]
    fams = hip._entry_templates(listed)
    # disk: directories and family of the listed entries
    disk: dict[str, set[str]] = {}
    for fam, tpls in fams.items():
        s = _sensor_of(fam, prefixes)
        dirs = {os.path.dirname(str(x)) for tpl, _ in tpls for x in tpl}
        found = set()
        for d in dirs:
            found |= _days(glob.glob(os.path.join(d or ".", os.path.basename(fam) + "*")))
        disk.setdefault(s or fam, set()).update(x for x in found if _in_windows(x, windows))
    scenes: dict[str, set[str]] = {}
    for e in entries:
        b, d = hip._entry_base(e), hip._entry_date(e)
        s = _sensor_of(b, prefixes) or b.split(d)[0] if d else None
        if d:
            scenes.setdefault(s, set()).add(d)
    rows = []
    for s in sorted(set(disk) | set(scenes) | {x for x in prefixes if prefixes[x] and _belongs(yml, instrument, x)}):
        p = prefixes.get(s)
        heads = set()
        if p:
            heads = {x for x in _days(glob.glob(os.path.join(out_dir, f"{p}.*{hip.HEAD_SUFFIX}")))
                     if _in_windows(x, windows)}
        for y in _years(windows):
            f = (lambda S: len({x for x in S if x.startswith(str(y))}))
            rows.append({"instrument": instrument, "key": key, "sensor": s, "year": y, "model_yaml": cfg_path,
                         "out_dir": out_dir, "disk_days": f(disk.get(s, set())),
                         "scene_days": f(scenes.get(s, set())), "head_days": f(heads)})
    return rows


def _belongs(yml, instrument, sensor) -> bool:
    try:
        from sit_fuse.pipelines.habs.hab_sensor_groups import instrument_for_sensor
        return instrument_for_sensor(yml.get("instruments", {}) or {}, sensor) == instrument
    except Exception:  # pragma: no cover
        return sensor == instrument


def product_days(yml, species, windows) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sit_fuse.pipelines.habs.uq.hab_uq_common import SENSORS, species_spec
    uq = yml.get("uq", {}) or {}
    root = Path(uq.get("out_dir", os.path.join(yml.get("final_product_dir", "."), "uq")))
    rows, frows = [], []
    for sp in species:
        tok = species_spec(sp, (uq.get("species") or {}).get(sp)).fname_token
        for s, info in SENSORS.items():
            days = {x for x in _days(glob.glob(str(root / "products" / s / f"{info.prefix}.*.{tok}.uq.tif")))
                    if _in_windows(x, windows)}
            for y in _years(windows):
                rows.append({"sensor": s, "year": y, "species": sp,
                             "uq_days": len({x for x in days if x.startswith(str(y))})})
        fd = {x for x in _days(glob.glob(str(root / "fused" / f"SITFUSE_MULTI.*.{tok}.fused_uq.tif")))
              if _in_windows(x, windows)}
        for y in _years(windows):
            frows.append({"species": sp, "year": y, "fused_days": len({x for x in fd if x.startswith(str(y))})})
    return pd.DataFrame(rows), pd.DataFrame(frows)


def diagnose(r) -> str:
    if r.get("disk_days", 0) == 0:
        return "no_rrs_on_disk"
    if r.get("scene_days", 0) < 0.5 * r["disk_days"]:
        return "naming_mismatch"
    if r.get("head_days", 0) < 0.8 * r.get("scene_days", 0):
        return "heads_missing"
    uq = [v for k, v in r.items() if k.startswith("uq_days_") and pd.notna(v)]
    if uq and min(uq) < 0.8 * r.get("head_days", 0):
        return "uq_missing"
    return "ok"


def run_audit(yml, windows, instruments=None, species=None, out=None, require=None,
              require_stage: str = "uq") -> int:
    from sit_fuse.pipelines.habs import hab_inference_pipeline as hip
    from sit_fuse.pipelines.habs.uq.hab_uq_common import SENSORS
    prefixes = {s: i.prefix for s, i in SENSORS.items()}
    rows = []
    for inst, key in hip._select(yml, instruments):
        try:
            rows += audit_instrument(yml, inst, key, windows, prefixes)
        except Exception as e:      # report and continue (e.g. missing model YAML)
            LOG.error("%s/%s: audit failed: %s", inst, key, e)
            rows.append({"instrument": inst, "key": key, "sensor": inst, "error": str(e)})
    df = pd.DataFrame(rows)
    sp = species or list((yml.get("uq", {}) or {}).get("species_run") or [])
    fused = pd.DataFrame()
    if sp and len(df) and "year" in df:
        uqd, fused = product_days(yml, sp, windows)
        if len(uqd):
            wide = uqd.pivot_table(index=["sensor", "year"], columns="species", values="uq_days").add_prefix("uq_days_")
            df = df.merge(wide.reset_index(), on=["sensor", "year"], how="left")
    if len(df) and "disk_days" in df:
        df["diagnosis"] = [diagnose(r) if pd.notna(r.get("disk_days")) else "error" for r in df.to_dict("records")]
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(df.drop(columns=[c for c in ("model_yaml", "out_dir") if c in df]).to_string(index=False))
        if len(fused):
            print(fused.pivot_table(index="year", columns="species", values="fused_days").to_string())
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        if len(fused):
            fused.to_csv(str(out).replace(".csv", "_fused.csv"), index=False)
    bad = []
    for s in require or []:
        g = df[df["sensor"] == s] if "sensor" in df else df.iloc[:0]
        if g.empty:
            bad.append(f"{s}: not in any instrument")
            continue
        for _, r in g.iterrows():
            uq = [r[c] for c in g.columns if c.startswith("uq_days_")]
            if r.get("head_days", 0) == 0 or (require_stage == "uq" and uq and min(uq) == 0):
                bad.append(f"{s} {r.get('year')}: heads {r.get('head_days')}, uq {uq} ({r.get('diagnosis')})")
    for b in bad:
        LOG.error("required platform missing: %s", b)
    return 1 if bad else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--window", action="append", required=True, metavar="START:END")
    ap.add_argument("--instruments", default=None)
    ap.add_argument("--species", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--require", default=None, help="comma list of platforms that must have heads (and uq) days")
    ap.add_argument("--require-stage", choices=["heads", "uq"], default="uq",
                    help="heads: gate on head outputs only (before infer); uq: heads and per-sensor products")
    a = ap.parse_args()
    from sit_fuse.pipelines.habs.hab_inference_pipeline import _read_yaml
    sys.exit(run_audit(_read_yaml(a.yaml), [w.split(":", 1) for w in a.window],
                       None if a.instruments is None else a.instruments.split(","),
                       None if a.species is None else a.species.split(","), a.out,
                       None if a.require is None else a.require.split(","), a.require_stage))
