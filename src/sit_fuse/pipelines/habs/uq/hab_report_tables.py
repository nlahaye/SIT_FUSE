"""Report tables for HAB UQ runs (management / paper tables) from finished evaluation outputs.

Reads <uq.out_dir>/evaluation/hab_uq_evaluation_<token>.json and hab_uq_test_predictions_<token>.csv
(+ hab_uq_fused_test_predictions_<token>.csv when present) of one or more runs and writes

  skill_<tag>.csv          per run x species: weighted F1 (per sensor, fused, original method,
                           always-most-common-bin), QWK (95% CI), RPSS, recalibrated 90% coverage,
                           test samples / rows
  year_calibration_<tag>.csv  per run x species x product (per_sensor | fused) x year x alert threshold:
                           observed rate, mean forecast probability, ratio, weighted F1, QWK
  subsets_<tag>.csv        evaluation subsets (oos_piers, ifcb, ...) when present

Always-most-common-bin is the test-set majority bin (weighted), an optimistic reference.
Runs are name=uq_out_dir; several runs (e.g. calibration variants) give one long table each.

  python -m sit_fuse.pipelines.habs.uq.hab_report_tables \
      --run production=/mnt/.../uq_full_2023_2026_ifcb --run year_balanced=/mnt/.../calib/year_balanced/uq \
      --out /mnt/.../reports/next_steps
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("hab-report-tables")


def _wf1(y, p, w) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="weighted", sample_weight=w, zero_division=0))


def _qwk(y, p, w, C) -> float:
    from sklearn.metrics import cohen_kappa_score
    try:
        return float(cohen_kappa_score(y, p, weights="quadratic", labels=list(range(C)), sample_weight=w))
    except ValueError:
        return float("nan")


def _get(d, *keys, default=np.nan):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _pcols(df):
    return sorted((c for c in df.columns if c.startswith("p_") and c[2:].isdigit()), key=lambda c: int(c[2:]))


def skill_row(run: str, ev: Path, token: str) -> dict | None:
    jp, cp = ev / f"hab_uq_evaluation_{token}.json", ev / f"hab_uq_test_predictions_{token}.csv"
    if not jp.exists():
        return None
    j = json.loads(jp.read_text())
    o = j.get("overall") or {}
    row = {"run": run, "species": j.get("species", token), "token": token,
           "wf1_per_sensor": _get(o, "uq", "weighted_f1"),
           "wf1_original_method": _get(o, "production_baseline", "weighted_f1"),
           "wf1_fused": _get(j, "fused", "uq", "weighted_f1"),
           "qwk": _get(o, "uq", "qwk"), "qwk_lo95": _get(j, "overall_ci", "qwk", "lo95"),
           "qwk_hi95": _get(j, "overall_ci", "qwk", "hi95"),
           "qwk_original_method": _get(o, "production_baseline", "qwk"),
           "qwk_fused": _get(j, "fused", "uq", "qwk"),
           "rpss": _get(o, "uq", "rpss_vs_climatology"), "rpss_fused": _get(j, "fused", "uq", "rpss_vs_climatology"),
           "coverage90_recal": _get(o, "uq", "interval_90_recal_coverage"),
           "coverage90_nominal": _get(o, "uq", "interval_90_coverage"),
           "model": _get(j, "run_config", "model_trial", default=None),
           "calibration_reweight": json.dumps(_get(j, "run_config", "calibration_reweight", default=None))}
    if cp.exists():
        d = pd.read_csv(cp)
        y, w = d["ref_label"].astype(int).to_numpy(), d["weight"].to_numpy(float)
        maj = int(pd.Series(w).groupby(y).sum().idxmax())
        row.update(wf1_most_common_bin=_wf1(y, np.full(len(y), maj), w), most_common_bin=maj,
                   n_test_samples=int(d["sample_uid"].nunique()), n_test_rows=int(len(d)),
                   test_start=str(d["date"].min())[:10], test_end=str(d["date"].max())[:10])
    return row


def year_rows(run: str, ev: Path, token: str) -> list[dict]:
    out = []
    for product, fname in (("per_sensor", f"hab_uq_test_predictions_{token}.csv"),
                           ("fused", f"hab_uq_fused_test_predictions_{token}.csv")):
        cp = ev / fname
        if not cp.exists():
            continue
        d = pd.read_csv(cp)
        pc = _pcols(d)
        if not pc or "value" not in d:
            continue
        P = d[pc].to_numpy(float)
        P /= np.clip(P.sum(1, keepdims=True), 1e-12, None)
        C = P.shape[1]
        d["year"] = pd.to_datetime(d["date"]).dt.year
        pred = d["pred_label"].astype(int).to_numpy() if "pred_label" in d else P.argmax(1)
        j = json.loads((ev / f"hab_uq_evaluation_{token}.json").read_text()) \
            if (ev / f"hab_uq_evaluation_{token}.json").exists() else {}
        blk = (j.get("overall") or {}).get("uq", {}).get("exceedance") or {}
        thresholds = [float(t) for t in blk] or [None]
        edges = (j.get("spec") or {}).get("edges")
        for yr, ix in d.groupby("year").indices.items():
            g = d.iloc[ix]
            y, w = g["ref_label"].astype(int).to_numpy(), g["weight"].to_numpy(float)
            base = {"run": run, "token": token, "product": product, "year": int(yr),
                    "n_samples": int(g["sample_uid"].nunique()), "wf1": _wf1(y, pred[ix], w),
                    "qwk": _qwk(y, pred[ix], w, C)}
            for t in thresholds:
                rec = dict(base, threshold=t)
                if t is not None:
                    obs = (g["value"].to_numpy(float) >= t).astype(float)
                    pe = _exceed(P[ix], t, edges, j, token)
                    rec.update(observed_rate=float(np.average(obs, weights=w)),
                               mean_p=float(np.average(pe, weights=w)) if pe is not None else np.nan)
                    rec["ratio"] = rec["mean_p"] / rec["observed_rate"] if rec["observed_rate"] > 0 else np.nan
                out.append(rec)
    return out


def _exceed(P, t, edges, j, token):
    """P(value >= t) from class probabilities (bin edges from the species spec)."""
    try:
        from sit_fuse.pipelines.habs.uq.hab_uq_common import exceedance_probability, species_spec
        spec = species_spec(j.get("species") or token)
        return exceedance_probability(P, spec.edges, t, spec.detection_floor)
    except Exception as e:                                  # noqa: BLE001 - report without the column
        LOG.warning("%s: exceedance at %s unavailable (%s)", token, t, e)
        return None


def collect(runs: dict[str, str], out: str, tokens: list[str] | None = None, tag: str = "report") -> dict[str, str]:
    Path(out).mkdir(parents=True, exist_ok=True)
    skills, years, subs = [], [], []
    for run, d in runs.items():
        ev = Path(d) / "evaluation"
        toks = tokens or sorted(p.stem.replace("hab_uq_evaluation_", "") for p in ev.glob("hab_uq_evaluation_*.json"))
        for t in toks:
            r = skill_row(run, ev, t)
            if r is None:
                LOG.warning("%s: no evaluation for %s", run, t)
                continue
            skills.append(r)
            years += year_rows(run, ev, t)
        ss = ev / "subsets" / "subset_summary.csv"
        if ss.exists():
            subs.append(pd.read_csv(ss).assign(run=run))
    paths = {}
    for name, rows in (("skill", skills), ("year_calibration", years)):
        if rows:
            p = Path(out) / f"{name}_{tag}.csv"
            pd.DataFrame(rows).to_csv(p, index=False)
            paths[name] = str(p)
    if subs:
        p = Path(out) / f"subsets_{tag}.csv"
        pd.concat(subs, ignore_index=True).to_csv(p, index=False)
        paths["subsets"] = str(p)
    LOG.info("wrote %s", paths)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="name=uq_out_dir (repeatable)")
    ap.add_argument("--token", action="append", default=None, help="species fname token(s); default: all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="report")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collect(dict(x.split("=", 1) for x in a.run), a.out, a.token, a.tag)


if __name__ == "__main__":
    main()
