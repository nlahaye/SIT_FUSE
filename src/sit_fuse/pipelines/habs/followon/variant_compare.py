"""Paired comparison of HAB UQ variants on shared test samples.

Reads <uq.out_dir>/evaluation/hab_uq_test_predictions_<token>.csv of every variant, pairs rows on
(sample_uid, sensor) (or sample_uid only with --pool-sensors: per-sample mean probabilities), and
scores each variant and its difference from the baseline with a block bootstrap over split_group.

Scores (sample weights from the 'weight' column when present)
  rps                  ranked probability score
  rpss_vs_baseline     1 - RPS_variant / RPS_baseline
  nll                  -log p(observed bin)
  brier_ge<k>, auc_ge<k>   exceedance P(bin >= k), k = 1..C-1
  rate_ge<k>, mean_p_ge<k> observed rate and mean forecast probability (calibration in the large)
Strata: overall, year, season, sensor (and Location_Code with --by-site).
Also: test-sample counts per variant and, when models/matchups_<token>.parquet exists, matchup
row counts and median match distance per sensor.

  python -m sit_fuse.pipelines.habs.followon.variant_compare \
      --variant baseline=/mnt/.../uq_full_2023_2026_ifcb --variant anc_all=/mnt/.../followon/ancillary/anc_all/uq \
      --baseline baseline --token pseudo_nitzschia_seriata_bloom --out /mnt/.../followon/ancillary/compare
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def load_predictions(uq_dir: str, token: str) -> pd.DataFrame:
    p = Path(uq_dir) / "evaluation" / f"hab_uq_test_predictions_{token}.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    df["sample_uid"] = df["sample_uid"].astype(str)
    return df


def prob_cols(df: pd.DataFrame) -> list[str]:
    cols = sorted((c for c in df.columns if c.startswith("p_") and c[2:].isdigit()), key=lambda c: int(c[2:]))
    if not cols:
        raise ValueError("no p_<k> columns")
    return cols


def pool_sensors(df: pd.DataFrame) -> pd.DataFrame:
    pc = prob_cols(df)
    meta = df.groupby("sample_uid").first().drop(columns=pc)
    P = df.groupby("sample_uid")[pc].mean()
    out = meta.join(P).reset_index()
    out["sensor"] = "pooled"
    return out


def align(frames: dict[str, pd.DataFrame], keys=("sample_uid", "sensor")) -> dict[str, pd.DataFrame]:
    idx = None
    for df in frames.values():
        k = pd.MultiIndex.from_frame(df[list(keys)].astype(str))
        idx = k if idx is None else idx.intersection(k)
    out = {}
    for n, df in frames.items():
        d = df.set_index(pd.MultiIndex.from_frame(df[list(keys)].astype(str)))
        d = d[~d.index.duplicated()].loc[idx]
        out[n] = d.reset_index(drop=True)
    return out


def rps(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """RPS normalized by C-1 (as hab_uq_common.ranked_probability_score)."""
    C = P.shape[1]
    Ocdf = (np.arange(C)[None, :] >= y[:, None]).astype(float)
    return np.sum((np.cumsum(P, 1)[:, :-1] - Ocdf[:, :-1]) ** 2, 1) / max(C - 1, 1)


def auc(score: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    w = np.ones_like(score) if w is None else w
    order = np.argsort(score, kind="mergesort")
    s, yy, ww = score[order], y[order], w[order]
    # weighted Mann-Whitney with ties
    uniq, inv = np.unique(s, return_inverse=True)
    wp = np.bincount(inv, weights=ww * (yy == 1), minlength=uniq.size)
    wn = np.bincount(inv, weights=ww * (yy == 0), minlength=uniq.size)
    cn = np.cumsum(wn) - wn
    return float(np.sum(wp * (cn + 0.5 * wn)) / (wp.sum() * wn.sum()))


def scores(P: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict[str, float]:
    C = P.shape[1]
    out = {"n": int(len(y)), "rps": float(np.average(rps(P, y), weights=w)),
           "nll": float(np.average(-np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)), weights=w))}
    for k in range(1, C):
        pe = P[:, k:].sum(1)
        ye = (y >= k).astype(int)
        out[f"brier_ge{k}"] = float(np.average((pe - ye) ** 2, weights=w))
        out[f"auc_ge{k}"] = auc(pe, ye, w)
        out[f"rate_ge{k}"] = float(np.average(ye, weights=w))
        out[f"mean_p_ge{k}"] = float(np.average(pe, weights=w))
    return out


def _arrays(df):
    pc = prob_cols(df)
    P = df[pc].to_numpy(float)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    y = df["ref_label"].to_numpy().astype(int)
    w = df["weight"].to_numpy(float) if "weight" in df else np.ones(len(df))
    return P, y, w


def paired_block(frames: dict[str, pd.DataFrame], base: str, n_boot: int, seed: int) -> list[dict]:
    arr = {n: _arrays(d) for n, d in frames.items()}
    groups = frames[base]["split_group"].astype(str).to_numpy() if "split_group" in frames[base] else \
        frames[base]["sample_uid"].astype(str).to_numpy()
    codes, ug = pd.factorize(groups)
    members = [np.flatnonzero(codes == g) for g in range(len(ug))]
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([members[g] for g in rng.integers(0, len(ug), len(ug))]) for _ in range(n_boot)]
    pb = {n: scores(*a) for n, a in arr.items()}
    rows = []
    for n, a in arr.items():
        rec = {"variant": n, **pb[n]}
        rec["rpss_vs_baseline"] = 1 - pb[n]["rps"] / pb[base]["rps"] if pb[base]["rps"] > 0 else np.nan
        if n != base and n_boot > 0 and len(ug) > 3:
            keys = [k for k in pb[n] if k.startswith(("rps", "nll", "brier_ge", "auc_ge"))]
            dd = {k: [] for k in keys}
            Pb, yb, wb = arr[base]
            Pv, yv, wv = a
            for ix in draws:
                sv, sb = scores(Pv[ix], yv[ix], wv[ix]), scores(Pb[ix], yb[ix], wb[ix])
                for k in keys:
                    dd[k].append(sv[k] - sb[k])
            for k, v in dd.items():
                v = np.asarray(v, float)
                v = v[np.isfinite(v)]
                rec[f"d_{k}"] = pb[n][k] - pb[base][k]
                if v.size >= 10:
                    rec[f"d_{k}_lo95"], rec[f"d_{k}_hi95"] = np.quantile(v, [0.025, 0.975])
                    rec[f"d_{k}_p"] = float(min(1.0, 2 * min(np.mean(v <= 0), np.mean(v >= 0))))
        rec["n_groups"] = int(len(ug))
        rows.append(rec)
    return rows


def matchup_summary(uq_dir: str, token: str) -> pd.DataFrame | None:
    p = Path(uq_dir) / "models" / f"matchups_{token}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, columns=None)
    except Exception as e:  # pragma: no cover
        LOG.warning("matchups %s: %s", p, e)
        return None
    agg = {"sample_uid": "nunique"}
    if "match_dist_km" in df:
        agg["match_dist_km"] = "median"
    out = df.groupby(["sensor", "split"]).agg(agg).rename(columns={"sample_uid": "n_samples"})
    out["n_rows"] = df.groupby(["sensor", "split"]).size()
    return out.reset_index()


def compare(variants: dict[str, str], baseline: str, token: str, out: str, n_boot: int = 500, seed: int = 0,
            pool: bool = False, by_site: bool = False) -> pd.DataFrame:
    if baseline not in variants:
        raise ValueError(f"baseline {baseline!r} not among variants {list(variants)}")
    raw = {n: load_predictions(d, token) for n, d in variants.items()}
    counts = {n: {"n_rows": len(d), "n_samples": int(d["sample_uid"].nunique()),
                  "sensors": sorted(d["sensor"].astype(str).unique().tolist())} for n, d in raw.items()}
    if pool:
        raw = {n: pool_sensors(d) for n, d in raw.items()}
    al = align(raw, ("sample_uid", "sensor"))
    base = al[baseline]
    base["year"] = pd.to_datetime(base["date"]).dt.year.astype(str)
    strata = ["year", "season", "sensor"] + (["Location_Code"] if by_site else [])
    rows = [dict(stratum="overall", level="all", **r) for r in paired_block(al, baseline, n_boot, seed)]
    for col in strata:
        if col not in base:
            continue
        for lev, ix in base.groupby(col).indices.items():
            if len(ix) < 20:
                continue
            sub = {n: d.iloc[ix].reset_index(drop=True) for n, d in al.items()}
            rows += [dict(stratum=col, level=str(lev), **r)
                     for r in paired_block(sub, baseline, n_boot if col != "Location_Code" else 0, seed)]
    res = pd.DataFrame(rows)
    os.makedirs(out, exist_ok=True)
    tag = token + ("_pooled" if pool else "")
    res.to_csv(Path(out) / f"variant_compare_{tag}.csv", index=False)
    mus = []
    for n, d in variants.items():
        m = matchup_summary(d, token)
        if m is not None:
            mus.append(m.assign(variant=n))
    if mus:
        pd.concat(mus).to_csv(Path(out) / f"variant_matchups_{token}.csv", index=False)
    with open(Path(out) / f"variant_compare_{tag}.json", "w") as f:
        json.dump({"variants": variants, "baseline": baseline, "counts": counts,
                   "n_paired": int(len(base))}, f, indent=1)
    LOG.info("%s: %d paired rows; overall rps %s", token, len(base),
             res[res.stratum == "overall"].set_index("variant")["rps"].round(4).to_dict())
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", action="append", required=True, help="name=uq_out_dir")
    p.add_argument("--baseline", required=True)
    p.add_argument("--token", action="append", required=True, help="species fname token (repeatable)")
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--pool-sensors", action="store_true")
    p.add_argument("--by-site", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    v = dict(x.split("=", 1) for x in a.variant)
    for t in a.token:
        try:
            compare(v, a.baseline, t, a.out, a.n_boot, pool=a.pool_sensors, by_site=a.by_site)
        except FileNotFoundError as e:
            LOG.warning("%s skipped: %s", t, e)


if __name__ == "__main__":
    main()
