"""Interpretability of the HAB UQ heads (model-level, group-level and cluster-level).

Subcommands (all read a trained UQ output directory; nothing is retrained)
  importance   grouped permutation importance of the calibrated sample-level predictions on the
               test split; permutation within season x site strata (keeps the seasonal / site
               structure, so importance is not credited to proxies of season or site). Losses:
               RPS and exceedance Brier at the species alert bin. Per-sample loss differences
               are kept, so groups / species can be contrasted with a block bootstrap.
               Optional SHAP (shap installed): TreeSHAP of P(bin >= alert) for random forests,
               permutation SHAP otherwise, aggregated to feature groups.
  contrast     seriata vs delicatissima (or any two species): per-group importance differences
               with block-bootstrap CI and the Spearman correlation of the group rankings.
  conditional  does the species product add information beyond the total-phytoplankton product?
               OLS of the observed log10 concentration on the total-phytoplankton expected bin,
               then adding the species expected bin and P(>= alert): Delta R^2 with block-bootstrap CI.
  composites   cluster -> environment: per parent cluster, mean SST / SST anomaly / SST gradient /
               upwelling / standard products at the cluster's pixels on sampled scene days, and a
               Mann-Whitney test of each cluster vs all other pixels (BH-adjusted q values).

The physics-only baseline is a HAB variant with features.groups [ancillary, physical]
(variants.py), scored against the production heads with variant_compare.

  python -m sit_fuse.pipelines.habs.followon.interpret importance -y HAB.yaml --species Pseudo_nitzschia_seriata_group \
      --out /mnt/.../followon/interpret --n-repeats 5 --shap 2000
  python -m sit_fuse.pipelines.habs.followon.interpret contrast --a DIR/importance_<tokA>_samples.parquet \
      --b DIR/importance_<tokB>_samples.parquet --out DIR
  python -m sit_fuse.pipelines.habs.followon.interpret conditional -y HAB.yaml --species Pseudo_nitzschia_seriata_group \
      --reference Total_Phytoplankton --out DIR
  python -m sit_fuse.pipelines.habs.followon.interpret composites -y HAB.yaml --sensors s3a,s3b,jpss1 \
      --start 2023-01-01 --end 2026-09-30 --every 5 --out DIR
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- feature groups
def feature_group(c: str) -> str:
    """Group of a UQ feature column (mirrors hab_uq_common.select_feature_columns + extras)."""
    from sit_fuse.train.sf_uq_utils import CLUSTER_STRUCTURE_PREFIXES
    if c.startswith("anc_"):
        k = c[4:]
        return ("ancillary:sst" if k.startswith("sst") else "ancillary:upwelling" if k.startswith(("cuti", "beuti"))
                else "ancillary:discharge" if k.startswith("q_") else "ancillary:indices")
    if c.startswith("phys_"):
        return "physical"
    if c.startswith("sensor_is_"):
        return "sensor"
    if c.startswith("sp_tier_"):
        return "spatial_tier"
    if c == "abs_day_offset":
        return "timing"
    if c.startswith(CLUSTER_STRUCTURE_PREFIXES) or "_delta_r" in c:
        return "cluster_structure"
    if c.startswith("nbr_r"):
        return "neighborhood_context" if ("_sa_" in c or "_vote_" in c) else "neighborhood"
    if c.startswith(("sa_parent_", "sa_joint_")):
        return "soft_assignment"
    if c.startswith("agree_joint_"):
        return "agreement"
    if c.startswith("assign_vote"):
        return "assignment_ensemble"
    if "_top" in c and c.endswith("_prob"):
        return "head_topk_probs"
    return "representation"


# ----------------------------------------------------------------------------- tables
def save_table(df: pd.DataFrame, stem: Path) -> Path:
    """<stem>.parquet, or <stem>.csv.gz without a parquet engine."""
    try:
        p = Path(str(stem) + ".parquet")
        df.to_parquet(p)
    except ImportError:
        p = Path(str(stem) + ".csv.gz")
        df.to_csv(p, index=False)
    return p


def read_table(path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        for alt in (Path(str(p).replace(".parquet", ".csv.gz")), Path(str(p) + ".parquet"), Path(str(p) + ".csv.gz")):
            if alt.exists():
                p = alt
                break
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


# ----------------------------------------------------------------------------- loading
def _uq_root(yml) -> Path:
    uq = yml["uq"]
    return Path(uq.get("out_dir", os.path.join(yml.get("final_product_dir", "."), "uq")))


def load_bundle_frame(yml, use_key):
    import joblib
    from sit_fuse.pipelines.habs.uq.hab_uq_common import species_spec
    spec = species_spec(use_key, (yml["uq"].get("species") or {}).get(use_key))
    root = _uq_root(yml) / "models"
    b = joblib.load(root / f"hab_uq_{spec.fname_token}.joblib")
    mpq, mcsv = root / f"matchups_{spec.fname_token}.parquet", root / f"matchups_{spec.fname_token}.csv.gz"
    frame = read_table(mpq if mpq.exists() else mcsv)
    return b, frame, spec


def alert_bin(spec) -> int:
    """First bin whose lower edge is >= the first alert threshold (default: upper half of the bins)."""
    if spec.alert_thresholds:
        e = np.asarray(spec.edges, float)
        k = int(np.searchsorted(e, float(spec.alert_thresholds[0]), side="left"))
        return int(np.clip(k, 1, spec.n_classes - 1))
    return spec.n_classes // 2


class Predictor:
    """Calibrated sample-level probabilities of a UQ bundle for a fixed set of matchup rows."""

    def __init__(self, bundle, rows: pd.DataFrame):
        from sit_fuse.pipelines.habs.uq.hab_uq_train import add_sensor_columns
        self.b = bundle
        self.C = bundle["n_classes"]
        self.cols = list(bundle["feature_columns"])
        r = add_sensor_columns(rows, bundle["sensors"])
        for c in self.cols:
            if c not in r.columns:
                r[c] = 0.0
        self.rows = r.reset_index(drop=True)
        self.X = self.rows[self.cols].to_numpy(np.float64)

    def predict(self, X=None):
        from sit_fuse.train.sf_uq_utils import full_proba
        from sit_fuse.pipelines.habs.uq.hab_uq_matchups import sample_level
        X = self.X if X is None else X
        Xdf = pd.DataFrame(X, columns=self.cols)
        outP, outM = [], []
        for s in self.b["sensors"]:
            m = (self.rows["sensor"] == s).to_numpy()
            if not m.any():
                continue
            key = "pooled" if self.b["model_scope"] == "pooled" else s
            P = full_proba(self.b["fits"][key]["model"], Xdf[m], self.C)
            Ps, meta = sample_level(P, self.rows[m])
            cal = self.b["calibrators"].get(s) or self.b["calibrators"].get("_pooled")
            outP.append(cal.transform(Ps) if cal is not None else Ps)
            outM.append(meta)
        return np.vstack(outP), pd.concat(outM, ignore_index=True)


def _losses(P, y, k_alert):
    from sit_fuse.pipelines.habs.uq.hab_uq_common import ranked_probability_score
    pe = P[:, k_alert:].sum(1)
    return {"rps": ranked_probability_score(P, y), "brier_alert": (pe - (y >= k_alert)) ** 2}


def _strata(rows: pd.DataFrame) -> np.ndarray:
    m = pd.to_datetime(rows["date"]).dt.month
    season = m.map(lambda x: {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2}.get(x, 3))
    site = rows["Location_Code"].astype(str) if "Location_Code" in rows else ""
    return pd.factorize(season.astype(str) + "|" + site)[0]


def permutation_importance(pred: Predictor, k_alert: int, n_repeats: int = 5, seed: int = 0,
                           groups: dict[str, list[int]] | None = None):
    """Per-sample loss increases (permuted - original), averaged over repeats, per feature group.

    Permutation is of whole samples within season x site strata: every pixel row of one
    sample takes the feature values of the rows of another sample of the same stratum
    (pixel counts differ, so rows are drawn with replacement from the donor sample)."""
    rng = np.random.default_rng(seed)
    P0, meta = pred.predict()
    y = meta["ref_label"].to_numpy().astype(int)
    L0 = _losses(P0, y, k_alert)
    if groups is None:
        g = {}
        for j, c in enumerate(pred.cols):
            g.setdefault(feature_group(c), []).append(j)
        groups = g
    rows = pred.rows
    skey = rows["sample_uid"].astype(str) + "|" + rows["sensor"].astype(str)
    codes, uniq = pd.factorize(skey)
    members = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    strat_rows = _strata(rows)
    strat_s = np.array([strat_rows[m[0]] for m in members])
    out = {}
    for gname, idx in groups.items():
        acc = {k: np.zeros(len(y)) for k in L0}
        for r in range(n_repeats):
            X = pred.X.copy()
            for st in np.unique(strat_s):
                ss = np.flatnonzero(strat_s == st)
                if ss.size < 2:
                    continue
                donor = rng.permutation(ss)
                for a, d in zip(ss, donor):
                    ra, rd = members[a], members[d]
                    X[np.ix_(ra, idx)] = pred.X[np.ix_(rng.choice(rd, ra.size, replace=True), idx)]
            P, m2 = pred.predict(X)
            L = _losses(P, m2["ref_label"].to_numpy().astype(int), k_alert)
            for k in acc:
                acc[k] += (L[k] - L0[k]) / n_repeats
        out[gname] = acc
        LOG.info("permutation %s (%d cols): dRPS %.4f dBrier %.4f", gname, len(idx),
                 np.average(acc["rps"], weights=meta["weight"]), np.average(acc["brier_alert"], weights=meta["weight"]))
    samples = meta[["sample_uid", "sensor", "date", "split_group", "Location_Code", "weight", "ref_label"]].copy()
    for gname, acc in out.items():
        for k, v in acc.items():
            samples[f"d_{k}__{gname}"] = v
    for k, v in L0.items():
        samples[f"base_{k}"] = v
    return samples, {g: len(i) for g, i in groups.items()}


def block_ci(values: np.ndarray, weights: np.ndarray, groups: np.ndarray, n_boot=500, seed=0):
    codes, ug = pd.factorize(groups)
    members = [np.flatnonzero(codes == g) for g in range(len(ug))]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        ix = np.concatenate([members[g] for g in rng.integers(0, len(ug), len(ug))])
        d.append(np.average(values[ix], weights=weights[ix]))
    return np.quantile(d, [0.025, 0.975]) if d else (np.nan, np.nan)


def summarize_importance(samples: pd.DataFrame, n_boot=500) -> pd.DataFrame:
    rows = []
    w = samples["weight"].to_numpy(float)
    g = samples["split_group"].astype(str).to_numpy()
    for c in [c for c in samples.columns if c.startswith("d_")]:
        loss, grp = c[2:].split("__", 1)
        v = samples[c].to_numpy(float)
        lo, hi = block_ci(v, w, g, n_boot)
        base = np.average(samples[f"base_{loss}"], weights=w)
        rows.append({"group": grp, "loss": loss, "delta": float(np.average(v, weights=w)), "lo95": lo, "hi95": hi,
                     "relative": float(np.average(v, weights=w) / base) if base > 0 else np.nan})
    return pd.DataFrame(rows).sort_values(["loss", "delta"], ascending=[True, False])


def shap_groups(pred: Predictor, k_alert: int, n: int = 2000, seed: int = 0) -> pd.DataFrame | None:
    try:
        import shap
    except ImportError:
        LOG.warning("shap not installed; SHAP step skipped (pip install shap)")
        return None
    from sit_fuse.train.sf_uq_utils import full_proba
    rng = np.random.default_rng(seed)
    ix = rng.choice(len(pred.X), min(n, len(pred.X)), replace=False)
    X = pred.X[ix]
    key = "pooled" if pred.b["model_scope"] == "pooled" else pred.b["sensors"][0]
    model = pred.b["fits"][key]["model"]
    name = type(model).__name__

    def f(Z):
        return full_proba(model, pd.DataFrame(Z, columns=pred.cols), pred.C)[:, k_alert:].sum(1)
    if "RandomForest" in name or "ExtraTrees" in name:
        ex = shap.TreeExplainer(model)
        sv = ex.shap_values(pd.DataFrame(X, columns=pred.cols))
        cls = list(model.classes_)
        arr = np.asarray(sv)
        if arr.ndim == 3 and arr.shape[-1] == len(cls):       # (n, d, C)
            S = sum(arr[:, :, i] for i, c in enumerate(cls) if int(c) >= k_alert)
        else:                                                  # list of (n, d) per class
            S = sum(np.asarray(sv[i]) for i, c in enumerate(cls) if int(c) >= k_alert)
        method = "tree"
    else:
        bg = X[rng.choice(len(X), min(100, len(X)), replace=False)]
        ex = shap.PermutationExplainer(f, bg)
        S = ex(X[: min(len(X), 500)]).values
        method = "permutation"
    S = np.asarray(S)
    rows = {}
    for j, c in enumerate(pred.cols):
        rows.setdefault(feature_group(c), []).append(j)
    return pd.DataFrame([{"group": g, "mean_abs_shap": float(np.abs(S[:, idx].sum(1)).mean()),
                          "n_features": len(idx), "method": method, "model": name} for g, idx in rows.items()]) \
        .sort_values("mean_abs_shap", ascending=False)


def cmd_importance(a):
    import yaml
    with open(a.yaml) as f:
        yml = yaml.safe_load(f)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for sp in a.species:
        b, frame, spec = load_bundle_frame(yml, sp)
        rows = frame[frame["split"] == a.split]
        if a.max_samples:
            keep = rows["sample_uid"].drop_duplicates().sample(min(a.max_samples, rows["sample_uid"].nunique()),
                                                                random_state=0)
            rows = rows[rows["sample_uid"].isin(keep)]
        pred = Predictor(b, rows)
        k = alert_bin(spec)
        samples, sizes = permutation_importance(pred, k, a.n_repeats, a.seed)
        tok = spec.fname_token
        save_table(samples, out / f"importance_{tok}_samples")
        summ = summarize_importance(samples, a.n_boot).assign(n_features=lambda d: d["group"].map(sizes))
        summ.to_csv(out / f"importance_{tok}.csv", index=False)
        if a.shap:
            sh = shap_groups(pred, k, a.shap, a.seed)
            if sh is not None:
                sh.to_csv(out / f"shap_groups_{tok}.csv", index=False)
        with open(out / f"importance_{tok}.json", "w") as f:
            json.dump({"species": sp, "alert_bin": k, "n_samples": int(len(samples)), "model": b.get("model_name"),
                       "split": a.split, "n_repeats": a.n_repeats, "strata": "season x Location_Code"}, f, indent=1)


def cmd_contrast(a):
    A, B = read_table(a.a), read_table(a.b)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from scipy.stats import spearmanr
    rows = []
    for loss in ("rps", "brier_alert"):
        ga = {c.split("__", 1)[1] for c in A.columns if c.startswith(f"d_{loss}__")}
        gb = {c.split("__", 1)[1] for c in B.columns if c.startswith(f"d_{loss}__")}
        groups = sorted(ga & gb)
        # normalized importance (share of the summed positive importance): comparable across species
        def shares(df, ix=None):
            d = df if ix is None else df.iloc[ix]
            v = np.array([np.average(d[f"d_{loss}__{g}"], weights=d["weight"]) for g in groups])
            return v / max(np.clip(v, 0, None).sum(), 1e-12)
        sa, sb = shares(A), shares(B)
        rng = np.random.default_rng(a.seed)
        ca, ua = pd.factorize(A["split_group"].astype(str))
        cb, ub = pd.factorize(B["split_group"].astype(str))
        ma = [np.flatnonzero(ca == i) for i in range(len(ua))]
        mb = [np.flatnonzero(cb == i) for i in range(len(ub))]
        dd, rho = [], []
        for _ in range(a.n_boot):
            ia = np.concatenate([ma[i] for i in rng.integers(0, len(ua), len(ua))])
            ib = np.concatenate([mb[i] for i in rng.integers(0, len(ub), len(ub))])
            xa, xb = shares(A, ia), shares(B, ib)
            dd.append(xa - xb)
            rho.append(spearmanr(xa, xb).correlation)
        dd = np.asarray(dd)
        for j, g in enumerate(groups):
            rows.append({"loss": loss, "group": g, "share_a": sa[j], "share_b": sb[j], "diff": sa[j] - sb[j],
                         "lo95": np.quantile(dd[:, j], 0.025), "hi95": np.quantile(dd[:, j], 0.975)})
        rows.append({"loss": loss, "group": "_spearman_rank", "diff": spearmanr(sa, sb).correlation,
                     "lo95": np.nanquantile(rho, 0.025), "hi95": np.nanquantile(rho, 0.975)})
    na, nb = (Path(x).name.split(".")[0].replace("importance_", "").replace("_samples", "") for x in (a.a, a.b))
    pd.DataFrame(rows).to_csv(out / f"contrast_{na}_vs_{nb}.csv", index=False)


def _ols_r2(X, y, w):
    X1 = np.column_stack([np.ones(len(y)), X])
    W = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(X1 * W, y * W[:, 0], rcond=None)
    r = y - X1 @ beta
    ybar = np.average(y, weights=w)
    return 1 - np.sum(w * r ** 2) / np.sum(w * (y - ybar) ** 2)


def cmd_conditional(a):
    import yaml
    from sit_fuse.pipelines.habs.uq.hab_uq_common import species_spec
    from sit_fuse.pipelines.habs.followon.variant_compare import load_predictions, prob_cols
    with open(a.yaml) as f:
        yml = yaml.safe_load(f)
    root = _uq_root(yml)
    ref_spec = species_spec(a.reference, (yml["uq"].get("species") or {}).get(a.reference))
    R = load_predictions(root, ref_spec.fname_token)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sp in a.species:
        spec = species_spec(sp, (yml["uq"].get("species") or {}).get(sp))
        S = load_predictions(root, spec.fname_token)
        k = alert_bin(spec)
        pr, ps = prob_cols(R), prob_cols(S)
        Rr = R[["sample_uid", "sensor"] + pr].rename(columns={c: "r_" + c for c in pr})
        J = S.merge(Rr, on=["sample_uid", "sensor"], how="inner").dropna(subset=["log10_value"])
        if len(J) < 30:
            LOG.warning("%s: %d joint samples; skipped", sp, len(J))
            continue
        Pr = J[["r_" + c for c in pr]].to_numpy(float)
        Ps = J[ps].to_numpy(float)
        e_ref = Pr @ np.arange(Pr.shape[1])
        e_sp = Ps @ np.arange(Ps.shape[1])
        x_alert = Ps[:, k:].sum(1)
        y = J["log10_value"].to_numpy(float)
        w = J["weight"].to_numpy(float) if "weight" in J else np.ones(len(J))
        g = J["split_group"].astype(str).to_numpy()
        fr = lambda ix: _ols_r2(e_ref[ix, None], y[ix], w[ix])
        ff = lambda ix: _ols_r2(np.column_stack([e_ref[ix], e_sp[ix], x_alert[ix]]), y[ix], w[ix])
        allix = np.arange(len(y))
        r0, r1 = fr(allix), ff(allix)
        codes, ug = pd.factorize(g)
        mem = [np.flatnonzero(codes == i) for i in range(len(ug))]
        rng = np.random.default_rng(a.seed)
        d = []
        for _ in range(a.n_boot):
            ix = np.concatenate([mem[i] for i in rng.integers(0, len(ug), len(ug))])
            d.append(ff(ix) - fr(ix))
        rows.append({"species": sp, "reference": a.reference, "n": len(y), "r2_reference": r0, "r2_full": r1,
                     "delta_r2": r1 - r0, "lo95": np.quantile(d, 0.025), "hi95": np.quantile(d, 0.975)})
    pd.DataFrame(rows).to_csv(out / "conditional_vs_reference.csv", index=False)


def cmd_composites(a):
    import yaml
    from scipy.stats import mannwhitneyu
    from sit_fuse.pipelines.habs.uq.hab_uq_common import instrument_dirs, pixel_centers
    from sit_fuse.pipelines.habs.uq.hab_uq_matchups import load_scene
    from sit_fuse.pipelines.habs.followon.ancillary import AncillaryProvider
    from sit_fuse.xai.cluster_interpret import bh_adjust
    from sit_fuse.train.sf_uq_utils import InvalidClusters, head_valid_mask
    with open(a.yaml) as f:
        yml = yaml.safe_load(f)
    uq = dict(yml["uq"])
    uq["features"] = {**uq.get("features", {}), "groups": ["ancillary:sst", "ancillary:upwelling"]
                      + (["physical"] if (uq.get("ancillary") or {}).get("physical") else [])}
    prov = AncillaryProvider(uq)
    invalid = InvalidClusters.from_config(uq)
    rng = np.random.default_rng(a.seed)
    days = pd.date_range(a.start, a.end, freq=f"{a.every}D")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    allrows = []
    for s in a.sensors.split(","):
        dirs = instrument_dirs(yml, s)
        parts = []
        for d in days:
            sc = load_scene(dirs, s, d, uq)
            if sc is None:
                continue
            bands, prof, _, _ = sc
            lab = bands.get("parent_label")
            if lab is None:
                continue
            ok = head_valid_mask(bands, invalid) & np.isfinite(lab) & (lab >= 0)
            r, c = np.nonzero(ok)
            if r.size > a.pixels_per_scene:
                k = rng.choice(r.size, a.pixels_per_scene, replace=False)
                r, c = r[k], c[k]
            LAT, LON = pixel_centers(prof)
            env = prov.at(s, d, LAT[r, c], LON[r, c])
            parts.append(pd.DataFrame({"sensor": s, "date": d, "parent": lab[r, c].astype(int), **env}))
        if not parts:
            LOG.warning("%s: no scenes", s)
            continue
        df = pd.concat(parts, ignore_index=True)
        save_table(df, out / f"composites_pixels_{s}")
        vars_ = [v for v in df.columns if v.startswith(("anc_", "phys_"))]
        for cl, sub in df.groupby("parent"):
            rest = df[df["parent"] != cl]
            for v in vars_:
                x, z = sub[v].dropna().to_numpy(), rest[v].dropna().to_numpy()
                if x.size < 30 or z.size < 30:
                    continue
                mw = mannwhitneyu(x, z)
                p = mw.pvalue
                allrows.append({"sensor": s, "parent": int(cl), "var": v, "n": int(x.size), "mean": float(x.mean()),
                                "mean_rest": float(z.mean()), "diff": float(x.mean() - z.mean()),
                                "effect_auc": float(mw.statistic / (x.size * z.size)), "p": p})
    if allrows:
        bh_adjust(allrows, "p")            # adds q_bh in place
        res = pd.DataFrame(allrows)
        res.to_csv(out / "cluster_environment_composites.csv", index=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("importance")
    i.add_argument("-y", "--yaml", required=True)
    i.add_argument("--species", action="append", required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--split", default="test")
    i.add_argument("--n-repeats", type=int, default=5)
    i.add_argument("--n-boot", type=int, default=500)
    i.add_argument("--max-samples", type=int, default=0)
    i.add_argument("--shap", type=int, default=0, help="rows explained by SHAP (0 = skip)")
    i.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("contrast")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--n-boot", type=int, default=500)
    c.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("conditional")
    d.add_argument("-y", "--yaml", required=True)
    d.add_argument("--species", action="append", required=True)
    d.add_argument("--reference", default="Total_Phytoplankton")
    d.add_argument("--out", required=True)
    d.add_argument("--n-boot", type=int, default=500)
    d.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("composites")
    m.add_argument("-y", "--yaml", required=True)
    m.add_argument("--sensors", required=True)
    m.add_argument("--start", required=True)
    m.add_argument("--end", required=True)
    m.add_argument("--every", type=int, default=5)
    m.add_argument("--pixels-per-scene", type=int, default=4000)
    m.add_argument("--out", required=True)
    m.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"importance": cmd_importance, "contrast": cmd_contrast, "conditional": cmd_conditional,
     "composites": cmd_composites}[a.cmd](a)


if __name__ == "__main__":
    main()
