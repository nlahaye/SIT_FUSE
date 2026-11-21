"""
Figures for the C-HARM follow-on (2023 OLCI/MODIS rerun, exceedance recalibration, pDA threshold
decision rule, coastal-edge and tuned-cutoff comparison). Reads only finished outputs; every figure
is skipped with a warning when its inputs are missing, so the job can run at any point.

    python -m sit_fuse.pipelines.habs.uq.hab_followon_figures -y <hab yaml> \
        [--figs coverage,reliability,decision,f1,points,coastal,maps_ts,map_panels] \
        [--before-eval /path/to/previous/evaluation] [--dates 2023-01-01,2024-06-15] [--out DIR]

  coverage     days with per-sensor *.uq.tif per platform x month (one species), 2023-2026
  reliability  exceedance reliability per species x alert threshold; dashed = --before-eval run
  decision     test confusion matrices of the bundle decision rule (pred_label) per species
  f1           weighted F1 by sensor and season, decision rule vs production (station-day bootstrap)
  points       delta ROC AUC (SIT-FUSE - C-HARM) with 95 % CI, vs charm_p and charm_p_offcoast
  coastal      C-HARM mean P and observed event rate by coast distance, per extraction variant
  maps_ts      daily Spearman and equal-area kappa vs C-HARM per stream (maps/*/map_metrics)
  map_panels   C-HARM | ours | equal-area alert masks for the listed dates (CHARM_CMP stacks)
Default --out: <uq.out_dir>/figures/followon.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("hab-followon-figures")
SENSOR_ORDER = ["pace", "s3a", "s3b", "snpp", "jpss1", "jpss2", "modis"]
SENSOR_NAME = {"pace": "PACE", "s3a": "S3A", "s3b": "S3B", "snpp": "SNPP", "jpss1": "NOAA-20",
               "jpss2": "NOAA-21", "modis": "MODIS"}
FIGS = ("coverage", "reliability", "decision", "f1", "points", "coastal", "maps_ts", "map_panels")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 150})
    return plt


def _uq_root(yml) -> Path:
    uq = yml.get("uq", {}) or {}
    return Path(uq.get("out_dir", os.path.join(yml.get("final_product_dir", "."), "uq")))


def _species(yml):
    return list((yml.get("uq", {}) or {}).get("species_run") or [])


def _token(yml, sp):
    from sit_fuse.pipelines.habs.uq.hab_uq_common import species_spec
    return species_spec(sp, ((yml.get("uq") or {}).get("species") or {}).get(sp)).fname_token


def fig_coverage(yml, out: Path, species=None):
    from sit_fuse.pipelines.habs.uq.hab_uq_common import SENSORS
    plt = _plt()
    sp = species or (_species(yml) or ["Pseudo_nitzschia_total"])[0]
    tok = _token(yml, sp)
    rows = []
    for s in SENSOR_ORDER:
        if s not in SENSORS:
            continue
        for p in glob.glob(str(_uq_root(yml) / "products" / s / f"{SENSORS[s].prefix}.*.{tok}.uq.tif")):
            d = os.path.basename(p).split(".")[1]
            rows.append({"sensor": s, "month": pd.Timestamp(d).to_period("M")})
    if not rows:
        LOG.warning("coverage: no products for %s", sp)
        return None
    df = pd.DataFrame(rows).groupby(["sensor", "month"]).size().unstack(fill_value=0)
    df = df.reindex([s for s in SENSOR_ORDER if s in df.index])
    months = pd.period_range(df.columns.min(), df.columns.max(), freq="M")
    df = df.reindex(columns=months, fill_value=0)
    fig, ax = plt.subplots(figsize=(max(8, 0.18 * len(months)), 2.8))
    im = ax.imshow(df.values, aspect="auto", cmap="viridis", vmin=0, vmax=31)
    ax.set_yticks(range(len(df.index)), [SENSOR_NAME.get(s, s) for s in df.index])
    ticks = [i for i, m in enumerate(months) if m.month in (1, 7)]
    ax.set_xticks(ticks, [str(months[i]) for i in ticks], rotation=45, ha="right")
    fig.colorbar(im, ax=ax, label="days with product")
    ax.set_title(f"Per-sensor product days per month ({sp})", loc="left")
    fig.tight_layout()
    f = out / f"coverage_{tok}.png"
    fig.savefig(f)
    plt.close(fig)
    return f


def _eval_jsons(d):
    return {json.load(open(p))["species"]: json.load(open(p))
            for p in sorted(glob.glob(str(Path(d) / "hab_uq_evaluation_*.json")))}


def fig_reliability(yml, out: Path, before=None):
    plt = _plt()
    cur = _eval_jsons(_uq_root(yml) / "evaluation")
    old = _eval_jsons(before) if before else {}
    panels = [(sp, t) for sp, d in cur.items() for t in (d["overall"]["uq"].get("exceedance") or {})]
    if not panels:
        LOG.warning("reliability: no evaluation JSONs")
        return None
    n = len(panels)
    cols = min(4, n)
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(3.2 * cols, 3.1 * np.ceil(n / cols)), squeeze=False)
    for ax, (sp, t) in zip(axes.flat, panels):
        for d, ls, lab in ((cur.get(sp), "-", "current"), (old.get(sp), "--", "before")):
            e = ((d or {}).get("overall", {}).get("uq", {}).get("exceedance") or {}).get(t)
            if not e or not e.get("reliability"):
                continue
            r = pd.DataFrame(e["reliability"])
            ax.plot(r["mean_predicted"], r["observed_frequency"], ls, marker="o", ms=3,
                    label=f"{lab}: Brier {e.get('brier', np.nan):.3f}, base rate {e.get('base_rate', np.nan):.3f}")
        m = max(0.05, ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, m], [0, m], color="grey", lw=0.8)
        ax.set_xlim(0, m); ax.set_ylim(0, m)
        short = sp.replace("Pseudo_nitzschia_", "PN ").replace("_group", "").replace("_", " ")
        ax.set_title(f"{short} >= {float(t):g}", fontsize=8)
        ax.set_xlabel("mean predicted"); ax.set_ylabel("observed frequency")
        ax.legend(fontsize=6, frameon=False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Exceedance reliability, test split (per-sensor rows)", x=0.01, ha="left")
    fig.tight_layout()
    f = out / "reliability_exceedance.png"
    fig.savefig(f)
    plt.close(fig)
    return f


def _preds(yml):
    return {os.path.basename(p)[len("hab_uq_test_predictions_"):-4]: p
            for p in sorted(glob.glob(str(_uq_root(yml) / "evaluation" / "hab_uq_test_predictions_*.csv")))}


def fig_decision(yml, out: Path):
    plt = _plt()
    pr = _preds(yml)
    if not pr:
        LOG.warning("decision: no test predictions")
        return None
    fig, axes = plt.subplots(1, len(pr), figsize=(3.0 * len(pr), 3.0), squeeze=False)
    for ax, (tok, p) in zip(axes.flat, pr.items()):
        d = pd.read_csv(p, usecols=lambda c: c in ("ref_label", "pred_label", "weight"))
        C = int(max(d["ref_label"].max(), d["pred_label"].max())) + 1
        M = pd.crosstab(d["ref_label"], d["pred_label"]).reindex(index=range(C), columns=range(C), fill_value=0)
        ax.imshow(np.log10(M.values + 1), cmap="Blues")
        for i in range(C):
            for j in range(C):
                ax.text(j, i, int(M.values[i, j]), ha="center", va="center", fontsize=7)
        ax.set_xlabel("decision bin"); ax.set_ylabel("observed bin")
        ax.set_title(tok.replace("_", " "), fontsize=8)
    fig.suptitle("Test confusion matrices, bundle decision rule", x=0.01, ha="left")
    fig.tight_layout()
    f = out / "decision_confusion.png"
    fig.savefig(f)
    plt.close(fig)
    return f


def fig_f1(yml, out: Path, n_boot=300):
    from sklearn.metrics import f1_score
    plt = _plt()
    files = []
    for tok, p in _preds(yml).items():
        d = pd.read_csv(p)
        if "pred_label" not in d or "production_bin" not in d:
            continue
        C = len([c for c in d.columns if c.startswith("p_") and c[2:].isdigit()])
        rng = np.random.default_rng(0)
        strata = [("sensor", [s for s in SENSOR_ORDER if (d["sensor"] == s).any()]),
                  ("season", [s for s in ("DJF", "MAM", "JJA", "SON") if (d.get("season") == s).any()])]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [7, 4]}, sharey=True)
        for ax, (col, levels) in zip(axes, strata):
            for j, lv in enumerate(levels):
                g = d[d[col] == lv]
                grp = g["split_group"].to_numpy() if "split_group" in g else np.arange(len(g))
                ug = np.unique(grp)
                ix = {k: np.where(grp == k)[0] for k in ug}
                res = {"pred_label": [], "production_bin": []}
                for _ in range(n_boot):
                    i = np.concatenate([ix[k] for k in rng.choice(ug, ug.size)])
                    b = g.iloc[i]
                    for c in res:
                        res[c].append(f1_score(b["ref_label"], b[c], labels=list(range(C)), average="weighted",
                                               sample_weight=b.get("weight"), zero_division=0))
                for c, off, colr in (("pred_label", -0.19, "#2a6f97"), ("production_bin", 0.19, "#c9a227")):
                    ax.boxplot([res[c]], positions=[j + off], widths=0.32, patch_artist=True, showfliers=False,
                               boxprops=dict(facecolor=colr, alpha=0.8), medianprops=dict(color="black"))
            ax.set_xticks(range(len(levels)), [f"{SENSOR_NAME.get(l, l)}\n(n={(d[col] == l).sum()})" for l in levels])
            ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3); ax.set_title(col)
        axes[0].set_ylabel(f"weighted F1 ({n_boot} station-day bootstrap draws)")
        h = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.8) for c in ("#2a6f97", "#c9a227")]
        fig.legend(h, ["SIT-FUSE UQ (decision rule)", "production baseline"], loc="upper right", ncol=2, frameon=False)
        fig.suptitle(f"{tok.replace('_', ' ')}: weighted F1, test split", x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        f = out / f"f1_{tok}.png"
        fig.savefig(f)
        plt.close(fig)
        files.append(f)
    return files or None


def _cmp(yml) -> Path:
    from sit_fuse.pipelines.habs.uq.hab_charm_compare import cmp_dir
    return cmp_dir(yml)


def fig_points(yml, out: Path, lead=0):
    plt = _plt()
    p = _cmp(yml) / "summary_paired.csv"
    if not p.exists():
        LOG.warning("points: %s missing", p)
        return None
    d = pd.read_csv(p)
    d = d[d["lead"] == lead]
    auc = [c for c in d.columns if c.startswith("d_roc_auc") or c.startswith("d_auc")]
    if not auc:
        LOG.warning("points: no AUC difference columns in %s", p)
        return None
    key = sorted(auc, key=len)[0]
    lo = next((c for c in d.columns if c.startswith(key) and ("lo" in c[len(key):] or "2.5" in c)), None)
    hi = next((c for c in d.columns if c.startswith(key) and ("hi" in c[len(key):] or "97.5" in c)), None)
    keep = d["stream"].str.match(r"^(direct_fused|fused|combined_\w+_fused|direct_type_\w+|type_\w+)$")
    d = d[keep]
    if d.empty:
        return None
    targets = list(d["target"].unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(4.4 * len(targets), 0.32 * d.groupby("target").size().max() + 1.6),
                             squeeze=False)
    for ax, t in zip(axes.flat, targets):
        g = d[d["target"] == t].sort_values("stream")
        streams = list(dict.fromkeys(g["stream"]))
        for k, (vs, colr, off) in enumerate((("charm", "#2a6f97", -0.15), ("charm_offcoast", "#c0504d", 0.15))):
            h = g[g["vs"] == vs].set_index("stream").reindex(streams)
            y = np.arange(len(streams)) + off
            x = h[key].to_numpy(float)
            err = None if lo is None or hi is None else np.vstack([x - h[lo].to_numpy(float), h[hi].to_numpy(float) - x])
            ax.errorbar(x, y, xerr=err, fmt="o", ms=3, color=colr, label=f"vs {vs.replace('_', ' ')}")
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_yticks(range(len(streams)), streams, fontsize=7)
        n_ev = g["n_events"].max() if "n_events" in g else None
        ax.set_title(f"{t}: delta ROC AUC (SIT-FUSE - C-HARM)" + (f", {int(n_ev)} events" if n_ev == n_ev and n_ev else ""),
                     fontsize=8)
        ax.legend(fontsize=7, frameon=True, loc="best")
    fig.tight_layout()
    f = out / "points_delta_auc.png"
    fig.savefig(f)
    plt.close(fig)
    return f


def fig_coastal(yml, out: Path, lead=0):
    plt = _plt()
    p = _cmp(yml) / "summary_coastal_edge.csv"
    if not p.exists():
        LOG.warning("coastal: %s missing", p)
        return None
    d = pd.read_csv(p)
    d = d[(d["lead"] == lead) & (d["stratum"] == "coast_bin")]
    if d.empty:
        return None
    order = ["<6km", "6-12km", "12-24km", ">=24km"]
    targets = list(d["target"].unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 3.2), squeeze=False)
    for ax, t in zip(axes.flat, targets):
        g = d[d["target"] == t]
        x = np.arange(len(order))
        for k, s in enumerate(("charm_p", "charm_p_nearest", "charm_p_offcoast")):
            h = g[g["stream"] == s].set_index("level").reindex(order)
            ax.bar(x + (k - 1) * 0.25, h["mean_p"], 0.25, label=s.replace("charm_p", "C-HARM").replace("_", " "))
        h = g[g["stream"] == "charm_p"].set_index("level").reindex(order)
        ax.plot(x, h["event_rate"], "k-o", ms=3, label="observed event rate")
        ax.set_xticks(x, [f"{o}\n(n={int(n) if n == n else 0})" for o, n in zip(order, h["n"])], fontsize=7)
        ax.set_ylim(0, 1); ax.set_title(f"{t}: C-HARM P by coast distance of nearest pixel", fontsize=8)
        ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    f = out / "coastal_edge.png"
    fig.savefig(f)
    plt.close(fig)
    return f


def fig_maps_ts(yml, out: Path):
    plt = _plt()
    files = sorted(glob.glob(str(_cmp(yml) / "maps" / "*" / "*" / "map_metrics_*.csv")))
    files = [p for p in files if os.path.getsize(p) > 1]
    if not files:
        LOG.warning("maps_ts: no map metrics")
        return None
    m = pd.concat([pd.read_csv(p, parse_dates=["date"]) for p in files], ignore_index=True)
    m = m[m.get("lead", 0) == 0]
    targets = list(m["target"].unique())
    mets = [c for c in ("spearman", "kappa_eqarea", "kappa_tuned") if c in m]
    fig, axes = plt.subplots(len(mets), len(targets), figsize=(5 * len(targets), 2.4 * len(mets)), squeeze=False,
                             sharex=True)
    for j, t in enumerate(targets):
        g = m[m["target"] == t]
        for i, c in enumerate(mets):
            ax = axes[i, j]
            for s, h in g.groupby("stream"):
                r = h.set_index("date")[c].sort_index().rolling("15D", min_periods=3).median()
                ax.plot(r.index, r.values, lw=0.9, label=s)
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_title(f"{t}: {c} vs C-HARM (15-day median)", fontsize=8)
            if i == 0:
                ax.legend(fontsize=6, ncol=3, frameon=False)
    fig.tight_layout()
    f = out / "map_agreement_timeseries.png"
    fig.savefig(f)
    plt.close(fig)
    return f


def fig_map_panels(yml, out: Path, dates=None, lead=0):
    import rasterio
    plt = _plt()
    from sit_fuse.pipelines.habs.uq.hab_charm_compare import points_stream_name, tuned_cutoffs
    files = []
    base = _cmp(yml) / "maps"
    for t in ("pn", "pda", "cda"):
        cands = sorted(glob.glob(str(base / t / "*" / f"CHARM_CMP.*.{t}.L{lead}.tif")))
        if not cands:
            continue
        want = [str(pd.Timestamp(x).strftime("%Y%m%d")) for x in (dates or [])] or \
            [os.path.basename(cands[len(cands) // 2]).split(".")[1]]
        cuts = tuned_cutoffs(yml, t, lead)
        for ds in want:
            p = next((c for c in cands if f".{ds}." in c), None)
            if p is None:
                continue
            with rasterio.open(p) as r:
                names = list(r.descriptions)
                X = r.read().astype(float)
                arr = {n: np.where(X[i] == r.nodata, np.nan, X[i]) for i, n in enumerate(names) if n}
                ext = (r.bounds.left, r.bounds.right, r.bounds.bottom, r.bounds.top)
            ours = "combined" if "combined" in arr else "fused"
            if "charm" not in arr or ours not in arr:
                continue
            ch, us = arr["charm"], arr[ours]
            cb = cuts.get("charm_p", 0.5)
            ok = np.isfinite(ch) & np.isfinite(us)
            frac = float((ch[ok] >= cb).mean()) if ok.any() else np.nan
            qa = float(np.quantile(us[ok], 1 - frac)) if ok.any() and 0 < frac < 1 else np.nan
            fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
            for ax, a, title in ((axes[0], ch, "C-HARM P"), (axes[1], us, f"SIT-FUSE {ours} P")):
                im = ax.imshow(a, extent=ext, vmin=0, vmax=1 if a is ch else max(0.05, np.nanpercentile(a, 99)),
                               cmap="magma")
                fig.colorbar(im, ax=ax, shrink=0.75)
                ax.set_title(title, fontsize=9)
            agree = np.full(ch.shape, np.nan)
            if np.isfinite(qa):
                A, B = us >= qa, ch >= cb
                agree[ok] = np.select([A[ok] & B[ok], A[ok] & ~B[ok], ~A[ok] & B[ok]], [3, 2, 1], 0)
            from matplotlib.colors import ListedColormap
            axes[2].imshow(agree, extent=ext, cmap=ListedColormap(["#dddddd", "#4f81bd", "#c0504d", "#7a3b8f"]),
                           vmin=-0.5, vmax=3.5)
            axes[2].set_title(f"alert areas: C-HARM >= {cb:.2f}, ours >= q (equal area {frac:.0%})", fontsize=8)
            h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#dddddd", "#4f81bd", "#c0504d", "#7a3b8f")]
            axes[2].legend(h, ["neither", "C-HARM only", "ours only", "both"], fontsize=6, loc="lower left")
            key = points_stream_name(t, ours)
            if ours == "combined":
                key = next((k for k in cuts if k.startswith("combined_") and k.endswith("_fused")), key)
            tc = cuts.get(key)
            fig.suptitle(f"{t.upper()} {pd.Timestamp(ds).date()} (lead {lead})"
                         + (f"; ours tuned cutoff {tc:.2f}" if tc is not None else ""), x=0.01, ha="left")
            fig.tight_layout()
            f = out / f"map_panel_{t}_{ds}.png"
            fig.savefig(f)
            plt.close(fig)
            files.append(f)
    return files or None


def run(yml, figs=FIGS, out=None, before=None, dates=None):
    out = Path(out or (_uq_root(yml) / "figures" / "followon"))
    out.mkdir(parents=True, exist_ok=True)
    made = {}
    for name in figs:
        try:
            fn = globals()[f"fig_{name}"]
            kw = {"before": before} if name == "reliability" else ({"dates": dates} if name == "map_panels" else {})
            made[name] = fn(yml, out, **kw)
        except Exception as e:      # a missing input skips one figure, not the job
            LOG.warning("figure %s skipped: %s", name, e)
            made[name] = None
    LOG.info("figures -> %s: %s", out, {k: (None if v is None else str(v)) for k, v in made.items()})
    return made


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--figs", default=",".join(FIGS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--before-eval", default=None)
    ap.add_argument("--dates", default=None, help="comma list for map_panels")
    a = ap.parse_args()
    from sit_fuse.pipelines.habs.uq.hab_charm import read_yaml_any
    run(read_yaml_any(a.yaml), [f for f in a.figs.split(",") if f], a.out, a.before_eval,
        None if not a.dates else a.dates.split(","))
