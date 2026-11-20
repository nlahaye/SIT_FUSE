"""
Manuscript figures for SIT-FUSE probabilistic products (smoke now, fire later).

Inputs are a VizContext (geo products, evaluation outputs, comparison references,
basemap). Every figure writes PNG + PDF and returns its paths; tables are CSV.

Map panels are declared as strings, so new products / references need config only:
    mask:<model>                 binary mask (pred_label)
    prob:<model>                 P(class) (prob_band)
    envelope:<model>             likely (q05>=thr) / expected (p>=thr) / possible (q95>=thr)
    width:<model>                ensemble 90% interval width (q95 - q05)
    unc:<model>:<band>           any uncertainty band
    ref:<reference>              comparison product (raster reference or HMS polygons)
    diff:<a>:<b>                 agreement map (both / a only / b only); a, b = <model> or ref:<name>,
                                 e.g. diff:hist_gradient_boosting:ref:adp_smoke or
                                 diff:ref:adp_smoke:ref:viirs_aerosol_smoke (reference vs reference)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import data as D
from . import plotting as P
from .products import MODEL_COLORS, MODEL_LABELS, SPLIT_COLORS, ProductSpec

LOG = logging.getLogger("uq-viz")
KM_PER_DEG = 111.32


@dataclass
class VizContext:
    index: D.GeoProductIndex
    spec: ProductSpec
    evals: dict[str, D.ModelEval]
    refs: dict[str, Any]
    ref_cfg: dict[str, dict]
    basemap: dict | None
    cfg: dict
    out: Path
    primary_model: str = "hist_gradient_boosting"
    baseline_model: str = "hard_assignment"
    labels: dict[str, str] = field(default_factory=lambda: dict(MODEL_LABELS))
    _frames: pd.DataFrame | None = None
    _area: np.ndarray | None = None

    # ------------------------------------------------------------------ helpers
    def label(self, m: str) -> str:
        return self.labels.get(m, self.ref_cfg.get(m, {}).get("label", m))

    def short(self, m: str) -> str:
        return self.label(m).split(" (")[0]

    def color(self, m: str) -> str:
        return MODEL_COLORS.get(m, self.ref_cfg.get(m, {}).get("color", "#444444"))

    @property
    def frames(self) -> pd.DataFrame:
        if self._frames is None:
            fc = self.cfg.get("frames", {})
            self._frames = D.assign_frames(self.index.scenes(self.primary_model), fc.get("mode", "scan"),
                                           int(fc.get("minutes", 60)))
        return self._frames

    @property
    def pixel_area(self) -> np.ndarray:
        if self._area is None:
            t = self.index.transform
            h, w = self.index.shape
            lat = t.f + (np.arange(h) + 0.5) * t.e
            a = (abs(t.a) * KM_PER_DEG) * (abs(t.e) * KM_PER_DEG) * np.cos(np.deg2rad(lat))
            self._area = np.repeat(a[:, None], w, axis=1).astype(np.float64)
        return self._area

    def threshold(self, model: str) -> float:
        t = self.index.df.loc[self.index.df.model == model, "threshold"]
        return float(t.iloc[0]) if len(t) and np.isfinite(t.iloc[0]) else 0.5

    def frame_pairs(self, frame_id: str) -> list[str]:
        return list(self.frames.loc[self.frames.frame_id == frame_id, "pair_id"])

    def view_extent(self, frame_id: str | None = None, zoom: bool = True):
        """Map view: config `event_extents[event]` ([W, E, S, N]) when zoom, else the product grid."""
        if zoom and frame_id is not None:
            ev = self.frames.loc[self.frames.frame_id == frame_id, "event"].iloc[0]
            e = self.cfg.get("event_extents", {}).get(ev)
            if e:
                return list(e)
        return self.index.extent()

    def frame_time(self, frame_id: str) -> pd.Timestamp:
        return self.frames.loc[self.frames.frame_id == frame_id, "frame_time"].iloc[0]

    def band(self, model: str, frame_id: str, band: str):
        return D.mosaic(self.index, model, self.frame_pairs(frame_id), band, self.spec.fill_seams)

    def reference(self, name: str, frame_id: str):
        """(raster on product grid or None, polygons GeoDataFrame or None)"""
        r = self.refs.get(name)
        if r is None:
            return None, None
        if isinstance(r, D.HMSPolygons):
            t = self.frame_time(frame_id)
            return r.raster(t, self.index), r.select(t)
        return D.mosaic_reference(r, self.index, self.frame_pairs(frame_id), self.primary_model,
                                  self.spec.prob_band), None

    def hms_rank(self, name: str, frame_id: str) -> np.ndarray | None:
        """HMS density class per pixel (0 none, 1 light, 2 medium, 3 heavy); max over overlapping polygons."""
        from rasterio.features import rasterize
        r = self.refs.get(name)
        if not isinstance(r, D.HMSPolygons) or r.points:
            return None
        g = r.select(self.frame_time(frame_id))
        if g is None:
            return None
        if len(g) == 0:
            return np.zeros(self.index.shape, np.float32)
        g = g.sort_values("rank")
        return rasterize(((geom, float(k)) for geom, k in zip(g.geometry, g["rank"])), out_shape=self.index.shape,
                         transform=self.index.transform, fill=0).astype(np.float32)


# ---------------------------------------------------------------------------
# frame statistics (shared by figures, animations, manuscript tables)
# ---------------------------------------------------------------------------

def _agreement(row: dict, pre: str, r: np.ndarray, mk: np.ndarray, p: np.ndarray, v: np.ndarray, A: np.ndarray):
    """Area-weighted contingency vs a two-sided reference (smoke / clear / no data) on reference-valid pixels."""
    rv = v & np.isfinite(r)
    ref = rv & (r >= 0.5)
    clr = rv & (r < 0.5)
    tp, fp, fn = A[ref & mk].sum(), A[clr & mk].sum(), A[ref & ~mk].sum()
    row.update({f"{pre}_valid_area": A[rv].sum(), f"{pre}_area": A[ref].sum(), f"{pre}_clear_area": A[clr].sum(),
                f"{pre}_tp": tp, f"{pre}_fp": fp, f"{pre}_fn": fn,
                f"{pre}_iou": tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan,
                f"{pre}_hit": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
                f"{pre}_fpr": fp / A[clr].sum() if clr.any() else np.nan,
                f"{pre}_mean_p_smoke": float(np.nanmean(p[ref])) if ref.any() else np.nan,
                f"{pre}_mean_p_clear": float(np.nanmean(p[clr])) if clr.any() else np.nan})


def frame_statistics(ctx: VizContext, models: list[str], hms_name: str | None, cache: Path | None = None,
                     agreement_refs: list[str] | None = None):
    """Per frame x model: discrete / expected / interval areas; agreement with two-sided references
    (`agreement_refs`, e.g. ADP primary and VIIRS secondary: columns <ref>_iou, _hit, _fpr, _tp, ...);
    and consistency with a positive-only reference (HMS: hms_* columns)."""
    agreement_refs = [r for r in (agreement_refs or []) if r in ctx.refs]
    if cache is not None and cache.exists():
        fs = pd.read_csv(cache, dtype={"date": str})
        fs["frame_time"] = pd.to_datetime(fs.frame_time, utc=True, format="mixed")
        if all(f"{r}_iou" in fs for r in agreement_refs):
            return fs
        LOG.info("frame statistics cache lacks %s; recomputing", agreement_refs)
    rows = []
    A = ctx.pixel_area
    s = ctx.spec
    for fid, fr in ctx.frames.groupby("frame_id"):
        t = fr.frame_time.iloc[0]
        hms_r = hms_g = rank = None
        if hms_name:
            hms_r, hms_g = ctx.reference(hms_name, fid)
            rank = ctx.hms_rank(hms_name, fid)
        two = {r: ctx.reference(r, fid)[0] for r in agreement_refs}
        for m in models:
            p = ctx.band(m, fid, s.prob_band)
            if p is None:
                continue
            thr = ctx.threshold(m)
            v = np.isfinite(p)
            mask = ctx.band(m, fid, s.mask_band)
            mk = (np.nan_to_num(mask) >= 0.5) & v if mask is not None else (np.nan_to_num(p) >= thr) & v
            row = {"frame_id": fid, "frame_time": t, "date": fr.date.iloc[0], "split": fr.split.iloc[0],
                   "event": fr.event.iloc[0], "n_granules": len(fr), "model": m,
                   "valid_area": A[v].sum(), "discrete_area": A[mk].sum(),
                   "expected_area": (A * np.nan_to_num(p))[v].sum()}
            for nm, b in (("q05", s.q_lo_band), ("q95", s.q_hi_band)):
                q = ctx.band(m, fid, b) if b else None
                row[f"{nm}_area"] = A[(np.nan_to_num(q) >= thr) & v].sum() if q is not None else np.nan
            for rn, rr in two.items():
                if rr is not None:
                    _agreement(row, rn, rr, mk, p, v, A)
            if hms_r is not None:
                h = (hms_r >= 0.5) & v
                row["hms_area"] = A[h].sum()
                row["hms_detect_frac"] = A[h & mk].sum() / A[h].sum() if h.any() else np.nan
                row["pred_outside_hms_frac"] = A[mk & ~h].sum() / A[mk].sum() if mk.any() else np.nan
                row["mean_p_in_hms"] = float(np.nanmean(p[h])) if h.any() else np.nan
                row["mean_p_out_hms"] = float(np.nanmean(p[v & ~h])) if (v & ~h).any() else np.nan
                if rank is not None:
                    for k, nm in enumerate(("none", "light", "medium", "heavy")):
                        sel = v & (rank == k)
                        row[f"mean_p_hms_{nm}"] = float(np.nanmean(p[sel])) if sel.any() else np.nan
                        row[f"frac_flag_hms_{nm}"] = float(mk[sel].mean()) if sel.any() else np.nan
                        row[f"n_hms_{nm}"] = int(sel.sum())
            rows.append(row)
    df = pd.DataFrame(rows).sort_values(["frame_time", "model"])
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    return df


def hms_consistency_table(fs: pd.DataFrame, splits=("holdout",)) -> pd.DataFrame:
    d = fs[fs.split.isin(splits) & fs.hms_area.gt(0)] if "hms_area" in fs else fs.iloc[0:0]
    if d.empty:
        return d
    rows = []
    for (ev, m), g in d.groupby(["event", "model"]):
        w = g.hms_area
        r = {"event": ev, "model": m, "n_frames": len(g),
             "hms_detect_frac": np.average(g.hms_detect_frac, weights=w),
             "pred_outside_hms_frac": np.nanmean(g.pred_outside_hms_frac),
             "mean_p_in_hms": np.nanmean(g.mean_p_in_hms), "mean_p_out_hms": np.nanmean(g.mean_p_out_hms)}
        for nm in ("none", "light", "medium", "heavy"):
            c = f"mean_p_hms_{nm}"
            if c in g:
                n = g[f"n_hms_{nm}"].fillna(0)
                r[c] = np.average(g[c].fillna(0), weights=n) if n.sum() else np.nan
                r[f"frac_flag_hms_{nm}"] = np.average(g[f"frac_flag_hms_{nm}"].fillna(0), weights=n) if n.sum() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure: data partition
# ---------------------------------------------------------------------------

def fig_partition(ctx: VizContext, name="fig_partition"):
    sc = ctx.index.scenes(ctx.primary_model)
    events = list(dict.fromkeys(sc.sort_values("scene_time").event))
    fig, axes = plt.subplots(1, len(events), figsize=(7.2, 2.1), sharey=True,
                             gridspec_kw={"width_ratios": [max(1, sc[sc.event == e].date.nunique()) for e in events]})
    axes = np.atleast_1d(axes)
    ev_lab = ctx.cfg.get("event_labels", {})
    for ax, ev in zip(axes, events):
        d = sc[sc.event == ev]
        g = d.groupby(["date", "split"]).size().reset_index(name="n")
        dates = sorted(d.date.unique())
        for i, dt in enumerate(dates):
            r = g[g.date == dt]
            sp, n = r.split.iloc[0], r.n.sum()
            ax.bar(i, n, color=SPLIT_COLORS.get(sp, "#ccc"), edgecolor="k", linewidth=0.4)
            ax.text(i, n + 1, str(n), ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(range(len(dates)), [pd.Timestamp(x).strftime("%m-%d") for x in dates], rotation=0)
        ax.set_title(ev_lab.get(ev, ev))
        ax.set_ylim(0, sc.groupby("date").size().max() * 1.22)
    axes[0].set_ylabel("TEMPO granule scenes")
    used = [s for s in ("dev", "calibration", "holdout") if s in set(sc.split)]
    fig.legend([Patch(fc=SPLIT_COLORS[s], ec="k", lw=0.4) for s in used], used, loc="upper center",
               ncol=len(used), frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    return P.save(fig, ctx.out, name)


# ---------------------------------------------------------------------------
# Figure: skill summary with bootstrap CIs
# ---------------------------------------------------------------------------

SKILL_METRICS = [("roc_auc", "ROC-AUC", True), ("pr_auc", "PR-AUC", True), ("target_macro_iou", "Smoke IoU", True),
                 ("brier", "Brier score", False), ("nll", "NLL", False), ("classwise_ece", "ECE", False)]


def fig_skill(ctx: VizContext, models: list[str], name="fig_skill"):
    models = [m for m in models if m in ctx.evals]
    fig, axes = plt.subplots(1, len(SKILL_METRICS), figsize=(7.2, 2.3), sharey=True)
    y = np.arange(len(models))[::-1]
    for ax, (k, lab, hib) in zip(axes, SKILL_METRICS):
        for yi, m in zip(y, models):
            s = ctx.evals[m].summary
            pt = ctx.evals[m].metric(k)
            ci = s["bootstrap_ci"]["metrics"].get(k, {})
            if pt is None:
                continue
            lo, hi = ci.get("lo", pt), ci.get("hi", pt)
            ax.errorbar(pt, yi, xerr=[[max(pt - lo, 0)], [max(hi - pt, 0)]], fmt="o", ms=4, color=ctx.color(m),
                        ecolor=ctx.color(m), elinewidth=1.1, capsize=2)
        ax.set_title(lab + ("\n(higher better)" if hib else "\n(lower better)"), fontsize=7)
        ax.grid(axis="x", lw=0.3, alpha=0.6)
        ax.tick_params(axis="x", labelsize=6.5)
    axes[0].set_yticks(y, [ctx.label(m) for m in models])
    axes[0].set_ylim(-0.6, len(models) - 0.4)
    fig.tight_layout(w_pad=0.6)
    return P.save(fig, ctx.out, name)


def _overall(ev, k: str):
    return ev.metric(k)


def _stratum_label(ctx, key: str) -> str:
    el = ctx.cfg.get("event_labels", {})
    sl = {"pos_q0": "lower smoke fraction", "pos_q1": "higher smoke fraction"}
    sl.update(ctx.cfg.get("stratum_labels", {}))
    ev, _, q = key.partition("|")
    return el.get(ev, ev) + (f", {sl.get(q, q)}" if q else "")


def probability_histograms(ctx: VizContext, models: list[str], split="holdout", bins=50) -> pd.DataFrame:
    """Histogram of P over evaluated (or split) scenes, from the geo rasters."""
    edges = np.linspace(0, 1, bins + 1)
    out = {"lo": edges[:-1], "hi": edges[1:]}
    for m in models:
        ev = ctx.evals.get(m)
        pids = set(ev.per_scene.pair_id) if ev is not None and ev.per_scene is not None else \
            set(ctx.index.scenes(m).query("split == @split").pair_id)
        h = np.zeros(bins)
        for p in ctx.index.df[(ctx.index.df.model == m) & ctx.index.df.pair_id.isin(pids)].path:
            a = D._read_band(p, ctx.spec.prob_band)
            if a is not None:
                h += np.histogram(a[np.isfinite(a)], edges)[0]
        out[m] = h / max(h.sum(), 1)
    return pd.DataFrame(out)


def fig_skill_by_event(ctx: VizContext, models: list[str], name="fig_skill_by_date"):
    models = [m for m in models if m in ctx.evals]
    dates = sorted(ctx.evals[models[0]].summary["by_date"].keys())
    mets = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("target_macro_iou", "Smoke IoU"),
            ("classwise_ece", "ECE (lower better)")]
    fig, axes = plt.subplots(1, len(mets), figsize=(7.2, 2.2))
    w = 0.8 / len(models)
    dl = ctx.cfg.get("date_labels", {})
    for ax, (k, lab) in zip(axes, mets):
        for i, m in enumerate(models):
            bd = ctx.evals[m].summary["by_date"]
            vals = [bd.get(d, {}).get(k, np.nan) for d in dates]
            ax.bar(np.arange(len(dates)) + (i - (len(models) - 1) / 2) * w, vals, w, color=ctx.color(m),
                   label=ctx.label(m))
        ax.set_xticks(range(len(dates)), [dl.get(d, d).replace(" ", "\n", 1) for d in dates], fontsize=6.3)
        ax.set_title(lab)
        ax.grid(axis="y", lw=0.3, alpha=0.6)
        if k in ("roc_auc",):
            ax.set_ylim(0.5, 1.0)
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper center", ncol=min(6, len(models)), frameon=False, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    return P.save(fig, ctx.out, name)


# ---------------------------------------------------------------------------
# Figure: reliability + sharpness
# ---------------------------------------------------------------------------

def fig_reliability(ctx: VizContext, models: list[str], name="fig_reliability", by_group_model=None):
    models = [m for m in models if m in ctx.evals and ctx.evals[m].reliability is not None]
    by_group_model = by_group_model or ctx.primary_model
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=0.6)
    for m in models:
        r = ctx.evals[m].reliability
        r = r[r.group == "overall"]
        ece = _overall(ctx.evals[m], "classwise_ece")
        ax.plot(r.mean_predicted, r.observed_frequency, "-o", ms=2.5, lw=1, color=ctx.color(m),
                label=f"{ctx.label(m)} (ECE {ece:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted P(smoke)", ylabel="Observed ADP smoke frequency",
           title="(a) Reliability, all held-out scenes")
    ax.set_aspect("equal")
    ax.legend(fontsize=5.6, loc="upper left", frameon=False)
    # by event for primary model
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", lw=0.6)
    r = ctx.evals[by_group_model].reliability
    gl = ctx.cfg.get("event_labels", {})
    for gname, g in r[r.group != "overall"].groupby("group"):
        lab = _stratum_label(ctx, gname.split("=", 1)[-1])
        ax.plot(g.mean_predicted, g.observed_frequency, "-o", ms=2.5, lw=1, label=lab)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted P(smoke)", title=f"(b) {ctx.label(by_group_model)} by stratum")
    ax.set_aspect("equal")
    ax.legend(fontsize=5.6, loc="upper left", frameon=False)
    # sharpness: distribution of forecast probabilities on the evaluated scenes
    ax = axes[2]
    hist = probability_histograms(ctx, models)
    hist.to_csv(ctx.out / "tables" / "probability_histograms.csv", index=False)
    for m in models:
        ax.step(hist.lo, hist[m], where="post", lw=1, color=ctx.color(m))
    ax.set_yscale("log")
    ax.set(xlabel="Predicted P(smoke)", ylabel="Fraction of pixels", title="(c) Sharpness (held-out pixels)",
           xlim=(0, 1))
    fig.tight_layout()
    return P.save(fig, ctx.out, name)


# ---------------------------------------------------------------------------
# Figure: uncertainty usefulness (risk-coverage, calibration of uncertainty)
# ---------------------------------------------------------------------------

UNC_LABELS = {"predictive_entropy": "Predictive entropy", "ensemble_mutual_information": "Ensemble MI",
              "ensemble_total_entropy": "Ensemble total entropy", "ensemble_variance": "Ensemble variance",
              "assignment_vote_variance": "Assignment vote variance", "parent_entropy": "Parent-cluster entropy",
              "child_entropy": "Child-cluster entropy", "margin_uncertainty": "Margin",
              "ensemble_expected_entropy": "Ensemble expected entropy"}


def fig_uncertainty(ctx: VizContext, models: list[str], measures: list[str], name="fig_uncertainty"):
    pm = ctx.primary_model
    ev = ctx.evals[pm]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax = axes[0]
    rc = ev.risk_coverage
    cmap = plt.get_cmap("tab10")
    for i, u in enumerate(measures):
        g = rc[rc.uncertainty == u].sort_values("coverage")
        if g.empty:
            continue
        ax.plot(g.coverage, g.risk, lw=1.1, color=cmap(i), label=UNC_LABELS.get(u, u))
    full = rc[rc.coverage == rc.coverage.max()].risk.mean()
    ax.axhline(full, color="k", ls=":", lw=0.7, label="No abstention")
    sr = ev.summary.get("selective_rule", {})
    rr = sr.get("rules", {}).get(sr.get("measure", ""), {})
    if rr.get("coverage"):
        ax.plot(rr["coverage"], rr["retained_error"], "k*", ms=7, label="Selective rule")
    ax.set(xlim=(0, 1), ylim=(0, max(0.3, full * 1.4)), xlabel="Coverage (fraction retained)",
           ylabel="Pixel error rate", title=f"(a) Risk-coverage, {ctx.label(pm)}")
    ax = axes[1]
    uc = ev.unc_calibration
    ax.plot([0, 0.5], [0, 0.5], "k--", lw=0.6)
    for i, u in enumerate(measures):
        g = uc[uc.uncertainty == u]
        if g.empty or not g.implied_error.notna().any():
            continue
        ax.plot(g.implied_error, g.observed_error, "-o", ms=2.5, lw=1, color=cmap(i))
    ax.set(xlabel="Error implied by uncertainty", ylabel="Observed error", xlim=(0, 0.5), ylim=(0, 0.5),
           title="(b) Uncertainty calibration")
    ax.set_aspect("equal")
    # error-detection AUROC across models for each measure
    ax = axes[2]
    meas = [u for u in measures]
    w = 0.8 / max(1, len(models))
    for i, m in enumerate([m for m in models if m in ctx.evals]):
        e = ctx.evals[m]
        vals = [_overall(e, f"{u}_error_auroc") if _overall(e, f"{u}_error_auroc") is not None else np.nan for u in meas]
        ax.barh(np.arange(len(meas)) + (i - (len(models) - 1) / 2) * w, vals, w, color=ctx.color(m), label=ctx.label(m))
    ax.axvline(0.5, color="k", lw=0.6, ls=":")
    ax.set_yticks(range(len(meas)), [UNC_LABELS.get(u, u) for u in meas], fontsize=6)
    ax.invert_yaxis()
    ax.set(xlim=(0.4, 0.9), xlabel="Error-detection AUROC", title="(c) Error-detection skill")
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = ax.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.legend(h1, l1, loc="lower left", bbox_to_anchor=(0.0, -0.02), ncol=4, frameon=False, fontsize=6)
    fig.legend(h2, l2, loc="lower right", bbox_to_anchor=(1.0, -0.02), ncol=1, frameon=False, fontsize=6)
    for a in axes:
        a.xaxis.labelpad = 1
    return P.save(fig, ctx.out, name)


# ---------------------------------------------------------------------------
# Figure: areas (scene-level intervals + hourly time series)
# ---------------------------------------------------------------------------

def fig_areas(ctx: VizContext, fs: pd.DataFrame | None, name="fig_areas"):
    pm, bm = ctx.primary_model, ctx.baseline_model
    ps = ctx.evals[pm].per_scene
    dates = sorted(ps.date.unique())
    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, len(dates), height_ratios=[1.15, 1], hspace=0.55, wspace=0.3)
    ax = fig.add_subplot(gs[0, :2] if len(dates) > 2 else gs[0, 0])
    sc = 1e-3
    gl = ctx.cfg.get("date_labels", {})
    cmap = plt.get_cmap("Dark2")
    for i, d in enumerate(dates):
        g = ps[ps.date == d]
        cov = ((g.c1_ref_area_km2 >= g.c1_conformal_lo) & (g.c1_ref_area_km2 <= g.c1_conformal_hi)).mean()
        ax.errorbar(g.c1_ref_area_km2 * sc, g.c1_discrete_area * sc,
                    yerr=[(g.c1_discrete_area - g.c1_conformal_lo).clip(lower=0) * sc,
                          (g.c1_conformal_hi - g.c1_discrete_area).clip(lower=0) * sc],
                    fmt="o", ms=2.6, lw=0.5, alpha=0.75, color=cmap(i),
                    label=f"{gl.get(d, d)}: 90% coverage {cov:.2f}")
    mx = max(ps.c1_ref_area_km2.max(), ps.c1_conformal_hi.max()) * sc * 1.05
    ax.plot([0, mx], [0, mx], "k--", lw=0.6)
    ax.set(xlim=(0, mx), ylim=(0, mx), xlabel="ADP smoke area (10$^3$ km$^2$)",
           ylabel=f"{ctx.label(pm)} area (10$^3$ km$^2$)",
           title="(a) Scene smoke area with split-conformal 90% intervals")
    ax.legend(fontsize=5.8, frameon=False, loc="upper left")
    # coverage by group summary
    ax2 = fig.add_subplot(gs[0, 2:] if len(dates) > 2 else gs[0, 1])
    ai = ctx.evals[pm].summary.get("areas", {})
    aic = None
    p = ctx.cfg.get("evaluation_dir")
    if p:
        f = Path(p) / pm / "area_interval_calibration.json"
        if f.exists():
            import json
            aic = json.loads(f.read_text()).get("class_1")
    if aic:
        names, cov, ns = [], [], []
        for gname, g in aic.get("groups", {}).items():
            names.append(ctx.cfg.get("short_event_labels", {}).get(gname, gname) + f"\n(n={g['n_evaluated']})")
            cov.append(g["evaluated_coverage"])
        if aic.get("pooled_fallback"):
            pf = aic["pooled_fallback"]
            names.append(f"Pooled\nfallback\n(n={pf['n_evaluated']})")
            cov.append(pf["evaluated_coverage"])
        names.append(f"All\n(n={len(ps)})")
        cov.append(aic["evaluated_coverage"])
        a1 = ai.get("class_1", {})
        names += ["Analytic", "Ensemble"]
        cov += [a1.get("analytic_interval_coverage", np.nan), a1.get("ensemble_interval_coverage", np.nan)]
        cols = ["#1b9e77"] * (len(cov) - 2) + ["#bbbbbb", "#bbbbbb"]
        ax2.bar(range(len(cov)), cov, color=cols, edgecolor="k", lw=0.4)
        ax2.axhline(aic.get("level", 0.9), color="k", ls="--", lw=0.7)
        names[-2:] = ["Analytic\n(binomial)", "Ensemble\nspread"]
        ax2.set_xticks(range(len(cov)), [n.replace("\n", " ", 1) if n.startswith("Pooled") else n for n in names],
                       fontsize=5.6, rotation=35, ha="right", rotation_mode="anchor")
        for i, c in enumerate(cov):
            ax2.text(i, c + 0.02, f"{c:.2f}", ha="center", fontsize=6)
        ax2.set(ylim=(0, 1.05), ylabel="Empirical coverage", title="(b) Area-interval coverage (nominal 0.90)")
    # hourly time series per holdout date
    for i, d in enumerate(dates):
        ax = fig.add_subplot(gs[1, i])
        g = ps[ps.date == d].copy()
        g["t"] = pd.to_datetime(g.pair_id.str.extract(r"(\d{8}T\d{6})Z")[0], format="%Y%m%dT%H%M%S", utc=True)
        g["scan"] = g.pair_id.str.extract(r"_S(\d{3})G")[0]
        a = g.groupby("scan").agg(t=("t", "median"), ref=("c1_ref_area_km2", "sum"), disc=("c1_discrete_area", "sum"),
                                  exp=("c1_expected_area", "sum"), lo=("c1_conformal_lo", "sum"),
                                  hi=("c1_conformal_hi", "sum")).sort_values("t")
        x = np.arange(len(a))                     # scans in time order (overnight gaps collapsed)
        ax.fill_between(x, a.lo * sc, a.hi * sc, color=ctx.color(pm), alpha=0.18, lw=0, label="Summed 90% intervals")
        ax.plot(x, a.ref * sc, "k-o", ms=2.5, lw=1, label="ADP")
        ax.plot(x, a.disc * sc, "-", color=ctx.color(pm), lw=1.2, label=f"{ctx.label(pm)} mask")
        ax.plot(x, a.exp * sc, "--", color=ctx.color(pm), lw=1.0, label="Expected area $\\Sigma p$")
        if bm in ctx.evals and ctx.evals[bm].per_scene is not None:
            b = ctx.evals[bm].per_scene
            b = b[b.date == d].copy()
            b["scan"] = b.pair_id.str.extract(r"_S(\d{3})G")[0]
            bb = b.groupby("scan").c1_discrete_area.sum().reindex(a.index)
            ax.plot(x, bb.values * sc, "-", color=ctx.color(bm), lw=1.0, label=ctx.label(bm))
        step = max(1, len(a) // 6)
        ax.set_xticks(x[::step], [f"{t:%H}" for t in a.t.iloc[::step]])
        ax.set_title(f"({chr(99 + i)}) {gl.get(d, d)}")
        ax.set_xlabel("Scan time (UTC hour)")
        if i == 0:
            ax.set_ylabel("Scan smoke area (10$^3$ km$^2$)")
    h, l_ = ax.get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=6.5)
    return P.save(fig, ctx.out, name)


# ---------------------------------------------------------------------------
# Figure: comparison with positive-only HMS polygons
# ---------------------------------------------------------------------------

def fig_hms(ctx: VizContext, fs: pd.DataFrame, models: list[str], name="fig_hms_consistency"):
    """Quick-look, positive-only comparison: P by HMS density class (polygons) or inside/outside the HMS daily
    extent (daily rasters) per event, and the fraction of HMS area masked on held-out scans."""
    if "hms_area" not in fs:
        LOG.warning("no HMS statistics; skipping %s", name)
        return []
    d = fs[fs.hms_area.notna()]
    evs = list(dict.fromkeys(d.sort_values("frame_time").event))
    ms = [m for m in models if m in set(d.model)]
    dens = "n_hms_light" in d and d["n_hms_light"].fillna(0).sum() > 0
    cls = ["none", "light", "medium", "heavy"] if dens else ["out", "in"]
    el = ctx.cfg.get("event_labels", {})
    fig, axes = plt.subplots(1, len(evs) + 1, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1] * len(evs) + [0.8]})
    w = 0.8 / len(ms)
    for ax, ev in zip(axes, evs):
        for i, m in enumerate(ms):
            g = d[(d.model == m) & (d.event == ev)]
            vals = []
            for c in cls:
                if dens:
                    n = g[f"n_hms_{c}"].fillna(0)
                    vals.append(np.average(g[f"mean_p_hms_{c}"].fillna(0), weights=n) if n.sum() else np.nan)
                else:
                    vals.append(np.nanmean(g[f"mean_p_{c}_hms"]))
            ax.bar(np.arange(len(cls)) + (i - (len(ms) - 1) / 2) * w, vals, w, color=ctx.color(m), label=ctx.label(m))
        if dens:
            ax.set_xticks(range(4), ["outside", "light", "medium", "heavy"])
            ax.set_xlabel("HMS smoke density class")
        else:
            ax.set_xticks(range(2), ["outside HMS", "inside HMS"])
            ax.set_xlabel("HMS daily smoke extent")
        ax.set_title(f"{el.get(ev, ev)} (all analysed scans)")
    axes[0].set_ylabel("Mean P(smoke) / mask fraction")
    ax = axes[-1]
    tab = hms_consistency_table(fs, ("holdout",))
    for i, m in enumerate(ms):
        vals = [tab[(tab.event == e) & (tab.model == m)].hms_detect_frac.mean() for e in evs]
        ax.bar(np.arange(len(evs)) + (i - (len(ms) - 1) / 2) * w, vals, w, color=ctx.color(m))
    ax.set_xticks(range(len(evs)), [ctx.cfg.get("short_event_labels", {}).get(e, e) for e in evs])
    ax.set(ylim=(0, max(0.5, float(np.nanmax(tab.hms_detect_frac)) * 1.25) if len(tab) else 1),
           ylabel="Fraction of HMS area masked", title="Held-out scans")
    h, l_ = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(h, l_, loc="upper center", ncol=len(ms), frameon=False, bbox_to_anchor=(0.5, 1.0))
    for k, a in enumerate(axes):
        P.panel_label(a, f"({chr(97 + k)})")
    return P.save(fig, ctx.out, name)


# ---------------------------------------------------------------------------
# Figure: scan-level agreement with the two-sided references (ADP primary, VIIRS secondary)
# ---------------------------------------------------------------------------

def reference_agreement_table(fs: pd.DataFrame, refs: list[str], splits=("holdout",), by="date") -> pd.DataFrame:
    """Area-pooled contingency per (date|event) x model x reference from frame statistics."""
    rows = []
    d = fs[fs.split.isin(splits)]
    for r in refs:
        if f"{r}_tp" not in d:
            continue
        g0 = d[d[f"{r}_valid_area"].fillna(0) > 0]
        for keys, g in list(g0.groupby([by, "model"])) + [(("all", m), gm) for m, gm in g0.groupby("model")]:
            tp, fp, fn = g[f"{r}_tp"].sum(), g[f"{r}_fp"].sum(), g[f"{r}_fn"].sum()
            clr, ref = g[f"{r}_clear_area"].sum(), g[f"{r}_area"].sum()
            rows.append({by: keys[0], "model": keys[1], "reference": r, "n_frames": len(g),
                         "ref_valid_area_km2": g[f"{r}_valid_area"].sum(), "ref_smoke_area_km2": ref,
                         "iou": tp / (tp + fp + fn) if tp + fp + fn > 0 else np.nan,
                         "hit_rate": tp / ref if ref > 0 else np.nan,
                         "precision": tp / (tp + fp) if tp + fp > 0 else np.nan,
                         "false_alarm_rate": fp / clr if clr > 0 else np.nan,
                         "mean_p_ref_smoke": np.average(g[f"{r}_mean_p_smoke"].fillna(0), weights=g[f"{r}_area"])
                         if ref > 0 else np.nan,
                         "mean_p_ref_clear": np.average(g[f"{r}_mean_p_clear"].fillna(0), weights=g[f"{r}_clear_area"])
                         if clr > 0 else np.nan})
    return pd.DataFrame(rows)


def fig_reference_agreement(ctx: VizContext, fs: pd.DataFrame, models: list[str], primary: str,
                            secondary: str | None = None, split="holdout", name="fig_reference_agreement",
                            min_secondary_smoke_km2: float = 500.0, min_primary_smoke_frac: float = 0.02):
    """Rows: IoU and false-alarm rate per scan through each holdout day. Lines: primary reference (ADP) on
    ADP-valid pixels; markers: secondary reference (VIIRS) at matched overpasses (IoU only where VIIRS saw at
    least `min_secondary_smoke_km2` of smoke; false-alarm rate on VIIRS-clear pixels)."""
    if f"{primary}_iou" not in fs:
        LOG.warning("no %s statistics; skipping %s", primary, name)
        return []
    d = fs[fs.split == split]
    dates = sorted(d.date.unique())
    ms = [m for m in models if m in set(d.model)]
    gl = ctx.cfg.get("date_labels", {})
    hrs = ctx.cfg.get("case_utc_hours")
    fig, axes = plt.subplots(2, len(dates), figsize=(7.2, 3.9), sharey="row", squeeze=False)
    pl = _ref_label(ctx, primary).replace("\n", " ")
    sl = _ref_label(ctx, secondary).replace("\n", " ") if secondary else None
    for j, day in enumerate(dates):
        g = d[d.date == day].sort_values("frame_time")
        if hrs:
            h = g.frame_time.dt.hour
            g = g[(h >= hrs[0] - 1) & (h <= hrs[1] + 1)]
        # complete scans only (partial dawn/dusk scans cover a few granules and give unstable ratios)
        g = g[g.valid_area >= 0.5 * g.valid_area.median()]
        for m in ms:
            gm = g[g.model == m]
            x = gm.frame_time.dt.hour + gm.frame_time.dt.minute / 60
            # IoU only where the reference saw a non-trivial smoke area (else IoU is dominated by a few pixels)
            ok = gm[f"{primary}_area"].fillna(0) >= min_primary_smoke_frac * gm[f"{primary}_valid_area"].fillna(0).clip(lower=1)
            ok &= gm[f"{primary}_area"].fillna(0) > 0
            axes[0, j].plot(x[ok], gm[f"{primary}_iou"][ok], "-o", ms=2.2, lw=1.1, color=ctx.color(m),
                            label=f"{ctx.short(m)} vs {pl}")
            axes[1, j].plot(x, gm[f"{primary}_fpr"], "-o", ms=2.2, lw=1.1, color=ctx.color(m))
            if secondary and f"{secondary}_iou" in gm:
                sv = gm[f"{secondary}_valid_area"].fillna(0) > 0
                big = sv & (gm[f"{secondary}_area"].fillna(0) >= min_secondary_smoke_km2)
                axes[0, j].plot(x[big], gm[f"{secondary}_iou"][big], "D", ms=4.2, mfc="white", mec=ctx.color(m),
                                mew=1.1, ls="none", label=f"{ctx.short(m)} vs {sl}")
                axes[1, j].plot(x[sv], gm[f"{secondary}_fpr"][sv], "D", ms=4.2, mfc="white", mec=ctx.color(m),
                                mew=1.1, ls="none")
        if secondary and f"{secondary}_area" in g:
            gp = g[g.model == ms[0]]
            sv = gp[f"{secondary}_valid_area"].fillna(0) > 0
            tot = gp.loc[sv, f"{secondary}_area"].sum()
            ns = int(sv.sum())
            axes[0, j].text(0.13, 0.97, f"{sl}: {ns} matched scan{'s' if ns != 1 else ''}, {tot:,.0f} km$^2$ smoke",
                            transform=axes[0, j].transAxes, ha="left", va="top", fontsize=5.4, color="#333")
        axes[0, j].set_title(gl.get(day, day), fontsize=7.5)
        axes[1, j].set_xlabel("Scan time (UTC hour)")
        for a in axes[:, j]:
            a.grid(lw=0.3, alpha=0.5)
    axes[0, 0].set_ylabel("Smoke IoU")
    axes[1, 0].set_ylabel("False-alarm rate\n(masked / reference-clear area)")
    axes[0, 0].set_ylim(0, float(np.nanmax([a.get_ylim()[1] for a in axes[0]])) * 1.12)
    axes[1, 0].set_ylim(0, None)
    h, l_ = [], []
    for a in axes[0]:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in l_:
                h.append(hh)
                l_.append(ll)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(h, l_, loc="upper center", ncol=min(4, len(l_)), frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=6.2)
    for k, a in enumerate(axes.ravel()):
        P.panel_label(a, f"({chr(97 + k)})")
    return P.save(fig, ctx.out, name)


# ---------------------------------------------------------------------------
# Reference vs reference (e.g. VIIRS vs ADP): disagreement among the comparison products
# ---------------------------------------------------------------------------

PAIR_CATS = (("both_smoke", "Both smoke"), ("a_only", "{a} only"), ("b_only", "{b} only"), ("both_clear", "Both clear"))


def reference_pair_statistics(ctx: VizContext, a: str, b: str, models: list[str], cache: Path | None = None,
                              splits=None) -> pd.DataFrame:
    """Per scan where both two-sided references have data (e.g. ADP and a matched VIIRS overpass), on pixels
    valid in both and in the product:
      pair columns   joint_valid_area, a_valid_area, b_valid_area, area_both_smoke, area_a_only, area_b_only,
                     area_both_clear, ab_iou, ab_agreement (area fraction), ab_kappa (area-weighted Cohen's kappa),
                     a_smoke_frac, b_smoke_frac
      model columns  (one row per model) iou_vs_a, iou_vs_b on the same joint pixels, and for each agreement
                     category (both_smoke, a_only, b_only, both_clear): mean_p_<cat>, flag_frac_<cat>
    Reference-vs-reference columns do not depend on the model and are repeated on each model row."""
    if cache is not None and cache.exists():
        ps = pd.read_csv(cache, dtype={"date": str})
        ps["frame_time"] = pd.to_datetime(ps.frame_time, utc=True, format="mixed")
        if {"ab_iou", "iou_vs_a"} <= set(ps) and not ps.empty and \
                (ps.reference_a.iloc[0], ps.reference_b.iloc[0]) == (a, b) and set(models) <= set(ps.model):
            return ps
        LOG.info("reference pair cache %s does not match %s / %s and the models; recomputing", cache, a, b)
    ra, rb = ctx.refs.get(a), ctx.refs.get(b)
    if not (isinstance(ra, D.RasterReference) and isinstance(rb, D.RasterReference)):
        LOG.warning("reference pair %s / %s: both must be raster references; skipped", a, b)
        return pd.DataFrame()
    A = ctx.pixel_area
    s = ctx.spec
    fr = ctx.frames if splits is None else ctx.frames[ctx.frames.split.isin(splits)]
    rows = []
    for fid, g in fr.groupby("frame_id"):
        # cheap pre-check: the sparser reference (VIIRS) must have a file for at least one granule
        if not any(rb._lookup(p) for p in g.pair_id) or not any(ra._lookup(p) for p in g.pair_id):
            continue
        xa, xb = ctx.reference(a, fid)[0], ctx.reference(b, fid)[0]
        if xa is None or xb is None:
            continue
        p0 = ctx.band(ctx.primary_model, fid, s.prob_band)
        pv = np.isfinite(p0) if p0 is not None else np.ones(A.shape, bool)
        va, vb = pv & np.isfinite(xa), pv & np.isfinite(xb)
        j = va & vb
        if not j.any():
            continue
        sa, sb = j & (xa >= 0.5), j & (xb >= 0.5)
        cat = {"both_smoke": sa & sb, "a_only": sa & ~sb, "b_only": ~sa & sb, "both_clear": j & ~sa & ~sb}
        W = A[j].sum()
        ar = {k: A[m].sum() for k, m in cat.items()}
        po = (ar["both_smoke"] + ar["both_clear"]) / W
        pa, pb = A[sa].sum() / W, A[sb].sum() / W
        pe = pa * pb + (1 - pa) * (1 - pb)
        den = ar["both_smoke"] + ar["a_only"] + ar["b_only"]
        base = {"frame_id": fid, "frame_time": g.frame_time.iloc[0], "date": g.date.iloc[0], "split": g.split.iloc[0],
                "event": g.event.iloc[0], "reference_a": a, "reference_b": b,
                "a_valid_area": A[va].sum(), "b_valid_area": A[vb].sum(), "joint_valid_area": W,
                **{f"area_{k}": v for k, v in ar.items()},
                "ab_iou": ar["both_smoke"] / den if den > 0 else np.nan, "ab_agreement": po,
                "ab_kappa": (po - pe) / (1 - pe) if pe < 1 else np.nan, "a_smoke_frac": pa, "b_smoke_frac": pb}
        for m in models:
            p = ctx.band(m, fid, s.prob_band)
            if p is None:
                continue
            mask = ctx.band(m, fid, s.mask_band)
            v = np.isfinite(p)
            mk = (np.nan_to_num(mask) >= 0.5) & v if mask is not None else (np.nan_to_num(p) >= ctx.threshold(m)) & v
            row = dict(base, model=m)
            for nm, ref_s in (("a", sa), ("b", sb)):
                tp, fp, fn = A[ref_s & mk].sum(), A[j & ~ref_s & mk].sum(), A[ref_s & ~mk].sum()
                row[f"iou_vs_{nm}"] = tp / (tp + fp + fn) if tp + fp + fn > 0 else np.nan
                row[f"tp_vs_{nm}"], row[f"fp_vs_{nm}"], row[f"fn_vs_{nm}"] = tp, fp, fn
            for k, mm in cat.items():
                sel = mm & v
                row[f"mean_p_{k}"] = float(np.average(p[sel], weights=A[sel])) if sel.any() else np.nan
                row[f"flag_frac_{k}"] = float(A[sel & mk].sum() / A[sel].sum()) if sel.any() else np.nan
            rows.append(row)
    ps = pd.DataFrame(rows)
    if ps.empty:
        LOG.warning("reference pair %s / %s: no scans with both references", a, b)
        return ps
    ps = ps.sort_values(["frame_time", "model"])
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        ps.to_csv(cache, index=False)
    LOG.info("reference pair %s / %s: %d scans with both references", a, b, ps.frame_id.nunique())
    return ps


def reference_pair_table(ps: pd.DataFrame, splits=("holdout",), by="date") -> pd.DataFrame:
    """Area-pooled reference-vs-reference agreement per (date|event|split) and overall, with each model's IoU
    against either reference on the jointly valid pixels and its mean P / flagged fraction per category."""
    if ps.empty:
        return ps
    d = ps[ps.split.isin(splits)] if splits else ps
    rows = []
    for key, g in list(d.groupby(by)) + [("all", d)]:
        f1 = g.drop_duplicates("frame_id")
        W = f1.joint_valid_area.sum()
        ar = {k: f1[f"area_{k}"].sum() for k, _ in PAIR_CATS}
        po = (ar["both_smoke"] + ar["both_clear"]) / W if W else np.nan
        pa, pb = (ar["both_smoke"] + ar["a_only"]) / W, (ar["both_smoke"] + ar["b_only"]) / W
        pe = pa * pb + (1 - pa) * (1 - pb)
        den = ar["both_smoke"] + ar["a_only"] + ar["b_only"]
        pair = {by: key, "reference_a": f1.reference_a.iloc[0], "reference_b": f1.reference_b.iloc[0],
                "n_scans": len(f1), "joint_valid_area_km2": W, **{f"area_{k}_km2": v for k, v in ar.items()},
                "ab_iou": ar["both_smoke"] / den if den else np.nan, "ab_agreement": po,
                "ab_kappa": (po - pe) / (1 - pe) if pe < 1 else np.nan,
                "a_smoke_frac": pa, "b_smoke_frac": pb,
                "b_smoke_confirmed_by_a": ar["both_smoke"] / (ar["both_smoke"] + ar["b_only"])
                if ar["both_smoke"] + ar["b_only"] else np.nan,
                "a_smoke_confirmed_by_b": ar["both_smoke"] / (ar["both_smoke"] + ar["a_only"])
                if ar["both_smoke"] + ar["a_only"] else np.nan}
        for m, gm in g.groupby("model"):
            r = dict(pair, model=m)
            for nm in ("a", "b"):
                tp, fp, fn = gm[f"tp_vs_{nm}"].sum(), gm[f"fp_vs_{nm}"].sum(), gm[f"fn_vs_{nm}"].sum()
                r[f"iou_vs_{nm}"] = tp / (tp + fp + fn) if tp + fp + fn else np.nan
            for k, _ in PAIR_CATS:
                w = gm[f"area_{k}"].where(gm[f"mean_p_{k}"].notna(), 0)
                r[f"mean_p_{k}"] = np.average(gm[f"mean_p_{k}"].fillna(0), weights=w) if w.sum() else np.nan
                r[f"flag_frac_{k}"] = np.average(gm[f"flag_frac_{k}"].fillna(0), weights=w) if w.sum() else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def fig_reference_disagreement(ctx: VizContext, ps: pd.DataFrame, models: list[str], a: str, b: str,
                               splits=("holdout",), name="fig_reference_disagreement"):
    """(a) Per matched scan: jointly valid area split into both smoke / a only / b only, with the a-b IoU;
    (b) mean model P(smoke) and (c) flagged fraction in each agreement category, per event."""
    d = ps[ps.split.isin(splits)] if (splits and not ps.empty) else ps
    if d.empty:
        LOG.warning("no scans with both %s and %s in %s; skipping %s", a, b, splits, name)
        return []
    la, lb = key_label(ctx, f"ref:{a}"), key_label(ctx, f"ref:{b}")
    cats = [(k, l_.format(a=la, b=lb)) for k, l_ in PAIR_CATS]
    ccol = {"both_smoke": P.DIFF_COLORS[1], "a_only": P.DIFF_COLORS[2], "b_only": P.DIFF_COLORS[3],
            "both_clear": "#bdbdbd"}
    f1 = d.drop_duplicates("frame_id").sort_values("frame_time").reset_index(drop=True)
    ms = [m for m in models if m in set(d.model)]
    evs = list(dict.fromkeys(f1.event))
    sev = ctx.cfg.get("short_event_labels", {})
    fig = plt.figure(figsize=(7.2, 4.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.55, wspace=0.22, left=0.08, right=0.99,
                          top=0.94, bottom=0.13)
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(f1))
    sc = 1e-3
    bot = np.zeros(len(f1))
    for k in ("both_smoke", "a_only", "b_only"):
        v = f1[f"area_{k}"].to_numpy() * sc
        ax.bar(x, v, 0.75, bottom=bot, color=ccol[k], label=dict(cats)[k])
        bot += v
    min_km2 = float(ctx.cfg.get("min_secondary_smoke_km2", 500))
    for xi, (h, iou) in enumerate(zip(bot, f1.ab_iou)):
        if np.isfinite(iou) and h / sc >= min_km2:   # IoU only where the references saw some smoke
            ax.text(xi, h + bot.max() * 0.02, f"{iou:.2f}", ha="center", va="bottom", fontsize=5.6)
    ax.set_xticks(x, [f"{sev.get(e, e)}\n{t:%m-%d %H:%M}" for e, t in zip(f1.event, f1.frame_time)], fontsize=5.4)
    ax.set_ylabel("Area (10$^3$ km$^2$)", fontsize=7)
    ax.set_ylim(0, bot.max() * 1.18 if bot.max() > 0 else 1)
    ax.set_xlim(-1.1, len(f1) - 0.4)               # room for the panel label
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_title(f"{lb} vs {la} on pixels valid in both, at matched scans (numbers: {la}-{lb} IoU)", fontsize=7.5)
    ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper right")
    P.panel_label(ax, "(a)")
    tab = reference_pair_table(d, None, by="event")
    w = 0.8 / max(1, len(ms))
    for col, (key, ylab) in enumerate((("mean_p", "Mean P(smoke)"), ("flag_frac", "Fraction flagged as smoke"))):
        ax = fig.add_subplot(gs[1, col])
        xt, xl = [], []
        for ie, ev in enumerate(evs):
            for ic, (k, l_) in enumerate(cats):
                xc = ie * (len(cats) + 0.8) + ic
                for im, m in enumerate(ms):
                    r = tab[(tab.event == ev) & (tab.model == m)]
                    val = float(r[f"{key}_{k}"].iloc[0]) if len(r) else np.nan
                    ax.bar(xc + (im - (len(ms) - 1) / 2) * w, val, w, color=ctx.color(m),
                           label=ctx.label(m) if (ie == 0 and ic == 0) else None)
                xt.append(xc)
                xl.append(l_.replace(" only", "\nonly").replace("Both ", "Both\n"))
            ax.text(ie * (len(cats) + 0.8) + (len(cats) - 1) / 2, 0.97, sev.get(ev, ev), ha="center",
                    va="top", fontsize=7, fontweight="bold", transform=ax.get_xaxis_transform())
        ax.set_xticks(xt, xl, fontsize=5.2)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(ylab, fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        P.panel_label(ax, f"({chr(98 + col)})")
    h, l_ = fig.axes[1].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=len(ms), frameon=False, fontsize=6.3, bbox_to_anchor=(0.5, 0.0))
    return P.save(fig, ctx.out, name)


# ---------------------------------------------------------------------------
# Map panels
# ---------------------------------------------------------------------------

class FrameData:
    """Lazy per-frame arrays for map panels and animation frames."""

    def __init__(self, ctx: VizContext, frame_id: str, zoom: bool = True):
        self.ctx, self.fid = ctx, frame_id
        self.view = ctx.view_extent(frame_id, zoom)
        self._c: dict[tuple, Any] = {}

    def band(self, model, band):
        k = ("b", model, band)
        if k not in self._c:
            self._c[k] = self.ctx.band(model, self.fid, band)
        return self._c[k]

    def ref(self, name):
        k = ("r", name)
        if k not in self._c:
            self._c[k] = self.ctx.reference(name, self.fid)
        return self._c[k]

    def mask(self, model):
        s = self.ctx.spec
        m = self.band(model, s.mask_band)
        if m is None:
            p = self.band(model, s.prob_band)
            m = None if p is None else np.where(np.isfinite(p), (p >= self.ctx.threshold(model)).astype(float), np.nan)
        return m

    def positive(self, key):
        """binary layer for a model or ref:name (positive-only refs: 0 outside, valid where the product is valid)."""
        if key.startswith("ref:"):
            r, _ = self.ref(key[4:])
            if r is None:
                return None
            v = np.isfinite(self.band(self.ctx.primary_model, self.ctx.spec.prob_band))
            return np.where(v & np.isfinite(r), r, np.nan)
        return self.mask(key)


def diff_keys(parts: list[str]) -> tuple[str, str]:
    """('a', 'b') from the tokens after 'diff'; either side may be <model> or ref:<name>."""
    if parts[0] == "ref":
        return f"ref:{parts[1]}", ":".join(parts[2:])
    return parts[0], ":".join(parts[1:])


def key_label(ctx: VizContext, key: str, short: bool = True) -> str:
    """Display label of a model or ref:<name> key."""
    if key.startswith("ref:"):
        rc = ctx.ref_cfg.get(key[4:], {})
        return rc.get("short_label", rc.get("label", key[4:])) if short else rc.get("label", key[4:])
    return (ctx.short(key) if short else ctx.label(key)).split(" (")[0]


def draw_panel(ax, fd: FrameData, spec_str: str, overlays: list[str], cbar=True, title=True, fig=None):
    ctx, s = fd.ctx, fd.ctx.spec
    ext = ctx.index.extent()
    parts = spec_str.split(":")
    kind = parts[0]
    im, legend, ttl = None, None, spec_str
    if kind == "mask":
        m = parts[1]
        a = fd.mask(m)
        cm, nm = P.mask_cmap(s.mask_color if m == ctx.primary_model else ctx.color(m))
        if a is not None:
            P.show(ax, a, ext, cm, nm)
        ttl = f"{ctx.short(m)} mask"
        legend = [Patch(fc=ctx.color(m), label=f"{s.label} (thr {ctx.threshold(m):.2f})")]
    elif kind == "prob":
        m = parts[1]
        a = fd.band(m, s.prob_band)
        if a is not None:
            im = P.show(ax, a, ext, s.prob_cmap, P.prob_norm(s))
        ttl, clab = f"{ctx.short(m)} P({s.label.lower()})", f"P({s.label.lower()})"
    elif kind == "envelope":
        m = parts[1]
        p = fd.band(m, s.prob_band)
        lo = fd.band(m, s.q_lo_band) if s.q_lo_band else None
        hi = fd.band(m, s.q_hi_band) if s.q_hi_band else None
        if p is not None:
            cm, nm = P.envelope_cmap(s.envelope_colors)
            P.show(ax, P.envelope(p, lo, hi, ctx.threshold(m)), ext, cm, nm)
        ttl = f"{ctx.short(m)} probabilistic extent"
        legend = [Patch(fc=c, label=l_) for c, l_ in zip(s.envelope_colors,
                                                          ["possible (q95$\\geq$thr)", "expected (p$\\geq$thr)",
                                                           "likely (q05$\\geq$thr)"])]
        if lo is None:
            legend = legend[1:2]
    elif kind == "width":
        m = parts[1]
        lo, hi = fd.band(m, s.q_lo_band), fd.band(m, s.q_hi_band)
        if lo is not None and hi is not None:
            im = P.show(ax, hi - lo, ext, "cividis", plt.Normalize(0, float(ctx.cfg.get("width_vmax", 0.4))))
        ttl, clab = f"{ctx.short(m)} ensemble 90% width", "q95 - q05 of P"
    elif kind == "unc":
        m, b = parts[1], parts[2]
        a = fd.band(m, b)
        vmax = float(ctx.cfg.get("unc_vmax", {}).get(b, np.nanpercentile(a, 99) if a is not None and
                                                      np.isfinite(a).any() else 1))
        if a is not None:
            im = P.show(ax, a, ext, s.unc_cmap, plt.Normalize(0, vmax))
        ttl = f"{ctx.short(m)} " + s.uncertainty_bands.get(b, UNC_LABELS.get(b, b))
        clab = UNC_LABELS.get(b, b)
    elif kind == "ref":
        r, poly = fd.ref(parts[1])
        rc = ctx.ref_cfg.get(parts[1], {})
        lab = rc.get("label", parts[1])
        if r is not None:
            cm, nm = P.mask_cmap(rc.get("color", s.reference_color))
            v = np.isfinite(fd.band(ctx.primary_model, s.prob_band))
            P.show(ax, np.where(v, r, np.nan), ext, cm, nm)
        ttl = lab + (" (positive-only)" if rc.get("positive_only") else "")
        legend = [Patch(fc=rc.get("color", s.reference_color), label=lab)]
    elif kind == "diff":
        a_key, b_key = diff_keys(parts[1:])
        a, b = fd.positive(a_key), fd.positive(b_key)
        if a is not None and b is not None:
            cm, nm = P.diff_cmap()
            P.show(ax, P.difference(a, b), ext, cm, nm)
        al, bl = key_label(ctx, a_key, short=a_key.startswith("ref:")), key_label(ctx, b_key)
        ttl = f"{al} vs {bl}"
        legend = [Patch(fc=c, ec="#999", lw=0.3, label=l_) for c, l_ in
                  zip(P.DIFF_COLORS[1:], ["both", f"{al} only", f"{bl} only"])]
    if kind == "ref":
        ref_keys = [parts[1]]
    elif kind == "diff":
        ref_keys = [k[4:] for k in diff_keys(parts[1:]) if k.startswith("ref:")]
    else:
        ref_keys = []
    ref_keys = [k for k in ref_keys if _two_sided(ctx, k)]
    if ref_keys:
        p = fd.band(ctx.primary_model, s.prob_band)
        rs = [fd.ref(k)[0] for k in ref_keys]
        if p is not None and all(r is not None for r in rs):
            # grey where any two-sided reference in the panel has no data
            miss = np.zeros(p.shape, bool)
            for r in rs:
                miss |= ~np.isfinite(r)
            gap = np.where(np.isfinite(p) & miss, 1.0, np.nan)
            P.show(ax, gap, ext, ListedColormap([NODATA_COLOR]), plt.Normalize(0, 1))
            if legend is not None:
                legend = legend + [Patch(fc=NODATA_COLOR, label="no reference data" if len(rs) == 1
                                         else "no data in either reference")]
    P.setup_map(ax, fd.view, ctx.basemap, ticks=False)
    for ov in (overlays if kind in ctx.cfg.get("overlay_panels", ["prob", "envelope"]) else []):
        rr, poly = fd.ref(ov)
        rc = ctx.ref_cfg.get(ov, {})
        if poly is not None:
            P.outline(ax, poly, rc.get("outline_color", "#00c8ff"), lw=0.7)
        elif rr is not None and np.nanmax(np.nan_to_num(rr)) >= 0.5:
            P.contour(ax, np.nan_to_num(rr), ctx.index, rc.get("outline_color", "#00c8ff"), lw=0.6, ls="--")
    ax.set_xlabel("")
    ax.set_ylabel("")
    if title:
        ax.set_title(ttl, fontsize=7.2, pad=2)
    if cbar and im is not None:
        cax = ax.inset_axes([0.06, -0.075, 0.88, 0.035])
        cb = (fig or ax.figure).colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=6, length=2, pad=1)
        cb.set_label(clab, fontsize=6.3, labelpad=1)
    elif legend:
        wide = len(legend) > 2 or sum(len(h.get_label()) for h in legend) > float(ctx.cfg.get("legend_row_chars", 34))
        ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=1 if wide else len(legend), fontsize=5.8, frameon=False, handlelength=1, borderaxespad=0.2, labelspacing=0.2)
    return ax


NODATA_COLOR = "#d9d9d9"


def _two_sided(ctx: VizContext, name: str) -> bool:
    """Raster references with explicit negatives (ADP, VIIRS, FDCC): no-data is drawn grey, not as 'no smoke'."""
    rc = ctx.ref_cfg.get(name, {})
    return isinstance(ctx.refs.get(name), D.RasterReference) and not rc.get("positive_only", False)


def pick_case_frames(ctx: VizContext, fs: pd.DataFrame, split="holdout", per_date=1, min_granules=None):
    """Most smoke-laden complete scans per date (ADP area if evaluated, else primary-model area)."""
    pm = ctx.primary_model
    ps = ctx.evals[pm].per_scene if pm in ctx.evals else None
    fr = ctx.frames[ctx.frames.split == split]
    hrs = ctx.cfg.get("case_utc_hours")          # e.g. [15, 23]: avoid high solar-zenith scans
    if hrs:
        h = fr.frame_time.dt.hour
        fr = fr[(h >= hrs[0]) & (h <= hrs[1])]
    min_g = min_granules or int(fr.groupby("frame_id").size().quantile(0.75))
    out = []
    for d, g in fr.groupby("date"):
        cnt = g.groupby("frame_id").size()
        cand = cnt[cnt >= min_g].index
        if ps is not None:
            sc = ps[ps.pair_id.isin(g.pair_id)].merge(g[["pair_id", "frame_id"]], on="pair_id")
            score = sc.groupby("frame_id").c1_ref_area_km2.sum().reindex(cand).dropna()
        else:
            score = fs[(fs.model == pm) & fs.frame_id.isin(cand)].set_index("frame_id").discrete_area
        out += list(score.sort_values(ascending=False).index[:per_date])
    return out


def fig_case_maps(ctx: VizContext, frame_ids: list[str], panels: list[list[str]], overlays: list[str],
                  name="fig_case", zoom=True):
    """One figure per case frame; `panels` is a grid (rows of panel strings)."""
    paths = []
    for fid in frame_ids:
        fd = FrameData(ctx, fid, zoom)
        nr, nc = len(panels), max(len(r) for r in panels)
        v = fd.view
        aspect = (v[3] - v[2]) / np.cos(np.deg2rad(0.5 * (v[2] + v[3]))) / (v[1] - v[0])
        W, mx, wsp = 7.2, 0.06, 0.1                 # inches
        pw = (W - 2 * mx - (nc - 1) * wsp) / nc
        ph = pw * aspect
        below, ttl_h, top, bot = 0.54, 0.2, 0.32, 0.12   # colorbar/legend band, panel title, suptitle, footer
        H = top + bot + nr * (ttl_h + ph + below)
        fig = plt.figure(figsize=(W, H))
        axes = np.empty((nr, nc), dtype=object)
        for i in range(nr):
            y0 = H - top - (i + 1) * (ttl_h + ph + below) + below
            for j in range(nc):
                axes[i, j] = fig.add_axes([(mx + j * (pw + wsp)) / W, y0 / H, pw / W, ph / H])
        k = 0
        for i, row in enumerate(panels):
            for j in range(nc):
                ax = axes[i, j]
                if j >= len(row):
                    ax.axis("off")
                    continue
                draw_panel(ax, fd, row[j], overlays, fig=fig)
                P.panel_label(ax, f"({chr(97 + k)})")
                k += 1
        t = ctx.frame_time(fid)
        ev = ctx.frames.loc[ctx.frames.frame_id == fid, "event"].iloc[0]
        sp = ctx.frames.loc[ctx.frames.frame_id == fid, "split"].iloc[0]
        fig.suptitle(f"{ctx.cfg.get('event_labels', {}).get(ev, ev)}  {t:%Y-%m-%d %H:%M} UTC  "
                     f"(TEMPO scan {fid.split('_S')[-1]}, {sp} day)", fontsize=8.5, y=1 - 0.12 / H)
        if overlays:
            fig.text(0.5, 0.04 / H, overlay_note(ctx, overlays), ha="center", fontsize=6.5, color="#333")
        paths += P.save(fig, ctx.out, f"{name}_{fid}")
    return paths


def overlay_note(ctx: VizContext, overlays: list[str]) -> str:
    parts = []
    for o in overlays:
        rc = ctx.ref_cfg.get(o, {})
        lab = rc.get("label", o)
        if isinstance(ctx.refs.get(o), D.HMSPolygons):
            parts.append(f"cyan outlines: {lab} polygons valid at scan time")
        else:
            parts.append(f"dashed cyan: {lab} (quick reference)")
    t = "; ".join(parts)
    return t[:1].upper() + t[1:]


def fig_sequence(ctx: VizContext, frame_ids: list[str], rows: list[str], overlays: list[str], name="fig_sequence"):
    """Hourly evolution: columns = scans, rows = panel kinds (e.g. prob:{pm}, envelope:{pm})."""
    fds = [FrameData(ctx, f) for f in frame_ids]
    v = fds[0].view
    aspect = (v[3] - v[2]) / np.cos(np.deg2rad(0.5 * (v[2] + v[3]))) / (v[1] - v[0])
    nc, nr = len(frame_ids), len(rows)
    W, mx, wsp, ttl_h, below, top = 7.2, 0.06, 0.06, 0.2, 0.08, 0.3
    pw = (W - 2 * mx - (nc - 1) * wsp) / nc
    ph = pw * aspect
    leg = 0.55
    H = top + nr * (ttl_h + ph + below) + leg
    fig = plt.figure(figsize=(W, H))
    k = 0
    for i, r in enumerate(rows):
        y0 = H - top - (i + 1) * (ttl_h + ph + below) + below
        for j, fd in enumerate(fds):
            ax = fig.add_axes([(mx + j * (pw + wsp)) / W, y0 / H, pw / W, ph / H])
            last = j == nc - 1
            draw_panel(ax, fd, r, overlays, cbar=False, title=False, fig=fig)
            if ax.get_legend():
                ax.get_legend().remove()
            if i == 0:
                ax.set_title(f"{ctx.frame_time(fd.fid):%m-%d %H:%M} UTC", fontsize=7, pad=2)
            if j == 0:
                ax.text(-0.04, 0.5, draw_label(ctx, r), transform=ax.transAxes, rotation=90, va="center", ha="right",
                        fontsize=7)
            P.panel_label(ax, f"({chr(97 + k)})")
            k += 1
            if last and i == nr - 1:
                pass
    # shared legends / colorbars
    s = ctx.spec
    hs = []
    for r in rows:
        kind = r.split(":")[0]
        if kind == "envelope":
            hs += [Patch(fc=c, label=l_) for c, l_ in zip(s.envelope_colors, ["possible (q95$\\geq$thr)",
                                                                             "expected (p$\\geq$thr)",
                                                                             "likely (q05$\\geq$thr)"])]
        if kind == "mask":
            m = r.split(":")[1]
            hs.append(Patch(fc=ctx.color(m), label=f"{ctx.short(m)} mask"))
        if kind == "ref":
            rc = ctx.ref_cfg.get(r.split(":")[1], {})
            hs.append(Patch(fc=rc.get("color", s.reference_color), label=rc.get("label", r.split(":")[1])))
        if kind == "diff":
            a_key, b_key = diff_keys(r.split(":")[1:])
            al, bl = key_label(ctx, a_key), key_label(ctx, b_key)
            hs += [Patch(fc=c, ec="#999", lw=0.3, label=l_) for c, l_ in
                   zip(P.DIFF_COLORS[1:], ["both", f"{al} only", f"{bl} only"])]
        if kind in ("ref", "diff") and any(_two_sided(ctx, k) for k in re.findall(r"ref:([A-Za-z0-9_.-]+)", r)
                                           + ([r.split(":")[1]] if kind == "ref" else [])):
            if not any(h.get_label() == "no reference data" for h in hs):
                hs.append(Patch(fc=NODATA_COLOR, label="no reference data"))
    for o in overlays:
        lab = ctx.ref_cfg.get(o, {}).get("label", o)
        poly = isinstance(ctx.refs.get(o), D.HMSPolygons)
        hs.append(Line2D([], [], color="#00c8ff", lw=1, ls="-" if poly else "--",
                         label=f"{lab} polygons" if poly else f"{lab} (daily, quick reference)"))
    if hs:
        fig.legend(handles=hs, loc="lower right", bbox_to_anchor=(0.99, 0.0), ncol=3, fontsize=5.8, frameon=False,
                   handlelength=1.2, columnspacing=0.8, labelspacing=0.25)
    if any(r.startswith("prob:") for r in rows):
        cax = fig.add_axes([0.06, 0.22 / H, 0.35, 0.07 / H])
        sm = plt.cm.ScalarMappable(norm=P.prob_norm(s), cmap=s.prob_cmap)
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(f"P({s.label.lower()})", fontsize=6.5, labelpad=1)
        cb.ax.tick_params(labelsize=6, length=2)
    ev = ctx.frames.loc[ctx.frames.frame_id == frame_ids[0], "event"].iloc[0]
    fig.suptitle(f"{ctx.cfg.get('event_labels', {}).get(ev, ev)}: hourly evolution", fontsize=8.5,
                 y=1 - 0.08 / H)
    return P.save(fig, ctx.out, name)


def draw_label(ctx: VizContext, r: str) -> str:
    p = r.split(":")
    if p[0] == "prob":
        return f"{ctx.short(p[1])} P({ctx.spec.label.lower()})"
    if p[0] == "envelope":
        return f"{ctx.short(p[1])} extent"
    if p[0] == "mask":
        return f"{ctx.short(p[1])} mask"
    if p[0] == "unc":
        return UNC_LABELS.get(p[2], p[2])
    if p[0] == "ref":
        rc = ctx.ref_cfg.get(p[1], {})
        return rc.get("short_label", rc.get("label", p[1]))
    if p[0] == "diff":
        a, b = diff_keys(p[1:])
        return f"{key_label(ctx, a)} vs {key_label(ctx, b)}"
    return r


# ---------------------------------------------------------------------------
# Additional two-sided references (e.g. VIIRS aerosol type): evaluation-based
# skill against each reference, and overpass case maps
# ---------------------------------------------------------------------------

REF_METRICS = [("roc_auc", "ROC-AUC", True), ("pr_auc", "PR-AUC", True), ("target_macro_iou", "Smoke IoU", True),
               ("classwise_ece", "ECE (lower better)", False)]
REF_TABLE_METRICS = [k for k, _, _ in REF_METRICS] + ["target_macro_f1", "brier", "nll"]
DEFAULT_REF_LABELS = {"primary": "ADP", "viirs_aerosol_smoke": "VIIRS", "consensus_level": "ADP +\nVIIRS neg.",
                      "consensus": "All refs\nagree"}


def _ref_label(ctx: VizContext, r: str) -> str:
    lab = dict(DEFAULT_REF_LABELS)
    lab.update(ctx.cfg.get("reference_labels", {}))
    rc = ctx.ref_cfg.get(r, {})
    return lab.get(r, rc.get("short_label", rc.get("label", r)))


def reference_skill_table(ctx: VizContext, models: list[str], references: list[str]) -> pd.DataFrame:
    """Held-out skill of each model against the primary reference and each additional reference.

    Reads evaluate_uq.py outputs only (no rasters): `by_reference.<name>` for references attached with
    attach_references.py, `by_label_level.consensus` for the consensus label level. Scene counts are the
    scenes on which the reference was observed."""
    rows = []
    for m in models:
        ev = ctx.evals.get(m)
        if ev is None:
            continue
        s = ev.summary
        cov = s.get("by_label_level", {}).get("coverage", {}) or {}
        rows.append({"model": m, "reference": "primary", "n_scenes": s.get("n_scenes"),
                     **{k: ev.metric(k) for k in REF_TABLE_METRICS}})
        for r in references:
            if r == "consensus_level":
                v, n = s.get("by_label_level", {}).get("consensus"), cov.get("n_scenes_covered")
            else:
                v = s.get("by_reference", {}).get(r)
                n = (cov.get("scenes_with_reference_observed", {}) or {}).get(r)
            if not isinstance(v, dict):
                LOG.info("reference skill: %s has no '%s' results", m, r)
                continue
            rows.append({"model": m, "reference": r, "n_scenes": n, **{k: v.get(k) for k in REF_TABLE_METRICS}})
    return pd.DataFrame(rows)


def fig_reference_skill(ctx: VizContext, models: list[str], references: list[str], name="fig_reference_skill"):
    tab = reference_skill_table(ctx, models, references)
    if tab.empty or set(tab.reference) == {"primary"}:
        LOG.warning("no additional-reference results in the evaluation outputs; skipping %s", name)
        return [], tab
    refs = [r for r in ["primary"] + list(references) if r in set(tab.reference)]
    ms = [m for m in models if m in set(tab.model)]
    fig, axes = plt.subplots(1, len(REF_METRICS), figsize=(7.2, 2.7))
    w = 0.8 / len(ms)
    xl = []
    for r in refs:
        n = tab[tab.reference == r].n_scenes.dropna()
        xl.append(_ref_label(ctx, r) + (f"\n(n={int(n.iloc[0])})" if len(n) else ""))
    for ax, (k, lab, _) in zip(axes, REF_METRICS):
        for i, m in enumerate(ms):
            vals = [tab[(tab.model == m) & (tab.reference == r)][k].astype(float).mean() for r in refs]
            ax.bar(np.arange(len(refs)) + (i - (len(ms) - 1) / 2) * w, vals, w, color=ctx.color(m), label=ctx.label(m))
        ax.set_xticks(range(len(refs)), xl, fontsize=5.2)
        ax.set_title(lab, fontsize=7.5)
        ax.grid(axis="y", lw=0.3, alpha=0.6)
        if k == "roc_auc":
            ax.set_ylim(0.5, 1.0)
    h, l_ = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(h, l_, loc="upper center", ncol=len(ms), frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
    for j, a in enumerate(axes):
        P.panel_label(a, f"({chr(97 + j)})")
    return P.save(fig, ctx.out, name), tab


def pick_reference_frames(ctx: VizContext, ref_name: str, split="holdout", per_event=1, min_granules=1,
                          fs: pd.DataFrame | None = None):
    """Scans with the most granules covered by a per-scene reference raster (e.g. a VIIRS overpass)."""
    ref = ctx.refs.get(ref_name)
    if not isinstance(ref, D.RasterReference):
        return []
    fr = ctx.frames[ctx.frames.split == split].copy()
    hrs = ctx.cfg.get("case_utc_hours")
    if hrs:
        h = fr.frame_time.dt.hour
        fr = fr[(h >= hrs[0]) & (h <= hrs[1])]
    fr["has_ref"] = [bool(ref._lookup(p)) and Path(ref._lookup(p)).exists() for p in fr.pair_id]
    cnt = fr[fr.has_ref].groupby(["event", "frame_id"]).size().rename("n").reset_index()
    cnt = cnt[cnt.n >= min_granules]
    if fs is not None and f"{ref_name}_valid_area" in fs:
        # most reference smoke first, then most reference-valid area (e.g. VIIRS-clear false-alarm checks)
        sc = fs[fs.model == ctx.primary_model].set_index("frame_id")
        cnt["smoke"] = cnt.frame_id.map(sc[f"{ref_name}_area"]).fillna(0)
        cnt["valid"] = cnt.frame_id.map(sc[f"{ref_name}_valid_area"]).fillna(0)
        cnt["n"] = cnt.smoke * 10 + cnt.valid
    out = []
    for _, g in cnt.groupby("event"):
        out += list(g.sort_values("n", ascending=False).frame_id[:per_event])
    if not out:
        LOG.warning("reference %s: no %s scans with reference rasters", ref_name, split)
    return out
