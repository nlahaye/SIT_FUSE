"""
Joint TEMPO smoke + fire visuals and fire-evaluation figures.

Smoke comes from the exported geo rasters (GeoProductIndex); fire from the exported rasters when present,
otherwise from the exported fire region polygons and scene footprints (GeoPackage / parquet). Both are
matched per TEMPO granule (YYYYMMDDTHHMMSSZ_SxxxGyy), so frames are the same TEMPO scans in both products.
Fire evaluation figures read only the evaluate_uq.py outputs (eval_summary.json, per_scene_metrics.csv,
reliability_class1.csv, risk_coverage.csv, model_comparison.csv).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import data as D
from . import plotting as PL
from .products import MODEL_COLORS, MODEL_LABELS

LOG = logging.getLogger("uq_viz.joint")
GRANULE_RE = re.compile(r"(\d{8}T\d{6}Z)_S(\d{3})G(\d{2})")
KM_PER_DEG = 111.195


def granule_key(pair_id: str) -> str | None:
    m = GRANULE_RE.search(str(pair_id))
    return m.group(0) if m else None


def _frame_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    k = df.pair_id.astype(str).str.extract(GRANULE_RE)
    df["granule_key"] = df.pair_id.map(granule_key)
    df["scan"] = k[1]
    df["scene_time"] = pd.to_datetime(df.scene_time, utc=True)
    df["date"] = df.scene_time.dt.strftime("%Y-%m-%d")
    df["frame_id"] = df["date"] + "_S" + df["scan"]
    df["event"] = df.split_stratum.astype(str).str.split("|").str[0]
    return df


def _read_vec(path):
    import geopandas as gpd
    p = Path(path)
    return gpd.read_parquet(p) if p.suffix == ".parquet" else gpd.read_file(p)


class FireVectors:
    """Fire regions + scene footprints exported by export_geo_products.py (one model)."""

    def __init__(self, polygons: str, footprints: str):
        self.poly = _frame_cols(_read_vec(polygons))
        self.fp = _frame_cols(_read_vec(footprints))
        LOG.info("fire vectors: %d regions, %d scenes", len(self.poly), len(self.fp))

    def frame(self, frame_id: str):
        return self.poly[self.poly.frame_id == frame_id], self.fp[self.fp.frame_id == frame_id]

    def frame_areas(self) -> pd.DataFrame:
        cols = [c for c in ("area_discrete_km2", "area_expected_km2", "area_ensemble_lo_km2",
                            "area_ensemble_hi_km2") if c in self.fp]
        g = self.fp.groupby("frame_id")
        out = g[cols].sum()
        out["frame_time"] = g.scene_time.median()
        out["split"] = g.classifier_split.first()
        out["event"] = g.event.first()
        out["date"] = g.date.first()
        return out.reset_index()


# ---------------------------------------------------------------------------
# fire evaluation figures
# ---------------------------------------------------------------------------

def _summary(eval_dir: Path, model: str) -> dict:
    return json.loads((eval_dir / model / "eval_summary.json").read_text())


def _label(m):
    return MODEL_LABELS.get(m, m).split(" (")[0]


def fig_fire_skill(eval_dir: Path, models: list[str], out: Path, date_labels: dict, primary: str):
    mc = pd.read_csv(eval_dir / "model_comparison.csv").set_index("model")
    models = [m for m in models if m in mc.index]
    s0 = _summary(eval_dir, primary)
    prev = s0["overall"]["threshold_free"]["class_1"]["prevalence"]
    cons_prev = s0["consensus_level_detail"]["threshold_free"]["class_1"]["prevalence"]
    fig, axs = plt.subplots(1, 4, figsize=(11.5, 3.0), gridspec_kw=dict(width_ratios=[1, 1, 1.15, 1.6]))
    x = np.arange(len(models))
    cols = [MODEL_COLORS.get(m, "#666") for m in models]

    def bars(ax, key, scale=1.0, lo=None, hi=None):
        v = mc.loc[models, key].values / scale
        e = None
        if f"{key}_lo" in mc:
            e = np.vstack([v - mc.loc[models, f"{key}_lo"].values / scale,
                           mc.loc[models, f"{key}_hi"].values / scale - v]).clip(0)
        ax.bar(x, v, color=cols, yerr=e, capsize=2, error_kw=dict(lw=0.7))
        ax.set_xticks(x, [_label(m) for m in models], rotation=40, ha="right", fontsize=7)

    bars(axs[0], "roc_auc")
    axs[0].axhline(0.5, color="k", lw=0.6, ls=":")
    axs[0].set_ylim(0.4, 1.0)
    axs[0].set_title("ROC-AUC vs FDCC", fontsize=8.5)
    bars(axs[1], "pr_auc", scale=prev)
    axs[1].axhline(1.0, color="k", lw=0.6, ls=":")
    axs[1].set_yscale("log")
    axs[1].set_title("PR-AUC / prevalence (lift)", fontsize=8.5)
    w = 0.38
    axs[2].bar(x - w / 2, mc.loc[models, "target_macro_f1"], w, color=cols, label="all FDCC labels")
    axs[2].bar(x + w / 2, mc.loc[models, "consensus_level_target_macro_f1"], w, color=cols, alpha=0.45,
               hatch="///", edgecolor="#333", lw=0.3, label="VIIRS-confirmed negatives")
    axs[2].set_xticks(x, [_label(m) for m in models], rotation=40, ha="right", fontsize=7)
    axs[2].set_title("F1 at calibrated cutoff", fontsize=8.5)
    axs[2].legend(fontsize=6.5, frameon=False, loc="upper right")
    # by date, primary + two comparisons
    ax = axs[3]
    show = [m for m in (primary, "random_forest", "logistic_regression", "hard_assignment") if m in models]
    dates = sorted(s0["by_date"])
    for k, m in enumerate(show):
        bd = _summary(eval_dir, m)["by_date"]
        v = [bd[d].get("roc_auc", np.nan) for d in dates]
        ax.plot(np.arange(len(dates)), v, "o-", ms=3.5, lw=1.0, color=MODEL_COLORS.get(m), label=_label(m))
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xticks(np.arange(len(dates)), [date_labels.get(d, d[5:]) for d in dates], rotation=40, ha="right",
                  fontsize=7)
    ax.set_ylim(0.3, 1.0)
    ax.set_title("ROC-AUC by holdout day", fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=2)
    for a, l in zip(axs, "abcd"):
        a.text(-0.02, 1.08, f"({l})", transform=a.transAxes, fontsize=9, fontweight="bold", ha="right")
        a.tick_params(labelsize=7)
    fig.tight_layout()
    LOG.info("prevalence %.2e (consensus level %.2e)", prev, cons_prev)
    return PL.save(fig, out, "fig_fire_skill")


def fig_fire_calibration(eval_dir: Path, model: str, out: Path, event_labels: dict):
    d = eval_dir / model
    rel = pd.read_csv(d / "reliability_class1.csv")
    rel = rel[(rel.group == "overall") & (rel.n > 0)]
    rc = pd.read_csv(d / "risk_coverage.csv")
    ps = pd.read_csv(d / "per_scene_metrics.csv")
    fig, axs = plt.subplots(1, 4, figsize=(12.0, 3.0))
    ax = axs[0]
    lo = max(1e-6, rel.mean_predicted[rel.mean_predicted > 0].min() * 0.5)
    ax.plot([lo, 1], [lo, 1], color="k", lw=0.6, ls=":")
    ok = rel.observed_frequency > 0
    ax.loglog(rel.mean_predicted[ok], rel.observed_frequency[ok], "o-", ms=3, color=MODEL_COLORS.get(model))
    for _, r in rel[~ok].iterrows():
        ax.plot(r.mean_predicted, lo, "v", ms=3, color=MODEL_COLORS.get(model), alpha=0.5)
    ax.set_xlim(lo, 1)
    ax.set_ylim(lo * 0.8, 1)
    ax.set_xlabel("mean predicted fire probability", fontsize=7.5)
    ax.set_ylabel("observed FDCC fire frequency", fontsize=7.5)
    ax.set_title("Reliability (equal-mass bins)", fontsize=8.5)
    ax = axs[1]
    for u, c in (("predictive_entropy", "#d95f02"), ("ensemble_mutual_information", "#7570b3"),
                 ("assignment_vote_variance", "#1b9e77")):
        r = rc[rc.uncertainty == u]
        if len(r):
            ax.plot(r.coverage, r.risk * 1e3, lw=1.1, color=c, label=u.replace("_", " "))
    ax.set_xlabel("coverage (retained fraction)", fontsize=7.5)
    ax.set_ylabel("error rate of retained pixels (x1e-3)", fontsize=7.5)
    ax.set_title("Risk-coverage", fontsize=8.5)
    ax.legend(fontsize=6.3, frameon=False, loc="upper left")
    ax = axs[2]
    pos = ps[ps.c1_ref_area_km2 > 0].copy()
    ev = pos.split_stratum.str.split("|").str[0]
    evc = {"oregon_fires_2026": "#1b9e77", "la_fires_2025": "#d95f02", "park_fire_2024": "#7570b3"}
    for e, g in pos.groupby(ev):
        lo_ = (g.c1_expected_area - g.c1_conformal_lo).clip(lower=0) if "c1_conformal_lo" in g else None
        ax.scatter(g.c1_ref_area_km2, g.c1_expected_area.clip(lower=1), s=7, color=evc.get(e, "#666"),
                   label=event_labels.get(e, e), alpha=0.8, lw=0)
    mx = max(pos.c1_ref_area_km2.max(), pos.c1_expected_area.max()) * 1.5
    ax.plot([5, mx], [5, mx], color="k", lw=0.6, ls=":")
    ax.set_xlim(30, mx)
    ax.set_ylim(5, mx)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FDCC fire area per scene (km$^2$)", fontsize=7.5)
    ax.set_ylabel("expected fire area (km$^2$)", fontsize=7.5)
    ax.set_title("Scene fire area", fontsize=8.5)
    ax.legend(fontsize=6.3, frameon=False, loc="upper left")
    ax = axs[3]
    v = ps.mean_ensemble_variance
    q = pd.qcut(v.rank(method="first"), 10, labels=False)
    g = ps.groupby(q).agg(u=("mean_ensemble_variance", "median"), e=("error_rate", "mean"))
    ax.plot(g.u, g.e * 1e3, "o-", ms=3.5, color=MODEL_COLORS.get(model))
    from scipy.stats import spearmanr
    rho = spearmanr(ps.mean_ensemble_variance, ps.error_rate, nan_policy="omit").statistic
    ax.set_xscale("log")
    ax.set_xlabel("scene mean ensemble variance", fontsize=7.5)
    ax.set_ylabel("scene error rate (x1e-3)", fontsize=7.5)
    ax.set_title("Scene uncertainty vs error", fontsize=8.5)
    ax.text(0.04, 0.95, f"Spearman $\\rho$ = {rho:.2f}\n(all holdout scenes;\ndecile medians shown)",
            transform=ax.transAxes, va="top", fontsize=6.8)
    for a, l in zip(axs, "abcd"):
        a.text(-0.02, 1.08, f"({l})", transform=a.transAxes, fontsize=9, fontweight="bold", ha="right")
        a.tick_params(labelsize=7)
    fig.tight_layout()
    return PL.save(fig, out, "fig_fire_calibration_uncertainty")


def size_table(eval_dir: Path, models: list[str], n_bins: int = 4) -> pd.DataFrame:
    """Scene F1 and detection fraction by FDCC fire-area quantile bin (FDCC-positive holdout scenes)."""
    rows = []
    base = pd.read_csv(eval_dir / models[0] / "per_scene_metrics.csv")
    pos = base[base.c1_ref_area_km2 > 0]
    edges = np.quantile(pos.c1_ref_area_km2, np.linspace(0, 1, n_bins + 1))
    for m in models:
        f = eval_dir / m / "per_scene_metrics.csv"
        if not f.exists():
            continue
        ps = pd.read_csv(f)
        ps = ps[ps.c1_ref_area_km2 > 0].copy()
        ps["bin"] = np.clip(np.searchsorted(edges, ps.c1_ref_area_km2, side="right") - 1, 0, n_bins - 1)
        ps["event"] = ps.split_stratum.str.split("|").str[0]
        for b, g in ps.groupby("bin"):
            rows.append(dict(model=m, bin=int(b), lo=edges[b], hi=edges[b + 1], n=len(g),
                             median_area=g.c1_ref_area_km2.median(), mean_f1=g.target_macro_f1.mean(),
                             detected=(g.target_macro_f1 > 0).mean(),
                             la_share=(g.event == "la_fires_2025").mean()))
    return pd.DataFrame(rows)


def fig_fire_size(eval_dir: Path, models: list[str], out: Path, event_labels: dict, pixel_km2: float):
    tab = size_table(eval_dir, models)
    tab.to_csv(out / "fire_size_detection.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0))
    nb = tab.bin.nunique()
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        t = tab[tab.model == m]
        x = t.bin.values + (k - (len(models) - 1) / 2) * w
        axs[0].bar(x, t.mean_f1, w, color=MODEL_COLORS.get(m), label=_label(m))
        axs[1].bar(x, t.detected, w, color=MODEL_COLORS.get(m))
    t0 = tab[tab.model == models[0]]
    lbl = [f"{a:.0f}\n({a / pixel_km2:.0f} px)" for a in t0.median_area]
    for a in axs[:2]:
        a.set_xticks(np.arange(nb), lbl, fontsize=7)
        a.set_xlabel("median FDCC fire area per scene, km$^2$ (TEMPO pixels)", fontsize=7.5)
    axs[0].set_ylabel("mean scene F1", fontsize=7.5)
    axs[0].set_title("Scene F1 by fire-size quartile", fontsize=8.5)
    axs[0].legend(fontsize=6.3, frameon=False, loc="upper left")
    axs[1].set_ylabel("fraction of scenes with any overlap", fontsize=7.5)
    axs[1].set_title("Detection by fire-size quartile", fontsize=8.5)
    ps = pd.read_csv(eval_dir / models[0] / "per_scene_metrics.csv")
    ps = ps[ps.c1_ref_area_km2 > 0]
    ev = ps.split_stratum.str.split("|").str[0]
    evc = {"oregon_fires_2026": "#1b9e77", "la_fires_2025": "#d95f02", "park_fire_2024": "#7570b3"}
    bins = np.geomspace(max(1, ps.c1_ref_area_km2.min()), ps.c1_ref_area_km2.max() * 1.05, 18)
    for e, g in ps.groupby(ev):
        axs[2].hist(g.c1_ref_area_km2, bins=bins, histtype="step", lw=1.3, color=evc.get(e, "#666"),
                    label=f"{event_labels.get(e, e)} (n={len(g)})")
    axs[2].axvline(pixel_km2, color="k", lw=0.7, ls=":")
    axs[2].text(pixel_km2 * 1.08, axs[2].get_ylim()[1] * 0.92, "1 TEMPO\npixel", fontsize=6.5, va="top")
    axs[2].set_xscale("log")
    axs[2].set_xlabel("FDCC fire area per scene (km$^2$)", fontsize=7.5)
    axs[2].set_ylabel("holdout scenes", fontsize=7.5)
    axs[2].set_title("Fire size by event (holdout)", fontsize=8.5)
    axs[2].legend(fontsize=6.3, frameon=False, loc="upper right")
    for a, l in zip(axs, "abc"):
        a.text(-0.02, 1.08, f"({l})", transform=a.transAxes, fontsize=9, fontweight="bold", ha="right")
        a.tick_params(labelsize=7)
    fig.tight_layout()
    return PL.save(fig, out, "fig_fire_size_detection")


# ---------------------------------------------------------------------------
# joint smoke + fire maps
# ---------------------------------------------------------------------------

class JointContext:
    def __init__(self, smoke_index: D.GeoProductIndex, smoke_model: str, fire: FireVectors,
                 fire_comp: FireVectors | None, adp: D.RasterReference | None, basemap, cfg: dict):
        self.idx, self.sm, self.fire, self.fire_comp, self.adp = smoke_index, smoke_model, fire, fire_comp, adp
        self.basemap, self.cfg = basemap, cfg
        sc = smoke_index.scenes(smoke_model).copy()
        sc["granule_key"] = sc.pair_id.map(granule_key)
        self.frames = D.assign_frames(sc, "scan")
        t = self.idx.df.loc[self.idx.df.model == smoke_model, "threshold"]
        self.smoke_thr = float(t.iloc[0]) if len(t) and np.isfinite(t.iloc[0]) else 0.5
        self.fire_thr = float(cfg.get("fire_threshold") or np.nan)
        h, w = self.idx.shape
        tr = self.idx.transform
        lat = tr.f + (np.arange(h) + 0.5) * tr.e
        a = (abs(tr.a) * KM_PER_DEG) * (abs(tr.e) * KM_PER_DEG) * np.cos(np.deg2rad(lat))
        self.pix_area = np.repeat(a[:, None], w, axis=1)

    def pairs(self, fid):
        return list(self.frames.loc[self.frames.frame_id == fid, "pair_id"])

    def smoke(self, fid, band="p_1"):
        return D.mosaic(self.idx, self.sm, self.pairs(fid), band, True)

    def adp_frame(self, fid):
        if self.adp is None:
            return None
        return D.mosaic_reference(self.adp, self.idx, self.pairs(fid), self.sm, "p_1")

    def frame_time(self, fid):
        f = self.frames.loc[self.frames.frame_id == fid, "frame_time"]
        if len(f):
            return f.iloc[0]
        return self.fire.fp.loc[self.fire.fp.frame_id == fid, "scene_time"].median()

    def extent(self, event):
        return self.cfg.get("fire_extents", {}).get(event) or self.cfg.get("event_extents", {}).get(event) \
            or self.idx.extent()

    def smoke_frame_areas(self) -> pd.DataFrame:
        fp = pd.read_parquet(self.cfg["smoke_footprints"]) if self.cfg.get("smoke_footprints") else None
        if fp is None:
            return pd.DataFrame()
        fp = _frame_cols(fp)
        g = fp.groupby("frame_id")
        out = g[["area_discrete_km2", "area_expected_km2"]].sum()
        for c in ("area_ensemble_lo_km2", "area_ensemble_hi_km2"):
            if c in fp:
                out[c] = g[c].sum()
        out["frame_time"] = g.scene_time.median()
        out["date"] = g.date.first()
        out["split"] = g.classifier_split.first()
        return out.reset_index()


FIRE_NORM = LogNorm(vmin=1e-2, vmax=0.5)
FIRE_CMAP = "YlOrRd"
UNC_CMAP = "cividis"


def draw_joint(ax, ctx: JointContext, fid: str, event: str, layer: str = "smoke", legend=True, ticks=True):
    """layer: smoke (p_1 + ADP contour + fire regions) | uncertainty (smoke entropy + fire outlines)."""
    ext = ctx.extent(event)
    PL.setup_map(ax, ext, ctx.basemap, ticks)
    if ticks:
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(1 if ext[1] - ext[0] <= 10 else 2))
        ax.yaxis.set_major_locator(MultipleLocator(1))
    gext = ctx.idx.extent()
    poly, fp = ctx.fire.frame(fid)
    if len(fp):
        fp.plot(ax=ax, facecolor="white", edgecolor="none", zorder=1)
    if layer == "smoke":
        p = ctx.smoke(fid)
        im = None
        if p is not None:
            im = PL.show(ax, p, gext, "Blues", Normalize(0, 1), zorder=2)
        a = ctx.adp_frame(fid)
        if a is not None and np.isfinite(a).any():
            PL.contour(ax, np.nan_to_num(a) >= 0.5, ctx.idx, color="#33a02c", lw=0.8, ls="-")
    else:
        u = ctx.smoke(fid, "predictive_entropy")
        im = PL.show(ax, u, gext, UNC_CMAP, Normalize(0, 1), zorder=2) if u is not None else None
    if len(poly):
        if layer == "smoke":
            poly.plot(ax=ax, column="max_p_1", cmap=FIRE_CMAP, norm=FIRE_NORM, edgecolor="#7f0000", linewidth=0.5,
                      zorder=7)
        else:
            poly.boundary.plot(ax=ax, color="#ff2a2a", linewidth=0.9, zorder=7)
    if ctx.fire_comp is not None:
        cp, _ = ctx.fire_comp.frame(fid)
        if len(cp):
            cp.boundary.plot(ax=ax, color="#6a3d9a", linewidth=0.5, linestyle="--", zorder=8)
    if len(fp):
        fp.dissolve().boundary.plot(ax=ax, color="#888888", linewidth=0.4, zorder=6)
    for mk in ctx.cfg.get("markers", {}).get(event, []):
        ax.plot(mk["lon"], mk["lat"], marker="*", ms=8, mfc="white", mec="k", mew=0.6, zorder=9)
        ax.annotate(mk.get("name", ""), (mk["lon"], mk["lat"]), xytext=tuple(mk.get("offset", (4, 4))), textcoords="offset points",
                    fontsize=6.5, zorder=9, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7))
    return im


def _legend(ax, with_comp=True):
    h = [Patch(facecolor="#6baed6", label="smoke probability"),
         Line2D([], [], color="#33a02c", lw=1.0, label="ADP smoke"),
         Patch(facecolor="#fd8d3c", edgecolor="#7f0000", label="fire region (HGB)")]
    if with_comp:
        h.append(Line2D([], [], color="#6a3d9a", lw=0.8, ls="--", label="fire region (hard assignment)"))
    h.append(Line2D([], [], color="#888888", lw=0.6, label="TEMPO scene edge"))
    h.append(Line2D([], [], marker="*", ls="", mfc="white", mec="k", ms=7, label="incident location"))
    ax.legend(handles=h, fontsize=6.3, loc="lower left", framealpha=0.85)


def fig_joint_cases(ctx: JointContext, cases: list[dict], out: Path):
    n = len(cases)
    fig, axs = plt.subplots(n, 2, figsize=(8.6, 3.9 * n), squeeze=False)
    ims = []
    for i, c in enumerate(cases):
        fid, ev = c["frame"], c["event"]
        t = ctx.frame_time(fid)
        sp = ctx.fire.fp.loc[ctx.fire.fp.frame_id == fid, "classifier_split"]
        sp = sp.iloc[0] if len(sp) else ""
        im1 = draw_joint(axs[i, 0], ctx, fid, ev, "smoke")
        im2 = draw_joint(axs[i, 1], ctx, fid, ev, "uncertainty")
        ims.append((im1, im2))
        for j, s in enumerate(("smoke probability + fire regions", "smoke predictive entropy + fire outlines")):
            PL.panel_label(axs[i, j], f"({'abcdefgh'[2 * i + j]}) {c.get('label', ev)} {t:%Y-%m-%d %H:%M} UTC"
                                      f" [{sp}]\n{s}")
        _legend(axs[i, 0], ctx.fire_comp is not None)
    fig.tight_layout(rect=(0, 0.03 + 0.03 / n, 1, 1))
    yb, hb = 0.012 + 0.015 / n, 0.012 / n * 2
    cax = fig.add_axes([0.07, yb, 0.26, hb])
    fig.colorbar(plt.cm.ScalarMappable(Normalize(0, 1), "Blues"), cax=cax, orientation="horizontal") \
        .set_label("smoke probability", fontsize=7)
    cax = fig.add_axes([0.39, yb, 0.26, hb])
    fig.colorbar(plt.cm.ScalarMappable(FIRE_NORM, FIRE_CMAP), cax=cax, orientation="horizontal") \
        .set_label("peak fire probability in region", fontsize=7)
    cax = fig.add_axes([0.71, yb, 0.26, hb])
    fig.colorbar(plt.cm.ScalarMappable(Normalize(0, 1), UNC_CMAP), cax=cax, orientation="horizontal") \
        .set_label("smoke predictive entropy", fontsize=7)
    for cx in fig.axes[-3:]:
        cx.tick_params(labelsize=6.5)
    return PL.save(fig, out, c.get("name", "fig_joint_cases") if n == 1 else "fig_joint_cases")


def joint_series(ctx: JointContext, date: str, smoke_eval: Path | None, fire_eval: Path | None,
                 smoke_model: str, fire_model: str, splits=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-scan smoke and fire areas for one UTC day; `splits` keeps only scans of those partitions
    (fire partition decides, so both series cover the same scans)."""
    sa = ctx.smoke_frame_areas()
    sa = sa[sa.date == date].copy() if len(sa) else sa
    fa = ctx.fire.frame_areas()
    fa = fa[fa.date == date].copy()
    if splits:
        fa = fa[fa.split.isin(splits)]
        if len(sa):
            sa = sa[sa.frame_id.isin(fa.frame_id)]
    for d, ev, model, col in ((sa, smoke_eval, smoke_model, "smoke_ref_km2"), (fa, fire_eval, fire_model,
                                                                               "fire_ref_km2")):
        if ev is None or not len(d):
            continue
        f = Path(ev) / model / "per_scene_metrics.csv"
        if not f.exists():
            continue
        ps = _frame_cols(pd.read_csv(f).assign(scene_time=lambda x: x.pair_id.str.extract(r"(\d{8}T\d{6})Z")[0]
                                               .pipe(pd.to_datetime, utc=True)))
        r = ps.groupby("frame_id").c1_ref_area_km2.sum()
        d[col] = d.frame_id.map(r)
    return sa.sort_values("frame_time"), fa.sort_values("frame_time")


def _break_gaps(d: pd.DataFrame, hours: float = 2.0) -> pd.DataFrame:
    """Insert NaN rows where consecutive scans are more than `hours` apart (no lines across the night)."""
    if len(d) < 2:
        return d
    d = d.sort_values("frame_time").reset_index(drop=True)
    gap = d.frame_time.diff() > pd.Timedelta(hours=hours)
    if not gap.any():
        return d
    rows = []
    for i, r in d.iterrows():
        if gap.iloc[i]:
            rows.append({"frame_time": d.frame_time.iloc[i - 1] + (r.frame_time - d.frame_time.iloc[i - 1]) / 2})
        rows.append(r.to_dict())
    return pd.DataFrame(rows)


def _series_axes(ax, sa, fa, t_now=None, fs=7):
    sa, fa = _break_gaps(sa), _break_gaps(fa)
    if len(sa):
        ax.fill_between(sa.frame_time, sa.get("area_ensemble_lo_km2", sa.area_expected_km2) / 1e3,
                        sa.get("area_ensemble_hi_km2", sa.area_expected_km2) / 1e3, color="#9ecae1", alpha=0.6, lw=0, label="smoke ensemble range")
        ax.plot(sa.frame_time, sa.area_expected_km2 / 1e3, color="#2171b5", lw=1.2, label="smoke expected")
        ax.plot(sa.frame_time, sa.area_discrete_km2 / 1e3, color="#2171b5", lw=0.8, ls="--", label="smoke mask")
        if "smoke_ref_km2" in sa:
            ax.plot(sa.frame_time, sa.smoke_ref_km2 / 1e3, "s", ms=3, color="#33a02c", label="ADP smoke")
    ax.set_ylabel("smoke area (10$^3$ km$^2$)", fontsize=fs, color="#2171b5")
    a2 = ax.twinx()
    if len(fa):
        if "area_ensemble_lo_km2" in fa:
            a2.fill_between(fa.frame_time, fa.area_ensemble_lo_km2, fa.area_ensemble_hi_km2, color="#fdae6b",
                            alpha=0.45, lw=0, label="fire ensemble range")
        a2.plot(fa.frame_time, fa.area_expected_km2, color="#d94801", lw=1.2, label="fire expected")
        a2.plot(fa.frame_time, fa.area_discrete_km2, color="#d94801", lw=0.8, ls="--", label="fire mask")
        if "fire_ref_km2" in fa:
            a2.plot(fa.frame_time, fa.fire_ref_km2, "^", ms=3.5, color="#7f0000", label="FDCC fire")
    a2.set_ylabel("fire area (km$^2$)", fontsize=fs, color="#d94801")
    ax.set_zorder(a2.get_zorder() + 1)
    ax.patch.set_visible(False)
    a2.tick_params(labelsize=fs - 0.5)
    ax.tick_params(labelsize=fs - 0.5)
    if t_now is not None:
        ax.axvline(t_now, color="k", lw=0.8)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="UTC"))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    return a2, h1 + h2, l1 + l2


def fig_joint_timeseries(ctx: JointContext, days: list[dict], smoke_eval, fire_eval, smoke_model, fire_model,
                         out: Path):
    fig, axs = plt.subplots(1, len(days), figsize=(5.2 * len(days), 2.9), squeeze=False)
    tabs = []
    for k, d in enumerate(days):
        sa, fa = joint_series(ctx, d["date"], smoke_eval, fire_eval, smoke_model, fire_model, d.get("splits"))
        tabs.append(sa.assign(kind="smoke", day=d["date"]))
        tabs.append(fa.assign(kind="fire", day=d["date"]))
        ax = axs[0, k]
        _, h, l = _series_axes(ax, sa, fa)
        ax.set_title(f"({'abcd'[k]}) {d.get('label', d['date'])}: hourly smoke and fire area", fontsize=8.5)
        ax.set_xlabel("UTC", fontsize=7)
        if k == 0:
            hl = (h, l)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(*hl, fontsize=6.8, frameon=False, loc="lower center", ncol=8)
    pd.concat(tabs).to_csv(out / "joint_area_series.csv", index=False)
    return PL.save(fig, out, "fig_joint_timeseries")


def animate_joint(ctx: JointContext, date: str, label: str, event: str, smoke_eval, fire_eval, smoke_model,
                  fire_model, out: Path, fps: float = 1.5, gif: bool = True, dpi: int = 110, utc_hours=None,
                  splits=None):
    import subprocess
    import tempfile
    sa, fa = joint_series(ctx, date, smoke_eval, fire_eval, smoke_model, fire_model, splits)
    if utc_hours:
        h0, h1 = utc_hours
        keep = lambda d: d[(d.frame_time.dt.hour >= h0) & (d.frame_time.dt.hour < h1)]
        sa, fa = (keep(sa) if len(sa) else sa), keep(fa)
    fids = sorted(set(fa.frame_id) | (set(sa.frame_id) if len(sa) else set()))
    if not fids:
        LOG.warning("no frames for %s", date)
        return []
    tmp = Path(tempfile.mkdtemp(prefix="joint_anim_"))
    for i, fid in enumerate(fids):
        fig = plt.figure(figsize=(9.0, 6.6))
        gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.0], hspace=0.42, wspace=0.12, top=0.93, bottom=0.07,
                              left=0.08, right=0.92)
        a0, a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
        draw_joint(a0, ctx, fid, event, "smoke")
        draw_joint(a1, ctx, fid, event, "uncertainty", ticks=False)
        t = ctx.frame_time(fid)
        sp = ctx.fire.fp.loc[ctx.fire.fp.frame_id == fid, "classifier_split"]
        sp = sp.iloc[0] if len(sp) else ""
        PL.panel_label(a0, "smoke probability + fire regions")
        PL.panel_label(a1, "smoke entropy + fire outlines")
        _legend(a0, ctx.fire_comp is not None)
        for ax_, nm, cm, lb in ((a0, Normalize(0, 1), "Blues", "smoke probability"),
                                (a1, Normalize(0, 1), UNC_CMAP, "smoke predictive entropy")):
            cb = fig.colorbar(plt.cm.ScalarMappable(nm, cm), ax=ax_, orientation="horizontal", fraction=0.035,
                              pad=0.1, aspect=40)
            cb.set_label(lb, fontsize=6.5)
            cb.ax.tick_params(labelsize=6)
        _, h, l = _series_axes(a2, sa, fa, t_now=t, fs=6.5)
        a2.legend(h, l, fontsize=5.8, frameon=False, loc="upper left", ncol=4)
        fig.suptitle(f"{label}  {t:%Y-%m-%d %H:%M} UTC  scan {fid.split('_S')[-1]}  [{sp}]", fontsize=9.5,
                     y=0.985)
        fig.savefig(tmp / f"f{i:03d}.png", dpi=dpi)
        plt.close(fig)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"anim_joint_{date}"
    vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "f%03d.png"),
                    "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", f"{stem}.mp4"], check=True)
    paths = [Path(f"{stem}.mp4")]
    if gif:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                        str(tmp / "f%03d.png"), "-vf", "scale=720:-1:flags=lanczos,split[a][b];[a]palettegen[p];"
                        "[b][p]paletteuse", f"{stem}.gif"], check=True)
        paths.append(Path(f"{stem}.gif"))
    keep = Path(f"{stem}_frames")
    keep.mkdir(exist_ok=True)
    for f in sorted(tmp.glob("*.png")):
        f.replace(keep / f.name)
    return paths
