"""
Figures for the fused GOES + TEMPO product written by fuse_multiscale_products.py.

    python make_fusion_visuals.py -c fusion_visuals.yaml [--only skill reliability timeseries cases]

Reads, from the fusion output_dir (taken from the fusion config unless fusion_dir is given):
  fusion_evaluation.csv, fusion_reliability.csv, fused_frames.csv, fusion_model_<grid>.json,
  <grid>/fused/rasters/fused_<product>_<grid>_<time>Z.tif

Figures (output_dir):
  fusion_skill_<grid>          metric bars, fused vs single-source and rule baselines, per reference (scope: all)
  fusion_skill_pattern_<grid>  same, split by availability pattern (both / tempo_only / goes_only)
  fusion_reliability_<grid>    reliability curves per reference and method
  fusion_timeseries_<grid>_<date>   expected area with 5-95 % band, discrete area, availability fractions
  fusion_case_<grid>_<time>    TEMPO p, GOES p, fused p, fused 90 % width, n_sources, reference(s), agreement
plus fusion_skill_summary.csv (fused minus best baseline per grid / reference / scope / metric).

Config keys (see fusion_visuals.example.yaml):
  fusion_config, fusion_dir (optional), output_dir, cache_dir, basemap.local, formats
  grids, references, methods, metrics, scopes
  timeseries: {splits: [holdout], dates: [...] (default: all dates in those splits)}
  cases: {frames: [ISO times] or auto: {n: 6, splits: [holdout], by: area_expected_km2}, grid, extent: [W,E,S,N]}
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize

from uq_viz import data as D
from uq_viz import plotting as P
import fuse_multiscale_products as F

LOG = logging.getLogger("make_fusion_visuals")

METHOD_COLORS = {"fused": "#d95f02", "tempo": "#1b9e77", "goes": "#7570b3", "noisy_or": "#e7298a",
                 "max": "#66a61e", "mean": "#a6761d"}
METHOD_LABELS = {"fused": "fused (log-linear)", "tempo": "TEMPO only", "goes": "GOES only",
                 "noisy_or": "noisy-OR", "max": "max", "mean": "mean"}
LOWER_BETTER = {"nll", "brier", "ece"}


def _csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:     # production runs (no references) write empty evaluation tables
        return pd.DataFrame()


def _methods(df, want):
    have = list(dict.fromkeys(df.method))
    return [m for m in (want or have) if m in have]


def fig_skill(ev: pd.DataFrame, grid: str, refs, methods, metrics, out, fmt, scope="all"):
    d = ev[(ev.grid == grid) & (ev.scope == scope)]
    refs = [r for r in refs if r in set(d.reference)]
    if d.empty or not refs:
        return
    fig, axes = plt.subplots(len(refs), len(metrics), figsize=(1.9 * len(metrics), 1.9 * len(refs)), squeeze=False)
    for i, rn in enumerate(refs):
        dr = d[d.reference == rn].set_index("method")
        for j, mt in enumerate(metrics):
            ax = axes[i, j]
            ms = [m for m in methods if m in dr.index]
            v = [float(dr.loc[m, mt]) for m in ms]
            ax.bar(range(len(ms)), v, color=[METHOD_COLORS.get(m, "0.5") for m in ms])
            ax.set_xticks(range(len(ms)))
            ax.set_xticklabels(ms, rotation=60, ha="right", fontsize=6)
            ax.set_title(f"{mt}{' (lower better)' if mt in LOWER_BETTER else ''}" if i == 0 else "")
            if j == 0:
                ax.set_ylabel(rn)
    fig.suptitle(f"Fused vs baselines, grid {grid}, scope {scope}", y=1.01)
    fig.tight_layout()
    P.save(fig, out, f"fusion_skill_{grid}", fmt)


def fig_skill_pattern(ev, grid, refs, methods, metrics, out, fmt, scopes=("both", "tempo_only", "goes_only")):
    d = ev[(ev.grid == grid) & ev.scope.isin(scopes)]
    scopes = [s for s in scopes if s in set(d.scope)]
    refs = [r for r in refs if r in set(d.reference)]
    if d.empty or not refs:
        return
    fig, axes = plt.subplots(len(refs), len(metrics), figsize=(2.3 * len(metrics), 1.9 * len(refs)), squeeze=False)
    w = 0.8 / max(len(methods), 1)
    for i, rn in enumerate(refs):
        dr = d[d.reference == rn]
        for j, mt in enumerate(metrics):
            ax = axes[i, j]
            for k, m in enumerate(methods):
                dm = dr[dr.method == m].set_index("scope")
                v = [float(dm.loc[s, mt]) if s in dm.index else np.nan for s in scopes]
                ax.bar(np.arange(len(scopes)) + (k - (len(methods) - 1) / 2) * w, v, w,
                       color=METHOD_COLORS.get(m, "0.5"), label=METHOD_LABELS.get(m, m))
            ax.set_xticks(range(len(scopes)))
            ax.set_xticklabels([{"both": "both", "tempo_only": "TEMPO", "goes_only": "GOES"}.get(s, s)
                                for s in scopes], fontsize=6.5)
            ax.set_title(mt if i == 0 else "")
            if j == 0:
                ax.set_ylabel(rn)
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle(f"Skill by availability pattern, grid {grid}", y=1.01)
    fig.tight_layout()
    P.save(fig, out, f"fusion_skill_pattern_{grid}", fmt)


def skill_summary(ev: pd.DataFrame, metrics) -> pd.DataFrame:
    rows = []
    for (g, rn, sc), d in ev.groupby(["grid", "reference", "scope"]):
        d = d.set_index("method")
        if "fused" not in d.index:
            continue
        base = d.drop(index="fused")
        for mt in metrics:
            if mt not in d or base.empty:
                continue
            b = base[mt].min() if mt in LOWER_BETTER else base[mt].max()
            bm = base[mt].idxmin() if mt in LOWER_BETTER else base[mt].idxmax()
            f = float(d.loc["fused", mt])
            gain = (b - f) if mt in LOWER_BETTER else (f - b)
            rows.append({"grid": g, "reference": rn, "scope": sc, "metric": mt, "fused": f, "best_baseline": bm,
                         "best_baseline_value": float(b), "fused_gain": float(gain), "n": int(d.loc["fused", "n"])})
    return pd.DataFrame(rows)


def fig_reliability(rel: pd.DataFrame, grid, refs, methods, out, fmt, scope="all"):
    d = rel[(rel.grid == grid) & (rel.scope == scope)]
    refs = [r for r in refs if r in set(d.reference)]
    if d.empty or not refs:
        return
    fig, axes = plt.subplots(1, len(refs), figsize=(2.8 * len(refs), 2.8), squeeze=False)
    for i, rn in enumerate(refs):
        ax = axes[0, i]
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
        for m in methods:
            dm = d[(d.reference == rn) & (d.method == m)].sort_values("bin")
            if dm.empty:
                continue
            ax.plot(dm.mean_p, dm.obs_freq, marker="o", ms=2.5, lw=1.4 if m == "fused" else 0.9,
                    color=METHOD_COLORS.get(m, "0.5"), label=METHOD_LABELS.get(m, m))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed frequency" if i == 0 else "")
        ax.set_title(f"{rn}: reliability, grid {grid}, scope {scope}", fontsize=7.5)
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    P.save(fig, out, f"fusion_reliability_{grid}", fmt)


def fig_timeseries(fr: pd.DataFrame, grid, splits, dates, out, fmt):
    d = fr[(fr.grid == grid) & fr.split.isin(splits)].copy()
    if d.empty:
        return
    d["t"] = pd.to_datetime(d.scene_time, utc=True)
    d["date"] = d.t.dt.strftime("%Y-%m-%d")
    for day in (dates or sorted(d.date.unique())):
        dd = d[d.date == str(day)].sort_values("t")
        if dd.empty:
            continue
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.2, 3.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
        a1.fill_between(dd.t, dd.area_q05_km2, dd.area_q95_km2, color="#fdae6b", alpha=0.5, lw=0,
                        label="5-95 % ensemble")
        a1.plot(dd.t, dd.area_expected_km2, color="#d95f02", lw=1.4, label="expected area")
        a1.plot(dd.t, dd.area_discrete_km2, color="0.25", lw=0.9, ls="--", label="thresholded area")
        a1.set_ylabel("area (km²)")
        a1.legend(frameon=False, ncol=3, loc="upper left")
        a1.set_title(f"Fused product, grid {grid}, {day} ({', '.join(sorted(set(dd.split)))})")
        a2.stackplot(dd.t, dd.frac_both, dd.frac_tempo_only, dd.frac_goes_only,
                     colors=["#d95f02", "#1b9e77", "#7570b3"], labels=["both", "TEMPO only", "GOES only"])
        a2.set_ylim(0, 1)
        a2.set_ylabel("grid fraction")
        a2.legend(frameon=False, ncol=3, loc="upper left", fontsize=6)
        a2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
        a2.set_xlabel("UTC")
        fig.tight_layout()
        P.save(fig, out, f"fusion_timeseries_{grid}_{day}", fmt)


def _bands(path):
    import rasterio
    with rasterio.open(path) as s:
        desc = list(s.descriptions)
        arr = {n: s.read(i + 1).astype(np.float32) for i, n in enumerate(desc) if n}
        b = s.bounds
        tags = s.tags()
    if "n_sources" in arr:
        arr["n_sources"][arr["n_sources"] == 0] = np.nan
    return arr, [b.left, b.right, b.bottom, b.top], tags


def pick_cases(fr: pd.DataFrame, grid, cc: dict) -> list[pd.Timestamp]:
    if cc.get("frames"):
        return [pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo else pd.Timestamp(t, tz="UTC")
                for t in cc["frames"]]
    au = cc.get("auto", {"n": 4})
    d = fr[(fr.grid == grid) & fr.split.isin(au.get("splits", ["holdout"]))].copy()
    if d.empty:
        return []
    d["t"] = pd.to_datetime(d.scene_time, utc=True)
    d["date"] = d.t.dt.date
    by = au.get("by", "area_expected_km2")
    # strongest frame per day, then the top n days, so cases span different events
    best = d.sort_values(by, ascending=False).groupby("date").head(1).head(int(au.get("n", 4)))
    return sorted(best.t)


def fig_cases(fdir: Path, product, grid, times, refs, thr_default, basemap, extent, out, fmt):
    idx = D.GeoProductIndex(fdir / grid, product, ["fused"])
    prob = ListedColormap(plt.get_cmap("YlOrRd")(np.linspace(0.05, 1, 256)))
    agree_cm, agree_nm = P.diff_cmap()
    nsrc_cm, nsrc_nm = ListedColormap(["#7fc97f", "#d95f02"]), BoundaryNorm([0.5, 1.5, 2.5], 2)
    for t in times:
        path = fdir / grid / "fused" / "rasters" / f"fused_{product}_{grid}_{t:%Y%m%dT%H%M%S}Z.tif"
        if not path.exists():
            LOG.warning("no fused raster %s", path.name)
            continue
        a, ext, tags = _bands(path)
        thr = float(tags.get("threshold", thr_default))
        ex = extent or ext
        width = a["ensemble_p1_q95"] - a["ensemble_p1_q05"] if "ensemble_p1_q95" in a else None
        panels = [("TEMPO p (on grid)", a.get("tempo_p1"), prob, Normalize(0, 1)),
                  ("GOES p (on grid)", a.get("goes_p1"), prob, Normalize(0, 1)),
                  ("fused p", a["p_1"], prob, Normalize(0, 1)),
                  ("fused 90 % width (q95 - q05)", width, plt.get_cmap("viridis"), Normalize(0, 0.5)),
                  ("sources (1 or 2)", a.get("n_sources"), nsrc_cm, nsrc_nm)]
        fused_mask = np.where(np.isfinite(a["p_1"]), (a["p_1"] >= thr).astype(np.float32), np.nan)
        for rn, ref in refs.items():
            y = F.ref_on_grid(ref, t, idx)
            if y is None:
                continue
            panels.append((f"{rn} reference", y, ListedColormap(["#ffffff", "#525252"]),
                           BoundaryNorm([-0.5, 0.5, 1.5], 2)))
            panels.append((f"fused (p >= {thr:.2f}) vs {rn}", P.difference(fused_mask, y), agree_cm, agree_nm))
        panels = [p for p in panels if p[1] is not None]
        nc = min(4, len(panels))
        nr = int(np.ceil(len(panels) / nc))
        aspect = (ex[3] - ex[2]) / max(ex[1] - ex[0], 1e-6) / np.cos(np.deg2rad(0.5 * (ex[2] + ex[3])))
        pw = 2.6
        fig, axes = plt.subplots(nr, nc, figsize=(pw * nc, nr * (0.62 * pw * aspect + 0.6)), squeeze=False)
        for k, (title, arr, cm, nm) in enumerate(panels):
            ax = axes.flat[k]
            P.setup_map(ax, ex, basemap)
            dec = 0 if max(ex[1] - ex[0], ex[3] - ex[2]) >= 8 else 1
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, d=dec: f"{abs(v):.{d}f}°{'W' if v < 0 else 'E'}"))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, d=dec: f"{v:.{d}f}°N"))
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            im = P.show(ax, arr, ext, cm, nm)
            ax.set_title(title)
            P.panel_label(ax, "abcdefghijkl"[k])
            if "vs" in title:
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=[0, 1, 2, 3])
                cb.ax.set_yticklabels(["neither", "both", "fused only", "ref only"], fontsize=6)
            elif "sources" in title:
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=[1, 2])
            elif "reference" not in title:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        fig.suptitle(f"Fused {product}, grid {grid}, {t:%Y-%m-%d %H:%M} UTC, split {tags.get('classifier_split', '?')}",
                     y=1.0)
        fig.tight_layout()
        P.save(fig, out, f"fusion_case_{grid}_{t:%Y%m%dT%H%M}", fmt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="subset of: skill reliability timeseries cases")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(Path(a.config).read_text())
    fcfg = yaml.safe_load(Path(cfg["fusion_config"]).read_text())
    fdir = Path(cfg.get("fusion_dir") or fcfg["output_dir"])
    product = fcfg.get("product", "smoke")
    out = Path(cfg.get("output_dir") or fdir / "visuals")
    out.mkdir(parents=True, exist_ok=True)
    fmt = tuple(cfg.get("formats", ["png", "pdf"]))
    want = set(a.only or ["skill", "reliability", "timeseries", "cases"])
    grids = cfg.get("grids") or list((fcfg.get("grids") or {"fine": {}}).keys())
    metrics = cfg.get("metrics", ["pr_auc", "f1", "iou", "brier", "ece"])

    ev = _csv(fdir / "fusion_evaluation.csv")
    refs_all = cfg.get("references") or (list(dict.fromkeys(ev.reference)) if len(ev) else [])
    methods = _methods(ev, cfg.get("methods")) if len(ev) else []
    if len(ev):
        skill_summary(ev, metrics).to_csv(out / "fusion_skill_summary.csv", index=False)
    for g in grids:
        mj = fdir / f"fusion_model_{g}.json"
        if mj.exists():
            LOG.info("grid %s model: %s", g, json.dumps(json.loads(mj.read_text()))[:400])
        if "skill" in want and len(ev):
            fig_skill(ev, g, refs_all, methods, metrics, out, fmt)
            fig_skill_pattern(ev, g, refs_all, methods, cfg.get("pattern_metrics", ["pr_auc", "brier"]), out, fmt,
                              tuple(cfg.get("scopes", ["both", "tempo_only", "goes_only"])))
        rel = _csv(fdir / "fusion_reliability.csv")
        if "reliability" in want and len(rel):
            fig_reliability(rel, g, refs_all, methods, out, fmt)
    fr = _csv(fdir / "fused_frames.csv")
    if len(fr):
        LOG.info("fused frames by split:\n%s", fr.groupby(["grid", "split"]).size().to_string())
    ts = cfg.get("timeseries", {})
    if "timeseries" in want and len(fr):
        for g in grids:
            fig_timeseries(fr, g, ts.get("splits", ["holdout"]), ts.get("dates"), out, fmt)
    cc = cfg.get("cases", {})
    if "cases" in want and len(fr):
        g = cc.get("grid", grids[0])
        times = pick_cases(fr, g, cc)
        refs = {k: v for k, v in F.load_references(fcfg).items() if not cfg.get("references") or k in refs_all}
        idx = D.GeoProductIndex(fdir / g, product, ["fused"])
        bm = cfg.get("basemap", {})
        basemap = D.load_basemap(Path(cfg.get("cache_dir", out / "cache")), cc.get("extent") or idx.extent(),
                                 local=bm.get("local")) if bm.get("enabled", True) else None
        fig_cases(fdir, product, g, times, refs, 0.5, basemap, cc.get("extent"), out, fmt)
    LOG.info("figures in %s", out)


if __name__ == "__main__":
    main()
