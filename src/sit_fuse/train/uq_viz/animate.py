"""
Hourly animations: multi-panel maps per TEMPO scan + a synced area time series.

Each frame is rendered to PNG, then encoded with ffmpeg (MP4, H.264) and Pillow (GIF).
The strip below the maps shows either scan-level agreement with the primary reference (IoU vs ADP, with
the secondary reference, e.g. VIIRS, as markers at matched overpasses, and the ADP-VIIRS IoU as black
stars) or domain smoke areas.
Panels use the same panel strings as figures.fig_case_maps, so fire (or GOES-only /
fused) products need config only.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plotting as P
from .figures import FrameData, VizContext, draw_panel

LOG = logging.getLogger("uq-viz")


def _strip(ax, ctx: VizContext, fs: pd.DataFrame, order: pd.Series, fid_now: str, models, hms_name):
    """Area time series over the animated scans (x = scan order; overnight gaps collapsed)."""
    sc = 1e-3
    pm = ctx.primary_model
    x = {f: k for k, f in enumerate(order.index)}
    d = fs[fs.frame_id.isin(x)].copy()
    d["x"] = d.frame_id.map(x)
    g = d[d.model == pm].sort_values("x")
    if g.q05_area.notna().any():
        ax.fill_between(g.x, g.q05_area * sc, g.q95_area * sc, color=ctx.color(pm), alpha=0.18, lw=0,
                        label=f"{ctx.short(pm)} q05-q95 extent")
    ax.plot(g.x, g.expected_area * sc, "--", color=ctx.color(pm), lw=1, label=f"{ctx.short(pm)} expected area")
    for m in models:
        gm = d[d.model == m].sort_values("x")
        ax.plot(gm.x, gm.discrete_area * sc, "-", color=ctx.color(m), lw=1.2, label=f"{ctx.short(m)} mask")
    if hms_name and "hms_area" in g:
        ax.plot(g.x, g.hms_area * sc, ":", color="#00a0d0", lw=1.2,
                label=ctx.ref_cfg.get(hms_name, {}).get("label", hms_name))
    ax.axvline(x[fid_now], color="k", lw=0.8)
    ax.set_ylabel("Domain area\n(10$^3$ km$^2$)", fontsize=6.5)
    step = max(1, len(order) // 10)
    ax.set_xticks(range(0, len(order), step), [f"{pd.Timestamp(t):%H:%M}" for t in order.values[::step]])
    ax.set_xlim(-0.3, len(order) - 0.7)
    ax.set_xlabel("Scan time (UTC)", fontsize=6.5, labelpad=1)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.6, ncol=5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.grid(lw=0.3, alpha=0.5)


def _strip_agreement(ax, ctx: VizContext, fs: pd.DataFrame, order: pd.Series, fid_now: str, models,
                     primary: str, secondary: str | None, pair_stats: pd.DataFrame | None = None):
    """IoU with the primary reference per scan (lines), with the secondary reference at overpasses (markers),
    and between the two references at overpasses (black stars; reference disagreement)."""
    x = {f: k for k, f in enumerate(order.index)}
    d = fs[fs.frame_id.isin(x)].copy()
    d["x"] = d.frame_id.map(x)
    lab = lambda r: ctx.ref_cfg.get(r, {}).get("short_label", ctx.ref_cfg.get(r, {}).get("label", r))
    for m in models:
        g = d[d.model == m].sort_values("x")
        ok = (g[f"{primary}_area"].fillna(0) > 0) & (g[f"{primary}_area"].fillna(0) >= float(
            ctx.cfg.get("min_primary_smoke_frac", 0.02)) * g[f"{primary}_valid_area"].fillna(0))
        ax.plot(g.x[ok], g[f"{primary}_iou"][ok], "-", color=ctx.color(m), lw=1.2,
                label=f"{ctx.short(m)} IoU vs {lab(primary)}")
        if secondary and f"{secondary}_iou" in g:
            big = g[f"{secondary}_area"].fillna(0) >= float(ctx.cfg.get("min_secondary_smoke_km2", 500))
            if big.any():
                ax.plot(g.x[big], g[f"{secondary}_iou"][big], "D", ms=4, mfc="white", mec=ctx.color(m), mew=1,
                        ls="none", label=f"{ctx.short(m)} IoU vs {lab(secondary)}")
    if secondary and pair_stats is not None and not pair_stats.empty:
        q = pair_stats[pair_stats.frame_id.isin(x)].drop_duplicates("frame_id").copy()
        smoke = q.area_both_smoke + q.area_a_only + q.area_b_only
        q = q[smoke >= float(ctx.cfg.get("min_secondary_smoke_km2", 500))]
        if len(q):
            ax.plot(q.frame_id.map(x), q.ab_iou, "*", ms=6.5, color="k", ls="none",
                    label=f"{lab(primary)} vs {lab(secondary)} IoU")
    if secondary and f"{secondary}_valid_area" in d:
        sv = d[(d.model == models[0]) & (d[f"{secondary}_valid_area"].fillna(0) > 0)]
        for xx in sv.x:
            ax.axvspan(xx - 0.4, xx + 0.4, color="#1b9e77", alpha=0.12, lw=0)
    ys = [l.get_ydata() for l in ax.get_lines()]
    ymax = max([float(np.nanmax(np.asarray(y, float))) for y in ys if len(y) and np.isfinite(np.asarray(y, float)).any()]
               or [0.0])
    ax.set_ylim(0, max(0.6, ymax * 1.12))
    ax.axvline(x[fid_now], color="k", lw=0.8)
    ax.set_ylabel("Smoke IoU", fontsize=6.5)
    step = max(1, len(order) // 10)
    ax.set_xticks(range(0, len(order), step), [f"{pd.Timestamp(t):%H:%M}" for t in order.values[::step]])
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xlabel(f"Scan time (UTC); shaded: {lab(secondary)} overpass matched" if secondary else "Scan time (UTC)",
                  fontsize=6.5, labelpad=1)
    ax.tick_params(labelsize=6)
    nh = len(ax.get_legend_handles_labels()[1])
    ax.legend(fontsize=5.6, ncol=nh if nh <= 5 else 4, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.grid(lw=0.3, alpha=0.5)


def animate_date(ctx: VizContext, fs: pd.DataFrame, date: str, panels: list[str], overlays: list[str],
                 strip_models: list[str], hms_name: str | None, out_dir: Path, fps: float = 2.0,
                 min_granules: int = 1, dpi: int = 130, formats=("mp4", "gif"), utc_hours=None,
                 strip_reference: bool = False, strip: str = "area", primary_ref: str | None = None,
                 secondary_ref: str | None = None, pair_stats: pd.DataFrame | None = None) -> list[Path]:
    fr = ctx.frames[ctx.frames.date == date]
    if utc_hours:                                   # e.g. [14, 23]: daytime scans only
        h = fr.frame_time.dt.hour
        fr = fr[(h >= utc_hours[0]) & (h <= utc_hours[1])]
    cnt = fr.groupby("frame_id").size()
    order = fr.drop_duplicates("frame_id").set_index("frame_id").frame_time.loc[cnt[cnt >= min_granules].index]
    order = order.sort_values()
    if order.empty:
        LOG.warning("no frames for %s", date)
        return []
    ev = fr.event.iloc[0]
    split = fr.split.iloc[0]
    tmp = out_dir / f"_frames_{ctx.spec.name}_{date}"
    tmp.mkdir(parents=True, exist_ok=True)
    pngs = []
    for k, (fid, t) in enumerate(order.items()):
        fd = FrameData(ctx, fid)
        n = len(panels)
        v = fd.view
        aspect = (v[3] - v[2]) / np.cos(np.deg2rad(0.5 * (v[2] + v[3]))) / (v[1] - v[0])
        pw = 2.3
        fig = plt.figure(figsize=(pw * n + 0.3, pw * aspect + 2.4))
        gs = fig.add_gridspec(2, n, height_ratios=[pw * aspect + 0.55, 1.05], hspace=0.42, wspace=0.06,
                              left=0.06, right=0.99, top=1 - 0.32 / (pw * aspect + 2.4), bottom=0.09)
        for j, p in enumerate(panels):
            ax = fig.add_subplot(gs[0, j])
            draw_panel(ax, fd, p, overlays, fig=fig)
        ax = fig.add_subplot(gs[1, :])
        if strip == "agreement" and primary_ref and f"{primary_ref}_iou" in fs:
            _strip_agreement(ax, ctx, fs, order, fid, strip_models, primary_ref, secondary_ref, pair_stats)
        else:
            _strip(ax, ctx, fs, order, fid, strip_models, hms_name if strip_reference else None)
        fig.suptitle(f"SIT-FUSE TEMPO {ctx.spec.label.lower()}  |  "
                     f"{ctx.cfg.get('event_labels', {}).get(ev, ev)}  |  {t:%Y-%m-%d %H:%M} UTC  |  {split} day",
                     fontsize=8.5)
        p = tmp / f"f{k:03d}.png"
        fig.savefig(p, dpi=dpi, bbox_inches=None)
        plt.close(fig)
        pngs.append(p)
    stem = out_dir / f"anim_{ctx.spec.name}_{ev}_{date}"
    outs = []
    if "mp4" in formats and shutil.which("ffmpeg"):
        mp4 = stem.with_suffix(".mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "f%03d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", str(mp4)], check=True)
        outs.append(mp4)
    if "gif" in formats:
        from PIL import Image
        ims = [Image.open(p).convert("RGB") for p in pngs]
        w = 900
        ims = [im.resize((w, int(im.height * w / im.width)), Image.LANCZOS) for im in ims]
        pal = [im.quantize(colors=200, method=Image.Quantize.MEDIANCUT) for im in ims]
        gif = stem.with_suffix(".gif")
        pal[0].save(gif, save_all=True, append_images=pal[1:], duration=int(1000 / fps), loop=0, optimize=True)
        outs.append(gif)
    keep = ctx.cfg.get("animations", {}).get("keep_frames", False)
    if not keep:
        shutil.rmtree(tmp, ignore_errors=True)
    LOG.info("animation %s: %d frames -> %s", date, len(pngs), ", ".join(o.name for o in outs))
    return outs
