"""Shared map / style helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize  # noqa: E402

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}
plt.rcParams.update(STYLE)

NODATA_COLOR = "#ececec"
LAND_EDGE = "#4d4d4d"


def save(fig, out_dir: Path, name: str, formats=("png", "pdf")) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def setup_map(ax, extent, basemap=None, ticks=True):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    lat0 = 0.5 * (extent[2] + extent[3])
    ax.set_aspect(1.0 / np.cos(np.deg2rad(lat0)))
    ax.set_facecolor(NODATA_COLOR)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    if basemap:
        for name, g in basemap.items():
            lw = {"coastline": 0.6, "borders": 0.6, "states": 0.35}.get(name, 0.4)
            g.plot(ax=ax, color=LAND_EDGE, linewidth=lw, zorder=4)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    if ticks:
        ax.tick_params(length=2, pad=1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):.0f}°W" if v < 0 else f"{v:.0f}°E"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°N"))
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def show(ax, a, extent, cmap, norm, zorder=2):
    m = np.ma.masked_invalid(a)
    return ax.imshow(m, extent=extent, origin="upper", cmap=cmap, norm=norm, interpolation="nearest", zorder=zorder)


def prob_norm(spec):
    if spec.log_prob:
        return LogNorm(vmin=max(spec.prob_vmin, 1e-6), vmax=spec.prob_vmax)
    return Normalize(vmin=spec.prob_vmin, vmax=spec.prob_vmax)


def mask_cmap(color):
    return ListedColormap(["#ffffff", color]), BoundaryNorm([-0.5, 0.5, 1.5], 2)


def envelope(p, lo, hi, thr):
    """0 none, 1 possible (q95 >= thr), 2 expected (p >= thr), 3 likely (q05 >= thr); NaN where p is NaN."""
    out = np.where(np.isfinite(p), 0.0, np.nan)
    if hi is not None:
        out[np.nan_to_num(hi) >= thr] = 1
    out[np.nan_to_num(p) >= thr] = 2
    if lo is not None:
        out[np.nan_to_num(lo) >= thr] = 3
    return out


def envelope_cmap(colors):
    return ListedColormap(["#ffffff", *colors]), BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)


DIFF_LABELS = ["neither", "both", "new product only", "comparison only"]
DIFF_COLORS = ["#ffffff", "#6a3d9a", "#d95f02", "#1f78b4"]


def difference(new, comp):
    """0 neither, 1 both, 2 new only, 3 comparison only; NaN where either is NaN."""
    v = np.isfinite(new) & np.isfinite(comp)
    n, c = (np.nan_to_num(new) >= 0.5), (np.nan_to_num(comp) >= 0.5)
    out = np.full(new.shape, np.nan, np.float32)
    out[v & ~n & ~c] = 0
    out[v & n & c] = 1
    out[v & n & ~c] = 2
    out[v & ~n & c] = 3
    return out


def diff_cmap():
    return ListedColormap(DIFF_COLORS), BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)


def outline(ax, gdf, color, lw=0.9, ls="-", zorder=6, label=None):
    if gdf is None or len(gdf) == 0:
        return
    gdf.boundary.plot(ax=ax, color=color, linewidth=lw, linestyle=ls, zorder=zorder)


def panel_label(ax, s):
    ax.text(0.015, 0.985, s, transform=ax.transAxes, va="top", ha="left", fontsize=8.5, fontweight="bold",
            zorder=10, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))


def contour(ax, a, index, color="#00c8ff", lw=0.6, ls="--"):
    """0.5 contour of a binary raster on the product grid (e.g. HMS daily smoke extent)."""
    t = index.transform
    h, w = a.shape
    x = t.c + (np.arange(w) + 0.5) * t.a
    y = t.f + (np.arange(h) + 0.5) * t.e
    ax.contour(x, y, a.astype(float), levels=[0.5], colors=[color], linewidths=lw, linestyles=ls, zorder=5)
