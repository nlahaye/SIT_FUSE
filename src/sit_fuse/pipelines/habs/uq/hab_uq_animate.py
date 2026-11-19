"""
Animations of the fused HAB UQ product streams.

Input: daily SITFUSE_MULTI.<date>.L3m.DAY.<token>.fused_uq.tif written by the
fuse stage (hab_multisensor_fusion.run_fusion). Each species has two streams:
  fused     observation-day pooling across sensors (gaps where no sensor saw the pixel)
  filtered  ordinal Bayesian filter over days (gap-filled, uncertainty grows with
            days since observation)

One MP4 (H.264, yuv420p, plays in PowerPoint) or GIF per species x stream, four
panels per frame:
  expected log10 concentration (discrete, in-situ bin edges)
  P(concentration >= first alert threshold)
  normalized predictive entropy
  sensors contributing (fused) | days since observation (filtered)
optional in-situ samples within +/-overlay_days, coloured by the same bins
(test-role days marked when a day-roles file is configured).

Frames are rendered one at a time (no frame buffer), so memory is one scene.
A poster PNG (day with the largest alert-probability area) and a manifest.json
are written next to the animations.

  python -m sit_fuse.pipelines.habs.uq.hab_uq_animate -y <hab yaml> [--streams filtered,fused]
or the 'animate' stage of hab_uq_pipeline. Config (all optional), uq.animation:
  out_dir: <uq.out_dir>/animations
  streams: [filtered, fused]
  format: mp4            # mp4 | gif
  fps: 4
  dpi: 110
  insitu_overlay: true
  overlay_days: 1
  land_mask: land_10     # Natural Earth land drawn grey (regionmask); none to skip
  extent: null           # [lon_min, lon_max, lat_min, lat_max]; default product extent
  threshold_index: 0     # which alert threshold to map
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sit_fuse.pipelines.habs.uq.hab_uq_common import NODATA, SpeciesSpec, species_spec

LOG = logging.getLogger(__name__)

STREAMS = {"fused": "Fused, observed days", "filtered": "Fused, temporally filtered"}
SPECIES_LABEL = {"Pseudo_nitzschia_seriata_group": "P. seriata group",
                 "Pseudo_nitzschia_delicatissima_group": "P. delicatissima group",
                 "pDA": "Particulate domoic acid", "Total_Phytoplankton": "Total phytoplankton",
                 "Alexandrium_spp": "Alexandrium spp."}
DATE_RE = re.compile(r"SITFUSE_MULTI\.(\d{8})\.L3m\.DAY\.")
DEFAULTS = {"streams": ["filtered", "fused"], "format": "mp4", "fps": 4, "dpi": 110,
            "insitu_overlay": True, "overlay_days": 1, "land_mask": "land_10", "extent": None,
            "threshold_index": 0}


def _thr_name(t: float) -> str:
    return f"{t:g}".replace(".", "p").replace("+", "")


def _fmt_conc(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def list_fused(fused_dir: str, token: str, start=None, end=None) -> list[tuple[pd.Timestamp, str]]:
    hits = glob.glob(os.path.join(fused_dir, "**", f"SITFUSE_MULTI.*.L3m.DAY.{token}.fused_uq.tif"),
                     recursive=True)
    out = []
    for p in hits:
        m = DATE_RE.search(os.path.basename(p))
        if not m:
            continue
        d = pd.Timestamp(m.group(1))
        if (start is None or d >= pd.Timestamp(start)) and (end is None or d <= pd.Timestamp(end)):
            out.append((d, p))
    return sorted(out)


def read_layers(path: str, names: list[str]):
    """Requested bands by description (missing -> None), NODATA -> NaN, plus profile."""
    import rasterio
    with rasterio.open(path) as src:
        desc = list(src.descriptions)
        prof = src.profile.copy()
        out = {}
        for n in names:
            if n in desc:
                a = src.read(desc.index(n) + 1).astype(np.float32)
                a[(a == NODATA) | ~np.isfinite(a)] = np.nan
                out[n] = a
            else:
                out[n] = None
    return out, prof


def stream_layers(stream: str, spec: SpeciesSpec, threshold_index: int = 0) -> dict[str, str]:
    p = "filtered" if stream == "filtered" else "fused"
    thr = spec.alert_thresholds[min(threshold_index, len(spec.alert_thresholds) - 1)] \
        if spec.alert_thresholds else None
    lay = {"conc": f"{p}_expected_log10_conc", "entropy": f"{p}_predictive_entropy",
           "support": "days_since_observation" if p == "filtered" else "n_sensors"}
    if thr is not None:
        lay["exceed"] = f"{p}_exceed_p_{_thr_name(thr)}"
    return lay


def land_grid(prof: dict[str, Any], source: str | None):
    if not source or source == "none":
        return None
    try:
        import regionmask
        from sit_fuse.pipelines.habs.uq.hab_uq_common import pixel_centers
        lat, lon = pixel_centers(prof)
        lat1, lon1 = np.asarray(lat), np.asarray(lon)
        if lat1.ndim == 2:      # regular grid: 1-D axes are enough
            lat1, lon1 = lat1[:, 0], lon1[0, :]
        regions = getattr(regionmask.defined_regions.natural_earth_v5_1_2, source)
        return np.isfinite(np.asarray(regions.mask(lon1, lat1)))
    except Exception as e:  # offline or missing package: animate without land
        LOG.warning("land mask %s unavailable (%s); no land shading", source, e)
        return None


def insitu_points(yml: dict[str, Any], spec: SpeciesSpec, start, end) -> pd.DataFrame | None:
    from sit_fuse.pipelines.habs.uq.hab_insitu import insitu_filter_config, load_insitu, species_frame
    uq = yml.get("uq", {})
    path = uq.get("insitu_file", yml.get("insitu_file"))
    if not path or not os.path.exists(path):
        return None
    pad = pd.Timedelta(days=int(uq.get("animation", {}).get("overlay_days", 1)) + 1)
    df = species_frame(load_insitu(path, uq.get("max_depth_m", 1.0), pd.Timestamp(start) - pad,
                                   pd.Timestamp(end) + pad, filters=insitu_filter_config(yml)), spec)
    if df.empty:
        return None
    roles = {}
    rf = uq.get("splits", {}).get("date_roles_file")
    if rf and os.path.exists(rf):
        from sit_fuse.pipelines.habs.uq.hab_uq_train import read_date_roles
        roles = read_date_roles(rf)
    df["role"] = df["date"].dt.strftime("%Y%m%d").map(lambda d: roles.get(d, "")) if roles else ""
    return df[["date", "Latitude", "Longitude", "value", "ref_class", "role", "Location_Code"]]


class _Figure:
    """Four-panel figure; artists updated in place for each frame."""

    def __init__(self, spec, stream, lay, prof, land, extent, dpi, max_support):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap

        self.plt = plt
        t = prof["transform"]
        H, W = prof["height"], prof["width"]
        full = [t.c, t.c + t.a * W, t.f + t.e * H, t.f]
        self.extent = full
        view = extent or full
        e = np.log10(np.maximum(np.asarray(spec.edges, float), spec.detection_floor))
        e[-1] = max(e[-1], e[-2] + 1)
        conc_cmap = plt.get_cmap("YlOrRd", len(e) - 1)
        self.conc_norm = BoundaryNorm(e, conc_cmap.N)
        self.conc_cmap = conc_cmap
        ent_cmap = plt.get_cmap("magma_r").copy()
        exc_cmap = plt.get_cmap("viridis").copy()
        if lay["support"] == "n_sensors":
            sup_cmap = ListedColormap(plt.get_cmap("Blues", max_support + 2)(np.arange(1, max_support + 2)))
            sup_norm = BoundaryNorm(np.arange(0.5, max_support + 1.5), sup_cmap.N)
            sup_label, sup_ticks = "Sensors contributing", list(range(1, max_support + 1))
        else:
            sup_cmap = plt.get_cmap("cividis_r", 8)
            sup_norm = BoundaryNorm([0, 1, 2, 3, 5, 7, 10, 14, 21], sup_cmap.N)
            sup_label, sup_ticks = "Days since observation", [0, 1, 2, 3, 5, 7, 10, 14, 21]
        for c in (ent_cmap, exc_cmap, sup_cmap, conc_cmap):
            c.set_bad("#e8eef2")

        aspect = (view[3] - view[2]) / max(view[1] - view[0], 1e-6)
        fw = 12.8                                # 2 panels + colorbars across
        fh = float(np.clip(2 * 5.4 * aspect + 0.9, 5.0, 10.0))
        self.fig, axs = plt.subplots(2, 2, figsize=(fw, fh), dpi=dpi, constrained_layout=True)
        self.fig.set_facecolor("white")
        self.axs = axs.ravel()
        thr = spec.alert_thresholds[0] if spec.alert_thresholds else None
        panels = [("conc", f"Expected concentration ({spec.units})", conc_cmap, self.conc_norm),
                  ("exceed", f"P(>= {_fmt_conc(thr) if thr else ''} {spec.units})", exc_cmap, None),
                  ("entropy", "Predictive entropy (0 = certain)", ent_cmap, None),
                  ("support", sup_label, sup_cmap, sup_norm)]
        self.ims = {}
        blank = np.full((H, W), np.nan, np.float32)
        for ax, (key, title, cmap, norm) in zip(self.axs, panels):
            kw = {"norm": norm} if norm is not None else {"vmin": 0.0, "vmax": 1.0}
            im = ax.imshow(blank, extent=full, origin="upper", cmap=cmap, interpolation="nearest", **kw)
            if land is not None:
                ax.imshow(np.where(land, 1.0, np.nan), extent=full, origin="upper",
                          cmap=ListedColormap(["#bdbdbd"]), vmin=0, vmax=1, interpolation="nearest")
            ax.set_xlim(view[0], view[1])
            ax.set_ylim(view[2], view[3])
            ax.set_title(title, fontsize=11, loc="left")
            ax.tick_params(labelsize=8)
            ax.set_aspect("equal")
            cb = self.fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
            cb.ax.tick_params(labelsize=8)
            if key == "conc":
                cb.set_ticks(e)
                cb.set_ticklabels([_fmt_conc(10 ** v) for v in e])
            elif key == "support":
                cb.set_ticks(sup_ticks)
            self.ims[key] = im
        self.scat = self.axs[0].scatter([], [], s=46, c=[], cmap=conc_cmap, norm=self.conc_norm,
                                        edgecolors="k", linewidths=0.9, zorder=5)
        self.test_ring = self.axs[0].scatter([], [], s=150, facecolors="none", edgecolors="#00a6ff",
                                             linewidths=1.6, zorder=6)
        self.suptitle = self.fig.suptitle("", fontsize=13, x=0.01, ha="left")
        self.footer = self.fig.text(0.99, 0.995, "", fontsize=9, ha="right", va="top", color="#555555")
        self.label = SPECIES_LABEL.get(spec.use_key, spec.use_key)
        self.stream_label = STREAMS.get(stream, stream)

    def update(self, date, arrays, pts, footer):
        for k, im in self.ims.items():
            a = arrays.get(k)
            im.set_data(a if a is not None else np.full(im.get_array().shape, np.nan, np.float32))
        if pts is not None and len(pts):
            self.scat.set_offsets(np.c_[pts["Longitude"], pts["Latitude"]])
            self.scat.set_array(np.log10(np.maximum(pts["value"].to_numpy(float), 1e-12)))
            tp = pts[pts["role"] == "test"]
            self.test_ring.set_offsets(np.c_[tp["Longitude"], tp["Latitude"]] if len(tp) else np.empty((0, 2)))
        else:
            self.scat.set_offsets(np.empty((0, 2)))
            self.scat.set_array(np.empty(0))
            self.test_ring.set_offsets(np.empty((0, 2)))
        self.suptitle.set_text(f"{self.label}  |  {self.stream_label}  |  {pd.Timestamp(date):%Y-%m-%d}")
        self.footer.set_text(footer)

    def close(self):
        self.plt.close(self.fig)


def _mask_unobserved(arrays: dict[str, np.ndarray | None], support_kind: str) -> dict[str, np.ndarray | None]:
    """Hide pixels no sensor saw today (fused: n_sensors 0) or has ever seen (filtered:
    days_since_observation NaN, i.e. prior only), in every panel."""
    sup = arrays.get("support")
    if sup is None:
        return arrays
    hide = ~np.isfinite(sup)
    if support_kind == "n_sensors":
        hide |= sup <= 0
    return {k: (None if v is None else np.where(hide, np.nan, v)) for k, v in arrays.items()}


def _writer(fmt: str, fps: int):
    from matplotlib import animation
    if fmt == "mp4" and animation.writers.is_available("ffmpeg"):
        return animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=-1,
                                      extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                                  "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]), "mp4"
    if fmt == "mp4":
        LOG.warning("ffmpeg not available; writing GIF")
    return animation.PillowWriter(fps=fps), "gif"


def animate_stream(files: list[tuple[pd.Timestamp, str]], spec: SpeciesSpec, stream: str, out_base: str,
                   cfg: dict[str, Any], insitu: pd.DataFrame | None = None,
                   sensors: list[str] | None = None) -> dict[str, Any] | None:
    lay = stream_layers(stream, spec, int(cfg.get("threshold_index", 0)))
    names = list(lay.values())
    first = None
    for _, p in files:
        a, prof = read_layers(p, names)
        if a[lay["conc"]] is not None:
            first = prof
            break
    if first is None:
        LOG.warning("%s/%s: no %s layers in %d files", spec.use_key, stream, lay["conc"], len(files))
        return None
    land = land_grid(first, cfg.get("land_mask", "land_10"))
    fig = _Figure(spec, stream, lay, first, land, cfg.get("extent"), int(cfg.get("dpi", 110)),
                  max(len(sensors or []), 1))
    writer, ext = _writer(cfg.get("format", "mp4"), int(cfg.get("fps", 4)))
    out = f"{out_base}.{ext}"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    od = pd.Timedelta(days=int(cfg.get("overlay_days", 1)))
    n, best, stats = 0, (-1.0, None), []
    try:
        with writer.saving(fig.fig, out, dpi=int(cfg.get("dpi", 110))):
            for d, p in files:
                a, prof = read_layers(p, names)
                if a[lay["conc"]] is None:      # fused stream: day with no observations
                    continue
                if prof["width"] != first["width"] or prof["height"] != first["height"]:
                    LOG.warning("%s: grid differs from first frame; skipped", p)
                    continue
                arrays = _mask_unobserved({k: a[v] for k, v in lay.items()}, lay["support"])
                pts = None
                if insitu is not None and cfg.get("insitu_overlay", True):
                    pts = insitu[(insitu["date"] >= d - od) & (insitu["date"] <= d + od)]
                valid = np.isfinite(arrays["conc"])
                ex = arrays.get("exceed")
                area = float(np.nanmean(ex[valid] >= 0.5)) if ex is not None and valid.any() else 0.0
                cov = float(valid.mean())
                stats.append({"date": f"{d:%Y-%m-%d}", "valid_fraction": cov, "alert_area_fraction": area})
                if area > best[0]:
                    best = (area, d)
                n_pts = 0 if pts is None else len(pts)
                fig.update(d, arrays, pts,
                           f"SIT-FUSE HAB UQ  |  valid {cov:.0%}  |  in-situ +/-{od.days} d: {n_pts}"
                           + ("  (rings: held-out test days)" if pts is not None and (pts["role"] == "test").any()
                              else ""))
                writer.grab_frame(facecolor="white")
                n += 1
        poster = None
        if best[1] is not None:
            d = best[1]
            a, _ = read_layers(dict(files)[d], names)
            pts = None if insitu is None else insitu[(insitu["date"] >= d - od) & (insitu["date"] <= d + od)]
            fig.update(d, _mask_unobserved({k: a[v] for k, v in lay.items()}, lay["support"]), pts, f"SIT-FUSE HAB UQ  |  poster frame {d:%Y-%m-%d}")
            poster = f"{out_base}.poster.png"
            fig.fig.savefig(poster, dpi=int(cfg.get("dpi", 110)), facecolor="white")
    finally:
        fig.close()
    if n == 0:
        os.remove(out) if os.path.exists(out) else None
        return None
    LOG.info("%s/%s: %d frames -> %s", spec.use_key, stream, n, out)
    return {"species": spec.use_key, "label": SPECIES_LABEL.get(spec.use_key, spec.use_key), "stream": stream,
            "stream_label": STREAMS.get(stream, stream), "path": out, "poster": poster, "n_frames": n,
            "start": files[0][0].strftime("%Y-%m-%d"), "end": files[-1][0].strftime("%Y-%m-%d"),
            "fps": int(cfg.get("fps", 4)), "poster_date": None if best[1] is None else f"{best[1]:%Y-%m-%d}",
            "layers": lay, "units": spec.units, "alert_thresholds": spec.alert_thresholds, "daily": stats}


def animate_species(yml: dict[str, Any], use_key: str, out_root, streams=None) -> list[dict[str, Any]]:
    uq = yml["uq"]
    cfg = {**DEFAULTS, **(uq.get("animation") or {})}
    spec = species_spec(use_key, uq.get("species", {}).get(use_key))
    fused_dir = str(Path(out_root) / "fused")
    out_dir = Path(cfg.get("out_dir") or Path(out_root) / "animations")
    start, end = uq.get("start_date", yml.get("start_date")), uq.get("end_date", yml.get("end_date"))
    files = list_fused(fused_dir, spec.fname_token, start, end)
    if not files:
        LOG.warning("%s: no fused products in %s (run the fuse stage first)", use_key, fused_dir)
        return []
    insitu = insitu_points(yml, spec, files[0][0], files[-1][0]) if cfg.get("insitu_overlay", True) else None
    res = []
    for st in streams or cfg["streams"]:
        r = animate_stream(files, spec, st, str(out_dir / f"SITFUSE_MULTI.{spec.fname_token}.{st}"), cfg,
                           insitu, uq.get("sensors"))
        if r:
            res.append(r)
    _update_manifest(out_dir, res)
    return res


def _update_manifest(out_dir: Path, entries: list[dict[str, Any]]):
    out_dir.mkdir(parents=True, exist_ok=True)
    mf = out_dir / "manifest.json"
    old = json.loads(mf.read_text()) if mf.exists() else []
    keep = [e for e in old if (e["species"], e["stream"]) not in {(r["species"], r["stream"]) for r in entries}]
    mf.write_text(json.dumps(keep + entries, indent=2, default=str))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--streams", default=None, help="comma list: filtered,fused")
    ap.add_argument("--species", default=None, help="comma list of use keys (default uq.species_run)")
    a = ap.parse_args()
    from sit_fuse.utils import read_yaml
    y = read_yaml(a.yaml)
    root = Path(y["uq"].get("out_dir", os.path.join(y.get("final_product_dir", "."), "uq")))
    keys = a.species.split(",") if a.species else (y["uq"].get("species_run") or [])
    for k in keys:
        animate_species(y, k, root, a.streams.split(",") if a.streams else None)
