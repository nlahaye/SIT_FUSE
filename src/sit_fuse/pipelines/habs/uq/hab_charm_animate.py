"""
Animations of the SIT-FUSE vs C-HARM comparison on the C-HARM 3 km grid.

Input: daily CHARM_CMP.<date>.<target>.L<lead>.tif stacks from hab_charm_compare maps (bands
charm, charm_chla, fused, combined (pn), type_<family>) and, when present, the consistency flags
(CONSISTENCY.<date>.tif, any_flag) from hab_charm_compare consistency.

Layouts (one MP4 / GIF per target x lead x layout, frames rendered one at a time):
  compare  C-HARM | SIT-FUSE fused | SIT-FUSE VIIRS | fused - C-HARM   (shared 0-1 scale)
  types    C-HARM and every SIT-FUSE stream (fused, size-class combination, each instrument type)

In-situ overlays on every panel: stations that reported the target (hollow) and samples within
+/-overlay_days (filled red when the observation exceeds the target threshold, white when
below). SIT-FUSE pixels failing a consistency check are hatched. The C-HARM version and lead are
printed on each frame.

  python -m sit_fuse.pipelines.habs.uq.hab_charm_animate -y <hab yaml> --target pn [--layout compare]
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sit_fuse.pipelines.habs.uq import hab_charm as hc
from sit_fuse.pipelines.habs.uq.hab_charm_compare import TARGETS, cfg_of, cmp_dir
from sit_fuse.pipelines.habs.uq.hab_uq_animate import _writer, land_grid  # noqa: F401
from sit_fuse.pipelines.habs.uq.hab_uq_common import NODATA, species_spec

LOG = logging.getLogger(__name__)
LABEL = {"pn": "Pseudo-nitzschia >= 10k cells/L", "pda": "pDA >= 0.5 ng/mL", "cda": "cDA >= 10 pg/cell"}
STREAM_LABEL = {"charm": "C-HARM", "fused": "SIT-FUSE fused", "combined": "SIT-FUSE size-class combination",
                "type_viirs": "SIT-FUSE VIIRS", "type_olci": "SIT-FUSE OLCI", "type_modis": "SIT-FUSE MODIS",
                "type_oci": "SIT-FUSE PACE OCI"}
RE = re.compile(r"CHARM_CMP\.(\d{8})\.")


def _read(path):
    import rasterio
    with rasterio.open(path) as src:
        names = list(src.descriptions)
        a = src.read().astype(np.float32)
        prof, tags = src.profile.copy(), src.tags()
    a[(a == NODATA) | ~np.isfinite(a)] = np.nan
    return dict(zip(names, a)), prof, tags


def _flags(cons_dir: Path, ds: str, prof):
    hits = glob.glob(str(cons_dir / "*" / f"CONSISTENCY.{ds}.tif"))
    if not hits:
        return None
    import rasterio
    with rasterio.open(hits[0]) as src:
        names = list(src.descriptions)
        if "any_flag" not in names:
            return None
        a = src.read(names.index("any_flag") + 1).astype(np.float32)
        sp = src.profile.copy()
    a[(a == NODATA) | ~np.isfinite(a)] = np.nan
    return hc.regrid_to(a, sp, prof, "max")


def _insitu(yml, target):
    from sit_fuse.pipelines.habs.uq.hab_charm_compare import _insitu as load
    from sit_fuse.pipelines.habs.uq.hab_insitu import species_frame
    key = TARGETS[target]["species"]
    df = species_frame(load(yml), species_spec(key, yml["uq"].get("species", {}).get(key)))
    return None if df.empty else df[["date", "Latitude", "Longitude", "value", "Location_Code"]]


def animate_target(yml, target: str, lead: int = 0, layout: str = "compare") -> dict | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    c = cfg_of(yml)
    ua = {k: v for k, v in (yml["uq"].get("animation") or {}).items() if k in ("fps", "dpi", "format", "overlay_days")}
    acfg = {"fps": 4, "dpi": 110, "format": "mp4", "overlay_days": 1, **ua, **(c.get("animation") or {})}
    base = cmp_dir(yml)
    files = sorted(glob.glob(str(base / "maps" / target / "*" / f"CHARM_CMP.*.{target}.L{lead}.tif")))
    if not files:
        LOG.warning("%s L%d: no comparison stacks (run maps first)", target, lead)
        return None
    first, prof, _ = _read(files[0])
    thr = TARGETS[target]["threshold"]
    ins = _insitu(yml, target)
    sites = None if ins is None else ins.groupby("Location_Code")[["Latitude", "Longitude"]].median()
    if layout == "compare":
        panels = ["charm", "fused" if "fused" in first else "combined", "type_viirs", "diff"]
    else:
        panels = ["charm"] + [k for k in ("fused", "combined", "type_viirs", "type_olci", "type_modis", "type_oci")
                              if k in first]
    ncol = 2 if len(panels) <= 4 else 3
    nrow = int(np.ceil(len(panels) / ncol))
    t = prof["transform"]
    ext = [t.c, t.c + t.a * prof["width"], t.f + t.e * prof["height"], t.f]
    aspect = (ext[3] - ext[2]) / max(ext[1] - ext[0], 1e-6)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.6 * ncol + 0.6, float(np.clip(4.4 * nrow * aspect, 4, 11))),
                            dpi=int(acfg["dpi"]), constrained_layout=True)
    axs = np.atleast_1d(axs).ravel()
    for ax in axs[len(panels):]:
        ax.set_visible(False)
    pcm = plt.get_cmap("viridis").copy()
    pcm.set_bad("#e8eef2")
    dcm = plt.get_cmap("RdBu_r").copy()
    dcm.set_bad("#e8eef2")
    blank = np.full((prof["height"], prof["width"]), np.nan, np.float32)
    land = ~np.isfinite(first.get("charm_chla", first["charm"]))
    ims, hatch_ax, scat = {}, [], []
    for ax, k in zip(axs, panels):
        diff = k == "diff"
        im = ax.imshow(blank, extent=ext, origin="upper", interpolation="nearest",
                       cmap=dcm if diff else pcm, vmin=-1 if diff else 0, vmax=1)
        ax.imshow(np.where(land, 1.0, np.nan), extent=ext, origin="upper", cmap=ListedColormap(["#bdbdbd"]),
                  vmin=0, vmax=1, interpolation="nearest")
        title = "SIT-FUSE fused - C-HARM" if diff else STREAM_LABEL.get(k, k)
        ax.set_title(title, fontsize=10, loc="left")
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("difference" if diff else "probability", fontsize=8)
        if sites is not None:
            ax.scatter(sites["Longitude"], sites["Latitude"], s=14, facecolors="none", edgecolors="#333333",
                       linewidths=0.6, alpha=0.8, zorder=4)
        scat.append(ax.scatter([], [], s=36, c=[], cmap=ListedColormap(["#ffffff", "#d7301f"]), vmin=0, vmax=1,
                               edgecolors="k", linewidths=0.8, zorder=5))
        ims[k] = im
        if k != "charm" and not k.startswith("type_"):
            hatch_ax.append(ax)          # flags come from the fused products
    sup = fig.suptitle("", fontsize=11, x=0.01, ha="left")
    writer, ext_ = _writer(acfg["format"], int(acfg["fps"]))
    out_dir = base / "animations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = str(out_dir / f"CHARM_CMP.{target}.L{lead}.{layout}.{ext_}")
    od = pd.Timedelta(days=int(acfg["overlay_days"]))
    hatches, n = [], 0
    try:
        with writer.saving(fig, out, dpi=int(acfg["dpi"])):
            for p in files:
                d = pd.Timestamp(RE.search(os.path.basename(p)).group(1))
                a, pr, tags = _read(p)
                if pr["width"] != prof["width"] or pr["height"] != prof["height"]:
                    continue
                prim = panels[1] if layout == "compare" else "fused"
                for k, im in ims.items():
                    if k == "diff":
                        v = (a.get(prim) - a["charm"]) if a.get(prim) is not None else None
                    else:
                        v = a.get(k)
                    im.set_data(blank if v is None else v)
                for h in hatches:
                    h.remove()
                hatches = []
                fl = _flags(base / "consistency", f"{d:%Y%m%d}", pr)
                if fl is not None and np.nanmax(np.nan_to_num(fl, nan=0)) > 0:
                    for ax in hatch_ax:
                        hatches.append(ax.contourf(np.linspace(ext[0], ext[1], pr["width"]),
                                                   np.linspace(ext[3], ext[2], pr["height"]),
                                                   np.nan_to_num(fl, nan=0), levels=[0.5, 1.5], colors="none",
                                                   hatches=["////"], zorder=3))
                pts = None if ins is None else ins[(ins["date"] >= d - od) & (ins["date"] <= d + od)]
                if pts is not None:      # exceedances drawn last, on top of co-located non-exceedances
                    pts = pts.sort_values("value")
                for sc in scat:
                    if pts is not None and len(pts):
                        sc.set_offsets(np.c_[pts["Longitude"], pts["Latitude"]])
                        sc.set_array((pts["value"].to_numpy(float) >= thr).astype(float))
                    else:
                        sc.set_offsets(np.empty((0, 2)))
                        sc.set_array(np.empty(0))
                sup.set_text(f"{LABEL[target]}  |  {d:%Y-%m-%d}  |  C-HARM {tags.get('charm_version', '?')}, "
                             f"{'nowcast' if lead == 0 else f'{lead}-day forecast'}\n"
                             f"hollow: stations; filled: samples +/-{od.days} d (red = observed exceedance)"
                             + ("; hatched: consistency check failed" if hatches else ""))
                writer.grab_frame(facecolor="white")
                n += 1
    finally:
        plt.close(fig)
    LOG.info("%s L%d %s: %d frames -> %s", target, lead, layout, n, out)
    entry = {"target": target, "lead": lead, "layout": layout, "path": out, "n_frames": n,
             "start": RE.search(os.path.basename(files[0])).group(1),
             "end": RE.search(os.path.basename(files[-1])).group(1)}
    mf = out_dir / "manifest.json"
    old = json.loads(mf.read_text()) if mf.exists() else []
    old = [e for e in old if (e["target"], e["lead"], e["layout"]) != (target, lead, layout)]
    mf.write_text(json.dumps(old + [entry], indent=2))
    return entry


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--target", choices=list(TARGETS), required=True)
    ap.add_argument("--lead", type=int, default=0)
    ap.add_argument("--layout", default="compare,types")
    a = ap.parse_args()
    y = hc.read_yaml_any(a.yaml)
    for lay in a.layout.split(","):
        animate_target(y, a.target, a.lead, lay)
