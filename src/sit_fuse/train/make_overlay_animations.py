#!/usr/bin/env python
"""
Copyright [2022-26], by the California Institute of Technology and Chapman University.
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.

Multi-panel animations of SIT-FUSE probabilistic smoke (and optionally fire) products overlaid, see-through, on the
observations they were derived from: TEMPO RGB, TEMPO UV RGB and GOES ABI natural true colour built from the
16-band GOES GeoTIFFs.

    python make_overlay_animations.py -c overlay_animation.yaml [--workers 8] [--shard i/N] [--dry-run]

Inputs (YAML; see overlay_animation.example.yaml)
  time_range: [start, end] (UTC), bbox: [W, E, S, N] (deg), resolution (deg)
  observations: tempo_rgb {dir, suffix .RGB.tif}, tempo_uv_rgb {dir, suffix .UV_RGB.tif},
                goes {dir, pattern, bands, kappa0 | kappa0_from, sza_correction, gamma, night}
  products: tempo_smoke, goes_smoke (+ optional tempo_fire, goes_fire, fused_smoke): {geo_dir, model, product, sensor}
            fused products (sensor: fused, model: fused, geo_dir: <fusion output_dir>/<grid>) are matched in time
            like GOES (nearest scene within frames.goes_window_minutes)
  panels: [{title, obs, overlays: [...]}]   default: one panel per observation, overlaid with that sensor's
          products (smoke, then fire when configured), plus one panel of fused products over GOES true colour
Frames (frames.base)
  tempo  one frame per TEMPO scan (granules in the box mosaicked); GOES scene nearest the scan time
         (within frames.goes_window_minutes)
  goes   one frame per GOES scene (optionally every frames.goes_every_minutes); TEMPO scan nearest in time
         (frames.tempo_causal: latest scan at or before the frame, up to frames.tempo_max_age_minutes)
Output: <output_dir>/anim_<name>_<segment>.mp4|.gif (one segment per day with split_by: day), frames_<segment>.csv
(the files and times shown in every panel), optional PNG frames (keep_frames).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from uq_viz import overlay as O

LOG = logging.getLogger("overlay-animations")

SENSOR_OBS = {"tempo": ("tempo_rgb", "tempo_uv_rgb"), "goes": ("goes",)}
OBS_TITLES = {"tempo_rgb": "TEMPO RGB", "tempo_uv_rgb": "TEMPO UV RGB", "goes": "GOES ABI true colour"}


def _resolve(base: Path, p):
    if p is None:
        return None
    if isinstance(p, (list, tuple)):
        return [_resolve(base, x) for x in p]
    p = Path(str(p)).expanduser()
    return str(p if p.is_absolute() else (base / p).resolve())


def obs_sensor(name, oc):
    return oc.get("sensor") or ("goes" if "goes" in name.lower() or "abi" in name.lower() else "tempo")


def default_panels(obs_cfg, products):
    panels = []
    for name, oc in obs_cfg.items():
        s = obs_sensor(name, oc)
        ov = sorted([k for k, p in products.items() if p.sensor == s], key=lambda k: products[k].product != "smoke")
        what = " + ".join(sorted({products[k].product for k in ov}, key=lambda k: (k != "smoke", k != "fire", k)))
        panels.append({"title": f"{OBS_TITLES.get(name, name)}" + (f" + {s.upper()} {what}" if ov else ""),
                       "obs": name, "overlays": ov})
    fused = sorted([k for k, p in products.items() if p.sensor == "fused"], key=lambda k: products[k].product != "smoke")
    if fused:  # one extra panel: the fused product(s) over GOES true colour (else the first observation)
        under = next((n for n, oc in obs_cfg.items() if obs_sensor(n, oc) == "goes"), next(iter(obs_cfg), None))
        what = " + ".join(dict.fromkeys(products[k].product for k in fused))
        panels.append({"title": (f"{OBS_TITLES.get(under, under)} + " if under else "") + f"fused {what}",
                       "obs": under, "overlays": fused})
    return panels


def build(cfg: dict, base: Path):
    t0, t1 = (O.utc(t) for t in cfg["time_range"])
    W, E, S, N = (float(v) for v in cfg["bbox"])
    grid = O.Grid(W, E, S, N, float(cfg.get("resolution", 0.01)))
    fc = cfg.get("frames") or {}
    shift = float(fc.get("tempo_day_shift_hours", -6))
    obs_cfg = {}
    for k, v in (cfg.get("observations") or {}).items():
        if v and v.get("dir"):
            v = dict(v, dir=_resolve(base, v["dir"]))
            if v.get("kappa0_from"):
                v["kappa0_from"] = _resolve(base, v["kappa0_from"])
            obs_cfg[k] = v
    obs = {}
    for k, v in obs_cfg.items():
        obs[k] = O.GoesRGB(k, v, grid, t0, t1) if obs_sensor(k, v) == "goes" else O.TempoRGB(k, v, grid, t0, t1, shift)
    prods = {}
    for k, v in (cfg.get("products") or {}).items():
        if v and v.get("geo_dir") and v.get("enabled", True):
            v = dict(v, geo_dir=_resolve(base, v["geo_dir"]), manifest=_resolve(base, v.get("manifest")))
            prods[k] = O.ProductSource(k, v, t0, t1, grid, shift)
    panels = cfg.get("panels") or default_panels(obs_cfg, prods)
    for pn in panels:
        bad = [o for o in pn.get("overlays", []) if o not in prods]
        if bad:
            LOG.warning("panel %r: overlays %s not configured; dropped", pn.get("title"), bad)
            pn["overlays"] = [o for o in pn.get("overlays", []) if o in prods]
        if pn.get("obs") and pn["obs"] not in obs:
            LOG.warning("panel %r: observation %r not configured", pn.get("title"), pn["obs"])

    base_kind = fc.get("base", "tempo" if any(obs_sensor(k, v) == "tempo" for k, v in obs_cfg.items())
                       or any(p.sensor == "tempo" for p in prods.values()) else "goes")
    gw = float(fc.get("goes_window_minutes", 10))
    tmax = float(fc.get("tempo_max_age_minutes", 60))
    causal = bool(fc.get("tempo_causal", True))
    # frame times
    if base_kind == "tempo":
        ids = {}
        for src in [o for o in obs.values() if isinstance(o, O.TempoRGB)] + \
                   [p for p in prods.values() if p.sensor == "tempo"]:
            for f, t in src.df.groupby("frame").time.median().items():
                ids.setdefault(f, []).append(t)
        times = pd.Series({f: min(v) for f, v in ids.items()}).sort_values()
    else:
        ts = []
        for o in obs.values():
            if isinstance(o, O.GoesRGB):
                ts += list(o.df.time)
        for p in prods.values():
            if p.sensor in ("goes", "fused"):
                ts += list(p.df.time)
        ts = pd.Series(sorted(set(ts)), dtype="datetime64[ns, UTC]")
        ev = fc.get("goes_every_minutes")
        if ev and len(ts):
            ts = ts.groupby(ts.dt.floor(f"{int(ev)}min")).first()
        times = pd.Series(list(ts), index=[t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in ts])
    uh = fc.get("utc_hours")
    if uh and len(times):
        h = pd.DatetimeIndex(times.values).hour
        times = times[(h >= uh[0]) & (h <= uh[1])]

    plan = []
    for fid, t in times.items():
        fr = {"frame_id": fid, "time": t, "panels": []}
        goes_obs = {k: o.nearest(t, gw) for k, o in obs.items() if isinstance(o, O.GoesRGB)}
        for pn in panels:
            oname = pn.get("obs")
            o = obs.get(oname)
            ospec = None
            if isinstance(o, O.TempoRGB):
                ospec = o.spec(fid) if base_kind == "tempo" else None
                if base_kind != "tempo" and len(o.df):
                    ft = o.df.groupby("frame").time.median()
                    dt = (t - ft) if causal else (ft - t).abs()
                    dt = dt[(dt >= pd.Timedelta(0)) & (dt <= pd.Timedelta(minutes=tmax))]
                    ospec = o.spec(dt.idxmin()) if len(dt) else None
            elif isinstance(o, O.GoesRGB):
                ospec = o.spec_for(goes_obs[oname])
            ref_t = ospec["time"] if ospec else t
            ovs, missing = [], []
            for k in pn.get("overlays", []):
                p = prods[k]
                if p.sensor == "tempo":
                    s = p.for_tempo_frame(fid) if base_kind == "tempo" else \
                        p.nearest(t, tmax, causal=causal, max_age_minutes=tmax)
                else:
                    s = p.nearest(ref_t, gw)
                (ovs.append(s) if s else missing.append(k))
            fr["panels"].append({"title": pn.get("title", oname or "products"), "obs": ospec, "obs_name": oname,
                                 "overlays": ovs, "missing": missing})
        if any(pn["obs"] or pn["overlays"] for pn in fr["panels"]):
            plan.append(fr)
    styles = {k: dict(v, **((cfg.get("style") or {}).get(k) or {})) for k, v in O.DEFAULT_STYLE.items()}
    return grid, plan, styles


def segments(plan, split_by):
    if split_by in (None, "none", False):
        return {"all": plan}
    out = {}
    for fr in plan:
        key = fr["frame_id"][:10] if fr["frame_id"][:4].isdigit() and fr["frame_id"][4] == "-" \
            else f"{fr['time']:%Y-%m-%d}"
        out.setdefault(key, []).append(fr)
    return out


def write_movie(pngs, stem: Path, fps, formats, gif_width):
    outs = []
    if not pngs:
        return outs
    if "mp4" in formats and shutil.which("ffmpeg"):
        lst = stem.parent / f".{stem.name}_frames.txt"
        lst.write_text("".join(f"file '{Path(p).resolve()}'\nduration {1.0 / fps:.4f}\n" for p in pngs)
                       + f"file '{Path(pngs[-1]).resolve()}'\n")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white", "-r", str(max(fps, 1)),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(stem.with_suffix(".mp4"))],
                       check=True)
        lst.unlink(missing_ok=True)
        outs.append(stem.with_suffix(".mp4"))
    elif "mp4" in formats:
        LOG.warning("ffmpeg not found; mp4 skipped")
    if "gif" in formats:
        from PIL import Image
        ims = [Image.open(p).convert("RGB") for p in pngs]
        ims = [im.resize((gif_width, int(im.height * gif_width / im.width)), Image.LANCZOS) for im in ims]
        pal = [im.quantize(colors=220, method=Image.Quantize.MEDIANCUT) for im in ims]
        pal[0].save(stem.with_suffix(".gif"), save_all=True, append_images=pal[1:], duration=int(1000 / fps),
                    loop=0, optimize=True)
        outs.append(stem.with_suffix(".gif"))
    return outs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="write the frame tables only (no rendering)")
    from sit_fuse.parallel_utils import add_parallel_args, cpu_map, parse_shard, setup_compute, shard_indices
    add_parallel_args(ap, shard=True, merge=False)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfgp = Path(a.config).resolve()
    cfg = yaml.safe_load(cfgp.read_text())
    base = cfgp.parent
    n_workers, threads = setup_compute(cfg, a.num_threads, a.workers)
    out = Path(_resolve(base, cfg.get("output_dir", "overlay_animations")))
    out.mkdir(parents=True, exist_ok=True)
    grid, plan, styles = build(cfg, base)
    segs = segments(plan, cfg.get("split_by", "day"))
    keys = sorted(segs)
    keys = [keys[i] for i in shard_indices(len(keys), parse_shard(a.shard))]
    LOG.info("%d frames in %d segments; this job: %s", len(plan), len(segs), ", ".join(keys) or "none")
    basemap = None
    bm = cfg.get("basemap") or {}
    if bm.get("enabled", True):
        from uq_viz.data import load_basemap
        local = {k: (f"zip://{_resolve(base, v)}" if str(v).endswith(".zip") else _resolve(base, v))
                 for k, v in (bm.get("local") or {}).items()}
        basemap = load_basemap(Path(_resolve(base, cfg.get("cache_dir", out / "_cache"))), grid.extent, local=local)
    static = {"grid": grid, "styles": styles, "basemap": basemap, "title": cfg.get("title"),
              "ncols": cfg.get("ncols"), "panel_width": cfg.get("panel_width", 3.6), "dpi": cfg.get("dpi", 130),
              "ticks": cfg.get("ticks", True), "background": cfg.get("background", "#202020")}
    name = cfg.get("name", "overlay")
    made = []
    for key in keys:
        frs = segs[key]
        O.frames_table(frs).to_csv(out / f"frames_{name}_{key}.csv", index=False)
        if a.dry_run:
            continue
        tmp = out / f"_frames_{name}_{key}"
        tmp.mkdir(parents=True, exist_ok=True)
        pngs = cpu_map(O.render_frame, [(fr, static, str(tmp / f"f{k:04d}.png")) for k, fr in enumerate(frs)],
                       n_workers, threads)
        made += write_movie(pngs, out / f"anim_{name}_{key}", float(cfg.get("fps", 3)),
                            tuple(cfg.get("formats", ["mp4", "gif"])), int(cfg.get("gif_width", 1100)))
        if not cfg.get("keep_frames", False):
            shutil.rmtree(tmp, ignore_errors=True)
        LOG.info("%s: %d frames", key, len(frs))
    (out / f"overlay_manifest_{name}{'_' + a.shard.replace('/', 'of') if a.shard else ''}.json").write_text(
        json.dumps({"segments": keys, "files": [str(p) for p in made]}, indent=1))
    LOG.info("wrote %d files to %s", len(made), out)


if __name__ == "__main__":
    main()
