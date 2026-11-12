"""
Copyright [2022-26], by the California Institute of Technology and Chapman University.
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.

Why are there no (or few) valid training samples? Per-channel validity check for any SIT-FUSE config.

Applies exactly the training-time rule (datasets/scene_sampling.preprocess_read + valid_center_mask):
a pixel is bad if ANY channel is non-finite, < valid_min, > valid_max or == fill_value, and a sample
needs a fully valid (2 * pixel_padding + 1)^2 window. A single channel that is bad everywhere
therefore removes every sample.

For each checked scene and channel it reports the fraction non-finite / below valid_min / above
valid_max / equal to fill, the finite min / max, and how many pixels would become valid if that channel
alone were dropped ("unlock"). The channels with the largest unlock are the culprits.

  python -m sit_fuse.datasets.valid_qa -y config/model/copernicus_shark_v2_split.yaml \
      [--n_scenes 5] [--files a.zarr b.zarr] [--names <tag>_env_meta.json] [--out qa.csv]

--names: a JSON with a "channels" list (e.g. the NatureNet env meta), or a text file with one name per line.
"""
import argparse
import csv
import json
import os

import numpy as np
import yaml

from sit_fuse.datasets import scene_sampling as ss

_EPS = 0.00000000005


def channel_names(path, n):
    if not path:
        return [str(i) for i in range(n)]
    if path.endswith(".json"):
        with open(path) as f:
            names = list(json.load(f)["channels"])
    else:
        with open(path) as f:
            names = [l.strip() for l in f if l.strip()]
    if len(names) != n:
        print("WARNING: %d names for %d channels; names may not line up" % (len(names), n))
    return [names[i] if i < len(names) else str(i) for i in range(n)]


def scene_qa(dat, chan_dim, delete_chans, valid_min, valid_max, fill_value, pixel_padding):
    """Per-channel stats and overall valid counts for one (C, H, W)-like array (any chan_dim)."""
    dat = np.asarray(dat, dtype=np.float32)
    if dat.ndim == 2:
        dat = dat[None]
        chan_dim = 0
    if dat.ndim != 3:
        raise ValueError("expected a single 3-D scene, got shape %s" % (dat.shape,))
    if delete_chans:
        dat = np.delete(dat, delete_chans, chan_dim)
    dat = np.moveaxis(dat, chan_dim, 0)
    C, H, W = dat.shape
    nonfin = ~np.isfinite(dat)
    below = (dat < valid_min - _EPS) if valid_min is not None else np.zeros_like(nonfin)
    above = (dat > valid_max - _EPS) if valid_max is not None else np.zeros_like(nonfin)
    isfill = (dat == fill_value) if fill_value is not None else np.zeros_like(nonfin)
    bad = nonfin | below | above | isfill
    n_bad = bad.sum(0)
    valid_px = n_bad == 0
    p = int(pixel_padding or 0)
    win = ss.valid_center_mask(~valid_px, p)
    # "any channel good" approximates the water / data footprint
    footprint = (~bad).any(0)
    rows = []
    for c in range(C):
        good = np.isfinite(dat[c]) & ~bad[c]
        fin = dat[c][np.isfinite(dat[c]) & ~isfill[c]]
        unlock_px = valid_px | ((n_bad == 1) & bad[c])
        rows.append(dict(
            channel=c,
            frac_bad_in_footprint=float(bad[c][footprint].mean()) if footprint.any() else 1.0,
            frac_nonfinite=float(nonfin[c].mean()), frac_below_min=float(below[c].mean()),
            frac_above_max=float(above[c].mean()), frac_fill=float(isfill[c].mean()),
            finite_min=float(fin.min()) if fin.size else float("nan"),
            finite_max=float(fin.max()) if fin.size else float("nan"),
            n_good=int(good.sum()),
            unlock_pixels=int(unlock_px.sum() - valid_px.sum()),
            unlock_windows=int(ss.valid_center_mask(~unlock_px, p).sum() - win.sum())))
    return dict(shape=(C, H, W), footprint=int(footprint.sum()), valid_pixels=int(valid_px.sum()),
                valid_windows=int(win.sum()), channels=rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-channel validity check for SIT-FUSE training scenes.")
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--files", nargs="*", default=None, help="scenes to check (default: spread over files_train)")
    ap.add_argument("--n_scenes", type=int, default=5)
    ap.add_argument("--names", default=None, help="channel names: JSON with 'channels' or a text file")
    ap.add_argument("--out", default=None, help="CSV with one row per scene x channel")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args(argv)
    with open(a.yaml) as f:
        conf = yaml.safe_load(f)
    d = conf["data"]
    from sit_fuse.utils import get_read_func
    read = get_read_func(d["reader_type"])
    files = a.files
    if not files:
        allf = ss.expand_filenames(d.get("files_train") or d.get("files_test") or [])
        if not allf:
            raise SystemExit("no files in data.files_train / files_test")
        idx = np.unique(np.linspace(0, len(allf) - 1, min(a.n_scenes, len(allf))).astype(int))
        files = [allf[i] for i in idx]
    print("rule: bad if non-finite, < valid_min (%s), > valid_max (%s) or == fill_value (%s) in ANY channel; "
          "window %dx%d" % (d.get("valid_min"), d.get("valid_max"), d.get("fill_value"),
                             2 * int(d.get("pixel_padding", 0)) + 1, 2 * int(d.get("pixel_padding", 0)) + 1))
    names, out_rows, agg = None, [], {}
    for fn in files:
        dat = read(fn, **(d.get("reader_kwargs") or {}))
        if dat is None:
            print("SKIP (reader returned None):", fn)
            continue
        q = scene_qa(dat, int(d.get("chan_dim", 0)), d.get("delete_chans") or [], d.get("valid_min"),
                     d.get("valid_max"), d.get("fill_value"), d.get("pixel_padding", 0))
        if names is None:
            names = channel_names(a.names, q["shape"][0])
            nc = d.get("number_channels")
            if nc and int(nc) != q["shape"][0]:
                print("NOTE: number_channels = %s but the scene has %d channels" % (nc, q["shape"][0]))
        print("\n%s  shape %s  data footprint %d px  valid px %d  valid windows %d" %
              (os.path.basename(str(fn)), q["shape"], q["footprint"], q["valid_pixels"], q["valid_windows"]))
        for r in q["channels"]:
            r["name"] = names[r["channel"]]
            r["scene"] = os.path.basename(str(fn))
            out_rows.append(r)
            g = agg.setdefault(r["channel"], dict(name=r["name"], unlock=0, bad=[], mn=np.inf, mx=-np.inf))
            g["unlock"] += r["unlock_windows"]
            g["bad"].append(r["frac_bad_in_footprint"])
            if np.isfinite(r["finite_min"]):
                g["mn"], g["mx"] = min(g["mn"], r["finite_min"]), max(g["mx"], r["finite_max"])
        worst = sorted(q["channels"], key=lambda r: (-r["unlock_windows"], -r["frac_bad_in_footprint"]))[: a.top]
        print("  %-4s %-28s %9s %8s %8s %8s %8s %12s %12s %10s" % ("ch", "name", "bad@fp", "nonfin", "<min", ">max",
                                                                  "fill", "min", "max", "unlock_win"))
        for r in worst:
            print("  %-4d %-28s %9.4f %8.4f %8.4f %8.4f %8.4f %12.4g %12.4g %10d" % (
                r["channel"], r["name"][:28], r["frac_bad_in_footprint"], r["frac_nonfinite"], r["frac_below_min"],
                r["frac_above_max"], r["frac_fill"], r["finite_min"], r["finite_max"], r["unlock_windows"]))
    if agg:
        print("\nAcross scenes (bad@fp = share of the data footprint where the channel is bad):")
        for c, g in sorted(agg.items(), key=lambda kv: (-kv[1]["unlock"], -np.mean(kv[1]["bad"])))[: a.top]:
            print("  %-4d %-28s mean bad@fp %.4f  unlock windows %d  finite range [%.4g, %.4g]" %
                  (c, g["name"][:28], float(np.mean(g["bad"])), g["unlock"], g["mn"], g["mx"]))
        always = [g["name"] for g in agg.values() if min(g["bad"]) > 0.999]
        if always:
            print("\nChannels bad over the whole footprint in every checked scene:", ", ".join(always))
            print("Fix them upstream, or drop them with data.delete_chans (indices) and lower number_channels.")
    if a.out and out_rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["scene", "channel", "name"] + [k for k in out_rows[0]
                                                                              if k not in ("scene", "channel", "name")])
            w.writeheader()
            w.writerows(out_rows)
        print("WROTE", a.out)


if __name__ == "__main__":
    main()
