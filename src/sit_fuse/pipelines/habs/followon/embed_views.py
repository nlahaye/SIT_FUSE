"""Encoder embeddings at co-located pixels, written as Rep_Taxonomy paired-view bundles.

Used for the same-instrument representation study (OLCI S3A/S3B, VIIRS SNPP/JPSS1/JPSS2):
which encoder (shared across platforms vs per platform) gives platform-consistent and
physically informative representations. The comparison itself (CKA, k-NN Jaccard,
Procrustes, cross-predictability, label agreement, probes, transfer, contrasts with a
block bootstrap) runs in Rep_Taxonomy: reptaxonomy.intercompare.paired_views.

  export  one encoder applied to one platform's scenes -> per-scene npz
          (lat, lon, row, col, embedding [float16], root/parent label, parent max prob).
          Pixels: every --stride-th row and column of the valid pixels (deterministic, so the
          same pixels are selected for every encoder / platform on the shared L3m grid),
          capped at --max-per-scene. Resumable: finished scenes are skipped.
          Pixel lat/lon come from the geo_zarr (loc[row, col] = (lat, lon)), so rows are
          keyed by date and grid location and align across platforms and instruments on
          the same 4 km L3m grid.
  pair    several export directories -> one bundle: keys "YYYYMMDD|lat|lon", groups
          "<14-day block>|<tile>" (bootstrap / CV blocks), view__<name>, labels__<name>,
          optional target__<var> from the ancillary provider (SST, upwelling, standard
          L3m products), domain = platform of the first view. Only rows present in every
          view are kept (intersection), so views are exactly paired.

  python -m sit_fuse.pipelines.habs.followon.embed_views export -y HAB.yaml --instrument olci --key no_trop \
      --platform-re '^S3A' --name olci_joint_s3a --out /mnt/.../embeds/olci_joint_s3a --start 2024-01-01 --end 2025-12-31
  python -m sit_fuse.pipelines.habs.followon.embed_views pair --view olci_joint_s3a=/mnt/.../embeds/olci_joint_s3a \
      --view olci_joint_s3b=/mnt/.../embeds/olci_joint_s3b -y HAB.yaml --targets sst,sst_anom,phys_chlor_a \
      --out /mnt/.../embeds/bundle_olci_joint.npz
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- export
def select_pixels(coord: np.ndarray, stride: int, max_n: int, seed: int = 0) -> np.ndarray:
    """Indices of dataset rows on the stride lattice (row % s == 0 and col % s == 0)."""
    li, si = (1, 2) if coord.shape[1] > 2 else (0, 1)
    r, c = coord[:, li].astype(int), coord[:, si].astype(int)
    ix = np.flatnonzero((r % stride == 0) & (c % stride == 0))
    if max_n and ix.size > max_n:
        ix = np.sort(np.random.default_rng(seed).choice(ix, max_n, replace=False))
    return ix


def load_geo(yml) -> np.ndarray | None:
    from sit_fuse.pipelines.habs.hab_inference_pipeline import geo_zarr_path
    if not yml.get("geo_zarr"):
        return None
    import zarr
    return np.asarray(zarr.open(geo_zarr_path(yml), mode="r")[:])


def embed_scene(model, data, idx: np.ndarray, tiled: bool, batch=None):
    """(E, parent, maxprob) for dataset rows idx (one pass over the batches, rows kept on the fly)."""
    import torch
    from sit_fuse.pipelines.habs.hab_inference_pipeline import _device, iter_batches
    from sit_fuse.pipelines.habs.uq.hab_head_export import _embed, _root_probs
    dev = _device(model)
    want = np.zeros(len(np.asarray(data.targets_full)), bool)
    want[idx] = True
    E, P, start = [], [], 0
    with torch.no_grad():
        for xb in iter_batches(data, tiled, batch):
            n = xb.shape[0]
            m = want[start:start + n]
            start += n
            if not m.any():
                continue
            x = xb[torch.as_tensor(m)].to(device=dev, dtype=torch.float32)
            e = _embed(model, x)
            p = _root_probs(model, x)
            E.append(e.detach().float().cpu().numpy().reshape(e.shape[0], -1))
            P.append(p.detach().float().cpu().numpy())
    E = np.concatenate(E)
    P = np.concatenate(P)
    return E, P.argmax(1).astype(np.int32), P.max(1).astype(np.float32)


def entry_date(entry) -> str | None:
    from sit_fuse.pipelines.habs.hab_inference_pipeline import _entry_date
    return _entry_date(entry)


def cmd_export(a):
    import yaml
    from sit_fuse.pipelines.habs import hab_inference_pipeline as hip
    from sit_fuse.compute_resources import apply_compute_limits
    with open(a.yaml) as f:
        yml = yaml.safe_load(f)
    inf = copy.deepcopy(yml.get("inference", {}) or {})
    if a.start:
        inf["start_date"] = a.start
        inf.pop("date_windows", None)
    if a.end:
        inf["end_date"] = a.end
    cfg_path = a.model_config or hip.model_config_path(yml, a.instrument, a.key)
    sf_yaml = hip.merge_compute_conf(yml, hip._read_yaml(cfg_path))
    apply_compute_limits(sf_yaml)
    out_dir = hip.instrument_out_dir(yml, a.instrument, a.key, sf_yaml)
    sf_yaml["output"]["out_dir"] = a.model_dir or hip.instrument_model_dir(yml, a.instrument, a.key, out_dir)
    hip.apply_data_overrides(sf_yaml, inf)
    src = sf_yaml
    if a.entries_from:            # scenes listed in another model YAML (encoder applied to another platform)
        src = copy.deepcopy(sf_yaml)
        other = hip._read_yaml(a.entries_from)
        for k in ("files_train", "files_test"):
            src["data"][k] = other["data"].get(k) or []
    entries = hip.scene_entries(src, inf)
    if a.platform_re:
        rx = re.compile(a.platform_re)
        entries = [e for e in entries if rx.search(hip._entry_base(e))]
    if a.every > 1:
        entries = entries[::a.every]
    LOG.info("%s: %d scenes (model %s, weights %s)", a.name, len(entries), cfg_path, sf_yaml["output"]["out_dir"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    geo = load_geo(yml)
    if geo is None:
        raise ValueError("geo_zarr is required (pixel lat/lon for cross-platform keys)")
    from sit_fuse.datasets.dataset_utils import get_prediction_dataset
    from sit_fuse.inference.generate_output import prep_model
    model = None
    tiled = bool(sf_yaml["data"].get("tile", False))
    n_done = 0
    for e in entries:
        d = entry_date(e)
        f = out / f"{a.name}.{d}.npz"
        if f.exists() and not a.overwrite:
            continue
        data = get_prediction_dataset(sf_yaml, e)[0]
        if data is None or data.data_full is None:
            continue
        if model is None:
            model = prep_model(sf_yaml)
            model.eval()
        coord = np.asarray(data.targets_full)
        idx = select_pixels(coord, a.stride, a.max_per_scene, a.seed)
        if idx.size == 0:
            continue
        E, lab, mp = embed_scene(model, data, idx, tiled, a.batch_size)
        li, si = (1, 2) if coord.shape[1] > 2 else (0, 1)
        r, c = coord[idx, li].astype(int), coord[idx, si].astype(int)
        ok = (r < geo.shape[0]) & (c < geo.shape[1])
        lat = np.where(ok, geo[np.minimum(r, geo.shape[0] - 1), np.minimum(c, geo.shape[1] - 1), 0], np.nan)
        lon = np.where(ok, geo[np.minimum(r, geo.shape[0] - 1), np.minimum(c, geo.shape[1] - 1), 1], np.nan)
        tmp = f.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, date=d, lat=lat.astype(np.float32), lon=lon.astype(np.float32),
                            row=r.astype(np.int32), col=c.astype(np.int32), embed=E.astype(np.float16),
                            parent=lab, maxprob=mp, platform=hip._entry_base(e)[:8])
        os.replace(tmp, f)
        n_done += 1
    with open(out / f"{a.name}.meta.json", "w") as fh:
        json.dump({"name": a.name, "model_config": cfg_path, "model_dir": sf_yaml["output"]["out_dir"],
                   "instrument": a.instrument, "platform_re": a.platform_re, "entries_from": a.entries_from,
                   "stride": a.stride, "max_per_scene": a.max_per_scene, "n_scenes": len(entries)}, fh, indent=1)
    LOG.info("%s: %d new scenes", a.name, n_done)


# ----------------------------------------------------------------------------- pair
def _grid_key(lat, lon, decimals: int):
    return np.char.add(np.char.add(np.round(lat, decimals).astype(str), "|"), np.round(lon, decimals).astype(str))


def load_view(d: str, name: str | None = None, decimals: int = 3) -> pd.DataFrame:
    files = sorted(Path(d).glob("*.npz"))
    files = [f for f in files if not f.name.endswith(".tmp.npz") and (name is None or f.name.startswith(name + "."))]
    parts = []
    for f in files:
        with np.load(f) as z:
            ok = np.isfinite(z["lat"]) & np.isfinite(z["lon"])
            k = np.char.add(str(z["date"]) + "|", _grid_key(z["lat"][ok], z["lon"][ok], decimals))
            parts.append(pd.DataFrame({"key": k, "date": str(z["date"]), "lat": z["lat"][ok], "lon": z["lon"][ok],
                                       "parent": z["parent"][ok], "maxprob": z["maxprob"][ok],
                                       "platform": str(z["platform"]),
                                       "embed": list(z["embed"][ok].astype(np.float32))}))
    if not parts:
        raise FileNotFoundError(f"no npz in {d}")
    df = pd.concat(parts, ignore_index=True).drop_duplicates("key")
    return df.set_index("key")


def block_groups(dates: pd.Series, lat, lon, block_days: int, tile_deg: float) -> np.ndarray:
    t = pd.to_datetime(dates, format="%Y%m%d")
    blk = ((t - pd.Timestamp("2000-01-01")).dt.days // block_days).astype(str)
    ti = np.floor(np.asarray(lat) / tile_deg).astype(int).astype(str)
    tj = np.floor(np.asarray(lon) / tile_deg).astype(int).astype(str)
    return (blk + "|" + pd.Series(ti, index=blk.index) + "_" + pd.Series(tj, index=blk.index)).to_numpy()


def ancillary_targets(yml, df: pd.DataFrame, names: list[str], sensor: str) -> dict[str, np.ndarray]:
    from sit_fuse.pipelines.habs.followon.ancillary import AncillaryProvider
    uq = dict(yml["uq"])
    groups = set()
    for n in names:
        groups.add("physical" if n.startswith("phys_") else
                   "ancillary:upwelling" if n.startswith(("cuti", "beuti")) else
                   "ancillary:discharge" if n.startswith("q_") else
                   "ancillary:sst" if n.startswith("sst") else "ancillary:indices")
    uq["features"] = {**uq.get("features", {}), "groups": sorted(groups)}
    prov = AncillaryProvider(uq)
    out = {n: np.full(len(df), np.nan) for n in names}
    for d, ix in df.groupby("date").indices.items():
        env = prov.at(sensor, pd.Timestamp(d), df["lat"].to_numpy()[ix], df["lon"].to_numpy()[ix])
        for n in names:
            v = env.get(n if n.startswith("phys_") else "anc_" + n, env.get(n))
            if v is not None:
                out[n][ix] = v
    return out


def cmd_pair(a):
    import yaml
    sys_path_rt = a.rep_taxonomy
    if sys_path_rt:
        import sys
        sys.path.insert(0, sys_path_rt)
    from reptaxonomy.intercompare.paired_views import save_bundle
    views = {}
    for spec in a.view:
        name, d = spec.split("=", 1)
        views[name] = load_view(d, name, a.decimals)
    common = None
    for v in views.values():
        common = v.index if common is None else common.intersection(v.index, sort=False)
    if common is None or len(common) == 0:
        raise ValueError("views share no (date, grid cell) rows")
    first = next(iter(views.values())).loc[common]
    if a.max_rows and len(common) > a.max_rows:
        # keep whole dates so blocks stay intact
        dates = first["date"].drop_duplicates().sample(frac=1.0, random_state=a.seed)
        n_per = first.groupby("date").size()
        keep = dates[np.cumsum(n_per.loc[dates].to_numpy()) <= a.max_rows]
        common = first.index[first["date"].isin(keep)]
        first = first.loc[common]
    groups = block_groups(first["date"], first["lat"], first["lon"], a.block_days, a.tile_deg)
    arr_views = {n: np.stack(v.loc[common, "embed"].to_numpy()).astype(np.float32) for n, v in views.items()}
    labels = {n: v.loc[common, "parent"].to_numpy().astype(np.int64) for n, v in views.items()}
    targets = {f"maxprob_{n}": v.loc[common, "maxprob"].to_numpy(float) for n, v in views.items()}
    if a.targets:
        with open(a.yaml) as f:
            yml = yaml.safe_load(f)
        targets.update(ancillary_targets(yml, first, a.targets.split(","), a.sensor))
    domain = first["platform"].to_numpy() if a.domain == "platform" else None
    save_bundle(a.out, np.asarray(common), groups, arr_views, labels, targets, domain=domain)
    meta = {"views": list(views), "n_rows": int(len(common)), "n_groups": int(len(set(groups))),
            "n_dates": int(first["date"].nunique()), "block_days": a.block_days, "tile_deg": a.tile_deg,
            "targets": sorted(targets)}
    with open(str(a.out) + ".json", "w") as fh:
        json.dump(meta, fh, indent=1)
    LOG.info("bundle %s: %s", a.out, meta)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("-y", "--yaml", required=True)
    e.add_argument("--instrument", required=True)
    e.add_argument("--key", default="no_trop")
    e.add_argument("--model-config", help="model YAML (default: inference.model_configs of the HAB YAML)")
    e.add_argument("--model-dir", help="weights directory (default: instruments.<I>.<key>.model_dir)")
    e.add_argument("--entries-from", help="take scene lists (files_train/files_test) from this model YAML")
    e.add_argument("--platform-re", help="regex on the scene file name, e.g. '^S3A' or '^JPSS1'")
    e.add_argument("--name", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--start")
    e.add_argument("--end")
    e.add_argument("--every", type=int, default=1, help="every n-th scene")
    e.add_argument("--stride", type=int, default=4)
    e.add_argument("--max-per-scene", type=int, default=3000)
    e.add_argument("--batch-size", type=int)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--overwrite", action="store_true")
    q = sub.add_parser("pair")
    q.add_argument("--view", action="append", required=True, help="name=export_dir")
    q.add_argument("--out", required=True)
    q.add_argument("-y", "--yaml", help="HAB YAML (needed for --targets)")
    q.add_argument("--targets", help="comma list: sst,sst_anom,sst_grad,cuti,beuti,phys_chlor_a,phys_nflh,...")
    q.add_argument("--sensor", default="s3a", help="sensor used for physical (L3m product) targets")
    q.add_argument("--domain", choices=["platform", "none"], default="platform")
    q.add_argument("--block-days", type=int, default=14)
    q.add_argument("--tile-deg", type=float, default=32 * 0.0417)
    q.add_argument("--decimals", type=int, default=3)
    q.add_argument("--max-rows", type=int, default=0)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--rep-taxonomy", help="path to the Rep_Taxonomy checkout if not installed")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"export": cmd_export, "pair": cmd_pair}[a.cmd](a)


if __name__ == "__main__":
    main()
