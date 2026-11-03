#!/usr/bin/env python3
"""
Export SIT-FUSE hierarchical clustering-head distributions per scene.

Runs the frozen encoder -> root (parent) IIC head -> child IIC heads exactly as
Heir_DC.forward() does, but keeps the probability information that
generate_output.py discards:

  * top-K parent cluster IDs + probabilities, parent entropy / margin / max
  * for each of the top-R parents, the top-K child IDs + probabilities from
    that parent's child head (Heir_DC only evaluates the argmax parent)
  * child entropy / margin / max for the argmax parent, joint probability
  * the SIT-FUSE joint label (child + num_child_classes * parent)

Notes on the existing SIT-FUSE outputs
--------------------------------------
* MultiPrototypes heads already end in nn.Softmax. Their outputs are used here
  as probabilities directly. generate_output.run_inference() applies a second
  softmax to the root-only ("no_heir") output, which flattens probabilities;
  do not use those values for UQ.
* The existing ".proba" raster is only max_j q(child j | argmax parent).
* The existing cluster raster stores joint_label / 1000. This exporter writes
  integer joint labels.

Usage:
    python export_head_outputs.py export_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sit_fuse.datasets.dataset_utils import get_prediction_dataset
from sit_fuse.inference.generate_output import prep_model
from sit_fuse.utils import get_read_func, read_yaml

try:  # package import (pipelines); bare import kept for script use from train/
    from sit_fuse.train.sf_uq_utils import (
        FORMAT_VERSION,
        HEAD_NODATA,
        head_band_names,
        write_band_stack,
    )
except ImportError:  # pragma: no cover
    from sf_uq_utils import (
        FORMAT_VERSION,
        HEAD_NODATA,
        head_band_names,
        write_band_stack,
    )

LOG = logging.getLogger("export-head-outputs")


# ---------------------------------------------------------------------------
# Head distributions
# ---------------------------------------------------------------------------

def _first(out):
    while isinstance(out, (list, tuple)):
        out = out[0]
    return out


def _as_prob_rows(y: torch.Tensor) -> torch.Tensor:
    """Root-head output -> (N_pixels, C), pixel order matching Heir_DC."""
    if y.ndim == 2:
        P = y
    elif y.ndim == 4:                       # B, C, H, W
        P = y.permute(0, 2, 3, 1).reshape(-1, y.shape[1])
    elif y.ndim == 3:                       # C, H, W
        P = y.permute(1, 2, 0).reshape(-1, y.shape[0])
    else:
        raise ValueError(f"Unsupported root-head output shape {tuple(y.shape)}")
    return _ensure_probs(P)


def _ensure_probs(P: torch.Tensor) -> torch.Tensor:
    P = P.float()
    s = P.sum(dim=1)
    if (P.min() < -1e-6) or (torch.abs(s - 1.0).max() > 1e-3):
        # head did not end in softmax; treat as logits
        P = torch.softmax(P, dim=1)
    return P


def _as_embed_rows(x: torch.Tensor) -> torch.Tensor:
    """Encoder output -> (N_pixels, D), same flattening as Heir_DC.forward."""
    if x.ndim == 2:
        return x
    if x.ndim == 4:
        return torch.flatten(x.permute(1, 0, 2, 3), start_dim=1).permute(1, 0)
    raise NotImplementedError(
        f"Encoder output with ndim={x.ndim} is not supported by this exporter."
    )


def _summaries(P: torch.Tensor, k: int):
    k = min(k, P.shape[1])
    top_p, top_i = torch.topk(P, k=k, dim=1)
    Pc = torch.clamp(P, min=1e-12)
    ent = -(Pc * torch.log(Pc)).sum(dim=1) / np.log(P.shape[1])
    margin = top_p[:, 0] - (top_p[:, 1] if k > 1 else 0.0)
    return top_i, top_p, top_p.sum(dim=1), top_p[:, 0], margin, ent


@torch.no_grad()
def head_distributions(model, x: torch.Tensor, cfg: dict, return_aux: bool = False):
    """
    Returns (N_pixels, n_bands) float32 in head_band_names() order.
    return_aux=True also returns the full root-head probabilities P (N, n_parent)
    and encoder embeddings E (N, D) as torch tensors, so callers (e.g. the HAB
    inference pipeline: MC input noise, embedding OOD) need no second forward.
    """
    Kp = int(cfg.get("top_k_parent", 5))
    Kc = int(cfg.get("top_k_child", 3))
    R = int(cfg.get("n_parent_ranks", 2))
    num_child = int(model.num_classes)
    if R > Kp:
        raise ValueError("head_export.n_parent_ranks must be <= top_k_parent.")

    out = model.forward(x, return_embed=True)
    if out is None:
        return None
    embed = _first(out[2])  # Heir_DC returns the unflattened encoder output
    y = _first(model.pretrained_model.forward(x))

    P = _as_prob_rows(y)
    E = _as_embed_rows(embed)
    if E.shape[0] != P.shape[0]:
        raise RuntimeError(
            f"Embedding rows ({E.shape[0]}) != root-head rows ({P.shape[0]})."
        )
    n = P.shape[0]
    dev = P.device
    cfg["_num_parent_classes"] = int(P.shape[1])

    p_top_i, p_top_p, _, p_max, p_margin, p_ent = _summaries(P, Kp)

    child_ids = torch.full((R, n, Kc), -1, dtype=torch.long, device=dev)
    child_ps = torch.zeros((R, n, Kc), device=dev)
    child_mass = torch.zeros((R, n), device=dev)
    c_max = torch.zeros(n, device=dev)
    c_margin = torch.zeros(n, device=dev)
    c_ent = torch.zeros(n, device=dev)
    present = torch.zeros(n, device=dev)

    tree = model.clust_tree["1"]
    for r in range(R):
        parents = p_top_i[:, r]
        for pid in torch.unique(parents).tolist():
            head = tree.get(str(int(pid)))
            if head is None:
                continue
            idx = torch.where(parents == pid)[0]
            Q = _ensure_probs(_first(head.forward(E[idx])))
            ti, tp, tm, tmax, tmarg, tent = _summaries(Q, Kc)
            child_ids[r, idx, :ti.shape[1]] = ti
            child_ps[r, idx, :tp.shape[1]] = tp
            child_mass[r, idx] = tm
            if r == 0:
                c_max[idx], c_margin[idx], c_ent[idx] = tmax, tmarg, tent
                present[idx] = 1.0

    parent = p_top_i[:, 0]
    child = torch.where(present > 0, child_ids[0, :, 0], torch.zeros_like(parent))
    joint = child + num_child * parent
    child_band = torch.where(present > 0, child_ids[0, :, 0], torch.full_like(parent, -1))
    joint_prob = p_max * torch.where(present > 0, c_max, torch.ones_like(c_max))

    cols = [joint.float(), parent.float(), child_band.float(), present]
    for i in range(Kp):
        cols += [p_top_i[:, i].float(), p_top_p[:, i]]
    cols += [p_top_p[:, :Kp].sum(dim=1), p_max, p_margin, p_ent]
    for r in range(R):
        for j in range(Kc):
            cols += [child_ids[r, :, j].float(), child_ps[r, :, j]]
        cols += [child_mass[r]]
    cols += [c_max, c_margin, c_ent, joint_prob]
    rows = torch.stack(cols, dim=1).cpu().numpy().astype(np.float32)
    if return_aux:
        return rows, P, E
    return rows


# ---------------------------------------------------------------------------
# Scene assembly (mirrors generate_output.plot_clusters placement)
# ---------------------------------------------------------------------------

def assemble_scene(coord: np.ndarray, rows: np.ndarray, tile_hw, pixel_padding=1):
    line_ind, samp_ind = (1, 2) if coord.shape[1] > 2 else (0, 1)
    H = int(coord[:, line_ind].max()) + 1 + pixel_padding
    W = int(coord[:, samp_ind].max()) + 1 + pixel_padding
    nb = rows.shape[-1]
    out = np.full((nb, H, W), HEAD_NODATA, dtype=np.float32)

    if tile_hw is None:
        out[:, coord[:, line_ind].astype(int), coord[:, samp_ind].astype(int)] = rows.T
    else:
        th, tw = tile_hw
        tiles = rows.reshape(coord.shape[0], th, tw, nb)
        for i in range(coord.shape[0]):
            r0, c0 = int(coord[i, line_ind]), int(coord[i, samp_ind])
            r1, c1 = min(H, r0 + th), min(W, c0 + tw)
            out[:, r0:r1, c0:c1] = np.moveaxis(tiles[i, :r1 - r0, :c1 - c0], -1, 0)
    return out


def _georef_info(path) -> dict | None:
    """
    Return grid info if `path` opens with rasterio and carries georeferencing
    (CRS + non-identity affine transform, or GCPs); otherwise None.
    """
    import rasterio
    from rasterio.errors import RasterioError
    from rasterio.transform import Affine

    try:
        with rasterio.open(path) as src:
            info = {"height": src.height, "width": src.width,
                    "transform": src.transform, "crs": src.crs}
            gcps, gcp_crs = src.gcps
    except (RasterioError, OSError, ValueError):
        return None
    if info["crs"] is not None and info["transform"] != Affine.identity():
        return info
    if gcps and gcp_crs is not None:
        info.update(transform=Affine.identity(), crs=None, gcps=(list(gcps), gcp_crs))
        return info
    return None


def _split_latlon(ll: np.ndarray, order: str):
    """SIT-FUSE *_geo readers return (2, H, W) or (H, W, 2); most are [lat, lon]."""
    ll = np.squeeze(np.asarray(ll, dtype=np.float64))
    if ll.ndim != 3 or 2 not in (ll.shape[0], ll.shape[-1]):
        return None
    a, b = (ll[0], ll[1]) if ll.shape[0] == 2 else (ll[..., 0], ll[..., 1])
    order = str(order).lower()
    if order == "auto":  # latitude never exceeds 90 deg in magnitude
        order = "lon_lat" if np.nanmax(np.abs(a)) > 90.0 >= np.nanmax(np.abs(b)) else "lat_lon"
    return (a, b) if order == "lat_lon" else (b, a)


def _grid_from_latlon(lat: np.ndarray, lon: np.ndarray, n_gcps: int = 20, rtol: float = 1e-3):
    """
    Regular lat/lon grid -> EPSG:4326 affine transform; otherwise GCPs
    (n_gcps x n_gcps, pixel centres) carrying EPSG:4326 coordinates.
    """
    from rasterio.control import GroundControlPoint
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    H, W = lat.shape
    good = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 360)
    if not good.any():
        return None
    if good.all() and H > 1 and W > 1:
        dlat = np.diff(lat[:, 0])
        dlon = np.diff(lon[0, :])
        regular = (np.allclose(lat, lat[:, :1], rtol=0, atol=abs(dlat).mean() * rtol)
                   and np.allclose(lon, lon[:1, :], rtol=0, atol=abs(dlon).mean() * rtol)
                   and np.allclose(dlat, dlat.mean(), rtol=rtol)
                   and np.allclose(dlon, dlon.mean(), rtol=rtol))
        if regular:
            dy, dx = dlat.mean(), dlon.mean()
            transform = Affine(dx, 0.0, lon[0, 0] - dx / 2.0, 0.0, dy, lat[0, 0] - dy / 2.0)
            return {"height": H, "width": W, "transform": transform, "crs": CRS.from_epsg(4326)}

    rr = np.unique(np.linspace(0, H - 1, min(n_gcps, H)).round().astype(int))
    cc = np.unique(np.linspace(0, W - 1, min(n_gcps, W)).round().astype(int))
    gcps = [GroundControlPoint(row=r + 0.5, col=c + 0.5, x=float(lon[r, c]), y=float(lat[r, c]))
            for r in rr for c in cc if good[r, c]]
    if len(gcps) < 3:
        return None
    return {"height": H, "width": W, "transform": Affine.identity(), "crs": None,
            "gcps": (gcps, CRS.from_epsg(4326))}


def _geo_reader_grid(fname, sf_yaml: dict, georef_cfg: dict):
    """Geolocation from the SIT-FUSE geo reader (data.geo_reader_type), if configured."""
    d = sf_yaml.get("data", {})
    reader = d.get("geo_reader_type")
    if not reader:
        return None
    try:
        ll = get_read_func(reader)(fname, **(d.get("geo_reader_kwargs") or {}))
    except Exception as exc:  # reader-specific failures -> fall through to no-geo
        LOG.warning("geo reader %s failed for %s: %s", reader, fname, exc)
        return None
    if ll is None:
        return None
    split = _split_latlon(ll, georef_cfg.get("latlon_order", "auto"))
    if split is None:
        LOG.warning("geo reader %s returned unexpected shape %s", reader, np.shape(ll))
        return None
    return _grid_from_latlon(*split, n_gcps=int(georef_cfg.get("n_gcps", 20)))


def resolve_georef_source(fname, explicit_ref, mode: str, sf_yaml: dict | None = None,
                          georef_cfg: dict | None = None):
    """
    Pick the grid used to georeference one scene, in order:
      1. explicit georef.reference_geotiffs entry
      2. the input file(s) themselves, if rasterio sees a CRS/transform or GCPs
      3. lat/lon from the SIT-FUSE geo reader (data.geo_reader_type /
         geo_reader_kwargs via sit_fuse.utils.get_read_func): regular grids
         become an EPSG:4326 affine transform, swaths become GCPs
      4. none (no-geo)

    mode (georef.use_input_files): auto | true (fail if 2-3 find nothing) | false
    Returns (source description or None, grid info or None).
    """
    if explicit_ref:
        info = _georef_info(explicit_ref)
        if info is None:
            raise ValueError(f"reference GeoTIFF has no georeferencing: {explicit_ref}")
        return explicit_ref, info

    mode = str(mode).lower()
    if mode in ("false", "no", "off", "none"):
        return None, None
    candidates = fname if isinstance(fname, (list, tuple)) else [fname]
    for cand in candidates:
        info = _georef_info(cand)
        if info is not None:
            return cand, info
    if sf_yaml is not None:
        info = _geo_reader_grid(fname, sf_yaml, georef_cfg or {})
        if info is not None:
            kind = "affine" if "gcps" not in info else "gcps"
            return f"geo_reader:{sf_yaml['data']['geo_reader_type']}:{kind}", info
    if mode in ("true", "yes", "on", "require"):
        raise ValueError(f"No georeferenced input file found for {fname}")
    return None, None


def georeference(stack: np.ndarray, grid: dict | None, subset_inds, base: str = ""):
    """Place stack into the reference grid (as generate_cluster_geotiffs does)."""
    from rasterio.transform import Affine

    if grid is None:
        H, W = stack.shape[1:]
        profile = {"height": H, "width": W, "transform": Affine.identity(), "crs": None}
        return stack, profile

    profile = dict(grid)
    H, W = profile["height"], profile["width"]
    full = np.full((stack.shape[0], H, W), HEAD_NODATA, dtype=np.float32)
    if subset_inds:
        r0, r1, c0, c1 = map(int, subset_inds)
        h, w = min(r1 - r0, stack.shape[1], H - r0), min(c1 - c0, stack.shape[2], W - c0)
        full[:, r0:r0 + h, c0:c0 + w] = stack[:, :h, :w]
    else:
        # SIT-FUSE no-geo outputs carry 1 px of padding; larger mismatches
        # usually mean the wrong grid or missing subset_inds.
        dh, dw = stack.shape[1] - H, stack.shape[2] - W
        if dh < 0 or dw < 0 or dh > 1 or dw > 1:
            LOG.warning("%s: output %s vs grid %s; check subset_inds / reference grid",
                        base, stack.shape[1:], (H, W))
        h, w = min(H, stack.shape[1]), min(W, stack.shape[2])
        full[:, :h, :w] = stack[:, :h, :w]
    return full, profile


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def select_input_files(cfg: dict, sf_yaml: dict) -> list:
    """
    Input entries to export, in priority order:
      files          explicit list
      matchup_file   sit_fuse_input of matchup records (optionally only
                     matchup_splits, e.g. [dev, calibration, holdout]); covers
                     scenes from both data.files_train and data.files_test
      input_lists    SIT-FUSE YAML lists, default [files_test]
    """
    if cfg.get("files"):
        return list(cfg["files"])
    if cfg.get("matchup_file"):
        recs = read_yaml(cfg["matchup_file"]).get("matchups", [])
        keep = set(cfg["matchup_splits"]) if cfg.get("matchup_splits") else None
        files, seen = [], set()
        for r in recs:
            if keep is not None and r.get("classifier_split") not in keep:
                continue
            entry = r.get("sit_fuse_input")
            if entry is None:
                raise KeyError("Matchup records lack sit_fuse_input; enable encoder_split in the generator.")
            key = str(entry)
            if key not in seen:
                seen.add(key)
                files.append(entry)
        LOG.info("Exporting %d inputs referenced by %s", len(files), cfg["matchup_file"])
        return files
    files = []
    for key in cfg.get("input_lists", ["files_test"]):
        files.extend(sf_yaml["data"].get(key) or [])
    return files


def run(cfg: dict) -> None:
    sf_yaml = read_yaml(cfg["sit_fuse_yaml"])
    ex = cfg.get("head_export", {})
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = prep_model(sf_yaml)
    model.eval()
    device = next(model.pretrained_model.parameters()).device

    files = select_input_files(cfg, sf_yaml)
    georef_cfg = cfg.get("georef", {}) or {}
    refs = georef_cfg.get("reference_geotiffs") or [None] * len(files)
    subsets = georef_cfg.get("subset_inds") or [[]] * len(files)
    use_inputs = georef_cfg.get("use_input_files", "auto")
    if not (len(files) == len(refs) == len(subsets)):
        raise ValueError("files, georef.reference_geotiffs and georef.subset_inds must align.")

    tiled = bool(sf_yaml["data"].get("tile", False))
    batch = int(ex.get("batch_size", 5 if tiled else 10000))
    Kp, Kc, R = int(ex.get("top_k_parent", 5)), int(ex.get("top_k_child", 3)), int(ex.get("n_parent_ranks", 2))
    names = head_band_names(Kp, Kc, R)

    for fname, ref_tif, sub in zip(files, refs, subsets):
        base = os.path.basename(fname[0] if isinstance(fname, list) else fname)
        data, _ = get_prediction_dataset(sf_yaml, fname)
        if data is None or data.data_full is None:
            LOG.warning("Skipping %s: no valid samples", base)
            continue

        loader = DataLoader(data, batch_size=batch, shuffle=False, num_workers=0, drop_last=False)
        chunks, tile_hw = [], None
        for xb in loader:
            x = xb[0].to(device=device, dtype=torch.float32)
            rows = head_distributions(model, x, ex)
            if rows is None:
                raise RuntimeError(f"Heir_DC.forward returned None for a batch of {base}")
            if x.ndim == 4:
                tile_hw = (x.shape[2], x.shape[3])
            chunks.append(rows)
        rows = np.concatenate(chunks, axis=0)

        coord = np.asarray(data.targets_full)
        stack = assemble_scene(coord, rows, tile_hw)
        geo_src, grid = resolve_georef_source(fname, ref_tif, use_inputs, sf_yaml, georef_cfg)
        stack, profile = georeference(stack, grid, sub, base)
        LOG.info("%s: georeference source = %s", base, geo_src or "none (no-geo)")

        tags = {
            "format_version": FORMAT_VERSION,
            "num_child_classes": int(model.num_classes),
            "num_parent_classes": int(ex["_num_parent_classes"]),
            "top_k_parent": Kp, "top_k_child": Kc, "n_parent_ranks": R,
            "sit_fuse_yaml": cfg["sit_fuse_yaml"],
            "source_file": str(fname),
            "georeferenced": geo_src is not None,
            "georef_source": geo_src or "none",
        }
        out_path = out_dir / f"{base}.head_outputs.tif"
        write_band_stack(out_path, dict(zip(names, stack)), profile, tags, nodata=HEAD_NODATA)
        LOG.info("Wrote %s (%d bands)", out_path, len(names))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("yaml_config", type=Path)
    args = ap.parse_args()
    run(read_yaml(args.yaml_config))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
