"""
Copyright [2022-26], by the California Institute of Technology and Chapman University.
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.

Memory-efficient sample extraction and multi-scene sampling for SFDataset.

The legacy loader read every scene into memory, materialised every pixel
neighbourhood (view_as_windows -> reshape copies (2p+1)^2 x the scene), deleted
fill-containing rows, concatenated scene-by-scene, shuffled through Python
lists, and only then subset / stratified. This module instead works one scene
at a time:

  * valid window centres come from an integral image of the bad-pixel mask, so
    only windows that are actually kept are ever copied;
  * prediction / full-set builds keep the legacy row order, values and targets;
  * training subsets are drawn while scenes stream through, so peak memory is
    bounded by the candidate pool (~candidate_factor x subset_training samples)
    instead of the full set of windows from all scenes;
  * the scaler is partial_fit on exactly the same pixels, in the same order, as
    before, and applied once to the selected samples.

Sampling across scenes is weighted reservoir sampling (Efraimidis-Spirakis
keys, log(u) / w), drawn per scene with exact order statistics, so a scene with
n valid pixels costs O(k) random numbers for the k candidates it can contribute.

Config (all optional) lives in ``data.sampling``; see DEFAULTS.
"""
import glob
import math
import os
import re

import numpy as np
from skimage.util import view_as_windows

FILL = -999999
_EPS = 0.00000000005

DEFAULTS = {
    "seed": None,                  # int for reproducible subsets / shuffles
    "n_jobs": 1,                   # scenes read + preprocessed concurrently (threads)
    "scene_balance": 0.0,          # 0 = every valid pixel equally likely (legacy),
                                   # 1 = every scene contributes ~equally, in between = mix
    "group_by": None,              # regex on file basename; balance groups (e.g. date/sensor)
    "groups": None,                # {basename or path: group} (e.g. written by scene_split); overrides group_by
    "spatial_block": None,         # int or [rows, cols]: block size (window-centre grid) for spatial balance
    "spatial_balance": 0.0,        # 0 = blocks weighted by valid pixels (legacy), 1 = every block ~equal
    "cache_train": False,          # save train_data.npy / .indices.npy / .files.json after the first build,
                                   # so later stages (DC, child heads) and seed variants reuse the same sample
    "max_samples_per_scene": None, # hard cap on candidates from one scene
    "candidate_factor": 2.0,       # k-means candidate pool = factor x subset_training
    "kmeans_fit_samples": None,    # fit k-means on at most this many candidates
    "n_clusters": 40,              # k-means strata (legacy value)
    "allocation": "proportional",  # proportional (legacy) | sqrt | equal
    "max_scenes": None,            # randomly use at most this many training scenes
    "chunk_size": 1000000,         # rows per scaler.transform / k-means predict call
    "mmap": False,                 # memory-map train_data.npy in read_data_preprocessed
    "legacy": False,               # use the original (pre-optimisation) code path
}


def sampling_conf(conf=None):
    out = dict(DEFAULTS)
    if conf:
        out.update({k: v for k, v in dict(conf).items() if v is not None or k in ("seed",)})
    return out


def base_seed(conf):
    seed = conf.get("seed")
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 63))
        conf["seed"] = seed
    return int(seed)


# --------------------------------------------------------------------------- #
# Per-scene preprocessing (identical semantics to the legacy loader)
# --------------------------------------------------------------------------- #
def preprocess_read(dat, chan_dim, delete_chans, valid_min, valid_max, fill_value,
                    transform_chans=(), transform_values=()):
    """Return (list of (H, W, C) float32 scenes, increment) with bad values == FILL."""
    dat = np.asarray(dat)
    if dat.dtype != np.float32 or not dat.flags.writeable:
        dat = dat.astype(np.float32)
    if dat.ndim == 2:
        dat = np.expand_dims(dat, chan_dim)
    inc = dat.ndim - 3
    cd = chan_dim + inc

    for t in range(len(transform_chans or [])):
        slc = [slice(None)] * dat.ndim
        slc[cd] = slice(transform_chans[t], transform_chans[t] + 1)
        tmp = dat[tuple(slc)]  # view
        if valid_min is not None:
            tmp[tmp < valid_min - _EPS] = transform_values[t]
        if valid_max is not None:
            tmp[tmp > valid_max - _EPS] = transform_values[t]

    if delete_chans is not None and len(delete_chans) > 0:
        dat = np.delete(dat, delete_chans, cd)

    bad = ~np.isfinite(dat)
    if valid_min is not None:
        bad |= dat < valid_min - _EPS
    if valid_max is not None:
        bad |= dat > valid_max - _EPS
    if fill_value is not None:
        bad |= dat == fill_value
    dat[bad] = FILL
    del bad

    dat = np.moveaxis(dat, cd, 2 + inc)
    if dat.ndim == 3:
        return [np.ascontiguousarray(dat)], inc
    return [np.ascontiguousarray(d) for d in dat], inc


def bad_pixel_mask(scene, fill_mode="eq"):
    """(H, W) mask of pixels that disqualify any window containing them."""
    if fill_mode == "lt9000":   # scene_arr path with scaling: values < -9000 become fill
        return (scene < -9000).any(axis=2)
    return (scene == FILL).any(axis=2)


def scaler_pixel_mask(scene):
    """Legacy scaler-fit predicate: min over channels > -9000."""
    return scene.min(axis=2) > -9000


def valid_center_mask(bad, p):
    """Mask over the (H-2p, W-2p) grid of window centres with no bad pixel in the window."""
    H, W = bad.shape
    s = 2 * p + 1
    if H < s or W < s:
        return np.zeros((max(H - 2 * p, 0), max(W - 2 * p, 0)), dtype=bool)
    if p == 0:
        return ~bad
    dt = np.int32 if H * W < 2 ** 31 - 1 else np.int64
    ii = np.zeros((H + 1, W + 1), dtype=dt)
    np.cumsum(np.cumsum(bad, axis=0, dtype=dt), axis=1, out=ii[1:, 1:])
    win = ii[s:, s:] - ii[:-s, s:] - ii[s:, :-s] + ii[:-s, :-s]
    return win == 0


def extract_windows(scene, rows, cols, p):
    """Flattened (wy, wx, C) windows for centres (rows + p, cols + p); legacy feature order."""
    if p == 0:
        return scene[rows, cols, :]
    s = 2 * p + 1
    v = view_as_windows(scene, (s, s, scene.shape[2]))
    return v[rows, cols, 0].reshape(len(rows), -1)


def crop_labels(lab, p, out_shape):
    lab = np.asarray(lab)
    lab = lab[p:lab.shape[0] - p, p:lab.shape[1] - p]
    return lab[:out_shape[0], :out_shape[1]]


def binarize_labels(lab):
    lab = np.asarray(lab).astype(np.int32)
    lab[lab < 0] = 0
    lab[lab > 0] = 1
    return lab


def transform_in_chunks(scaler, X, n_chan, chunk_rows):
    """In-place per-channel scaling of flattened windows (bitwise == scene-level scaling)."""
    if X.shape[0] == 0:
        return X
    flat = X.reshape(-1, n_chan)
    step = max(1, int(chunk_rows))
    for i in range(0, flat.shape[0], step):
        flat[i:i + step] = scaler.transform(flat[i:i + step])
    return X


def targets_dtype(max_val):
    return np.int16 if max_val < np.iinfo(np.int16).max else np.int32


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #
def top_keys(rng, n, k, w):
    """Top-k of n iid Efraimidis-Spirakis keys log(U)/w, descending, in O(k)."""
    if k <= 0:
        return np.empty(0, dtype=np.float64)
    e = rng.standard_exponential(k)
    logu = -np.cumsum(e / (n - np.arange(k, dtype=np.float64)))
    return logu / w


def draw_candidates(rng, n, k, w):
    """Uniform k-subset of range(n) (random order) with matching top-k keys."""
    k = int(min(k, n))
    if k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    idx = rng.choice(n, size=k, replace=False, shuffle=True)
    return idx, top_keys(rng, n, k, w)


def allocate(counts, total, mode="proportional", population=None):
    """
    Integer quotas summing to min(total, sum(counts)), capped at counts (water-filling).
    population: optional true stratum sizes used for the weights when `counts` are
    capped candidate pools (defaults to counts).
    """
    counts = np.asarray(counts, dtype=np.int64)
    pop = counts if population is None else np.asarray(population, dtype=np.int64)
    q = np.zeros_like(counts)
    if total >= counts.sum():
        return counts.copy()
    if mode == "sqrt":
        w = np.sqrt(pop.astype(np.float64))
    elif mode == "equal":
        w = (pop > 0).astype(np.float64)
    else:
        w = pop.astype(np.float64)
    active = counts > 0
    remaining = int(total)
    while remaining > 0 and active.any():
        idx = np.nonzero(active)[0]
        share = remaining * w[idx] / w[idx].sum()
        cap = counts[idx] - q[idx]
        sat = share >= cap
        if sat.any():
            q[idx[sat]] = counts[idx[sat]]
            remaining -= int(cap[sat].sum())
            active[idx[sat]] = False
            continue
        fl = np.floor(share).astype(np.int64)
        q[idx] += fl
        rem = remaining - int(fl.sum())
        if rem > 0:
            order = np.argsort(-(share - fl), kind="stable")[:rem]
            q[idx[order]] += 1
        remaining = 0
    return q


def kmeans_strata(X, n_clusters=40, fit_samples=None, rng=None, chunk_rows=1000000):
    """Legacy MiniBatchKMeans settings; optional fit on a random subset; chunked predict."""
    from sklearn.cluster import MiniBatchKMeans
    n = X.shape[0]
    n_clusters = int(min(n_clusters, n))
    fit_X = X
    if fit_samples is not None and n > int(fit_samples):
        rng = rng if rng is not None else np.random.default_rng(0)
        fit_X = X[np.sort(rng.choice(n, size=int(fit_samples), replace=False))]
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, max_iter=50,
                         n_init="auto").fit(fit_X)
    lab = np.empty(n, dtype=np.int32)
    step = max(1, int(chunk_rows))
    for i in range(0, n, step):
        lab[i:i + step] = km.predict(X[i:i + step])
    return lab


def select_by_strata(keys, strata, total, mode, population=None):
    """
    Indices: per stratum, the highest-key items up to its allocated quota.
    population: optional {stratum: true size} for allocation weights.
    """
    uniq, inv = np.unique(strata, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq))
    pop = None
    if population is not None:
        pop = np.array([population.get(int(u), c) for u, c in zip(uniq, counts)], dtype=np.int64)
    quotas = allocate(counts, total, mode, pop)
    sel = []
    by_stratum = {}
    for j, u in enumerate(uniq):
        if quotas[j] == 0:
            continue
        members = np.nonzero(inv == j)[0]
        if quotas[j] < len(members):
            members = members[np.argsort(-keys[members], kind="stable")[:quotas[j]]]
        sel.append(members)
        by_stratum[u] = members
    if not sel:
        return np.empty(0, dtype=np.int64), {}
    return np.concatenate(sel), by_stratum


class CandidatePool(object):
    """Keeps the top-`cap` keyed candidates per stratum while scenes stream in."""

    def __init__(self, cap):
        self.cap = int(cap)
        self._parts = {}
        self._sizes = {}

    def add(self, stratum, keys, windows, targets):
        if len(keys) == 0:
            return
        self._parts.setdefault(stratum, []).append((keys, windows, targets))
        self._sizes[stratum] = self._sizes.get(stratum, 0) + len(keys)
        if self._sizes[stratum] > 2 * self.cap:
            self._compact(stratum)

    def _compact(self, stratum):
        parts = self._parts[stratum]
        keys = np.concatenate([p[0] for p in parts])
        win = np.concatenate([p[1] for p in parts])
        tgt = np.concatenate([p[2] for p in parts])
        if len(keys) > self.cap:
            keep = np.argpartition(-keys, self.cap - 1)[:self.cap]
            keep.sort()
            keys, win, tgt = keys[keep], win[keep], tgt[keep]
        self._parts[stratum] = [(keys, win, tgt)]
        self._sizes[stratum] = len(keys)

    def finalize(self):
        out = {}
        for s in list(self._parts.keys()):
            self._compact(s)
            out[s] = self._parts.pop(s)[0]
        return out


# --------------------------------------------------------------------------- #
# File list helpers
# --------------------------------------------------------------------------- #
def expand_filenames(filenames):
    out = []
    for f in filenames:
        if isinstance(f, str) and any(ch in f for ch in "*?["):
            out.extend(sorted(glob.glob(f)))
        else:
            out.append(f)
    return out


def _basename(f):
    while isinstance(f, (list, tuple)) and len(f) > 0:
        f = f[0]
    return os.path.basename(str(f))


def _group_from_map(groups, f):
    """Look a file up in a {path or basename: group} map (path, then basename, then basename sans ext)."""
    for k in (str(f) if not isinstance(f, (list, tuple)) else None, _basename(f)):
        if k is not None and k in groups:
            return str(groups[k])
    b = _basename(f)
    stem = b.split(".")[0]
    if stem in groups:
        return str(groups[stem])
    return None


def scene_groups(filenames, group_by, groups=None):
    """
    Group id per file. `groups` ({path or basename: group}, e.g. strata from scene_split) wins;
    otherwise regex `group_by` on the basename (group 1, else whole match, else basename).
    Files missing from `groups` form their own group "__other__".
    """
    if groups:
        names = []
        for f in filenames:
            g = _group_from_map(groups, f)
            names.append("__other__" if g is None else g)
        uniq, inv = np.unique(np.array(names, dtype=object).astype(str), return_inverse=True)
        return inv, np.bincount(inv, minlength=len(uniq))
    if not group_by:
        return np.zeros(len(filenames), dtype=np.int64), np.array([len(filenames)])
    pat = re.compile(group_by)
    names = []
    for f in filenames:
        b = _basename(f)
        m = pat.search(b)
        names.append((m.group(1) if m.groups() else m.group(0)) if m else b)
    uniq, inv = np.unique(np.array(names, dtype=object).astype(str), return_inverse=True)
    return inv, np.bincount(inv, minlength=len(uniq))


# --------------------------------------------------------------------------- #
# Spatial balance within scenes
# --------------------------------------------------------------------------- #
def block_shape(spatial_block):
    if spatial_block is None:
        return None
    if isinstance(spatial_block, (list, tuple)):
        by, bx = int(spatial_block[0]), int(spatial_block[1])
    else:
        by = bx = int(spatial_block)
    if by <= 0 or bx <= 0:
        return None
    return by, bx


def block_ids(flat, grid_shape, blk):
    """Spatial block id of each flat index on the (H, W) window-centre grid."""
    rows, cols = np.divmod(flat, grid_shape[1])
    nbx = -(-grid_shape[1] // blk[1])
    return (rows // blk[0]) * nbx + (cols // blk[1])


def spatial_candidates(rng, flat, grid_shape, blk, beta, scene_mass, k):
    """
    Weighted candidates over one scene's valid flat indices, balanced across spatial blocks.

    The scene's total sampling mass `scene_mass` is split across its blocks in proportion to
    n_b ** (1 - beta) (beta = 0: proportional to valid pixels, i.e. the legacy uniform weighting;
    beta = 1: every block gets the same mass). Each pixel in block b gets weight
    scene_mass * share_b / n_b, and Efraimidis-Spirakis keys are drawn per block (O(k_b)).
    Because the block grid is the same for every scene on a common grid, block ids also index
    the same locations across time. Returns (indices into flat, keys), at most k items.
    """
    n = len(flat)
    k = int(min(k, n))
    if k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    bid = block_ids(flat, grid_shape, blk)
    ub, inv, cnt = np.unique(bid, return_inverse=True, return_counts=True)
    share = cnt.astype(np.float64) ** (1.0 - float(beta))
    share /= share.sum()
    order = np.argsort(inv, kind="stable")
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    picks, keys = [], []
    for j in range(len(ub)):
        members = order[starts[j]:starts[j] + cnt[j]]
        w = float(scene_mass) * share[j] / cnt[j]
        p, kk = draw_candidates(rng, len(members), min(k, len(members)), w)
        picks.append(members[p])
        keys.append(kk)
    picks = np.concatenate(picks)
    keys = np.concatenate(keys)
    if len(keys) > k:
        top = np.argpartition(-keys, k - 1)[:k]
        picks, keys = picks[top], keys[top]
    return picks, keys


def iterate_parallel(func, items, n_jobs):
    """Ordered results, computed with up to n_jobs threads (bounded look-ahead)."""
    n_jobs = int(n_jobs or 1)
    if n_jobs <= 1 or len(items) <= 1:
        for it in items:
            yield func(it)
        return
    from joblib import Parallel, delayed
    par = Parallel(n_jobs=n_jobs, backend="threading", return_as="generator",
                   pre_dispatch="n_jobs")
    for res in par(delayed(func)(it) for it in items):
        yield res
