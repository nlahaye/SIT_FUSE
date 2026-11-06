"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.

SHAP attributions for SIT-FUSE cluster assignments.

What is explained (xai.output_mode):
  "assigned_logprob" (default)  For every cluster c, f_c(x) = log p(c | x) evaluated on the samples assigned to c.
                                For Heir_DC, p(c | x) = p_top(t | x) * p_sub(s | t, x) with c = s + num_classes * t.
  "assigned_prob"               Same, on the probability scale.
  "label"                       Legacy: the integer label itself with an identity link. Labels are categorical, so the
                                attributions are not meaningful; kept only for comparison with earlier runs.

Players (xai.attribution_level):
  "channel" (default)           One player per input channel; all tile positions of a channel move together.
  "channel_centre_surround"     Two players per channel: the centre pixel and the surrounding tile.
  "group"                       User-defined groups of channels (xai.feature_groups), e.g. correlated variables.
  "feature"                     One player per flattened input (C * tile**2); summaries are aggregated back to channels.

Value function: interventional (baseline) Shapley values,
  v(S) = sum_b w_b f(x_S, b_notS),  with b from a weighted background (xai.background: "kmeans" | "sample" | "zero").
With <= xai.exact_max_players players the Shapley values are computed exactly by enumerating all coalitions;
otherwise shap.KernelExplainer is run on the same coalition game.
"""

# General Imports
import argparse
import hashlib
import itertools
import json
import math
import os
import warnings
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42

OUTPUT_MODES = ("assigned_logprob", "assigned_prob", "label")
LEVELS = ("channel", "channel_centre_surround", "group", "feature")
BACKGROUNDS = ("kmeans", "sample", "zero")
PROB_EPS = 1e-12


# ============================== OPTIONS ===============================

def xai_options(yml_conf, **overrides):
    """Integer options (validated) plus string/list options from the yml 'xai' block and CLI overrides."""
    int_defaults = {
        "max_samples_per_cluster": 200,
        "max_windows_per_cluster": 200,
        "batch_size": 512,
        "max_evals": None,
        "seed": SEED,
        "n_background": 50,
        "background_pool": 20000,
        "exact_max_players": 12,
        "max_clusters": 0,
        "min_cluster_samples": 10,
    }
    other_defaults = {
        "output_mode": "assigned_logprob",
        "attribution_level": "channel",
        "background": "kmeans",
        "feature_names": None,
        "feature_groups": None,
        "clusters": None,
        "data_split": "train",
    }
    xconf = yml_conf.get("xai", {}) or {}
    options = dict(int_defaults)
    options.update(other_defaults)
    options.update({k: v for k, v in xconf.items() if k in options})
    options.update({k: v for k, v in overrides.items() if k in options and v is not None})
    for key in int_defaults:
        if options[key] is not None:
            options[key] = int(options[key])
    for key in ("max_samples_per_cluster", "max_windows_per_cluster", "max_clusters"):
        if options[key] < 0:
            raise ValueError(f"xai.{key} must be nonnegative (0 means all).")
    for key in ("batch_size", "max_evals", "n_background", "background_pool", "exact_max_players"):
        if options[key] is not None and options[key] <= 0:
            raise ValueError(f"xai.{key} must be positive.")
    if options["seed"] < 0 or options["seed"] >= 2**32:
        raise ValueError("xai.seed must be in [0, 2**32).")
    if options["output_mode"] not in OUTPUT_MODES:
        raise ValueError(f"xai.output_mode must be one of {OUTPUT_MODES}.")
    if options["attribution_level"] not in LEVELS:
        raise ValueError(f"xai.attribution_level must be one of {LEVELS}.")
    if options["background"] not in BACKGROUNDS:
        raise ValueError(f"xai.background must be one of {BACKGROUNDS}.")
    if options["data_split"] not in ("train", "test"):
        raise ValueError("xai.data_split must be 'train' or 'test'.")
    if options["attribution_level"] == "group" and not options["feature_groups"]:
        raise ValueError("xai.attribution_level 'group' needs xai.feature_groups {group: [channel names]}.")
    return options


# ============================== MODEL WRAPPERS ===============================

def prepare_explanation_model(model, device):
    model.eval()
    # Hierarchical heads live in ordinary dictionaries, not registered modules.
    for tier in getattr(model, "clust_tree", {}).values():
        for head in tier.values():
            if isinstance(head, torch.nn.Module):
                head.to(device)
                head.eval()


def _first(out):
    while isinstance(out, (tuple, list)):
        out = out[0]
    return out


def is_hierarchical(model):
    return hasattr(model, "clust_tree") and "1" in getattr(model, "clust_tree", {})


def make_batched_output_fn(model, device, batch_size=512, temporal=False):
    """Hard labels (hierarchical labels for Heir_DC) or RTDBN probabilities, in bounded batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    prepare_explanation_model(model, device)

    def f(x):
        outputs = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                t = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size])).to(device)
                if temporal:
                    output = torch.nn.functional.softmax(model(t.float()), dim=1)
                elif is_hierarchical(model):
                    _, output, _, _ = model.forward(t, return_embed=False)
                else:
                    output = _first(model.forward(t))
                    if output.ndim == 2 and output.shape[1] > 1:
                        output = torch.argmax(output, dim=1, keepdim=True)
                outputs.append(output.detach().cpu().numpy())
        if not outputs:
            width = model.num_classes if temporal else 1
            return np.empty((0, width), dtype=np.float32)
        return np.concatenate(outputs, axis=0)
    return f


def _hier_embedding(model, x):
    """Embedding fed to the Heir_DC sub-heads, for the flat (N, features) encoders only."""
    enc_type = getattr(model, "encoder_type", None)
    if getattr(model, "encoder", None) is None:
        return x
    if enc_type == "pca":
        return torch.from_numpy(model.pretrained_model.pretrained_model(x.cpu().numpy())).type(x.dtype).to(x.device)
    if any(s in str(enc_type) for s in ("byol", "conv", "clay", "ijepa", "mae", "unet")):
        raise NotImplementedError(f"assigned-cluster probabilities not implemented for encoder_type={enc_type}; "
                                  "use xai.output_mode 'label'.")
    return model.encoder(x)  # same branch as Heir_DC.forward for flat (dbn / default) encoders


def make_label_prob_fn(model, device, label, batch_size=512, log=True):
    """f(x) -> (N,) [log] p(label | x).

    Non-hierarchical models: forward() returns the softmax over clusters.
    Heir_DC: label = sub + num_classes * top, p = p_top[top] * p_sub[sub] (p_sub = 1 if the top cluster has no sub-head).
    """
    prepare_explanation_model(model, device)
    label = int(label)
    hier = is_hierarchical(model)
    if hier:
        nc = int(model.num_classes)
        top, sub = label // nc, label % nc
        head = model.clust_tree["1"].get(str(top))
    else:
        top, sub, head = label, None, None

    def f(x):
        x = np.asarray(x, dtype=np.float32)
        out = np.empty(len(x), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                t = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size])).to(device)
                if hier:
                    y = _first(model.pretrained_model.forward(t))
                else:
                    y = _first(model.forward(t))
                if y.ndim != 2:
                    raise NotImplementedError("assigned-cluster probabilities need (N, clusters) outputs; "
                                              "use xai.output_mode 'label' for conv/tile models.")
                p = y[:, top].double()
                if hier and head is not None:
                    ps = _first(head.forward(_hier_embedding(model, t)))
                    p = p * ps[:, sub].double()
                out[start:start + len(t)] = p.cpu().numpy()
        return np.log(np.clip(out, PROB_EPS, None)) if log else out
    return f


def check_label_prob_fn(model, device, x, labels, batch_size=512, n_check=200):
    """p(assigned label | x) must be a valid probability and, for the top level, the assigned top cluster must be the argmax."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x), size=min(n_check, len(x)), replace=False)
    for lab in np.unique(labels[idx]):
        sel = idx[labels[idx] == lab]
        p = make_label_prob_fn(model, device, lab, batch_size, log=False)(x[sel])
        assert np.all(np.isfinite(p)) and np.all(p >= 0) and np.all(p <= 1 + 1e-6), f"bad probabilities for {lab}"
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(x[sel])).to(device)
            y = _first(model.pretrained_model.forward(t) if is_hierarchical(model) else model.forward(t))
        top = lab // int(model.num_classes) if is_hierarchical(model) else lab
        agree = float(np.mean(torch.argmax(y, dim=1).cpu().numpy() == top))
        assert agree > 0.99, f"label {lab}: top-level argmax agrees with the assigned label for only {agree:.2%}"


# ============================== PLAYERS ===============================

def build_players(n_inputs, n_channels, tile_size, level, feature_names=None, feature_groups=None):
    """Return (player_names, list of input-index arrays). Inputs are laid out as (channel, row, col) per sample."""
    if n_inputs != n_channels * tile_size ** 2:
        raise ValueError(f"{n_inputs} inputs != {n_channels} channels * {tile_size}^2")
    names = list(feature_names) if feature_names else [f"chan_{i}" for i in range(n_channels)]
    if len(names) != n_channels:
        raise ValueError(f"xai.feature_names has {len(names)} names for {n_channels} channels")
    per = tile_size ** 2
    chan_idx = [np.arange(c * per, (c + 1) * per) for c in range(n_channels)]
    centre = (tile_size // 2) * tile_size + tile_size // 2
    if level == "channel":
        return names, chan_idx
    if level == "channel_centre_surround":
        if tile_size == 1:
            return names, chan_idx
        pn, pi = [], []
        for c, n in enumerate(names):
            pn += [f"{n}|centre", f"{n}|surround"]
            pi += [chan_idx[c][[centre]], np.delete(chan_idx[c], centre)]
        return pn, pi
    if level == "group":
        lookup = {n: c for c, n in enumerate(names)}
        used, pn, pi = set(), [], []
        for g, members in feature_groups.items():
            missing = [m for m in members if m not in lookup]
            if missing:
                raise ValueError(f"xai.feature_groups[{g}] has unknown channels {missing}")
            dup = used.intersection(members)
            if dup:
                raise ValueError(f"channels {sorted(dup)} appear in more than one group")
            used.update(members)
            pn.append(str(g))
            pi.append(np.concatenate([chan_idx[lookup[m]] for m in members]))
        for n in names:  # ungrouped channels are their own players
            if n not in used:
                pn.append(n)
                pi.append(chan_idx[lookup[n]])
        return pn, pi
    # "feature"
    pn = [f"{n}|r{r}c{c}" for n in names for r in range(tile_size) for c in range(tile_size)]
    return pn, [np.array([i]) for i in range(n_inputs)]


def channel_aggregate(phi, player_names):
    """Sum feature-level or centre/surround attributions to one value per channel name (exact by additivity)."""
    chans = []
    for n in player_names:
        base = n.split("|")[0]
        if base not in chans:
            chans.append(base)
    out = np.zeros(phi.shape[:-1] + (len(chans),), dtype=phi.dtype)
    for j, n in enumerate(player_names):
        out[..., chans.index(n.split("|")[0])] += phi[..., j]
    return out, chans


# ============================== BACKGROUND ===============================

def get_background(background_type, data, n_samples=50, seed=SEED):
    """Weighted background (rows, weights). 'kmeans': weighted centroids; 'sample': random rows; 'zero': one zero row."""
    data = np.asarray(data, dtype=np.float32)
    if background_type == "zero":
        return np.zeros((1, data.shape[1]), dtype=np.float32), np.ones(1)
    rng = np.random.default_rng(seed)
    if background_type == "sample":
        idx = rng.choice(len(data), size=min(n_samples, len(data)), replace=False)
        return data[idx], np.full(len(idx), 1.0 / len(idx))
    if background_type == "kmeans":
        from sklearn.cluster import KMeans
        k = min(n_samples, len(data))
        km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(data)
        w = np.bincount(km.labels_, minlength=k).astype(np.float64)
        keep = w > 0
        return km.cluster_centers_[keep].astype(np.float32), w[keep] / w[keep].sum()
    raise ValueError(background_type)


# ============================== SHAPLEY ===============================

def coalition_values(f, x, bg, bg_w, player_idx, masks, chunk_rows=200000):
    """v[n, m] = sum_b w_b f(x_n on players in masks[m], background b elsewhere)."""
    n, d = x.shape
    nb = len(bg)
    v = np.empty((n, len(masks)))
    on = np.zeros((len(masks), d), dtype=bool)
    for m, mask in enumerate(masks):
        for j in np.flatnonzero(mask):
            on[m, player_idx[j]] = True
    per_mask = n * nb
    masks_per_chunk = max(1, chunk_rows // max(per_mask, 1))
    for m0 in range(0, len(masks), masks_per_chunk):
        sel = on[m0:m0 + masks_per_chunk]
        # (M, N, B, D)
        z = np.where(sel[:, None, None, :], x[None, :, None, :], bg[None, None, :, :])
        fz = f(z.reshape(-1, d)).reshape(len(sel), n, nb)
        v[:, m0:m0 + len(sel)] = (fz @ bg_w).T
    return v


def exact_shapley(f, x, bg, bg_w, player_idx):
    """Exact interventional Shapley values by enumerating all 2^G coalitions. Returns (phi (N, G), base (N,), fx (N,))."""
    g = len(player_idx)
    masks = np.array(list(itertools.product([0, 1], repeat=g)), dtype=bool)
    v = coalition_values(f, x, bg, bg_w, player_idx, masks)
    pos = {tuple(m): i for i, m in enumerate(masks)}
    sizes = masks.sum(1)
    w = np.array([math.factorial(s) * math.factorial(g - s - 1) / math.factorial(g) if s < g else 0.0
                  for s in sizes])
    phi = np.zeros((len(x), g))
    for i, m in enumerate(masks):
        for j in np.flatnonzero(~m):
            mj = m.copy()
            mj[j] = True
            phi[:, j] += w[i] * (v[:, pos[tuple(mj)]] - v[:, i])
    base, fx = v[:, pos[(False,) * g]], v[:, pos[(True,) * g]]
    return phi, base, fx


def kernel_shapley(f, x, bg, bg_w, player_idx, nsamples="auto", seed=SEED):
    """shap.KernelExplainer on the same coalition game (players switched jointly)."""
    import shap
    g = len(player_idx)
    phi = np.zeros((len(x), g))
    base = np.zeros(len(x))
    fx = np.zeros(len(x))
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        for n in range(len(x)):
            def game(z, _x=x[n:n + 1]):
                return coalition_values(f, _x, bg, bg_w, player_idx, z.astype(bool))[0]
            expl = shap.KernelExplainer(game, np.zeros((1, g)), link="identity")
            phi[n] = np.asarray(expl.shap_values(np.ones((1, g)), nsamples=nsamples, silent=True)).reshape(-1)
            base[n] = float(np.asarray(expl.expected_value).reshape(-1)[0])
            fx[n] = game(np.ones((1, g)))[0]
    finally:
        np.random.set_state(state)
    return phi, base, fx


def shapley(f, x, bg, bg_w, player_idx, exact_max_players=12, nsamples="auto", seed=SEED):
    if len(player_idx) <= exact_max_players:
        return exact_shapley(f, x, bg, bg_w, player_idx)
    return kernel_shapley(f, x, bg, bg_w, player_idx, nsamples=nsamples, seed=seed)


def sample_cluster_indices(labels, max_per_cluster, seed=SEED):
    """Keep rare clusters and deterministically sample larger clusters; 0 keeps all."""
    if max_per_cluster < 0:
        raise ValueError("max_per_cluster must be nonnegative.")
    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(labels):
        inds = np.flatnonzero(labels == label)
        if max_per_cluster and len(inds) > max_per_cluster:
            inds = rng.choice(inds, size=max_per_cluster, replace=False)
        selected.extend(inds)
    return np.sort(np.asarray(selected, dtype=np.int64))


def explain_clusters(model, device, X, labels, options, player_names, player_idx, bg, bg_w, clusters=None):
    """Per-cluster attributions of f_c on the samples assigned to c. Returns {cluster: dict(phi, base, fx, index)}."""
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    uniq, counts = np.unique(labels, return_counts=True)
    order = uniq[np.argsort(-counts)]
    if clusters is not None:
        order = [c for c in order if c in set(int(v) for v in clusters)]
    order = [c for c in order if np.sum(labels == c) >= options["min_cluster_samples"]]
    if options["max_clusters"]:
        order = order[:options["max_clusters"]]
    rng = np.random.default_rng(options["seed"])
    results = {}
    for c in order:
        inds = np.flatnonzero(labels == c)
        if options["max_samples_per_cluster"] and len(inds) > options["max_samples_per_cluster"]:
            inds = np.sort(rng.choice(inds, options["max_samples_per_cluster"], replace=False))
        log = options["output_mode"] == "assigned_logprob"
        f = make_label_prob_fn(model, device, c, options["batch_size"], log=log)
        phi, base, fx = shapley(f, X[inds], bg, bg_w, player_idx, options["exact_max_players"],
                                options["max_evals"] or "auto", options["seed"])
        # Efficiency: attributions must sum to f(x) - E_b f(b).
        gap = np.max(np.abs(phi.sum(1) - (fx - base))) if len(inds) else 0.0
        scale = max(1.0, float(np.max(np.abs(fx - base))) if len(inds) else 1.0)
        if gap > 1e-3 * scale and len(player_idx) <= options["exact_max_players"]:
            raise AssertionError(f"cluster {c}: Shapley efficiency violated (max gap {gap:.3g})")
        results[int(c)] = dict(phi=phi, base=base, fx=fx, index=inds, n_assigned=int(np.sum(labels == c)))
        print(f"cluster {c}: {len(inds)} samples, mean f = {fx.mean():.3f}, base = {base.mean():.3f}")
    return results


# ============================== OUTPUTS ===============================

def write_summaries(results, player_names, out_dir, tag):
    """mean |phi| and mean phi per cluster x player (and per cluster x channel if players are finer than channels)."""
    import pandas as pd
    rows_abs, rows_sgn = [], []
    for c, r in results.items():
        rows_abs.append(dict(cluster=c, n_assigned=r["n_assigned"], n_explained=len(r["index"]),
                             **dict(zip(player_names, np.abs(r["phi"]).mean(0)))))
        rows_sgn.append(dict(cluster=c, n_assigned=r["n_assigned"], n_explained=len(r["index"]),
                             **dict(zip(player_names, r["phi"].mean(0)))))
    paths = {}
    for name, rows in (("mean_abs_shap", rows_abs), ("mean_shap", rows_sgn)):
        p = os.path.join(out_dir, f"{name}_by_cluster_{tag}.csv")
        pd.DataFrame(rows).to_csv(p, index=False)
        paths[name] = p
    if any("|" in n for n in player_names):
        rows = []
        for c, r in results.items():
            agg, chans = channel_aggregate(r["phi"], player_names)
            rows.append(dict(cluster=c, **dict(zip(chans, np.abs(agg).mean(0)))))
        p = os.path.join(out_dir, f"mean_abs_shap_by_cluster_channel_{tag}.csv")
        pd.DataFrame(rows).to_csv(p, index=False)
        paths["channel"] = p
        if any(n.endswith("|centre") for n in player_names):
            rows = []
            for c, r in results.items():
                a = np.abs(r["phi"]).mean(0)
                cen = sum(a[j] for j, n in enumerate(player_names) if n.endswith("|centre"))
                sur = sum(a[j] for j, n in enumerate(player_names) if n.endswith("|surround"))
                rows.append(dict(cluster=c, centre=cen, surround=sur, surround_share=sur / max(cen + sur, 1e-12)))
            p = os.path.join(out_dir, f"centre_surround_by_cluster_{tag}.csv")
            pd.DataFrame(rows).to_csv(p, index=False)
            paths["centre_surround"] = p
    return paths


def player_values(X, player_idx):
    """Per-player feature value for plots: mean over the player's inputs."""
    return np.stack([X[:, idx].mean(1) for idx in player_idx], axis=1)


def save_summary_plot(out_dir, shap_values, X, label, feature_names, tag=""):
    import shap
    filename = f"shap_summary_plot_{label}{'_' + tag if tag else ''}.png"
    shap.plots.violin(shap_values, features=X, feature_names=list(feature_names), show=False)
    plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.close("all")


def save_heatmap(csv_path, out_png, title):
    import pandas as pd
    df = pd.read_csv(csv_path).set_index("cluster")
    df = df.drop(columns=[c for c in ("n_assigned", "n_explained") if c in df.columns])
    m = df.values / np.maximum(df.values.sum(1, keepdims=True), 1e-12)  # share of attribution per cluster
    fig, ax = plt.subplots(figsize=(1.0 + 0.45 * m.shape[1], 1.0 + 0.25 * m.shape[0]))
    im = ax.imshow(m, aspect="auto", cmap="viridis")
    ax.set_xticks(range(m.shape[1]), df.columns, rotation=90)
    ax.set_yticks(range(m.shape[0]), df.index)
    ax.set_xlabel("player")
    ax.set_ylabel("cluster")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="share of mean |SHAP|")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_results(results, player_names, X, player_idx, out_dir, tag, settings):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"shap_{tag}.npz"),
                        player_names=np.array(player_names),
                        clusters=np.array(sorted(results)),
                        **{f"phi_{c}": r["phi"] for c, r in results.items()},
                        **{f"base_{c}": r["base"] for c, r in results.items()},
                        **{f"fx_{c}": r["fx"] for c, r in results.items()},
                        **{f"index_{c}": r["index"] for c, r in results.items()})
    with open(os.path.join(out_dir, f"shap_{tag}_settings.json"), "w") as f:
        json.dump(settings, f, indent=1, default=str)
    paths = write_summaries(results, player_names, out_dir, tag)
    save_heatmap(paths["mean_abs_shap"], os.path.join(out_dir, f"mean_abs_shap_heatmap_{tag}.png"),
                 f"SHAP by cluster ({tag})")
    return paths


def load_results(path):
    z = np.load(path, allow_pickle=False)
    names = [str(s) for s in z["player_names"]]
    res = {int(c): dict(phi=z[f"phi_{c}"], base=z[f"base_{c}"], fx=z[f"fx_{c}"], index=z[f"index_{c}"])
           for c in z["clusters"]}
    return res, names


# ============================== MAIN ===============================

def checkpoint_inventory(yml_conf, out_dir):
    save_dir = out_dir
    logger = yml_conf.get("logger", {})
    if logger.get("use_wandb"):
        save_dir = os.path.join(save_dir, logger["log_out_dir"])
    checkpoints = {}
    for directory in ("encoder", "full_model", "full_model_heir"):
        for path in sorted(glob(os.path.join(save_dir, directory, "**", "*"), recursive=True)):
            if os.path.isfile(path) and path.endswith((".ckpt", ".pkl", ".pt", ".pth")):
                stat = os.stat(path)
                checkpoints[path] = [stat.st_size, stat.st_mtime_ns]
    return checkpoints


def load_inputs_and_model(yml_conf, options):
    from sit_fuse.inference.generate_output import get_model
    from sit_fuse.datasets.dataset_utils import get_prediction_dataset
    files = yml_conf["data"]["files_train" if options["data_split"] == "train" else "files_test"]
    data, _ = get_prediction_dataset(yml_conf, files)
    if len(data.data_full) == 0:
        raise ValueError("No samples available for SHAP.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(yml_conf, data.data_full.shape[1]).to(device)
    return np.asarray(data.data_full, dtype=np.float32), files, model, device


def run(yml_conf, model, device, X, files, options, out_dir):
    n_channels = int(yml_conf['data']['number_channels'])
    padding = int(yml_conf['data'].get('pixel_padding', 0))
    tile_size = 2 * padding + 1
    names, idx = build_players(X.shape[1], n_channels, tile_size, options["attribution_level"],
                               options["feature_names"], options["feature_groups"])
    labels = make_batched_output_fn(model, device, options["batch_size"])(X).reshape(-1).astype(np.int64)
    assert len(labels) == len(X)

    if options["output_mode"] == "label":
        warnings.warn("xai.output_mode 'label' explains a categorical label with an identity link; "
                      "attributions are not interpretable. Use 'assigned_logprob'.")
        rng = np.random.default_rng(options["seed"])
        pool = X[rng.choice(len(X), size=min(options["background_pool"], len(X)), replace=False)]
        bg, bg_w = get_background(options["background"], pool, options["n_background"], options["seed"])
        lab_fn = make_batched_output_fn(model, device, options["batch_size"])
        sel = sample_cluster_indices(labels, options["max_samples_per_cluster"], options["seed"])
        phi, base, fx = shapley(lambda z: lab_fn(z).reshape(-1).astype(np.float64), X[sel], bg, bg_w, idx,
                                options["exact_max_players"], options["max_evals"] or "auto", options["seed"])
        results = {int(c): dict(phi=phi[labels[sel] == c], base=base[labels[sel] == c], fx=fx[labels[sel] == c],
                                index=sel[labels[sel] == c], n_assigned=int(np.sum(labels == c)))
                   for c in np.unique(labels[sel])}
    else:
        check_label_prob_fn(model, device, X, labels, options["batch_size"])
        rng = np.random.default_rng(options["seed"])
        pool = X[rng.choice(len(X), size=min(options["background_pool"], len(X)), replace=False)]
        bg, bg_w = get_background(options["background"], pool, options["n_background"], options["seed"])
        results = explain_clusters(model, device, X, labels, options, names, idx, bg, bg_w, options["clusters"])

    tag = f"{options['output_mode']}_{options['attribution_level']}_{options['background']}_bg"
    settings = dict({k: v for k, v in options.items()}, input_shape=list(X.shape), input_files=list(files),
                    tag=tag, player_names=names, n_background_rows=int(len(bg)),
                    input_digest=hashlib.sha256(np.ascontiguousarray(X[:100000]).tobytes()).hexdigest(),
                    config_digest=hashlib.sha256(json.dumps(yml_conf, sort_keys=True, default=str).encode()).hexdigest(),
                    checkpoints=checkpoint_inventory(yml_conf, yml_conf['output']['out_dir']))
    np.save(os.path.join(out_dir, f"assigned_labels_{options['data_split']}.npy"), labels)
    np.savez_compressed(os.path.join(out_dir, f"background_{tag}.npz"), bg=bg, w=bg_w)
    paths = save_results(results, names, X, idx, out_dir, tag, settings)

    plot_dir = os.path.join(out_dir, "shap_plots")
    os.makedirs(plot_dir, exist_ok=True)
    try:
        for c, r in results.items():
            if len(r["index"]) > 1:
                save_summary_plot(plot_dir, r["phi"], player_values(X[r["index"]], idx), c, names, tag)
    except Exception as e:  # plotting must not discard computed values
        warnings.warn(f"summary plots failed: {e}")
    return results, names, paths


def main(**kwargs):
    if not kwargs.get('yaml'):
        raise Exception('No yaml path specified.')
    from sit_fuse.utils import read_yaml
    yml_conf = read_yaml(kwargs['yaml'])
    options = xai_options(yml_conf, **kwargs)
    out_dir = os.path.join(yml_conf['output']['out_dir'], "xai")
    os.makedirs(out_dir, exist_ok=True)
    X, files, model, device = load_inputs_and_model(yml_conf, options)
    run(yml_conf, model, device, X, files, options, out_dir)


try:
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
except ImportError:  # allows importing the pure-numpy helpers without torch
    torch = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", required=True, help="SIT-FUSE YAML file.")
    parser.add_argument("-e", "--max-evals", dest="max_evals", type=int,
                        help="KernelSHAP coalition samples when players > exact_max_players.")
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, help="Inference batch size (default 512).")
    parser.add_argument("--max-samples-per-cluster", dest="max_samples_per_cluster", type=int,
                        help="Samples explained per cluster (default 200; 0: all).")
    parser.add_argument("--max-windows-per-cluster", dest="max_windows_per_cluster", type=int)
    parser.add_argument("--seed", type=int, help="Sampling seed (default 42).")
    parser.add_argument("--output-mode", dest="output_mode", choices=OUTPUT_MODES)
    parser.add_argument("--attribution-level", dest="attribution_level", choices=LEVELS)
    parser.add_argument("--background", choices=BACKGROUNDS)
    parser.add_argument("--n-background", dest="n_background", type=int)
    parser.add_argument("--max-clusters", dest="max_clusters", type=int)
    parser.add_argument("--data-split", dest="data_split", choices=("train", "test"))
    args = parser.parse_args()
    from timeit import default_timer as timer
    start = timer()
    main(**vars(args))
    print(f"{timer() - start:.1f} s")
