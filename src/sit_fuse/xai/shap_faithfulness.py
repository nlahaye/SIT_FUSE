"""
Faithfulness and stability checks for the cluster SHAP attributions written by xai.py.

1. Deletion / insertion curves: remove (or add) players in order of |SHAP| and track f_c = log p(c | x) under the same
   background; compare with random orders. A faithful attribution drops f_c faster than random (deletion AUC lower,
   insertion AUC higher).
2. Rank stability: Spearman / Kendall / top-k Jaccard of the per-cluster mean |SHAP| ranking across backgrounds,
   seeds and sample sizes.
3. Model-based permutation importance per cluster (drop in mean f_c when a player's inputs are permuted across the
   pool) and its rank agreement with SHAP.

Usage:
  python -m sit_fuse.xai.shap_faithfulness -y <sit-fuse yaml>
  (reads xai.* options and xai.faithfulness.* : n_random_orders, stability_backgrounds, stability_seeds,
   stability_sample_sizes, n_permutations, top_k)
"""
import argparse
import os

import numpy as np

from sit_fuse.xai import xai as X


def _norm_curve(curve, base, fx):
    den = np.where(np.abs(fx - base) > 1e-12, fx - base, np.nan)
    return (curve - base[:, None]) / den[:, None]


def deletion_insertion(f, x, phi, bg, bg_w, player_idx, n_random=10, seed=0):
    """Curves (N, G+1) of f with the top-k players removed (deletion) or kept (insertion), plus random baselines.

    All curves are normalised to 1 at the full input and 0 at the background expectation, so AUCs are comparable
    across samples and clusters.
    """
    n, g = phi.shape
    rng = np.random.default_rng(seed)

    def curves(order):
        masks_del = np.ones((n, g + 1, g), dtype=bool)
        masks_ins = np.zeros((n, g + 1, g), dtype=bool)
        for k in range(1, g + 1):
            rows = np.arange(n)[:, None]
            masks_del[rows, k:, order[:, k - 1:k]] = False
            masks_ins[rows, k:, order[:, k - 1:k]] = True
        dele = np.empty((n, g + 1))
        ins = np.empty((n, g + 1))
        for i in range(n):
            vd = X.coalition_values(f, x[i:i + 1], bg, bg_w, player_idx, masks_del[i])[0]
            vi = X.coalition_values(f, x[i:i + 1], bg, bg_w, player_idx, masks_ins[i])[0]
            dele[i], ins[i] = vd, vi
        return dele, ins

    order = np.argsort(-np.abs(phi), axis=1)
    d, i = curves(order)
    base, fx = d[:, -1], d[:, 0]
    d, i = _norm_curve(d, base, fx), _norm_curve(i, base, fx)
    rd, ri = [], []
    for _ in range(n_random):
        o = np.stack([rng.permutation(g) for _ in range(n)])
        a, b = curves(o)
        rd.append(_norm_curve(a, base, fx))
        ri.append(_norm_curve(b, base, fx))
    auc = lambda c: np.nanmean(np.trapezoid(c, dx=1.0 / g, axis=1)) if hasattr(np, "trapezoid") \
        else np.nanmean(np.trapz(c, dx=1.0 / g, axis=1))
    return dict(deletion=np.nanmean(d, 0), insertion=np.nanmean(i, 0),
                deletion_random=np.nanmean(np.stack(rd), (0, 1)), insertion_random=np.nanmean(np.stack(ri), (0, 1)),
                auc_deletion=auc(d), auc_insertion=auc(i),
                auc_deletion_random=float(np.mean([auc(c) for c in rd])),
                auc_insertion_random=float(np.mean([auc(c) for c in ri])))


def rank_agreement(a, b, top_k=3):
    from scipy.stats import spearmanr, kendalltau
    a, b = np.asarray(a, float), np.asarray(b, float)
    rho = spearmanr(a, b).correlation if np.std(a) > 0 and np.std(b) > 0 else np.nan
    tau = kendalltau(a, b).correlation if np.std(a) > 0 and np.std(b) > 0 else np.nan
    k = min(top_k, len(a))
    ta, tb = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    return dict(spearman=rho, kendall=tau, topk_jaccard=len(ta & tb) / len(ta | tb))


def permutation_importance(f, x_cluster, pool, player_idx, n_perm=5, seed=0):
    """Mean drop in f over x_cluster when each player's inputs are replaced by the same player's inputs from random pool rows."""
    rng = np.random.default_rng(seed)
    f0 = f(x_cluster).mean()
    imp = np.zeros(len(player_idx))
    for j, idx in enumerate(player_idx):
        drops = []
        for _ in range(n_perm):
            xp = x_cluster.copy()
            xp[:, idx] = pool[rng.integers(0, len(pool), len(xp))][:, idx]
            drops.append(f0 - f(xp).mean())
        imp[j] = np.mean(drops)
    return imp


def run(yml_conf, model, device, data, options, out_dir):
    import pandas as pd
    fconf = (yml_conf.get("xai", {}) or {}).get("faithfulness", {}) or {}
    n_random = int(fconf.get("n_random_orders", 10))
    n_curve = int(fconf.get("n_curve_samples", 50))
    top_k = int(fconf.get("top_k", 3))
    n_perm = int(fconf.get("n_permutations", 5))
    backgrounds = fconf.get("stability_backgrounds", ["kmeans", "sample", "zero"])
    seeds = [int(s) for s in fconf.get("stability_seeds", [0, 1, 2])]
    sizes = [int(s) for s in fconf.get("stability_sample_sizes", [50, 100, 200])]

    n_ch = int(yml_conf["data"]["number_channels"])
    tile = 2 * int(yml_conf["data"].get("pixel_padding", 0)) + 1
    names, pidx = X.build_players(data.shape[1], n_ch, tile, options["attribution_level"],
                                  options["feature_names"], options["feature_groups"])
    labels = X.make_batched_output_fn(model, device, options["batch_size"])(data).reshape(-1).astype(np.int64)
    rng = np.random.default_rng(options["seed"])
    pool = data[rng.choice(len(data), size=min(options["background_pool"], len(data)), replace=False)]

    def explain(bg_type, seed, n_samples):
        o = dict(options, background=bg_type, seed=seed, max_samples_per_cluster=n_samples)
        bg, w = X.get_background(bg_type, pool, o["n_background"], seed)
        return X.explain_clusters(model, device, data, labels, o, names, pidx, bg, w, options["clusters"]), bg, w

    ref, bg, w = explain(options["background"], options["seed"], options["max_samples_per_cluster"])
    curve_rows, perm_rows = [], []
    curves_out = {}
    for c, r in ref.items():
        f = X.make_label_prob_fn(model, device, c, options["batch_size"], log=options["output_mode"] != "assigned_prob")
        sub = r["index"][:n_curve]
        res = deletion_insertion(f, data[sub], r["phi"][:len(sub)], bg, w, pidx, n_random, options["seed"])
        curves_out.update({f"{k}_{c}": v for k, v in res.items() if isinstance(v, np.ndarray)})
        curve_rows.append(dict(cluster=c, **{k: v for k, v in res.items() if not isinstance(v, np.ndarray)},
                               deletion_gain=res["auc_deletion_random"] - res["auc_deletion"],
                               insertion_gain=res["auc_insertion"] - res["auc_insertion_random"]))
        imp = permutation_importance(f, data[r["index"]], pool, pidx, n_perm, options["seed"])
        perm_rows.append(dict(cluster=c, **rank_agreement(np.abs(r["phi"]).mean(0), imp, top_k),
                              **{f"perm_{n}": v for n, v in zip(names, imp)}))
    tag = f"{options['output_mode']}_{options['attribution_level']}"
    pd.DataFrame(curve_rows).to_csv(os.path.join(out_dir, f"faithfulness_deletion_insertion_{tag}.csv"), index=False)
    pd.DataFrame(perm_rows).to_csv(os.path.join(out_dir, f"faithfulness_permutation_{tag}.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, f"faithfulness_curves_{tag}.npz"), **curves_out)

    # Stability of mean |SHAP| rankings relative to the reference run.
    stab_rows = []
    variants = [("background", b, options["seed"], options["max_samples_per_cluster"]) for b in backgrounds
                if b != options["background"]]
    variants += [("seed", options["background"], s, options["max_samples_per_cluster"]) for s in seeds
                 if s != options["seed"]]
    variants += [("sample_size", options["background"], options["seed"], n) for n in sizes
                 if n != options["max_samples_per_cluster"]]
    for kind, b, s, n in variants:
        alt, _, _ = explain(b, s, n)
        for c in ref:
            if c in alt:
                stab_rows.append(dict(cluster=c, variant=kind, background=b, seed=s, n_samples=n,
                                      **rank_agreement(np.abs(ref[c]["phi"]).mean(0),
                                                       np.abs(alt[c]["phi"]).mean(0), top_k)))
    stab = pd.DataFrame(stab_rows)
    stab.to_csv(os.path.join(out_dir, f"shap_rank_stability_{tag}.csv"), index=False)
    if len(stab):
        stab.groupby("variant")[["spearman", "kendall", "topk_jaccard"]].describe().to_csv(
            os.path.join(out_dir, f"shap_rank_stability_summary_{tag}.csv"))
    plot_curves(curve_rows, curves_out, os.path.join(out_dir, f"faithfulness_curves_{tag}.png"))
    return curve_rows, perm_rows, stab_rows


def plot_curves(rows, curves, out_png):
    import matplotlib.pyplot as plt
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for key, ax, title in (("deletion", axes[0], "Deletion"), ("insertion", axes[1], "Insertion")):
        shap_c = np.nanmean([curves[f"{key}_{r['cluster']}"] for r in rows], 0)
        rand_c = np.nanmean([curves[f"{key}_random_{r['cluster']}"] for r in rows], 0)
        k = np.arange(len(shap_c))
        ax.plot(k, shap_c, label="SHAP order")
        ax.plot(k, rand_c, "--", label="random order")
        ax.set_xlabel("players removed" if key == "deletion" else "players added")
        ax.set_ylabel("normalised f_c")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main(**kwargs):
    from sit_fuse.utils import read_yaml
    yml_conf = read_yaml(kwargs["yaml"])
    options = X.xai_options(yml_conf, **kwargs)
    if options["output_mode"] == "label":
        raise ValueError("faithfulness checks need a probabilistic output_mode")
    out_dir = os.path.join(yml_conf["output"]["out_dir"], "xai")
    os.makedirs(out_dir, exist_ok=True)
    data, _, model, device = X.load_inputs_and_model(yml_conf, options)
    run(yml_conf, model, device, data, options, out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-y", "--yaml", required=True)
    p.add_argument("--attribution-level", dest="attribution_level", choices=X.LEVELS)
    p.add_argument("--max-clusters", dest="max_clusters", type=int)
    main(**vars(p.parse_args()))
