"""
Stability of SIT-FUSE environmental clusters across training runs (seeds, subsets, architectures).

Every run provides one label raster per scene (same scenes, same order). For each pair of runs:
  ARI (Hubert & Arabie 1985), NMI, and Hungarian-matched per-cluster Jaccard overlap;
  optionally, distances between matched cluster centroids in raw-variable space (z-scored).

YAML:
  scenes: [scene_a.zarr, ...]          optional raw inputs for centroid matching (read with the SIT-FUSE config)
  sit_fuse_config: path.yaml           SIT-FUSE config used to read the raw inputs
  variable_names: [...]
  runs:
    seed0: [labels_scene_a.tif, ...]
    seed1: [...]
  label_scale: 1                       cluster id = rint(value * label_scale); 1000 for SIT-FUSE .no_geo.tif rasters
  max_pixels: 500000
  min_cluster_pixels: 50
  out_dir: /path/to/output
"""
import argparse
import itertools
import os

import numpy as np

from sit_fuse.xai.cluster_interpret import clean_labels, read_label_raster


def contingency(a, b):
    ua, ai = np.unique(a, return_inverse=True)
    ub, bi = np.unique(b, return_inverse=True)
    tab = np.zeros((len(ua), len(ub)), dtype=np.int64)
    np.add.at(tab, (ai, bi), 1)
    return tab, ua, ub


def hungarian_match(a, b, min_pixels=50):
    """One-to-one matching maximising overlap; returns rows (cluster_a, cluster_b, jaccard, n_a, n_b)."""
    from scipy.optimize import linear_sum_assignment
    tab, ua, ub = contingency(a, b)
    na, nb = tab.sum(1), tab.sum(0)
    jac = tab / np.maximum(na[:, None] + nb[None, :] - tab, 1)
    r, c = linear_sum_assignment(-jac)
    rows = [dict(cluster_a=int(ua[i]), cluster_b=int(ub[j]), jaccard=float(jac[i, j]), n_a=int(na[i]), n_b=int(nb[j]))
            for i, j in zip(r, c) if na[i] >= min_pixels]
    matched = {int(ua[i]) for i in r}
    rows += [dict(cluster_a=int(u), cluster_b=None, jaccard=0.0, n_a=int(n), n_b=0)
             for u, n in zip(ua, na) if int(u) not in matched and n >= min_pixels]
    return rows


def centroid_distances(Xp, a, b, matches):
    """z-scored Euclidean distance between the medians of matched clusters."""
    mu, sd = Xp.mean(0), Xp.std(0)
    sd[sd == 0] = 1
    Z = (Xp - mu) / sd
    out = []
    for m in matches:
        if m["cluster_b"] is None:
            continue
        ca, cb = Z[a == m["cluster_a"]], Z[b == m["cluster_b"]]
        if len(ca) and len(cb):
            out.append(dict(m, centroid_dist=float(np.linalg.norm(np.median(ca, 0) - np.median(cb, 0)))))
    return out


def pooled_labels(run_files, fill=-1, max_pixels=500000, seed=0, scenes=None, label_scale=1):
    """Pixels valid in every run (and finite in all raw channels if scenes are given), subsampled once for all runs."""
    names = list(run_files)
    n_sc = {len(v) for v in run_files.values()}
    if len(n_sc) != 1:
        raise ValueError("all runs must list the same number of scenes")
    per_run = {k: [] for k in names}
    raw = []
    for s in range(n_sc.pop()):
        labs = [clean_labels(read_label_raster(run_files[k][s]), fill) for k in names]
        if len({l.shape for l in labs}) != 1:
            raise ValueError(f"scene {s}: label shapes differ across runs")
        ok = np.all([np.isfinite(l) for l in labs], 0).reshape(-1)
        if scenes is not None:
            flat = scenes[s].reshape(scenes[s].shape[0], -1).T
            ok &= np.all(np.isfinite(flat), 1)
            raw.append(flat[ok])
        for k, l in zip(names, labs):
            v = l.reshape(-1)[ok]
            per_run[k].append(np.rint(v * label_scale).astype(np.int64) if label_scale != 1 else v.astype(np.int64))
    y = {k: np.concatenate(v) for k, v in per_run.items()}
    Xp = np.concatenate(raw) if scenes is not None else None
    n = len(next(iter(y.values())))
    idx = np.arange(n)
    if max_pixels and n > max_pixels:
        idx = np.sort(np.random.default_rng(seed).choice(n, max_pixels, replace=False))
    return {k: v[idx] for k, v in y.items()}, (Xp[idx] if Xp is not None else None)


def run(labels, out_dir, Xp=None, min_pixels=50):
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    os.makedirs(out_dir, exist_ok=True)
    pair_rows, match_rows = [], []
    for ka, kb in itertools.combinations(labels, 2):
        a, b = labels[ka], labels[kb]
        m = hungarian_match(a, b, min_pixels)
        if Xp is not None:
            dists = {(d["cluster_a"], d["cluster_b"]): d["centroid_dist"] for d in centroid_distances(Xp, a, b, m)}
            for r in m:
                r["centroid_dist"] = dists.get((r["cluster_a"], r["cluster_b"]), np.nan)
        w = np.array([r["n_a"] for r in m], float)
        jac = np.array([r["jaccard"] for r in m])
        pair_rows.append(dict(run_a=ka, run_b=kb, ari=adjusted_rand_score(a, b), nmi=normalized_mutual_info_score(a, b),
                              n_clusters_a=len(np.unique(a)), n_clusters_b=len(np.unique(b)),
                              mean_matched_jaccard=float(jac.mean()) if len(jac) else np.nan,
                              weighted_matched_jaccard=float((w * jac).sum() / w.sum()) if w.sum() else np.nan,
                              share_clusters_jaccard_gt_0_5=float(np.mean(jac > 0.5)) if len(jac) else np.nan,
                              n_pixels=len(a)))
        match_rows += [dict(run_a=ka, run_b=kb, **r) for r in m]
    pd.DataFrame(pair_rows).to_csv(os.path.join(out_dir, "cluster_stability_pairs.csv"), index=False)
    pd.DataFrame(match_rows).to_csv(os.path.join(out_dir, "cluster_stability_matches.csv"), index=False)
    if match_rows:
        mr = pd.DataFrame(match_rows)
        # Per reference cluster: how reproducible is it across the other runs?
        mr.groupby(["run_a", "cluster_a"])["jaccard"].agg(["mean", "min", "count"]).reset_index().to_csv(
            os.path.join(out_dir, "cluster_stability_per_cluster.csv"), index=False)
    return pair_rows, match_rows


def main(path):
    import yaml
    with open(path) as f:
        conf = yaml.safe_load(f)
    scenes = None
    if conf.get("scenes"):
        from sit_fuse.utils import read_yaml
        from sit_fuse.xai.cluster_interpret import read_raw_scene
        sconf = read_yaml(conf["sit_fuse_config"])
        scenes = [read_raw_scene(sconf, s) for s in conf["scenes"]]
    labels, Xp = pooled_labels(conf["runs"], conf.get("label_fill", -1), conf.get("max_pixels", 500000),
                               conf.get("seed", 0), scenes, float(conf.get("label_scale", 1)))
    run(labels, conf["out_dir"], Xp, conf.get("min_cluster_pixels", 50))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-y", "--yaml", required=True)
    main(p.parse_args().yaml)
