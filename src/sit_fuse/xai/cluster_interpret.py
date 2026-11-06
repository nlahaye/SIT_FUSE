"""
Physical interpretation of SIT-FUSE environmental clusters from raw (unscaled) inputs and label rasters.

Products (all under <out_dir>/xai/interpret/):
  cluster_profiles_long.csv     cluster x variable: n, median, q25, q75, mean, std, z (cluster median vs scene-pool mean/std)
  cluster_profiles_wide.csv     cluster x variable medians (input to NatureNet reward_env_projection)
  cluster_profile_heatmap.png   z-score heatmap of the most frequent clusters
  cluster_occurrence.npz/.png   per-pixel occurrence frequency of the most frequent clusters
  cluster_monthly_frequency.csv share of labelled pixels per cluster and month (if dates are given)
  surrogate_metrics.csv         decision tree / gradient boosting from raw variables to cluster, GroupKFold by scene
  surrogate_tree_rules.txt      rules of a shallow tree fitted to all scenes
  external_enrichment.csv       per-cluster enrichment of SST fronts, Okubo-Weiss eddy cores and optional masks
  external_association.csv      NMI / Cramer's V between cluster labels and each external feature

YAML (xai.interpret block, in the SIT-FUSE config):
  label_files: [...]            one label raster per input file (same order as data.files_train or `data_files`)
  data_files:  [...]            optional; defaults to data.files_train
  variable_names: [...]         names of the input channels (defaults to xai.feature_names)
  dates: [...]                  optional ISO dates per scene for the monthly summary
  label_fill: -1                labels < 0, NaN or == label_fill are ignored
  n_top_clusters: 30            clusters kept by frequency (the rest are pooled as "other" in the surrogate)
  max_pixels_per_scene: 20000
  surrogate_depths: [3, 4, 6]
  n_splits: 5
  external:
    sst: "thetao"               channel used for fronts
    front_percentile: 90        gradient-magnitude percentile defining front pixels (per scene)
    u: "uo"                     channels for Okubo-Weiss
    v: "vo"
    ow_threshold: 0.2           eddy core where W < -ow_threshold * std(W)
    pixel_km: null              grid spacing in km (null: per-pixel units)
    masks: {upwelling: [...]}   optional binary rasters, one per scene
"""
import argparse
import os
import warnings

import numpy as np

ANCILLARY_DEFAULTS = dict(label_fill=-1, n_top_clusters=30, max_pixels_per_scene=20000, surrogate_depths=[3, 4, 6],
                          n_splits=5, min_cluster_pixels=50, seed=0)


# ============================== IO ===============================

def read_label_raster(path):
    if path.endswith(".npy"):
        return np.load(path).astype(np.float64)
    if path.endswith(".zarr"):
        import zarr
        return np.asarray(zarr.open(path, mode="r")).astype(np.float64)
    try:
        from osgeo import gdal
        ds = gdal.Open(path)
        return ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    except ImportError:
        import rasterio
        with rasterio.open(path) as ds:
            return ds.read(1).astype(np.float64)


def read_raw_scene(yml_conf, path):
    """Raw (unscaled) input channels as (C, H, W), invalid values set to NaN."""
    from sit_fuse.utils import get_read_func
    d = yml_conf["data"]
    arr = np.asarray(get_read_func(d["reader_type"])(path, **(d.get("reader_kwargs") or {})), dtype=np.float64)
    arr = np.moveaxis(arr, int(d.get("chan_dim", 0)), 0)
    if d.get("delete_chans"):
        arr = np.delete(arr, d["delete_chans"], axis=0)
    bad = ~np.isfinite(arr)
    if d.get("fill_value") is not None:
        bad |= arr == float(d["fill_value"])
    if d.get("valid_min") is not None:
        bad |= arr < float(d["valid_min"])
    if d.get("valid_max") is not None:
        bad |= arr > float(d["valid_max"])
    arr[bad] = np.nan
    return arr


def clean_labels(lab, fill):
    lab = np.asarray(lab, dtype=np.float64).copy()
    bad = ~np.isfinite(lab) | (lab < 0) | (lab == fill)
    lab[bad] = np.nan
    return lab


# ============================== EXTERNAL FEATURES ===============================

def gradient_magnitude(field, pixel_km=None):
    gy, gx = np.gradient(field)
    g = np.hypot(gx, gy)
    return g / pixel_km if pixel_km else g


def front_mask(sst, percentile=90.0, pixel_km=None):
    g = gradient_magnitude(sst, pixel_km)
    if np.all(~np.isfinite(g)):
        return np.zeros_like(sst, dtype=bool), g
    thr = np.nanpercentile(g, percentile)
    return np.isfinite(g) & (g >= thr), g


def okubo_weiss(u, v, pixel_km=None):
    """W = s_n^2 + s_s^2 - omega^2 (Okubo 1970; Weiss 1991)."""
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    if pixel_km:
        du_dy, du_dx, dv_dy, dv_dx = (a / pixel_km for a in (du_dy, du_dx, dv_dy, dv_dx))
    sn = du_dx - dv_dy
    ss = dv_dx + du_dy
    om = dv_dx - du_dy
    return sn ** 2 + ss ** 2 - om ** 2


def eddy_mask(u, v, threshold=0.2, pixel_km=None):
    w = okubo_weiss(u, v, pixel_km)
    s = np.nanstd(w)
    return np.isfinite(w) & (w < -threshold * s), w


# ============================== STATISTICS ===============================

def enrichment_table(labels, feature, clusters, name):
    """Per cluster: share of pixels with the feature vs the rest; odds ratio, Fisher exact p, BH-adjusted q."""
    from scipy.stats import fisher_exact
    rows = []
    ok = np.isfinite(labels) & np.isfinite(feature.astype(float))
    lab, feat = labels[ok].astype(np.int64), feature[ok].astype(bool)
    for c in clusters:
        inc = lab == c
        a = int(np.sum(inc & feat))
        b = int(np.sum(inc & ~feat))
        cc = int(np.sum(~inc & feat))
        d = int(np.sum(~inc & ~feat))
        if a + b == 0:
            continue
        orat = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (cc + 0.5))  # Haldane-Anscombe correction
        p = fisher_exact([[a, b], [cc, d]])[1] if min(a + b, cc + d) > 0 else np.nan
        rows.append(dict(feature=name, cluster=int(c), n=a + b, share_in_cluster=a / (a + b),
                         share_elsewhere=cc / max(cc + d, 1), odds_ratio=orat, log2_or=np.log2(orat), p=p))
    return bh_adjust(rows)


def bh_adjust(rows, key="p"):
    p = np.array([r[key] for r in rows], dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        pv = p[ok]
        order = np.argsort(pv)
        m = len(pv)
        adj = pv[order] * m / np.arange(1, m + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        out = np.empty(m)
        out[order] = np.minimum(adj, 1.0)
        q[ok] = out
    for r, v in zip(rows, q):
        r["q_bh"] = v
    return rows


def cramers_v(a, b):
    from scipy.stats import chi2_contingency
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    tab = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(tab, (ai, bi), 1)
    tab = tab[tab.sum(1) > 0][:, tab.sum(0) > 0]
    if min(tab.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(tab, correction=False)[0]
    return float(np.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1))))


# ============================== ANALYSES ===============================

def collect_pixels(scenes, labels, max_per_scene, seed=0):
    """Pool valid pixels (all channels finite, label valid). Returns X (N, C), y (N,), scene id (N,), flat index (N,)."""
    rng = np.random.default_rng(seed)
    xs, ys, gs, fs = [], [], [], []
    for s, (arr, lab) in enumerate(zip(scenes, labels)):
        if arr.shape[1:] != lab.shape:
            raise ValueError(f"scene {s}: data {arr.shape[1:]} and labels {lab.shape} differ")
        flat = arr.reshape(arr.shape[0], -1).T
        lf = lab.reshape(-1)
        ok = np.flatnonzero(np.all(np.isfinite(flat), 1) & np.isfinite(lf))
        if max_per_scene and len(ok) > max_per_scene:
            ok = np.sort(rng.choice(ok, max_per_scene, replace=False))
        xs.append(flat[ok])
        ys.append(lf[ok].astype(np.int64))
        gs.append(np.full(len(ok), s))
        fs.append(ok)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(fs)


def cluster_profiles(Xp, y, names, clusters):
    import pandas as pd
    mu, sd = np.nanmean(Xp, 0), np.nanstd(Xp, 0)
    long_rows, wide_rows = [], []
    for c in clusters:
        sel = Xp[y == c]
        if not len(sel):
            continue
        med = np.median(sel, 0)
        q25, q75 = np.percentile(sel, [25, 75], axis=0)
        for j, n in enumerate(names):
            long_rows.append(dict(cluster=int(c), variable=n, n=len(sel), median=med[j], q25=q25[j], q75=q75[j],
                                  mean=sel[:, j].mean(), std=sel[:, j].std(),
                                  z=(med[j] - mu[j]) / sd[j] if sd[j] > 0 else np.nan))
        wide_rows.append(dict(cluster=int(c), n=len(sel), frac=len(sel) / len(y), **dict(zip(names, med))))
    return pd.DataFrame(long_rows), pd.DataFrame(wide_rows)


def plot_profile_heatmap(long_df, out_png, max_rows=40):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not len(long_df):
        return
    z = long_df.pivot(index="cluster", columns="variable", values="z")
    n = long_df.groupby("cluster")["n"].first().reindex(z.index)
    z = z.loc[n.sort_values(ascending=False).index[:max_rows]]
    fig, ax = plt.subplots(figsize=(1.5 + 0.5 * z.shape[1], 1.0 + 0.25 * z.shape[0]))
    lim = np.nanmax(np.abs(z.values)) if np.isfinite(z.values).any() else 1.0
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(z.shape[1]), z.columns, rotation=90)
    ax.set_yticks(range(z.shape[0]), z.index)
    ax.set_xlabel("variable")
    ax.set_ylabel("cluster (by frequency)")
    fig.colorbar(im, ax=ax, label="z-score of cluster median")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


def occurrence_maps(labels, clusters):
    """Fraction of valid scenes in which each pixel carries cluster c (scenes must share a grid)."""
    shapes = {l.shape for l in labels}
    if len(shapes) != 1:
        warnings.warn("label rasters have different shapes; occurrence maps skipped")
        return None
    stack = np.stack(labels)
    valid = np.isfinite(stack).sum(0)
    out = np.stack([(stack == c).sum(0) / np.maximum(valid, 1) for c in clusters])
    out[:, valid == 0] = np.nan
    return out


def plot_occurrence(maps, clusters, out_png, n_show=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = min(n_show, len(clusters))
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if k < n:
            im = ax.imshow(maps[k], vmin=0, vmax=1, cmap="magma")
            ax.set_title(f"cluster {clusters[k]}")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="occurrence frequency", shrink=0.8)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


def monthly_frequency(labels, dates, clusters):
    import pandas as pd
    rows = []
    for lab, d in zip(labels, dates):
        m = pd.Timestamp(d).month
        v = lab[np.isfinite(lab)].astype(np.int64)
        tot = max(len(v), 1)
        for c in clusters:
            rows.append(dict(month=m, cluster=int(c), n=int(np.sum(v == c)), total=tot))
    df = pd.DataFrame(rows).groupby(["month", "cluster"], as_index=False)[["n", "total"]].sum()
    df["share"] = df["n"] / df["total"]
    return df


def surrogate_models(Xp, y, groups, names, clusters, depths=(3, 4, 6), n_splits=5, seed=0):
    """Raw variables -> cluster, cross-validated by scene. Clusters outside `clusters` are pooled as -1 ('other')."""
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.tree import DecisionTreeClassifier, export_text
    yy = np.where(np.isin(y, clusters), y, -1)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("surrogate CV needs at least two scenes")
    cv = GroupKFold(n_splits=min(n_splits, n_groups))
    models = {f"tree_d{d}": (lambda d=d: DecisionTreeClassifier(max_depth=d, min_samples_leaf=20, random_state=seed))
              for d in depths}
    models["hgb"] = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    rows, per_cluster = [], []
    labels_sorted = np.unique(yy)
    for name, make in models.items():
        pred = np.empty_like(yy)
        for tr, te in cv.split(Xp, yy, groups):
            pred[te] = make().fit(Xp[tr], yy[tr]).predict(Xp[te])
        maj = np.bincount(yy[yy >= 0]).argmax() if np.any(yy >= 0) else -1
        rows.append(dict(model=name, accuracy=accuracy_score(yy, pred),
                         balanced_accuracy=balanced_accuracy_score(yy, pred),
                         macro_f1=f1_score(yy, pred, average="macro"),
                         majority_baseline=float(np.mean(yy == maj)), n=len(yy), n_scenes=n_groups))
        f1s = f1_score(yy, pred, average=None, labels=labels_sorted)
        per_cluster += [dict(model=name, cluster=int(c), f1=v, support=int(np.sum(yy == c)))
                        for c, v in zip(labels_sorted, f1s)]
    rules = export_text(DecisionTreeClassifier(max_depth=min(depths), min_samples_leaf=20, random_state=seed)
                        .fit(Xp, yy), feature_names=list(names))
    return pd.DataFrame(rows), pd.DataFrame(per_cluster), rules


def external_analysis(scenes, labels, names, clusters, ext, masks=None):
    import pandas as pd
    from sklearn.metrics import normalized_mutual_info_score
    feats = {}
    sst, u, v = ext.get("sst"), ext.get("u"), ext.get("v")
    pk = ext.get("pixel_km")
    if sst is not None:
        j = names.index(sst)
        feats["sst_front"] = [front_mask(s[j], float(ext.get("front_percentile", 90)), pk)[0] for s in scenes]
    if u is not None and v is not None:
        ju, jv = names.index(u), names.index(v)
        feats["eddy_core"] = [eddy_mask(s[ju], s[jv], float(ext.get("ow_threshold", 0.2)), pk)[0] for s in scenes]
    for mname, files in (masks or {}).items():
        feats[mname] = [read_label_raster(f) > 0 for f in files]
    enr, assoc = [], []
    lab_all = np.concatenate([l.reshape(-1) for l in labels])
    for fname, fl in feats.items():
        f_all = np.concatenate([m.reshape(-1) for m in fl]).astype(float)
        ok = np.isfinite(lab_all)
        enr += enrichment_table(lab_all, f_all, clusters, fname)
        a, b = lab_all[ok].astype(np.int64), f_all[ok].astype(np.int64)
        assoc.append(dict(feature=fname, prevalence=float(b.mean()), nmi=normalized_mutual_info_score(a, b),
                          cramers_v=cramers_v(a, b), n=int(ok.sum())))
    return pd.DataFrame(enr), pd.DataFrame(assoc), feats


# ============================== DRIVER ===============================

def interpret_config(yml_conf):
    x = yml_conf.get("xai", {}) or {}
    conf = dict(ANCILLARY_DEFAULTS)
    conf.update(x.get("interpret", {}) or {})
    if "label_files" not in conf:
        raise ValueError("xai.interpret.label_files is required (one label raster per scene)")
    conf.setdefault("data_files", yml_conf["data"]["files_train"])
    conf.setdefault("variable_names", x.get("feature_names"))
    if len(conf["label_files"]) != len(conf["data_files"]):
        raise ValueError("xai.interpret.label_files and data_files must have the same length")
    return conf


def run(scenes, labels, names, conf, out_dir, dates=None):
    os.makedirs(out_dir, exist_ok=True)
    assert len(scenes) == len(labels) and len(scenes) > 0
    assert all(s.shape[0] == len(names) for s in scenes), "variable_names must match the number of channels"
    labels = [clean_labels(l, conf["label_fill"]) for l in labels]
    Xp, y, g, _ = collect_pixels(scenes, labels, conf["max_pixels_per_scene"], conf["seed"])
    if not len(y):
        raise ValueError("no valid pixels")
    uniq, cnt = np.unique(y, return_counts=True)
    order = uniq[np.argsort(-cnt)]
    clusters = [int(c) for c in order[:conf["n_top_clusters"]] if cnt[uniq == c][0] >= conf["min_cluster_pixels"]]
    long_df, wide_df = cluster_profiles(Xp, y, names, order)
    long_df.to_csv(os.path.join(out_dir, "cluster_profiles_long.csv"), index=False)
    wide_df.to_csv(os.path.join(out_dir, "cluster_profiles_wide.csv"), index=False)
    plot_profile_heatmap(long_df[long_df.cluster.isin(clusters)], os.path.join(out_dir, "cluster_profile_heatmap.png"))
    maps = occurrence_maps(labels, clusters)
    if maps is not None:
        np.savez_compressed(os.path.join(out_dir, "cluster_occurrence.npz"), maps=maps, clusters=np.array(clusters))
        plot_occurrence(maps, clusters, os.path.join(out_dir, "cluster_occurrence.png"))
    if dates:
        monthly_frequency(labels, dates, clusters).to_csv(os.path.join(out_dir, "cluster_monthly_frequency.csv"),
                                                          index=False)
    out = dict(clusters=clusters, profiles=wide_df)
    if len(np.unique(g)) >= 2:
        m, pc, rules = surrogate_models(Xp, y, g, names, clusters, conf["surrogate_depths"], conf["n_splits"],
                                        conf["seed"])
        m.to_csv(os.path.join(out_dir, "surrogate_metrics.csv"), index=False)
        pc.to_csv(os.path.join(out_dir, "surrogate_per_cluster_f1.csv"), index=False)
        with open(os.path.join(out_dir, "surrogate_tree_rules.txt"), "w") as f:
            f.write(rules)
        out["surrogate"] = m
    else:
        warnings.warn("surrogate models skipped: need >= 2 scenes for scene-grouped CV")
    ext = conf.get("external") or {}
    if ext:
        enr, assoc, _ = external_analysis(scenes, labels, names, clusters, ext, ext.get("masks"))
        enr.to_csv(os.path.join(out_dir, "external_enrichment.csv"), index=False)
        assoc.to_csv(os.path.join(out_dir, "external_association.csv"), index=False)
        out["enrichment"], out["association"] = enr, assoc
    return out


def main(yaml_path):
    from sit_fuse.utils import read_yaml
    yml_conf = read_yaml(yaml_path)
    conf = interpret_config(yml_conf)
    names = conf["variable_names"] or [f"chan_{i}" for i in range(int(yml_conf["data"]["number_channels"]))]
    scenes = [read_raw_scene(yml_conf, f) for f in conf["data_files"]]
    labels = [read_label_raster(f) for f in conf["label_files"]]
    out_dir = os.path.join(yml_conf["output"]["out_dir"], "xai", "interpret")
    run(scenes, labels, list(names), conf, out_dir, conf.get("dates"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-y", "--yaml", required=True)
    main(p.parse_args().yaml)
