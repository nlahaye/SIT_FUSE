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
  reference_<name>_agreement.csv    NMI, ARI, homogeneity, completeness, V-measure, Cramer's V and coverage between
                                    the cluster labels and a categorical reference map (e.g. MBON Seascapes)
  reference_<name>_contingency.csv  cluster x reference class pixel counts
  reference_<name>_per_cluster.csv  dominant reference class, purity and pixel count per cluster
  reference_<name>_per_class.csv    dominant cluster and its share per reference class
  reference_<name>_heatmap.png      row-normalised contingency of the most frequent clusters

YAML (xai.interpret block, in the SIT-FUSE config):
  label_files: [...]            one label raster per input file (same order as data.files_train or `data_files`)
  data_files:  [...]            optional; defaults to data.files_train
  variable_names: [...]         names of the input channels (defaults to xai.feature_names)
  dates: [...]                  optional ISO dates per scene for the monthly summary
  label_fill: -1                labels < 0, NaN or == label_fill are ignored
  label_scale: 1                cluster id = rint(raster value * label_scale). SIT-FUSE .no_geo.tif rasters store
                                id / 1000, so use 1000 for them (with 1 the ids are truncated to 0..4). With a
                                scale != 1 the CSVs carry cluster_id (integer id) and cluster = id / label_scale
                                (the raster-value convention NatureNet reward_env_projection expects).
  level: {level: parent | coarse, k: 20, hierarchy_json: <cluster_hierarchy.json>, label_input: fine,
          num_child: 5}       optional environment level (sit_fuse.postprocessing.cluster_hierarchy); ids are
                                then parent / coarse ids. CLI: --level / --k / --hierarchy_json. Output goes to
                                xai/interpret_<level> (interpret_k<K> for coarse) unless --out_subdir is given.
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
  reference:                    optional categorical maps on the label grid, one per scene
    seascapes:
      pattern: ".../seascape_sharks_t{t}.tif"   {t} = scene index from t_regex on the data file name; also {name},
                                                {stem}, {out_dir}; or files: [...] in data_files order
      t_regex: "_t(\\d+)"
      scale: 1                  class = rint(value * scale); classes <= 0, NaN = no class
      names_csv: ".../seascape_classes.csv"     optional (id, name) table for the per-cluster output
  Pattern placeholders for --label_pattern are the same, so class rasters can be profiled as labels
  (--label_pattern <seascape pattern> --label_scale 1 --out_subdir interpret_seascape).
  --reference_only skips the profiles, surrogates and external features (agreement tables only, every level).
"""
import argparse
import os
import warnings

import numpy as np

ANCILLARY_DEFAULTS = dict(label_fill=-1, label_scale=1, level=None, reference=None, n_top_clusters=30, max_pixels_per_scene=20000, surrogate_depths=[3, 4, 6],
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


def level_labels(lab, conf):
    """Cleaned label raster (NaN = invalid) -> integer ids (float array, NaN = invalid) at the configured level."""
    scale = float(conf.get("label_scale", 1) or 1)
    v = np.asarray(lab, dtype=np.float64)
    ok = np.isfinite(v)
    if scale == 1 and ok.any() and np.any(np.abs(v[ok] - np.rint(v[ok])) > 1e-6):
        warnings.warn("label values are not integers with label_scale 1 (SIT-FUSE tifs store id / 1000: set "
                      "xai.interpret.label_scale: 1000); ids are truncated")
    ids = np.full(v.shape, -1, dtype=np.int64)
    ids[ok] = np.rint(v[ok] * scale).astype(np.int64) if scale != 1 else v[ok].astype(np.int64)
    lv = conf.get("level") or {}
    if lv.get("level") and lv["level"] != "fine":
        from sit_fuse.postprocessing.cluster_hierarchy import apply_level, load_level_map
        lm = None
        if lv["level"] == "coarse":
            lm, _ = load_level_map(lv["hierarchy_json"], lv["k"])
        ids = apply_level(ids, lv["level"], int(lv.get("num_child", 5)), lv.get("label_input", "fine"), lm,
                          lv.get("unmapped", "error"))
    out = ids.astype(np.float64)
    out[ids < 0] = np.nan
    return out


def _id_columns(df, scale):
    """cluster -> cluster_id (int) and cluster = id / label_scale (raster-value convention), if scale != 1."""
    if scale == 1 or "cluster" not in df.columns:
        return df
    df = df.copy()
    df.insert(0, "cluster_id", df["cluster"].astype(int))
    df["cluster"] = df["cluster_id"] / float(scale)
    return df


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



# ============================== REFERENCE MAPS ===============================

def expand_pattern(pattern, data_file, out_dir="", t_regex=r"_t(\d+)", n_classes=None):
    """Per-scene path from a pattern with {out_dir}, {name}, {stem}, {t} (t from t_regex on the file name).
    Shell wildcards must match exactly one file."""
    import glob
    import re
    name = os.path.basename(str(data_file).rstrip("/"))
    fmt = dict(out_dir=str(out_dir).rstrip("/"), name=name, stem=os.path.splitext(name)[0], n_classes=n_classes)
    if "{t}" in pattern:
        m = re.search(t_regex, name)
        if not m:
            raise ValueError(f"t_regex {t_regex} does not match {name}")
        fmt["t"] = int(m.group(1))
    path = pattern.format(**fmt)
    if glob.has_magic(path):
        hits = sorted(glob.glob(path))
        if len(hits) != 1:
            raise ValueError(f"{path}: {len(hits)} matches (need exactly one)")
        path = hits[0]
    return path


def _resize_nearest(a, shape):
    if a.shape == tuple(shape):
        return a
    iy = np.minimum((np.arange(shape[0]) + 0.5) * a.shape[0] / shape[0], a.shape[0] - 1).astype(int)
    ix = np.minimum((np.arange(shape[1]) + 0.5) * a.shape[1] / shape[1], a.shape[1] - 1).astype(int)
    return a[np.ix_(iy, ix)]


def reference_ids(arr, scale=1, shape=None):
    """Reference raster -> float ids (NaN = no class: NaN, <= 0)."""
    a = np.asarray(arr, dtype=np.float64)
    if shape is not None and a.shape != tuple(shape):
        warnings.warn(f"reference map {a.shape} resized (nearest) to the label grid {tuple(shape)}")
        a = _resize_nearest(a, shape)
    out = np.full(a.shape, np.nan)
    ok = np.isfinite(a)
    q = np.rint(a[ok] * float(scale))
    out[ok] = np.where(q > 0, q, np.nan)
    return out


def contingency(labels, refs):
    """Pixel counts over pixels valid in both: (cluster ids, class ids, table[cluster, class])."""
    a = np.concatenate([l.reshape(-1) for l in labels])
    b = np.concatenate([r.reshape(-1) for r in refs])
    ok = np.isfinite(a) & np.isfinite(b)
    ca, ai = np.unique(a[ok].astype(np.int64), return_inverse=True)
    cb, bi = np.unique(b[ok].astype(np.int64), return_inverse=True)
    tab = np.zeros((len(ca), len(cb)), dtype=np.float64)
    np.add.at(tab, (ai, bi), 1)
    n_lab = int(np.isfinite(a).sum())
    return ca, cb, tab, n_lab


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def agreement_metrics(tab):
    """NMI (arithmetic), ARI, homogeneity / completeness / V-measure (clusters vs reference) and Cramer's V."""
    n = tab.sum()
    if n == 0:
        return dict(n=0)
    pk, pc = tab.sum(1) / n, tab.sum(0) / n
    hk, hc = _entropy(pk), _entropy(pc)
    pij = tab / n
    nz = pij > 0
    mi = float((pij[nz] * np.log(pij[nz] / np.outer(pk, pc)[nz])).sum())
    comb = lambda x: x * (x - 1) / 2.0
    sij, sa, sb, sn = comb(tab).sum(), comb(tab.sum(1)).sum(), comb(tab.sum(0)).sum(), comb(n)
    exp = sa * sb / sn if sn else 0.0
    den = 0.5 * (sa + sb) - exp
    ari = float((sij - exp) / den) if den else np.nan
    hom = 1.0 - (hc - mi) / hc if hc > 0 else 1.0           # H(C|K) = H(C) - MI
    com = 1.0 - (hk - mi) / hk if hk > 0 else 1.0
    v = 2 * hom * com / (hom + com) if (hom + com) > 0 else 0.0
    nmi = mi / (0.5 * (hk + hc)) if (hk + hc) > 0 else 1.0
    t = tab[tab.sum(1) > 0][:, tab.sum(0) > 0]
    cv = np.nan
    if min(t.shape) >= 2:
        e = np.outer(t.sum(1), t.sum(0)) / n
        cv = float(np.sqrt(((t - e) ** 2 / e).sum() / (n * (min(t.shape) - 1))))
    return dict(n=int(n), n_clusters=int(tab.shape[0]), n_classes=int(tab.shape[1]), nmi=float(nmi), ari=ari,
                homogeneity=float(hom), completeness=float(com), v_measure=float(v), cramers_v=cv,
                mutual_info=mi, h_clusters=hk, h_reference=hc)


def reference_agreement(labels, refs, name, out_dir, scale=1, class_names=None, n_show=30):
    """Agreement tables between cluster labels and one reference map; returns the metrics row."""
    import pandas as pd
    ca, cb, tab, n_lab = contingency(labels, refs)
    m = agreement_metrics(tab)
    m = dict(reference=name, coverage=m.get("n", 0) / max(n_lab, 1), **m)
    cname = lambda c: (class_names or {}).get(int(c), "")
    long = pd.DataFrame([dict(cluster=int(a), ref_class=int(b), ref_name=cname(b), n=int(tab[i, j]))
                         for i, a in enumerate(ca) for j, b in enumerate(cb) if tab[i, j] > 0])
    rows = []
    for i, a in enumerate(ca):
        r = tab[i]
        j = int(r.argmax())
        rows.append(dict(cluster=int(a), n=int(r.sum()), ref_class=int(cb[j]), ref_name=cname(cb[j]),
                         purity=float(r[j] / r.sum()), n_ref_classes=int((r > 0).sum())))
    per_cluster = pd.DataFrame(rows).sort_values("n", ascending=False)
    rows = []
    for j, b in enumerate(cb):
        c = tab[:, j]
        i = int(c.argmax())
        rows.append(dict(ref_class=int(b), ref_name=cname(b), n=int(c.sum()), cluster=int(ca[i]),
                         share=float(c[i] / c.sum()), n_clusters=int((c > 0).sum())))
    per_class = pd.DataFrame(rows).sort_values("n", ascending=False)
    os.makedirs(out_dir, exist_ok=True)
    pre = os.path.join(out_dir, f"reference_{name}")
    pd.DataFrame([m]).to_csv(pre + "_agreement.csv", index=False)
    _id_columns(long, scale).to_csv(pre + "_contingency.csv", index=False)
    _id_columns(per_cluster, scale).to_csv(pre + "_per_cluster.csv", index=False)
    pc = per_class.copy()
    if scale != 1:
        pc.insert(pc.columns.get_loc("cluster"), "cluster_id", pc["cluster"].astype(int))
        pc["cluster"] = pc["cluster_id"] / float(scale)
    pc.to_csv(pre + "_per_class.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = np.argsort(-tab.sum(1))[:n_show]
        R = tab[top] / tab[top].sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(1.0 + 0.35 * len(cb), 1.5 + 0.25 * len(top)))
        im = ax.imshow(R, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(len(top)), [str(int(ca[i])) for i in top], fontsize=7)
        ax.set_xticks(range(len(cb)), [f"{int(b)} {cname(b)[:18]}".strip() for b in cb], rotation=90, fontsize=7)
        ax.set_xlabel(f"{name} class")
        ax.set_ylabel("cluster")
        ax.set_title(f"{name}: NMI {m.get('nmi', np.nan):.2f}, V {m.get('v_measure', np.nan):.2f}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04, label="share of cluster pixels")
        fig.tight_layout()
        fig.savefig(pre + "_heatmap.png", dpi=150)
        plt.close(fig)
    except Exception as e:          # noqa: BLE001 - plotting is optional
        warnings.warn(f"reference heatmap skipped: {e}")
    return m


def load_references(conf, data_files, out_dir="", shape_of=None):
    """{name: (list of reference arrays, scale, class_names)} from conf['reference']."""
    import pandas as pd
    out = {}
    for name, r in (conf.get("reference") or {}).items():
        if r.get("files"):
            files = list(r["files"])
        else:
            files = [expand_pattern(r["pattern"], f, out_dir, r.get("t_regex", r"_t(\d+)")) for f in data_files]
        if len(files) != len(data_files):
            raise ValueError(f"reference {name}: {len(files)} files for {len(data_files)} scenes")
        names = None
        if r.get("names_csv") and os.path.exists(r["names_csv"]):
            t = pd.read_csv(r["names_csv"])
            names = dict(zip(t["id"].astype(int), t["name"].astype(str)))
        out[name] = (files, float(r.get("scale", 1)), names)
    return out

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


def run_references(labels, references, out_dir, scale=1):
    """labels: level ids (NaN = invalid). references: {name: (arrays, scale, class_names)}. Returns the metrics."""
    import pandas as pd
    rows = []
    for name, (arrs, rscale, cnames) in (references or {}).items():
        if len(arrs) != len(labels):
            raise ValueError(f"reference {name}: {len(arrs)} maps for {len(labels)} label rasters")
        refs = [reference_ids(a, rscale, l.shape) for a, l in zip(arrs, labels)]
        m = reference_agreement(labels, refs, name, out_dir, scale, cnames)
        print(f"reference {name}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                                for k, v in m.items()))
        rows.append(m)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "reference_agreement.csv"), index=False)
    return rows


def run(scenes, labels, names, conf, out_dir, dates=None, references=None, reference_only=False):
    os.makedirs(out_dir, exist_ok=True)
    assert len(labels) > 0
    labels = [level_labels(clean_labels(l, conf["label_fill"]), conf) for l in labels]
    sc = float(conf.get("label_scale", 1) or 1)
    ref_rows = run_references(labels, references, out_dir, sc)
    if reference_only:
        return dict(reference=ref_rows)
    assert scenes is not None and len(scenes) == len(labels)
    assert all(s.shape[0] == len(names) for s in scenes), "variable_names must match the number of channels"
    Xp, y, g, _ = collect_pixels(scenes, labels, conf["max_pixels_per_scene"], conf["seed"])
    if not len(y):
        raise ValueError("no valid pixels")
    uniq, cnt = np.unique(y, return_counts=True)
    order = uniq[np.argsort(-cnt)]
    clusters = [int(c) for c in order[:conf["n_top_clusters"]] if cnt[uniq == c][0] >= conf["min_cluster_pixels"]]
    long_df, wide_df = cluster_profiles(Xp, y, names, order)
    _id_columns(long_df, sc).to_csv(os.path.join(out_dir, "cluster_profiles_long.csv"), index=False)
    _id_columns(wide_df, sc).to_csv(os.path.join(out_dir, "cluster_profiles_wide.csv"), index=False)
    plot_profile_heatmap(long_df[long_df.cluster.isin(clusters)], os.path.join(out_dir, "cluster_profile_heatmap.png"))
    maps = occurrence_maps(labels, clusters)
    if maps is not None:
        np.savez_compressed(os.path.join(out_dir, "cluster_occurrence.npz"), maps=maps, clusters=np.array(clusters))
        plot_occurrence(maps, clusters, os.path.join(out_dir, "cluster_occurrence.png"))
    if dates:
        monthly_frequency(labels, dates, clusters).to_csv(os.path.join(out_dir, "cluster_monthly_frequency.csv"),
                                                          index=False)
    out = dict(clusters=clusters, profiles=wide_df, reference=ref_rows)
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


def main(yaml_path, level=None, k=None, hierarchy_json=None, label_input=None, out_subdir=None, label_pattern=None,
         label_scale=None, reference_only=False, no_reference=False):
    from sit_fuse.utils import read_yaml
    yml_conf = read_yaml(yaml_path)
    conf = interpret_config(yml_conf)
    sf_out = yml_conf["output"]["out_dir"]
    if label_pattern:
        if not out_subdir:
            raise ValueError("--label_pattern needs --out_subdir (e.g. interpret_seascape), so xai/interpret is kept")
        conf["label_files"] = [expand_pattern(label_pattern, f, sf_out) for f in conf["data_files"]]
    if label_scale is not None:
        conf["label_scale"] = float(label_scale)
    if level:
        lv = dict(conf.get("level") or {})
        lv["level"] = level
        if k is not None:
            lv["k"] = int(k)
        if hierarchy_json:
            lv["hierarchy_json"] = hierarchy_json
        if label_input:
            lv["label_input"] = label_input
        lv.setdefault("num_child", int(yml_conf.get("cluster", {}).get("heir", {}).get("num_classes", 5)))
        conf["level"] = lv
    lv = conf.get("level") or {}
    if out_subdir is None:
        out_subdir = "interpret"
        if lv.get("level") == "parent":
            out_subdir = "interpret_parent"
        elif lv.get("level") == "coarse":
            out_subdir = f"interpret_k{int(lv['k'])}"
    names = conf["variable_names"] or [f"chan_{i}" for i in range(int(yml_conf["data"]["number_channels"]))]
    labels = [read_label_raster(f) for f in conf["label_files"]]
    references = {}
    if not no_reference:
        for rname, (files, rscale, cnames) in load_references(conf, conf["data_files"], sf_out).items():
            if [os.path.abspath(f) for f in files] == [os.path.abspath(f) for f in conf["label_files"]]:
                print(f"reference {rname} skipped (it is the label set)")
                continue
            references[rname] = ([read_label_raster(f) for f in files], rscale, cnames)
    if reference_only and not references:
        raise ValueError("--reference_only needs an xai.interpret.reference block")
    scenes = None if reference_only else [read_raw_scene(yml_conf, f) for f in conf["data_files"]]
    out_dir = os.path.join(sf_out, "xai", out_subdir)
    run(scenes, labels, list(names), conf, out_dir, conf.get("dates"), references, reference_only)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-y", "--yaml", required=True)
    p.add_argument("--level", choices=["fine", "parent", "coarse"], default=None)
    p.add_argument("--k", type=int, default=None, help="coarse cut K (cluster_hierarchy.json)")
    p.add_argument("--hierarchy_json", default=None)
    p.add_argument("--label_input", choices=["fine", "parent"], default=None)
    p.add_argument("--out_subdir", default=None)
    p.add_argument("--label_pattern", default=None,
                   help="per-scene label rasters ({out_dir} {name} {stem} {t}), e.g. Seascapes class rasters")
    p.add_argument("--label_scale", type=float, default=None, help="overrides xai.interpret.label_scale")
    p.add_argument("--reference_only", action="store_true", help="agreement tables only (no raw scenes)")
    p.add_argument("--no_reference", action="store_true", help="skip the xai.interpret.reference maps")
    a = p.parse_args()
    main(a.yaml, a.level, a.k, a.hierarchy_json, a.label_input, a.out_subdir, a.label_pattern, a.label_scale,
         a.reference_only, a.no_reference)
