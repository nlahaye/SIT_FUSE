"""
Multi-scale SIT-FUSE environment levels: fine (heir) -> parent (first-stage) -> coarse (agglomerated parents).

    python -m sit_fuse.postprocessing.cluster_hierarchy verify -y config/hierarchy/shark_hierarchy_v2.yaml [--n_scenes 5]
    python -m sit_fuse.postprocessing.cluster_hierarchy build  -y config/hierarchy/shark_hierarchy_v2.yaml

Label encoding (Heir_DC, from models/deep_cluster/heir_dc.py and inference/generate_output.py):
  fine id   = num_child * parent + child    (num_child = cluster.heir.num_classes, child 0-based; parents without a
                                             child model get child 0)
  raster    = id / 1000 (float32 GeoTIFF), no data = -1 / 1000
  no_heir   = parent id / 1000 (written only with output.generate_intermediate_output: True)
So the parent level can be derived from the fine rasters (fine // num_child); the no_heir rasters are used by
`verify` to confirm this on real output. Id 0 is a valid cluster (no data is negative).

verify: on the first n scenes, checks fine // num_child == no_heir (if present), that ids are integers after
        scaling, the child range (0-based), and the share of pixels with fine id 0 (dropped by the old "> 0" rule).
build:  one pass over the scenes (one in memory at a time): per-parent pixel counts and channel means of the raw
        inputs (all pixels, no sampling) -> standardised profiles (z vs the pixel pool), optional equal weight per
        xai.feature_groups group -> Ward linkage of parents with >= min_parent_pixels pixels -> cuts at `ks`.
        Smaller parents inherit the labels of their nearest well-sampled parent, so all cuts stay nested.
        Coarse ids are 0..K-1 ordered by pixel share (largest first).
Outputs in hierarchy.out_dir:
  cluster_hierarchy.json  parents, pixel counts, cuts {K: {parent: coarse}}, settings (read by NatureNet
                          env_data_format cluster_level: coarse and by cluster_interpret --level coarse)
  parent_profiles.csv     parent, n, frac, core, standardised channel means
  cut_summary.csv         K, R^2 (pixel-weighted share of profile variance between coarse clusters), min / max
                          pixel share, n parents per cluster, nesting check
  linkage.npy, dendrogram.png

YAML:
  model_yaml: <merged SIT-FUSE config, e.g. copernicus_shark_v2_xai.yaml>  data reader, files, output.out_dir,
              cluster.heir.num_classes, xai.feature_names / feature_groups (from make_xai_config)
  hierarchy:
    out_dir, scenes (train | test; test = every complete scene), scene_stride, label_pattern, no_heir_pattern,
    label_input (fine | parent), label_scale (1000), num_child (null = cluster.heir.num_classes),
    variables (shell patterns, null = all), group_weighting (equal | none), min_parent_pixels, linkage (ward),
    ks: [..]
"""
import argparse
import fnmatch
import glob
import json
import os
import sys

import numpy as np
import yaml


# ============================== label levels (shared with cluster_interpret) ===============================

def to_ids(values, label_scale=1000, zero_valid=True):
    """Raster values -> integer cluster ids (int64), -1 = invalid (NaN, negative, or 0 if not zero_valid)."""
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v) & ((v >= 0) if zero_valid else (v > 0))
    out = np.full(v.shape, -1, dtype=np.int64)
    out[ok] = np.rint(v[ok] * label_scale).astype(np.int64)
    out[out < 0] = -1
    return out


def parent_ids(ids, num_child, label_input="fine"):
    ids = np.asarray(ids, dtype=np.int64)
    if label_input == "parent":
        return ids.copy()
    if label_input != "fine":
        raise ValueError(f"label_input must be fine | parent, got {label_input}")
    return np.where(ids >= 0, ids // int(num_child), -1)


def load_level_map(path, k):
    """{parent id: coarse id} for cut K from cluster_hierarchy.json, plus the file's settings."""
    with open(path) as f:
        h = json.load(f)
    cuts = h["cuts"]
    if str(k) not in cuts:
        raise ValueError(f"{path}: no cut K = {k}; available {sorted(int(c) for c in cuts)}")
    return {int(p): int(c) for p, c in cuts[str(k)].items()}, h


def apply_level(ids, level="fine", num_child=5, label_input="fine", level_map=None, unmapped="error"):
    """Integer ids at the raster level -> ids at `level` (fine | parent | coarse), -1 = invalid."""
    ids = np.asarray(ids, dtype=np.int64)
    if level == "fine":
        if label_input != "fine":
            raise ValueError("cluster_level fine needs fine (heir) rasters")
        return ids.copy()
    par = parent_ids(ids, num_child, label_input)
    if level == "parent":
        return par
    if level != "coarse":
        raise ValueError(f"level must be fine | parent | coarse, got {level}")
    if not level_map:
        raise ValueError("cluster_level coarse needs a level map (hierarchy_json + level_k)")
    keys = np.array(sorted(level_map), dtype=np.int64)
    vals = np.array([level_map[k] for k in keys], dtype=np.int64)
    out = np.full(par.shape, -1, dtype=np.int64)
    v = par >= 0
    j = np.clip(np.searchsorted(keys, par[v]), 0, len(keys) - 1)
    hit = keys[j] == par[v]
    if not hit.all():
        bad = np.unique(par[v][~hit])
        if unmapped == "error":
            raise ValueError(f"{len(bad)} parent clusters are not in the hierarchy, e.g. {bad[:10].tolist()}; "
                             "rebuild it over every scene (hierarchy.scenes: test) or set unmapped: invalid")
    res = np.full(j.shape, -1, dtype=np.int64)
    res[hit] = vals[j[hit]]
    out[v] = res
    return out


# ============================== config / IO ===============================

def _read_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def load_conf(path):
    conf = _read_yaml(path)
    mp = conf["model_yaml"]
    if not os.path.isabs(mp):                      # relative to this yaml
        mp = os.path.join(os.path.dirname(os.path.abspath(path)), mp)
    model = _read_yaml(mp)
    h = dict(scenes="test", scene_stride=1, label_input="fine", label_scale=1000, num_child=None,
             variables=None, group_weighting="equal", min_parent_pixels=200, linkage="ward",
             ks=[10, 15, 20, 25, 30],
             label_pattern="{out_dir}/{name}.clust.data_*clusters.no_geo.tif",
             no_heir_pattern="{out_dir}/{name}.no_heir.clust.data_*clusters.no_geo.tif")
    h.update(conf.get("hierarchy") or {})
    if h["num_child"] is None:
        h["num_child"] = int(model["cluster"]["heir"]["num_classes"])
    h.setdefault("out_dir", os.path.join(model["output"]["out_dir"], "hierarchy"))
    return model, h


def scene_files(model, h):
    key = "files_test" if h["scenes"] == "test" else "files_train"
    files = list(model["data"].get(key) or [])
    if not files:
        raise ValueError(f"data.{key} is empty in the model yaml")
    return files[::max(1, int(h["scene_stride"]))]


def label_file(model, pattern, scene):
    name = os.path.basename(scene[0] if isinstance(scene, list) else scene)
    p = pattern.format(out_dir=model["output"]["out_dir"].rstrip("/"), name=name,
                       stem=os.path.splitext(name)[0])
    if glob.has_magic(p):
        hits = sorted(glob.glob(p))
        if len(hits) > 1:
            raise ValueError(f"{p} matches {len(hits)} files")
        return hits[0] if hits else None
    return p if os.path.exists(p) else None


def select_variables(names, patterns):
    if not patterns:
        return list(range(len(names)))
    idx = []
    for pat in patterns:
        idx += [i for i, n in enumerate(names) if fnmatch.fnmatchcase(n, pat) and i not in idx]
    if not idx:
        raise ValueError("hierarchy.variables matches no channel")
    return idx


def variable_weights(names, groups, mode):
    """Equal total weight per feature group (squared-distance contribution), 1 otherwise."""
    w = np.ones(len(names))
    if mode != "equal" or not groups:
        return w
    for members in groups.values():
        m = [i for i, n in enumerate(names) if n in set(members)]
        if m:
            w[m] = 1.0 / np.sqrt(len(m))
    return w


# ============================== verify ===============================

def verify(model, h, n_scenes=5):
    from sit_fuse.xai.cluster_interpret import read_label_raster
    nc, sc = int(h["num_child"]), float(h["label_scale"])
    rows, worst = [], 1.0
    for s in scene_files(model, h)[:n_scenes]:
        fl = label_file(model, h["label_pattern"], s)
        if fl is None:
            print("missing fine raster for", s)
            continue
        v = read_label_raster(fl)
        fin = np.isfinite(v) & (v >= 0)
        frac_int = float(np.mean(np.abs(v[fin] * sc - np.rint(v[fin] * sc)) < 1e-3)) if fin.any() else np.nan
        ids = to_ids(v, sc, zero_valid=True)
        ok = ids >= 0
        child = ids[ok] % nc
        r = dict(scene=os.path.basename(fl), valid=int(ok.sum()), integer_ids=frac_int,
                 n_fine=int(len(np.unique(ids[ok]))), n_parent=int(len(np.unique(ids[ok] // nc))),
                 child_min=int(child.min()) if ok.any() else -1, child_max=int(child.max()) if ok.any() else -1,
                 zero_id_share=float(np.mean(ids[ok] == 0)) if ok.any() else np.nan, parent_agreement=np.nan)
        fn = label_file(model, h["no_heir_pattern"], s)
        if fn is not None:
            p = to_ids(read_label_raster(fn), sc, zero_valid=True)
            both = ok & (p >= 0)
            r["parent_agreement"] = float(np.mean(ids[both] // nc == p[both])) if both.any() else np.nan
            worst = min(worst, r["parent_agreement"])
        rows.append(r)
        print(r)
    if not rows:
        raise SystemExit("no label rasters found; check label_pattern / output.out_dir")
    if all(np.isnan(r["parent_agreement"]) for r in rows):
        print("NOTE: no no_heir rasters (set output.generate_intermediate_output: True and rerun inference to check "
              "fine // num_child against the first-stage labels)")
    elif worst < 0.999:
        raise SystemExit(f"fine // {nc} disagrees with no_heir on {100 * (1 - worst):.2f}% of pixels; "
                         "use label_input: parent with the no_heir rasters")
    if min(r["integer_ids"] for r in rows) < 0.999:
        raise SystemExit(f"raster values * {sc} are not integers; check label_scale")
    print("OK: fine = %d * parent + child, child %d-based" % (nc, min(r["child_min"] for r in rows)))
    return rows


# ============================== build ===============================

def accumulate(model, h, names, var_idx):
    from sit_fuse.xai.cluster_interpret import read_label_raster, read_raw_scene
    pat = h["label_pattern"] if h["label_input"] == "fine" else h["no_heir_pattern"]
    V = len(var_idx)
    n = np.zeros(0, np.int64)
    s = np.zeros((0, V))
    g_sum, g_sq, g_n, used, skipped = np.zeros(V), np.zeros(V), 0, 0, []
    for scene in scene_files(model, h):
        lf = label_file(model, pat, scene)
        if lf is None:
            skipped.append(scene)
            continue
        arr = read_raw_scene(model, scene[0] if isinstance(scene, list) else scene)[var_idx]
        par = parent_ids(to_ids(read_label_raster(lf), h["label_scale"]), h["num_child"], h["label_input"])
        if arr.shape[1:] != par.shape:
            raise ValueError(f"{scene}: data {arr.shape[1:]} vs labels {par.shape}")
        X = arr.reshape(V, -1).T
        p = par.reshape(-1)
        ok = (p >= 0) & np.all(np.isfinite(X), 1)
        X, p = X[ok], p[ok]
        if not len(p):
            continue
        m = int(p.max()) + 1
        if m > len(n):
            n = np.concatenate([n, np.zeros(m - len(n), np.int64)])
            s = np.vstack([s, np.zeros((m - len(s), V))])
        n += np.bincount(p, minlength=len(n))
        for j in range(V):
            s[:, j] += np.bincount(p, weights=X[:, j], minlength=len(n))
        g_sum += X.sum(0)
        g_sq += (X ** 2).sum(0)
        g_n += len(p)
        used += 1
    if skipped:
        print(f"WARNING: {len(skipped)} scenes without label rasters skipped, e.g. {skipped[:3]}")
    if not g_n:
        raise ValueError("no valid pixels")
    parents = np.flatnonzero(n > 0)
    mean = s[parents] / n[parents, None]
    mu = g_sum / g_n
    sd = np.sqrt(np.maximum(g_sq / g_n - mu ** 2, 0))
    sd[sd == 0] = 1.0
    print(f"{used} scenes, {g_n} valid pixels, {len(parents)} parent clusters")
    return parents, n[parents], (mean - mu) / sd


def nested(cuts, ks):
    """True if every cluster of each finer cut lies inside one cluster of the next coarser cut."""
    ks = sorted(ks)
    for a, b in zip(ks[:-1], ks[1:]):
        pairs = {}
        for p, cb in cuts[b].items():
            if pairs.setdefault(cb, cuts[a][p]) != cuts[a][p]:
                return False
    return True


def build(model, h):
    from scipy.cluster.hierarchy import fcluster, linkage
    names = list(model.get("xai", {}).get("feature_names") or [])
    if not names:
        raise ValueError("xai.feature_names is empty; use the make_xai_config output as model_yaml")
    var_idx = select_variables(names, h.get("variables"))
    vnames = [names[i] for i in var_idx]
    w = variable_weights(vnames, model.get("xai", {}).get("feature_groups"), h["group_weighting"])
    parents, cnt, Z = accumulate(model, h, names, var_idx)
    X = Z * w
    core = cnt >= int(h["min_parent_pixels"])
    if core.sum() < 2:
        raise ValueError("fewer than 2 parents with >= min_parent_pixels pixels")
    L = linkage(X[core], method=h["linkage"])
    ks = sorted(int(k) for k in h["ks"] if int(k) <= core.sum())
    if len(ks) < len(h["ks"]):
        print(f"NOTE: cuts above {core.sum()} core parents dropped")
    # small parents -> nearest core parent (inherits its label at every K, keeps the cuts nested)
    ci = np.flatnonzero(core)
    near = np.arange(len(parents))
    for i in np.flatnonzero(~core):
        near[i] = ci[np.argmin(((X[ci] - X[i]) ** 2).sum(1))]
    cuts, summary = {}, []
    tot = ((X - np.average(X, axis=0, weights=cnt)) ** 2).sum(1) @ cnt
    for k in ks:
        lab_core = fcluster(L, k, criterion="maxclust")
        lab = np.zeros(len(parents), np.int64)
        lab[ci] = lab_core
        lab = lab[near]
        share = np.bincount(lab, weights=cnt)
        order = np.argsort(-share[np.unique(lab)])
        remap = {int(u): i for i, u in enumerate(np.unique(lab)[order])}
        lab = np.array([remap[int(x)] for x in lab])
        K = len(remap)
        within = 0.0
        for c in range(K):
            m = lab == c
            within += ((X[m] - np.average(X[m], axis=0, weights=cnt[m])) ** 2).sum(1) @ cnt[m]
        sh = np.bincount(lab, weights=cnt) / cnt.sum()
        npar = np.bincount(lab)
        summary.append(dict(K=k, n_clusters=K, r2=float(1 - within / tot), min_share=float(sh.min()),
                            max_share=float(sh.max()), min_parents=int(npar.min()), max_parents=int(npar.max())))
        cuts[k] = {int(p): int(c) for p, c in zip(parents, lab)}
        if K != k:
            print(f"WARNING: cut K = {k} gave {K} clusters (ties)")
    ok = nested(cuts, ks)
    for r in summary:
        r["nested"] = ok
        print(r)
    if not ok:
        raise ValueError("cuts are not nested")
    return dict(parents=parents, cnt=cnt, Z=Z, core=core, L=L, vnames=vnames, w=w, cuts=cuts, summary=summary,
                ks=ks)


def write(res, model, h):
    import pandas as pd
    out = h["out_dir"]
    os.makedirs(out, exist_ok=True)
    js = dict(version=1, label_input=h["label_input"], label_scale=h["label_scale"], num_child=h["num_child"],
              linkage=h["linkage"], group_weighting=h["group_weighting"], min_parent_pixels=h["min_parent_pixels"],
              scenes=h["scenes"], scene_stride=h["scene_stride"], variables=res["vnames"],
              parents=res["parents"].tolist(), parent_pixels=res["cnt"].tolist(),
              core_parents=res["parents"][res["core"]].tolist(),
              cuts={str(k): {str(p): c for p, c in m.items()} for k, m in res["cuts"].items()},
              summary=res["summary"])
    with open(os.path.join(out, "cluster_hierarchy.json"), "w") as f:
        json.dump(js, f, indent=1)
    prof = pd.DataFrame(res["Z"], columns=res["vnames"])
    prof.insert(0, "core", res["core"])
    prof.insert(0, "frac", res["cnt"] / res["cnt"].sum())
    prof.insert(0, "n", res["cnt"])
    prof.insert(0, "parent", res["parents"])
    for k in res["ks"]:
        prof[f"coarse_K{k}"] = [res["cuts"][k][int(p)] for p in res["parents"]]
    prof.to_csv(os.path.join(out, "parent_profiles.csv"), index=False)
    pd.DataFrame(res["summary"]).to_csv(os.path.join(out, "cut_summary.csv"), index=False)
    np.save(os.path.join(out, "linkage.npy"), res["L"])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram
        fig, ax = plt.subplots(figsize=(10, 4))
        dendrogram(res["L"], truncate_mode="lastp", p=max(res["ks"]), ax=ax, no_labels=True)
        for k in res["ks"]:
            hgt = 0.5 * (res["L"][-k, 2] + res["L"][-(k - 1), 2]) if 1 < k < len(res["L"]) else res["L"][-1, 2]
            ax.axhline(hgt, ls=":", lw=0.8, color="grey")
            ax.text(ax.get_xlim()[1], hgt, f" K={k}", va="center", fontsize=7)
        ax.set_ylabel("Ward distance")
        ax.set_title("Parent-cluster hierarchy")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "dendrogram.png"), dpi=200)
        plt.close(fig)
    except Exception as e:      # plotting is optional
        print("dendrogram skipped:", e)
    print("wrote", os.path.join(out, "cluster_hierarchy.json"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["verify", "build"])
    p.add_argument("-y", "--yaml", required=True)
    p.add_argument("--n_scenes", type=int, default=5)
    a = p.parse_args(argv)
    model, h = load_conf(a.yaml)
    if a.cmd == "verify":
        verify(model, h, a.n_scenes)
    else:
        write(build(model, h), model, h)


if __name__ == "__main__":
    main()
