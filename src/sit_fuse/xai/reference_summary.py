"""
Collect the cluster_interpret reference agreement tables of every level into one table.

    python -m sit_fuse.xai.reference_summary -d /data/nlahaye/output/Learnergy/COP_SHARK_V2/xai

Reads <xai dir>/interpret*/reference_agreement.csv (interpret = fine, interpret_parent, interpret_k<K>, ...) and
writes <xai dir>/reference_agreement_levels.csv plus a Markdown copy (manuscript table). One row per level and
reference: n_clusters, n_classes, coverage, NMI, ARI, homogeneity, completeness, V-measure, Cramer's V.
"""
import argparse
import glob
import os
import re


def level_name(d):
    b = os.path.basename(d.rstrip("/"))
    return "fine" if b == "interpret" else b.replace("interpret_", "")


def _order(name):
    m = re.fullmatch(r"k(\d+)", name)
    return (0, 0) if name == "fine" else (1, 0) if name == "parent" else (2, -int(m.group(1))) if m else (3, name)


def collect(xai_dir):
    import pandas as pd
    rows = []
    for f in sorted(glob.glob(os.path.join(xai_dir, "interpret*", "reference_agreement.csv"))):
        df = pd.read_csv(f)
        df.insert(0, "level", level_name(os.path.dirname(f)))
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"no interpret*/reference_agreement.csv under {xai_dir}")
    out = pd.concat(rows, ignore_index=True)
    out = out.iloc[sorted(range(len(out)), key=lambda i: (out.reference[i], _order(out.level[i])))]
    return out.reset_index(drop=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--xai_dir", required=True)
    a = p.parse_args(argv)
    df = collect(a.xai_dir)
    cols = [c for c in ("level", "reference", "n_clusters", "n_classes", "coverage", "nmi", "ari", "homogeneity",
                        "completeness", "v_measure", "cramers_v") if c in df]
    path = os.path.join(a.xai_dir, "reference_agreement_levels.csv")
    df.to_csv(path, index=False)
    try:
        md = df[cols].to_markdown(index=False, floatfmt=".3f")
    except ImportError:                 # tabulate not installed
        md = df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    with open(path.replace(".csv", ".md"), "w") as f:
        f.write(md + "\n")
    print(md)
    print("wrote", path)


if __name__ == "__main__":
    main()
