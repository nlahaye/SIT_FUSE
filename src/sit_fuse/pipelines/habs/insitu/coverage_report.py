"""
Per-product coverage report for an in-situ HAB table produced by pull_insitu_hab
(combined_augmented.csv or by_source/*.csv; the *_strict.csv file has no
source/method columns, so it is reported as a single "unknown" source).

  python -m sit_fuse.pipelines.habs.insitu.coverage_report \
      -i insitu_augmented/combined_augmented.csv -o insitu_augmented/coverage \
      --start 2018-01-01 --end 2027-01-01

A "product" is a concentration column (Total_Phytoplankton, Pseudo_nitzschia_*,
pDA, ...). An observation is a row with a finite value for that product.

Outputs (out_dir):
  coverage_summary.csv   product x source x method: obs, positives, stations, days, first/last,
                         months covered / months in range, median and max revisit gap (days)
  coverage_station.csv   product x station x source: same metrics plus lat/lon
  coverage_monthly.csv   product x station x month observation counts (long form)
  coverage_yearly.csv    product x source x year observation counts (wide)
  coverage_bins.csv      product x source x concentration bin counts (UQ bin edges; class balance)
  coverage_sensors.csv   product x sensor: observation days inside each sensor's record
  coverage_gaps.csv      station gaps longer than min_gap_days inside the requested range
  coverage_report.md     readable summary
  heatmap_{product}.png  station x month counts (if matplotlib is installed)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sit_fuse.pipelines.habs.uq.hab_uq_common import DEFAULT_SPECIES
    BIN_EDGES = {k: list(v.edges) for k, v in DEFAULT_SPECIES.items()}
except Exception:  # pragma: no cover
    _C = [0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    BIN_EDGES = {"Total_Phytoplankton": _C, "Pseudo_nitzschia_seriata_group": _C,
                 "Pseudo_nitzschia_delicatissima_group": _C, "Alexandrium_spp": _C,
                 "pDA": [0, 0.1, 1, 10, 100, 1000, 10000]}

CORE_PRODUCTS = ["Total_Phytoplankton", "Pseudo_nitzschia_seriata_group",
                 "Pseudo_nitzschia_delicatissima_group", "pDA"]
OTHER_PRODUCTS = ["Alexandrium_spp", "Akashiwo_sanguinea", "Dinophysis_spp", "Lingulodinium_polyedra",
                  "Prorocentrum_spp", "Ceratium_spp", "Cochlodinium_spp", "Gymnodinium_spp",
                  "Other_Diatoms", "Other_Dinoflagellates", "tDA", "dDA", "Avg_Chloro"]

# approximate start of the ocean-colour science record used for matchups (edit to your archive)
SENSOR_START = {"AQUA_MODIS": "2002-07-04", "SNPP_VIIRS": "2012-01-02", "S3A_OLCI": "2016-04-26",
                "JPSS1_VIIRS": "2018-01-05", "S3B_OLCI": "2018-05-15", "JPSS2_VIIRS": "2023-03-01",
                "PACE_OCI": "2024-03-05"}


def _load(df_or_path) -> pd.DataFrame:
    if isinstance(df_or_path, pd.DataFrame):
        df = df_or_path.copy()
    else:
        paths = [df_or_path] if isinstance(df_or_path, (str, Path)) else list(df_or_path)
        df = pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)
    t = None
    if "sample_time_utc" in df.columns:
        t = pd.to_datetime(df["sample_time_utc"], utc=True, errors="coerce", format="mixed")
    tt = pd.to_datetime(df["time"], format="%m/%d/%y", errors="coerce", utc=True) if "time" in df.columns else None
    if t is None:
        t = tt
    elif tt is not None:
        t = t.fillna(tt)
    df["_t"] = t.dt.tz_convert(None)
    df["_day"] = df["_t"].dt.floor("D")
    for c in ("source", "method"):
        df[c] = df[c].fillna("unknown").astype(str) if c in df.columns else "unknown"
    df["Location_Code"] = df["Location_Code"].astype(str)
    return df[df["_t"].notna()]


def _gaps(days: pd.Series) -> np.ndarray:
    d = np.sort(days.dropna().unique())
    return np.diff(d).astype("timedelta64[D]").astype(float) if len(d) > 1 else np.array([])


def _metrics(g: pd.DataFrame, v: pd.Series, months_in_range: int) -> dict:
    gp = _gaps(g["_day"])
    return dict(n_obs=int(len(g)), n_positive=int((v.loc[g.index] > 0).sum()),
                n_stations=int(g["Location_Code"].nunique()), n_days=int(g["_day"].nunique()),
                first=g["_t"].min().date(), last=g["_t"].max().date(),
                months_with_data=int(g["_t"].dt.to_period("M").nunique()), months_in_range=months_in_range,
                month_fraction=round(g["_t"].dt.to_period("M").nunique() / max(months_in_range, 1), 3),
                median_revisit_days=float(np.median(gp)) if len(gp) else np.nan,
                max_gap_days=float(gp.max()) if len(gp) else np.nan)


def coverage_report(df_or_path, out_dir, start=None, end=None, products=None, include_extra=False,
                    min_gap_days: int = 60, sensors: dict | None = None, plots: bool = True) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = _load(df_or_path)
    t0 = pd.Timestamp(start) if start else df["_t"].min().normalize()
    t1 = pd.Timestamp(end) if end else df["_t"].max().normalize() + pd.Timedelta("1D")
    df = df[(df["_t"] >= t0) & (df["_t"] < t1)]
    months = pd.period_range(t0, t1 - pd.Timedelta("1D"), freq="M")
    if products is None:
        products = [c for c in CORE_PRODUCTS + OTHER_PRODUCTS if c in df.columns]
        if include_extra:
            products += [c for c in df.columns if c.startswith("extra_") and not c.endswith("_sigma_dex")]
    sensors = {k: pd.Timestamp(v) for k, v in (sensors or SENSOR_START).items()}

    summ, stat, mon, yr, bins, sens, gaps = [], [], [], [], [], [], []
    for p in products:
        v = pd.to_numeric(df[p], errors="coerce")
        d = df[np.isfinite(v)]
        if d.empty:
            summ.append(dict(product=p, source="(none)", method="", n_obs=0))
            continue
        vp = v.loc[d.index]
        for (s, m), g in d.groupby(["source", "method"]):
            summ.append(dict(product=p, source=s, method=m, **_metrics(g, v, len(months))))
            e = BIN_EDGES.get(p)
            if e is not None:
                k = np.clip(np.searchsorted(e, vp.loc[g.index].to_numpy(), side="right") - 1, 0, len(e) - 2)
                cnt = np.bincount(k, minlength=len(e) - 1)
                bins.append(dict(product=p, source=s, method=m,
                                 **{f"bin{i}_[{e[i]:g},{e[i + 1]:g})": int(c) for i, c in enumerate(cnt)}))
        summ.append(dict(product=p, source="ALL", method="ALL", **_metrics(d, v, len(months))))
        for (loc, s, m), g in d.groupby(["Location_Code", "source", "method"]):
            r = _metrics(g, v, len(months))
            r.pop("n_stations")
            stat.append(dict(product=p, Location_Code=loc, source=s, method=m,
                             latitude=pd.to_numeric(g["latitude"], errors="coerce").median(),
                             longitude=pd.to_numeric(g["longitude"], errors="coerce").median(), **r))
            dd = np.sort(g["_day"].unique())
            edges = np.concatenate([[np.datetime64(t0)], dd, [np.datetime64(t1)]])
            for a, b in zip(edges[:-1], edges[1:]):
                n = (b - a).astype("timedelta64[D]").astype(int)
                if n > min_gap_days:
                    gaps.append(dict(product=p, Location_Code=loc, source=s, method=m,
                                     gap_start=pd.Timestamp(a).date(), gap_end=pd.Timestamp(b).date(), days=n))
        mc = d.groupby(["Location_Code", "method", d["_t"].dt.to_period("M")]).size().rename("n_obs").reset_index()
        mc.columns = ["Location_Code", "method", "month", "n_obs"]
        mon.append(mc.assign(product=p))
        yc = d.groupby(["source", d["_t"].dt.year]).size().unstack(fill_value=0)
        yr.append(yc.assign(product=p).reset_index())
        for sn, st in sensors.items():
            w = d[d["_t"] >= st]
            sens.append(dict(product=p, sensor=sn, record_start=st.date(), n_obs=len(w),
                             n_days=int(w["_day"].nunique()), n_stations=int(w["Location_Code"].nunique())))

    S = pd.DataFrame(summ)
    S.to_csv(out / "coverage_summary.csv", index=False)
    ST = pd.DataFrame(stat)
    ST.to_csv(out / "coverage_station.csv", index=False)
    M = pd.concat(mon, ignore_index=True) if mon else pd.DataFrame()
    M.to_csv(out / "coverage_monthly.csv", index=False)
    Y = pd.concat(yr, ignore_index=True).fillna(0) if yr else pd.DataFrame()
    if len(Y):
        Y = Y[["product", "source"] + sorted(c for c in Y.columns if c not in ("product", "source"))]
    Y.to_csv(out / "coverage_yearly.csv", index=False)
    B = pd.DataFrame(bins)
    B.to_csv(out / "coverage_bins.csv", index=False)
    SN = pd.DataFrame(sens)
    SN.to_csv(out / "coverage_sensors.csv", index=False)
    G = pd.DataFrame(gaps)
    G.to_csv(out / "coverage_gaps.csv", index=False)
    _markdown(out, S, ST, B, SN, G, t0, t1, min_gap_days)
    if plots and len(M):
        _heatmaps(out, M, months)
    return S


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(x, float) and np.isnan(x)) else
                                       (f"{x:g}" if isinstance(x, float) else str(x)) for x in r) + " |")
    return "\n".join(lines) + "\n"


def _markdown(out, S, ST, B, SN, G, t0, t1, min_gap_days):
    md = [f"# In-situ coverage report\n\nRange: {t0.date()} to {t1.date()} (end exclusive)\n"]
    keep = ["source", "method", "n_obs", "n_positive", "n_stations", "n_days", "first", "last",
            "month_fraction", "median_revisit_days", "max_gap_days"]
    for p, g in S.groupby("product", sort=False):
        md.append(f"\n## {p}\n")
        md.append(_md_table(g[[c for c in keep if c in g.columns]]))
        if len(B) and p in set(B["product"]):
            b = B[B["product"] == p].drop(columns=["product"])
            md.append("\nConcentration-bin counts (UQ bin edges)\n\n" + _md_table(b.dropna(axis=1, how="all")))
        if len(SN):
            sn = SN[SN["product"] == p].drop(columns=["product"])
            md.append("\nObservations inside each sensor record\n\n" + _md_table(sn))
        if len(ST):
            st = ST[ST["product"] == p][["Location_Code", "source", "method", "n_obs", "n_days", "first",
                                        "last", "month_fraction", "max_gap_days"]]
            md.append("\nStations\n\n" + _md_table(st.sort_values(["source", "Location_Code"])))
    if len(G):
        top = G.sort_values("days", ascending=False).groupby("product").head(10)
        md.append(f"\n## Longest station gaps (> {min_gap_days} days, top 10 per product)\n\n" + _md_table(top))
    (out / "coverage_report.md").write_text("\n".join(md))


def _heatmaps(out, M, months):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    for p, g in M.groupby("product"):
        g = g.assign(row=g["Location_Code"] + " (" + g["method"] + ")")
        piv = g.pivot_table(index="row", columns="month", values="n_obs", aggfunc="sum")
        piv = piv.reindex(columns=months)
        fig, ax = plt.subplots(figsize=(max(8, len(months) * 0.09), 0.3 * len(piv) + 1.6))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#eeeeee")                      # grey = no observations
        im = ax.imshow(np.ma.masked_invalid(np.log10(piv.to_numpy())), aspect="auto", cmap=cmap,
                       interpolation="nearest")
        ax.set_yticks(range(len(piv)), piv.index, fontsize=7)
        jan = [i for i, m in enumerate(months) if m.month == 1]
        ax.set_xticks(jan, [str(months[i].year) for i in jan], fontsize=7)
        ax.set_title(f"{p}: observations per station-month (log10 n; grey = none)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        fig.tight_layout()
        fig.savefig(out / f"heatmap_{p}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", nargs="+", required=True, help="combined_augmented.csv or by_source/*.csv")
    ap.add_argument("-o", "--out-dir", default="coverage")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--products", nargs="+")
    ap.add_argument("--include-extra", action="store_true", help="also report extra_* columns")
    ap.add_argument("--min-gap-days", type=int, default=60)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    coverage_report(a.input, a.out_dir, a.start, a.end, a.products, a.include_extra, a.min_gap_days,
                    plots=not a.no_plots)
    print(f"wrote {a.out_dir}/coverage_report.md")
