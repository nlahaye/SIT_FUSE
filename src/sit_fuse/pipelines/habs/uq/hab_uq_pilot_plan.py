"""
Plan a short-turnaround HAB UQ pilot over a high-signal sub-window.

Reads the in-situ table exactly as the UQ pipeline does (load_insitu +
species_frame, uq.max_depth_m) and reports, per species, how many samples and
alert-level events (first alert threshold) fall in each scene-split role.
It then ranks candidate windows and, for the chosen window, either

  --mode scene_split  keeps the pipeline's day roles (encoder/context-held-out
                      test days; no extra leakage). Report only.
  --mode resplit      re-splits the window into train/calibration/test by
                      time block, stratified so every split gets events, and
                      writes a date_roles CSV for uq.splits.date_roles_file.
                      Test days may then include scene-split train days
                      (seen, unlabelled, by the encoder; labelled by the
                      production context assignment), so the production
                      baseline is optimistic on them; the overlap is reported.

Usage
  python -m sit_fuse.pipelines.habs.uq.hab_uq_pilot_plan -y <hab yaml> \
      --species Pseudo_nitzschia_seriata_group Pseudo_nitzschia_delicatissima_group pDA \
      --window-days 150 [--start 2025-02-01 --end 2025-06-30] \
      [--mode resplit --block-days 14 --buffer-days 2 --out pilot_date_roles.csv]

Several disjoint windows: set uq.date_windows in the YAML (or pass
--windows 2024-05-01:2024-06-30 2024-12-01:2025-06-15 ...). Ranking is then
skipped and --mode resplit deals 14-day blocks of all windows together (blocks
never span two windows), so every split draws from every window's events.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import yaml

from sit_fuse.pipelines.habs.uq.hab_insitu import insitu_filter_config, load_insitu, species_frame
from sit_fuse.pipelines.habs.uq.hab_uq_common import date_windows, insitu_date_kwargs, species_spec, window_days

ROLES = ("train", "calibration", "test")


def species_events(yml, species, start=None, end=None):
    """One row per (sample, species): date, station, value, event (>= first alert threshold)."""
    uq = yml["uq"]
    dk = {"start_date": start, "end_date": end} if (start or end) else insitu_date_kwargs(uq)
    ins = load_insitu(uq.get("insitu_file", yml.get("insitu_file")), max_depth_m=uq.get("max_depth_m", 1.0),
                      filters=insitu_filter_config(yml), **dk)
    out = []
    for k in species:
        spec = species_spec(k, uq.get("species", {}).get(k))
        f = species_frame(ins, spec)
        if f.empty:
            continue
        thr = spec.alert_thresholds[0] if spec.alert_thresholds else spec.edges[1]
        out.append(pd.DataFrame({"species": k, "date": f["date"].dt.normalize(),
                                 "Location_Code": f["Location_Code"].astype(str),
                                 "value": f["value"], "event": f["value"] >= thr, "threshold": thr}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def scene_roles(yml):
    try:
        from sit_fuse.pipelines.habs.hab_scene_split import uq_date_roles
        return uq_date_roles(yml) or {}
    except Exception as e:  # pragma: no cover
        print("scene_split roles unavailable:", e)
        return {}


def summarize(ev, roles, by="role"):
    ev = ev.copy()
    ev["role"] = ev["date"].dt.strftime("%Y%m%d").map(lambda d: roles.get(d, "excluded")) if roles else "all"
    ev["month"] = ev["date"].dt.to_period("M")
    g = ev.groupby(["species", by]).agg(samples=("event", "size"), events=("event", "sum"),
                                        event_days=("date", lambda d: d[ev.loc[d.index, "event"]].nunique()))
    return ev, g


def rank_windows(ev, roles, window_days, step_days=14, mode="scene_split"):
    """Score = min over species of events in the scoring split (test for scene_split, all for resplit)."""
    lo, hi = ev["date"].min(), ev["date"].max()
    rows = []
    for s in pd.date_range(lo, max(lo, hi - pd.Timedelta(days=window_days)), freq=f"{step_days}D"):
        e = s + pd.Timedelta(days=window_days)
        w = ev[(ev["date"] >= s) & (ev["date"] <= e)]
        if mode == "scene_split" and roles:
            w = w[w["date"].dt.strftime("%Y%m%d").map(lambda d: roles.get(d)) == "test"]
        per = w.groupby("species")["event"].sum().reindex(ev["species"].unique(), fill_value=0)
        n = ev[(ev["date"] >= s) & (ev["date"] <= e)].groupby("species").size()
        rows.append({"start": s.date(), "end": e.date(), "score_min_events": int(per.min()),
                     **{f"events_{k}": int(v) for k, v in per.items()},
                     "samples": int(n.max()) if len(n) else 0,
                     "sample_days": int(ev[(ev["date"] >= s) & (ev["date"] <= e)]["date"].nunique())})
    return pd.DataFrame(rows).sort_values(["score_min_events", "samples"], ascending=False)


def resplit(ev, start, end, block_days=14, buffer_days=2, fractions=(0.6, 0.2, 0.2), seed=0,
            scene_roles_map=None, windows=None):
    """
    Day roles for [start, end]: time blocks ranked by event share (summed over
    species, each normalized by its window total), then dealt to roles in groups
    matching `fractions`, so events are spread over all
    splits (stratified random, seeded). Days within buffer_days of a test
    block are excluded. With scene_roles_map, test is given preferentially to
    the blocks with the most scene-split test (encoder/context-held-out) days.
    `windows` ([(start, end), ...]) restricts the days to those windows; blocks
    are numbered within each window so none spans a gap.
    """
    if windows:
        parts, off = [], 0
        for s, e in windows:
            d = pd.date_range(s, e, freq="D")
            b = (d - s).days // block_days
            parts.append(pd.Series(b + off, index=d))
            off += int(b.max()) + 1
        bmap = pd.concat(parts)
        days, blk = bmap.index, bmap.to_numpy()
    else:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        days = pd.date_range(start, end, freq="D")
        blk = ((days - start).days // block_days).to_numpy()
        bmap = pd.Series(blk, index=days)
    w = ev[ev["date"].isin(days)].copy()
    w["block"] = w["date"].map(bmap)
    tot = w.groupby("species")["event"].sum().clip(lower=1)
    share = (w[w["event"]].groupby(["block", "species"]).size() / tot).groupby(level=0).sum()
    nsamp = w.groupby("block").size()
    blocks = pd.DataFrame({"block": np.unique(blk)})
    blocks["share"] = blocks["block"].map(share).fillna(0.0)
    blocks["n"] = blocks["block"].map(nsamp).fillna(0)
    rng = np.random.default_rng(seed)
    blocks["tie"] = rng.random(len(blocks))
    blocks = blocks.sort_values(["share", "n", "tie"], ascending=False).reset_index(drop=True)
    unit = 5 if all(abs(f * 5 - round(f * 5)) < 1e-9 for f in fractions) else 10
    pattern = (["test"] * round(fractions[2] * unit) + ["calibration"] * round(fractions[1] * unit)
               + ["train"] * round(fractions[0] * unit))
    # stratified random: each consecutive group of len(pattern) ranked blocks gets a
    # seeded permutation of the pattern, so all splits see similar event shares
    held = np.zeros(len(blocks))
    if scene_roles_map:
        dd = pd.Series(days.strftime("%Y%m%d"), index=blk)
        frac = dd.map(lambda d: scene_roles_map.get(d) == "test").groupby(level=0).mean()
        held = blocks["block"].map(frac).fillna(0).to_numpy()
    role = []
    for i in range(0, len(blocks), len(pattern)):
        grp = list(rng.permutation(pattern)[: len(blocks) - i])
        n_test = grp.count("test")
        if n_test and held[i:i + len(grp)].max() > 0:
            pick = np.argsort(-held[i:i + len(grp)], kind="stable")[:n_test]
            rest = [r for r in grp if r != "test"]
            grp = ["test" if j in pick else rest.pop(0) for j in range(len(grp))]
        role.extend(grp)
    blocks["held_out_frac"] = held
    blocks["role"] = role
    role_of_block = dict(zip(blocks["block"], blocks["role"]))
    roles = pd.Series([role_of_block[b] for b in blk], index=days)
    if buffer_days > 0:
        # by calendar distance, so a gap between windows is not treated as adjacency
        is_test = roles.eq("test").to_numpy()
        t = np.sort(days[is_test].values.astype("datetime64[D]").astype(np.int64))
        di = days.values.astype("datetime64[D]").astype(np.int64)
        near = np.zeros(len(days), dtype=bool)
        if len(t):
            j = np.clip(np.searchsorted(t, di), 1, len(t)) - 1
            jn = np.clip(j + 1, 0, len(t) - 1)
            near = np.minimum(np.abs(di - t[j]), np.abs(di - t[jn])) <= buffer_days
        roles[(~is_test) & near] = "excluded"
    return roles, blocks


def best_seed(ev, start, end, a, roles=None, windows=None) -> int:
    """Seed whose resplit maximizes min alert events over species x {calibration, test} (ties: more test)."""
    best = None
    for seed in range(a.seed_search):
        r, _ = resplit(ev, start, end, a.block_days, a.buffer_days, tuple(a.fractions), seed, roles,
                       windows=windows)
        m = {d.strftime("%Y%m%d"): v for d, v in r.items()}
        e = ev[ev["event"]]
        t = e.assign(role=e["date"].dt.strftime("%Y%m%d").map(m)).groupby(["species", "role"]).size()
        t = t.unstack(fill_value=0).reindex(columns=["calibration", "test"], fill_value=0)
        t = t.reindex(ev["species"].unique(), fill_value=0)
        key = (int(t.to_numpy().min()), int(t["test"].sum()), -seed)
        if best is None or key > best[0]:
            best = (key, seed)
    return best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--species", nargs="+", default=None, help="default: uq.species_run or the four main products")
    ap.add_argument("--start"), ap.add_argument("--end")
    ap.add_argument("--windows", nargs="+", default=None,
                    help="START:END ... (default uq.date_windows); overrides --start/--end and ranking")
    ap.add_argument("--window-days", type=int, default=150)
    ap.add_argument("--mode", choices=["scene_split", "resplit"], default="scene_split")
    ap.add_argument("--block-days", type=int, default=14)
    ap.add_argument("--buffer-days", type=int, default=2)
    ap.add_argument("--fractions", type=float, nargs=3, default=[0.6, 0.2, 0.2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-search", type=int, default=0,
                    help="try seeds 0..N-1 (resplit) and keep the one maximizing the minimum alert-event "
                         "count over species x {calibration, test}")
    ap.add_argument("--out", default="pilot_date_roles.csv")
    a = ap.parse_args()

    with open(a.yaml) as f:                      # as sit_fuse.utils.read_yaml (no heavy imports)
        yml = yaml.load(f, Loader=yaml.FullLoader)
    if a.windows:
        yml["uq"]["date_windows"] = [w.split(":") for w in a.windows]
    windows = date_windows(yml["uq"]) if yml["uq"].get("date_windows") else None
    species = a.species or yml["uq"].get("species_run") or [
        "Total_Phytoplankton", "Pseudo_nitzschia_seriata_group", "Pseudo_nitzschia_delicatissima_group", "pDA"]
    ev = species_events(yml, species)
    if ev.empty:
        raise SystemExit("no in-situ rows for the requested species")
    roles = scene_roles(yml)
    pd.set_option("display.width", 200), pd.set_option("display.max_rows", 200)

    ev, g = summarize(ev, roles)
    print("\n== thresholds:", ev.groupby("species")["threshold"].first().to_dict())
    print("\n== full range, by scene-split role\n", g.to_string())
    m = ev.groupby(["month", "species"])["event"].agg(["size", "sum"]).unstack("species")
    print("\n== monthly samples (size) / events (sum)\n", m.to_string())

    if windows:
        start, end = windows[0][0], windows[-1][1]
        print("\n== date windows: " + ", ".join(f"{s.date()} .. {e.date()}" for s, e in windows))
        w = ev[ev["date"].isin(window_days(windows))].copy()
        for s, e in windows:
            ww = w[(w["date"] >= s) & (w["date"] <= e)]
            print(f"  {s.date()} .. {e.date()}: samples per species "
                  f"{ww.groupby('species').size().to_dict()}, events {ww.groupby('species')['event'].sum().to_dict()}")
    else:
        print(f"\n== top {a.window_days}-day windows (mode={a.mode})")
        rk = rank_windows(ev, roles, a.window_days, mode=a.mode)
        print(rk.head(8).to_string(index=False))
        start = pd.Timestamp(a.start) if a.start else pd.Timestamp(rk.iloc[0]["start"])
        end = pd.Timestamp(a.end) if a.end else pd.Timestamp(rk.iloc[0]["end"])
        print(f"\n== chosen window {start.date()} .. {end.date()}")
        w = ev[(ev["date"] >= start) & (ev["date"] <= end)].copy()
    if a.mode == "scene_split":
        _, gw = summarize(w, roles)
        print(gw.to_string())
        print("\nYAML: uq.start_date / uq.end_date =", start.date(), end.date(), "(keep scene_split day roles)")
        return
    seed = a.seed
    if a.seed_search > 0:
        seed = best_seed(ev, start, end, a, roles, windows)
        print(f"\n== seed search over {a.seed_search} seeds -> seed {seed}")
    r, blocks = resplit(ev, start, end, a.block_days, a.buffer_days, tuple(a.fractions), seed, roles,
                        windows=windows)
    rmap = {d.strftime("%Y%m%d"): v for d, v in r.items()}
    _, gw = summarize(w, rmap)
    print(gw.to_string())
    if roles:
        tdays = [d for d, v in rmap.items() if v == "test"]
        held = np.mean([roles.get(d) == "test" for d in tdays]) if tdays else 0.0
        print(f"\npilot test days that are scene-split test days (encoder/context-held-out): {held:.0%}")
    pd.DataFrame({"date": list(rmap), "role": list(rmap.values())}).to_csv(a.out, index=False)
    span = ("uq.date_windows as above" if windows else
            f"uq.start_date/end_date = {start.date()} / {end.date()}")
    print(f"\nwrote {a.out}  ->  uq.splits.date_roles_file: {a.out}; {span}")


if __name__ == "__main__":
    main()
