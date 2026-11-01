#!/usr/bin/env python3
"""
Encoder-aware, stratified train/calibration/holdout splits for SIT-FUSE
classifier matchups (HMS, ADP, FDCC generators).

Called once by each matchup generator after build_matchups(). Each matched
record gains:

  scene_id            unique per SIT-FUSE scene x product (raster identity)
  sit_fuse_input      the data.files_train / data.files_test entry it came from
  encoder_split       encoder_train | encoder_test | unknown
  split_date          truth date (YYYY-MM-DD) used for grouping
  split_group         grouping unit (date, or N-day block); never split
  split_stratum       stratification key of the group
  classifier_split    dev | calibration | holdout | excluded
  annotation_raster   generic alias of the truth raster (classifier_utils)
  truth_positive_fraction   (optional) used for stratification; computed on
                      pixels outside excluded clusters when invalid_clusters is set
  excluded_cluster_fraction (optional) share of the matched area (valid cluster
                      pixels that overlap valid reference pixels) in excluded
                      clusters (invalid_clusters)
  excluded_scene_fraction   (optional) same share over the whole cluster raster
  head_outputs        (optional) expected export_head_outputs.py raster

Rules
  1. Encoder split: each SIT-FUSE output is traced to its input file. Outputs
     are named basename(files_*[i][0]) + suffix by SIT-FUSE, so the longest
     input basename that prefixes the output name wins. Optional key_patterns
     (e.g., TEMPO "YYYYMMDDTHHMMSSZ_SsssGgg") give a fallback match.
  2. Groups: all records sharing split_group go to one partition. A group is
     eligible for calibration/holdout only if every record is encoder_test and
     the group lies more than buffer_days from any encoder-training date
     (dates parsed from data.files_train, even when no train outputs exist).
  3. Stratification over eligible groups: event window (or year-month),
     optionally crossed with product and with a truth positive-fraction bin.
     Holdout and calibration counts are allocated per stratum (largest
     remainder), with each non-trivial stratum keeping at least one dev group.
  4. Ineligible groups: encoder_train/mixed groups -> dev (or excluded with
     encoder_train_policy: exclude); buffer groups -> dev (or excluded).
  4b. Observing day (optional classifier_split.observing_day): split_date is
     the observing day of the scene time, (t - day_start_utc_hour).date(),
     instead of the reference date, so one daily scan sequence that crosses
     00 UTC (TEMPO over the US: ~11Z to ~01:30Z) stays in one group. Encoder
     training dates use the same rule. link_shared_truth (default false)
     merges groups that share a reference raster, so one reference never
     spans partitions; with daily references matched by UTC date, evening
     (00-02Z) scenes share the next day's raster and chain consecutive days
     into one group. Reference rasters that span partitions are reported. The summary reports the minimum time gap between
     partitions and between calibration/holdout and encoder-training scenes.
  5. Excluded clusters (optional `invalid_clusters` block, same format as the
     UQ config, or `from_uq_config: uq_config.yaml` to share one list): the
     SIT-FUSE cluster raster (sit_fuse_context_free) of each record is read.
     A record is dropped only when its matched area (valid cluster pixels
     overlapping valid reference pixels) has fewer than `min_kept_pixels`
     (default 1) pixels outside excluded clusters, i.e. the matched area is
     entirely excluded; a record whose whole scene is excluded is dropped for
     the same reason. Containing an excluded cluster never drops a record:
     those pixels are removed downstream. `max_scene_fraction` (default null,
     off) optionally also drops records whose matched-area excluded share
     exceeds it. The stratification positive fraction ignores excluded pixels.

Reuses: sit_fuse.utils.read_yaml, classifier_utils.read_raster.
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger("matchup-splits")

ENCODER_TRAIN, ENCODER_TEST, UNKNOWN = "encoder_train", "encoder_test", "unknown"
DEV, CAL, HOLDOUT, EXCLUDED = "dev", "calibration", "holdout", "excluded"

SPLIT_FIELDS = [
    "scene_id", "pair_id", "product", "sit_fuse_context_free", "sit_fuse_input",
    "encoder_split", "scene_time", "truth_date", "split_date", "split_group", "split_stratum",
    "truth_positive_fraction", "excluded_cluster_fraction", "excluded_scene_fraction",
    "classifier_split", "status", "notes",
]


# ---------------------------------------------------------------------------
# Encoder (DBN / cluster) split from the SIT-FUSE training YAML
# ---------------------------------------------------------------------------

def _entry_basenames(entry) -> list[str]:
    files = entry if isinstance(entry, (list, tuple)) else [entry]
    return [os.path.basename(str(f)) for f in files]


def _parse_dates(names, pattern: str) -> set[dt.date]:
    out = set()
    rx = re.compile(pattern)
    for n in names:
        m = rx.search(n)
        if m:
            tok = m.group(1)
            for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y%j"):
                try:
                    out.add(dt.datetime.strptime(tok, fmt).date())
                    break
                except ValueError:
                    continue
    return out


def _scene_time(name: str, pattern: str):
    """Scene datetime (UTC) from a file name; None if the pattern does not match."""
    m = re.search(pattern, os.path.basename(str(name)))
    if not m:
        return None
    tok = m.group(1)
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M", "%Y%m%d%H%M%S", "%Y%j%H%M%S"):
        try:
            return dt.datetime.strptime(tok, fmt)
        except ValueError:
            continue
    return None


def _observing_day(t: dt.datetime, start_hour: float) -> dt.date:
    return (t - dt.timedelta(hours=float(start_hour))).date()


class _Union:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _min_gap_hours(a, b):
    """Minimum |t_a - t_b| in hours between two lists of datetimes."""
    if not a or not b:
        return None
    A = np.array(sorted(x.timestamp() for x in a))
    B = np.array(sorted(x.timestamp() for x in b))
    idx = np.clip(np.searchsorted(A, B), 1, len(A) - 1) if len(A) > 1 else np.zeros(len(B), int)
    d = np.abs(B - A[idx])
    if len(A) > 1:
        d = np.minimum(d, np.abs(B - A[idx - 1]))
    return float(d.min() / 3600.0)


class EncoderSplit:
    """data.files_train / data.files_test of the SIT-FUSE encoder+cluster YAML."""

    def __init__(self, sf_yaml: dict[str, Any], key_patterns=None,
                 date_pattern: str = r"(\d{8})T\d{6}Z"):
        data = sf_yaml["data"]
        self.entries: dict[str, tuple[str, Any]] = {}
        dup = set()
        # train first so that a file listed in both is treated as seen by the encoder
        for role, key in ((ENCODER_TRAIN, "files_train"), (ENCODER_TEST, "files_test")):
            for entry in data.get(key) or []:
                b = _entry_basenames(entry)[0]
                if b in self.entries:
                    dup.add(b)
                    continue
                self.entries[b] = (role, entry)
        if dup:
            LOG.warning("%d inputs appear in both files_train and files_test; "
                        "treated as encoder_train.", len(dup))
        self._by_len = sorted(self.entries, key=len, reverse=True)
        self.key_patterns = [re.compile(p) for p in (key_patterns or [])]
        self._by_key = {}
        for b in self.entries:
            k = self._key(b)
            if k is not None:
                self._by_key.setdefault(k, b)
        self.train_dates = _parse_dates(
            [b for b, (r, _) in self.entries.items() if r == ENCODER_TRAIN], date_pattern)
        self.test_dates = _parse_dates(
            [b for b, (r, _) in self.entries.items() if r == ENCODER_TEST], date_pattern)

    def _key(self, name: str):
        for rx in self.key_patterns:
            m = rx.search(name)
            if m:
                return m.group(1) if m.groups() else m.group(0)
        return None

    def lookup(self, sit_fuse_path) -> tuple[str, Any]:
        name = os.path.basename(str(sit_fuse_path))
        for b in self._by_len:
            if name.startswith(b):
                return self.entries[b]
        k = self._key(name)
        if k is not None and k in self._by_key:
            return self.entries[self._by_key[k]]
        return UNKNOWN, None

    @classmethod
    def from_config(cls, split_cfg: dict[str, Any]) -> "EncoderSplit":
        from sit_fuse.utils import read_yaml
        enc = split_cfg.get("encoder_split", {})
        sf = read_yaml(enc["sit_fuse_yaml"])
        return cls(sf, enc.get("key_patterns"),
                   enc.get("date_pattern", r"(\d{8})T\d{6}Z"))


# ---------------------------------------------------------------------------
# Grouping and strata
# ---------------------------------------------------------------------------

def _as_date(value) -> dt.date:
    """YYYY-MM-DD[...] string, date or datetime -> date."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _group_of(d: dt.date, cfg: dict[str, Any]) -> str:
    mode = str(cfg.get("group_by", "date")).lower()
    if mode == "date":
        return d.isoformat()
    if mode == "block":
        n = int(cfg.get("block_days", 3))
        origin = dt.date.fromisoformat(str(cfg.get("block_origin", "2000-01-01")))
        start = origin + dt.timedelta(days=((d - origin).days // n) * n)
        return f"{start.isoformat()}_{n}d"
    raise ValueError("classifier_split.group_by must be 'date' or 'block'.")


def _event_of(d: dt.date, windows) -> str:
    for w in windows or []:
        if dt.date.fromisoformat(str(w["start"])[:10]) <= d <= dt.date.fromisoformat(str(w["end"])[:10]):
            return str(w["name"])
    return f"{d:%Y-%m}"


def _invalid_clusters_cls():
    try:
        from sit_fuse.train.sf_uq_utils import InvalidClusters
    except ImportError:  # running from a flat directory
        from sf_uq_utils import InvalidClusters
    return InvalidClusters


def cluster_exclusion(cluster_path, invalid, block: dict[str, Any]):
    """
    Excluded mask on the SIT-FUSE cluster grid. Cluster GeoTIFF values are
    joint_label / geotiff_scale; parent exclusion needs num_child_classes
    (joint = child + num_child_classes * parent).
    Returns (excluded bool array, valid bool array, profile).
    """
    from sit_fuse.train.classifier_utils import read_raster
    arr, prof = read_raster(Path(cluster_path))
    arr = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(arr) & (arr >= 0)
    if prof.get("nodata") is not None:
        valid &= arr != prof["nodata"]
    # e.g. [0.0] if the .full_geo.tif zero padding should not count as cluster 0
    for v in block.get("cluster_nodata_values") or []:
        valid &= arr != float(v)
    joint = np.where(valid, np.rint(arr * float(block.get("geotiff_scale", 1000))), -1)
    parent = None
    if invalid.parent:
        nch = block.get("num_child_classes")
        if nch is None:
            raise ValueError("invalid_clusters.parent at the matchup step needs num_child_classes.")
        parent = np.where(valid, joint // int(nch), -1)
    return invalid.mask(joint, parent) & valid, valid, prof


def _to_grid(mask, src_prof, dst_prof):
    """Nearest-neighbour resample of a boolean mask onto another raster grid."""
    if (mask.shape == (dst_prof["height"], dst_prof["width"])
            and (src_prof.get("crs") is None or dst_prof.get("crs") is None
                 or (src_prof["crs"] == dst_prof["crs"] and src_prof["transform"] == dst_prof["transform"]))):
        return mask
    if src_prof.get("crs") is None or dst_prof.get("crs") is None:
        raise ValueError("Cannot align cluster and truth rasters without CRS.")
    from rasterio.warp import Resampling, reproject
    out = np.zeros((dst_prof["height"], dst_prof["width"]), dtype=np.uint8)
    reproject(mask.astype(np.uint8), out, src_transform=src_prof["transform"], src_crs=src_prof["crs"],
              dst_transform=dst_prof["transform"], dst_crs=dst_prof["crs"],
              resampling=Resampling.nearest, src_nodata=None, dst_nodata=0)
    return out.astype(bool)


_TAG_WARNED: set = set()


def reference_nodata_values(prof, configured=(), label_values=(0, 1), mode="auto", path=None):
    """
    NoData values for a reference raster at the matchup step. Configured
    labels win over the raster tag: with mode auto (default) a NoData tag that
    equals a label value (e.g., ADP masks with NoData=0 where 0 is
    background) is ignored; `use` applies it, `ignore` never does.
    """
    labels = {float(v) for v in (label_values or [])}
    nod = [float(v) for v in (configured or [])]
    bad = [v for v in nod if v in labels]
    if bad:
        raise ValueError(f"Reference nodata_values {bad} are also label_values {sorted(labels)}.")
    tag = prof.get("nodata")
    mode = str(mode or "auto").lower()
    if tag is not None and np.isfinite(float(tag)) and mode != "ignore":
        if mode == "auto" and float(tag) in labels:
            if float(tag) not in _TAG_WARNED:
                _TAG_WARNED.add(float(tag))
                LOG.warning("Ignoring reference raster NoData tag %s (e.g. %s): it is a label value. "
                            "Set raster_nodata: use to apply it.", tag,
                            os.path.basename(str(path)) if path else "")
        else:
            nod.append(float(tag))
    return nod


def truth_valid_mask(path, nodata_values=(), label_values=(0, 1), mode="auto"):
    """Valid reference pixels (finite, not nodata) and the reference profile."""
    from sit_fuse.train.classifier_utils import read_raster
    arr, prof = read_raster(Path(path))
    arr = np.asarray(arr)
    ok = np.isfinite(arr)
    for v in reference_nodata_values(prof, nodata_values, label_values, mode, path):
        ok &= arr != v
    return ok, prof


def matched_area_exclusion(ex, ok, cprof, truth_path, nodata_values=(), label_values=(0, 1), mode="auto"):
    """
    Excluded-cluster counts over the matched area on the cluster grid:
    valid cluster pixels that overlap valid reference pixels (reference
    validity resampled to the cluster grid, nearest). Returns
    (n_matched, n_kept) or None when the rasters cannot be aligned.
    """
    try:
        tvalid, tprof = truth_valid_mask(truth_path, nodata_values, label_values, mode)
        tv = _to_grid(tvalid, tprof, cprof)
    except Exception as exc:  # no CRS on one side and different shapes
        LOG.warning("%s: matched area not computed (%s); using the whole scene", truth_path, exc)
        return None
    area = ok & tv
    return int(area.sum()), int((area & ~ex).sum())


def _label_values(pos_cfg):
    """Reference values that are labels (never NoData): label_values, else 0 + positive_values (or 1)."""
    if pos_cfg.get("label_values") is not None:
        return list(pos_cfg["label_values"])
    return [0] + list(pos_cfg.get("positive_values") or [1])


def positive_fraction(path, pos_cfg: dict[str, Any], exclude=None) -> float:
    """
    Fraction of valid truth pixels that are positive (classifier_utils.read_raster).
    exclude: optional (mask, profile) of excluded-cluster pixels on any grid.
    """
    from sit_fuse.train.classifier_utils import read_raster
    arr, prof = read_raster(Path(path))
    arr = np.asarray(arr)
    bad = ~np.isfinite(arr)
    if exclude is not None:
        bad |= _to_grid(exclude[0], exclude[1], prof)
    for v in reference_nodata_values(prof, pos_cfg.get("nodata_values"), _label_values(pos_cfg),
                                     pos_cfg.get("raster_nodata", "auto"), path):
        bad |= arr == v
    valid = arr[~bad]
    if valid.size == 0:
        return float("nan")
    pv = pos_cfg.get("positive_values")
    pos = np.isin(valid, pv) if pv else valid > 0
    return float(pos.mean())


def load_yaml_any(path):
    try:
        from sit_fuse.utils import read_yaml
        return read_yaml(path)
    except Exception:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)


def _largest_remainder(sizes: dict[str, int], frac: float, reserve: dict[str, int],
                       minimum: dict[str, int] | None = None) -> dict[str, int]:
    """
    Allocate round(frac * total) items across strata (largest remainder),
    capped by availability (size - reserve), with an optional per-stratum
    minimum that takes precedence over proportionality.
    """
    total = sum(sizes.values())
    avail = {k: max(0, sizes[k] - reserve.get(k, 0)) for k in sizes}
    raw = {k: frac * sizes[k] for k in sizes}
    alloc = {k: min(avail[k], max(int(math.floor(raw[k])), (minimum or {}).get(k, 0))) for k in sizes}
    target = max(int(round(frac * total)), sum(alloc.values()))
    order = sorted(sizes, key=lambda k: (raw[k] - math.floor(raw[k]), sizes[k]), reverse=True)
    i = 0
    while sum(alloc.values()) < target and any(alloc[k] < avail[k] for k in sizes):
        k = order[i % len(order)]
        if alloc[k] < avail[k]:
            alloc[k] += 1
        i += 1
    return alloc


# ---------------------------------------------------------------------------
# Main entry point used by the matchup generators
# ---------------------------------------------------------------------------

def apply_splits(matchups: list[dict[str, Any]], cfg: dict[str, Any], *,
                 truth_field: str, truth_date_field: str,
                 encoder_split: EncoderSplit | None = None) -> tuple[list[dict], list[dict], dict]:
    """
    cfg: the generator YAML (uses `encoder_split`, `classifier_split`,
    `head_outputs` sections). Returns (kept matchups, split rows, summary).
    """
    enc_cfg = cfg.get("encoder_split", {}) or {}
    cs = cfg.get("classifier_split", {}) or {}
    ho_cfg = cfg.get("head_outputs", {}) or {}
    rows, kept = [], []
    if not enc_cfg.get("enabled", True) and not cs.get("enabled", True):
        return matchups, rows, {}

    es = encoder_split or (EncoderSplit.from_config(cfg) if enc_cfg.get("sit_fuse_yaml") else None)
    unmatched_policy = str(enc_cfg.get("unmatched_policy", "drop")).lower()
    od_cfg = cs.get("observing_day") or {}
    t_pattern = od_cfg.get("time_pattern", r"(\d{8}T\d{6})Z")

    # ---- per-record annotation -------------------------------------------
    seen_ids = Counter()
    for m in matchups:
        role, entry = (es.lookup(m["sit_fuse_context_free"]) if es else (UNKNOWN, None))
        if role == UNKNOWN and es is not None and unmatched_policy == "drop":
            rows.append({**{k: m.get(k, "") for k in SPLIT_FIELDS},
                         "encoder_split": UNKNOWN, "status": "dropped",
                         "notes": "SIT-FUSE output not traceable to files_train/files_test"})
            continue
        m = dict(m)
        m["annotation_raster"] = m[truth_field]
        d = _as_date(m[truth_date_field])
        m["truth_date"] = d.isoformat()
        m["encoder_split"] = role
        if entry is not None:
            m["sit_fuse_input"] = entry
        base = _entry_basenames(entry)[0] if entry is not None else Path(m["sit_fuse_context_free"]).name
        t = _scene_time(base, t_pattern) or _scene_time(m["sit_fuse_context_free"], t_pattern)
        if t is not None:
            m["scene_time"] = t.strftime("%Y-%m-%dT%H:%M:%SZ")
        if od_cfg:
            if t is None:
                raise ValueError(f"observing_day.time_pattern {t_pattern!r} does not match {base!r}.")
            d = _observing_day(t, od_cfg.get("day_start_utc_hour", 0))
        m["split_date"] = d.isoformat()
        sid = f"{m.get('product', 'x')}_{Path(base).stem}"
        seen_ids[sid] += 1
        m["scene_id"] = sid if seen_ids[sid] == 1 else f"{sid}_{seen_ids[sid]:02d}"
        if ho_cfg.get("directory"):
            m["head_outputs"] = str(Path(ho_cfg["directory"]) /
                                    f"{base}{ho_cfg.get('suffix', '.head_outputs.tif')}")
        m["split_group"] = _group_of(d, cs)
        kept.append(m)

    if not kept:
        raise ValueError("No matchups remain after encoder-split tracing.")

    # ---- merge groups that share a reference raster ----------------------
    if cs.get("link_shared_truth", False):
        uf = _Union()
        first = {}
        for m in kept:
            g = m["split_group"]
            uf.find(g)
            key = str(m["annotation_raster"])
            if key in first:
                uf.union(first[key], g)
            else:
                first[key] = g
        members = defaultdict(set)
        for m in kept:
            members[uf.find(m["split_group"])].add(m["split_group"])
        merged = {g: "+".join(sorted(ms)) for ms in members.values() if len(ms) > 1 for g in ms}
        if merged:
            LOG.info("link_shared_truth: merged %d groups into %d (shared reference rasters)",
                     len(merged), len({v for v in merged.values()}))
            for m in kept:
                m["split_group"] = merged.get(m["split_group"], m["split_group"])

    # ---- optional excluded clusters (invalid_clusters) -------------------
    pos_cfg = (cs.get("stratify", {}) or {}).get("positive_fraction", {}) or {}
    inv_block = cfg.get("invalid_clusters") or {}
    excl = {}
    if inv_block:
        IC = _invalid_clusters_cls()
        invalid = IC.from_config(cfg)
        if inv_block.get("from_uq_config"):   # matchup-only keys stay in the generator YAML
            inv_block = {**(load_yaml_any(inv_block["from_uq_config"]).get("invalid_clusters") or {}),
                         **{k: v for k, v in inv_block.items() if k != "from_uq_config"}}
        if invalid.parent and inv_block.get("num_child_classes") is None and enc_cfg.get("sit_fuse_yaml"):
            sf = load_yaml_any(enc_cfg["sit_fuse_yaml"])
            inv_block = {**inv_block, "num_child_classes": int(sf["cluster"]["heir"]["num_classes"])}
            LOG.info("num_child_classes=%d from %s (cluster.heir.num_classes)",
                     inv_block["num_child_classes"], enc_cfg["sit_fuse_yaml"])
        max_frac = inv_block.get("max_scene_fraction")
        min_kept = int(inv_block.get("min_kept_pixels", 1))
        truth_nod = list(inv_block.get("truth_nodata_values") or pos_cfg.get("nodata_values") or [])
        if invalid:
            LOG.info("Excluded clusters at matchup step: %d joint, %d parent", len(invalid.joint), len(invalid.parent))
            still, reasons = [], Counter()
            for m in kept:
                ex, ok, cprof = cluster_exclusion(m["sit_fuse_context_free"], invalid, inv_block)
                n_ok, n_ex = int(ok.sum()), int((ex & ok).sum())
                m["excluded_scene_fraction"] = n_ex / n_ok if n_ok else None
                ma = matched_area_exclusion(ex, ok, cprof, m["annotation_raster"], truth_nod,
                                            _label_values(pos_cfg), pos_cfg.get("raster_nodata", "auto")) \
                    if m.get("annotation_raster") else None
                if ma is None or ma[0] == 0:
                    if ma is not None:
                        LOG.warning("%s: no overlap between valid cluster and reference pixels; "
                                    "excluded share taken over the whole scene", m["scene_id"])
                    n_area, n_kept = n_ok, n_ok - n_ex
                else:
                    n_area, n_kept = ma
                f = (1.0 - n_kept / n_area) if n_area else None
                m["excluded_cluster_fraction"] = f
                note = None
                if n_ok and n_ok - n_ex < min_kept:
                    note = "scene entirely in excluded clusters"
                elif n_area and n_kept < min_kept:
                    note = "matched area entirely in excluded clusters"
                elif max_frac is not None and f is not None and f > float(max_frac):
                    note = f"matched-area excluded fraction {f:.3f} > max_scene_fraction {float(max_frac):.3f}"
                if note:
                    reasons["above max_scene_fraction" if "max_scene" in note else note] += 1
                    rows.append({**{k: m.get(k, "") for k in SPLIT_FIELDS}, "status": "dropped",
                                 "notes": note})
                    continue
                excl[m["scene_id"]] = (ex, cprof)
                still.append(m)
            LOG.info("Excluded clusters: dropped %d / %d records %s", len(kept) - len(still), len(kept),
                     dict(reasons) if reasons else "")
            fr = [m["excluded_cluster_fraction"] for m in kept if m.get("excluded_cluster_fraction") is not None]
            if fr:
                LOG.info("Matched-area excluded fraction: median %.3f, max %.3f", float(np.median(fr)), float(np.max(fr)))
            kept = still
            if not kept:
                raise ValueError(
                    f"No matchups remain after excluded-cluster filtering ({dict(reasons)}). Check that "
                    "invalid_clusters values use the cluster GeoTIFF convention (geotiff_scale), that "
                    "max_scene_fraction is not too low, and the dropped rows in the split CSV.")

    # ---- optional truth positive fraction --------------------------------
    if pos_cfg.get("enabled", False):
        for m in kept:
            m["truth_positive_fraction"] = positive_fraction(m["annotation_raster"], pos_cfg,
                                                             exclude=excl.get(m["scene_id"]))

    # ---- group-level attributes ------------------------------------------
    groups = defaultdict(list)
    for m in kept:
        groups[m["split_group"]].append(m)

    buffer_days = int(cs.get("buffer_days", 1))
    train_dates = set(es.train_dates) if es else set()
    train_times = []
    if es:
        train_times = [t for t in (_scene_time(b, t_pattern) for b, (r, _) in es.entries.items()
                                   if r == ENCODER_TRAIN) if t is not None]
        if od_cfg and train_times:
            train_dates = {_observing_day(t, od_cfg.get("day_start_utc_hour", 0)) for t in train_times}
    train_dates |= {_as_date(m["split_date"]) for m in kept if m["encoder_split"] == ENCODER_TRAIN}
    strat = cs.get("stratify", {}) or {}
    windows = strat.get("event_windows") or []
    by_product = bool(strat.get("by_product", False))

    ginfo = {}
    for g, ms in groups.items():
        roles = {m["encoder_split"] for m in ms}
        dates = {_as_date(m["split_date"]) for m in ms}
        near = any(abs((d - t).days) <= buffer_days for d in dates for t in train_dates) if buffer_days >= 0 else False
        if roles == {ENCODER_TEST}:
            kind = "buffer" if (near and buffer_days > 0) else "eligible"
        elif ENCODER_TRAIN in roles:
            kind = "encoder_train" if roles == {ENCODER_TRAIN} else "mixed"
        else:
            kind = "unknown"
        events = sorted({_event_of(d, windows) for d in dates})
        key = ["+".join(events)]
        if by_product:
            key.append("+".join(sorted({str(m.get("product")) for m in ms})))
        pf = [m.get("truth_positive_fraction") for m in ms if m.get("truth_positive_fraction") is not None]
        pf = [x for x in pf if np.isfinite(x)]
        ginfo[g] = {"kind": kind, "key": key, "posfrac": float(np.mean(pf)) if pf else float("nan")}

    events_all = {i["key"][0] for i in ginfo.values()}
    events_ok = {i["key"][0] for i in ginfo.values() if i["kind"] == "eligible"}
    for ev in sorted(events_all - events_ok):
        LOG.warning("Event/stratum %r has no eligible groups (all encoder_train, mixed or within "
                    "buffer_days); it will not be calibrated or evaluated. Consider group_by: date "
                    "or a smaller buffer_days.", ev)

    # positive-fraction bins computed over eligible groups only (quantiles)
    eligible = sorted(g for g, i in ginfo.items() if i["kind"] == "eligible")
    nb = int(pos_cfg.get("n_bins", 2)) if pos_cfg.get("enabled", False) else 0
    if nb > 1 and eligible:
        vals = np.array([ginfo[g]["posfrac"] for g in eligible])
        fin = vals[np.isfinite(vals)]
        edges = np.unique(np.quantile(fin, np.linspace(0, 1, nb + 1)[1:-1])) if fin.size else np.array([])
        for g in ginfo:
            v = ginfo[g]["posfrac"]
            ginfo[g]["key"].append("pos_na" if not np.isfinite(v) else f"pos_q{int(np.searchsorted(edges, v, side='right'))}")
    for g in ginfo:
        ginfo[g]["stratum"] = "|".join(ginfo[g]["key"])

    # ---- stratified allocation over eligible groups -----------------------
    rng = np.random.default_rng(int(cs.get("random_state", 42)))
    f_ho = float(cs.get("holdout_fraction", 0.2))
    f_cal = float(cs.get("calibration_fraction", 0.15))
    if f_ho + f_cal >= 1.0:
        raise ValueError("holdout_fraction + calibration_fraction must be < 1.")
    by_stratum = defaultdict(list)
    for g in eligible:
        by_stratum[ginfo[g]["stratum"]].append(g)
    sizes = {s: len(v) for s, v in by_stratum.items()}
    etp = str(enc_cfg.get("encoder_train_policy", "dev")).lower()
    bp = str(cs.get("buffer_policy", "dev")).lower()
    # strata that already receive dev groups from ineligible groups
    # (encoder_train / mixed / buffer under the dev policies)
    dev_fed = set()
    for g, i in ginfo.items():
        if (i["kind"] in ("encoder_train", "mixed", "unknown") and etp != "exclude") or \
                (i["kind"] == "buffer" and bp != "exclude"):
            dev_fed.add(i["stratum"])
    # a stratum with >= 2 eligible groups keeps one for dev only when no
    # ineligible group supplies dev data for it (reserve_dev_group: always /
    # auto / never)
    rmode = str(cs.get("reserve_dev_group", "auto")).lower()
    reserve = {s: (1 if n >= 2 and (rmode == "always" or (rmode == "auto" and s not in dev_fed)) else 0)
               for s, n in sizes.items()}
    if bool(cs.get("singleton_strata_to_dev", False)):
        reserve = {s: max(r, 1) for s, r in reserve.items()}
    # per-stratum minimums: every stratum (event / regime) is evaluated and
    # calibrated when it has enough eligible groups
    m_ho = int(cs.get("min_holdout_groups_per_stratum", 1))
    m_cal = int(cs.get("min_calibration_groups_per_stratum", 1))
    n_ho = _largest_remainder(sizes, f_ho, reserve,
                              {k: m_ho for k, n in sizes.items() if n >= 2} if f_ho > 0 else None)
    # holdout first: at least min_holdout_groups_total overall when eligible
    # groups exist (a tiny pool of singleton strata otherwise rounds to 0)
    min_ho_total = int(cs.get("min_holdout_groups_total", 1)) if f_ho > 0 else 0
    while sum(n_ho.values()) < min_ho_total:
        cand = [s for s in sizes if sizes[s] - reserve[s] - n_ho[s] > 0]
        if not cand:
            break
        s_ = max(cand, key=lambda k: sizes[k])
        n_ho[s_] += 1
    reserve2 = {s: reserve[s] + n_ho[s] for s in sizes}
    n_cal = (_largest_remainder(sizes, f_cal, reserve2,
                                {k: m_cal for k, n in sizes.items() if n >= 3})
             if f_cal > 0 else {s: 0 for s in sizes})

    # guarantee a minimum number of calibration groups overall (calibration
    # mode calibration_groups needs at least one): take remaining eligible
    # groups first, then a reserved dev group, then a surplus holdout group
    min_cal_total = int(cs.get("min_calibration_groups_total", 1)) if f_cal > 0 else 0
    n_dev_inelig = sum(1 for i in ginfo.values()
                       if (i["kind"] in ("encoder_train", "mixed", "unknown") and etp != "exclude")
                       or (i["kind"] == "buffer" and bp != "exclude"))
    while sum(n_cal.values()) < min_cal_total:
        free = {s: sizes[s] - reserve[s] - n_ho[s] - n_cal[s] for s in sizes}
        n_dev = n_dev_inelig + sum(free.values()) + sum(reserve.values())
        cand = [s for s in sizes if free[s] > 0] if n_dev - 1 >= 2 else []
        if cand:
            src = "unassigned eligible group"
        else:
            # never leave fewer than two dev groups (grouped CV)
            cand = [s for s in sizes if reserve[s] > 0] if n_dev - 1 >= 2 else []
            src = "reserved dev group"
            if cand:
                s_ = max(cand, key=lambda k: sizes[k])
                reserve[s_] -= 1
                cand = [s_]
            else:
                cand = [s for s in sizes if n_ho[s] >= 2]
                src = "surplus holdout group"
                if cand:
                    s_ = max(cand, key=lambda k: n_ho[k])
                    n_ho[s_] -= 1
                    cand = [s_]
        if not cand:
            LOG.warning("Cannot assign %d calibration group(s) from %d eligible groups: the rest are "
                        "needed for holdout or to keep >= 2 dev groups. Lower holdout_fraction / "
                        "min_holdout_groups_per_stratum or stratify.positive_fraction.n_bins, add "
                        "eligible days, or use calibration.mode: cross_fit.", min_cal_total, len(eligible))
            break
        s_ = max(cand, key=lambda k: sizes[k])
        n_cal[s_] += 1
        LOG.info("Calibration minimum: %s in stratum %s -> calibration", src, s_)

    assign = {}
    for s, gs in by_stratum.items():
        gs = list(rng.permutation(sorted(gs)))
        for i, g in enumerate(gs):
            assign[g] = HOLDOUT if i < n_ho[s] else CAL if i < n_ho[s] + n_cal[s] else DEV
    for g, i in ginfo.items():
        if g in assign:
            continue
        if i["kind"] in ("encoder_train", "mixed", "unknown"):
            assign[g] = EXCLUDED if etp == "exclude" else DEV
        elif i["kind"] == "buffer":
            assign[g] = EXCLUDED if bp == "exclude" else DEV

    # ---- write back, enforce encoder-train policy at record level --------
    out = []
    for m in kept:
        g = m["split_group"]
        m["split_stratum"] = ginfo[g]["stratum"]
        part = assign[g]
        if etp == "exclude" and m["encoder_split"] != ENCODER_TEST:
            part = EXCLUDED
        m["classifier_split"] = part
        rows.append({**{k: m.get(k, "") for k in SPLIT_FIELDS},
                     "sit_fuse_input": str(m.get("sit_fuse_input", "")),
                     "status": "kept" if part != EXCLUDED else "excluded",
                     "notes": ginfo[g]["kind"]})
        if part != EXCLUDED:
            out.append(m)

    summary = summarize_splits(out, ginfo, assign)
    summary["temporal_separation_hours"] = separation_report(out, train_times, cs)
    parts_by_ref = defaultdict(set)
    for m in out:
        parts_by_ref[str(m["annotation_raster"])].add(m["classifier_split"])
    spanning = sorted(r for r, ps in parts_by_ref.items() if len(ps & {CAL, HOLDOUT}) >= 1 and len(ps) > 1)
    summary["reference_rasters_spanning_partitions"] = {
        "n": len(spanning), "examples": [os.path.basename(r) for r in spanning[:5]]}
    if spanning:
        LOG.warning("%d reference raster(s) label scenes in more than one partition, including calibration "
                    "or holdout (e.g., %s). Set classifier_split.link_shared_truth: true to keep each "
                    "reference in one partition.", len(spanning), os.path.basename(spanning[0]))
    if excl or inv_block:
        fr = [m["excluded_cluster_fraction"] for m in out if m.get("excluded_cluster_fraction") is not None]
        summary["invalid_clusters"] = {
            "max_scene_fraction": inv_block.get("max_scene_fraction"),
            "min_kept_pixels": inv_block.get("min_kept_pixels", 1),
            "n_dropped": sum(1 for r in rows if "excluded" in str(r.get("notes", "")) and r.get("status") == "dropped"),
            "mean_excluded_fraction": float(np.mean(fr)) if fr else None,
            "max_excluded_fraction": float(np.max(fr)) if fr else None}
    _check_summary(summary, cs)
    return out, rows, summary


def separation_report(matchups, train_times, cs) -> dict[str, Any]:
    """Minimum time gaps between partitions and to encoder-training scenes."""
    tt = defaultdict(list)
    for m in matchups:
        if m.get("scene_time"):
            tt[m["classifier_split"]].append(dt.datetime.strptime(m["scene_time"], "%Y-%m-%dT%H:%M:%SZ"))
    out = {}
    for a, b in ((DEV, CAL), (DEV, HOLDOUT), (CAL, HOLDOUT)):
        out[f"{a}_vs_{b}"] = _min_gap_hours(tt.get(a, []), tt.get(b, []))
    for part in (CAL, HOLDOUT):
        out[f"{part}_vs_encoder_train_files"] = _min_gap_hours(train_times, tt.get(part, []))
    req = cs.get("min_partition_gap_hours")
    for k, v in out.items():
        if v is not None:
            LOG.info("Minimum gap %-36s %8.2f h", k, v)
            if req is not None and v < float(req):
                LOG.warning("Minimum gap %s = %.2f h < min_partition_gap_hours %.1f", k, v, float(req))
    out["min_partition_gap_hours"] = req
    return out


def summarize_splits(matchups, ginfo, assign) -> dict[str, Any]:
    s: dict[str, Any] = {"n_records": len(matchups), "partitions": {}}
    for part in (DEV, CAL, HOLDOUT):
        ms = [m for m in matchups if m["classifier_split"] == part]
        gs = sorted({m["split_group"] for m in ms})
        pf = [m["truth_positive_fraction"] for m in ms
              if m.get("truth_positive_fraction") is not None and np.isfinite(m["truth_positive_fraction"])]
        s["partitions"][part] = {
            "n_records": len(ms), "n_groups": len(gs), "groups": gs,
            "encoder_split": dict(Counter(m["encoder_split"] for m in ms)),
            "strata": dict(Counter(ginfo[g]["stratum"] for g in gs)),
            "products": dict(Counter(str(m.get("product")) for m in ms)),
            "mean_positive_fraction": float(np.mean(pf)) if pf else None,
        }
    s["group_kinds"] = dict(Counter(i["kind"] for i in ginfo.values()))
    s["excluded_groups"] = sorted(g for g, p in assign.items() if p == EXCLUDED)
    return s


def _check_summary(summary, cs):
    p = summary["partitions"]
    for part, key in ((HOLDOUT, "holdout_fraction"), (CAL, "calibration_fraction")):
        if float(cs.get(key, 0.2 if part == HOLDOUT else 0.15)) > 0 and p[part]["n_groups"] == 0:
            LOG.warning("No eligible groups were assigned to %s. Check encoder split, "
                        "buffer_days and fractions.", part)
        mpf = p[part]["mean_positive_fraction"]
        if mpf is not None and mpf == 0.0:
            LOG.warning("%s partition has no positive truth pixels.", part)
    if p[DEV]["n_groups"] < 2:
        LOG.warning("Fewer than two dev groups: grouped CV will fail downstream.")
    for part in (DEV, CAL, HOLDOUT):
        LOG.info("%-11s records=%4d groups=%3d encoder=%s strata=%s", part,
                 p[part]["n_records"], p[part]["n_groups"], p[part]["encoder_split"], p[part]["strata"])


def write_split_csv(rows: list[dict[str, Any]], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SPLIT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def finalize(matchups, cfg, *, truth_field, truth_date_field, log=None):
    """
    Convenience wrapper for the generators: apply splits, write the split CSV
    (output.split_csv), and return (matchups, summary). No-op when the
    generator YAML has neither `encoder_split` nor `classifier_split`.
    """
    if "encoder_split" not in cfg and "classifier_split" not in cfg:
        return matchups, None
    kept, rows, summary = apply_splits(matchups, cfg, truth_field=truth_field,
                                       truth_date_field=truth_date_field)
    out = cfg.get("output", {})
    csv_path = out.get("split_csv") or str(Path(out["audit_csv"]).with_suffix("")) + "_splits.csv"
    write_split_csv(rows, csv_path)
    (log or LOG).info("Split table: %s", Path(csv_path).resolve())
    return kept, summary
