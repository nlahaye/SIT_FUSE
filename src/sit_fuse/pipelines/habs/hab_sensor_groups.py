"""
Sensor groups for the HAB pipelines: one model / one product stream for several
platforms of the same instrument.

  olci  : S3A_OLCI_ERRNT + S3B_OLCI_ERRNT           (Sentinel-3A / 3B OLCI, 11 Rrs bands)
  viirs : SNPP_VIIRS + JPSS1_VIIRS + JPSS2_VIIRS     (Suomi-NPP, NOAA-20, NOAA-21 VIIRS, 5 Rrs bands)

A group is used exactly like a single instrument key in the HAB YAMLs
(training: "OLCI" / "VIIRS"; post-inference / inference / UQ: "olci" / "viirs").
Each scene keeps its own platform prefix (e.g. JPSS1_VIIRS.20240501.L3m.DAY.),
so inference outputs, context-free GeoTIFFs and UQ products stay per scene and
per platform; only the model, the context assignment and the daily merged
product are shared. UQ keeps per-platform sensors (overpass times differ) and
finds their products in the group's out_dir.

Training input for a group (input_oc_dir / input_oc_re may each be a string or a list):

  VIIRS:
    no_trop:
      input_oc_dir: [/mnt/data/REFLECTANCES/SNPP_VIIRS/, /mnt/data/REFLECTANCES/JPSS1_VIIRS/,
                     /mnt/data/REFLECTANCES/JPSS2_VIIRS/]
      input_oc_re: '((?:SNPP|JPSS1|JPSS2)_VIIRS\\.(\\d{8})\\.L3m\\.DAY\\.)RRS.*\\.4km.*nc'
"""

from __future__ import annotations

import re

# platform key -> literal L3m file prefix (same values as INSTRUMENT_PREFIX)
PLATFORM_PREFIX = {
    "s3a": "S3A_OLCI_ERRNT",
    "s3b": "S3B_OLCI_ERRNT",
    "snpp": "SNPP_VIIRS",
    "jpss1": "JPSS1_VIIRS",
    "jpss2": "JPSS2_VIIRS",
}

# group key -> member platforms, in default merge priority (first wins where
# several same-day scenes have valid pixels; override with platform_priority)
SENSOR_GROUPS = {
    "olci": ["s3a", "s3b"],
    "viirs": ["snpp", "jpss1", "jpss2"],
}

# literal tag used for directory names and in-situ input_file_type of a group
GROUP_TAG = {
    "olci": "S3_OLCI_ERRNT",
    "viirs": "VIIRS",
}

# training-side group keys (upper case, like "JPSS2_VIIRS") -> group
TRAINING_GROUP_KEYS = {"OLCI": "olci", "VIIRS": "viirs"}


def group_of(instrument) -> str | None:
    """Group key for a group instrument name (olci / OLCI / viirs / VIIRS), else None."""
    if instrument is None:
        return None
    s = str(instrument)
    if s in SENSOR_GROUPS:
        return s
    return TRAINING_GROUP_KEYS.get(s) or (s.lower() if s.lower() in SENSOR_GROUPS else None)


def group_members(instrument) -> list[str]:
    g = group_of(instrument)
    return list(SENSOR_GROUPS[g]) if g else []


def member_prefixes(instrument, priority=None) -> list[str]:
    """Literal platform prefixes of a group, in merge priority order."""
    members = group_members(instrument)
    if priority:
        order = [p for p in priority if p in members] + [m for m in members if m not in priority]
        members = order
    return [PLATFORM_PREFIX[m] for m in members]


def prefix_regex(instrument, single_prefix: dict) -> str:
    """Regex alternative for a group (non-capturing, so group numbering is unchanged),
    or the single-instrument prefix."""
    if group_of(instrument):
        return "(?:" + "|".join(re.escape(p) for p in member_prefixes(instrument)) + ")"
    return single_prefix[instrument]


def instrument_tag(instrument, single_prefix: dict) -> str:
    """Literal name for output directories / in-situ input_file_type."""
    g = group_of(instrument)
    return GROUP_TAG[g] if g else single_prefix[instrument]


def input_file_type(instrument, single_prefix: dict) -> str:
    """input_file_type for insitu_hab_to_multi_hist: '+'-joined member prefixes for a group."""
    if group_of(instrument):
        return "+".join(member_prefixes(instrument))
    return single_prefix[instrument]


def expand_sensors(names) -> list[str]:
    """Replace group keys by their member platforms (order kept, no duplicates)."""
    out = []
    for n in names or []:
        for s in (group_members(n) or [n]):
            if s not in out:
                out.append(s)
    return out


def instrument_for_sensor(instruments: dict, sensor: str) -> str | None:
    """Instrument key of a HAB YAML that holds this platform's products (itself or its group)."""
    if sensor in instruments:
        return sensor
    for inst in instruments:
        if sensor in group_members(inst):
            return inst
    return None


def oc_sources(entry: dict) -> list[tuple[str, str]]:
    """[(input_oc_dir, input_oc_re)] for one instruments.<inst>.<key> block.

    input_oc_dir and input_oc_re may each be a string or a list; a single regex
    is used for every directory, otherwise the lists are paired."""
    dirs = entry.get("input_oc_dir")
    res = entry.get("input_oc_re")
    if dirs is None:
        return []
    dirs = [dirs] if isinstance(dirs, str) else list(dirs)
    if res is None or isinstance(res, str):
        res = [res] * len(dirs)
    else:
        res = list(res)
        if len(res) == 1:
            res = res * len(dirs)
        if len(res) != len(dirs):
            raise ValueError("input_oc_re must be one regex or one per input_oc_dir")
    return list(zip(dirs, res))
