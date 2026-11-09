#!/usr/bin/env python3
"""
sync_l3m_rrs.py -- Sync daily L3m remote-sensing reflectance (Rrs) files from
NASA OB.DAAC (via Earthdata / CMR) into existing per-instrument directories.

For every requested instrument and date range it will:
  1. Inventory what is already on disk (recursively, any layout).
  2. Download standard (refined) daily L3m Rrs files that are missing.
     Where no standard file exists yet, fall back to the NRT file (optional).
  3. For every local NRT file whose standard counterpart is now available,
     download the standard file, verify it, then delete (or archive) the NRT
     file -- and report every replacement (log + CSV + optional webhook).

Supported instruments (key -> CMR collections)
  PACE_OCI     PACE_OCI_L3M_AOP        / PACE_OCI_L3M_AOP_NRT     (v3.2; Rrs lives in the AOP file)
  S3A_OLCI     OLCIS3A_L3m_ERR_RRS     / OLCIS3A_L3m_ERR_RRS_NRT
  S3B_OLCI     OLCIS3B_L3m_ERR_RRS     / OLCIS3B_L3m_ERR_RRS_NRT
  NPP_VIIRS    VIIRSN_L3m_RRS          / VIIRSN_L3m_RRS_NRT       (files are named SNPP_VIIRS.*)
  JPSS1_VIIRS  VIIRSJ1_L3m_RRS         / VIIRSJ1_L3m_RRS_NRT
  JPSS2_VIIRS  VIIRSJ2_L3m_RRS         / VIIRSJ2_L3m_RRS_NRT
  AQUA_MODIS   MODISA_L3m_RRS          / MODISA_L3m_RRS_NRT

Authentication: uses `earthaccess.login()` (~/.netrc, EARTHDATA_USERNAME /
EARTHDATA_PASSWORD env vars, or interactive prompt).

Examples
  python sync_l3m_rrs.py --start 2025-06-01 --end 2025-09-30 \
      --dir PACE_OCI=/data/pace --dir AQUA_MODIS=/data/modisa --dir S3A_OLCI=/data/s3a

  # all instruments from a JSON config {"PACE_OCI": "/data/pace", ...}, preview only
  python sync_l3m_rrs.py --start 2025-06-01 --end 2025-06-30 --config dirs.json --dry-run

  # also upgrade every NRT file on disk, even outside the date range; archive instead of delete
  python sync_l3m_rrs.py --start 2025-06-01 --end 2025-06-30 --config dirs.json \
      --replace-all-nrt --nrt-archive-dir /data/nrt_archive

Requires: pip install earthaccess
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import re
import shutil
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("sync_l3m_rrs")

OBDAAC_PUBLIC = "https://obdaac-tea.earthdatacloud.nasa.gov/ob-cumulus-prod-public/"


# ----------------------------------------------------------------------------
# Instrument configuration
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Instrument:
    key: str
    std_collection: str
    nrt_collection: str
    file_prefix: str  # first token of the OB.DAAC file name
    # Regex applied to the product token (e.g. "Rrs_443"). None = keep all files
    # in the suite (PACE ships every Rrs band in one AOP file).
    product_regex: str | None = r"^Rrs_\d+$"


INSTRUMENTS: dict[str, Instrument] = {
    "PACE_OCI": Instrument("PACE_OCI", "PACE_OCI_L3M_AOP", "PACE_OCI_L3M_AOP_NRT", "PACE_OCI", None),
    "S3A_OLCI": Instrument("S3A_OLCI", "OLCIS3A_L3m_ERR_RRS", "OLCIS3A_L3m_ERR_RRS_NRT", "S3A_OLCI_ERRNT"),
    "S3B_OLCI": Instrument("S3B_OLCI", "OLCIS3B_L3m_ERR_RRS", "OLCIS3B_L3m_ERR_RRS_NRT", "S3B_OLCI_ERRNT"),
    "NPP_VIIRS": Instrument("NPP_VIIRS", "VIIRSN_L3m_RRS", "VIIRSN_L3m_RRS_NRT", "SNPP_VIIRS"),
    "JPSS1_VIIRS": Instrument("JPSS1_VIIRS", "VIIRSJ1_L3m_RRS", "VIIRSJ1_L3m_RRS_NRT", "JPSS1_VIIRS"),
    "JPSS2_VIIRS": Instrument("JPSS2_VIIRS", "VIIRSJ2_L3m_RRS", "VIIRSJ2_L3m_RRS_NRT", "JPSS2_VIIRS"),
    "AQUA_MODIS": Instrument("AQUA_MODIS", "MODISA_L3m_RRS", "MODISA_L3m_RRS_NRT", "AQUA_MODIS"),
}
ALIASES = {"SNPP_VIIRS": "NPP_VIIRS", "VIIRSN": "NPP_VIIRS", "NOAA20_VIIRS": "JPSS1_VIIRS",
           "NOAA21_VIIRS": "JPSS2_VIIRS", "MODISA": "AQUA_MODIS", "PACE": "PACE_OCI"}

# e.g. AQUA_MODIS.20251001.L3m.DAY.RRS.Rrs_443.4km.NRT.nc
#      PACE_OCI.20251001.L3m.DAY.AOP.V3_2.4km.nc
#      PACE_OCI.20240601.L3m.DAY.RRS.V3_0.Rrs.4km.nc      (older PACE naming)
#      S3A_OLCI_ERRNT.20251001.L3m.DAY.RRS.Rrs_665.4km.nc
FNAME_RE = re.compile(
    r"^(?P<prefix>[A-Z0-9_]+)\.(?P<date>\d{8})\.L3m\.DAY\.(?P<suite>[A-Z_]+)\."
    r"(?:(?P<version>V\d+_\d+)\.)?(?:(?P<product>[A-Za-z0-9_]+)\.)?"
    r"(?P<res>4km|9km|0p1deg)(?P<nrt>\.NRT)?\.nc$"
)


@dataclass
class L3mName:
    fname: str
    prefix: str
    date: dt.date
    suite: str
    version: str | None
    product: str | None
    res: str
    nrt: bool

    @property
    def std_name(self) -> str:
        """File name of the standard (non-NRT) counterpart."""
        return self.fname.replace(".NRT.nc", ".nc") if self.nrt else self.fname

    @property
    def slot(self) -> tuple:
        """Version-independent identity: (date, suite, product, res)."""
        return (self.date, self.suite, self.product, self.res)


def parse_name(fname: str) -> L3mName | None:
    m = FNAME_RE.match(fname)
    if not m:
        return None
    d = m.groupdict()
    return L3mName(fname, d["prefix"], dt.datetime.strptime(d["date"], "%Y%m%d").date(),
                   d["suite"], d["version"], d["product"], d["res"], bool(d["nrt"]))


def wanted(name: L3mName, inst: Instrument, res: str, include_aux: bool) -> bool:
    if name.prefix != inst.file_prefix or name.res != res:
        return False
    if inst.key == "PACE_OCI":
        return name.suite == "AOP"
    if name.suite != "RRS":
        return False
    if include_aux or inst.product_regex is None:
        return True
    return bool(name.product and re.match(inst.product_regex, name.product))


# ----------------------------------------------------------------------------
# Remote search
# ----------------------------------------------------------------------------
def search_remote(short_name: str, start: dt.date, end: dt.date, inst: Instrument,
                  res: str, include_aux: bool) -> dict[str, str]:
    """Return {file_name: https_url} for daily L3m files in [start, end]."""
    import earthaccess

    try:
        results = earthaccess.search_data(
            short_name=short_name,
            temporal=(start.isoformat(), end.isoformat() + "T23:59:59"),
            granule_name=f"*.L3m.DAY.*{res}*",
            count=-1,
        )
    except Exception as exc:  # collection missing, CMR hiccup, etc.
        log.warning("  CMR search failed for %s: %s", short_name, exc)
        return {}

    out: dict[str, str] = {}
    for g in results:
        links = [u for u in g.data_links(access="external") if u.endswith(".nc")]
        fname = None
        url = None
        for u in links:
            fname = u.rsplit("/", 1)[-1]
            if "earthdatacloud" in u:  # prefer direct cloud download over OPeNDAP
                url = u
                break
        if fname is None:
            continue
        url = url or OBDAAC_PUBLIC + fname
        name = parse_name(fname)
        if name and start <= name.date <= end and wanted(name, inst, res, include_aux):
            out[fname] = url
    return out


# ----------------------------------------------------------------------------
# Local inventory
# ----------------------------------------------------------------------------
@dataclass
class Inventory:
    std: dict[str, Path] = field(default_factory=dict)     # std file name -> path
    nrt: dict[str, Path] = field(default_factory=dict)     # std-equivalent name -> NRT path
    slots: dict[tuple, list[Path]] = field(default_factory=dict)  # for version checks


def inventory(root: Path, inst: Instrument, res: str, include_aux: bool,
              staging_name: str) -> Inventory:
    inv = Inventory()
    for p in root.rglob("*.L3m.DAY.*.nc"):
        if staging_name in p.parts:
            continue
        name = parse_name(p.name)
        if name is None or name.prefix != inst.file_prefix or name.res != res:
            continue
        # Older PACE RRS.V3_x files are tracked for reporting only.
        if inst.key == "PACE_OCI" and name.suite == "RRS":
            inv.slots.setdefault(("PACE_OLD", name.date, name.res), []).append(p)
            continue
        if not wanted(name, inst, res, include_aux):
            continue
        (inv.nrt if name.nrt else inv.std)[name.std_name] = p
    return inv


def target_dir(root: Path, date: dt.date, layout: str) -> Path:
    if layout == "year":
        return root / f"{date:%Y}"
    if layout == "year/month":
        return root / f"{date:%Y}" / f"{date:%m}"
    if layout == "year/doy":
        return root / f"{date:%Y}" / f"{date:%j}"
    return root


# ----------------------------------------------------------------------------
# Download + verification
# ----------------------------------------------------------------------------
HDF5_SIG = b"\x89HDF\r\n\x1a\n"


def looks_valid(path: Path) -> bool:
    """Cheap integrity check: non-trivial size and NetCDF4/HDF5 signature."""
    try:
        if path.stat().st_size < 1024:
            return False
        with open(path, "rb") as fh:
            head = fh.read(8)
        return head == HDF5_SIG or head[:3] == b"CDF"
    except OSError:
        return False


def download(urls: list[str], staging: Path, threads: int) -> dict[str, Path]:
    """Download URLs into staging; return {file_name: path} for verified files."""
    import earthaccess

    if not urls:
        return {}
    staging.mkdir(parents=True, exist_ok=True)
    earthaccess.download(urls, local_path=str(staging), threads=threads)
    ok: dict[str, Path] = {}
    for u in urls:
        fname = u.rsplit("/", 1)[-1]
        p = staging / fname
        if p.exists() and looks_valid(p):
            ok[fname] = p
        else:
            log.error("  download failed or invalid: %s", fname)
            p.unlink(missing_ok=True)
    return ok


# ----------------------------------------------------------------------------
# Main sync logic
# ----------------------------------------------------------------------------
@dataclass
class Event:
    instrument: str
    action: str  # downloaded_std | downloaded_nrt | replaced_nrt | nrt_retained | failed | outdated_local_version
    file: str
    path: str
    detail: str = ""


def sync_instrument(inst: Instrument, root: Path, args) -> list[Event]:
    events: list[Event] = []
    root.mkdir(parents=True, exist_ok=True)
    staging = root / args.staging_name
    log.info("== %s  (%s)", inst.key, root)

    inv = inventory(root, inst, args.res, args.include_aux, args.staging_name)
    log.info("  local: %d standard, %d NRT files", len(inv.std), len(inv.nrt))

    # Window used for the standard search (optionally widened to cover all local NRT files).
    s_start, s_end = args.start, args.end
    if args.replace_all_nrt and inv.nrt:
        nrt_dates = [parse_name(p.name).date for p in inv.nrt.values()]
        s_start, s_end = min(s_start, min(nrt_dates)), max(s_end, max(nrt_dates))

    std_remote = search_remote(inst.std_collection, s_start, s_end, inst, args.res, args.include_aux)
    log.info("  remote standard files: %d", len(std_remote))
    nrt_remote: dict[str, str] = {}
    if not args.no_nrt:
        nrt_remote = search_remote(inst.nrt_collection, args.start, args.end, inst, args.res, args.include_aux)
        log.info("  remote NRT files: %d", len(nrt_remote))

    in_range = lambda fn: args.start <= parse_name(fn).date <= args.end  # noqa: E731

    to_get_std: list[str] = []   # brand-new standard files
    to_replace: list[str] = []   # standard files that replace a local NRT file
    for fn in sorted(std_remote):
        if fn in inv.std:
            continue
        if fn in inv.nrt:
            to_replace.append(fn)
        elif in_range(fn):
            to_get_std.append(fn)

    to_get_nrt: list[str] = []
    for fn in sorted(nrt_remote):
        std_fn = parse_name(fn).std_name
        if std_fn in std_remote or std_fn in inv.std or std_fn in inv.nrt:
            continue
        to_get_nrt.append(fn)

    for std_fn, p in sorted(inv.nrt.items()):
        if std_fn not in std_remote:
            events.append(Event(inst.key, "nrt_retained", p.name, str(p), "standard not yet available"))

    for key, paths in sorted(inv.slots.items()):
        for p in paths:
            events.append(Event(inst.key, "outdated_local_version", p.name, str(p),
                                "pre-V3_2 PACE RRS file; current product is L3m AOP V3_2"))

    log.info("  plan: %d new standard, %d NRT->standard replacements, %d new NRT",
             len(to_get_std), len(to_replace), len(to_get_nrt))

    if args.dry_run:
        for fn in to_get_std:
            events.append(Event(inst.key, "would_download_std", fn, ""))
        for fn in to_replace:
            events.append(Event(inst.key, "would_replace_nrt", fn, str(inv.nrt[fn])))
        for fn in to_get_nrt:
            events.append(Event(inst.key, "would_download_nrt", fn, ""))
        return events

    # -- Download in chunks so a crash mid-run leaves completed work in place.
    def process(batch: list[str], remote: dict[str, str], kind: str):
        for i in range(0, len(batch), args.chunk):
            chunk = batch[i:i + args.chunk]
            got = download([remote[fn] for fn in chunk], staging, args.threads)
            for fn in chunk:
                if fn not in got:
                    events.append(Event(inst.key, "failed", fn, "", f"{kind} download failed"))
                    continue
                name = parse_name(fn)
                if kind == "replace":
                    nrt_path = inv.nrt[fn]
                    dest = nrt_path.parent / fn  # keep it where the NRT file lived
                else:
                    dest = target_dir(root, name.date, args.layout) / fn
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(got[fn]), dest)

                if kind == "replace":
                    if args.nrt_archive_dir:
                        arch = Path(args.nrt_archive_dir) / inst.key
                        arch.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(nrt_path), arch / nrt_path.name)
                        detail = f"NRT archived to {arch / nrt_path.name}"
                    else:
                        nrt_path.unlink()
                        detail = "NRT deleted"
                    log.info("  REPLACED %s -> %s (%s)", nrt_path.name, fn, detail)
                    events.append(Event(inst.key, "replaced_nrt", fn, str(dest),
                                        f"replaced {nrt_path.name}; {detail}"))
                else:
                    events.append(Event(inst.key, f"downloaded_{kind}", fn, str(dest)))

    process(to_replace, std_remote, "replace")
    process(to_get_std, std_remote, "std")
    process(to_get_nrt, nrt_remote, "nrt")

    shutil.rmtree(staging, ignore_errors=True)
    return events


# ----------------------------------------------------------------------------
# Reporting / notification
# ----------------------------------------------------------------------------
def write_report(events: list[Event], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instrument", "action", "file", "path", "detail"])
        for e in events:
            w.writerow([e.instrument, e.action, e.file, e.path, e.detail])


def summarize(events: list[Event]) -> str:
    counts: dict[tuple[str, str], int] = {}
    for e in events:
        counts[(e.instrument, e.action)] = counts.get((e.instrument, e.action), 0) + 1
    lines = ["L3m Rrs sync summary"]
    for inst in sorted({k[0] for k in counts}):
        parts = [f"{a}={n}" for (i, a), n in sorted(counts.items()) if i == inst]
        lines.append(f"  {inst}: " + ", ".join(parts))
    replaced = [e for e in events if e.action in ("replaced_nrt", "would_replace_nrt")]
    if replaced:
        lines.append(f"NRT -> standard replacements ({len(replaced)}):")
        lines += [f"  [{e.instrument}] {e.detail or e.path} -> {e.file}" for e in replaced[:200]]
        if len(replaced) > 200:
            lines.append(f"  ... and {len(replaced) - 200} more (see report CSV)")
    return "\n".join(lines)


def notify_webhook(url: str, message: str) -> None:
    """POST {"text": message} -- works with Slack / Teams / Mattermost incoming webhooks."""
    try:
        req = urllib.request.Request(url, data=json.dumps({"text": message}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30)
    except Exception as exc:
        log.warning("webhook notification failed: %s", exc)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, type=dt.date.fromisoformat, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, type=dt.date.fromisoformat, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--dir", action="append", default=[], metavar="INSTRUMENT=PATH",
                   help="Data directory per instrument (repeatable). Instruments given here form the set.")
    p.add_argument("--config", help='JSON file {"INSTRUMENT": "/path", ...}; merged with --dir')
    p.add_argument("--instruments", nargs="+",
                   help="Restrict to this subset of the instruments that have directories")
    p.add_argument("--res", default="4km", choices=["4km", "9km", "0p1deg"],
                   help="Spatial resolution (0p1deg is PACE-only). Default 4km")
    p.add_argument("--include-aux", action="store_true",
                   help="Also fetch non-Rrs files of the RRS suite (aot_*, angstrom) for multi-file sensors")
    p.add_argument("--no-nrt", action="store_true", help="Never download NRT files as a fallback")
    p.add_argument("--replace-all-nrt", action="store_true",
                   help="Upgrade every local NRT file, not only those inside --start/--end")
    p.add_argument("--nrt-archive-dir", help="Move replaced NRT files here instead of deleting them")
    p.add_argument("--layout", default="flat", choices=["flat", "year", "year/month", "year/doy"],
                   help="Where NEW files go inside each instrument dir (replacements stay beside the NRT file)")
    p.add_argument("--threads", type=int, default=4, help="Parallel downloads")
    p.add_argument("--chunk", type=int, default=50, help="Files per download batch")
    p.add_argument("--staging-name", default=".l3m_staging", help="Temp subdirectory name")
    p.add_argument("--report", default=None, help="CSV report path (default: ./l3m_rrs_sync_<timestamp>.csv)")
    p.add_argument("--notify-webhook", help="Incoming-webhook URL to receive the summary (Slack etc.)")
    p.add_argument("--dry-run", action="store_true", help="Search and plan only; no downloads or deletions")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.end < args.start:
        p.error("--end is before --start")

    dirs: dict[str, str] = {}
    if args.config:
        dirs.update(json.loads(Path(args.config).read_text()))
    for item in args.dir:
        if "=" not in item:
            p.error(f"--dir expects INSTRUMENT=PATH, got {item!r}")
        k, v = item.split("=", 1)
        dirs[k] = v
    resolved: dict[str, Path] = {}
    for k, v in dirs.items():
        key = ALIASES.get(k.upper(), k.upper())
        if key not in INSTRUMENTS:
            p.error(f"Unknown instrument {k!r}; choose from {', '.join(INSTRUMENTS)}")
        resolved[key] = Path(v).expanduser()
    if args.instruments:
        subset = {ALIASES.get(i.upper(), i.upper()) for i in args.instruments}
        missing = subset - resolved.keys()
        if missing:
            p.error(f"No directory given for: {', '.join(sorted(missing))}")
        resolved = {k: v for k, v in resolved.items() if k in subset}
    if not resolved:
        p.error("Provide at least one --dir INSTRUMENT=PATH (or --config)")
    if args.res == "0p1deg" and set(resolved) - {"PACE_OCI"}:
        p.error("0p1deg is only produced for PACE_OCI")
    args.dirs = resolved
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    import earthaccess

    if not args.dry_run:
        auth = earthaccess.login()  # netrc -> env vars -> interactive
        if not auth or not getattr(auth, "authenticated", True):
            log.error("Earthdata login failed")
            return 2

    events: list[Event] = []
    for key, root in args.dirs.items():
        try:
            events += sync_instrument(INSTRUMENTS[key], root, args)
        except Exception as exc:
            log.exception("%s failed: %s", key, exc)
            events.append(Event(key, "failed", "", str(root), f"instrument error: {exc}"))

    report = Path(args.report or f"l3m_rrs_sync_{dt.datetime.now():%Y%m%dT%H%M%S}.csv")
    write_report(events, report)
    summary = summarize(events)
    print("\n" + summary + f"\nFull report: {report.resolve()}")
    if args.notify_webhook:
        notify_webhook(args.notify_webhook, summary)
    return 1 if any(e.action == "failed" for e in events) else 0


if __name__ == "__main__":
    sys.exit(main())
