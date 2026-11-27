"""Temporal stack reader: the current day plus the previous days of a daily L3m product.

Wraps any SIT-FUSE reader (e.g. ``viirs_oc``) and returns, for scene date d, the days
d-(L-1) .. d of the same platform as one (L * (B + 1), H, W) array, time-major
(channel = t * (B + 1) + b, t = 0 is the oldest day, t = L-1 the current day). The extra
channel per day is the number of days between that slot and the observation it holds.

Per pixel:
  * missing days are filled by carrying the last valid value forward (leading gaps are
    back-filled from the first valid day);
  * the pixel is set invalid (-999999 in every channel) when it has no valid
    observation on the current day (``require_current``) or fewer than
    ``min_valid_days`` valid days in the window.

So the scene keeps the valid-pixel support of the current day, and the pixel + neighbourhood
windows SFDataset extracts (pixel_padding) carry the same spatial context as the DBN input.

reader_kwargs (data.reader_kwargs with data.reader_type: temporal_stack):
  base_reader       reader name for one day (required), e.g. "viirs_oc"
  temporal_days     L, window length in days (default 8)
  min_valid_days    minimum valid days per pixel (default 3)
  require_current   current-day observation required (default True)
  add_age           append the days-since-observation channel (default True)
  date_regex        regex with one group for the YYYYMMDD date (default r"\\.(\\d{8})\\.")
  cache_days        per-process cache of single-day reads (default 4096)
  valid_range       [min, max] of valid base-reader values (default [-30000, 64000000],
                    the HAB template valid_min / valid_max)
  all remaining kwargs are passed to the base reader.
"""

from __future__ import annotations

import datetime as _dt
import re
import threading
from collections import OrderedDict

import numpy as np

FILL = -999999.0

_CACHE: "OrderedDict[str, np.ndarray | None]" = OrderedDict()
_LOCK = threading.Lock()

_OWN_KEYS = ("base_reader", "temporal_days", "min_valid_days", "require_current", "add_age",
             "date_regex", "cache_days", "valid_range")


def _shift(entry, date_str: str, new_str: str):
    if isinstance(entry, (list, tuple)):
        return type(entry)(_shift(e, date_str, new_str) for e in entry)
    return entry.replace(date_str, new_str) if isinstance(entry, str) else entry


def _first_str(entry):
    while isinstance(entry, (list, tuple)):
        entry = entry[0]
    return str(entry)


def _key(entry) -> str:
    return repr(entry)


def _read_day(read, entry, base_kwargs, cache_days, required):
    k = _key(entry)
    with _LOCK:
        if k in _CACHE:
            _CACHE.move_to_end(k)
            return _CACHE[k]
    try:
        arr = np.asarray(read(entry, **base_kwargs), dtype=np.float32)
    except Exception:
        if required:
            raise
        arr = None
    with _LOCK:
        _CACHE[k] = arr
        while len(_CACHE) > cache_days:
            _CACHE.popitem(last=False)
    return arr


def read_temporal_stack(filename, **kwargs):
    from sit_fuse.utils import get_read_func

    kw = dict(kwargs)
    base_name = kw.get("base_reader")
    if not base_name:
        raise ValueError("temporal_stack: reader_kwargs.base_reader is required")
    L = int(kw.get("temporal_days", 8))
    min_valid = int(kw.get("min_valid_days", 3))
    require_current = bool(kw.get("require_current", True))
    add_age = bool(kw.get("add_age", True))
    rx = re.compile(kw.get("date_regex", r"\.(\d{8})\."))
    cache_days = int(kw.get("cache_days", 4096))
    vmin, vmax = kw.get("valid_range", [-30000.0, 64000000.0])
    base_kwargs = {k: v for k, v in kw.items() if k not in _OWN_KEYS}
    read = get_read_func(base_name)

    m = rx.search(_first_str(filename))
    if m is None:
        raise ValueError(f"temporal_stack: no YYYYMMDD date in {filename!r}")
    d0 = m.group(1)
    t0 = _dt.datetime.strptime(d0, "%Y%m%d").date()

    days = []
    for lag in range(L - 1, -1, -1):                      # oldest first, current day last
        ds = (t0 - _dt.timedelta(days=lag)).strftime("%Y%m%d")
        days.append(_read_day(read, _shift(filename, d0, ds), base_kwargs, cache_days, required=(lag == 0)))

    cur = days[-1]
    if cur.ndim != 3:
        raise ValueError(f"temporal_stack: base reader must return (bands, rows, cols), got {cur.shape}")
    B, H, W = cur.shape
    stack = np.full((L, B, H, W), np.nan, dtype=np.float32)
    for t, a in enumerate(days):
        if a is not None and a.shape == cur.shape:
            stack[t] = a
    with np.errstate(invalid="ignore"):
        stack[~np.isfinite(stack) | (stack < vmin) | (stack > vmax)] = np.nan
    valid = np.all(np.isfinite(stack), axis=1)            # (L, H, W): every band valid that day

    # source index of the observation used at each slot (forward fill, then back fill)
    tix = np.arange(L)[:, None, None]
    fwd = np.where(valid, tix, -1)
    fwd = np.maximum.accumulate(fwd, axis=0)
    bwd = np.where(valid, tix, L)
    bwd = np.minimum.accumulate(bwd[::-1], axis=0)[::-1]
    src = np.where(fwd >= 0, fwd, bwd)                    # L where the pixel never has data
    has_any = src < L
    src_c = np.clip(src, 0, L - 1)

    filled = np.take_along_axis(stack, src_c[:, None, :, :].repeat(B, axis=1), axis=0)
    age = np.abs(tix - src_c).astype(np.float32)

    keep = valid.sum(axis=0) >= min_valid
    if require_current:
        keep &= valid[-1]
    keep &= has_any.all(axis=0)

    parts = [filled, age[:, None]] if add_age else [filled]
    out = np.concatenate(parts, axis=1)                  # (L, B(+1), H, W)
    out[:, :, ~keep] = FILL
    out[~np.isfinite(out)] = FILL
    return out.reshape(L * out.shape[1], H, W)


def clear_cache():
    with _LOCK:
        _CACHE.clear()
