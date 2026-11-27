"""Leave-one-year-out evaluation for the HAB UQ heads (follow-on: held-out year and no-day-gap checks).

The production split assigns day roles in short blocks interleaved through 2023-2026, so slowly varying,
region-wide predictors (monthly climate indices, upwelling, regional discharge, SST) can identify the
period of a test day and borrow labels from adjacent training blocks. Here every day of one calendar year
is test, a buffer of days on either side of that year is excluded, and all other days keep their
production role (production test days become train, so the training share is not reduced).

Subcommands
  roles   date_roles CSV for one held-out year (uq.splits.date_roles_file of the fold's variants)
  pool    concatenate the per-fold test predictions of one variant into one evaluation directory, so
          variant_compare scores every sample once, predicted by the model that never saw its year

  python -m sit_fuse.pipelines.habs.followon.year_holdout roles --src date_roles.csv --year 2024 \\
      --buffer-days 14 --out roles_y2024.csv
  python -m sit_fuse.pipelines.habs.followon.year_holdout pool --fold 2023=/.../prod_y2023/uq \\
      --fold 2024=/.../prod_y2024/uq ... --out /.../pooled/prod/uq

Notes
  Original 'excluded' days stay excluded (buffers, out-of-window or quality-excluded days).
  The SIT-FUSE encoder is self-supervised; it has seen unlabelled imagery from held-out years after 2023
  (its scene split spans 2024-2026), but no labels. 2023 is fully encoder-unseen.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

LOG = logging.getLogger(__name__)
ROLE_MAP = {"val": "calibration"}


def year_roles(src: pd.DataFrame, year: int, buffer_days: int = 14) -> pd.DataFrame:
    col = "role" if "role" in src.columns else "split"
    d = pd.to_datetime(src["date"].astype(str), format="mixed").dt.normalize()
    role = src[col].astype(str).str.strip().str.lower().replace(ROLE_MAP)
    lo, hi = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
    buf = pd.Timedelta(days=int(buffer_days))
    in_year = (d >= lo) & (d <= hi)
    near = ((d >= lo - buf) & (d < lo)) | ((d > hi) & (d <= hi + buf))
    out = role.where(role != "test", "train")                 # other years: production test days -> train
    out = out.mask(near & (role != "excluded"), "excluded")
    out = out.mask(in_year & (role != "excluded"), "test")
    return pd.DataFrame({"date": d.dt.strftime("%Y-%m-%d"), "role": out.to_numpy()})


def cmd_roles(a) -> None:
    src = pd.read_csv(a.src, dtype=str)
    r = year_roles(src, a.year, a.buffer_days)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out + ".tmp"
    r.to_csv(tmp, index=False)
    Path(tmp).replace(a.out)
    n = r["role"].value_counts().to_dict()
    LOG.info("held-out %d -> %s  %s", a.year, a.out, n)
    print(json.dumps({"year": a.year, "out": a.out, "days": n}))


def _no_test_matchups(json_path: Path) -> bool:
    try:
        return json.loads(json_path.read_text()).get("error") == "no test matchups"
    except (OSError, ValueError, AttributeError):
        return False


def cmd_pool(a) -> None:
    folds = dict(f.split("=", 1) for f in a.fold)
    out = Path(a.out) / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    tokens = sorted({p.name[len("hab_uq_test_predictions_"):-4]
                     for d in folds.values() for p in (Path(d) / "evaluation").glob("hab_uq_test_predictions_*.csv")})
    summary = {}
    for tok in tokens:
        parts, missing, empty = [], [], []
        for yr, d in folds.items():
            p = Path(d) / "evaluation" / f"hab_uq_test_predictions_{tok}.csv"
            if not p.exists():
                # a finished fold with no reference samples in its year (e.g. pDA / cDA / Alexandrium in 2026,
                # where the only 2026 rows are IFCB) writes the evaluation JSON with an error and no CSV
                if _no_test_matchups(Path(d) / "evaluation" / f"hab_uq_evaluation_{tok}.json"):
                    empty.append(yr)
                else:
                    missing.append(yr)
                continue
            df = pd.read_csv(p)
            df["holdout_year"] = int(yr)
            if "date" in df:      # only rows of the held-out year (guards against role-file mistakes)
                df = df[pd.to_datetime(df["date"]).dt.year == int(yr)]
            parts.append(df)
        if missing and not a.allow_missing:
            raise FileNotFoundError(f"{tok}: no test predictions for folds {missing} (fold not evaluated: rerun "
                                    f"its train_eval; folds with no test matchups are skipped automatically)")
        if empty:
            LOG.info("%s: no test matchups in folds %s (skipped)", tok, empty)
        if not parts:
            LOG.warning("%s: no fold has test predictions; not pooled", tok)
            continue
        df = pd.concat(parts, ignore_index=True)
        dup = df.duplicated(["sample_uid", "sensor"]).sum() if "sensor" in df else df.duplicated("sample_uid").sum()
        if dup:
            raise ValueError(f"{tok}: {dup} samples predicted by more than one fold")
        p = out / f"hab_uq_test_predictions_{tok}.csv"
        df.to_csv(str(p) + ".tmp", index=False)
        Path(str(p) + ".tmp").replace(p)
        summary[tok] = {"n": len(df), "by_year": df["holdout_year"].value_counts().sort_index().to_dict(),
                        "folds_without_test_matchups": empty, "folds_missing": missing}
    (Path(a.out) / "pool_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, default=str))


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("roles")
    r.add_argument("--src", required=True, help="production date_roles CSV (date, role)")
    r.add_argument("--year", type=int, required=True)
    r.add_argument("--buffer-days", type=int, default=14)
    r.add_argument("--out", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--fold", action="append", required=True, help="YEAR=<uq.out_dir> of the fold's variant")
    p.add_argument("--out", required=True)
    p.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args(argv)
    {"roles": cmd_roles, "pool": cmd_pool}[a.cmd](a)


if __name__ == "__main__":
    main()
