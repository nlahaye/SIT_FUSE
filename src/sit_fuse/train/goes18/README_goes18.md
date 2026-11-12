# GOES-18 smoke and fire UQ: configs and run order

Run from `src/sit_fuse/train`: `bash goes18/run_goes18.sh fire|smoke|all`.

## Changed code (copy over the existing files in `train/`)
- `sf_uq_utils.py`, `simple_classifiers_uq.py`: configurable `calibration.threshold_grid` (null = legacy 0.05-0.95; list; or `{scale: log, min, max, n, include_default}`); warning when the chosen threshold sits at the grid floor; grid recorded in metrics. No feature-table rebuild needed.
- `build_fdcc_matchups.py`, `adp_matchup_generator.py`, `build_simple_classifier_matchups.py`: ABI timestamps (`_s(\d{13})\d_e` -> `%Y%j%H%M%S`).
- `matchup_splits.py`: `encoder_split.inference_yamls` (inputs of inference-only YAMLs are encoder_test); 7-digit `YYYYJJJ` and 13-digit `YYYYJJJHHMMSS` tokens parsed as day-of-year.

## New configs
| Product | Primary (splits) | Attached | Export | UQ |
|---|---|---|---|---|
| Fire | `g18_fdcc_matchup.yaml` | `g18_firms_viirs_matchup.yaml` (consensus), `g18_firms_modis_landsat_matchup.yaml`, `g18_hms_fire_matchup.yaml` | `g18_export_fire.yaml` | `g18_uq_config_fire.yaml` |
| Smoke | `g18_adp_matchup.yaml` (TEMPO ADP) | `g18_viirs_smoke_matchup.yaml` (consensus), `g18_hms_smoke_matchup.yaml` | `g18_export_smoke.yaml` | `g18_uq_config_smoke.yaml` |

`config/model/goes_la_fire_infer.yaml`: the Park-trained fire encoder applied to the 237 unique LA scenes (inference only). Run it before the fire matchups if the LA fire outputs do not exist yet.

## Splits
Encoder-training days are 2024-07-26 and 07-30 for both GOES encoders, so 07-26 cannot be holdout. Pinned: holdout 2024-07-27, 2025-01-11, 2025-01-12; calibration 2024-07-29, 2025-01-09; all other eligible days dev.

## Check before running
- SIT-FUSE output names: `ls <out_dir> | grep full_geo | head` should show `OR_ABI-L1b-RadC-M6C01_G18_s<13+1 digits>_e...full_geo.tif`.
- FIRMS `tag` values: `tag.value_counts()` of each `fire_front_index.csv`; edit `tags` if the labels differ.
- `invalid_clusters.joint` is empty in both UQ configs; add GOES artifact clusters after inspecting the outputs.
- FDCC and the FIRMS rasters were built for TEMPO (0.05 deg FDCC; FIRMS grid as configured). For ~2 km GOES pixels, re-pull FDCC at 0.01 deg and rasterize FIRMS at <= 0.01 deg.
- FDCC is derived from the same ABI scans, and TEMPO ADP partly uses ABI; FIRMS VIIRS, VIIRS aerosol type and HMS are the independent references.
