#!/usr/bin/env bash
# =============================================================================
# GOES-18 smoke and fire UQ, matchups onward. Run from src/sit_fuse/train in
# the SIT-FUSE environment:   bash goes18/run_goes18.sh fire|smoke|all
# Generated matchup YAMLs / audits go to ./goes18/.
# =============================================================================
set -euo pipefail
WHAT=${1:-all}
MODELS="random_forest logistic_regression hist_gradient_boosting hard_assignment soft_assignment assignment_ensemble"
G=goes18

run_fire() {
  # 0. (once) LA scenes through the Park-trained fire encoder, if not done:
  #    python ../inference/generate_output.py -y ../config/model/goes_la_fire_infer.yaml
  #    ls /data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED_FIRE/ | grep full_geo | head   # check names

  # 1. Matchups: FDCC primary (with splits), then the attached references
  python build_fdcc_matchups.py                $G/g18_fdcc_matchup.yaml
  python firms_fire_front_matchup_generator.py $G/g18_firms_viirs_matchup.yaml
  python firms_fire_front_matchup_generator.py $G/g18_firms_modis_landsat_matchup.yaml
  python build_simple_classifier_matchups.py   $G/g18_hms_fire_matchup.yaml

  # 2. Attach (records keep the FDCC split)
  python attach_references.py $G/g18_fdcc_fire_classifier_matchups.yaml $G/g18_output_fdcc_hms_firms.yaml \
      --add firms_viirs=$G/g18_firms_viirs_classifier_matchups.yaml \
      --add firms_modis_landsat=$G/g18_firms_modis_landsat_classifier_matchups.yaml \
      --add hms_fire=$G/g18_hms_fire_classifier_matchups.yaml

  # 3. Head outputs for every matched scene (Park + LA)
  python export_head_outputs.py $G/g18_export_fire.yaml

  # 4-7. Features, training + calibration, inference, evaluation
  python build_uq_feature_table.py $G/g18_uq_config_fire.yaml
  python simple_classifiers_uq.py  $G/g18_uq_config_fire.yaml
  for m in $MODELS; do python classifier_inference_uq.py $G/g18_uq_config_fire.yaml --model "$m"; done
  python evaluate_uq.py $G/g18_uq_config_fire.yaml --models $MODELS

  # 8. Optional products / figures
  python export_geo_products.py $G/g18_uq_config_fire.yaml
  # python make_uq_visuals.py <visuals config pointing at UQ_FIRE>
}

run_smoke() {
  # 0. LA smoke inference is goes_la.yaml (already run with the same model/out_dir)
  #    ls /data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/ | grep full_geo | head

  # 1. Matchups: TEMPO ADP primary (with splits), then VIIRS and HMS
  python adp_matchup_generator.py            $G/g18_adp_matchup.yaml
  python adp_matchup_generator.py            $G/g18_viirs_smoke_matchup.yaml
  python build_simple_classifier_matchups.py $G/g18_hms_smoke_matchup.yaml

  # 2. Attach (records keep the ADP split)
  python attach_references.py $G/g18_adp_smoke_classifier_matchups.yaml $G/g18_adp_multi_ref.yaml \
      --add viirs_aerosol_smoke=$G/g18_viirs_smoke_classifier_matchups.yaml \
      --add hms_smoke=$G/g18_hms_smoke_classifier_matchups.yaml

  # 3-7.
  python export_head_outputs.py    $G/g18_export_smoke.yaml
  python build_uq_feature_table.py $G/g18_uq_config_smoke.yaml
  python simple_classifiers_uq.py  $G/g18_uq_config_smoke.yaml
  for m in $MODELS; do python classifier_inference_uq.py $G/g18_uq_config_smoke.yaml --model "$m"; done
  python evaluate_uq.py $G/g18_uq_config_smoke.yaml --models $MODELS

  # 8. Optional
  python export_geo_products.py $G/g18_uq_config_smoke.yaml
}

case "$WHAT" in
  fire)  run_fire ;;
  smoke) run_smoke ;;
  all)   run_fire; run_smoke ;;
  *) echo "usage: $0 fire|smoke|all"; exit 1 ;;
esac
