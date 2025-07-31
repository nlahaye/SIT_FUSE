#!/bin/bash

YAML_DIR=../config/postprocess/validation/test_train

for f in "$YAML_DIR"/*.yml; do
  if [[ ! ( "$f" == *test* || "$f" == *train* ) || "$f" == *jpss2* ]]; then
  continue
 fi
base=$(basename "$f")

  # Extract satellite (e.g., jpss2 → JPSS2)
  sat=$(echo "$base" | sed -E 's/hab_([^_]+).*/\1/' | tr '[:lower:]' '[:upper:]')

  # Extract split (test/train/full → uppercase)
  split=$(echo "$base" | sed -E 's/.*_([^_]+)_val\.yml/\1/' | tr '[:lower:]' '[:upper:]')

  # Extract region (the token *before* split) → also uppercase
  region=$(echo "$base" | sed -E "s/.*_([^_]+)_${split,,}_val\.yml/\1/" | tr '[:lower:]' '[:upper:]')

  echo "Running validation for $base -> ${sat}_${region}_${split}_VALIDATION.txt"
  python3 multi_hist_insitu.py -y "$f" > "${sat}_${region}_${split}_VALIDATION.txt"
done

done
