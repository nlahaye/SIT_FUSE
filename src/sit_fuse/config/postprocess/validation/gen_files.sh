#!/bin/bash

instruments=("aqua_modis" "s3a" "s3b" "snpp" "jpss1" "jpss2")
regions=("al" "pnd" "pns" "ca")
splits=("test" "train")
prefix="hab"

# Output directory
out_dir=./generated_yaml
mkdir -p "$out_dir"

for instrument in "${instruments[@]}"; do
  for region in "${regions[@]}"; do
    for split in "${splits[@]}"; do
      filename="${prefix}_${instrument}_viirs_insitu_clust_multiplot_n_ca_${region}_${split}_val.yml"
      touch "$out_dir/$filename"
      echo "Created: $filename"
    done
  done
done