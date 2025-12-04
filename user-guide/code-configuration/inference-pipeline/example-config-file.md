# Example Config File

```
training_config: "../../config/model/goes_tempo_test_3.yaml"

context_config: "../../config/postprocess/goes_tempo_geotiff_gen_4.yaml"


run_uid: "goes_park_smoke_5_min"
context_classes: "smoke"

config_dir: "../../config/"

reuse_gtiffs: False

update_inputs: True

input_dir: "/data/nlahaye/remoteSensing/GOES_TEMPO/new_data/"
input_pattern: "OR*ABI*tif"

run_inference: False

tile_tiers: []

run_context_assigned_geotiff_gen: False
```
