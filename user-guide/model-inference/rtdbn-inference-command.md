---
description: Command examples used to inference RTDBN models
---

# RTDBN Inference Command

RTDBN's inference flow is simpler than DBN's, as there is no `Heir_DC` step, no GeoTiff/context assignment, and no `.npy` index-file setup.

If you followed the RTDBN training steps, the model weights will already be in the correct format in `output_dir`:

```
encoder/scaler.pkl, encoder/rtdbn.ckpt, full_model/deep_cluster.ckpt
```

Set `encoder_type: rtdbn` in the YAML, then run:

```
cd <path_to_SIT_FUSE>/SIT_FUSE/src/sit_fuse/inference
```

```
python3 generate_output.py -y <path_to_yaml>
# Same YAML as in the previous steps. See RTDBN-Specific Config Options
```

Outputs a CSV per split (`rtdbn_test_clusters.csv`, and `rtdbn_val_clusters.csv` if `generate_intermediate_output` is set), with `trial_idx`, `timestep`, `cluster`, and `probability` columns; one row per window. RTDBN windows do not have a spatial coordinate, so this replaces the GeoTiff/zarr output that DBN produces.

The same `encoder_type: rtdbn` YAML also works with `calculate_flops.py` and `xai.py`, run the same way from their directories.
