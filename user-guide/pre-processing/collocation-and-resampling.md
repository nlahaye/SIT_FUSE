---
description: Collocation and resampling command used to convert data types.
---

# Collocation and Resampling

Occasionally, data is in a format that isn't ideal for your current purposes. This scripts helps convert one data type to another (e.g. NetCDF to GeoTiff)&#x20;

See [data-combination-colocation-and-resampling](../code-configuration/data-combination-colocation-and-resampling/ "mention") for guidance and examples on how to set up the YAML file(s).

```
cd <path_to_SIT_FUSE>/SIT_FUSE/src/sit_fuse/preprocess
```

```
python3 colocate_and_resample.py -y <path_to_yaml>
# E.g python3 colocate_and_resample.py -y ../config/preprocess/colocate_and_resample/fuse_goes_n_ca.yaml
```
