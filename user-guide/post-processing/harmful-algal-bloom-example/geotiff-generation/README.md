# GeoTiff generation

code config for&#x20;

specify how this gtiff gen is different from the other one...

`config/postprocess/geotiff_gen/hab/aqua_modis_only_oc_geotiff_gen_pad_n_ca.yaml`

* needs clusters. previous page will show JPSS2 and GOES cluster gen. plug clusters into here
* use the tifs generated from the collocation and resampling page. gotta do that after training/generate\_output

```
cd SIT_FUSE/src/sit_fuse/postprocessing 
```

```
python3 generate_cluster_geotiffs.py -y <path_to_yaml>
```
