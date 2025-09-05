---
description: >-
  Files starting with "fuse" in SIT_FUSE/src/sit_fuse/config/preprocessing/ are
  all relevant examples for this script.
---

# Example Config File

{% code title="fuse_s3a_only_oc_cluster_n_ca.yaml" %}
```yaml
low_res:
 data:
  valid_min: -100.0
  valid_max: 99999999999
  reader_type: "zarr_to_numpy"
  reader_kwargs:
   none: "None"
  geo_reader_type: "zarr_to_numpy"
  geo_reader_kwargs:
   no_arg: "no_arg"
  filenames: [
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240306.L3m.DAY..clust.data_77081clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240307.L3m.DAY..clust.data_54886clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240308.L3m.DAY..clust.data_78299clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240309.L3m.DAY..clust.data_78299clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240310.L3m.DAY..clust.data_78274clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240311.L3m.DAY..clust.data_78299clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240312.L3m.DAY..clust.data_78299clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240313.L3m.DAY..clust.data_78296clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240314.L3m.DAY..clust.data_78262clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240315.L3m.DAY..clust.data_78299clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240316.L3m.DAY..clust.data_78299clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240317.L3m.DAY..clust.data_78290clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240318.L3m.DAY..clust.data_78295clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240319.L3m.DAY..clust.data_78299clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240320.L3m.DAY..clust.data_78295clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240321.L3m.DAY..clust.data_78295clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240323.L3m.DAY..clust.data_78281clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240324.L3m.DAY..clust.data_78295clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240325.L3m.DAY..clust.data_78295clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240326.L3m.DAY..clust.data_78286clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240327.L3m.DAY..clust.data_78247clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240328.L3m.DAY..clust.data_78255clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240329.L3m.DAY..clust.data_78295clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240330.L3m.DAY..clust.data_78296clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240331.L3m.DAY..clust.data_78295clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240401.L3m.DAY..clust.data_78299clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240402.L3m.DAY..clust.data_69599clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240403.L3m.DAY..clust.data_64064clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240404.L3m.DAY..clust.data_64094clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240405.L3m.DAY..clust.data_78278clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240406.L3m.DAY..clust.data_78295clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240407.L3m.DAY..clust.data_69599clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240409.L3m.DAY..clust.data_78299clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240410.L3m.DAY..clust.data_78295clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240411.L3m.DAY..clust.data_78286clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240412.L3m.DAY..clust.data_77098clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240413.L3m.DAY..clust.data_69588clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240414.L3m.DAY..clust.data_69595clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240415.L3m.DAY..clust.data_64064clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240416.L3m.DAY..clust.data_78295clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240417.L3m.DAY..clust.data_78274clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240418.L3m.DAY..clust.data_34956clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240419.L3m.DAY..clust.data_77098clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240420.L3m.DAY..clust.data_78292clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240421.L3m.DAY..clust.data_34998clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240423.L3m.DAY..clust.data_64080clusters.zarr",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/S3A_OLCI_S_CA/S3A_OLCI_ERRNT.20240424.L3m.DAY..clust.data_77096clusters.zarr",


]


  geo_filenames: [
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
"/mnt/data/cal_coast_geo.zarr",
]

  chan_dim: 2
  geo_coord_dim: 2
  geo_lat_index: 0
  geo_lon_index: 1


high_res:
 data:
  valid_min: 0.0
  valid_max: 99999999999
  reader_type: "zarr_to_numpy"
  reader_kwargs:
   no_arg: "no_arg"
  geo_reader_type: "zarr_to_numpy"
  geo_reader_kwargs:
   no_arg: "no_arg"
  filenames: []
  geo_filenames: []
  chan_dim: 0
  geo_coord_dim: 2
  geo_lat_index: 0
  geo_lon_index: 1


fusion:
 projection_id: 4326
 description: "Generating S3A GeoTiffs"
 area_id: "GERD Dam"
 projection_proj4:
  proj: "longlat"
  datum: "WGS84"
 final_resolution: 0.055
 projection_units: "degrees"
 resample_radius: 5000
 resample_n_neighbors: 64
 resample_n_procs: 10
 resample_epsilon: 1.6
 use_bilinear: False




output_files: [
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240306.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240307.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240308.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240309.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240310.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240311.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240312.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240313.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240314.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240315.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240316.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240317.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240318.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240319.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240320.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240321.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240323.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240324.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240325.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240326.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240327.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240328.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240329.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240330.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240331.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240401.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240402.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240403.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240404.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240405.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240406.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240407.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240409.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240410.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240411.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240412.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240413.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240414.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240415.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240416.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240417.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240418.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240419.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240420.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240421.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240423.L3m.DAY.tif",
"/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20240424.L3m.DAY.tif",
]
```
{% endcode %}
