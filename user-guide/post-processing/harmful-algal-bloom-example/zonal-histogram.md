# Zonal Histogram

keep min thresh around 0.51

make sure label gtiffs is nested list

make sure class name is list

```
cd SIT_FUSE/src/sit_fuse/post_processing
```

```
python3 zonal_histogram.py -y <path_to_yaml>
```

{% code title="merge_algal_jpss1_viirs_n_ca_pseudo_nitzschia_delicatissima_no_heir.yaml" %}
```yaml

data:
 min_thresh: 0.51
 clust_gtiffs: [
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240305.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240307.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240308.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240309.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240310.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240311.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240312.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240313.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240314.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240315.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240316.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240317.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240318.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240319.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240320.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240321.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240322.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240323.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240324.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240325.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240326.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240327.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240328.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240329.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240330.L3m.DAY.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240331.L3m.DAY.tif",
]


 label_gtiffs: [[
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240305.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240307.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240308.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240309.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240310.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240311.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240312.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240313.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240314.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240315.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240316.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240317.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240318.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240319.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240320.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240321.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240322.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240323.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240324.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240325.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240326.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240327.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240328.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240329.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240330.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
"/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/JPSS2_VIIRS.20240331.L3m.DAY.no_heir.pseudo_nitzschia_delicatissima_bloom.tif",
]]


output:
 out_dir: "/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/"
 class_name: ["tiered_hab_pnd"]


```
{% endcode %}
