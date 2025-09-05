---
description: applys context from last step to create visualizations (tifs)
---

# Creating visualizations

```
python3 generate_cluster_geotiffs.py -y <path_to_yaml>
```

{% code title="merge_algal_jpss1_viirs_n_ca_alexandrium_no_heir.yaml" %}
```yaml
 
input_paths: ["/data/nlahaye/output/Learnergy/No_Data/",
              "/data/nlahaye/output/Learnergy/No_Data/",
              "/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS1_VIIRS_S_CA/"]

 
 
fname_str: "DAY.alexandrium_bloom.tif"
out_dir: "/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/MERGED_HAB/JPSS1_VIIRS/"
num_classes: 9
re_index: 5

gen_daily: true
gen_monthly: false

```
{% endcode %}
