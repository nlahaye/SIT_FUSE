---
description: assigns context using insitu data
---

# Assigning the labels

matching up the insitu data (labels) with the class outputs (like with the colors)

explain no\_heir lmao. sometimes the final layer misses the full context so having no\_heir we can check for false negatives&#x20;

make one for every feature/bloom

```
python3 multi_hist_insitu.py -y <path_to_yaml>
```

{% code title="hab_snpp_viirs_insitu_clust_multiplot_s_ca.yml" %}
```yaml
xl_fname: '/mnt/data/la_hab_data/combined.csv'
start_date: "2024-03-05"
end_date: "2025-06-11"
clusters: 120000
 
clusters_dir: '/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/SNPP_VIIRS_S_CA/'
radius_degrees: [0.09]
global_max: 10000000
ranges: [0,1000,10000,100000,1000000,10000000, 100000000]

input_file_type: 'SNPP_VIIRS'
use_key: ["Total_Phytoplankton", "Alexandrium_spp", "Pseudo_nitzschia_delicatissima_group", "Pseudo_nitzschia_seriata_group"]
```
{% endcode %}

