---
description: YAML configuration for class comparison
icon: bird
---

# Code configuration

{% code fullWidth="true" %}
```yaml
xl_fname: (str) Path to in-situ data (csv or xl)
start_date: (str) Start date of data
end_date: (str) End date of data
clusters: (int) The number of clusters
 
clusters_dir: (str) Path to clusters directory 
radius_degrees: (list) Radius around in-situ data point
# 0.1 = 0.45 km
global_max: (int)
ranges: (list)
input_file_type: (str) Type of file (instrument, no_heir, daily, etc)
### TODO make clearer code config options
use_key: (str) Algal bloom name (must correspond to what's in the in-situ data)
# if no key is used, default to Total_Phytoplankton
```
{% endcode %}

**E.g. configurations:**

{% tabs %}
{% tab title="no_heir_total_phytoplankton" %}
{% code title="hab_jpss2_viirs_insitu_clust_multiplot_no_heir.yml" fullWidth="true" %}
```yaml

xl_fname: '/mnt/data/la_hab_data/combined.csv'
start_date: "2024-03-05"
end_date: "2025-06-11"
clusters: 120000
 
clusters_dir: '/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/'
radius_degrees: [0.09]
global_max: 10000000
ranges: [0,1000,10000,100000,1000000,10000000, 100000000]
input_file_type: 'JPSS2_VIIRS_no_heir'
```
{% endcode %}
{% endtab %}

{% tab title="no_heir_alexandrium bloom" %}
{% code title="hab_jpss2_viirs_insitu_clust_multiplot_s_ca_al.yaml" %}
```yaml
use_key: 'Alexandrium_spp'

xl_fname: '/mnt/data/la_hab_data/combined.csv'
start_date: "2024-03-05"
end_date: "2025-06-11"
clusters: 120000
 
clusters_dir: '/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/'
radius_degrees: [0.09]
global_max: 10000000
ranges: [0,1000,10000,100000,1000000,10000000, 100000000]
input_file_type: 'JPSS2_VIIRS_no_heir'
```
{% endcode %}
{% endtab %}

{% tab title="final_total_phytoplankton" %}
{% code title="hab_jpss2_viirs_insitu_clust_multiplot_s_ca.yml" %}
```yaml

xl_fname: '/mnt/data/la_hab_data/combined.csv'
start_date: "2024-03-05"
end_date: "2025-06-11"
clusters: 120000
 
clusters_dir: '/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS/'
radius_degrees: [0.09]
global_max: 10000000
ranges: [0,1000,10000,100000,1000000,10000000, 100000000]
input_file_type: 'JPSS2_VIIRS'
```
{% endcode %}
{% endtab %}
{% endtabs %}

{% hint style="info" %}
More configurations can be found in `SIT_FUSE/src/sit_fuse/config/postprocess/insitu_class_match/`
{% endhint %}
