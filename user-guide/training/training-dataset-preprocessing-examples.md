---
description: >-
  Command examples used to preprocess training datasets. It will cut down on
  processing time if this is run and a static training dataset is used, instead
  of having to preprocess at all steps.
---

# Training Dataset Preprocessing Examples

Make sure config files have been updated to contain all desired paths and parameterizations.

```
cd <path_to_SIT_FUSE>/SIT_FUSE/sit_fuse/datasets/ 
# Can be run outside of the repo via command line or in a script as well
```

<pre><code><strong># generate vectorized dataset
</strong><strong>python3 sf_dataset.py -y &#x3C;path_to_yaml>
</strong># i.e. set &#x3C;path_to_yaml> to ../config/model/emas_fire_dbn_multi_layer_pl.yaml 
</code></pre>

<pre><code><strong># generate tiled dataset
</strong><strong>python3 sf_dataset_conv.py -y &#x3C;path_to_yaml>
</strong># i.e. set &#x3C;path_to_yaml> to ../config/model/emas_fire_dbn_multi_layer_pl.yaml 
</code></pre>
