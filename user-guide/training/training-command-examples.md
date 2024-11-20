---
description: Command examples used to train models
---

# Training Command Examples

Make sure config files have been updated to contain all desired paths and parameterizations.

```
cd <path_to_SIT_FUSE>/SIT_FUSE/sit_fuse/train/
# Can be run outside of the repo via command line or in a scripted fashion as well
```

<pre><code><strong># Pretrain encoder
</strong><strong>python3 pretrain_encoder.py -y &#x3C;path_to_yaml>
</strong># i.e. set &#x3C;path_to_yaml> to ../config/model/emas_fire_dbn_multi_layer_pl.yaml 
</code></pre>

```
# Finetune encoder and train deep clustering MLP head
python3 finetune_dc_mlp_head.py -y <path_to_yaml>
# i.e. set <path_to_yaml> to ../config/model/emas_fire_dbn_multi_layer_pl.yaml 
```

```
# Train heirarchichal deep clustering MLP heads
python3 train_heirarchichal_deep_cluster.py -y <path_to_yaml>
# i.e. set <path_to_yaml> to ../config/model/emas_fire_dbn_multi_layer_pl.yaml 
```
