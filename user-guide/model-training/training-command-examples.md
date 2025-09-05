---
description: Command examples used to train models
---

# Training Command

{% stepper %}
{% step %}
```
cd <path_to_SIT_FUSE>/src/sit_fuse/train/
# Can be run outside of the repo via command line or in a script as well
```
{% endstep %}

{% step %}
```
# Pretrain encoder
python3 pretrain_encoder.py -y <path_to_yaml>
# E.g. set <path_to_yaml> to ../config/model/emas_fire_dbn_multi_layer_pl.yaml 
```
{% endstep %}

{% step %}
```
# Finetune encoder and train deep clustering MLP head
python3 finetune_dc_mlp_head.py -y <path_to_yaml>
# Same YAML file as above
```
{% endstep %}

{% step %}
```
# Train heirarchichal deep clustering MLP heads
python3 train_heirarchichal_deep_cluster.py -y <path_to_yaml>
# Same YAML file as above
```
{% endstep %}
{% endstepper %}

Use the same YAML file as in [training-dataset-preprocessing-examples](training-dataset-preprocessing-examples/ "mention")
