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

## RTDBN Training

RTDBN training uses its own scripts, run from the same directory. The first script pre-trains the stacked RTRBM layers one layer at a time and saves the scaler and encoder weights; the second trains the deep clustering head on top of the frozen encoder. See [RTDBN-Specific Config Options](../code-configuration/code-configuration/config-options/rtdbn-specific-config-options.md) for the YAML options and [Example RTDBN Config](../code-configuration/code-configuration/example-configs/example-rtdbn-config.md) for a full example.

{% stepper %}
{% step %}
```
cd <path_to_SIT_FUSE>/src/sit_fuse/train/
```
{% endstep %}

{% step %}
```
# Pretrain RTDBN encoder (layer-wise); writes encoder/scaler.pkl and encoder/rtdbn.ckpt to output: out_dir
python3 pretrain_rtdbn.py -y <path_to_yaml>
```
{% endstep %}

{% step %}
```
# Train the deep clustering head on the frozen encoder; writes full_model/deep_cluster.ckpt to output: out_dir
python3 pretrain_rtdbn_dc.py -y <path_to_yaml>
# Same YAML file as above
```
{% endstep %}
{% endstepper %}

Once both steps finish, see [RTDBN Inference Command](../model-inference/rtdbn-inference-command.md) to run inference.
