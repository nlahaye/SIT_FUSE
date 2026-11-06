---
description: Command examples used to train RTDBN models
---

# RTDBN Training Command

RTDBN training uses its own scripts instead of the DBN scripts. The first script pre-trains the stacked RTRBM layers one layer at a time and saves the scaler and encoder weights; the second trains the deep clustering head on top of the frozen encoder. There is no heirarchichal deep clustering step.

See [RTDBN-Specific Config Options](../code-configuration/code-configuration/config-options/rtdbn-specific-config-options.md) for the YAML options and [Example RTDBN Config](../code-configuration/code-configuration/example-configs/example-rtdbn-config.md) for a full example.

{% stepper %}
{% step %}
```
cd <path_to_SIT_FUSE>/src/sit_fuse/train/
```
{% endstep %}

{% step %}
```
# Pretrain RTDBN encoder (layer-wise)
python3 pretrain_rtdbn.py -y <path_to_yaml>
# Writes encoder/scaler.pkl and encoder/rtdbn.ckpt to the out_dir set in the output section of the YAML
```
{% endstep %}

{% step %}
```
# Train the deep clustering head on the frozen encoder
python3 pretrain_rtdbn_dc.py -y <path_to_yaml>
# Same YAML file as above
# Writes full_model/deep_cluster.ckpt to the same out_dir
```
{% endstep %}
{% endstepper %}

Once both steps finish, see [RTDBN Inference Command](../model-inference/rtdbn-inference-command.md) to run inference.
