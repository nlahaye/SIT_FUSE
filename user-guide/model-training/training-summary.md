---
description: Concise steps to train the model. Make sure the environment is installed.
coverY: 0
---

# Full Training Example

An example training:

<pre><code><strong>cd /home/SIT_FUSE/sit_fuse/datasets
</strong>python3 sf_dataset.py -y ../config/model/gk2a_test_dbn_multi_layer_pl.yaml
cd ../train
python3 pretrain_encoder.py -y ../config/model/gk2a_test_dbn_multi_layer_pl.yaml
python3 finetune_dc_mlp_head.py -y ../config/model/gk2a_test_dbn_multi_layer_pl.yaml
python3 train_heirarchichal_deep_cluster.py -y ../config/model/gk2a_test_dbn_multi_layer_pl.yaml
cd ../inference
python3 generate_output.py -y ../config/model/gk2a_test_dbn_multi_layer_pl.yaml
</code></pre>

{% content-ref url="training-dataset-preprocessing-examples/" %}
[training-dataset-preprocessing-examples](training-dataset-preprocessing-examples/)
{% endcontent-ref %}

{% content-ref url="training-command-examples.md" %}
[training-command-examples.md](training-command-examples.md)
{% endcontent-ref %}

{% content-ref url="inference-command.md" %}
[inference-command.md](inference-command.md)
{% endcontent-ref %}
