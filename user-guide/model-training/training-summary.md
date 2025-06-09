---
description: Concise steps to train the model. Make sure the environment is installed.
---

# Summary

For more info on how to use the command line, see [here](https://ubuntu.com/tutorials/command-line-for-beginners#1-overview).

### Training Dataset Preprocessing

```
cd <path_to_SIT_FUSE>/src/sit_fuse/datasets/
# Can be run outside of the repo via command line or in a script as well
```

```
# generate vectorized dataset
python3 sf_dataset.py -y ../config/<folder>/<yaml_file>
# E.g. set <path_to_yaml> to ../config/model/emas_fire_dbn_multi_layer_pl.yaml 
```

The YAML file will have info on the training parameters and the files to be included in the training. Ensure the paths and parameters are up to date.&#x20;

See [code-configuration](../code-configuration/ "mention") for documentation on setting up different YAML files.

To learn more about YAML files, see [this website](https://aws.amazon.com/compare/the-difference-between-yaml-and-json/) and [creating-the-yaml-file.md](../post-processing/zonal-histogram/creating-the-yaml-file.md "mention").

<pre><code><strong># generate tiled dataset
</strong><strong>python3 sf_dataset_conv.py -y &#x3C;path_to_yaml>
</strong># Same YAML file
</code></pre>

### Training Command

```
cd <path_to_SIT_FUSE>/src/sit_fuse/train/
# Same process as above
```

```
# Pretrain encoder
python3 pretrain_encoder.py -y <path_to_yaml>
# Same YAML file as above
```

```
# Finetune encoder and train deep clustering MLP head
python3 finetune_dc_mlp_head.py -y <path_to_yaml>
```

```
# Train heirarchichal deep clustering MLP heads
python3 train_heirarchichal_deep_cluster.py -y <path_to_yaml>
```

### Inference Command

```
cd <path_to_SIT_FUSE>/src/sit_fuse/inference/
# Same process as above
```

```
python3 generate_output.py -y <path_to_yaml>
# Same yaml file as above
```

### Summary

An example training:

<pre><code><strong>cd /app/SIT_FUSE/sit_fuse/datasets
</strong>python3 sf_dataset.py -y ../config/model/gk2a_test_dbn_multi_layer_pl.yaml
python3 sf_dataset_conv.py -y &#x3C;path_to_yaml>
cd /train
python3 pretrain_encoder.py -y ../config/model/gk2a_test_dbn_multi_layer_pl.yaml
python3 finetune_dc_mlp_head.py -y ../config/model/gk2a_test_dbn_multi_layer_pl.yaml
python3 train_heirarchichal_deep_cluster.py -y ../config/model/gk2a_test_dbn_multi_layer_pl.yaml
cd /inference
python3 generate_output.py -y ../config/model/gk2a_test_dbn_multi_layer_pl.yaml
</code></pre>
