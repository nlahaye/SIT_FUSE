---
description: >-
  Concise steps to train the models, assuming the docker environment is set up
  and running in the terminal.
---

# Training Summary

### Training Dataset Preprocessing

In the terminal, first type:

```
cd <path_to_SIT_FUSE>/SIT_FUSE/sit_fuse/datasets/
```

An example of the full path would be:

```
app/SIT_FUSE/sit_fuse/datasets/
```

The path and steps to get to the specified folder will depend on the current directory of the terminal.&#x20;

If frequently getting a "No such file or directory" error, try:

```
ls
```

This will allow viewing of the directory contents and help navigation to the desired directory.

Next, type:

```
python3 sf_dataset.py -y ../config/<folder>/<yaml_file>
# Replace the example yaml file with the desired yaml file and update the folder accordingly
```

An example of the path and file would be:

```
../config/model/gk2a_test_dbn_multi_layer_pl.yaml
```

### Training Command

First, type:

```
cd <path_to_SIT_FUSE>/SIT_FUSE/sit_fuse/train/
# Same process as above
```

However, if already in the datasets directory, type:

```
cd ..
cd train
# Do this if already in SIT_FUSE/sit_fuse/datasets/
```

Then, type:

```
python3 pretrain_encoder.py -y <path_to_yaml>
# Use the same yaml file and path as above
```

```
python3 finetune_dc_mlp_head.py -y <path_to_yaml>
```

```
python3 train_heirarchichal_deep_cluster.py -y <path_to_yaml>
```

### Inference Command

First, type:

```
cd <path_to_SIT_FUSE>/SIT_FUSE/sit_fuse/inference/
# Same process as above
```

However, if already in the train directory, type:

```
cd ..
cd inference
# Do this if already in SIT_FUSE/sit_fuse/train/
```

Then, type:

```
python3 generate_output.py -y <path_to_yaml>
# Use the same yaml file and path as above
```

## Summary

An example training and the steps to implement it:

#### Training Dataset Preprocessing

<pre><code><strong>cd /app/SIT_FUSE/sit_fuse/datasets/
</strong></code></pre>

```
python3 sf_dataset.py -y ../config/model/gk2a_test_dbn_multi_layer_pl.yaml
```

#### Training Command

```
cd /train
```

```
python3 pretrain_encoder.py -y ../config/model/gk2a_test_dbn_multi_layer_pl.yaml
```

```
python3 finetune_dc_mlp_head.py -y ../config/model/gk2a_test_dbn_multi_layer_pl.yaml
```

```
python3 train_heirarchichal_deep_cluster.py -y ../config/model/gk2a_test_dbn_multi_layer_pl.yaml
```

#### Inference Command

```
cd /inference
```

```
python3 generate_output.py -y ../config/model/gk2a_test_dbn_multi_layer_pl.yaml
```
