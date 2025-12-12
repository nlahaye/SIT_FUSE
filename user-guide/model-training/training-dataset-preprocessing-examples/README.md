---
description: >-
  Command examples used to preprocess training datasets. It will cut down on
  processing time if this is run and a static training dataset is used, instead
  of having to preprocess at all steps.
---

# Training Dataset Preprocessing

**Disclaimer**: If you already have pre-trained model weights, skip to [inference-command.md](../../model-inference/inference-command.md "mention") and [code-configuration.md](code-configuration.md "mention").

***

**Before running our commands, we must have a YAML file updated with the specific paths and parameters needed for training.**

{% hint style="info" %}
See [code-configuration.md](code-configuration.md "mention") for info on setting up the YAML file
{% endhint %}

{% hint style="info" %}
To learn more about YAML files, see [this website](https://aws.amazon.com/compare/the-difference-between-yaml-and-json/) and [creating-the-yaml-file.md](../../post-processing/wildfire-example/zonal-histogram/creating-the-yaml-file.md "mention")
{% endhint %}

{% hint style="info" %}
For more info on how to use the command line, see [here](https://ubuntu.com/tutorials/command-line-for-beginners#1-overview).
{% endhint %}

***

After setting up the YAML file, we can run our commands:

```
cd <path_to_SIT_FUSE>/src/sit_fuse/datasets/
# Can be run outside of the repo via command line or in a script as well
```

{% tabs %}
{% tab title=" Generate vectorized dataset" %}
```
python3 sf_dataset.py -y ../config/<folder>/<yaml_file>
# E.g. set <path_to_yaml> to ../config/model/emas_fire_dbn_multi_layer_pl.yaml 
```

Workstreams that use classic DBNs, PCA-based encoding, or no encoder at all would use vector-based samples.
{% endtab %}

{% tab title="Generate tiled dataset" %}
```
python3 sf_dataset_conv.py -y <path_to_yaml>
```

Workstreams that use convolutional DBNs, CNNs, Vision Transformers would use tiled samples.
{% endtab %}
{% endtabs %}
