---
description: Config options for EMIT Data Preprocessing and Training Pipeline
---

# EMIT Training Pipeline Config Options

To be used with:

{% content-ref url="../../model-training/training-and-preprocessing-pipeline/emit-training-and-preprocessing-pipeline.md" %}
[emit-training-and-preprocessing-pipeline.md](../../model-training/training-and-preprocessing-pipeline/emit-training-and-preprocessing-pipeline.md)
{% endcontent-ref %}

```
file_re: A regular expression (compatible with Python's re library) that will allow the code to identify input files of interest. Group 1 of the regular expression should map to the filename's timestamp. This allows users to specify timestamps associated with training scenes, so the model can split the data accordingly. See the example for further details.

input_dir: The relative or absolute path to the input files

input_glob: A pattern compatible with the glob library to allow input file identification.

train_timestamps: Timestamps associated with the scenes to be used for training. The rest will be put in the "test scenes" list - not to be used for training, but output will be generated for them.

out_dir: The relative or absolute path to the output directory

run_uid: A unique identifier that will be used for generated config files and some output files.

config_dir: The relative or absolute path to the config directory.

```
