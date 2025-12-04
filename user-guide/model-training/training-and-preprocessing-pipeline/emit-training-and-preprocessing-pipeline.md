---
description: 'To run this pipeline run these commands:'
---

# EMIT Training and Preprocessing Pipeline

See info on associated config files:&#x20;

{% content-ref url="../../code-configuration/training-pipeline/emit-training-pipeline-config-options.md" %}
[emit-training-pipeline-config-options.md](../../code-configuration/training-pipeline/emit-training-pipeline-config-options.md)
{% endcontent-ref %}

```
cd src/sit_fuse/pipelines/training/
python3 emit_preprocess_and_train_pipeline.py -y <path to config file>
e.g. python3 emit_preprocess_and_train_pipeline.py -y ../../config/piplines/training/emit_example_config.yaml
```
