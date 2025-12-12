---
description: 'To run this pipeline run these commands:'
---

# Basic Inference Pipeline

See info on associated config files:&#x20;

{% content-ref url="../../code-configuration/inference-pipeline/basic-inference-config-options.md" %}
[basic-inference-config-options.md](../../code-configuration/inference-pipeline/basic-inference-config-options.md)
{% endcontent-ref %}

```
cd src/sit_fuse/pipelines/inference
python3 basic_inference_pipeline.py -y <path to config file>
e.g. python3 basic_inference_pipeline.py -y ../../config/piplines/inference/goes_la_smoke_segmentation.yaml 
```

