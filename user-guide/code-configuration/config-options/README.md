---
description: >-
  Top level config options and their descriptions with nested pages detailing
  subsections.
---

# Config Options

<pre data-full-width="true"><code><strong>encoder_type: The type of encoder used in the model architecture. 
</strong>    Valid values are "dbn", "ijepa", and "byol".
    
encoder: Options that are shared across encoder types. 
    Specifics are detailed in page linked below.
    
&#x3C;encoder type specific sections>: Sections named "dbn:", "byol:", and "ijepa:"
    are only needed relative to the encoder_type selected above. Each type has its
    own page linked below.
    
cluster: Options for training hierarchichal deep clustering model. Detailed in page linked below.
    
logger: Options for the logger. Detailed in page linked below.

scaler: Options for the input scaler. Detailed in page linked below.

output: Options for output. Detailed in page linked below.

data: Options for reading in data. Detailed in page linked below.
</code></pre>



{% content-ref url="encoder-config-options/" %}
[encoder-config-options](encoder-config-options/)
{% endcontent-ref %}

{% content-ref url="dbn-specific-config-options.md" %}
[dbn-specific-config-options.md](dbn-specific-config-options.md)
{% endcontent-ref %}

{% content-ref url="byol-specific-config-options.md" %}
[byol-specific-config-options.md](byol-specific-config-options.md)
{% endcontent-ref %}

{% content-ref url="i-jepa-specific-training-options.md" %}
[i-jepa-specific-training-options.md](i-jepa-specific-training-options.md)
{% endcontent-ref %}

{% content-ref url="deep-clustering-config-options/" %}
[deep-clustering-config-options](deep-clustering-config-options/)
{% endcontent-ref %}

{% content-ref url="logger-config-options.md" %}
[logger-config-options.md](logger-config-options.md)
{% endcontent-ref %}

{% content-ref url="scaler-config-options.md" %}
[scaler-config-options.md](scaler-config-options.md)
{% endcontent-ref %}

{% content-ref url="output-config-options.md" %}
[output-config-options.md](output-config-options.md)
{% endcontent-ref %}

{% content-ref url="data-config-options.md" %}
[data-config-options.md](data-config-options.md)
{% endcontent-ref %}
