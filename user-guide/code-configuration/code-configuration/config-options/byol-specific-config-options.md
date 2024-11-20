# BYOL-Specific Config Options

{% code fullWidth="true" %}
```
hidden_layer: Hidden layer within defined architecture using the BYOL framework to be used as
    sample representation.
    
projection_size: Starting size of projection of hidden_layer output.

projection_hidden_size: Size of output of MLP layers that project sample representation.

moving_average_decay: EMA decay rate for BYOL learning.
```
{% endcode %}
