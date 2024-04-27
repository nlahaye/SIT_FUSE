# Hierarchichal Deep Clustering Training Config Options

{% code fullWidth="true" %}
```
learning_rate: The learning rate for the encoder's optimizer. 
    A single float if using BYOL or I-JEPA. A list of floats (one for each RBM) if
    using a DBN.
    
batch_size: The batch size (integer) for the encoder training process.

epochs: The number of epochs to train the model for (integer).

accelerator: Type of accelerator being used for training (string i.e. "gpu", "tpu")

devices: Number of devices to run training on (integer).

gradient_clip_val: Maximum value a gradient can reach without being 
    clipped during training (float).
    
precision: Precision level to be used for values in training computations (string 
    or integer i.e. 32, "16-mixed")
    
save_dir: subset of output_dir to save encoder-specific data to

```
{% endcode %}



{% content-ref url="../../encoder-config-options/encoder-training-config-options/training-data-stratification-config-options.md" %}
[training-data-stratification-config-options.md](../../encoder-config-options/encoder-training-config-options/training-data-stratification-config-options.md)
{% endcontent-ref %}

