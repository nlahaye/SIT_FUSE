# Encoder Training Config Options

{% code fullWidth="true" %}
```
learning_rate: The learning rate for the encoder's optimizer. 
    A single float if using BYOL or I-JEPA. A list of floats (one for each RBM) if
    using a DBN.
    
momentum: Optimizer momentum hyperparameter for encoder training. 
    A single float if using BYOL or I-JEPA. A list of floats (one for each RBM) if
    using a DBN.
    
momentum_start_end: The start and end values for varying momentum during 
    the optimization process. A list of size 2 (start and end) if using
    BYOL or I-JEPA. Not used for DBNs.
    
weight_decay: The weight decay hyperparameter for encoder training.
    A single float if using BYOL or I-JEPA. A list of floats (one for each RBM) if
    using a DBN.
    
nesterov_acceleration: A boolean indicating whether or not to use nesterov acceleration
    during the encoder training. A list of floats (one for each RBM) if
    using a DBN. Not currently used for BYOL or I-JEPA.
    
batch_size: The batch size (integer) for the encoder training process.

epochs: The number of epochs to train the model for (integer).

accelerator: Type of accelerator being used for training (string i.e. "gpu", "tpu")

devices: Number of devices to run training on (integer).

gradient_clip_val: Maximum value a gradient can reach without being 
    clipped during training (float).
    
precision: Precision level to be used for values in training computations (string 
    or integer i.e. 32, "16-mixed")
    
save_dir: subset of output_dir to save encoder-specific data to

stratify_data: A subsection that details training data stratification. 
    Options detailed in page linked below.
```
{% endcode %}



{% content-ref url="training-data-stratification-config-options.md" %}
[training-data-stratification-config-options.md](training-data-stratification-config-options.md)
{% endcontent-ref %}

