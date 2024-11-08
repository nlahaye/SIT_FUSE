# DBN-Specific Config Options

{% code fullWidth="true" %}
```
model_type: A list of RBM types to be used (strings - one for each RBM in DBN). See Learnergy documentation
    For further information on this, typically "gaussian_selu" is used.
    
dbn_arch: A list of hidden layer shapes (integers - one for each RBM in DBN).

gibbs_steps: A list of the number of gibbs steps to be taken during training of each RBM.

temp: List of floats (temperature parameter for each RBM).

normalize_learnergy: List of booleans that notes whether or not to use the normalization functionality
    Within Learnergy.
    
batch_normalize: List of booleans that notes whether or not to use the batch normalization functionality
    Within Learnergy.
```
{% endcode %}
