# RTDBN-Specific Config Options

RTDBN training supports any number of stacked layers via greedy layer-wise pre-training; see [RTDBN Inference Command](../../../model-inference/rtdbn-inference-command.md) for running inference, FLOPs, and XAI once a model is trained.

{% code fullWidth="true" %}
```
model_type: A list of RTRBM types to be used (strings - one for each layer in the RTDBN).
    Currently, only "variance_gaussian" is registered for use with RTDBN (bernoulli and fixed-variance
    gaussian RTRBMs exist in Learnergy but aren't wired into RTDBN's model registry). A
    single-layer RTDBN (one entry in n_hidden) is equivalent to a standalone RTVarianceGaussianRBM.

n_visible: Number of visible units (integer; dimensionality of each timestep's input vector).

n_hidden: A list of hidden layer sizes (integers; one for each RTRBM layer in the RTDBN).

gibbs_steps: A list of the number of gibbs steps to be taken during training of each RTRBM layer.

temp: List of floats (temperature parameter for each RTRBM layer).
```
{% endcode %}
