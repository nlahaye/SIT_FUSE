# RTDBN-Specific Config Options

RTDBN training supports any number of stacked layers via greedy layer-wise pre-training; see [RTDBN Inference Command](../../../model-inference/rtdbn-inference-command.md) for running inference, FLOPs, and XAI once a model is trained, and [RTDBN Training Command](../../../model-training/rtdbn-training-command.md) for the training scripts. A full example is available at [Example RTDBN Config](../example-configs/example-rtdbn-config.md).

{% code fullWidth="true" %}
```
model_type: A list of RTRBM types to be used (strings - one for each layer in the RTDBN).
    Currently, only "variance_gaussian" is registered for use with RTDBN (bernoulli and fixed-variance
    gaussian RTRBMs exist in Learnergy but aren't wired into RTDBN's model registry). A
    single-layer RTDBN (one entry in n_hidden) is equivalent to a standalone RTVarianceGaussianRBM.

n_visible: Number of visible units (integer; dimensionality of each timestep's input vector).
    Must match the number of features in the data. Training stops with an error if it does not.

n_hidden: A list of hidden layer sizes (integers; one for each RTRBM layer in the RTDBN).

gibbs_steps: A list of the number of gibbs steps to be taken during training of each RTRBM layer.

temp: List of floats (temperature parameter for each RTRBM layer).
```
{% endcode %}

## Encoder Training Options

RTDBN encoder training (`pretrain_rtdbn.py`) reads the following options from the `encoder: training:` section. Options that are lists contain one entry per RTRBM layer. Layers are trained one at a time, and each layer is trained on the output of the layers below it.

{% code fullWidth="true" %}
```
learning_rate: List of floats (learning rate for each RTRBM layer).

momentum: List of floats (SGD momentum for each RTRBM layer).

weight_decay: List of floats (weight decay for each RTRBM layer).

nesterov_accel: List of booleans that notes whether or not to use Nesterov accelerated
    gradient for each RTRBM layer.

warmup_epochs: Integer. The learned per-feature variance (sigma) of the "variance_gaussian"
    layer is frozen for the first warmup_epochs epochs of each layer, and is trained along with
    the other parameters afterwards.

batch_size: Integer (number of windows per batch).

epochs: List of integers (number of epochs for each RTRBM layer). A single integer is
    also accepted and used for every layer.

accelerator: Lightning accelerator to use, e.g. "gpu" or "cpu".

devices: Integer (number of devices to train on).

precision: Lightning training precision, e.g. "32".

gradient_clip_val: Float (gradient clipping value).
```
{% endcode %}

## Data Options

RTDBN data is read from a folder of per-episode NumPy files, and not the image-based readers described under [Data Config Options](data-config-options/). Each file is named `episode_*.npy` and holds a 2-D array of shape (timesteps, features). Overlapping windows of `seq_len` timesteps (stride of one timestep) are cut from each episode; windows never span two episodes, and episodes shorter than `seq_len` are skipped. The data is split into train, validation, and test sets by episode. A StandardScaler (zero mean, unit variance) is fit on the training episodes only for the "variance_gaussian" model type, and is saved to `encoder/scaler.pkl`.

{% code fullWidth="true" %}
```
data_dir: Path to the folder containing the episode_*.npy files.

seq_len: Integer (number of timesteps in each window).

val_percent: Float. Fraction of episodes held out for validation.

test_percent: Float. Fraction of episodes held out for testing. Defaults to 0.0.

split_seed: Integer. Seed for the episode split. Defaults to 42.

num_loader_workers: Integer (number of data loader workers).

fill_value: Float. Value that marks invalid entries; windows containing it are dropped.
```
{% endcode %}

## Deep Clustering Options

RTDBN deep clustering (`pretrain_rtdbn_dc.py`) trains an IIC clustering head on top of the frozen RTDBN encoder. It reads the `cluster:` section.

{% code fullWidth="true" %}
```
num_classes: Integer (number of clusters).

gauss_noise_stdev: List of floats. The first entry is the standard deviation of the Gaussian
    noise used to create the second view of each window for the IIC loss.

lambda: Float (lambda value for the IIC loss).

training: Options for training the head: learning_rate, batch_size, epochs (integers),
    and accelerator, devices, precision, and gradient_clip_val (as above).
```
{% endcode %}

## Output Options

Trained models are written to `out_dir` from the [Output Config Options](output-config-options.md): `encoder/scaler.pkl` and `encoder/rtdbn.ckpt` from encoder training, and `full_model/deep_cluster.ckpt` from deep clustering.
