# Example RTDBN Config

{% code fullWidth="true" %}
```yaml
data:
  data_dir: "<path_to_episode_npy_files>"
  seq_len: 20
  val_percent: 0.15
  test_percent: 0.15
  split_seed: 42
  num_loader_workers: 0
  fill_value: -999999.0

output:
  out_dir: "<path_to_output_directory>"
  generate_intermediate_output: True

rtdbn:
  model_type: ["variance_gaussian"]
  n_visible: 80
  n_hidden: [64]
  gibbs_steps: [1]
  temp: [1.0]

encoder_type: "rtdbn"
encoder:
  training:
    learning_rate: [0.001]
    momentum: [0.0]
    weight_decay: [0.0]
    nesterov_accel: [False]
    warmup_epochs: 15
    batch_size: 32
    epochs: [30]
    accelerator: "gpu"
    devices: 1
    gradient_clip_val: 1.0
    precision: "32"

cluster:
  num_classes: 10
  gauss_noise_stdev: [0.01]
  lambda: 1.0
  training:
    learning_rate: 0.001
    batch_size: 32
    epochs: 30
    accelerator: "gpu"
    devices: 1
    gradient_clip_val: 1.0
    precision: "32"

logger:
  use_wandb: False
```
{% endcode %}
