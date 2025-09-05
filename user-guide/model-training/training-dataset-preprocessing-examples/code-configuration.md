---
description: YAML configuration for training process
---

# Code configuration

Visit [code-configuration](../../code-configuration/code-configuration/ "mention") for explicit documentation of configuration options and examples to follow.

***

### **Specific configurations to keep in mind:**

{% tabs %}
{% tab title="First Tab" %}
For `file_train`,

* If you're following the training steps, split the training and testing data 50/50.
* If you're using pre-trained model weights, make sure at least one file is in the training list.

For `files_train` and `files_test` , you can exclude the file extension (especially if there's a file for each band), as the code will append one based on the `reader_type` selected.

* For example, if you have 5 NetCDF files corresponding to one day:

```
SNPP_VIIRS.20250623.L3m.DAY.RRS.Rrs_410.4km.NRT.nc
SNPP_VIIRS.20250623.L3m.DAY.RRS.Rrs_443.4km.NRT.nc
SNPP_VIIRS.20250623.L3m.DAY.RRS.Rrs_486.4km.NRT.nc
SNPP_VIIRS.20250623.L3m.DAY.RRS.Rrs_551.4km.NRT.nc
SNPP_VIIRS.20250623.L3m.DAY.RRS.Rrs_671.4km.NRT.nc
```

* Instead of including them all in the YAML, simply include:

```
SNPP_VIIRS.20250623.L3m.DAY.
```
{% endtab %}

{% tab title="Second Tab" %}
For `output_dir`, set this to the path where the model weights are located (e.g. `home/data/MODEL_WEIGHTS`)
{% endtab %}
{% endtabs %}

More specific readers and reader keyword arguments can be found in the above link.

***

### **E.g. configurations**

{% tabs %}
{% tab title="Training from scratch" %}
{% code title="jpss2_viirs_only_n_ca_multi_layer_pl.yaml" %}
```yaml

data:
 tile: False
 tile_size: [3,3,32]
 tile_step: [1,1,32]
 num_loader_workers: 10
 val_percent: 0.1
 scale_data: True
 pixel_padding: 1
 number_channels: 5
 fill_value: -999999.0
 valid_min: -10000.0
 valid_max: 6400000
 subset_count: 1
 output_subset_count: 1
 reader_type: "viirs_oc"
 reader_kwargs: 
  start_lon: -128.00
  end_lon: -116.00
  start_lat: 30.00
  end_lat: 38.94
  # mask_shp: "/data/nlahaye/remoteSensing/Lakes/glwd_1.shp"
  mask_oceans: True
  nrt: True
 chan_dim: 0
 delete_chans: []
 transform_default:
  chans: []
  transform: []

 files_train: [
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240305.L3m.DAY.",
# "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240306.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240307.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240308.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240309.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240310.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240311.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240312.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240313.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240314.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240315.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240316.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240317.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240318.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240319.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240320.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240321.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240322.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240323.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240324.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240325.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240326.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240327.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240328.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240329.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240330.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240331.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240401.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240402.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240403.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240404.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240405.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240406.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240407.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240408.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240409.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240410.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240411.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240412.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240413.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240414.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240415.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240416.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240417.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240418.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240419.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240420.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240421.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240422.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240423.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240424.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240425.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240426.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240427.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240428.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240429.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20240430.L3m.DAY.",
 ]

 files_test: [
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241024.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241025.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241026.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241027.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241028.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241029.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241030.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241031.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241101.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241102.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241103.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241104.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241105.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241106.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241107.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241108.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241109.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241110.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241111.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241112.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241113.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241114.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241115.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241116.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241117.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241118.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241119.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241120.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241121.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241122.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241123.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241124.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241125.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241126.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241127.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241128.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241129.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241130.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241201.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241202.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241203.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241204.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241205.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241206.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241207.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241208.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241209.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241210.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241211.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241212.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241213.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241214.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241215.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241216.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241217.L3m.DAY.",
 "/mnt/data/REFLECTANCES/JPSS2_VIIRS/JPSS2_VIIRS.20241218.L3m.DAY.",
 ]


scaler:
 name: "standard" 

output:
 out_dir: "/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/JPSS2_VIIRS_FINAL/"
 generate_intermediate_output: True


dbn:
 model_type: ["gaussian_selu", "gaussian_selu"]
 dbn_arch: [2000, 1000] #[250, 500, 2000] #, 2000]
 gibbs_steps: [1,1] #, 7, 7] #, 10] #, 25]
 temp: [1.0, 1.0] #, 1.0, 1.0] ##[0.9, 0.75, 0.5] #, 1.0] #, 0.5] #, 0.5] 
 normalize_learnergy: [False, True]
 batch_normalize: [False, False]

encoder_type: "dbn"
encoder:
 tiled: False
 tune_scaler: False
 subset_training: 2000000
 overwrite_model: False
 training:
  learning_rate: [0.00001, 0.00001]
  momentum: [0.95,0.95]
  weight_decay: [0.0001, 0.0001]
  nesterov_accel: [True, True]
  batch_size: 128
  epochs: 30
  accelerator: "gpu"
  devices: 1
  gradient_clip_val: 0.1
  precision: "16-mixed"
  save_dir: "wandb_encoder"
  stratify_data:
   kmeans: True


logger:
 use_wandb: True
 log_model: True
 log_out_dir: "wandb_dbn"
 project: "SIT-FUSE"

cluster:
 gauss_noise_stdev:  [0.01]
 lambda: 1.0
 num_classes: 800
 training:
  learning_rate: 0.0001
  batch_size: 1000
  epochs: 30
  accelerator: "gpu"
  devices: 1
  gradient_clip_val: 0.1
  precision: "16-mixed"
  save_dir: "wandb_full_model"

 heir:
  tiers: 1
  gauss_noise_stdev:  [0.01]
  lambda: 1.0
  num_classes: 100
  training:
   accelerator: "gpu"
   devices: 1
   gradient_clip_val: 0.1
   precision: "16-mixed"
   save_dir: "wandb_dbn_full_model_heir"
   batch_size: 100
   min_samples: 1000
   learning_rate: 0.0001
   batch_size: 1000
   epochs: 30
```
{% endcode %}
{% endtab %}

{% tab title="Using pre-trained weights" %}
<pre class="language-yaml" data-title="snpp_viirs_only_n_ca_multi_layer_pl.yaml"><code class="lang-yaml"><strong>
</strong><strong>data:
</strong> tile: False
 tile_size: [3,3,32]
 tile_step: [1,1,32]
 num_loader_workers: 10
 val_percent: 0.1
 scale_data: True
 pixel_padding: 1
 number_channels: 5
 fill_value: -999999.0
 valid_min: -10000.0
 valid_max: 6400000
 subset_count: 1
 output_subset_count: 1
 reader_type: "viirs_oc"
 reader_kwargs:
  #nrt: True
  start_lon: -128.00
  end_lon: -116.00
  start_lat: 30.00
  end_lat: 38.94
  #mask_shp: "/data/nlahaye/remoteSensing/Lakes/glwd_1.shp"
  mask_oceans: True
 chan_dim: 0
 delete_chans: []
 transform_default:
  chans: []
  transform: []

 files_train: [
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250401.L3m.DAY.",
 ]

 files_test: [
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250402.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250403.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250404.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250405.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250406.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250407.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250408.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250409.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250410.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250411.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250412.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250413.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250414.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250415.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250416.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250417.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250418.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250419.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250420.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250421.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250422.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250423.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250424.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250425.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250426.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250427.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250428.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250429.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250430.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250501.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250502.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250503.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250504.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250505.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250506.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250507.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250508.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250509.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250510.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250511.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250512.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250513.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250514.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250515.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250516.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250517.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250518.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250519.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250520.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250521.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250522.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250523.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250524.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250525.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250526.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250527.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250528.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250529.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250530.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250531.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250601.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250602.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250603.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250604.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250605.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250606.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250607.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250608.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250609.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250610.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250611.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250612.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250613.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250614.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250615.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250616.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250617.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250618.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250619.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250620.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250621.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250622.L3m.DAY.",
"/mnt/data/REFLECTANCES/SNPP_VIIRS/SNPP_VIIRS.20250623.L3m.DAY."
]


scaler:
 name: "standard" 

output:
 out_dir: "/mnt/data/HAB_MODEL_WEIGHTS/OC_SIF_HABs_2025/SNPP_VIIRS_S_CA/"
 generate_intermediate_output: True


dbn:
 model_type: ["gaussian_selu", "gaussian_selu"]
 dbn_arch: [2000, 1000] #[250, 500, 2000] #, 2000]
 gibbs_steps: [1,1] #, 7, 7] #, 10] #, 25]
 temp: [1.0, 1.0] #, 1.0, 1.0] ##[0.9, 0.75, 0.5] #, 1.0] #, 0.5] #, 0.5] 
 normalize_learnergy: [False, True]
 batch_normalize: [False, False]

encoder_type: "dbn"
encoder:
 tiled: False
 tune_scaler: False
 subset_training: 2000000
 overwrite_model: False
 training:
  learning_rate: [0.00001, 0.00001]
  momentum: [0.95,0.95]
  weight_decay: [0.0001, 0.0001]
  nesterov_accel: [True, True]
  batch_size: 128
  epochs: 30
  accelerator: "gpu"
  devices: 1
  gradient_clip_val: 0.1
  precision: "16-mixed"
  save_dir: "wandb_encoder"
  stratify_data:
   kmeans: True


logger:
 use_wandb: True
 log_model: True
 log_out_dir: "wandb_dbn"
 project: "SIT-FUSE"

cluster:
 gauss_noise_stdev:  [0.01]
 lambda: 1.0
 num_classes: 800
 training:
  learning_rate: 0.0001
  batch_size: 1000
  epochs: 30
  accelerator: "gpu"
  devices: 1
  gradient_clip_val: 0.1
  precision: "16-mixed"
  save_dir: "wandb_full_model"

 heir:
  tiers: 1
  gauss_noise_stdev:  [0.01]
  lambda: 1.0
  num_classes: 100
  training:
   accelerator: "gpu"
   devices: 1
   gradient_clip_val: 0.1
   precision: "16-mixed"
   save_dir: "wandb_dbn_full_model_heir"
   batch_size: 100
   min_samples: 1000
   learning_rate: 0.0001
   batch_size: 1000
   epochs: 30
</code></pre>


{% endtab %}
{% endtabs %}
