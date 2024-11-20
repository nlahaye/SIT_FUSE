# Example I-JEPA Config

{% code fullWidth="true" %}
```yaml
data:
 tile: True #True
 tile_size: [3,3,34]
 tile_step: [1,1,34]
 val_percent: 0.0
 num_loader_workers: 10
 scale_data: True
 pixel_padding: 1
 number_channels: 3 #34
 fill_value: -9999.0
 valid_min:  -900.0
 valid_max: 100000000
 subset_count: 1
 output_subset_count: 1
 reader_type: "numpy"
 reader_kwargs:
  none: "none"
  #start_line: 1000
  #end_line: 1500
  #start_sample: 300
  #end_sample: 789
 chan_dim: 0
 delete_chans: [25,26,36,37]
 transform_default:
  chans: []
  transform: []


 files_train: [
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19910_10_20190806_1858_1910_V02_Georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19911_10_20190807_2004_2016_V02_georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19911_09_20190807_1947_2002_V02_georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19912_07_20190808_1806_1821_V02_georeferenced_scaled.tif.npy',
]

 files_test: [
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19911_05_20190807_1817_1833_V02_georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19910_06_20190806_1815_1824_V02_Georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19912_07_20190808_1806_1821_V02_georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19910_21_20190806_2111_2125_V02_Georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19910_19_20190806_2035_2048_V02_Georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19911_08_20190807_1928_1942_V02_georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19911_12_20190807_2039_2051_V02_georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19911_11_20190807_2021_2035_V02_georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19910_17_20190806_2000_2010_V02_Georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19910_16_20190806_1941_1956_V02_Georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19910_13_20190806_1923_1925_V02_Georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19910_20_20190806_2052_2106_V02_Georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19910_06_20190806_1815_1824_V02_Georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19910_08_20190806_1834_1846_V02_Georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19910_18_20190806_2018_2031_V02_Georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19910_15_20190806_1931_1938_V02_Georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19910_14_20190806_1926_1929_V02_Georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19910_10_20190806_1858_1910_V02_Georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19911_10_20190807_2004_2016_V02_georeferenced_scaled.tif.npy',
'/data/nlahaye/remoteSensing/eMAS/rad/eMASL1B_19911_09_20190807_1947_2002_V02_georeferenced_scaled.tif.npy']

scaler:
 name: "standard" 

output:
 out_dir: "/data/nlahaye/output/Learnergy/IJEPA_TEST_SMALL/"
 generate_train_output: True

logger:
 use_wandb: True
 log_model: True
 log_out_dir: "wandb_small_full"
 project: "SIT-FUSE"

ijepa:
 patch_size: 1
 embed_dim: 128
 encoder_heads: 8
 encoder_depth: 8
 decoder_depth: 6
 target_aspect_ratio: [1,1]
 target_scale: [1,1]
 context_aspect_ratio: 1
 context_scale: [1,1]
 number_target_blocks: 4 

encoder_type: "ijepa"
encoder:
 subset_training: -1
 overwrite_model: False
 tune_scaler: False
 training:
  weight_decay: 0.05
  momentum: 0.996
  momentum_start_end: [0.996, 1.0] 
  learning_rate: 0.001
  batch_size: 16
  epochs: 100
  accelerator: "gpu"
  devices: 6 
  gradient_clip_val: 0.1
  precision: "16-mixed"
  save_dir: "wandb_ijepa_small_full"
  stratify_data:
   kmeans: False  

cluster:
 gauss_noise_stdev:  [0.01]
 lambda: 1.0
 num_classes: 800
 training:
  learning_rate: 0.0001
  batch_size: 1000
  epochs: 30 
  accelerator: "gpu"
  devices: 3
  gradient_clip_val: 0.1
  precision: "16-mixed"
  save_dir: "wandb_ijepa_small_full"

 heir:
  tiers: 1
  gauss_noise_stdev:  [0.01]
  lambda: 1.0
  num_classes: 100
  training:
   accelerator: "gpu"
   devices: 2
   gradient_clip_val: 0.1
   precision: "32"
   save_dir: "wandb_ijepa_small_full"
   batch_size: 100
   min_samples: 1000
   epochs: 30
   lambda: 1.0
```
{% endcode %}
