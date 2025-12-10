


#TODO - update files_train, files_test, reader_kwargs/BB, out_dir, dbn/arch, reader_type
YAML_TEMPLATE_HAB_TRAIN = {

"data" : {
 "tile" : False,
 "tile_size" : [3,3,32],
 "tile_step" : [1,1,32],
 "num_loader_workers" : 10,
 "val_percent" : 0.1,
 "scale_data" : True,
 "pixel_padding" : 1,
 "number_channels" : 5, #ignore for now - dynamically read
 "fill_value" : -999.0,
 "valid_min" : -30000.0,
 "valid_max" : 64000000,
 "subset_count" : 1,
 "output_subset_count" : 1,
 "reader_type" : "oc_and_trop",
 "reader_kwargs" : {
  "start_lon" : -128.00,
  "end_lon" : -116.00,
  "start_lat" : 30.00,
  "end_lat" : 37.00,
  "mask_oceans" : True,
  "nrt" : True
 },
 "chan_dim" : 0,
 "delete_chans" : [], #[0,1,2,3,4], USED FOR TROP ONLY
 "transform_default" : {
  "chans" : [],
  "transform" : [],
 },

 "files_train" : [],
 "files_test" : [],
},

"scaler" : {
 "name" : "standard"
},
 
"output" : {
 "out_dir" : "/data/nlahaye/output/Learnergy/TROP_ONLY_S_CA/",
 "generate_intermediate_output" : True,
 "generate_train_output" : True 
},

"dbn" : {
 "model_type" : ["gaussian_selu", "gaussian_selu"],
 "dbn_arch" : [2000, 1000],
 "gibbs_steps" : [1,1],
 "temp" : [1.0, 1.0],
 "normalize_learnergy" : [False, True],
 "batch_normalize" : [False, False]
},

"encoder_type": "dbn",

"encoder": {
 "tiled": False,
 "tune_scaler" : False,
 "subset_training" : 2000000,
 "overwrite_model" : False,
 "training" : {
  "learning_rate" : [0.00001, 0.00001],
  "momentum" : [0.95,0.95],
  "weight_decay" : [0.0001, 0.0001],
  "nesterov_accel" : [True, True],
  "batch_size" : 128,
  "epochs" : 30,
  "accelerator" : "gpu",
  "devices" : 1,
  "gradient_clip_val" : 0.1,
  "precision" : "16-mixed",
  "save_dir" : "wandb_encoder",
  "stratify_data" : {
   "kmeans" : True
  }
  },
},

"logger" : {
 "use_wandb" : True,
 "log_model" : True,
 "log_out_dir" : "wandb_dbn",
 "project" : "SIT-FUSE",
},

"cluster" : {
 "gauss_noise_stdev" :  [0.01],
 "lambda" : 1.0,
 "num_classes" : 800,
 "training" : {
  "learning_rate" : 0.0001,
  "batch_size" : 1000,
  "epochs" : 30,
  "accelerator" : "gpu",
  "devices" : 1,
  "gradient_clip_val" : 0.1,
  "precision" : "16-mixed",
  "save_dir" : "wandb_full_model"
 },

 "heir" : {
  "tiers" : 1,
  "gauss_noise_stdev" :  [0.01],
  "lambda" : 1.0,
  "num_classes" : 100,
  "training" : {
   "accelerator" : "gpu",
   "devices" : 1,
   "gradient_clip_val" : 0.1,
   "precision" : "16-mixed",
   "save_dir" : "wandb_dbn_full_model_heir",
   "batch_size" : 100,
   "min_samples" : 1000,
   "learning_rate" : 0.0001,
   "batch_size" : 1000,
   "epochs" : 30,
  }
 }
}

}


REGION_BBS =  {
"s_ca" : {
 "start_lon" : -128.00,
 "end_lon" : -116.00,
 "start_lat" : 30.00,
 "end_lat" : 37.00,
},

"ca" : {
 "start_lon" : -128.00,
 "end_lon" : -116.00,
 "start_lat" : 30.00,
 "end_lat" : 38.94,
},


"w_us_coast": {
 "start_lon" : -135.00,
 "end_lon" : -116.00,
 "start_lat" : 30.00,
 "end_lat" : 49.0,
},

"gulf_of_mexico" : {
 "start_lon" : -97.8985,
 "end_lon" : -74.5301,
 "start_lat" : 18.1599,
 "end_lat" : 30.4159,
},


"red_sea" : {
  "start_lon": 32.0,
  "end_lon": 44.0,
  "start_lat": 12.0,
  "end_lat": 29.0,
},

}


READER_TYPE_MAP = {

    "TROP" : "oc_and_trop",
    "S3A_OLCI" : "s3_oc",
    "S3B_OLCI" : "s3_oc",
    "JPSS1_VIIRS" : "viirs_oc",
    "JPSS2_VIIRS" : "viirs_oc",
    "SNPP_VIIRS" : "viirs_oc",
    "AQUA_MODIS" : "modis_oc",
    "PACE_OCI" : "pace_oc",
    "GOES18_ABI": "gtiff" 
}

NUM_CHANNELS = {

    "TROP" : 1,
    "S3A_OLCI" : 11,
    "S3B_OLCI" : 11,
    "JPSS1_VIIRS" : 5,
    "JPSS2_VIIRS" : 5,
    "SNPP_VIIRS" :  5,
    "AQUA_MODIS" : 10,
    "PACE_OCI" : 1, #all channels in one file
    "GOES18_ABI": 1, #all channels in one file after tiff gen

}


DBN_ARCH_MAP = {
 
    "PACE_OCI" : [3000, 1500],
    "DEFAULT" : [2000, 1000]
}



