
YAML_TEMPLATE_PV_TRAIN = {

"data" : {
 "tile" : False,
 "num_loader_workers" : 10,
 "val_percent" : 0.1,
 "scale_data" : True,
 "pixel_padding" : 2,
 "number_channels" : 6, #ignore for now - dynamically read
 "fill_value" : 0.0,
 "valid_min" : 0.0000001,
 "valid_max" : 100000000,
 "subset_count" : 1,
 "output_subset_count" : 1,
 "reader_type" : "gtiff",
 "reader_kwargs" : {
  "none" : "none",
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
 "out_dir" : "/data/nlahaye/output/Learnergy/DBN_PV_MAPS/",
 "generate_intermediate_output" : True,
 "generate_train_output" : True 
},

"dbn" : {
 "model_type" : ["gaussian_selu", "gaussian_selu", "gaussian_selu"],
 "dbn_arch" : [3000, 2500, 2000],
 "gibbs_steps" : [1,1, 1],
 "temp" : [1.0, 1.0, 1.0],
 "normalize_learnergy" : [False, True, True],
 "batch_normalize" : [False, False, False]
},

"encoder_type": "dbn",

"encoder": {
 "tiled": False,
 "tune_scaler" : False,
 "subset_training" : 2000000,
 "overwrite_model" : False,
 "training" : {
  "learning_rate" : [0.00001, 0.00001, 0.00001],
  "momentum" : [0.95,0.95, 0.95],
  "weight_decay" : [0.0001, 0.0001,  0.0001],
  "nesterov_accel" : [True, True, True],
  "batch_size" : 128,
  "epochs" : 100,
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
  "epochs" : 100,
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
   "epochs" : 100,

  }
 }
}

}



