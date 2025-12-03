import numpy as np
import copy
import os
import re
import yaml
import glob

from sit_fuse.datasets.sf_dataset import run_data_prep
from sit_fuse.train.pretrain_encoder import run_pretraining
from sit_fuse.train.finetune_dc_mlp_head import run_tuning
from sit_fuse.train.train_heirarchichal_deep_cluster import run_heir_training

from sit_fuse.pipelines.training.oil_palm_preprocess_and_train_constants import *


def build_config_fname_cf_gtiff_gen(config_dir, run_uid):
 
    #currently not very dynamic. Modularized like other pipelines for future changes
    config_fname = os.path.join(config_dir, "model", "train_model_" + run_uid + ".yaml")

    return config_fname



def find_train_test_fpaths(yml_conf, config_dict):

    train_grid_inds = yml_conf["train_grid_inds"]

    gridded_re = yml_conf["gridded_file_re"]
 
    gridded_fnames = glob.glob(os.path.join(yml_conf["input_dir"], yml_conf["input_glob"]))

    train_fnames = []
    test_fnames = []

    for fname in gridded_fnames:
        print(fname, gridded_re)
        mtch = re.search(gridded_re, fname)
        if mtch:
            grid_idx = str(mtch.group(1))
            print(mtch.group(0), grid_idx)
            if grid_idx in train_grid_inds:
                train_fnames.append(fname)
            else:
                test_fnames.append(fname)

    config_dict["data"]["files_train"] = train_fnames
    config_dict["data"]["files_test"] = test_fnames

    return config_dict


def update_training_config(yml_conf, config_dir, yaml_train_template):
 
    config_dict = copy.deepcopy(yaml_train_template)
    config_dict = find_train_test_fpaths(yml_conf, config_dict)
    config_dict["output"]["out_dir"] = yml_conf["out_dir"]

    config_fname = build_config_fname_cf_gtiff_gen(config_dir, yml_conf["run_uid"])

    with open(config_fname, 'w') as fle:
        yaml.dump(config_dict, fle)

    return config_dict

def run_data_preprocessing(yml_conf):
 
    yaml_train_template = YAML_TEMPLATE_OIL_PALM_TRAIN
    config_dict = update_training_config(yml_conf, yml_conf["config_dir"], yaml_train_template)
 
    #pre-generate and store training subset and scaler
    run_data_prep(config_dict)
    return config_dict

def run_model_training(config_dict):

    run_pretraining(config_dict)
    run_tuning(config_dict)
    run_heir_training(config_dict)



