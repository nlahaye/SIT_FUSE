import numpy as np
import dask
import dask.array as da
from  datetime import datetime
import copy
import os 
import re
import yaml

from sit_fuse.utils import read_oc_geo

from sit_fuse.pipelines.habs.hab_training_constants import *
from sit_fuse.pipelines.data_transform_and_prep.goes_combine_and_clip_pipeline import goes_combine_and_clip_pipeline

from sit_fuse.datasets.sf_dataset import run_data_prep
from sit_fuse.train.pretrain_encoder import run_pretraining
from sit_fuse.train.finetune_dc_mlp_head import run_tuning
from sit_fuse.train.train_heirarchichal_deep_cluster import run_heir_training
 
def build_config_fname_data_prep_and_train(config_dir, instrument, with_trop = False):

    config_fname = os.path.join(config_dir, "model", "train_model_" + instrument)

    if with_trop:
        config_fname = config_fname + "_trop"

    config_fname = config_fname + ".yaml"
    return config_fname

def gen_geo_zarr_oc_daily(yml_conf):
 
    start_lon = yml_conf["start_lon"]
    end_lon = yml_conf["end_lon"]
    start_lat = yml_conf["start_lat"]
    end_lat = yml_conf["end_lat"]
    
    fglob = glob.glob(yml_conf["geo_zarr"]["oc_geo_fpattern"])
    loc = read_oc_geo(fglob[0])
 
    lat = np.array(loc[0])
    lon = np.array(loc[1])
    inds1 = np.where((lat >= start_lat) & (lat <= end_lat))
    inds2 = np.where((lon >= start_lon) & (lon <= end_lon))

    nind1, nind2 = np.meshgrid(inds2[0], inds1[0])
    lat = lat[inds1]
    lon = lon[inds2]
    loc = np.array(np.meshgrid(lat, lon))

    loc = np.moveaxis(loc, 0,2)
    loc = np.swapaxes(loc,0,1)
 
    data2 = da.from_array(loc)
    da.to_zarr(data2, os.path.join(yml_conf["geo_zarr"]["out_dir"], yml_conf["geo_zarr"]["out_uid"] + ".zarr"))


def build_trop_fname(yml_conf, datetm):

    dirpth = yml_conf["sif_dir"]
    fname = os.path.join(dirpth, "TROPO_redSIF_" + datetm.strftime("%Y-%m-%d") + "_ungridded.nc") 
    return fname



def find_train_test_fpaths(yml_conf, config_dict, instrument, key):
 
    train_min = datetime.strptime(yml_conf["train_min_date"], "%Y%m%d")
    train_max = datetime.strptime(yml_conf["train_max_date"], "%Y%m%d")

    print(train_min, train_max)

    train_fnames = []
    test_fnames = []

    for root, dirs, files in os.walk(yml_conf["instruments"][instrument][key]["input_oc_dir"]):
         
        for fle in files:

            mtch = re.search(str(yml_conf["instruments"][instrument][key]["input_oc_re"]), fle)
            if mtch:
                if "GOES" in instrument:
                    dt = datetime.strptime(mtch.group(2), "%Y%j%H%M%S")
                else:
                    dt = datetime.strptime(mtch.group(2), "%Y%m%d")
                lst_entry = os.path.join(root, mtch.group(1))
                #trop/no_trop
                if "troposif" == instrument or "no" not in key:
                    trop_fname = build_trop_fname(yml_conf, dt)
                     
                    lst_entry = [lst_entry, trop_fname]

                print(train_max > train_min, train_max, train_min, dt, dt >= train_min, dt <= train_max)    
                if dt >= train_min and dt <= train_max:
                    train_fnames.append(lst_entry)
                    print(dt, mtch.group(2))  
                else:
                    test_fnames.append(lst_entry)

        test_fnames = sorted(test_fnames)
        train_fnames = sorted(train_fnames)
        match_count = 0
        final_test_fnames = []
        for i in range(1, len(test_fnames)):
            if test_fnames[i] == test_fnames[i-1]:
                match_count += 1
            else:
                if match_count == NUM_CHANNELS[instrument] -1:
                    final_test_fnames.append(test_fnames[i-1])
                match_count = 0
  
        final_train_fnames = []
        match_count = 0
        for i in range(1, len(train_fnames)):
            if train_fnames[i] == train_fnames[i-1]:
                match_count += 1
            else:
                if match_count == NUM_CHANNELS[instrument] -1:
                    final_train_fnames.append(train_fnames[i-1])
                match_count = 0


    config_dict["data"]["files_train"] = list(set(final_train_fnames))
    config_dict["data"]["files_test"] = list(set(final_test_fnames))

    return config_dict


def update_training_config(yml_conf, config_dict, instrument, key):

    config_dict = find_train_test_fpaths(yml_conf, config_dict, instrument, key)

    region_bb = REGION_BBS[yml_conf["roi"]]
    for bb_key in region_bb:
        config_dict["data"]["reader_kwargs"][bb_key] = region_bb[bb_key]
    


    if "no" not in key:
            config_dict["data"]["reader_type"] = READER_TYPE_MAP["TROP"] 
    else:
        for rt_key in READER_TYPE_MAP:
            if rt_key == "TROP":
                continue

            if instrument in rt_key:
                config_dict["data"]["reader_type"] = READER_TYPE_MAP[rt_key] 


    print(config_dict["data"]["reader_type"], config_dict["data"]["files_train"])

    dbn_arch = DBN_ARCH_MAP["DEFAULT"] 
    for arch_key in DBN_ARCH_MAP:
        if instrument in arch_key:
            dbn_arch = DBN_ARCH_MAP[arch_key]

    config_dict["dbn"]["dbn_arch"] = dbn_arch

    config_dict["output"]["out_dir"] = os.path.join(yml_conf["output"], instrument + "_" + key)

    os.makedirs(config_dict["output"]["out_dir"], exist_ok=True)

    return config_dict


def run_data_preprocessing(yml_conf):

    instrument_dict = yml_conf["instruments"]
    config_dir = yml_conf["config_dir"]

    configs = {}
    
    for instrument in instrument_dict:
        if instrument not in configs:
            configs[instrument] = {}
        for key in instrument_dict[instrument]:
            if key not in configs[instrument]:
                configs[instrument][key] = {}

            trop = True
            if "no" in key: #Key is trop no_trop
                trop = False
              
            config_dict = copy.deepcopy(YAML_TEMPLATE_HAB_TRAIN)
 
            config_dict = update_training_config(yml_conf, config_dict, instrument, key)

            if "GOES" in instrument and instrument_dict[instrument][key]["run_preprocess"]:
                yml_conf["input_dir"] = instrument_dict[instrument][key]["input_oc_dir"]
                goes_combine_and_clip_pipeline(yml_conf)                

            if instrument_dict[instrument][key]["run_preprocess_and_scale"]:
                run_data_prep(config_dict)             

            configs[instrument][key] = config_dict

            config_fname = build_config_fname_data_prep_and_train(config_dir, instrument, with_trop = trop)

            with open(config_fname, 'w') as fle:
               yaml.dump(config_dict, fle)

    return configs



def run_model_training(yml_conf, configs):

    instrument_dict = yml_conf["instruments"]
    config_dir = yml_conf["config_dir"]

    for instrument in instrument_dict:
        for key in instrument_dict[instrument]:
            config_dict = configs[instrument][key]

            run_pretraining(config_dict) 
            run_tuning(config_dict)
            run_heir_training(config_dict)




