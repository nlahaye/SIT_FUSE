
import os
import numpy as np
import glob

import yaml

from sit_fuse.utils import read_yaml

from sit_fuse.postprocessing.generate_cluster_geotiffs import run_geotiff_gen
from sit_fuse.postprocessing.conv_and_cluster import conv_and_cluster


from sit_fuse.inference.generate_output import predict


def cluster_fname_builder(out_dir, gtiff_data, prob = True, no_heir = True):

    final_gtiff_data = []
    final_clust_data = []

    out_fname_base = out_dir
    for i in range(len(gtiff_data)):
        file_pattern = os.path.join(out_fname_base, os.path.basename(os.path.splitext(gtiff_data)[0]) + ".*clust.data.*zarr")
        fglob = glob.glob(file_pattern)

        fglob = sorted(fglob)

        for j in range(len(fglob)):
  
            if not prob and ".prob" in fglob[j]:
                continue

            if not no_heir and "no_heir" in fglob[j]:
                continue

            final_gtiff_data.append(gtiff_data[i])
            final_clust_data.append(fglob[j])

    return final_gtiff_data, final_clust_data


def input_fname_builder(yml_conf):

    #TODO - make this more multi-sensor / HAB friendly

    input_data = []

    input_dir = yml_conf["input_dir"]
    input_fle_pattern = os.path.join(input_dir, yml_conf["input_pattern"])
    fglob = glob.glob(input_fle_pattern)   
 
    fglob = sorted(fglob)

    return fglob

def build_config_fname_inference(config_dir, run_uid):

        config_fname = os.path.join(config_dir, "model", "inference_" + run_uid + ".yaml") 
        return config_fname

def build_config_fname_gtiff_gen(config_dir, run_uid):

    config_fname = os.path.join(config_dir, "postprocessing", "geotiff_gen_" + run_uid + ".yaml") 
    return config_fname

def update_config_inference(yml_conf, config_dict):
 
    if yml_conf["update_inputs"]:
        gtiff_data = input_fname_builder(yml_conf)
        config_dict["data"]["files_test"] = gtiff_data

    return config_dict

 
def update_config_gtiff_gen(yml_conf, context_conf, config_dict):

    reuse_gtiffs = yml_conf["reuse_gtiffs"]
  
    if not reuse_gtiffs:
        gtiff_data = input_fname_builder(yml_conf)
        gtiff_data, cluster_fnames = cluster_fname_builder(context_conf["output"]["out_dir"], gtiff_data)
     
        config_dict["data"]["gtiff_data"] = gtiff_data
        config_dict["data"]["cluster_fnames"] = cluster_fnames

    config_dict["context"]["name"] = yml_conf["run_uid"]

    return config_dict


def update_config_tiered_gtiff_gen(yml_conf, context_conf, config_dict):

    config_dict = update_config_gtiff_gen(yml_conf, context_conf, config_dict)

    tiered_masks = []
    tiered_classes = []
    for i in range(len(config_dict["data"]["cluster_fnames"])):

        drname = os.path.dirname(config_dict["data"]["cluster_fnames"][i])
        fname_base = os.path.basename(os.path.splitext(config_dict["data"]["cluster_fnames"][i])[0])
        fname_base_final = os.path.join(drname, fname_base)

        tiered_masks_sub = [] 
        for j in range(len(yml_conf["tile_tiers"])):
            tiered_masks_sub.append(fname_base_final + ".tile_cluster." + str(yml_conf["tile_tiers"][j]) + ".tif")
        
        tiered_masks.append(tiered_masks_sub)

        tiered_classes.append(context_conf["tiered_masking:"]["tiered_classes"])

    config_dict["tiered_masking:"]["tiered_classes"] = tiered_classes
    tiered_masks["tiered_masking:"]["masks"] = tiered_masks

    return config_dict


def update_config_conv_and_cluster(yml_conf):

    tiled_features_conf = {}
    tiled_features_conf["tile_size"] = yml_conf["tile_tiers"]

    clust_fnames = cluster_fname_builder(out_dir, gtiff_data, prob = False, no_heir = False)
 
    tiled_features_conf["train_fname"] = clust_fnames[0] #Arbitrary as assumption here is that training is completed already
    tiled_features_conf["test_fnames"] =  clust_fnames

    return tiled_features_conf


def run_prediction(yml_conf):

    predict(yml_conf)
     

def run_inferece_only(yml_conf, config_dict):

    #Assumes initial context asignment has been done - other pipelines automate that process

    print("Generating scnene-wide pixel-level predictions")
    #training_conf = read_yaml(yml_conf["training_config"])
    #config_dict = copy.deepcopy(training_conf)
    config_dict = update_config_inference(yml_conf, config_dict)
    run_prediction(config_dict)

    #Dump to file
    config_fname = build_config_fname_gtiff_gen(config_dir, "model", yml_conf["run_uid"] + "_inference.yaml")
    with open(config_fname, 'w') as fle:
        yaml.dump(config_dict, fle)



def run_basic_inference_geolocation(yml_conf, config_dict):
   
    #Assumes initial context asignment has been done - other pipelines automate that process
    run_inferece_only(yml_conf, config_dict)

    context_conf = read_yaml(yml_conf["context_config"])
    config_dict = copy.deepcopy(context_conf)

    #Generate config
    if "tiered_masking" in context_conf["context"]:
        tiled_features_conf = update_config_conv_and_cluster(yml_conf)
        config_dict = update_config_tiered_gtiff_gen(yml_conf, context_conf, config_dict)
    else:
        config_dict = update_config_gtiff_gen(yml_conf, context_conf, config_dict)
 
    #Dump to file
    config_fname = build_config_fname_gtiff_gen(config_dir, "postprocessing", yml_conf["run_uid"] + "_geolocated_clusters.yaml")
    with open(config_fname, 'w') as fle:
        yaml.dump(config_dict, fle)
 
    if "tiered_masking" in context_conf["context"]:

        config_fname = build_config_fname_gtiff_gen(config_dir, "postprocessing", yml_conf["run_uid"] + "_conv_and_cluster.yaml")
        with open(config_fname, 'w') as fle:
            yaml.dump(config_dict, fle)

        print("Generating feature pyramids from pixel-level predictions")
        conv_and_cluster(tiled_features_conf)

    print("Generating geolocated products")
    run_geotiff_gen(config_dict)

 




