
import os
import numpy as np
import glob

import yaml
import copy

from sit_fuse.utils import read_yaml

from sit_fuse.postprocessing.generate_cluster_geotiffs import run_geotiff_gen
from sit_fuse.postprocessing.conv_and_cluster import conv_and_cluster
from sit_fuse.postprocessing.contour_and_fill import contour_and_fill
 
from sit_fuse.inference.generate_output import predict


import sys

def cluster_fname_builder(out_dir, gtiff_data, prob = True, no_heir = True, tiff=False):

    final_gtiff_data = []
    final_clust_data = []

    out_fname_base = out_dir
    for i in range(len(gtiff_data)):
        file_pattern = os.path.join(out_fname_base, os.path.basename(os.path.splitext(gtiff_data[i])[0]) + "*clust.data*zarr")
        if tiff:
            file_pattern += "*full_geo.tif"
        fglob = glob.glob(file_pattern)
        
        fglob = sorted(fglob)

        for j in range(len(fglob)):
  
            if not prob and "prob" in fglob[j]:
                continue

            if not no_heir and "no_heir" in fglob[j]:
                continue

            print(gtiff_data[i])
            print(fglob[j], prob, no_heir)

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


def build_config_fname_contour(config_dir, run_uid):
    config_fname = os.path.join(config_dir, "postprocess", "contour_and_fill" + run_uid + ".yaml") 
    return config_fname

def build_config_fname_conv_and_cluster(config_dir, run_uid):
    config_fname = os.path.join(config_dir, "postprocess", "conv_and_cluster_" + run_uid + ".yaml")
    return config_fname

def build_config_fname_inference(config_dir, run_uid):

        config_fname = os.path.join(config_dir, "model", "inference_" + run_uid + ".yaml") 
        return config_fname

def build_config_fname_gtiff_gen(config_dir, run_uid):

    config_fname = os.path.join(config_dir, "postprocess", "geotiff_gen_" + run_uid + ".yaml") 
    return config_fname

def update_config_inference(yml_conf, config_dict):
 
    if yml_conf["update_inputs"]:
        gtiff_data = input_fname_builder(yml_conf)
        config_dict["data"]["files_test"] = gtiff_data

    return config_dict

 
def update_config_gtiff_gen(yml_conf, training_conf, config_dict, context_assign = True):

    reuse_gtiffs = yml_conf["reuse_gtiffs"]
  
    config_dict["context"]["apply_context"] = context_assign

    if not reuse_gtiffs:
        gtiff_data = input_fname_builder(yml_conf)
        gtiff_data, cluster_fnames = cluster_fname_builder(training_conf["output"]["out_dir"], gtiff_data)
        config_dict["data"]["clust_reader_type"] = "zarr_to_numpy"
        config_dict["gen_from_geotiffs"] = False
        config_dict["data"]["gtiff_data"] = gtiff_data
        config_dict["data"]["cluster_fnames"] = cluster_fnames

    config_dict["context"]["name"] = yml_conf["context_classes"]


    return config_dict

 
def update_config_contour_and_fill(config_dict, context_config):


    tmp = copy.deepcopy(context_config["data"]["cluster_fnames"])

    file_pattern = os.path.join(os.path.dirname(tmp[0]), "*" + context_config["context"]["name"] + ".tif")
    fglob = glob.glob(file_pattern)
 
    final_list = []
    for i in range(len(fglob)):
        if "prob" not in fglob[i] and "no_heir" not in fglob[i] and "FullColor" not in fglob[i]:
            final_list.append(fglob[i])


    config_dict["data"]["filename"] = final_list

    return config_dict

def update_config_tiered_gtiff_gen(yml_conf, training_conf, config_dict):

    config_dict = update_config_gtiff_gen(yml_conf, training_conf, config_dict)

    config_dict["context"]["apply_context"] = True #No need for tiered masking if not applying context

    tiered_masks = []
    tiered_classes = []
    cluster_fnames = []
    #TODO, create a non-tiered fnames list and only create single geotiff from them
    for i in range(len(config_dict["data"]["cluster_fnames"])):
        if "prob" in config_dict["data"]["cluster_fnames"][i] or "no_heir" in config_dict["data"]["cluster_fnames"][i]:
            continue
        else:
            cluster_fnames.append(config_dict["data"]["cluster_fnames"][i])

    config_dict["data"]["cluster_fnames"] = cluster_fnames
    for i in range(len(config_dict["data"]["cluster_fnames"])):

        drname = os.path.dirname(config_dict["data"]["cluster_fnames"][i])
        fname_base = os.path.basename(os.path.splitext(config_dict["data"]["cluster_fnames"][i])[0])
        fname_base_final = os.path.join(drname, fname_base)

        tiered_masks_sub = [] 
        for j in range(len(yml_conf["tile_tiers"])):
            tiered_masks_sub.append(fname_base_final + ".zarr.full_geo.tif.tile_cluster." + str(yml_conf["tile_tiers"][j]) + ".tif")
        
        tiered_masks.append(tiered_masks_sub)

        print(len(config_dict["context"]["tiered_masking"]["tiered_classes"]), len(config_dict["data"]["cluster_fnames"]))
        tiered_classes.append(config_dict["context"]["tiered_masking"]["tiered_classes"][0]) #Should be uniform across samples for these cases

    config_dict["context"]["tiered_masking"]["tiered_classes"] = tiered_classes
    config_dict["context"]["tiered_masking"]["masks"] = tiered_masks

    return config_dict


def update_config_conv_and_cluster(yml_conf, out_dir):

    tiled_features_conf = {}
    tiled_features_conf["tile_size"] = yml_conf["tile_tiers"]

    gtiff_data = input_fname_builder(yml_conf)
    final_gtiff_data, clust_fnames = cluster_fname_builder(out_dir, gtiff_data, prob = False, no_heir = False, tiff=True)

    print(clust_fnames[0], "UPDATAE CONFIG")

    tiled_features_conf["train_fname"] = clust_fnames[0] #Arbitrary as assumption here is that training is completed already
    tiled_features_conf["test_fnames"] =  clust_fnames

    return tiled_features_conf


def run_prediction(yml_conf):

    predict(yml_conf)

def run_embed_gen(yml_conf, fname, gen_image_shaped = True):

    embed, labels = gen_embeddings(yml_conf, fname, gen_image_shaped) 
    return embed, labels

def run_inference_only(yml_conf, config_dict):

    #Assumes initial context asignment has been done - other pipelines automate that process

    print("Generating scnene-wide pixel-level predictions")
    config_dict = update_config_inference(yml_conf, config_dict)
    run_prediction(config_dict)

    #Dump to file
    config_fname = build_config_fname_inference(yml_conf["config_dir"], yml_conf["run_uid"])
    with open(config_fname, 'w') as fle:
        yaml.dump(config_dict, fle)


def run_contour_and_fill(yml_conf, context_config):

    contour_conf = read_yaml(yml_conf["contour_config"]) 
    config_dict = copy.deepcopy(contour_conf)
 
    config_dict = update_config_contour_and_fill(config_dict, context_config)
 
    config_fname = build_config_fname_contour(yml_conf["config_dir"],  yml_conf["run_uid"])
    with open(config_fname, 'w') as fle:
        yaml.dump(config_dict, fle)
 
    contour_and_fill(config_dict)


def run_basic_inference_geolocation(yml_conf):
   
    #Assumes initial context asignment has been done - other pipelines automate that process
    training_conf = read_yaml(yml_conf["training_config"])
    config_dict = copy.deepcopy(training_conf)
    run_inference_only(yml_conf, config_dict)

    context_conf = read_yaml(yml_conf["context_config"])
    config_dict = copy.deepcopy(context_conf)
    config_dict["output"] = {"out_dir" : training_conf["output"]["out_dir"]}

    tiled_features_conf = None

    #Generate config
    if "tiered_masking" in context_conf["context"]:
        tiled_features_conf = update_config_conv_and_cluster(yml_conf, training_conf["output"]["out_dir"])
        config_dict = update_config_tiered_gtiff_gen(yml_conf, training_conf, config_dict)
    else:
        config_dict = update_config_gtiff_gen(yml_conf, training_conf, config_dict)
 
    #Dump to file
    config_fname = build_config_fname_gtiff_gen(yml_conf["config_dir"],  yml_conf["run_uid"] + "_geolocated_clusters")
    with open(config_fname, 'w') as fle:
        yaml.dump(config_dict, fle)

 
    if "tiered_masking" in context_conf["context"]:

        print("Generating geolocated products")
        config_dict["context"]["apply_context"] = False
        run_geotiff_gen(config_dict) #Need Geotiffs to do conv_and_cluster
        config_dict["context"]["apply_context"] = True
        
 
        config_fname = build_config_fname_conv_and_cluster(yml_conf["config_dir"], yml_conf["run_uid"])
        with open(config_fname, 'w') as fle:
            yaml.dump(tiled_features_conf, fle)

        print("Generating feature pyramids from pixel-level predictions")
        conv_and_cluster(tiled_features_conf)


    print("Generating geolocated context assigned products")
    run_geotiff_gen(config_dict) 

    if "contour_config" in yml_conf:
        run_contour_and_fill(yml_conf, config_dict)         




