
import os
import numpy as np
import glob
import re
import sys
import yaml
import copy

from sit_fuse.utils import read_yaml

from sit_fuse.postprocessing.generate_cluster_geotiffs import run_geotiff_gen
from sit_fuse.postprocessing.conv_and_cluster import conv_and_cluster
from sit_fuse.postprocessing.contour_and_fill import contour_and_fill
from sit_fuse.postprocessing.zonal_histogram import run_zonal_hist
from sit_fuse.postprocessing.class_compare import run_class_compare

from sit_fuse.pipelines.inference.inference_utils import run_inference_only, update_config_gtiff_gen, build_config_fname_gtiff_gen, \
update_config_conv_and_cluster, build_config_fname_conv_and_cluster, update_config_tiered_gtiff_gen
from sit_fuse.pipelines.context_assign.context_assign_constants import *




#TODO config gen, and testing
#Test current structure and single model
def run_zonal_histogram(config_dict):

    return run_zonal_hist(config_dict)


def update_config_class_compare(config_dict, zonal_hist_fpath):

    config_dict["dbf_list"] = [[zonal_hist_fpath]]
    return config_dict


def build_config_fname_zonal_hist(config_dir, run_uid):

    config_fname = os.path.join(config_dir, "postprocess", "zonal_histogram_" + run_uid)

    config_fname = config_fname + ".yaml"
    return config_fname

 
def build_config_fname_class_comp(config_dir, run_uid):

    config_fname = os.path.join(config_dir, "postprocess", "class_compare_" + run_uid + ".yaml")
    return config_fname


 
def class_compare(yml_conf, zonal_hist_fpath = None, tiered = False):
 
    #Generate config
    config_dict = copy.deepcopy(YAML_TEMPLATE_PIXEL_LAYER_CLASS_COMPARE)
    config_dict = update_config_class_compare(config_dict, zonal_hist_fpath) 

    if tiered:
        config_dict["dbf_percentage_thresh"] = 1.0 #Allow for any tiled features that include object type of interest
    else:
        config_dict["dbf_percentage_thresh"] = 0.9 #Slightly less constrained - remove majoritive negative clusters at the pixel level


    #Dump to file
    config_fname = build_config_fname_class_comp(yml_conf["config_dir"], yml_conf["run_uid"])
    with open(config_fname, 'w') as fle:
        yaml.dump(config_dict, fle)

    assignment, uncertain = run_class_compare(config_dict)
    classes = None
    if len(assignment) == 3: #Binary problem
        classes = assignment[1]
        classes.extend(assignment[2])
        #Take all certain and uncertain assignments for class of interest
        #Tiled filtering via conv and cluster will help filter out further
    else:
        classes = assignment #TODO - employ further logic here - need a test case

    return classes

def update_config_tiered_zonal_hist(yml_conf, training_conf, config_dict, tier):

    config_dict_ret = copy.deepcopy(config_dict)
    for i in range(len(config_dict_ret["output"]["class_name"])):
        config_dict_ret["output"][i] = config_dict_ret["output"]["class_name"][i] + "_" + str(tier)
    for i in range(len(config_dict_ret["data"]["clust_gtiffs"])):
        config_dict_ret["data"]["clust_gtiffs"][i] = config_dict_ret["data"]["clust_gtiffs"][i] + ".tile_cluster." + str(tier) + ".tif"
   

    return config_dict_ret


def run_context_assign_experiment(yml_conf):
    #Assumes initial context asignment has been done - other pipelines automate that process
    training_conf = read_yaml(yml_conf["training_config"])
    config_dict = copy.deepcopy(training_conf)
    if yml_conf["run_inference"]:
        #training_conf = read_yaml(yml_conf["training_config"])
        #config_dict = copy.deepcopy(training_conf)
        run_inference_only(yml_conf, config_dict)

    context_conf = read_yaml(yml_conf["context_config"])
    config_dict = copy.deepcopy(context_conf)

    #No tiered masking info yet
    config_dict = update_config_gtiff_gen(yml_conf, training_conf, config_dict, context_assign = False)

    #Dump to file
    config_fname = build_config_fname_gtiff_gen(yml_conf["config_dir"],  yml_conf["run_uid"] + "_geolocated_clusters")
    with open(config_fname, 'w') as fle:
        yaml.dump(config_dict, fle)

    #print("Generating geolocated products")
    #run_geotiff_gen(config_dict)

    #For now, this must be manually generated
    zonal_hist_conf = read_yaml(yml_conf["zonal_hist_config"])
 
    print("Generating geolocated products")
    run_geotiff_gen(config_dict)
 
    print("Generating zonal hist") 
    zonal_hist_fname, _ = run_zonal_histogram(zonal_hist_conf)

    print("Running pixel-level class assignment")
    classes = class_compare(yml_conf, zonal_hist_fname, tiered = False)

    #Add generated class list to config 
    config_dict["context"]["clusters"] = classes

    tiled_features_conf = update_config_conv_and_cluster(yml_conf, training_conf["output"]["out_dir"])

    config_fname = build_config_fname_conv_and_cluster(yml_conf["config_dir"], yml_conf["run_uid"])
    with open(config_fname, 'w') as fle:
        yaml.dump(tiled_features_conf, fle)

    conv_and_cluster(tiled_features_conf)

    tiered_classes = []
    for j in range(len(yml_conf["tile_tiers"])):
        tiered_zonal_hist_conf = update_config_tiered_zonal_hist(yml_conf, training_conf, zonal_hist_conf, yml_conf["tile_tiers"][j])
        config_fname = build_config_fname_zonal_hist(yml_conf["config_dir"], yml_conf["run_uid"] + "_tiered_masking_" + str(yml_conf["tile_tiers"][j]))
        with open(config_fname, 'w') as fle:
            yaml.dump(tiered_zonal_hist_conf, fle)

        print("Generating zonal hist. Tile size:", str(yml_conf["tile_tiers"][j]))
        zonal_hist_fname, _ = run_zonal_histogram(tiered_zonal_hist_conf) 

        print("Running tile-level class assignment. Tile size:", str(yml_conf["tile_tiers"][j]))
        classes = class_compare(yml_conf, zonal_hist_fname, tiered = True)

        classes.append(-1.0)
        tiered_classes.append(classes) 
       

    tiered_classes_full = []
    for i in range(len(config_dict["data"]["cluster_fnames"])):
        tiered_classes_full.append(tiered_classes)
    print(tiered_classes)
    print("Generating config for tiered masking")
    tiered_masking = copy.deepcopy(YAML_TEMPLATE_TIERED_MASKING) 

    tiered_masking["tiered_classes"] = tiered_classes_full
    config_dict["context"]["tiered_masking"] = tiered_masking
    config_dict = update_config_tiered_gtiff_gen(yml_conf, training_conf, config_dict)
   #Dump to file
    config_fname = build_config_fname_gtiff_gen(yml_conf["config_dir"],  yml_conf["run_uid"] + "_geolocated_clusters")
    with open(config_fname, 'w') as fle:
        yaml.dump(config_dict, fle)



    print("Generating geolocated products that incorporate tiered masking")
    run_geotiff_gen(config_dict)



"""
1) Run inference, if needed *
2) Generate full_geo geotiffs *
3) take specified labels and files and generate zonal_hist*
4) run class compare over ^ w/ high certainty threshold (0.9)** - Need to fix config setup on this step TODO
5) take positive assignments and uncertain assignments from ^: set as per-pixel classes *
6) Run conv and cluster, 
7) Use zonal hist -> class compare to generate tiered class masks
	6a) Test w/ 1 model and multiple models
7) Setup config w/ classes from (5) and (7): Run final map gen
8) Eval whether or not contouring is needed
"""







