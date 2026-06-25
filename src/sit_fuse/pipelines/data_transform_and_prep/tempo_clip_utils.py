import os
import copy
import re
import numpy as np
import yaml

from sit_fuse.utils import read_yaml
from sit_fuse.preprocessing.colocate_and_resample import resample_or_fuse_data

from sit_fuse.pipelines.data_transform_and_prep.tempo_clip_constants import *


def build_config_fname_cf_gtiff_gen(config_dir):

    #currently not very dynamic. Modularized like other pipelines for future changes
    config_fname = os.path.join(config_dir, "preprocess", "colocate_and_resample", "tempo_clip.yaml")

    return config_fname


def update_config_tempo_netcdf_to_gtiff(yml_conf, config):

    fdir = yml_conf["input_dir"]

    start_lon = yml_conf["start_lon"]
    end_lon = yml_conf["end_lon"]
    start_lat = yml_conf["start_lat"]
    end_lat = yml_conf["end_lat"]
 
    config["fusion"]["lon_bounds"] = [start_lon, end_lon]
    config["fusion"]["lat_bounds"] = [start_lat, end_lat]

   
    fnames = []
    #Find all files
    for root, dirs, files in os.walk(fdir):
        for fle in files:
            mtch = re.search(TEMPO_BASIC_RE, fle)
            if mtch:
                fnames.append(os.path.join(root, fle))

    #Sort based on (1) datetime and (2) channel
    #TEMPO filename-specific indices for start time
    fnames = sorted(fnames, key = lambda x: (os.path.basename(x)[17:31]), reverse=True)
    
    out_files = []
    for f in range(len(fnames)):
        out_files.append(fnames[f].replace(".nc", ".tif")) 

    config["low_res"]["data"]["filenames"] = fnames
    config["low_res"]["data"]["geo_filenames"] = fnames
    config["output_files"] = out_files

    config_fname = build_config_fname_cf_gtiff_gen(yml_conf["config_dir"])
    with open(config_fname, 'w') as fle:
        yaml.dump(config, fle)

    return config


def run_tempo_clip(yml_conf):

    config_dict = copy.deepcopy(YAML_TEMPLATE_TEMPO_NCDF_TO_GTFF)
    config_dict = update_config_tempo_netcdf_to_gtiff(yml_conf, config_dict)
    print(config_dict.keys())
    resample_or_fuse_data(config_dict)

    return config_dict








