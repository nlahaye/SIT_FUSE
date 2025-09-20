
from sit_fuse.utils import read_yaml
import os
import yaml


#This pipeline assumes context assignment and training have already been done
#Other pipelines will handle these functions.

def run_basic_inference(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    run_basic_inference_geolocation(yml_conf)


