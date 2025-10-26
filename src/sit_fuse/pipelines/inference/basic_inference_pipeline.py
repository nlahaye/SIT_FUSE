import argparse

from sit_fuse.utils import read_yaml
import os
import yaml


from sit_fuse.pipelines.inference.inference_utils import run_basic_inference_geolocation

#This pipeline assumes context assignment and training have already been done
#Other pipelines will handle these functions.

def run_basic_inference(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    run_basic_inference_geolocation(yml_conf)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #yml_conf = read_yaml(args.yaml)

    run_basic_inference(args.yaml)

