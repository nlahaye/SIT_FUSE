

import os
import argparse

from sit_fuse.utils import read_yaml

from sit_fuse.pipelines.habs.hab_training_utils import run_data_preprocessing, run_model_training


def run_hab_training_pipeline(yml_conf):
 
    zarr_fname = os.path.join(yml_conf["geo_zarr"]["out_dir"], yml_conf["geo_zarr"]["out_uid"] + ".zarr")
    if not os.path.exists(zarr_fname):
        gen_geo_zarr_oc_daily(yml_conf)
 
    configs = run_data_preprocessing(yml_conf)

    run_model_training(yml_conf, configs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

    run_hab_training_pipeline(yml_conf)


