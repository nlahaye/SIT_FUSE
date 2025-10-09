import os
import argparse

from sit_fuse.utils import read_yaml

from sit_fuse.pipelines.training.pv_preprocess_and_train_utils import run_data_preprocessing, run_model_training


def run_pv_preprocess_and_train_pipeline(yml_conf):

    config_dict = run_data_preprocessing(yml_conf)

    run_model_training(config_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

    run_pv_preprocess_and_train_pipeline(yml_conf)




