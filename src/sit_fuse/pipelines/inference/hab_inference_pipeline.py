
import argparse

from sit_fuse.utils import read_yaml
from sit_fuse.pipelines.habs.hab_post_inference_pipeline import run_hab_post_inference_pipeline
from sit_fuse.pipelines.habs.hab_training_utils import run_data_preprocessing

from sit_fuse.pipelines.inference.inference_utils import run_inference_only, update_config_inference, run_prediction, build_config_fname_inference

import yaml
import copy

def run_multi_sensor_hab_inference(yml_conf):

    config_dicts = run_data_preprocessing(yml_conf)

    #for instrument in yml_conf["instruments"]:
    #    for key in yml_conf["instruments"][instrument]:
    #        yml_conf["input_dir"] = yml_conf["instruments"][instrument][key]["input_dir"]
    #        yml_conf["input_pattern"] = yml_conf["instruments"][instrument][key]["input_pattern"]
    #
    #        if "input_clip_re" in yml_conf["instruments"][instrument][key]:
    #            yml_conf["input_clip_re"] =  yml_conf["instruments"][instrument][key]["input_clip_re"]
    #        elif "input_clip_re" in yml_conf:
    #            del yml_conf["input_clip_re"]
    #
    #        config_dict = config_dicts[instrument][key]
    #
    #        run_inference_only(yml_conf, config_dict)

    post_inference_conf = read_yaml(yml_conf["post_inference_config"])
    run_hab_post_inference_pipeline(post_inference_conf)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

    run_multi_sensor_hab_inference(yml_conf)



