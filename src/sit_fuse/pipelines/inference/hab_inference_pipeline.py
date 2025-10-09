
import argparse

from sit_fuse.utils import read_yaml

from sit_fuse.pipelines.habs.hab_post_inference_pipeline import run_hab_post_inference_pipeline
 
from sit_fuse.pipelines.inference.inference_utils import run_inferece_only, update_config_inference, run_prediction, build_config_fname_inference

import yaml

def run_multi_sensor_hab_inference(yml_conf):

    for instrument in yml_conf["instruments"]:
        for key in yml_conf["instruments"][instrument]:
            training_config  = yml_conf["instruments"][instrument][key]["training_config"]

            print(instrument, key)

            config_dict = read_yaml(training_config)
            config_dict = update_config_inference(yml_conf, config_dict)
            run_prediction(config_dict)

            #Dump to file
            config_fname = build_config_fname_inference(yml_conf["config_dir"], "model_" + \
                    yml_conf["run_uid"] + "_" + instrument + "_" + key + "_inference")
            with open(config_fname, 'w') as fle:
                yaml.dump(config_dict, fle)


    post_inference_conf = read_yaml(yml_conf["post_inference_config"])
    run_hab_post_inference_pipeline(post_inference_conf)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

    run_multi_sensor_hab_inference(yml_conf)



