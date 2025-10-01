
from sit_fuse.utils import read_yaml

from sit_fuse.pipelines.habs.hab_post_inference_pipeline import run_hab_post_inference_pipeline
 
from sit_fuse.pipelines.inference.inference_utils import run_inferece_only


def run_multi_sensor_hab_inference(yml_fpath):


    yml_conf = read_yaml(yml_fpath)

 
    for instrument in yml_conf["training_config"]:
        for key in yml_conf["training_config"][instrument]
            training_config = 
        training_config  = yml_conf[instrument][key]["training_config"]


    config_dict = copy.deepcopy(training_conf)
    config_dict = update_config_inference(yml_conf, training_conf, config_dict)
    run_prediction(config_dict)

    #Dump to file
    config_fname = build_config_fname_gtiff_gen(config_dir, "model", yml_conf["run_uid"] + "_inference.yaml")
    with open(config_fname, 'w') as fle:
        yaml.dump(config_dict, fle)


   run_inferece_only(yml_conf) 


   post_inference_conf = read_yaml(yml_conf["post_inference_config"])
   run_hab_post_inference_pipeline(post_inference_conf)




