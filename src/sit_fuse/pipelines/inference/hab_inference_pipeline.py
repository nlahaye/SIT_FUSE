
from sit_fuse.utils import read_yaml

from sit_fuse.pipelines.habs.hab_post_inference_pipeline import run_hab_post_inference_pipeline
 
from sit_fuse.pipelines.inference.inference_utils import run_inferece_only


def run_multi_sensor_hab_inference(yml_fpath):

   run_inferece_only(yml_conf) 


   post_inference_conf = read_yaml(yml_conf["post_inference_config"])
   run_hab_post_inference_pipeline(post_inference_conf)




