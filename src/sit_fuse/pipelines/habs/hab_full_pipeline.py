
import argparse

from sit_fuse.utils import read_yaml

from sit_fuse.pipelines.habs.hab_training_pipeline import run_hab_training_pipeline
from sit_fuse.pipelines.habs.hab_inference_pipeline import run_hab_inference_pipeline
from sit_fuse.pipelines.habs.hab_post_inference_pipeline import run_hab_post_inference_pipeline


def run_full_pipeline(yml_conf):
    """training -> inference (clusters [+ head outputs / MC / OOD]) -> post-inference [+ UQ]."""

    if yml_conf.get("run_training", True):
        run_hab_training_pipeline(yml_conf)
    if yml_conf.get("run_inference", True):
        run_hab_inference_pipeline(yml_conf)
    run_hab_post_inference_pipeline(yml_conf)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

    run_full_pipeline(yml_conf)
