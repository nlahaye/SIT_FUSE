import os
import argparse

from sit_fuse.pipelines.data_transform_and_prep.tempo_clip_utils import run_tempo_clip
from sit_fuse.utils import read_yaml

def tempo_clip_pipeline(yml_conf):

    run_tempo_clip(yml_conf)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary
    yml_conf = read_yaml(args.yaml)

    tempo_clip_pipeline(yml_conf)

