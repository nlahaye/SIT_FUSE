import argparse

from sit_fuse.utils import read_yaml
from sit_fuse.pipelines.setup.weight_download_constants import *

import os
import yaml
from huggingface_hub import hf_hub_download



def pull_weights(project_key, model_output_dir):

    repo_id = "nlahaye/" + PROJECT_TO_REPO_MAP[project_key]

    hf_hub_download(repo_id, "wandb", subfolder="model_weights", local_dir=model_output_dir, local_dir_use_symlinks =False)
    hf_hub_download(repo_id, "encoder_scaler.pkl", local_dir=model_output_dir, local_dir_use_symlinks =False)

    if len(tile_tiers) > 0:
        for i in range(len(tile_tiers)):
            hf_hub_download(repo_id, "model_" + str(tile_tiers[i]) + ".ckpt", subfolder="tiled_masking", \
                local_dir=model_output_dir, local_dir_use_symlinks =False)
            hf_hub_download(repo_id, "clust_" + str(tile_tiers[i]) + ".ckpt", subfolder="tiled_masking", \
                local_dir=model_output_dir, local_dir_use_symlinks =False)





def run_model_weight_download(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    pull_weights(yml_conf["project_key"], yml_conf["model_output_dir"])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    run_model_weight_download(args.yaml)




