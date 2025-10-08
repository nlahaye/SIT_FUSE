import os
import argparse

from sit_fuse.utils import read_yaml
from sit_fuse.pipelines.data_transform_and_prep.pv_tile_and_mask_utils import run_pv_grid_and_mask

def pv_grid_and_mask(yml_conf):

    gridded_fnames = run_pv_grid_and_mask(yml_conf)
    return gridded_fnames


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary
    yml_conf = read_yaml(args.yaml)

    pv_grid_and_mask(yml_conf)

