"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
import os
import argparse
import cv2
import re
import numpy as np
from osgeo import osr, gdal
from sit_fuse.preprocessing.misc_utils import clip_geotiff
from sit_fuse.utils import read_yaml


def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)

    filenames = yml_conf["data"]["filenames"]
    start_lon = yml_conf["data"]["start_lon"]
    start_lat = yml_conf["data"]["start_lat"]
    end_lon = yml_conf["data"]["end_lon"]
    end_lat = yml_conf["data"]["end_lat"]

    out_dir = yml_conf["output"]["out_dir"]

    bbox = (start_lon, start_lat, end_lon, end_lat)

  

    for i in range(len(filenames)):
        clip_geotiff(filenames[i], bbox, out_dir)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)


