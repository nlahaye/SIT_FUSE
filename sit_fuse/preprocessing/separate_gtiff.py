"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""

from osgeo import gdal, ogr
import os
import argparse
from misc_utils import gtiff_to_gtiff_multfile
from utils import numpy_to_torch, read_yaml, get_read_func



def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    sep_kwargs = yml_conf["kwargs"]
    for i in range(len(yml_conf["fnames"])):
        fname = yml_conf["fnames"][i]
        gtiff_to_gtiff_multfile(fname, yml_conf["number_channels"], **sep_kwargs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    main(args.yaml)



