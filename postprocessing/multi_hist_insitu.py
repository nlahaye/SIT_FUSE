


"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""

import numpy as np
from utils import numpy_to_torch, read_yaml, insitu_hab_to_multi_hist
from osgeo import gdal, osr
import argparse
import os
from pandas import DataFrame as df
from skimage.util import view_as_windows
from copy import deepcopy
from datetime import datetime

def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    start_date = datetime.strptime( yml_conf['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime( yml_conf['end_date'], '%Y-%m-%d')
    insitu_hab_to_multi_hist(yml_conf['xl_fname'], start_date, end_date,
		yml_conf['clusters_dir'], yml_conf['clusters'], yml_conf['radius_degrees'],
                yml_conf['ranges'], yml_conf['global_max'],
                yml_conf['files_test'],
		yml_conf['files_train'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    main(args.yaml)





