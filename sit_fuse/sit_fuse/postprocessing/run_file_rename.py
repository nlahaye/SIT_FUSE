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
import shutil
import datetime
from pandas import DataFrame as df
from skimage.util import view_as_windows
from copy import deepcopy

def rename_files(start_date, clusters_dir):

    for ind in range(0, 900):

        clust_fname = os.path.join(clusters_dir, "sif_finalday_" + str(ind) + ".karenia_brevis_bloom.tif")
        dqi_fname = os.path.join(clusters_dir, "sif_finalday_" + str(ind) + ".karenia_brevis_bloom.DQI.tif")

        if not os.path.exists(clust_fname):
            clust_fname = clust_fname + "f"
            if not os.path.exists(clust_fname):
                continue

        days_since = ind
        end_date = start_date + datetime.timedelta(days=days_since)
        new_fname = os.path.join(clusters_dir, end_date.strftime("%Y%m%d") + "_karenia_brevis.tif")
        new_dqi_fname = os.path.join(clusters_dir, end_date.strftime("%Y%m%d") + "_karenia_brevis.DQI.tif")
        shutil.move(clust_fname, new_fname)
        shutil.move(dqi_fname, new_dqi_fname)



def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    start_date = datetime.datetime.strptime( yml_conf['start_date'], '%Y-%m-%d')
    rename_files(start_date, yml_conf['clusters_dir']) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    main(args.yaml)




