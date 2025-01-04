


"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""

import numpy as np
from sit_fuse.utils import numpy_to_torch, read_yaml, insitu_hab_to_multi_hist
from osgeo import gdal, osr
import argparse
import os
import pytz
from pandas import DataFrame as df
from skimage.util import view_as_windows
from copy import deepcopy
import datetime

def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    start_date = datetime.datetime.strptime( yml_conf['start_date'], '%Y-%m-%d') #.replace(tzinfo=datetime.timezone.utc)#.tz_localize(None)
    end_date = datetime.datetime.strptime( yml_conf['end_date'], '%Y-%m-%d') #.replace(tzinfo=datetime.timezone.utc) #.tz_localize(None)

    #start_date = pytz.utc.localize(start_date)
    #end_date = pytz.utc.localize(end_date)

    karenia = False
    if 'Karenia' in yml_conf['xl_fname'] or 'karenia' in yml_conf['xl_fname']:
        karenia = True

    insitu_hab_to_multi_hist(yml_conf['xl_fname'], start_date, end_date,
		yml_conf['clusters_dir'], yml_conf['clusters'], yml_conf['radius_degrees'],
                yml_conf['ranges'], yml_conf['global_max'], yml_conf['input_file_type'], karenia)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    main(args.yaml)





