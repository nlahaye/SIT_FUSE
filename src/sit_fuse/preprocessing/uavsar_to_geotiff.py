"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
import numpy as np
from utils import numpy_to_torch, read_yaml, get_read_func, get_lat_lon
from osgeo import gdal, osr
import argparse
import os
from pandas import DataFrame as df
from skimage.util import view_as_windows
from copy import deepcopy
from misc_utils import uavsar_to_geotiff

"""
def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    tile_data(yml_conf)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    main(args.yaml)
"""


data = ["/data/nlahaye/remoteSensing/UAVSAR/fire-scenes/L-band/SanAnd_26526_17122_004_171102_L090_CX_01/SanAnd_26526_17122_004_171102_L090HHHH_CX_01.grd",
"/data/nlahaye/remoteSensing/UAVSAR/fire-scenes/L-band/SanAnd_26526_17122_004_171102_L090_CX_01/SanAnd_26526_17122_004_171102_L090HVHV_CX_01.grd",
"/data/nlahaye/remoteSensing/UAVSAR/fire-scenes/L-band/SanAnd_26526_17122_004_171102_L090_CX_01/SanAnd_26526_17122_004_171102_L090VVVV_CX_01.grd"
]
#data = ["/data/nlahaye/remoteSensing/UAVSAR/fire-scenes/L-band/SanAnd_26526_17122_004_171102_L090_CX_01/SanAnd_26526_17122_004_171102_L090HHHH_CX_01.grd",
#"/data/nlahaye/remoteSensing/UAVSAR/fire-scenes/L-band/SanAnd_26526_17122_004_171102_L090_CX_01/SanAnd_26526_17122_004_171102_L090HVHV_CX_01.grd",
#"/data/nlahaye/remoteSensing/UAVSAR/fire-scenes/L-band/SanAnd_26526_17122_004_171102_L090_CX_01/SanAnd_26526_17122_004_171102_L090VVVV_CX_01.grd"
#]



#data = [
#[
#"/data/nlahaye/remoteSensing/UAVSAR/fire-scenes/L-band/caldor_08200_21049_026_210831_L090_CX_01/caldor_08200_21049_026_210831_L090HVHV_CX_01.grd",
#"/data/nlahaye/remoteSensing/UAVSAR/fire-scenes/L-band/caldor_08200_21049_026_210831_L090_CX_01/caldor_08200_21049_026_210831_L090HHHH_CX_01.grd",
#"/data/nlahaye/remoteSensing/UAVSAR/fire-scenes/L-band/caldor_08200_21049_026_210831_L090_CX_01/caldor_08200_21049_026_210831_L090VVVV_CX_01.grd"
#] #,
#[
#"/data/nlahaye/remoteSensing/UAVSAR/fire-scenes/L-band/caldor_26200_21048_013_210825_L090_CX_01/caldor_26200_21048_013_210825_L090HVHV_CX_01.grd" ,
#"/data/nlahaye/remoteSensing/UAVSAR/fire-scenes/L-band/caldor_26200_21048_013_210825_L090_CX_01/caldor_26200_21048_013_210825_L090HHHH_CX_01.grd",
#"/data/nlahaye/remoteSensing/UAVSAR/fire-scenes/L-band/caldor_26200_21048_013_210825_L090_CX_01/caldor_26200_21048_013_210825_L090VVVV_CX_01.grd"
#]
#]



out_dir = "/data/nlahaye/remoteSensing/UAVSAR/fire-scenes/L-band/Caldor_tiffs/"


reader_kwargs = {
    #"ann_fps" :[ "/data/nlahaye/remoteSensing/UAVSAR/fire-scenes/L-band/caldor_08200_21049_026_210831_L090_CX_01/uavsar.asf.alaska.edu_UA_caldor_08200_21049_026_210831_L090_CX_01_caldor_08200_21049_026_210831_L090_CX_01.ann",
    #"/data/nlahaye/remoteSensing/UAVSAR/fire-scenes/P-band/safire_14036_15102_007_150705_PL09043020_XX_01/safire_14036_15102_007_150705_PL09043020_05_XX_01.ann",
    #"/data/nlahaye/remoteSensing/UAVSAR/fire-scenes/L-band/caldor_26200_21048_013_210825_L090_CX_01/uavsar.asf.alaska.edu_UA_caldor_26200_21048_013_210825_L090_CX_01_caldor_26200_21048_013_210825_L090_CX_01.ann"
    #],
    "ann_fps": ["/data/nlahaye/remoteSensing/UAVSAR/fire-scenes/L-band/SanAnd_26526_17122_004_171102_L090_CX_01/uavsar.asf.alaska.edu_UA_SanAnd_26526_17122_004_171102_L090_CX_01_SanAnd_26526_17122_004_171102_L090_CX_01.ann"],
    "pol_modes": ['HVHV'], #,'HHHH','VVVV'],
    "start_line": 5400,
    "end_line": 6600,
    "start_sample": 10800,
    "end_sample": 19400,
    "clip": True}




uavsar_to_geotiff(data, out_dir, **reader_kwargs)



