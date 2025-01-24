"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""

from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import torch
from sw_approx import sw_approx
import zarr
import argparse
import numpy as np
from osgeo import gdal, osr
from sit_fuse.utils import get_read_func, read_yaml
import cv2
import os
import csv

from tqdm import tqdm
from tabulate import tabulate
from pprint import pprint

import diplib as dip

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import ot
import geopy.distance

from sklearn.cluster import MiniBatchKMeans


def load_image(fname):
    dat = gdal.Open(fname)
    imgData = dat.ReadAsArray()
    arr = imgData
    arr_tensor = torch.from_numpy(arr)

    return arr_tensor

def calc_swd(fname1, fname2):
    arr_tensor = load_image(fname1)
    arr_tensor2 = load_image(fname2)

    d2 = max(arr_tensor2.shape[1], arr_tensor.shape[1])
    d1 = max(arr_tensor2.shape[0], arr_tensor.shape[0])
    tmp2 = torch.zeros((d1,d2))
    tmp2[:arr_tensor2.shape[0],:arr_tensor2.shape[1]] =arr_tensor2
    tmp =torch.zeros((d1,d2))
    tmp[:arr_tensor.shape[0],:arr_tensor.shape[1]] = arr_tensor

    swd_value = sw_approx(tmp, tmp2)
    return swd_value        


def compare_parallel(data_fnames, njobs=70): 

    results = None
    with tqdm_joblib(tqdm(desc="My calculation", total=(len(data_fnames)* (len(data_fnames)-1)))) as progress_bar:
        results = Parallel(n_jobs=njobs)(
            delayed(calc_swd)(data_fnames[i], data_fnames[j])
            for i in range(len(data_fnames))
            for j in range(i + 1, len(data_fnames))
        )
 
    compare = torch.zeros((len(data_fnames),len(data_fnames)))
    index = 0
    for i in range(len(data_fnames)):
        for j in range(i + 1, len(data_fnames)):
            compare[i,j] = results[index]
            index += 1

    return compare

def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)
    #Assume GeoTiff only for the time being - geolocation info
    out_dir = yml_conf["data"]["out_dir"]
    data_fnames = yml_conf["data"]["filename"]
    data_reader =  yml_conf["data"]["reader_type"]
    data_reader_kwargs = yml_conf["data"]["reader_kwargs"]
    wrt_geotiff = yml_conf["write_geotiff"] 
    label_fname = yml_conf["data"]["label_fname"]


    conts = []

    compare_parallel(data_fnames)

    compare = compare.detach().cpu().numpy()
    print("DISTRIBUTION COMPARISIONS")
    print(tabulate(compare))

    kmeans = MiniBatchKMeans(n_clusters=50)
    kmeans.fit(compare)
    labels = kmeans.labels_

    label_dict = {}
    for i in range(len(data_fnames)):
        label_dict[data_fnames[i]] = [data_fnames[i], labels[i]]
 
    np.savez(os.path.join(out_dir, label_fname), **label_dict) 
    csv_file = os.path.join(out_dir, label_fname)+".csv"
    # Write the dictionary to a CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
 
        # Write the header row (keys of the dictionary)
        #writer.writerow(label_dict.keys())
 
        # Write the data rows (values of the dictionary)
        writer.writerows(label_dict.values())
 

def write_geotiff(dat, imgData, fname):

    nx = imgData.shape[1]
    ny = imgData.shape[0]
    geoTransform = dat.GetGeoTransform()
    wkt = dat.GetProjection()
    out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(geoTransform)
    out_ds.SetProjection(wkt)
    out_ds.GetRasterBand(1).WriteArray(imgData)
    out_ds.FlushCache()
    out_ds = None
 
def write_zarr(fname, imgData):

    zarr.save(fname + ".zarr", imgData)
    img = plt.imshow(imgData, vmin=-1, vmax=1)
    plt.savefig(fname + ".png", dpi=400, bbox_inches='tight') 
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)



