"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
from timeit import default_timer as timer
from osgeo import osr, gdal
import os
import numpy as np
import random
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse

import sys
sys.setrecursionlimit(4500)

from sit_fuse.utils import read_yaml, get_read_func, get_scaler

import pickle
from joblib import load, dump

#ML imports
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.cluster import MiniBatchKMeans

from skimage.util import view_as_windows

def main(yml_fpath):
        """
        Function used if code is called as executable. Generates data and indices in preprocessed format and 
        saves to files. Can be reaccessed via read_data_preprocessed.

        :param yml_fpath: Path to YAML configuration.
        
        Values required to be in YAML configuration file:

        /data : Sub-dictionary that contains parameters about dataset.
        /data/files_train : List of files tp be used for training.
        /data/pixel_padding : Number of pixels to extend per-pixel/per-sample 'neighborhood' away from center sample of focus. Can be 0.
        /data/reader_type : Name of reader key (see utils documentation) to get the appropriate data reader function.
        /data/reader_kwargs : Kwargs for reader function.
        /data/fill_value : Fill value to use for unusable pixels/samples.
        /data/chan_dim : Index of the channel dimension when it is read in from read_func
        /data/valid_min : Smallest valid value in data. All values less than this threshold will be set to a fill value and not used.
        /data/valid_max : Largest valid value in data. All values greater than this threshold will be set to a fill value and not used.
        /data/delete_chans : Set of channels to be unused/deleted prior to preprocessing. Can be empty.
        /data/scale_data : Boolean indicating whether or not to scale data.
        /data/transform_default/chans: Typically unused.Optional channels to have special transforms applied to pixels out of expected ranges prior to filling. Can be empty list ([]).
        /data/transform_default/transform: Typically unused. Optional values associated with transform_chans. Values to be used for out of range samples in each of the channels specified in transform_chans. Can be empty list ([]).
        /output/out_dir : Directory to write data to.
        """
        #Translate config to dictionary 
        yml_conf = read_yaml(yml_fpath)

        full_data_inds =  yml_conf["data"]["train_inds_full"]
        subset_inds = yml_conf["data"]["train_inds_subset"]

        train_gtiffs = yml_conf["data"]["train_gtiffs"]

        full_inds = np.load(full_data_inds, allow_pickle=True)
        sub_inds = np.load(subset_inds, allow_pickle=True)

        #final_inds = []
        sm = 0
        for i in range(sub_inds.shape[0]):
            ind_set = sub_inds[i] #full_inds[sub_inds[i],:]
            print(ind_set.shape)
            sm += ind_set.shape[0]
            #final_inds.extend(ind_set)
        #final_inds = np.array(final_inds)
        

        print(full_inds.shape, sm)

        for i in range(len(train_gtiffs)):
            file_inds = full_inds[:,0]
            print(file_inds.shape, file_inds.max(), file_inds.min()) 

            dat = gdal.Open(train_gtiffs[i])
            img_data = dat.ReadAsArray()
            print(img_data.shape)
            new_dat = np.zeros((img_data.shape[1],img_data.shape[2]))


            inds = np.where(file_inds == i)
            useable_inds = None
            if len(inds) > 0:
                useable_inds = full_inds[inds[0],1:]
                print(useable_inds.shape)

            if useable_inds is not None:
                for j in range(useable_inds.shape[0]):
                    new_dat[useable_inds[j,0],useable_inds[j,1]] = 1
                    new_dat[useable_inds[j,0]-1,useable_inds[j,1]-1] = 2
                    new_dat[useable_inds[j,0]-1,useable_inds[j,1]] = 2
                    new_dat[useable_inds[j,0]-1,useable_inds[j,1]+1] = 2
                    new_dat[useable_inds[j,0],useable_inds[j,1]-1] = 2
                    new_dat[useable_inds[j,0],useable_inds[j,1]+1] = 2
                    new_dat[useable_inds[j,0]+1,useable_inds[j,1]-1] = 2
                    new_dat[useable_inds[j,0]+1,useable_inds[j,1]] = 2
                    new_dat[useable_inds[j,0]+1,useable_inds[j,1]+1] = 2

            metadata=dat.GetMetadata()
            geoTransform = dat.GetGeoTransform()
            wkt = dat.GetProjection()
            gcpcount = dat.GetGCPCount()
            gcp = None
            gcpproj = None
            if gcpcount > 0:
                gcp = dat.GetGCPs()
                gcpproj = dat.GetGCPProjection()
            dat.FlushCache()
            dat = None

            nx = img_data.shape[2]
            ny = img_data.shape[1]

            file_ext = ".train_coverage"
            fname = train_gtiffs[i] + file_ext + ".tif"
            print(fname)
            #sys.exit(-1)
            out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Int16)
            out_ds.SetMetadata(metadata)
            out_ds.SetGeoTransform(geoTransform)
            out_ds.SetProjection(wkt)
            if gcpcount > 0:
                out_ds.SetGCPs(gcp, gcpproj)
            out_ds.GetRasterBand(1).WriteArray(new_dat)
            out_ds.FlushCache()
            out_ds = None
 


if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument("-y", "--yaml", help="YAML file for data config.")
        args = parser.parse_args()
        from timeit import default_timer as timer
        start = timer()
        main(args.yaml)
        end = timer()
        print(end - start) # Time in seconds, e.g. 5.38091952400282
