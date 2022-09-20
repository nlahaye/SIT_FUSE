#General Imports
import matplotlib
matplotlib.use('agg')

import os
import matplotlib.pyplot as plt
import numpy as np
import dask
import dask.array as da
import random

import sys
sys.setrecursionlimit(10**6)

#ML imports
import torch
from sklearn.cluster import Birch
#from dask_ml.preprocessing import StandardScaler
#from dask_ml.wrappers import Incremental
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

from utils import read_s6_netcdf

def main(yml_fpath):
  
    yml_conf = read_yaml(yml_fpath)

    scaler = StandardScaler()
    #scaler = MaxAbsScaler()
 
    files = yml_config["filenames"]

    for i in range(len(files)):
        data = np.log(read_s6_netcdf(files[i]))
        data = data.reshape((1, data.shape[1]*data.shape[2]))
        scaler.partial_fit(data)


    min_data = 1000000001
    max_data = -9999999
    for i in range(len(files)):
        data = np.log(read_s6_netcdf(files[i]))
        shape = data.shape
        data = scaler.transform(data.reshape((1, data.shape[1]*data.shape[2]))).reshape(shape)
        min_data = min(min_data, data.min())
        max_data = max(max_data, data.max())

 
        #plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
    
        img = plt.imshow(np.squeeze(data)) #, vmin=min_data, vmax=max_data)
        plt.grid(False)
        plt.axis('off')
        #cmap = ListedColormap(CMAP_COLORS[0:int(self.max_clust - (-1) + 1)])
        img.set_cmap('nipy_spectral')
        plt.colorbar()
        plt.savefig(files[i] + ".log.png", dpi=400, bbox_inches='tight')
        plt.clf() 
 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)





