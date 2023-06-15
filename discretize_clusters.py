#Input Parsing
import yaml
import argparse

import matplotlib
matplotlib.use('agg')

import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from CMAP import CMAP, CMAP_COLORS

import torch
import numpy as np

import dask
import dask.array as da

from utils import read_yaml

def plot_clusters(coord, labels, output_basename, min_clust, max_clust, pixel_padding = 1):

        n_clusters_local = max_clust - min_clust

        data = []
        max_dim1 = max(coord[:,1])
        max_dim2 = max(coord[:,2])
        strt_dim1 = 0
        strt_dim2 = 0
 
        if os.path.exists(output_basename + "_" + str(n_clusters_local) + "clusters.zarr"):
            return

        #1 subtracted to separate No Data from areas that have cluster value 0.
        data = np.zeros((((int)(max_dim1-strt_dim1)+1+pixel_padding), ((int)(max_dim2-strt_dim2)+pixel_padding+1))) - 1 
        labels = np.array(labels)
        print("ASSIGNING LABELS", min_clust, max_clust)
        print(data.shape, labels.shape, coord.shape)
        for i in range(labels.shape[0]):
            data[coord[i,1], coord[i,2]] = labels[i]

        print("FINISHED WITH LABEL ASSIGNMENT")
        print("FINAL DATA TO DASK")
        data2 = da.from_array(data)
        #del data

        da.to_zarr(data2,output_basename + "_" + str(n_clusters_local) + "clusters.zarr", overwrite=True)
        img = plt.imshow(data, vmin=-1, vmax=max_clust)
        print("HERE CLUSTERS MIN MAX MEAN STD", data.min(), data.max(), data.mean(), data.std())
        cmap = ListedColormap(CMAP_COLORS[0:int(max_clust - (-1) + 1)])
        img.set_cmap(cmap)
        plt.colorbar()
        plt.savefig(output_basename + "_" + str(n_clusters_local) + "clusters.png", dpi=400, bbox_inches='tight')
        plt.clf()


def main(yml_fpath):

    config = read_yaml(yml_fpath)    

    dat = config["data"]["filenames"] 

    for i in range(len(dat)):
        if not os.path.exists(dat[i]):
            continue

        print(dat[i])
        data = torch.load(dat[i]).numpy()
        indices = torch.load(dat[i] + ".indices")

        max_cluster = data.shape[1]
        min_cluster = 0
        disc_data = np.argmax(data, axis = 1)
        del data
        print(disc_data.shape, disc_data.mean(), np.unique(disc_data).shape)
    
        plot_clusters(indices, np.squeeze(disc_data), dat[i], min_cluster, max_cluster)



if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument("-y", "--yaml", help="YAML file for cluster discretization.")
        args = parser.parse_args()
        from timeit import default_timer as timer
        start = timer()
        main(args.yaml)
        end = timer()
        print(end - start) # Time in seconds, e.g. 5.38091952400282




