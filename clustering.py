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
from dask_ml.preprocessing import StandardScaler
from dask_ml.wrappers import Incremental

#Data
from dbnDatasets import DBNDataset
from utils import numpy_to_torch, read_yaml, get_read_func

#Input Parsing
import yaml
import argparse

#Plotting
from matplotlib.colors import ListedColormap
from CMAP import CMAP, CMAP_COLORS

#Serialization
import pickle

import warnings
warnings.simplefilter("ignore") 

class RSClustering(object):
  
    def __init__(self, pixel_padding = 1, branch = 5, thresh = 1e-5, 
        train_sample_size = 1000, clustering = None, n_clusters = 500, min_clust = None,
        max_clust = None, out_dir = "", train = True, reset_n_clusters = True):

        self.pixel_padding = pixel_padding
        self.branch = branch
        self.thresh = float(thresh)
        self.train_sample_size = train_sample_size
        self.n_clusters = n_clusters
        self.clustering = clustering
        self.min_clust = min_clust
        self.max_clust = max_clust 
        self.out_dir = out_dir
        self.train = train
        self.reset_n_clusters = reset_n_clusters

        if self.clustering is None:
            self.estimator = Birch(branching_factor=self.branch, threshold=self.thresh, n_clusters=None)
            self.clustering = Incremental(estimator=self.estimator)
        elif isinstance(self.clustering, str) and os.path.exists(self.clustering):
            with open(self.clustering, "rb") as f:
                self.clustering = pickle.load(f)

    def __plot_clusters__(self, coord, labels, output_basename):

        n_clusters = self.max_clust - self.min_clust + 1

        data = []
        max_dim1 = max(coord[:,1])
        max_dim2 = max(coord[:,2])
        strt_dim1 = 0
        strt_dim2 = 0
    
        tmp = np.array(coord[:,1]*max_dim2 + coord[:,2]) 
        coord_flat = da.from_array(tmp.reshape((1,tmp.shape[0])))

        #1 subtracted to seperate No Data from areas that have cluster value 0.
        data = da.zeros((((int)(max_dim1-strt_dim1)+1+self.pixel_padding)*((int)(max_dim2-strt_dim2)+self.pixel_padding+1)), chunks=2000) - 1
        da.slicing.setitem(data, labels, coord_flat)

        print("TO NUMPY...")
        data_tmp = np.array(data.compute()).reshape(((int)(max_dim1-strt_dim1)+1+self.pixel_padding, (int)(max_dim2-strt_dim2)+self.pixel_padding+1))
        print("BACK TO DASK")
        data2 = da.from_array(data_tmp)

        da.to_zarr(data2,output_basename + ".zarr")
        img = plt.imshow(data, vmin=self.min_clust, vmax=self.max_clust)
        cmap = ListedColormap(CMAP_COLORS[0:int(data.max() - data.min() + 1)])
        img.set_cmap(cmap)
        plt.colorbar()
        plt.savefig(output_basename + ".png", dpi=400)
        plt.clf()

        return data


    def __predict_cluster__(self, data):
        labels = self.clustering.predict(data)
        return labels


    def __train_clustering__(self, data):
        print(data.shape, data.min(), data.max(), data.mean())
        self.clustering.fit(data)
 
    def __train_scaler__(self, data):
        self.scaler = StandardScaler()
        self.scaler.fit(data)

    def __cluster_data__(self, data, indices, fname, scale = True):

        if scale:
            data = self.scaler.transform(data)

        labels = self.__predict_cluster__(data)
        self.__plot_clusters__(indices, labels, fname + ".clustering")


    #TODO make reader functionality generic - take in files, indices files, and reader type, like DBN
    def run_clustering(self, train_data, test_data):

        trn = []
        if self.train:
            print("TRAINING SCALERS")
            for i in range(len(train_data)):   
                trn.append(da.from_array(torch.load(train_data[i]).detach().numpy(), chunks=2000))
                #self.__train_scaler__(trn[i])
            trn = da.concatenate(trn)
            #shuffle data
            np.random.seed(42)
            index = np.random.choice(trn.shape[0], trn.shape[0], replace=False)
            trn = da.slicing.shuffle_slice(trn, index)
            trn = trn[:self.train_sample_size,:] #500000,:]

            self.__train_scaler__(trn)
            print("INIT TRAINING CLUSTERING MODEL")

            trn = self.scaler.transform(trn)
            self.__train_clustering__(trn) 

            print("FINAL CLUSTER TRAINING")
            if self.n_clusters is not None and self.reset_n_clusters == True:
                self.estimator = self.clustering.estimator.set_params(n_clusters=self.n_clusters)
                self.clustering.partial_fit(None)
            self.min_clust = 0
            self.max_clust = self.n_clusters

        train_indices = []
        trn = []
        print("RUNNING CLUSTERING")
        for i in range(len(train_data)):
            print("CLUSTERING", train_data[i])
            trn = da.from_array(torch.load(train_data[i]).detach().numpy(), chunks=2000)
            train_indices = torch.load(train_data[i] + ".indices")
 
            self.__cluster_data__(trn, train_indices, train_data[i], True)

        for i in range(len(test_data)):
            print("CLUSTERING", test_data[i])
            test = da.from_array(torch.load(test_data[i]).detach().numpy(), chunks=2000)
            train_indices = torch.load(test_data[i] + ".indices")
        
            self.__cluster_data__(test, test_indices, test_data[i], True) 

        print("CLUSTERING COMPLETE")

    def save_clustering(self):
        os.makedirs(self.out_dir, exist_ok = True)
        with open(os.path.join(self.out_dir, "clustering.pkl"), "wb") as f:
            pickle.dump(self.clustering, f, pickle.HIGHEST_PROTOCOL)

def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    pixel_padding = yml_conf["clustering"]["pixel_padding"]
    branch = yml_conf["clustering"]["branch"]
    thresh = yml_conf["clustering"]["thresh"]
    train_sample_size = yml_conf["clustering"]["train_sample_size"]
    n_clusters = yml_conf["clustering"]["n_clusters"]
    train = yml_conf["clustering"]["train"]
    reset_n_clusters = yml_conf["clustering"]["reset_n_clusters"]
    out_dir = yml_conf["output"]["out_dir"]
    model = yml_conf["clustering"]["model"]
    clustering = RSClustering(pixel_padding = pixel_padding, branch = branch,
        thresh = thresh, train_sample_size = train_sample_size,
        n_clusters = n_clusters, out_dir = out_dir, clustering = model,
        train = train, reset_n_clusters = reset_n_clusters)



    train_data = yml_conf["files_train"]
    test_data = yml_conf["files_test"]
    clustering.run_clustering(train_data, test_data)
    clustering.save_clustering()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)
       



