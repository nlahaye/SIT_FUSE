#General Imports
import matplotlib
matplotlib.use('agg')

import os
import matplotlib.pyplot as plt
import numpy as np
import random

import sys
sys.setrecursionlimit(4500)

#ML imports
import torch
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler

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

class RSClustering(object):
  
    def __init__(self, pixel_padding = 1, branch = 5, thresh = 1e-5, 
        train_sample_size = 1000, number_train_epochs = 75, n_clusters = 500, predict_data_chunk = 10000,
        clustering = None, min_clust = None, max_clust = None, out_dir = "", train = True, reset_n_clusters = True):

        self.pixel_padding = pixel_padding
        self.branch = branch
        self.thresh = float(thresh)
        self.train_sample_size = train_sample_size
        self.number_train_epochs = number_train_epochs
        self.predict_data_chunk = predict_data_chunk
        self.n_clusters = n_clusters
        self.clustering = clustering
        self.min_clust = min_clust
        self.max_clust = max_clust 
        self.out_dir = out_dir
        self.train = train
        self.reset_n_clusters = reset_n_clusters

        if self.clustering is None:
            self.clustering = Birch(branching_factor=self.branch, threshold=self.thresh, n_clusters=None)
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

        #1 subtracted to seperate No Data from areas that have cluster value 0.
        data = np.zeros(((int)(max_dim1-strt_dim1)+self.pixel_padding, (int)(max_dim2-strt_dim2)+self.pixel_padding)) - 1
        for i in range(len(labels)):
            data[int(coord[i,1]), int(coord[i,2])] = labels[i]

        np.save(output_basename + ".npy", data)
        img = plt.imshow(data, vmin=self.min_clust, vmax=self.max_clust)
        cmap = ListedColormap(CMAP_COLORS[0:int(data.max() - data.min() + 1)])
        img.set_cmap(cmap)
        plt.colorbar()
        plt.savefig(output_basename + ".png", dpi=400)
        plt.clf()

        return data


    def __predict_cluster__(self, data):
        
        labels =  np.zeros(data.shape[0])
        for i in range(0, data.shape[0], self.predict_data_chunk):
            end_ind = i + self.predict_data_chunk
            if end_ind > data.shape[0]:
                end_ind = data.shape[0]-1
            labels[i:end_ind+1] = self.clustering.predict(data[i:end_ind+1])
        return labels


    def __train_partial__(self, data):
        
        if data.shape[0] < self.train_sample_size:
            self.train_sample_size = 100
        elif int(self.number_train_epochs/self.train_sample_size) > int(data.shape[0]/self.train_sample_size):
            self.number_train_epochs = int(data.shape[0]/self.train_sample_size)
        indices = np.random.choice(data.shape[0], size=self.number_train_epochs, replace=False)
        for n in indices:
            fnl_ind = n + self.train_sample_size + 1
            if fnl_ind > data.shape[0]:
                fnl_ind = data.shape[0]
            subd = data[n:fnl_ind].numpy()
            self.clustering.partial_fit(subd)
       
 
    def __train_scalers__(self, data):
        self.scalers = []
        for n in range(data.shape[1]):
            self.scalers.append(StandardScaler())
            #slc = [slice(None)] * data.ndim
            #slc[1] = slice(n, n+1)
            subd = data[:,n]
            self.scalers[n].partial_fit(subd[np.where(subd > -9999)].reshape(-1, 1))      

    def __cluster_data__(self, data, indices, fname, scale = True):

        if scale:
            for c in range(data.shape[1]):
                subd = data[:,c].numpy()
                data[:,c] = torch.from_numpy(self.scalers[c].transform(subd.reshape(-1,1)).reshape(-1))

        labels = self.__predict_cluster__(data)
        for i in range(0, int(max(indices[:,0]))+1):
            inds = np.where(indices[:,0] == i)
            self.__plot_clusters__(indices[inds[0],:], labels[inds[0]], fname + ".clustering." + str(i))

    def run_clustering(self, train_data, test_data):

        if self.train:    
            train = torch.load(train_data)
            train_indices = torch.load(train_data + ".indices")
            self.__train_scalers__(train.data)

            for c in range(train.data.shape[1]):
                subd = train.data[:,c].numpy()
                train.data[:,c] = torch.from_numpy(self.scalers[c].transform(subd.reshape(-1,1)).reshape(-1))
            self.__train_partial__(train.data)          

            if self.n_clusters is not None and self.reset_n_clusters == True:
                self.clustering.set_params(n_clusters=self.n_clusters)
                self.clustering.partial_fit(None)
            self.min_clust = 0
            self.max_clust = self.n_clusters


        self.__cluster_data__(train.data, train_indices, train_data, False)

        test = torch.load(test_data)
        test_indices = torch.load(test_data + ".indices")
        
        self.__cluster_data__(test.data, test_indices, test_data, True) 

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
    number_train_epochs = yml_conf["clustering"]["number_train_epochs"]
    n_clusters = yml_conf["clustering"]["n_clusters"]
    predict_data_chunk = yml_conf["clustering"]["predict_data_chunk"]
    train = yml_conf["clustering"]["train"]
    reset_n_clusters = yml_conf["clustering"]["reset_n_clusters"]
    out_dir = yml_conf["output"]["out_dir"]
    model = yml_conf["clustering"]["model"]
    clustering = RSClustering(pixel_padding = pixel_padding, branch = branch,
        thresh = thresh, train_sample_size = train_sample_size, number_train_epochs = number_train_epochs,
        n_clusters = n_clusters, predict_data_chunk = predict_data_chunk, out_dir = out_dir, clustering = model,
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
       



