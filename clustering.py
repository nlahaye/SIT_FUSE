"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
#General Imports
import matplotlib
matplotlib.use('agg')

import os
import matplotlib.pyplot as plt
import numpy as np
import dask
import dask.array as da
import random
import pickle

import sys
import resource
max_rec = 10**6
# May segfault without this line. 100 is a guess at the size of each stack frame.
resource.setrlimit(resource.RLIMIT_STACK, [100*max_rec, resource.RLIM_INFINITY])
sys.setrecursionlimit(max_rec)

#ML imports
import torch
from sklearn.cluster import Birch
from dask_ml.preprocessing import StandardScaler
#from dask_ml.wrappers import Incremental
#from sklearn.preprocessing import StandardScaler

#Data
from dbn_datasets import DBNDataset
from utils import numpy_to_torch, read_yaml, get_read_func, get_scaler

#Input Parsing
import yaml
import argparse

#Plotting
from matplotlib.colors import ListedColormap
from CMAP import CMAP, CMAP_COLORS

#Serialization
from joblib import dump, load

import warnings
warnings.simplefilter("ignore") 

class RSClustering(object):
  
    def __init__(self, pixel_padding = 1, branch = 5, thresh = 1e-5, 
        train_sample_size = 1000, n_clusters = 500, clustering = None, min_clust = None,
        max_clust = None, out_dir = "", train = True, reset_n_clusters = True, scaler = None, train_scaler = True, chunks=2000):

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
        self.chunks = chunks  
 
        #TODO allow to load
        self.scaler = scaler
        self.train_scaler = train_scaler

        if self.clustering is None:
            #self.estimator = Birch(branching_factor=self.branch, threshold=self.thresh, n_clusters=None) #self.n_clusters)
            self.clustering = Birch(branching_factor=self.branch, threshold=self.thresh, n_clusters=None) #self.n_clusters)
            #self.clustering = Incremental(estimator=self.estimator)
        elif isinstance(self.clustering, str) and os.path.exists(self.clustering):
            #Given a known joblib/sklearn issue with highly recursive structures, cannot support reloading/online learning
            #     Can only support exporting centroids and doing predictions with initial model
            self.train = False
            self.clustering = Birch(branching_factor=self.branch, threshold=self.thresh, n_clusters=self.n_clusters)
            with open(os.path.join(os.path.dirname(self.clustering) + "cluster_scale.pkl")) as f2:
                self.scaler = load(f2)
            with open(self.clustering, "rb") as f:
                self.clustering.subcluster_centers_ = load(f)
            print("HERE LOADED SUBCLUST CENTERS")

    def __plot_clusters__(self, coord, labels, output_basename):

        n_clusters_local = self.max_clust - self.min_clust + 1
        if self.reset_n_clusters == True and self.n_clusters is not None:
            n_clusters_local = self.n_clusters

        data = []
        max_dim1 = max(coord[:,1])
        max_dim2 = max(coord[:,2])
        strt_dim1 = 0
        strt_dim2 = 0
   
        #print("PLOTTING") 
        #tmp = np.array((coord.shape[0]), dtype=np.int32)
        #tmp = coord[:,1].astype(np.int32)
        #tmp = (tmp*max_dim2)
        #tmp = tmp + coord[:,2].astype(np.int32) 
        #print("HERE0", max_dim1, max_dim2, coord.shape, tmp.shape)
        #coord_flat = da.from_array(tmp.reshape((1, tmp.shape[0]))) #(1,tmp.shape[0])
        #print("HERE ", tmp.min(), tmp.max(), tmp.mean(), coord_flat.min().compute(), coord_flat.max().compute(), coord_flat.mean().compute())

        #1 subtracted to separate No Data from areas that have cluster value 0.
        #data = da.zeros((((int)(max_dim1-strt_dim1)+1+self.pixel_padding)*((int)(max_dim2-strt_dim2)+self.pixel_padding+1)), chunks=2000) - 1
        data = np.zeros((((int)(max_dim1-strt_dim1)+1+self.pixel_padding), ((int)(max_dim2-strt_dim2)+self.pixel_padding+1))) - 1 #, chunks=2000) - 1
        #print("MOVING LABELS TO BE FULLY IN-MEMORY")
        labels = np.array(labels) #.compute())
        print("ASSIGNING LABELS")
        print(data.shape, labels.shape, coord.shape)
        for i in range(labels.shape[0]):
            data[coord[i,1], coord[i,2]] = labels[i]
        #da.slicing.setitem(data, labels, coord_flat)
        

        #print("TO NUMPY...")
        #data_tmp = np.array(data.compute()).reshape(((int)(max_dim1-strt_dim1)+1+self.pixel_padding, (int)(max_dim2-strt_dim2)+self.pixel_padding+1))
        print("FINISHED WITH LABEL ASSIGNMENT")
        print("FINAL DATA TO DASK")
        data2 = da.from_array(data)
        #del data

        da.to_zarr(data2,output_basename + "_" + str(n_clusters_local) + "clusters.zarr", overwrite=True)
        img = plt.imshow(data, vmin=-1, vmax=self.max_clust)
        print("HERE CLUSTERS MIN MAX MEAN STD", data.min(), data.max(), data.mean(), data.std()) 
        cmap = ListedColormap(CMAP_COLORS[0:int(self.max_clust - (-1) + 1)])
        img.set_cmap(cmap)
        plt.colorbar()
        plt.savefig(output_basename + "_" + str(n_clusters_local) + "clusters.png", dpi=400, bbox_inches='tight')
        plt.clf()

        return data2


    def __predict_cluster__(self, data):
        labels = self.clustering.predict(data).astype(np.int16)
        return labels


    def __train_clustering__(self, data):
        self.clustering.partial_fit(data) 
        #self.clustering.fit(data)
 
    def __train_scaler__(self, data):
        if self.scaler is None:
            self.scaler = StandardScaler()
        self.scaler.fit(data)
        #self.scaler.partial_fit(data)

    def __cluster_data__(self, data, indices, fname, scale = True):

        if scale:
            print("SCALING")
            data = self.scaler.transform(data)
            #for i in range(0, data.shape[0], 10000):
            #    data[i:i+10000] = self.scaler.transform(data[i:i+10000])
            #for c in range(data.shape[1]):
            #    tmp = np.array(data[:,c]).reshape(-1,1)
            #    tmp = self.scaler.transform().reshape(-1)
            #    data[:,c] = tmp

        print("HERE BEFORE PREDICTION")
        labels = np.zeros(data.shape[0], dtype=np.int16)
        for l in range(0, data.shape[0], 100000):
            start_ind = l
            end_ind = l + 100000
            if end_ind > data.shape[0]:
                end_ind = data.shape[0]
            labels[start_ind:end_ind] = self.__predict_cluster__(data[start_ind:end_ind,:])
        print("HERE AFTER PREDICTION")
        unique_files = np.unique(indices[:,0]).shape[0]
        self.min_clust = min(self.min_clust, min(labels))
        self.max_clust = max(self.max_clust, max(labels))
 
        if unique_files > 1:
            for i in range(unique_files):
                inds = np.where(indices[:,0] == i)
                self.__plot_clusters__(indices[inds[0],:], labels[inds[0]], fname + ".clustering" + str(i))
        else: 
            self.__plot_clusters__(indices, labels, fname + ".clustering")


    #TODO make reader functionality generic - take in files, indices files, and reader type, like DBN
    def run_clustering(self, train_data, test_data, scale = True):

        trn = []
        if self.train_scaler:
            if scale:
                print("TRAINING SCALERS")
                for i in range(len(train_data)): 
                    if not os.path.exists(train_data[i]):
                        continue
                    print(train_data[i])
                    #if ".data.input" in train_data[i]:
                    #    dat = da.from_array(torch.load(train_data[i]), chunks=20000)
                    #    #self.__train_scaler__(da.from_array(torch.load(train_data[i]), chunks=20000))
                    #else:
                    #    dat = da.from_array(torch.load(train_data[i]).detach().numpy(), chunks=20000)
                    #    #self.__train_scaler__(da.from_array(torch.load(train_data[i]).detach().numpy(), chunks=20000))
                    #for j in range(0, dat.shape[0], 1000000):
                    #    self.__train_scaler__(dat[j:j+1000000]) 
    
                    if ".data.input" in train_data[i]:
                        tmp = da.from_array(torch.load(train_data[i]), chunks=self.chunks)
                        if np.isnan(tmp.min().compute()) and np.isnan(tmp.min().compute()):
                            continue
                        trn.append(tmp)
                    else: 
                        tmp = da.from_array(torch.load(train_data[i]).detach().numpy(), chunks=self.chunks)
                        if np.isnan(tmp.min().compute()) and np.isnan(tmp.min().compute()):
                            continue
                        trn.append(tmp)
                    print(torch.load(train_data[i]).detach().numpy().min(), torch.load(train_data[i]).detach().numpy().max())
                    #self.__train_scaler__(trn[i])
                trn = da.concatenate(trn)
                self.__train_scaler__(trn)
                del trn
                #shuffle data
                np.random.seed(42)
        if self.train: 
            for i in range(len(train_data)):
                trn = []
                print(train_data[i])
                if ".data.input" in train_data[i]: 
                    tmp = da.from_array(torch.load(train_data[i]), chunks=self.chunks)
                    if np.isnan(tmp.min().compute()) and np.isnan(tmp.min().compute()):
                        continue
                    trn.append(tmp)
                else:
                    tmp = da.from_array(torch.load(train_data[i]).detach().numpy(), chunks=self.chunks)
                    if np.isnan(tmp.min().compute()) and np.isnan(tmp.min().compute()):
                            continue
                    trn.append(tmp)
                trn = da.concatenate(trn)    
                index = np.random.choice(trn.shape[0], trn.shape[0], replace=False)
                trn = da.slicing.shuffle_slice(trn, index)
                print(trn.shape, trn.min().compute(), trn.max().compute())
                trn = trn[:int(self.train_sample_size/len(train_data)),:]
                print(trn.shape)
                for j in range(0,trn.shape[0], 50000):
                    print("INIT TRAINING CLUSTERING MODEL")  
                    print(j, j+50000)
                    if scale:
                        trn[j:j+50000] = self.scaler.transform(trn[j:j+50000])
                    self.__train_clustering__(trn[j:j+50000]) 
                del trn

            #index = np.random.choice(trn.shape[0], trn.shape[0], replace=False)
            #trn = da.slicing.shuffle_slice(trn, index)
            #trn = trn[:self.train_sample_size,:] #500000,:]
            #print(trn.shape)

            #self.__train_scaler__(trn)
            #trn = trn[:self.train_sample_size,:] #500000,:]
            #print(trn.shape)
            #print("INIT TRAINING CLUSTERING MODEL")

            #trn = self.scaler.transform(trn)
            #self.__train_clustering__(trn) 

 
            print("HERE TEST", self.n_clusters, self.reset_n_clusters == True)
            if self.n_clusters is not None and self.reset_n_clusters == True:
                 print("FINAL CLUSTER TRAINING")
                 ##self.estimator = self.clustering._postfit_estimator
                 ##self.estimator.set_params(n_clusters=self.n_clusters)
                 self.clustering.set_params(n_clusters=self.n_clusters)
                 #self.clustering = Incremental(estimator=self.estimator)
                 self.clustering.partial_fit(None)
                #    #self.clustering.set_params(estimator__n_clusters=self.n_clusters)
                #    #print("HERE", self.estimator, self.clustering.estimator, self.estimator.get_params(), self.clustering.get_params())
                #    #self.clustering.fit(None)
                 self.min_clust = -1
                 self.max_clust = self.n_clusters
            else:
                self.min_clust = 999999
                self.max_clust = -999999

        train_indices = []
        trn = []
        print("RUNNING CLUSTERING")
        for i in range(len(train_data)):
            print("CLUSTERING", train_data[i])
            if ".data.input" in train_data[i]:
                trn = da.from_array(torch.load(train_data[i]), chunks=self.chunks)
            else:
                tmp = da.from_array(torch.load(train_data[i]).detach().numpy(), chunks=self.chunks)
                if np.isnan(tmp.min().compute()) and np.isnan(tmp.min().compute()):
                            continue
                trn = tmp
            train_indices = torch.load(train_data[i].replace(".input", "") + ".indices")
 
            self.__cluster_data__(trn, train_indices, os.path.join(self.out_dir, os.path.basename(train_data[i])), scale)

        for i in range(len(test_data)):
            print("CLUSTERING", test_data[i])
            if ".data.input" in test_data[i]:
                test = da.from_array(torch.load(test_data[i]), chunks=self.chunks)
            else:
                tmp = da.from_array(torch.load(test_data[i]).detach().numpy(), chunks=self.chunks)
                if np.isnan(tmp.min().compute()) and np.isnan(tmp.min().compute()):
                            continue
                test = tmp
            test_indices = torch.load(test_data[i].replace(".input", "") + ".indices")
        
            self.__cluster_data__(test, test_indices, os.path.join(self.out_dir, os.path.basename(test_data[i])), scale) 

        print("CLUSTERING COMPLETE")

    def save_clustering(self):
        os.makedirs(self.out_dir, exist_ok = True)
        with open(os.path.join(self.out_dir, "cluster_scale.pkl"), "wb") as f2:
            dump(self.scaler, f2, True, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.out_dir, "clustering_centroids.pkl"), "wb") as f:
            dump(self.clustering.subcluster_centers_, f, True, pickle.HIGHEST_PROTOCOL)
        #Given a known joblib/sklearn issue with highly recursive structures, cannot support reloading/online learning
        #     Can only support exporting centroids and doing predictions with initial model
        #with open(os.path.join(self.out_dir, "clustering_dummy_leaf.pkl"), "wb") as f:
        #    dump(self.clustering.dummy_leaf_, f, True, pickle.HIGHEST_PROTOCOL)
        #with open(os.path.join(self.out_dir, "clustering_root.pkl"), "wb") as f:
        #    dump(self.clustering.root_, f, True, pickle.HIGHEST_PROTOCOL)


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
    scale = yml_conf["scaler"]["scale"]
    scaler_type = yml_conf["scaler"]["name"]

    chunks = 2000
    if "chunks" in yml_conf["clustering"].keys():
        chunks = yml_conf["clustering"]["chunks"]


    scaler_train = True 
    if isinstance(model , str) and os.path.exists(model) and os.path.exists(os.path.join(out_dir, "cluster_scale.pkl")) and not train:
        scaler_train = False
        scaler = None
    else:
        scaler, scaler_train = get_scaler(scaler_type)

    clustering = RSClustering(pixel_padding = pixel_padding, branch = branch,
        thresh = thresh, train_sample_size = train_sample_size,
        n_clusters = n_clusters, out_dir = out_dir, clustering = model,
        train = train, reset_n_clusters = reset_n_clusters, scaler = scaler, train_scaler = scaler_train,
        chunks = chunks)



    train_data = yml_conf["files_train"]
    test_data = yml_conf["files_test"]
    clustering.run_clustering(train_data, test_data, scale)
    clustering.save_clustering()
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)
       



