"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn import metrics, manifold
from sklearn.preprocessing import MinMaxScaler
from pprint import pprint

#- Scientific stack
import scipy
from scipy.spatial.distance import pdist, squareform

#- Graph-stats 
from graspologic.embed import OmnibusEmbed, ClassicalMDS, AdjacencySpectralEmbed
from graspologic.simulations import rdpg


import gc

from resource import *
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

import pickle

from learnergy.models.deep import DBN, ConvDBN

from segmentation.models.gcn import GCN
from segmentation.models.deeplabv3_plus_xception import DeepLab
from segmentation.models.unet import UNetEncoder, UNetDecoder

import openTSNE

from sit_fuse.models.deep_cluster.dc import DeepCluster
from sit_fuse.models.encoders.cnn_encoder import DeepConvEncoder
from sit_fuse.models.deep_cluster.ijepa_dc import IJEPA_DC
from sit_fuse.models.deep_cluster.dbn_dc import DBN_DC
from sit_fuse.datasets.dataset_utils import get_prediction_dataset
from sit_fuse.models.deep_cluster.heir_dc import Heir_DC
from sit_fuse.utils import read_yaml
from sit_fuse.models.deep_cluster.multi_prototypes import MultiPrototypes

from tqdm import tqdm

import argparse
import os
import numpy as np
import sys

import dask
import dask.array as da

from osgeo import gdal, osr

from torchinfo import summary

import zarr

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sit_fuse.viz.CMAP import CMAP, CMAP_COLORS

from sit_fuse.inference.generate_output import run_inference, get_model

sys.setrecursionlimit(4500)


def run_tsne(embed, labels, final_labels, out_fname, pca_embed = False, indices = None):

    print(embed.shape, labels.shape, final_labels.shape)

    if embed.ndim > 2:
        embed = embed.reshape(-1, embed.shape[-1])
    final_labels = final_labels.flatten()
    labels = labels.flatten()
 
    if indices is not None:
        ind_tmp = zarr.load(indices)
    else:
        sub_ind_1 = np.where(final_labels == 0)[0]
        sub_ind_2 = np.where(final_labels == 1)[0]
        sub_ind_3 = np.where(final_labels == 2)[0]
        sub_ind_4 = np.where(final_labels == 3)[0]
        ind_final = np.concatenate((sub_ind_4, sub_ind_2))

        sub_sub_ind__3 = np.random.choice(sub_ind_3.shape[0], size=min(int(sub_ind_3.shape[0]), 10000), replace=False)
        sub_sub_ind__1 = np.random.choice(sub_ind_1.shape[0], size=min(int(sub_ind_1.shape[0]), 20000), replace=False)
        ind_tmp = np.concatenate((ind_final, sub_ind_3[sub_sub_ind__3], sub_ind_1[sub_sub_ind__1]))
 
        #indices = np.random.choice(embed.shape[0], size=min(int(embed.shape[0]*0.05), 100000), replace=False)
        #indices2 = np.where((final_labels == 1) | (final_labels == 3))[0]
        #ind_tmp = np.concatenate((list(indices), list(indices2)))
        print(np.unique(final_labels))
        zarr.save(out_fname + ".indices.zarr", ind_tmp)
        print(ind_tmp.shape, out_fname + ".indices.zarr")
    indices = np.unique(ind_tmp)
    print(indices.shape, "HERE INDICES")
    test_data = embed[indices, :]
    clust_data = labels[indices]
    clust_data_2 = final_labels[indices]
    print(test_data.shape, clust_data.shape, clust_data_2.shape)

    if pca_embed:
        tsne_data = test_data.cpu().numpy()
    else:
        #tsne_data = tsne.fit_transform(test_data)
        aff500 = openTSNE.affinity.PerplexityBasedNN(test_data,perplexity=2200, n_jobs=50, random_state=20)
        tsne_data = openTSNE.TSNE(n_jobs=50, verbose=True, metric="euclidean", exaggeration = 4,
                random_state=42).fit(affinities=aff500)
    #final = plot_tsne(tsne_data, clust_data, test_coord)

    #tsne_data = (tsne_data*100).astype(np.int32)

    shift_1 = abs(min(tsne_data[:,0]))
    shift_2 = abs(min(tsne_data[:,1]))

    tsne_data[:,0] = tsne_data[:,0] + shift_1
    tsne_data[:,1] = tsne_data[:,1] + shift_2

    scaler = MinMaxScaler()
    tsne_data = scaler.fit_transform(tsne_data)

    tsne_data = (tsne_data*1000).astype(np.int32)

    fnl = max(tsne_data[:,0])
    fnl2 = max(tsne_data[:,1])
    final = np.zeros((fnl+1, fnl2+1), dtype=np.float32) - 1.0
    final2 = np.zeros((fnl+1, fnl2+1), dtype=np.float32) - 1.0
    print(out_fname + ".TSNE_Clust.tif")
    for i in range(indices.shape[0]):
        final[tsne_data[i,0], tsne_data[i,1]] = clust_data[i] / 100.0
        final2[tsne_data[i,0], tsne_data[i,1]] = clust_data_2[i]

    out_ds = gdal.GetDriverByName("GTiff").Create(out_fname + ".TSNE_Clust.tif", final.shape[1], final.shape[0], 1, gdal.GDT_Float32)
    out_ds.GetRasterBand(1).WriteArray(final)
    out_ds.FlushCache()
    out_ds = None
 
    out_ds = gdal.GetDriverByName("GTiff").Create(out_fname + ".TSNE_Final.tif", final.shape[1], final.shape[0], 1, gdal.GDT_Float32)
    out_ds.GetRasterBand(1).WriteArray(final2)
    out_ds.FlushCache()
    out_ds = None



    out_ds = gdal.GetDriverByName("GTiff").Create(out_fname + ".TSNE_Clust_Round.tif", final.shape[1], final.shape[0], 1, gdal.GDT_Int32)
    out_ds.GetRasterBand(1).WriteArray(final.astype(np.int32))
    out_ds.FlushCache()
    out_ds = None

    return  test_data

def knn_graph(w, k, symmetrize=True, metric='euclidean'):
    '''
    :param w: A weighted affinity graph of shape [N, N] or 2-d array 
    :param k: The number of neighbors to use
    :param symmetrize: Whether to symmetrize the resulting graph
    :return: An undirected, binary, KNN graph of shape [N, N]
    '''
    w_shape = w.shape
    if w_shape[0] != w_shape[1]:
        w = np.array(squareform(pdist(w, metric=metric)))
            
    neighborhoods = np.argsort(w, axis=1)[:, -(k+1):-1]
    A = np.zeros_like(w)
    for i, neighbors in enumerate(neighborhoods):
        for j in neighbors:
            A[i, j] = 1
            if symmetrize:
                A[j, i] = 1
    return A

def build_knn_graph(embed, out_fname):
    if embed.ndim < 3:
        k=int(np.log(embed.shape[0]))
    else:
        k=int(np.log(embed.shape[0]* embed.shape[1]))
 
    knn_graph_out =  knn_graph(embed, k=k, symmetrize=True, metric='cosine')
    zarr.save(out_fname, knn_graph_out)


def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)

    test_fnames = yml_conf["data"]["files_test"]
    train_fnames = yml_conf["data"]["files_train"]
  
    data = None
    cntr = 0
    while data is None or data.data_full is None:
        data, _  = get_prediction_dataset(yml_conf, train_fnames[cntr])
        cntr = cntr + 1

    print(data.data_full.shape)

    model = get_model(yml_conf, data.data_full.shape[1])

    model = model.cuda()
    model.pretrained_model = model.pretrained_model.cuda()
    #model.pretrained_model.mlp_head = model.pretrained_model.mlp_head.cuda()
    if hasattr(model.pretrained_model, "pretrained_model"):
        model.pretrained_model.pretrained_model = model.pretrained_model.pretrained_model.cuda()
     
    for lab1 in model.clust_tree.keys():
        if lab1 == "0":
            continue
        for lab2 in model.lab_full.keys():
            if lab2 in model.clust_tree[lab1].keys():
                if model.clust_tree[lab1][lab2] is not None:
                    model.clust_tree[lab1][lab2] = model.clust_tree[lab1][lab2].cuda() 
  
 
    out_dir = yml_conf["output"]["out_dir"]

    tiled = yml_conf["data"]["tile"]

    tsne_perplexity = yml_conf["analysis"]["tsne"]["perplexity"]
    tsne_niter = yml_conf["analysis"]["tsne"]["niter"]
    tsne_njobs = yml_conf["analysis"]["tsne"]["njobs"]
    tsne_patience = yml_conf["analysis"]["tsne"]["patience"]
    tsne_lr = yml_conf["analysis"]["tsne"]["lr"]



    indices = [None]
    if "indices" in yml_conf["data"].keys():
        indices = yml_conf["data"]["indices"]

    embed_func = yml_conf["analysis"]["embed_func"]
    final_labels = yml_conf["data"]["final_labels"]

    knn_graphs = yml_conf["analysis"]["build_knn_graphs"]


    for i in range(len(test_fnames)):
        if isinstance(test_fnames[i], list):
            output_fle = os.path.join(out_dir, os.path.basename(os.path.splitext(test_fnames[i][0])[0]) + "." + embed_func)
        else:
            output_fle = os.path.join(out_dir, os.path.basename(os.path.splitext(test_fnames[i])[0]) + "." + embed_func)
        print(output_fle)
        data, output_file  = get_prediction_dataset(yml_conf, test_fnames[i])
        if data.data_full is None:
            print("SKIPPING", test_fnames[i], " No valid samples")
            continue
        context_labels = gdal.Open(final_labels[i]).ReadAsArray()
        print(data.targets_full.shape)
        if data.targets_full.ndim > 2:
            context_labels = context_labels[data.targets_full[0,:,1], data.targets_full[0,:,2]].flatten()
        else:
            if data.targets_full.shape[1] < 3:
                context_labels = context_labels[data.targets_full[:,0], data.targets_full[:,1]].flatten()
            else:
                print(data.targets_full[:,0].max(), data.targets_full[:,1].max(), data.targets_full[:,2].max())
                context_labels = context_labels[data.targets_full[:,1], data.targets_full[:,2]].flatten()
        if model.clust_tree_ckpt is not None:
            output, embed = run_inference(data, model, True, out_dir, output_fle + ".clust.data", tiled = tiled, return_embed =  True)
            output = output[:context_labels.shape[0]]
            embed = embed[:context_labels.shape[0]]
            is_pca = bool("pca" in embed_func )
            embed = run_tsne(embed, output, context_labels, output_fle, is_pca, indices[i])
            if knn_graphs and i == 0:	                
                build_knn_graph(embed, os.path.join(out_dir, embed_func + ".zarr"))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)

    print(getrusage(RUSAGE_SELF))


