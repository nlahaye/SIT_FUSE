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

import h5py

import copy
import gc

from resource import *
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

import pickle
import joblib

from learnergy.models.deep import DBN, ConvDBN

from segmentation.models.gcn import GCN
from segmentation.models.deeplabv3_plus_xception import DeepLab
from segmentation.models.unet import UNetEncoder, UNetDecoder

import openTSNE
import umap

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

FINAL_DCT = {
        "proj_y": [],
        "proj_x": [],
        "bb_x": [],
        "bb_y": [],
        "bb_width": [],
        "bb_height": [],
        "heir_label": [],
        "final_label": [],
        "no_heir_label": []
    }


def prep_projection(embed, labels, coord, final_labels, out_fname, pca_embed = False, indices = None, recon_arr = None, recon_lab = None, init_shape = [], final_labels_func = None):


    coord_coord_1 = 1
    coord_coord_2 = 2
    if coord.ndim < 3 and coord.shape[1] < 3:
        coord_coord_1 = 0
        coord_coord_2 = 1

    print("HERE INIT SHAPE", init_shape)
    if recon_lab is None or  recon_arr is None:
        if final_labels is not None:
            print("HERE ERROR", embed.shape, labels.shape, final_labels.shape, np.unique(labels), len(np.unique(labels)))
        if len(init_shape) > 0:
            original_shape = init_shape[0]
        else:
            original_shape = (max(coord[:,coord_coord_1])+1, max(coord[:,coord_coord_2])+1)
        #for i in range(coord.shape[1]):
        #    print(max(coord[:,i]), coord.shape, i)
        if embed.ndim  == 4:
            embed = np.transpose(embed, axes=(0,2,3,1))
            reconstructed_arr = np.zeros((original_shape[0], original_shape[1], embed.shape[3]), dtype=np.float32)
            reconstructed_labels = np.zeros((original_shape[0], original_shape[1]), dtype=np.float32) - 1
        else:
            reconstructed_arr = np.zeros((original_shape[0], original_shape[1], embed.shape[1]), dtype=np.float32)
            reconstructed_labels = np.zeros((original_shape[0], original_shape[1]), dtype=np.float32) - 1

        print(original_shape, coord.shape, reconstructed_arr.shape, reconstructed_labels.shape)
        print(embed.shape, labels.shape)

        if embed.ndim  == 4:
            for i in range(embed.shape[0]):
                print(coord[i], i, embed.shape, reconstructed_arr.shape) 
                if coord[i,coord_coord_2]+embed.shape[2] > reconstructed_arr.shape[1] or coord[i,coord_coord_1]+embed.shape[1] > reconstructed_arr.shape[0]:
                    continue
                reconstructed_arr[coord[i,coord_coord_1]:coord[i,coord_coord_1]+embed.shape[1], coord[i,coord_coord_2]:coord[i,coord_coord_2]+embed.shape[2], :] = embed[i,:,:,:]
                reconstructed_labels[coord[i,coord_coord_1]:coord[i,coord_coord_1]+labels.shape[1], coord[i,coord_coord_2]:coord[i,coord_coord_2]+labels.shape[2]] = labels[i,:,:]
                #reconstructed_final_labels[coord[i,0], coord[i,1]] = final_labels[i]
        else:
            for i in range(embed.shape[0]):
                reconstructed_arr[coord[i,coord_coord_1], coord[i,coord_coord_2]] = embed[i]
                reconstructed_labels[coord[i,coord_coord_1], coord[i,coord_coord_2]] = labels[i]
                #reconstructed_final_labels[coord[i,0], coord[i,1]] = final_labels[i]

        zarr.save(out_fname + ".embeddings.zarr", reconstructed_arr)
        zarr.save(out_fname + ".embedding_labels.zarr", reconstructed_labels)

    else:
        reconstructed_arr = recon_arr
        reconstructed_labels = recon_lab

    recon_coord = np.indices(reconstructed_arr.shape[0:2])
    recon_coord = np.moveaxis(recon_coord, 0,2)
    print(recon_coord.shape, reconstructed_arr.shape, recon_coord.max(), recon_coord[:,0].max(), recon_coord[:,1].max())
    recon_coord = recon_coord.reshape(-1,2)
    print(recon_coord.shape, reconstructed_arr.shape, recon_coord.max(), recon_coord[:,0].max(), recon_coord[:,1].max())

 
    if final_labels is not None:
        final_labels = final_labels[:reconstructed_labels.shape[0],:reconstructed_labels.shape[1]]
        final_labels = final_labels.flatten()
    print("INTERMEDIATE RECON SHAPE", reconstructed_arr.shape, reconstructed_labels.shape, np.unique(final_labels))
    embed = reconstructed_arr.reshape(-1, reconstructed_arr.shape[-1])
    labels = reconstructed_labels.reshape(-1, 1)
    labels = labels.flatten()
    #print("FLATTENED RECON SHAPE", embed.shape, final_labels.shape, labels.shape, indices)
    if indices is not None:
        ind_tmp = zarr.load(indices)
    elif final_labels is not None:
        #sub_inds = []
        #for label in np.unique(final_labels):
        #    if label > 0.0:
        #        sub_ind = np.where(final_labels == label)[0]
        
        ind_tmp = final_labels_func(final_labels)
  
        print(np.unique(final_labels))
        print(ind_tmp)
        print(out_fname + ".indices.zarr")
        zarr.save(out_fname + ".indices.zarr", ind_tmp)
        print(ind_tmp.shape, out_fname + ".indices.zarr")
    else:
        ind_tmp = np.where((labels > 0.0)) #TODO configurable #np.indices(labels.shape)
        zarr.save(out_fname + ".indices.zarr", ind_tmp)
    indices = np.unique(ind_tmp)
    print(indices.shape, "HERE INDICES", np.unique(labels[indices]), len(np.unique(labels[indices])), embed.shape, recon_coord.shape)
    test_data = embed[indices, :]
    clust_data = labels[indices]
    if final_labels is not None:
        clust_data_2 = final_labels[indices]
    print("RECON COORD TEST", recon_coord[:,0].max(), recon_coord[:,1].max())
    recon_coord = recon_coord[indices, :]
    print("RECON COORD TEST2", recon_coord[:,0].max(), recon_coord[:,1].max())
    
    #reducer = loop_dict["reducer"]
    #landmarks = loop_dict["landmarks"]

    return {"test_data":test_data, "recon_coord":recon_coord, "clust_data":clust_data, "clust_data_2":clust_data_2, "out_fname":out_fname}


def run_proj(full_embed, is_pca, n_neighbors=20, min_dist=0.1, spread = 1, n_components=2):

    if is_pca:
        return None
    reducer = umap.UMAP(metric="cosine", n_neighbors=n_neighbors, min_dist=min_dist, spread = spread, n_components=n_components)
    reducer.fit(full_embed)
    return reducer
  
    #if not pca_embed and reducer is None:
    #    reducer = umap.ParametricUMAP()
    #    reducer.optimizer = keras.optimizers.Adam(1e-5, clipvalue=0.1)

    #    reducer.fit(test_data)
     

    #if not pca_embed: 
 
    #    if landmarks is None:
    #        landmark_idx = list(np.random.choice(range(test_data.shape[0]), int(test_data.shape[0]/100), replace=False))
    #        new_landmarks = reducer.transform(test_data[landmark_idx]) 
    #    else:
    #        landmarks = np.stack([np.array([np.nan, np.nan])]*test_data.shape[0] + list(landmarks))
    #        reducer.landmark_loss_weight = 0.01
    #        reducer.fit(test_data, landmark_positions=landmarks)
    #        transformed = reducer.transform(test_data)
    #        landmark_idx = list(np.random.choice(range(test_data.shape[0]), int(test_data.shape[0]/100), replace=False))
    #        new_landmarks = np.concatenate((landmarks, transformed[landmark_idx])) 
    
 
    #return {"test_data":test_data, "reducer":reducer, "landmarks":new_landmarks, "recon_coord":recon_coord, "clust_data":clust_data, "clust_data_2":clust_data_2, "out_fname":out_fname}



def transform_and_save(recon_coord, test_data, pca_embed, out_fname, clust_data, clust_data_2, reducer = None, scaler = None):

    if not pca_embed:
        projection_data = reducer.transform(test_data)
    else:
        projection_data = test_data

    #if scaler is not None:
    #    projection_data = scaler.transform(projection_data)
 

    shift_1 = abs(min(projection_data[:,0]))
    shift_2 = abs(min(projection_data[:,1]))

    projection_data[:,0] = projection_data[:,0] + shift_1
    projection_data[:,1] = projection_data[:,1] + shift_2
 
    projection_data = (projection_data*10).astype(np.int32)

    fnl = int(max(projection_data[:,0]))
    fnl2 = int(max(projection_data[:,1]))
    final = np.zeros((fnl+1, fnl2+1), dtype=np.float32) - 1.0
    final2 = np.zeros((fnl+1, fnl2+1), dtype=np.float32) - 1.0
    print(out_fname + ".UMAP_Clust.tif", np.unique(final), np.unique(final2))


    final_dct = copy.deepcopy(FINAL_DCT)


    for i in range(recon_coord.shape[0]):

        final_dct["proj_y"].append(int(projection_data[i,0]))
        final_dct["proj_x"].append(int(projection_data[i,1]))
        final_dct["bb_x"].append(recon_coord[i,1])
        final_dct["bb_y"].append(recon_coord[i,0])
        final_dct["bb_width"].append(3) #TODO generalize
        final_dct["bb_height"].append(3)
        final_dct["heir_label"].append(clust_data[i] / 100.0)
        if clust_data_2 is not None:
            final_dct["final_label"].append(clust_data_2[i])
        final_dct["no_heir_label"].append(int(clust_data[i] / 100.0))

        final[int(projection_data[i,0]), int(projection_data[i,1])] = clust_data[i] / 100.0
        if clust_data_2 is not None:
            final2[int(projection_data[i,0]), int(projection_data[i,1])] = clust_data_2[i]

    np.save(out_fname + ".viz_dict.npy", final_dct)


    out_ds = gdal.GetDriverByName("GTiff").Create(out_fname + ".UMAP_Clust.tif", final.shape[1], final.shape[0], 1, gdal.GDT_Float32)
    out_ds.GetRasterBand(1).WriteArray(final)
    out_ds.FlushCache()
    out_ds = None
 
    if clust_data_2 is not None:
        out_ds = gdal.GetDriverByName("GTiff").Create(out_fname + ".UMAP_Final.tif", final.shape[1], final.shape[0], 1, gdal.GDT_Int16)
        out_ds.GetRasterBand(1).WriteArray(final2)
        out_ds.FlushCache()
        out_ds = None


    out_ds = gdal.GetDriverByName("GTiff").Create(out_fname + ".UMAP_Clust_Round.tif", final.shape[1], final.shape[0], 1, gdal.GDT_Int32)
    out_ds.GetRasterBand(1).WriteArray(final.astype(np.int32))
    out_ds.FlushCache()
    out_ds = None

    return projection_data


def knn_graph(w, k, symmetrize=True, metric='cosine'):
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



def emas_final_label_func(final_labels, labels):
    sub_ind_1 = np.where((final_labels == 0) & (labels > 0.0))[0]
    sub_ind_2 = np.where(final_labels == 1)[0]
    sub_ind_3 = np.where(final_labels == 2)[0]
    sub_ind_4 = np.where(final_labels == 3)[0]
    ind_final = np.concatenate((sub_ind_4, sub_ind_2))

    sub_sub_ind__3 = np.random.choice(sub_ind_3.shape[0], size=min(int(sub_ind_3.shape[0]), 5000), replace=False)
    sub_sub_ind__1 = np.random.choice(sub_ind_1.shape[0], size=min(int(sub_ind_1.shape[0]), 10000), replace=False)
    ind_tmp = np.concatenate((ind_final, sub_ind_3[sub_sub_ind__3], sub_ind_1[sub_sub_ind__1]))

    return ind_tmp
 

def chesapeake_final_label_func(final_labels):

    sub_inds = []
    final_inds = None

    for i in range(1, 13):
        sub_inds = np.where(final_labels == i)[0]
        print(len(sub_inds))
        if len(sub_inds) < 1:
            continue
        if len(sub_inds)  > 1000:
            sub_sub_inds  = np.random.choice(len(sub_inds), size=10, replace=False)
            sub_inds = sub_inds[sub_sub_inds]
        if final_inds is None:
            final_inds = np.array(sub_inds)
        else:
            print(final_inds.shape, sub_inds.shape)
            final_inds = np.concatenate((final_inds, sub_inds), axis=0)

    return final_inds



    
def mados_final_label_func(final_labels):

    sub_inds = []
    final_inds = None
    
    for i in range(1, 16):
        sub_inds = np.where(final_labels == i)[0]
        print(len(sub_inds))
        if len(sub_inds) < 1:
            continue
        if len(sub_inds)  > 1000:
            sub_sub_inds  = np.random.choice(len(sub_inds), size=1000, replace=False)
            sub_inds = sub_inds[sub_sub_inds]
        if final_inds is None:
            final_inds = np.array(sub_inds)
        else:
            print(final_inds.shape, sub_inds.shape)
            final_inds = np.concatenate((final_inds, sub_inds), axis=0)

    return final_inds

def hab_severity_final_labels_func(final_labels):
    sub_inds = []
    final_inds = None

    for i in range(0, 7):
        sub_inds = np.where(final_labels == i)[0]
        print(len(sub_inds))
        if len(sub_inds) < 1:
            continue
        if len(sub_inds)  > 1000:
            sub_sub_inds  = np.random.choice(len(sub_inds), size=1000, replace=False)
            sub_inds = sub_inds[sub_sub_inds]
        if final_inds is None:
            final_inds = np.array(sub_inds)
        else:
            print(final_inds.shape, sub_inds.shape)
            final_inds = np.concatenate((final_inds, sub_inds), axis=0)


    return final_inds


def binary_final_labels_func(final_labels):

    sub_inds = []
    final_inds = None

    for i in range(0, 2):
        sub_inds = np.where(final_labels == i)[0]
        print(len(sub_inds))
        if len(sub_inds) < 1:
            continue
        if len(sub_inds)  > 100:
            sub_sub_inds  = np.random.choice(len(sub_inds), size=100, replace=False)
            sub_inds = sub_inds[sub_sub_inds]
        if final_inds is None:
            final_inds = np.array(sub_inds)
        else:
            print(final_inds.shape, sub_inds.shape)
            final_inds = np.concatenate((final_inds, sub_inds), axis=0)

    return final_inds



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

    del data

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

    n_components = yml_conf["analysis"]["projection"]["n_components"]
    spread = yml_conf["analysis"]["projection"]["spread"]
    min_dist = yml_conf["analysis"]["projection"]["min_dist"]
    n_neighbors =  yml_conf["analysis"]["projection"]["n_neighbors"]
    

    run_projection = yml_conf["analysis"]["run_projection"]


    indices = [None]
    if "indices" in yml_conf["data"].keys():
        indices = yml_conf["data"]["indices"]

    embed_func = yml_conf["analysis"]["embed_func"]
    final_labels = yml_conf["data"]["final_labels"]

    final_labels_func = None
    if final_labels is not None:
        if "eMAS" in final_labels[0]:
            final_labels_func = emas_final_label_func 
        elif "MADOS" in final_labels[0]:
            final_labels_func = mados_final_label_func
        elif "hdf5" in final_labels[0]:
            final_labels_func = chesapeake_final_label_func
        elif "DAY" in final_labels[0]:
            final_labels_func = hab_severity_final_labels_func
        else:
            final_labels_func = binary_final_labels_func
 
    knn_graphs = yml_conf["analysis"]["build_knn_graphs"]

    #loop_dict = {"reducer":None, "landmarks":None}
    loop_dicts = []
    int_embed = None
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
        context_labels = None
        if len(final_labels) == len(test_fnames):
            if ".tif" in final_labels[i]:
                context_labels = gdal.Open(final_labels[i]).ReadAsArray()
            elif ".hdf5" in final_labels[i]:
                context_labels = (h5py.File(final_labels[i], 'r'))["label"][:]
        print(data.targets_full.shape, data.init_shape)
        #if data.targets_full.ndim > 2:
        #    context_labels = context_labels[data.targets_full[0,:,1], data.targets_full[0,:,2]].flatten()
        #else:
        #    if data.targets_full.shape[1] < 3:
        #        context_labels = context_labels[data.targets_full[:,0], data.targets_full[:,1]].flatten()
        #    else:
        #        print(data.targets_full[:,0].max(), data.targets_full[:,1].max(), data.targets_full[:,2].max())
        #        context_labels = context_labels[data.targets_full[:,1], data.targets_full[:,2]].flatten()
        output = None
        embed = None
        recon_arr = None
        recon_lab = None
        if model.pretrained_model is not None:
            print("GENERATING EMBEDDING")
            if not os.path.exists(output_fle + ".embeddings.zarr") or not os.path.exists(output_fle + ".embedding_labels.zarr"):
                output, embed, _ = run_inference(data, model, True, out_dir, output_fle + ".clust.data", tiled = tiled, return_embed =  True)
            else:
                recon_arr = zarr.load(output_fle + ".embeddings.zarr")
                recon_lab = zarr.load(output_fle + ".embedding_labels.zarr")
   
            is_pca = bool("pca" in embed_func )
            if run_projection:
                ind = None
                if indices[0] is not None:
                    ind = indices[i]
                loop_dict = prep_projection(embed, output, data.targets_full, context_labels, output_fle, is_pca, ind, recon_arr, recon_lab, data.init_shape, final_labels_func)
                loop_dicts.append(loop_dict)
                if int_embed is None:
                    int_embed = loop_dict["test_data"]
                else:
                    int_embed = np.concatenate((int_embed, loop_dict["test_data"]))
                print("INTERMEDIATE EMBEDDING SHAPE", int_embed.shape) 

    del model

    if run_projection:
        reducer = None
        scaler2 = None
        scaler1 = None
        reducer_fname = os.path.join(out_dir, 'umap_model.joblib')
        #scaler1_fname = os.path.join(out_dir, 'umap_pre_scaler.joblib')
        #scaler2_fname = os.path.join(out_dir, 'umap_post_scaler.joblib') 

        print(int_embed.min(), int_embed.max(), int_embed.mean())

        #if not os.path.exists(scaler1_fname) and not is_pca:
        #    print("FITTING SCALER 1")
        #    scaler1 = MinMaxScaler()
        #    scaler1.fit(int_embed) #loop_dicts[1]["test_data"])
        #    joblib.dump(scaler1, scaler1_fname)
        #elif not is_pca:
        #    scaler1 = joblib.load(scaler1_fname)
 
        #if not is_pca: 
        #    int_embed = scaler1.transform(int_embed)
        print(np.nanmin(int_embed), np.nanmax(int_embed), np.nanmean(int_embed), np.count_nonzero(np.isnan(int_embed)), np.count_nonzero(~np.isnan(int_embed)))
        print(int_embed.min(), int_embed.max(), int_embed.mean())
        if not os.path.exists(reducer_fname) and not is_pca:
            print("TRAINING UMAP")
            reducer = run_proj(int_embed, is_pca, n_neighbors=n_neighbors, min_dist=min_dist, spread = spread, n_components=n_components) #loop_dicts[1]["test_data"]), is_pca)
            joblib.dump(reducer, reducer_fname)
        elif not is_pca:
            reducer = joblib.load(reducer_fname)

        if reducer is not None:
            int_embed = reducer.transform(int_embed)

        #if not os.path.exists(scaler2_fname):
        #    print("FITTING SCALER 2")
        #    scaler2 = MinMaxScaler()
        #    if reducer is not None and scaler1 is not None:
        #        scaler2.fit(int_embed) #loop_dicts[1]["test_data"])))
        #    else:
        #        scaler2.fit(int_embed)
        #    joblib.dump(scaler2, scaler2_fname)
        #else:
        #    scaler2 = joblib.load(scaler2_fname)

        del int_embed 

        final_embed = None
        for i in range(len(test_fnames)):
            inp = loop_dicts[i]["test_data"]
            #if not is_pca:
            #    inp = scaler1.transform(inp)
            print("Generating UMAP for", loop_dicts[i]["out_fname"])
            projection_data = transform_and_save(loop_dicts[i]["recon_coord"], inp, \
                is_pca, loop_dicts[i]["out_fname"], loop_dicts[i]["clust_data"], loop_dicts[i]["clust_data_2"], \
                reducer = reducer, scaler=scaler2)
            if i % 2 > 0:
                if final_embed is None:
                    final_embed = projection_data
                else:
                    final_embed = np.concatenate((final_embed, projection_data))

        del reducer
        del loop_dicts
        print("Building KNN Graph")
        if knn_graphs:	                
            build_knn_graph(final_embed, os.path.join(out_dir, embed_func + ".zarr"))


def loop_main(meta_yml_fpath):
 
 meta_yml_conf = read_yaml(meta_yml_fpath)

 for m in range(len(meta_yml_conf["models"])):
     yml_conf = read_yaml(meta_yml_conf["models"][m])
     main(yml_conf)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    #loop_main(args.yaml)
    main(args.yaml)

    print(getrusage(RUSAGE_SELF))


