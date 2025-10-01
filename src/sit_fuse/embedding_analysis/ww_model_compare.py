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

from random import randint
import logging

from sklearn import metrics, manifold
from sklearn.preprocessing import MinMaxScaler
from pprint import pprint

#- Scientific stack
import scipy
from scipy.spatial.distance import pdist, squareform

#- Graph-stats 
from graspologic.embed import OmnibusEmbed, ClassicalMDS, AdjacencySpectralEmbed
from graspologic.simulations import rdpg

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
import weightwatcher as ww


sys.setrecursionlimit(4500)

def plot_metrics_histogram(metric, xlabel, title, series_name, \
    all_names, all_details, colors, log=False, valid_ids = []):
                                
    transparency = 1.0
    
    if len(valid_ids) == 0:
        valid_ids = range(0,len(all_details)-1)
        idname='all'
    else:
        idname='fnl'
         
    ind = 0
    #for im, details in enumerate(all_details):
    for key in all_details.keys():
        print("HERE PLOT METRICS", valid_ids)
        if key in valid_ids:
            if metric not in all_details[key]:
                continue
            vals = all_details[key][metric].to_numpy()
            print("HERE VALS", metric, vals)
            if log:
                vals = np.log10(np.array(vals+0.000001, dtype=np.float))


            plt.hist(vals, bins=100, label=key, alpha=transparency, color=colors[ind], density=True)
            transparency -= 0.15
            ind = ind + 1

    fulltitle = "Histogram: "+title+" "+xlabel
  
    #plt.legend()
    plt.title(title)
    plt.title(fulltitle)
    plt.xlabel(xlabel)
    
    figname = "img/{}_{}_{}_hist.png".format(series_name, idname, metric)
    print("saving {}".format(figname))
    plt.savefig(figname)
    plt.show()

def plot_metrics_depth(metric, ylabel, title, series_name, \
    all_names, all_details, colors, log=False, valid_ids = []):
    
    transparency = 1.0
      
    if len(valid_ids) == 0:
        valid_ids = range(len(all_details)-1)
        idname='all'
    else:
        idname='fnl'
        
         
    #for im, details in enumerate(all_details):
    ind = 0
    for key in all_details.keys():
        if key in valid_ids:
            #details = all_details[im]
            name = key #all_names[im]
            #x = details["layer_id"].to_numpy()
            print(metric, all_details)
            if metric not in all_details[key]:
                continue
            y = all_details[key][metric].to_numpy()
            x = [i for i in range(len(y))]
            if log:
                y = np.log10(np.array(y+0.000001, dtype=np.float))

            plt.scatter(x,y, label=name, color=colors[ind])
            ind = ind + 1

    #plt.legend()
    plt.title("Depth vs "+title+" "+ylabel)
    plt.xlabel("Layer id")
    plt.ylabel(ylabel)
    
    figname = "img/{}_{}_{}_depth.png".format(series_name, idname, metric)
    print("saving {}".format(figname))
    plt.savefig(figname)
    plt.show()


def plot_all_metric_histograms(\
    series_name, all_names, colors, all_summaries, all_details,  first_n_last_ids):
    
    metric = "log_norm"
    xlabel = r"Log Frobenius Norm $\log\Vert W_{F}\Vert$"
    title = series_name
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors)                        
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors, valid_ids = first_n_last_ids)

    metric = "alpha"
    xlabel = r"Alpha $\alpha$"
    title = series_name
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors)                           
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors, valid_ids = first_n_last_ids)


    metric = "alpha_weighted"
    xlabel = r"Weighted Alpha $\hat{\alpha}$"
    title = series_name
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors)                         
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors, valid_ids = first_n_last_ids)

    
    metric = "stable_rank"
    xlabel = r"Stable Rank $\mathcal{R}_{s}$"
    title = series_name
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors)   
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors, valid_ids = first_n_last_ids)

    metric = "log_spectral_norm"
    xlabel = r"Log Spectral Norm $\log\Vert\mathbf{W}\Vert_{\infty}$"
    title = series_name
    plot_metrics_histogram(metric, xlabel, title,  series_name, \
            all_names, all_details, colors)                          
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors, valid_ids = first_n_last_ids)


    metric = "mp_softrank"
    xlabel = r"Log MP Soft Rank $\mathcal{R}_{mp}$"
    title = series_name
    plot_metrics_histogram(metric, xlabel, title,  series_name, \
            all_names, all_details, colors)                         
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors, valid_ids = first_n_last_ids)

    
    metric = "log_alpha_norm"
    xlabel = r"Log $\alpha$-Norm $\log\Vert\mathbf{X}\Vert^{\alpha}_{\alpha}$"
    title = series_name
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors)                          
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors, \
            valid_ids = first_n_last_ids)
    

def plot_all_metric_vs_depth(\
    series_name, all_names, colors, all_summaries, all_details, first_n_last_ids):

    metric = "log_norm"
    xlabel = r"Log Frobenius Norm $\langle\log\;\Vert\mathbf{W}\Vert\rangle_{F}$"
    title = series_name
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = [])
    plot_metrics_depth(metric, xlabel, title,series_name, \
            all_names, all_details, colors, log=False, valid_ids = first_n_last_ids)

    metric = "alpha"
    xlabel = r"Alpha $\alpha$"
    title = series_name
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = [])
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = first_n_last_ids)

    metric = "alpha_weighted"
    xlabel = r"Weighted Alpha $\hat{\alpha}$"
    title = series_name
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = [])
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = first_n_last_ids)



    metric = "stable_rank"
    xlabel = r"Stable Rank $\log\;\mathcal{R}_{s}$"
    title = series_name
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = [])
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = first_n_last_ids)

    metric = "log_spectral_norm"
    xlabel = r"Log Spectral Norm $\log\;\Vert\mathbf{W}\Vert_{\infty}$"
    title = series_name
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = [])
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = first_n_last_ids)



    metric = "mp_softrank"
    xlabel = r"Log MP Soft Rank $\log\;\mathcal{R}_{mp}$"
    title = series_name
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = [])
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = first_n_last_ids)

    metric = "log_alpha_norm"
    xlabel = r"Log $\alpha$-Norm $\log\;\Vert\mathbf{X}\Vert^{\alpha}_{\alpha}$"
    title = series_name
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = [])
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = first_n_last_ids)
 

def get_colors(n):
    color = []

    for i in range(n):
        color.append('#%06X' % randint(0, 0xFFFFFF))

    return color

def main():

    yml_fpaths = ["/home/nlahaye/SIT_FUSE/src/sit_fuse/config/model/emas_fire_dbn_multi_layer_pl_NEW_rerun_TSNE.yaml",\
        "/home/nlahaye/SIT_FUSE/src/sit_fuse/config/model/emas_fire_byol_TSNE.yaml",\
        "/home/nlahaye/SIT_FUSE/src/sit_fuse/config/model/emas_fire_clay_small_TSNE.yaml",\
        "/home/nlahaye/SIT_FUSE/src/sit_fuse/config/model/emas_fire_ijepa_v5_TSNE.yaml"]

    names = ["DBN", "Pix_Cont_CNN", "Clay", "JEPA"]

    yml_fpath = yml_fpaths[0]
    yml_conf = read_yaml(yml_fpath)

    test_fnames = yml_conf["data"]["files_test"]
    train_fnames = yml_conf["data"]["files_train"]
  
    data = None
    cntr = 0
    while data is None or data.data_full is None:
        data, _  = get_prediction_dataset(yml_conf, train_fnames[cntr])
        cntr = cntr + 1

    print(data.data_full.shape)


    out_dir = yml_conf["output"]["out_dir"]

    for i in range(len(yml_fpaths)):
        yml_fp = yml_fpaths[i]
        yml_conf1 = read_yaml(yml_fp)
        model = get_model(yml_conf1, data.data_full.shape[1]).encoder
        model = model.cuda()
        for j in range(len(yml_fpaths)):           
            yml_fp2 = yml_fpaths[j]
            yml_conf2 = read_yaml(yml_fp2)
            model2 = get_model(yml_conf2, data.data_full.shape[1]).encoder
            model2 = model2.cuda()

            watcher = ww.WeightWatcher(log_level=logging.WARNING)
            print(names[i], names[j])
            d1, d2, _ = watcher.distances(model, model2, method=ww.constants.CKA)
            print(d1, d2)
            model2 = model2.cpu()
            del model2
        model = model.cpu()
        del model
 

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    #args = parser.parse_args()
    main() #args.yaml)
  
    print(getrusage(RUSAGE_SELF))


