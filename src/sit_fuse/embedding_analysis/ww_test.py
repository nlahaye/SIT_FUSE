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
    plt.tight_layout()
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
    plt.tight_layout()
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
    model = model.encoder

    del data

    model = model.cuda()
    #model.pretrained_model = model.pretrained_model.cuda()
    ##model.pretrained_model.mlp_head = model.pretrained_model.mlp_head.cuda()
    #if hasattr(model.pretrained_model, "pretrained_model"):
    #    model.pretrained_model.pretrained_model = model.pretrained_model.pretrained_model.cuda()
     
    #for lab1 in model.clust_tree.keys():
    #    if lab1 == "0":
    #        continue
    #    for lab2 in model.lab_full.keys():
    #        if lab2 in model.clust_tree[lab1].keys():
    #            if model.clust_tree[lab1][lab2] is not None:
    #                model.clust_tree[lab1][lab2] = model.clust_tree[lab1][lab2].cuda() 
  
 
    out_dir = yml_conf["output"]["out_dir"]


    watcher = ww.WeightWatcher(model=model, log_level=logging.DEBUG)
    details = watcher.analyze(model=model, plot=True, min_evals=50, max_evals=5000, randomize=True, mp_fit=True, pool=True, savefig=True,\
     layers=[]) #[ww.LAYER_TYPE.DENSE, ww.LAYER_TYPE.STACKED, ww.LAYER_TYPE.CONV1D, ww.LAYER_TYPE.CONV2D, ww.LAYER_TYPE.FLATTENED, ww.LAYER_TYPE.EMBEDDING, ww.LAYER_TYPE.BIDIRECTIONAL, ww.LAYER_TYPE.NORM])



    #details_2 = watcher.describe(model=model, layers=[0,1,2,3]) #,randomize=True, mp_fit=True, pool=True, min_evals=0, max_evals=100)
  
    #watcher.get_details()
    summaries = watcher.get_summary(details)
    watcher.get_ESD()


 
    #watcher.distances(model_1, model_2)
    metrics = ["log_norm","alpha","alpha_weighted","log_alpha_norm",\
        "log_spectral_norm","stable_rank","mp_softrank"]
 
    print(len(details), details.keys())
    colors = get_colors(len(details)) 
    all_names = []
    series_name = "Gaussian DBN 2-Layer" #"Pix-Wise Contrastive CNN" #"JEPA_Local" #"Clay" #"Gaussian DBN 2-Layer"
    first_n_last_ids = [series_name]
    details = {series_name: details}
    for i in range(len(details)):
        all_names.append("TEST" + str(i))    


    #plt.rcParams.update({'font.size': 20})
    plot_all_metric_vs_depth(series_name, all_names, colors, summaries, details, first_n_last_ids)
    plot_all_metric_histograms(series_name, all_names, colors, summaries, details,  first_n_last_ids)
 
    #from pylab import rcParams
    #rcParams['figure.figsize'] = 10,10
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)

    print(getrusage(RUSAGE_SELF))


