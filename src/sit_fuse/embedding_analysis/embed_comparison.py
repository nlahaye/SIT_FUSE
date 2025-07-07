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

import dask

import copy

from sklearn import metrics, manifold
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

from sklearn.preprocessing import RobustScaler

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

import zarr
import dask
import dask.array as da

from osgeo import gdal, osr

from torchinfo import summary

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sit_fuse.viz.CMAP import CMAP, CMAP_COLORS

from sit_fuse.inference.generate_output import run_inference, get_model

sys.setrecursionlimit(4500)



def bootstrap_null(graph, number_of_bootstraps=25, n_components=None, umap_n_neighbors=32, acorn=None, fname_uid=""):
    '''
    Constructs a bootstrap null distribution for the difference of latent positions of the nodes in the passed graph
    :param graph: [N, N] binary symmetric hollow matrix to model
    :param number_of_bootstraps: the number of bootstrap replications
    :param n_components: the number of components to use in initial ASE. selected automatically if None.
    :param umap_n_neighbors: the number of neighbors to use in umap
    :param acorn: rng seed to control for randomness in umap and ase
    :return: [2, N, number_of_bootstraps], n_components.
             The 0 column of the matrix is the ASE estimates, and the 1 column is the UMAP estimates.
             n_components is the number of selected components
    '''
    if acorn is not None:
        np.random.seed(acorn)

    ase_latents = AdjacencySpectralEmbed(n_components=n_components, svd_seed=acorn, n_elbows=5).fit_transform(graph)

    n, n_components = ase_latents.shape

    distances = np.zeros(((2, number_of_bootstraps, n)))
    distances = np.zeros((number_of_bootstraps, n))

    for i in tqdm(range(number_of_bootstraps)):
        print("BOOTSTRAP", i)
        graph_b = rdpg(ase_latents, directed=False)

        bootstrap_latents = OmnibusEmbed(n_components=n_components).fit_transform([graph, graph_b])
        #scaler = RobustScaler(quantile_range=(10.0,90.0))
        #data = copy.deepcopy(bootstrap_latents[0])
        #for k in range(1, len(bootstrap_latents)):
        #    data = np.concatenate((data, bootstrap_latents[k]))
        #scaler.fit(data)
        #del data 
        #for k in range(bootstrap_latents.shape[0]):
        #    bootstrap_latents[k] = scaler.transform(bootstrap_latents[k])
        #distances[i] = np.linalg.norm(bootstrap_latents[0] - bootstrap_latents[1], axis=1)
        print("HERE", n, n_components, distances[i], bootstrap_latents.shape, distances[i].shape)
        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #ct = ax.scatter(bootstrap_latents[0,:,0], bootstrap_latents[0,:,1], bootstrap_latents[0,:,2], marker='s', c="blue",
        #    label="INIT", s=100)
        #ax.scatter(bootstrap_latents[1,:,0], bootstrap_latents[1,:,1], bootstrap_latents[1,:,2], marker='.', c="red",
        #    label="AdjSpectral", s=100)
        #plt.legend(fontsize=20)
     
        #for j in range(200):
        #    idx = np.random.randint(n, size=1)
        #    ax.plot([bootstrap_latents[0,idx,0], bootstrap_latents[0,idx,1], bootstrap_latents[0,idx,2]], [bootstrap_latents[1,idx,0], bootstrap_latents[1,idx,1], bootstrap_latents[1,idx,2]])
        #plt.tight_layout()
        #plt.savefig("Bootstrap_Mapping_" + str(i) + fname_uid)
        #plt.clf()
    return distances.transpose((1, 0)) , n_components


def get_cdf(pvalues, num=26):
    linspace = np.linspace(0, 1, num=num)

    cdf = np.zeros(num)

    for i, ii in enumerate(linspace):
        cdf[i] = np.mean(pvalues <= ii)

    return cdf



def build_dist_mtx(embedding_functions, knn_graphs):
    dist_matrix = np.zeros((len(embedding_functions), len(embedding_functions)))
    for i, embed_function1 in enumerate(embedding_functions):
        for j, embed_function2 in enumerate(embedding_functions[i+1:], i+1):
            omni_embds = OmnibusEmbed(n_components=3).fit_transform([knn_graphs[embed_function1], knn_graphs[embed_function2]])
            temp_dist = np.linalg.norm(omni_embds[0] - omni_embds[1]) / np.linalg.norm( (omni_embds[0] + omni_embds[1])) # / 2 )
            dist_matrix[i,j] = temp_dist
            dist_matrix[j,i] = temp_dist
    return dist_matrix

def run_nomic_analysis(embedding_functions, knn_graphs, out_dir):

    #- now, we can "easily" learn a joint/aligned low-dimensional embedding of the two sets of embeddings
    omni_embds = OmnibusEmbed(n_components=3).fit_transform(list(knn_graphs.values())) #list()

    #scaler = RobustScaler(quantile_range=(10.0,90.0))
    #data = copy.deepcopy(omni_embds[0])
    print(omni_embds[0].shape, "OMNI SHAPE")
    #for i in range(1, len(embedding_functions)):
    #    data = np.concatenate((data, omni_embds[i]))
    #scaler.fit(data)
    #for i, embed_function in enumerate(embedding_functions):
    #   scaler.partial_fit(omni_embds[i])
    #for i, embed_function in enumerate(embedding_functions):
    #   omni_embds[i] = scaler.transform(omni_embds[i])
    #scaler = None
    #del data

    out_dct = {"embed_funcs": embedding_functions, "omni_embeds": omni_embds}
    np.save(out_dir + "/omni.viz_dict.npy", out_dct)   
 
    #colors = ListedColormap(CMAP_COLORS[0:int(len(embedding_functions) - (-1) + 1)])


    colors = ['red', 'black', 'blue', 'orange', 'green', "magenta", "cyan", "olive", "purple"]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i, embed_function in enumerate(embedding_functions): # < 5 < 40 < 5
        print(omni_embds.shape, colors[i], i, embed_function)
        #inds = np.where((omni_embds[i, :, 0] > 1) & (omni_embds[i, :, 1] > 20) & (omni_embds[i, :, 2] > 0))
        #ax.scatter(omni_embds[i, inds, 0], omni_embds[i, inds, 1], omni_embds[i, inds, 2], c=colors[i], label=embed_function)
        ax.scatter(omni_embds[i,:,0], omni_embds[i, :,1], omni_embds[i, :,2], c=colors[i], label=embed_function)
        #ax.set_cmap(cmap)
        # Shrink current axis by 20%
        #plt.axis('off')
    plt.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


    plt.show()
    print("Plotting Scatter Embed")
    plt.savefig(os.path.join(out_dir, "Embed_Scatter_multi" + ".png"))
    plt.clf()
    plt.close(fig)    

    #- A simple "check" to see if the two embedding functions represent document i differently
    #- is to look at the distance || omni[0][i] - omni[1][i] ||
    argsorted=np.argsort(np.linalg.norm(omni_embds[0] - omni_embds[1], axis=1))
    print(argsorted)
 
    #- i.e., dataset[argsorted[0]] has moved the most

    #- A more statistically rigorous way to determine *if* a document has moved
    #- is to use the hypothesis test described in the paper

    for i, embed_function in enumerate(embedding_functions):
        for j in range(i+1,len(embedding_functions)):
            print("HERE", i, j)
            embed_function2 = embedding_functions[j]
            null_dist, ase_n_components  = bootstrap_null(knn_graphs[embedding_functions[i]], n_components=None, number_of_bootstraps=20, fname_uid="_" + embedding_functions[i] + "_" + embedding_functions[j])
            test_statistics = np.linalg.norm(omni_embds[i] - omni_embds[j], axis=1)
            p_values = []

            print(test_statistics.shape, null_dist.shape, omni_embds[i].shape)

            for st, test_statistic in enumerate(test_statistics):
                print("DIST", np.mean(test_statistic), np.mean(null_dist[st]), test_statistic.shape, null_dist.shape)
                p_value = np.mean(test_statistic < null_dist[st])
                p_values.append(p_value)
    
    
            #- same joint embedding space as above, but this time just plotting nomic-ai/nomic-embed-text-v1.5
            #- and adding color intensity to represent p-value
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for d in range(omni_embds.shape[1]):
                ax.scatter(omni_embds[i, d, 0], omni_embds[i, d, 1], omni_embds[i, d, 2], label=embed_function, color=colors[i], alpha=1-p_values[d])
            plt.show()
            print("Plotting Null Hyp Scatter")
            plt.savefig(os.path.join(out_dir, "Embed_Scatter_Null_Hyp_" + embed_function + "_" + embed_function2 + ".png"))
            plt.clf()
            plt.close(fig) 
            #- Notice that the ranking of the p-values is related to but does not equal ranking of || omni[0][i] - omni[1][i] ||
            print(np.argsort(p_values)[::-1])

            #- Looking at distribution of p-values relative to the uniform dist
            #- there doesnt seem to be a systematic difference

            linspace=np.linspace(0, 1, num=25+1)
            cdf  = get_cdf(p_values, num=25+1)

            fig, ax = plt.subplots(1,1)

            ax.plot(linspace, cdf, label='observed')
            ax.plot(linspace, linspace, label='null / uniform (y=x)')
            ax.legend()
            plt.show()
            print("Plotting P-Value")
            plt.savefig(os.path.join(out_dir, "P_Value_Dist_" + embed_function + "_" + embed_function2 + ".png"))
            plt.clf()
            plt.close(fig)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #- Get low-dimensional representations of embedding models.
    #- "Families" of embedding models are close to each other in this space.
    dist_matrix = build_dist_mtx(embedding_functions, knn_graphs)
 
    print(dist_matrix.shape)
 
    colors = ['red', 'black', 'blue', 'orange', 'green', "magenta", "cyan", "olive", "purple"]
    cmds_embds = ClassicalMDS(n_components=3).fit_transform(dist_matrix)
    for i, cmds in enumerate(cmds_embds):
        ax.scatter(cmds[0], cmds[1], cmds[2], label=embedding_functions[i], c=colors[i])

    #ax.set_yscale('log')
    #ax.set_xscale('log')
    #ax.set_zscale('log')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
 

    plt.show()
    plt.savefig(os.path.join(out_dir, "Embed_Space_Dist_Mtx_" + embedding_functions[0] + ".png"))
    plt.close(fig)

def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)

    knn_graphs = yml_conf["analysis"]["knn_graphs"]
    embedding_functions = yml_conf["analysis"]["embedding_functions"]

    out_dir = yml_conf["output"]["out_dir"]
    final_labels = yml_conf["data"]["final_labels"]

    for key in knn_graphs.keys():
        knn_graphs[key] = zarr.load(knn_graphs[key]).astype(np.float32)
        print(knn_graphs[key].min(), knn_graphs[key].max(), np.unique(knn_graphs[key]))    
        knn_graphs[key] = knn_graphs[key].astype(np.int8)

        #print(knn_graphs[key].shape)
        #knn_graphs[key] = np.array(knn_graphs[key])
        ##middle = int(knn_graphs[key].shape[0]/2)
        ##ind1 = middle - 5000
        ##ind2 = middle + 5000
        #already randomized
        ##knn_graphs[key] = knn_graphs[key][ind1:ind2,ind1:ind2]
        print(knn_graphs[key].shape)

 
    run_nomic_analysis(embedding_functions, knn_graphs, out_dir)

				                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)

    print(getrusage(RUSAGE_SELF))


