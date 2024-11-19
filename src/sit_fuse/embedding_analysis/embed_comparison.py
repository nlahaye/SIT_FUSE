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



def bootstrap_null(graph, number_of_bootstraps=25, n_components=None, umap_n_neighbors=32, acorn=None):
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

    ase_latents = AdjacencySpectralEmbed(n_components=n_components, svd_seed=acorn).fit_transform(graph)

    n, n_components = ase_latents.shape

    distances = np.zeros(((2, number_of_bootstraps, n)))
    distances = np.zeros((number_of_bootstraps, n))

    for i in tqdm(range(number_of_bootstraps)):
        graph_b = rdpg(ase_latents, directed=False)

        bootstrap_latents = OmnibusEmbed(n_components=n_components).fit_transform([graph, graph_b])
        distances[i] = np.linalg.norm(bootstrap_latents[0] - bootstrap_latents[1], axis=1)

    return distances.transpose((1, 0)), n_components


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
            omni_embds = OmnibusEmbed(n_components=2).fit_transform([knn_graphs[embed_function1], knn_graphs[embed_function2]])
            temp_dist = np.linalg.norm(omni_embds[0] - omni_embds[1]) / np.linalg.norm( (omni_embds[0] + omni_embds[1]) / 2 )
            dist_matrix[i,j] = temp_dist
            dist_matrix[j,i] = temp_dist
    return dist_matrix

def run_nomic_analysis(embedding_functions, knn_graphs, out_dir):

    #- now, we can "easily" learn a joint/aligned low-dimensional embedding of the two sets of embeddings
    omni_embds = OmnibusEmbed(n_components=2).fit_transform(list(knn_graphs.values()))
    fig, ax = plt.subplots(1,1)

    #colors = ListedColormap(CMAP_COLORS[0:int(len(embedding_functions) - (-1) + 1)])

    colors = ['red', 'black', 'blue', 'orange', 'green']
    for i, embed_function in enumerate(embedding_functions):
        ax.scatter(omni_embds[i, :, 0], omni_embds[i, :, 1], c=colors[i], label=embed_function)
    #ax.set_cmap(cmap)
    ax.legend(loc='upper left')

    plt.show()
    plt.savefig(os.path.join(out_dir, "Embed_Scatter.png"))
    plt.clf()

    #- A simple "check" to see if the two embedding functions represent document i differently
    #- is to look at the distance || omni[0][i] - omni[1][i] ||
    argsorted=np.argsort(np.linalg.norm(omni_embds[0] - omni_embds[1], axis=1))
    print(argsorted)
 
    #- i.e., dataset[argsorted[0]] has moved the most

    #- A more statistically rigorous way to determine *if* a document has moved
    #- is to use the hypothesis test described in the paper

  
    for i, embed_function in enumerate(embedding_functions):
        for j, embed_function2 in enumerate(embedding_functions):
            null_dist = bootstrap_null(knn_graphs[embedding_functions[i]], number_of_bootstraps=100)[0]
            test_statistics = np.linalg.norm(omni_embds[i] - omni_embds[j], axis=1)
            p_values = []

            for st, test_statistic in enumerate(test_statistics):
                p_value = np.mean(test_statistic < null_dist[st])
                p_values.append(p_value)
    
    
            #- same joint embedding space as above, but this time just plotting nomic-ai/nomic-embed-text-v1.5
            #- and adding color intensity to represent p-value
            fig, ax = plt.subplots(1,1)
            for d in range(omni_embds.shape[1]):
                ax.scatter(omni_embds[i, d, 0], omni_embds[i, d, 1], label=embed_function, color=colors[i], alpha=1-p_values[j])
            plt.show()
            plt.savefig(os.path.join(out_dir, "Embed_Scatter_Null_Hyp_" + embed_function[i] + "_" + embed_function[j] + ".png"))
            plt.clf()
 
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
            plt.savefig(os.path.join(out_dir, "P_Value_Dist.png"))

    fig, ax = plt.subplots(1,1)

    #- Get low-dimensional representations of embedding models.
    #- "Families" of embedding models are close to each other in this space.
    dist_matrix = build_dist_mtx(embedding_functions, knn_graphs)
 

    cmds_embds = ClassicalMDS().fit_transform(dist_matrix)
    for i, cmds in enumerate(cmds_embds):
        ax.scatter(cmds[0], cmds[1], label=embedding_functions[i])
    ax.legend(loc='center left')
    plt.show()
    plt.savefig(os.path.join(out_dir, "Embed_Space_Dist_Mtx.png"))


def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)

    knn_graphs = yml_conf["analysis"]["knn_graphs"]
    embedding_functions = yml_conf["analysis"]["embedding_functions"]

    out_dir = yml_conf["output"]["out_dir"]
    final_labels = yml_conf["data"]["final_labels"]

    for key in knn_graphs.keys():
        knn_graphs[key] = zarr.load(knn_graphs[key])
        print(knn_graphs[key].shape)

 
    run_nomic_analysis(embedding_functions, knn_graphs, out_dir)

    for i in range(len(test_fnames)):
        if isinstance(test_fnames[i], list):
            output_fle = os.path.basename(test_fnames[i][0])
        else:
            output_fle = os.path.basename(test_fnames[i])
        data, output_file  = get_prediction_dataset(yml_conf, test_fnames[i])
        if data.data_full is None:
            print("SKIPPING", test_fnames[i], " No valid samples")
            continue
        context_labels = gdal.Open(final_labels[i]).ReadAsArray()
        if model.clust_tree_ckpt is not None:
            _, output, embed = run_inference(data, model, True, out_dir, output_fle + ".clust.data", tiled = tiled)
            run_tsne(embed, output, context_labels, output_fle) 

				                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)

    print(getrusage(RUSAGE_SELF))


