"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import torch
#from sw_approx import sw_approx
#from sliceduot.sliced_uot import unbalanced_sliced_ot, sliced_unbalanced_ot
import zarr
import argparse
import numpy as np
from osgeo import gdal, osr
from sit_fuse.utils import get_read_func, read_yaml
from sit_fuse.models.layers.random_conv_2d import RandomConv2d
import cv2
import os
import csv

from tqdm import tqdm
from tabulate import tabulate
from pprint import pprint

import diplib as dip
import scipy as sp

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import ot
import geopy.distance

from sklearn.cluster import MiniBatchKMeans

def convolutional_sliced_wasserstein(X, Y, conv_net, num_projections=50):
    """
    Computes the Convolutional Sliced Wasserstein distance between two sets of samples.

    Args:
        X: NumPy array of shape (n_samples1, height, width, channels) representing the first set of samples.
        Y: NumPy array of shape (n_samples2, height, width, channels) representing the second set of samples.
        num_projections: Number of random projections to use.
        filter_size: Size of the convolutional filters.
        num_filters: Number of convolutional filters.

    Returns:
        The Convolutional Sliced Wasserstein distance between X and Y.
    """
 
    with torch.no_grad():
        # Define the convolutional layer

        # Apply the convolutional layer to the samples
        X_features = conv_net(X) #.detach().numpy()
        Y_features = conv_net(Y) #.detach().numpy()

        print(X_features.shape, Y_features.shape)

        X_features = torch.squeeze(torch.nn.functional.interpolate(
            torch.unsqueeze(X_features,0),
            size=(min(X_features.shape[1:]), min(X_features.shape[1:])),
            mode="bilinear",
            align_corners=False,
        ))  # Resize to match labels size

        Y_features = torch.squeeze(torch.nn.functional.interpolate(
            torch.unsqueeze(Y_features, 0),
            size=(min(Y_features.shape[1:]), min(Y_features.shape[1:])),
            mode="bilinear",
            align_corners=False,
        ))  # Resize to match labels size

    # Flatten the feature maps
    ##X_features_flat = X_features.reshape(X_features.shape[0], -1).ravel()
    ##Y_features_flat = Y_features.reshape(Y_features.shape[0], -1).ravel()

    print(X_features.shape, Y_features.shape)
    ind1 = np.indices(X_features.shape, dtype=np.int32)
    ind2 =  np.indices(Y_features.shape, dtype=np.int32)
    print(ind1.shape, ind2.shape)

    C1 = torch.from_numpy(np.array(squareform(pdist(ind1.transpose(1,2,0).reshape(-1,2), 'minkowski', p=2))))
    C2 = torch.from_numpy(np.array(squareform(pdist(ind2.transpose(1,2,0).reshape(-1,2), 'minkowski', p=2))))


    #C1 = torch.squeeze(torch.pdist(X_features, 2)) ##.ravel()
    #C2 = torch.squeeze(torch.pdist(Y_features, 2)) ##.ravel()
 
    print(X_features.shape, Y_features.shape, C1.shape, C2.shape)

    M = torch.zeros((C1.shape[0], C2.shape[0]), dtype=torch.float16)

    alpha = 1e-3

    nx = ot.backend.get_backend(C1, C2)
    #print(p.shape, q.shape, X_features_flat.shape, Y_features_flat.shape, X_features.shape, Y_features.shape, X.shape, Y.shape)
    X_features = X_features.ravel()
    Y_features = Y_features.ravel()

    print(X_features.shape, Y_features.shape, C1.shape, C2.shape, M.shape)
    distance = ot.fused_gromov_wasserstein2(M, C1, C2, X_features,  Y_features, loss_fun="kl_loss", alpha=alpha, verbose=True, log=False)


    #C1 /= C1.max()
    #C2 /= C2.max()

    #indices = np.indices(X_features_flat.shape).transpose(1,0).reshape(-1,2).astype(np.int32)
    #C1 = np.array(squareform(pdist(indices.reshape(-1,2), 'minkowski', p=2)))
 
    #indices = np.indices(Y_features_flat.shape).transpose(1,0).reshape(-1,2).astype(np.int32)
    #C2 = np.array(squareform(pdist(indices.reshape(-1,2), 'minkowski', p=2)))


    # Conditional Gradient algorithm
    ##gw0, log0 = ot.gromov.gromov_wasserstein(
    ##    C1, C2, X_features_flat, Y_features_flat, "square_loss", verbose=False, log=True
    ##)
 
    ## Generate random projections
    #projections = np.random.randn(num_projections, X_features_flat.shape[1])
    #projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    # Project the feature maps
    #print(X_features_flat.shape, projections.shape, Y_features_flat.shape)
    #X_projected = X_features_flat @ projections.T
    #Y_projected = Y_features_flat @ projections.T

    # Compute the Sliced Wasserstein distance
    #sw_distances = [ot.wasserstein_1d(np.sort(X_projected[:, i]), np.sort(Y_projected[:, i])) for i in range(num_projections)]
    #csw_distance = np.mean(sw_distances)

    return distance

def load_image(fname):
    dat = gdal.Open(fname)
    imgData = dat.ReadAsArray()
    arr = imgData
    arr_tensor = arr

    if arr_tensor.shape[0] > arr_tensor.shape[1]:
        arr_tensor = arr_tensor.transpose(1,0) 

    return arr_tensor

def calc_swd(data_fnames, i, batch_size = 1000, im_resize = 128):
 
    if im_resize > 0.0:
        arr_tensor = torch.from_numpy(cv2.resize(load_image(data_fnames[i]), (im_resize,im_resize)))
    else:
        arr_tensor = torch.from_numpy(load_image(data_fnames[i]))

    swds = []   
 
   
    a = torch.ones(batch_size)/100
    b = torch.ones(batch_size)/100
    filter_size=3
    num_filters=16
   
    dilation = (2,3)
    stride = (2,3)

    conv_layer = torch.nn.Sequential(RandomConv2d(1, num_filters, filter_size), 
                    RandomConv2d(num_filters, num_filters*2, 5, dilation=dilation, stride=stride), 
                    RandomConv2d(num_filters*2, num_filters*4, 5, dilation=stride, stride=stride),
                    RandomConv2d(num_filters*4, num_filters*4, 5, stride=stride),
                    RandomConv2d(num_filters*4, 1, 1)).cuda()
 
    for j in range(i, len(data_fnames), batch_size):
        batch_tmp = []
        batch_tmp2 = [] 
        for m in range(j+1, min(j+batch_size+1, len(data_fnames))):
            batch_tmp = []
            batch_tmp2 = []
            ##for k in range(m, min(j+batch_size+1, len(data_fnames))):
 
            if im_resize > 0.0: 
                arr_tensor2 = torch.from_numpy(cv2.resize(load_image(data_fnames[m]), (im_resize,im_resize)))
            else:
                arr_tensor2 = torch.from_numpy(load_image(data_fnames[m]))   

                #d2 = max(arr_tensor2.shape[1], arr_tensor.shape[1])
                #d1 = max(arr_tensor2.shape[0], arr_tensor.shape[0])
                #tmp2 = arr_tensor2 #torch.zeros((d1,d2))
                #tmp2[:arr_tensor2.shape[0],:arr_tensor2.shape[1]] =arr_tensor2
                #tmp =torch.zeros((d1,d2))
                #tmp[:arr_tensor.shape[0],:arr_tensor.shape[1]] = arr_tensor
            batch_tmp2.append(arr_tensor2) 
            batch_tmp.append(arr_tensor)
             
            if len(batch_tmp) == 0:
                print(j+1, j+batch_size)
                continue
            batch_tmp = torch.stack(batch_tmp, dim=0)
            batch_tmp2 = torch.stack(batch_tmp2, dim=0)
            #print("HERE", batch_tmp.shape, batch_tmp2.shape)
        
            #get_random_projections(, n_projections, seed=None, backend=None, type_as=None)
            #swd_value, _, _, a_SUOT, b_SUOT, _ = unbalanced_sliced_ot(a, b, batch_tmp.cuda(), batch_tmp2.cuda(), p=2, num_projections=500, rho1=1, rho2=1, niter=10)
            swd_value = convolutional_sliced_wasserstein(batch_tmp.cuda(), batch_tmp2.cuda(), conv_layer)

            #swd_value = sw_approx(batch_tmp.cuda(), batch_tmp2.cuda())
 
            #print("HERE2", batch_tmp2.shape)
            swds.append(swd_value) #.extend(swd_value) #.detach().cpu().numpy())

    return swds
     


def compare_parallel(data_fnames, njobs=20, batch_size=1000, im_resize = 128): 

    results = None
    with tqdm_joblib(tqdm(desc="My calculation", total=(len(data_fnames)))) as progress_bar:
        results = Parallel(n_jobs=njobs)(
            delayed(calc_swd)(data_fnames, i, batch_size, im_resize)
            for i in range(len(data_fnames))
            #for j in range(i + 1, len(data_fnames))
        )
 
    compare = torch.zeros((len(data_fnames),len(data_fnames)))
    index = 0
    print("LOOPING", len(data_fnames), len(results[0]), len(results))
    for i in range(len(data_fnames)):
        print(len(results[i]))
        for j in range(len(results[i])):
            compare[i,i+j] = results[i][j].detach().cpu()
            compare[i+j, i]

    return compare

def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)
    #Assume GeoTiff only for the time being - geolocation info
    out_dir = yml_conf["data"]["out_dir"]
    data_fnames = yml_conf["data"]["filename"]
    data_reader =  yml_conf["data"]["reader_type"]
    data_reader_kwargs = yml_conf["data"]["reader_kwargs"]
    wrt_geotiff = yml_conf["write_geotiff"] 
    label_fname = yml_conf["data"]["label_fname"]
    batch_size = yml_conf["data"]["batch_size"]
    im_resize = yml_conf["data"]["im_resize"]
    njobs = yml_conf["njobs"]
    k = yml_conf["k_clusters"]
 

    conts = []

    compare = compare_parallel(data_fnames, njobs, batch_size, im_resize)

    compare = compare.numpy()
    print("DISTRIBUTION COMPARISIONS")
    print(tabulate(compare))

    kmeans = MiniBatchKMeans(n_clusters=k)
    kmeans.fit(compare)
    labels = kmeans.labels_

    label_dict = {}
    for i in range(len(data_fnames)):
        label_dict[data_fnames[i]] = [data_fnames[i], labels[i]]
 
    np.savez(os.path.join(out_dir, label_fname), **label_dict) 
    csv_file = os.path.join(out_dir, label_fname)+".csv"
    # Write the dictionary to a CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
 
        # Write the header row (keys of the dictionary)
        #writer.writerow(label_dict.keys())
 
        # Write the data rows (values of the dictionary)
        writer.writerows(label_dict.values())
 

def write_geotiff(dat, imgData, fname):

    nx = imgData.shape[1]
    ny = imgData.shape[0]
    geoTransform = dat.GetGeoTransform()
    wkt = dat.GetProjection()
    out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(geoTransform)
    out_ds.SetProjection(wkt)
    out_ds.GetRasterBand(1).WriteArray(imgData)
    out_ds.FlushCache()
    out_ds = None
 
def write_zarr(fname, imgData):

    zarr.save(fname + ".zarr", imgData)
    img = plt.imshow(imgData, vmin=-1, vmax=1)
    plt.savefig(fname + ".png", dpi=400, bbox_inches='tight') 
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)



