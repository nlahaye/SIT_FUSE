

import pickle
from collections import Counter
import cv2
import numpy as np
import os
from osgeo import gdal
import argparse
from sit_fuse.utils import numpy_to_torch, read_yaml, get_read_func
from sit_fuse.postprocessing.contour_and_fill import write_geotiff
from scipy.spatial import distance_matrix
import scipy.sparse as sp
#from pynndescent import NNDescent
import scipy.sparse
import rasterio
import geopandas
from scipy.spatial.distance import pdist, squareform

from pyresample import area_config, bilinear, geometry, data_reduce, create_area_def, kd_tree
from pyresample.utils.rasterio import get_area_def_from_raster
from rasterstats import zonal_stats

from joblib import dump, load

class PolygonAreaKNNGraph(object):

        def __init__(self):
            self.ul_x = -1
            self.ul_y = -1
            self.lr_x = -1
            self.lr_y = -1

            self.knn_adj = None
            self.knn_values = None

        def set_bb(self, ul_x, ul_y, lr_x, lr_y):
            self.ul_x = ul_x
            self.ul_y = ul_y
            self.lr_x = lr_x
            self.lr_y = lr_y

            bb_y = abs(self.lr_y - self.ul_y) + 1
            bb_x  = abs(self.lr_x - self.ul_x) + 1
            self.knn_adj = np.zeros((bb_y, bb_x))

            #print( ul_x, ul_y, lr_x, lr_y, bb_y, bb_x)

        def build_knn_graph(self, img, symmetrize=True, metric='euclidean'):
     
            self.knn_values = img[self.lr_y:self.ul_y+1,self.ul_x:self.lr_x+1]
            indices = np.indices(self.knn_values.shape).transpose(1,2,0).reshape(-1,2).astype(np.int16)

            k = 30
            index = NNDescent(indices, n_neighbors=k)
            #self.knn_adj_final = index._search_forest.multiply(index._search_forest.T).sqrt()
            ##self.knn_adj_final = scipy.sparse.lil_matrix((indices.shape[0], indices.shape[0]), dtype=np.int16)
            #self.knn_adj_final.rows = index._neighbor_graph[0]
            ##self.knn_adj_final.data = np.ones_like(index._neighbor_graph[0], dtype=np.int16)

            neighbors, distances = index.neighbor_graph
 
            inds4 = np.where(~np.isfinite(distances))
            neighbors[inds4] = 0
            distances[inds4] = 0


            # Construct the adjacency matrix (using a sparse matrix for efficiency)

            num_samples = indices.shape[0]
            self.knn_adj_final = scipy.sparse.csr_matrix((abs(distances.flatten()), (np.repeat(np.arange(num_samples), neighbors.shape[1]), neighbors.flatten())), shape=(num_samples, num_samples))

            #self.knn_adj_final = self.knn_adj_final.multiply(self.knn_adj_final.transpose())
  
            self.values_final = self.knn_values.flatten()


def gen_zonal_histogram_vector(zone_vector_path, value_raster_path, zonal_histogram = None, poly_knns = None):


    default_crs = 'EPSG:4326'

    dataset = rasterio.open(value_raster_path)
    arr = dataset.read(1)
    affine=dataset.transform
 
    zone = geopandas.read_file(zone_vector_path)
    # Set CRS to the default if needed. This does not
    # transform the shapefile to a difference CRS.
    if zone.crs is None:
        zone = zone.set_crs(default_crs)
 
    # Transform to a difference CRS.
    zone = zone.to_crs(dataset.crs)
    stats = zonal_stats(zone, arr, affine=affine)

    #print("HERE", stats)


def gen_zonal_histogram_multiclass(zone_raster_path, value_raster_path, \
    zonal_histogram = None, poly_knns = None, regrid = False, zone_min=0, zone_max=10):

    for i in range(zone_min+1, zone_max+2):
        zonal_histograms, poly_knns = gen_zonal_histogram(zone_raster_path, value_raster_path, zonal_histogram, poly_knns, i, regrid)
    
    return zonal_histograms, poly_knns


def gen_zonal_histogram(zone_raster_path, value_raster_path, zonal_histogram = None, poly_knns = None, zone_ind = 1, regrid = False):
    """
    Calculates the zonal histogram of a value raster, 
    based on the zones defined in a zone raster.

    Args:
        zone_raster_path (str): Path to the zone raster file.
        value_raster_path (str): Path to the value raster file.

    Returns:
        dict: A dictionary where keys are the unique zone values 
              and values are the histograms of the corresponding zones 
              in the value raster.
    """

    #print(zone_raster_path)

    if regrid:
        zone_array_0 = regrid_map(zone_raster_path, value_raster_path)
    else:
        zone_array_0 = gdal.Open(zone_raster_path).ReadAsArray()
 
    #write_geotiff(gdal.Open(value_raster_path), zone_array_0 , "regridded_polygons.tif")

    #print("HERE", np.unique(zone_array_0), zone_ind-1)
    zone_array_1 = zone_array_0
    zone_array_1[np.where(zone_array_1 == (zone_ind-1))] = 255
    zone_array_1[np.where(zone_array_1 != 255)] = 0
    zone_array_1 = zone_array_1.astype(np.uint8)


    value_array = gdal.Open(value_raster_path).ReadAsArray()
    #print(value_array.shape, np.unique(zone_array_1), zone_ind-1)
    zeros = np.zeros((value_array.shape[0],value_array.shape[1]))
    inds = [1,0]
 
    zone_array_2 = cv2.resize(zone_array_0, (value_array.shape[1],value_array.shape[0]), interpolation=cv2.INTER_NEAREST)
    zone_array = cv2.resize(zone_array_1, (value_array.shape[1],value_array.shape[0]), interpolation=cv2.INTER_NEAREST)
    zone_array[np.where(zone_array > 0)] = zone_ind

    unique_zones = np.unique(zone_array)
    #print(zone_raster_path, value_raster_path)
    if poly_knns is None:
            poly_knns = []
    if zonal_histogram is None:
            zonal_histograms = {}
    else:
        zonal_histograms = zonal_histogram 

    zone = zone_ind - 1
    #print("UNIQUE ZONES", unique_zones, zone_ind)
    contours, hierarchy = cv2.findContours(zone_array_1,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #if len(contours) < 1:
        
    #    continue

    #print(len(contours))    

    mask = zone_array #== zone
    masked_values = np.zeros(value_array.shape)-1
    inds = np.where(mask > 0)
    #print(masked_values.shape, value_array.shape, mask.shape)
    masked_values[inds] = value_array[inds]

    tuple_arr = tuple(map(tuple, masked_values))
    #print(masked_values.shape, masked_values.dtype, mask.dtype, value_array.dtype)
    c = Counter(tuple_arr)
    if zone not in zonal_histograms.keys():
        zonal_histograms[zone] = {} 
    #for b in range(len(bins)):
    for k1, v in c.items():
        for k in k1:
            k = round(float(k), 3)
            if k not in zonal_histograms[zone].keys():
                zonal_histograms[zone][k] = int(v)
            else:
                zonal_histograms[zone][k] = zonal_histograms[zone][k] + int(v)


    return zonal_histograms, poly_knns


def regrid_map(label_gtiff, clust_gtiff):
   
    labels = gdal.Open(label_gtiff).ReadAsArray()
    area_def = get_area_def_from_raster(label_gtiff)
    final_area_def = get_area_def_from_raster(clust_gtiff)
 
    #print(area_def, final_area_def) #, np.unique(labels))
    #print("PRE GRID", np.unique(labels))
    reprojected_labels = np.squeeze(kd_tree.resample_nearest(area_def, labels, final_area_def, radius_of_influence=5000, fill_value = -2))
    #print(np.unique(reprojected_labels))
    reprojected_labels = reprojected_labels.astype(np.int32)
    return reprojected_labels

def run_zonal_hist_outside(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)

    run_zonal_hist(yml_conf)

def run_zonal_hist(yml_conf):
    #Run 
    clust_gtiffs = yml_conf["data"]["clust_gtiffs"]
    label_gtiffs = yml_conf["data"]["label_gtiffs"]
    out_dir = yml_conf["output"]["out_dir"]
    out_tag = yml_conf["output"]["class_name"]
    regrid = yml_conf["data"]["regrid"]
    multiclass = yml_conf["data"]["multiclass"]
    zone_min = yml_conf["data"]["zone_min"]
    zone_max = yml_conf["data"]["zone_max"]

    if isinstance(out_tag, list):
        out_tag = out_tag[0]


    zonal_histogram = None
    poly_knns = []
    #print(len(clust_gtiffs), len(label_gtiffs[0]))

    #Assuming number of desired classes == number of sublists in label_gtiffs
    for j in range(len(label_gtiffs)):
        for i in range(len(label_gtiffs[0])):
            
            if ".shp" not in label_gtiffs[j][i]:
             
                if not multiclass:
                    zonal_histogram, poly_knns = gen_zonal_histogram(label_gtiffs[j][i], clust_gtiffs[i], \
                        zonal_histogram, poly_knns, zone_ind=j+1, regrid=regrid)
                else:
                    zonal_histogram, poly_knns = gen_zonal_histogram_multiclass(label_gtiffs[j][i], clust_gtiffs[i], \
                        zonal_histogram, poly_knns, regrid=regrid, zone_min=zone_min, zone_max=zone_max)
            else:
                zonal_histogram, poly_knns = gen_zonal_histogram_vector(label_gtiffs[j][i], clust_gtiffs[i], zonal_histogram, poly_knns)

        #print(len(zonal_histogram.keys()), len(zonal_histogram[list(zonal_histogram.keys())[0]].keys()))

        for zone in zonal_histogram.keys():
            values = zonal_histogram[zone].keys()
            del_vals = []
            for value in values:
                if zonal_histogram[zone][value] <  yml_conf["data"]["min_thresh"]:
                    del_vals.append(value)
            for value in del_vals:
                del  zonal_histogram[zone][value]

        #print(zonal_histogram)

        #print(zonal_histogram.keys(), len(zonal_histogram[list(zonal_histogram.keys())[0]].keys()))

    print("SAVING", os.path.join(out_dir, out_tag + "_hist_dict.pkl"))

    with open(os.path.join(out_dir, out_tag + "_hist_dict.pkl"), 'wb') as handle:
            dump(zonal_histogram, handle, True, pickle.HIGHEST_PROTOCOL)


    print("SAVING", os.path.join(out_dir, out_tag + "_base_cluster_polygon_knn_graphs.pkl"))
    with open(os.path.join(out_dir, out_tag + "_base_cluster_polygon_knn_graphs.pkl"), 'wb') as handle:
            dump(poly_knns, handle, True, pickle.HIGHEST_PROTOCOL)

        #numpy.save(os.path.join(out_dir, out_tag[-1] + "_base_cluster_polygon_knn_graphs.npz"), poly_knns, allow_pickle=True)
    return os.path.join(out_dir, out_tag + "_hist_dict.pkl"), os.path.join(out_dir, out_tag + "_base_cluster_polygon_knn_graphs.pkl")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    run_zonal_hist_outside(args.yaml)






