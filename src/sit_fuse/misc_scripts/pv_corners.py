import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from osgeo import osr, gdal
from skimage.util import view_as_windows
import cv2

fnames = [
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP_S1/large_urban_gambia_clipped_0_s1_s2_ls8.tif.clust.data_78994clusters.zarr.PV_Map.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP_S1/large_urban_gambia_clipped_10_s1_s2_ls8.tif.clust.data_79799clusters.zarr.PV_Map.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP_S1/large_urban_gambia_clipped_11_s1_s2_ls8.tif.clust.data_79799clusters.zarr.PV_Map.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP_S1/large_urban_gambia_clipped_12_s1_s2_ls8.tif.clust.data_79799clusters.zarr.PV_Map.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP_S1/large_urban_gambia_clipped_13_s1_s2_ls8.tif.clust.data_79799clusters.zarr.PV_Map.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP_S1/large_urban_gambia_clipped_14_s1_s2_ls8.tif.clust.data_79799clusters.zarr.PV_Map.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP_S1/large_urban_gambia_clipped_15_s1_s2_ls8.tif.clust.data_79799clusters.zarr.PV_Map.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP_S1/large_urban_gambia_clipped_1_s1_s2_ls8.tif.clust.data_79799clusters.zarr.PV_Map.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP_S1/large_urban_gambia_clipped_2_s1_s2_ls8.tif.clust.data_79799clusters.zarr.PV_Map.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP_S1/large_urban_gambia_clipped_3_s1_s2_ls8.tif.clust.data_79799clusters.zarr.PV_Map.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP_S1/large_urban_gambia_clipped_4_s1_s2_ls8.tif.clust.data_79799clusters.zarr.PV_Map.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP_S1/large_urban_gambia_clipped_5_s1_s2_ls8.tif.clust.data_79799clusters.zarr.PV_Map.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP_S1/large_urban_gambia_clipped_6_s1_s2_ls8.tif.clust.data_79799clusters.zarr.PV_Map.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP_S1/large_urban_gambia_clipped_7_s1_s2_ls8.tif.clust.data_79799clusters.zarr.PV_Map.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP_S1/large_urban_gambia_clipped_8_s1_s2_ls8.tif.clust.data_79799clusters.zarr.PV_Map.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP_S1/large_urban_gambia_clipped_9_s1_s2_ls8.tif.clust.data_79799clusters.zarr.PV_Map.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_ROAD_CLIP_S1/urban_small_road_clipped_s1_s2_ls8.tif.clust.data_78998clusters.zarr.PV_Map.tif",
]

lab_fnames = [
"/data/nlahaye/remoteSensing/PV_Mapping/pv_polygons_10.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/pv_polygons_11.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/pv_polygons_16.tif",
]


def write_geotiff(dat, imgData, fname):

    nx = imgData.shape[1]
    ny = imgData.shape[0]
    geoTransform = dat.GetGeoTransform()
    wkt = dat.GetProjection()
    gcpcount = dat.GetGCPCount()
    gcp = None
    gcpproj = None
    if gcpcount > 0:
        gcp = dat.GetGCPs()
        gcpproj = dat.GetGCPProjection()
    out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Byte)
    print(fname)
    out_ds.SetGeoTransform(geoTransform)
    out_ds.SetProjection(wkt)
    if gcpcount > 0:
        out_ds.SetGCPs(gcp, gcpproj)
    out_ds.GetRasterBand(1).WriteArray(imgData)
    out_ds.FlushCache()
    out_ds = None


data = None
targets = None

for i in range(len(fnames)):
    in_dat = gdal.Open(fnames[i]).ReadAsArray()
    #in_dat = in_dat[8:11,:,:]
    #in_dat = in_dat.transpose(1,2,0)
    print(in_dat.shape)

    in_dat[np.where(in_dat < 1)] = 0
    in_dat = cv2.normalize(in_dat, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
 
    imagegray = in_dat

    #converting the input image to grayscale im age using cvtColor() function
    #imagegray = cv2.cvtColor(in_dat, cv2.COLOR_BGR2GRAY)
     
    #using threshold() function to convert the grayscale image to binary image
    #_, imagethreshold = cv2.threshold(imagegray, 245, 255, cv2.THRESH_BINARY_INV)
    imagethreshold = imagegray
    #finding the contours in the given image using findContours() function
    imagecontours, _ = cv2.findContours(imagethreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(imagecontours))
    #for each of the contours detected, the shape of the contours is approximated using approxPolyDP() function and the contours are drawn in the image using drawContours() function
    zeros = np.zeros(imagegray.shape)
    for count in imagecontours:
        epsilon = 0.01 * cv2.arcLength(count, True)
        approximations = cv2.approxPolyDP(count, epsilon, True)

        print(len(approximations))
        
        if len(approximations) <= 20 and len(approximations) >= 3:
            cv2.drawContours(zeros, [approximations], -1, 1, thickness=cv2.FILLED)
            print("HERE", len(approximations), zeros.max())
    write_geotiff(gdal.Open(fnames[i]), zeros, fnames[i] + ".Polygons.tif")




