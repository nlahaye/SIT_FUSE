"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""

import zarr
import argparse
import numpy as np
from osgeo import gdal, osr
from utils import get_read_func, read_yaml
import cv2

from tabulate import tabulate
from pprint import pprint

import diplib as dip

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import ot
import geopy.distance


def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)
    #Assume GeoTiff only for the time being - geolocation info
    contour_fnames = yml_conf["data"]["contour_filename"]
    data_fnames = yml_conf["data"]["filename"]
    data_reader =  yml_conf["data"]["reader_type"]
    data_reader_kwargs = yml_conf["data"]["reader_kwargs"]
    wrt_geotiff = yml_conf["write_geotiff"] 
    mask_fnames = yml_conf["data"]["mask_filename"]


    conts = []
    comparison = np.zeros((len(data_fnames),len(data_fnames)))
    compare = np.zeros((len(data_fnames),len(data_fnames)))
    distances = np.zeros((len(data_fnames),len(data_fnames)))

    new_cs = osr.SpatialReference()
    new_cs.ImportFromEPSG(4326)

    centroids = []
    for i in range(len(data_fnames)):
        dat = gdal.Open(data_fnames[i])
        contData = gdal.Open(contour_fnames[i]).ReadAsArray()
 
        contData[np.where(contData < 0)] = 0
        contData[np.where(contData > 0)] = 1       
        contData = contData.astype(np.uint8)


        contours, hierarchy = cv2.findContours(contData, 
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
   
    
        conts.append(contours[0])

        old_cs = osr.SpatialReference()
        old_cs.ImportFromWkt(dat.GetProjectionRef())
        transform = osr.CoordinateTransformation(old_cs,new_cs)
        xoffset, px_w, rot1, yoffset, px_h, rot2 = dat.GetGeoTransform()
 
        M = cv2.moments(contours[0])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
  
        posX = px_w * cX + rot1 * cY + (px_w * 0.5) + (rot1 * 0.5) + xoffset
        posY = px_h * cX + rot2 * cY + (px_h * 0.5) + (rot2 * 0.5) + yoffset

        latlon = transform.TransformPoint(posX,posY)
        centroids.append(latlon)
            
   

    
    #comparisons = [cv2.CONTOURS_MATCH_I1,cv2.CONTOURS_MATCH_I2,cv2.CONTOURS_MATCH_I3]
    comparisons = [cv2.CONTOURS_MATCH_I3]
 
    pprint(data_fnames)
 
    print("CONTOUR CENTROID DISTANCES")
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            distances[i,j] = geopy.distance.distance(centroids[i], centroids[j]).km
    print(tabulate(distances))

    
    print("CONTOUR COMPARISONS")
    for k in range(len(comparisons)):
        for i in range(len(conts)):
            for j in range(len(conts)):
                comparison[i,j] = cv2.matchShapes(conts[i], conts[j], comparisons[k], 0.0)
        print(tabulate(comparison))


    for i in range(len(data_fnames)):
        dat = gdal.Open(data_fnames[i])
        imgData = dat.ReadAsArray()
        mask = gdal.Open(mask_fnames[i]).ReadAsArray()
        inds = np.where(mask <= 0)
        imgData[inds] = -1
        arr = imgData
        
        for j in range(len(data_fnames)):
            dat2 = gdal.Open(data_fnames[j])
            imgData2 = dat2.ReadAsArray()
            mask2 = gdal.Open(mask_fnames[j]).ReadAsArray()

            inds2 = np.where(mask2 <= 0)
            imgData2[inds2] = -1
            arr2 = imgData2
            d2 = max(arr2.shape[1], arr.shape[1])
            d1 = max(arr2.shape[0], arr.shape[0])
            tmp2 = np.zeros((d1,d2))
            tmp2[:arr2.shape[0],:arr2.shape[1]] = arr2
            tmp = np.zeros((d1,d2))
            tmp[:arr.shape[0],:arr.shape[1]] = arr

            compare[i,j] = ot.sliced_wasserstein_distance(tmp, tmp2, n_projections=1000, seed=1)



    print("DISTRIBUTION COMPARISIONS")
    print(tabulate(compare))



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



