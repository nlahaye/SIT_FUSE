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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)
    #Assume GeoTiff only for the time being - geolocation info
    data_fnames = yml_conf["data"]["filename"]
    data_reader =  yml_conf["data"]["reader_type"]
    data_reader_kwargs = yml_conf["data"]["reader_kwargs"]
    wrt_geotiff = yml_conf["write_geotiff"] 

    for i in range(len(data_fnames)):
        dat = gdal.Open(data_fnames[i])
        imgData = dat.ReadAsArray() 

        imgData[np.where(imgData < 0)] = 0
        imgData[np.where(imgData > 0)] = 1       
        imgData = imgData.astype(np.uint8)

        imgData2 = imgData.copy()
        imgData3 = imgData.copy()

        print(imgData.shape, imgData.dtype)

        
        print(imgData.min(), imgData.max())  
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = imgData.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
  
        # Floodfill from point (0, 0)
        cv2.floodFill(imgData3, mask, (0,0), 255);
        if wrt_geotiff:
            write_geotiff(dat, imgData3, data_fnames[i] + ".ImFill_Init.tif")
        else:
            write_zarr(data_fnames[i] + ".ImFill_Init", imgData3.astype(np.int32))
        cv2.waitKey(0)  

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(imgData3)
        if wrt_geotiff:
            write_geotiff(dat, im_floodfill_inv, data_fnames[i] + ".ImFill_Invert.tif")  
        else:
            write_zarr(data_fnames[i] + ".ImFill_Invert.zarr", im_floodfill_inv.astype(np.int32))
        # Combine the two images to get the foreground.
        im_out = imgData | im_floodfill_inv
        if wrt_geotiff:
            write_geotiff(dat, im_out, data_fnames[i] + ".ImFill_Final.tif")
        else:
            write_zarr(data_fnames[i] + ".ImFill_Final.zarr", im_out.astype(np.int32))
        cv2.waitKey(0) 


        # Find Canny edges
        edged = cv2.Canny(imgData2, 30, 200)
        cv2.waitKey(0)
     
        # Finding Contours
        # Use a copy of the image e.g. edged.copy()
        # since findContours alters the image
        contours, hierarchy = cv2.findContours(imgData2, 
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  
        if wrt_geotiff:
            write_geotiff(dat, imgData2, data_fnames[i] + ".Edged.tif") 
        else:
            write_zarr(data_fnames[i] + ".Edged.zarr", imgData2.astype(np.int32))
        #cv2.imshow('Canny Edges After Contouring', edged)
        #cv2.waitKey(0)
   
        print("Number of Contours found = " + str(len(contours)))
   
        # Draw all contours
        # -1 signifies drawing all contours
 
        for j in range(0,len(contours),100):
            zeros = np.zeros(imgData2.shape)
            cv2.drawContours(zeros, contours[j*10:(j+1)*10], -1, (0,255,0), 3)
            if wrt_geotiff:
                write_geotiff(dat, zeros, data_fnames[i] + ".Contours" + str(j) + ".tif") 
            else:
                write_zarr(data_fnames[i] + ".Contours.zarr", zeros)

        # cv2.imshow('Contours', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


def write_geotiff(dat, imgData, fname):

    nx = imgData.shape[1]
    ny = imgData.shape[0]
    geoTransform = dat.GetGeoTransform()
    wkt = dat.GetProjection()
    out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Byte)
    print(fname)
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



