"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
import os
import argparse
import cv2
import re
import numpy as np
from osgeo import osr, gdal
from sit_fuse.preprocessing.misc_utils import goes_to_geotiff
from sit_fuse.utils import read_yaml



def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)

    filenames = yml_conf["data"]["filenames"]
    data_reader_kwargs = yml_conf["data"]["reader_kwargs"]
    ref_band = yml_conf["data"]["reference_band"]

     
    for i in range(len(filenames)):

        dat1 = []
        nx = None
        ny = None
        geoTransform = None
        metadata = None
        wkt = None

        for j in range(len(filenames[i])):
            goes_to_geotiff(filenames[i][j])
            filenames[i][j] = filenames[i][j].replace(".nc", ".tif")
            dat = gdal.Open(filenames[i][j])            
            dat1.append(dat.ReadAsArray())

            if j == ref_band:
                nx = dat1[j].shape[1]
                ny = dat1[j].shape[0]
                geoTransform = dat.GetGeoTransform()
                metadata = dat.GetMetadata()
                wkt = dat.GetProjection()

            dat.FlushCache()
            dat = None


        refShp = dat1[ref_band].shape
        print(refShp, dat1[0].shape)
        for j in range(len(dat1)):
            if j == ref_band:
                continue
            elif dat1[j].shape[0] != refShp[0] or dat1[j].shape[1] != refShp[1]:
                dat1[j] = cv2.resize(dat1[j], (refShp[1],refShp[0]), interpolation=cv2.INTER_CUBIC)
        dat = np.array(dat1) 

        out_fname = re.sub("-M6C\d{2}_", "_FULL_", filenames[i][0])
        print(out_fname,  dat.shape)       

        out_ds = gdal.GetDriverByName("GTiff").Create(out_fname, nx, ny, dat.shape[0], gdal.GDT_Float32)
        out_ds.SetGeoTransform(geoTransform)
        out_ds.SetMetadata(metadata)
        out_ds.SetProjection(wkt)
        for c in range(dat.shape[0]):
            out_ds.GetRasterBand(c+1).WriteArray(dat[c,:,:])
        out_ds.FlushCache()
        out_ds = None




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)





