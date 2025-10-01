import os
import re
import zarr
import glob
import cv2
import numpy as np
from pprint import pprint
from osgeo import osr, gdal
from subprocess import DEVNULL, run, Popen, PIPE
from scipy.ndimage import uniform_filter
from scipy.ndimage import variance



data_smoke = "/data/nlahaye/SIT_FUSE_Geo/GOES_TEMPO_DATA/TEMPO_NO2_L3_V03_20240726T155132Z_S008.fire.tif"
data_fire = "/data/nlahaye/SIT_FUSE_Geo/GOES_TEMPO_DATA/TEMPO_NO2_L3_V03_20240726T155132Z_S008.smoke.tif"

dat1 = gdal.Open(data_smoke).ReadAsArray()
dat2 = gdal.Open(data_fire).ReadAsArray()


inds = np.where((dat1[0,:,:] <= 0.0) & (dat2[0,:,:] <= 0.0))
dat1 = dat1[1,:,:]
dat1[inds] = -9e+36

print(len(inds), dat1.shape)

dat = gdal.Open(data_smoke)
geoTransform = dat.GetGeoTransform()
wkt = dat.GetProjection()
dat.FlushCache()

out_fname  = "/data/nlahaye/SIT_FUSE_Geo/GOES_TEMPO_DATA/TEMPO_NO2_L3_V03_20240726T155132Z_S008.masked.tif"
nx = dat1.shape[1]
ny = dat1.shape[0]

out_ds = gdal.GetDriverByName("GTiff").Create(out_fname, nx, ny, 1, gdal.GDT_Float32)
out_ds.SetGeoTransform(geoTransform)
out_ds.SetProjection(wkt)
out_ds.GetRasterBand(1).WriteArray(dat1)
out_ds.FlushCache()
out_ds = None
del dat1
dat.FlushCache()
dat = None





