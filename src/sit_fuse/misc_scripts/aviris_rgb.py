
from osgeo import gdal, osr
import numpy as np
import os
import sys

fname = "/mnt/data/AVIRIS/Radiances/aviris_20190821_p00r23.tif"

dat = gdal.Open(fname)
x = dat.ReadAsArray()
print(x.shape)

nx = x.shape[2]
ny = x.shape[1]

metadata=dat.GetMetadata()
geoTransform = dat.GetGeoTransform()
wkt = dat.GetProjection()

out_fname = fname + ".rgb.tif"
out_ds = gdal.GetDriverByName("GTiff").Create(out_fname, nx, ny, 3, gdal.GDT_Float32)
out_ds.SetGeoTransform(geoTransform)
out_ds.SetProjection(wkt)

bands = [60,40,20]
out_ds.GetRasterBand(1).WriteArray(x[bands[0],:,:])
out_ds.GetRasterBand(2).WriteArray(x[bands[1],:,:])
out_ds.GetRasterBand(3).WriteArray(x[bands[2],:,:])


out_ds.FlushCache()
out_ds = None


