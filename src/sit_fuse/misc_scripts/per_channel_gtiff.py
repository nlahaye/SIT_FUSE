import cv2
import numpy as np
from osgeo import gdal
import os
from netCDF4 import Dataset
   
data = [
"/data/nlahaye/remoteSensing/eMAS_new/eMASL1B_19916_14_20190816_2313_2327_V03.tif",
"/data/nlahaye/remoteSensing/eMAS_new/eMASL1B_19910_16_20190806_1941_1956_V03.tif"]
#"/data/nlahaye/remoteSensing/AVIRIS_GTIFF/aviris_20190821_p00r27.tif"]
#"/data/nlahaye/remoteSensing/EMIT_WQ/EMIT_L2A_RFL_001_20230419T173941_2310912_005.tif",
#"/data/nlahaye/remoteSensing/EMIT_WQ/EMIT_L2A_RFL_001_20230729T223029_2321015_001.tif",
#"/data/nlahaye/remoteSensing/EMIT_WQ/EMIT_L2A_RFL_001_20240621T160541_2417311_026.tif",
#"/data/nlahaye/remoteSensing/EMIT_WQ/EMIT_L2A_RFL_001_20240621T160553_2417311_027.tif",
#"/data/nlahaye/remoteSensing/EMIT_WQ/EMIT_L2A_RFL_001_20240626T152144_2417810_030.tif",
#"/data/nlahaye/remoteSensing/EMIT_WQ/EMIT_L2A_RFL_001_20240626T152155_2417810_031.tif"]




for tiff in data:
 
    fbase, _ = os.path.splitext(tiff)
 
    dat = gdal.Open(tiff)
    x = dat.ReadAsArray()
    metadata = dat.GetMetadata()
    geoTransform = dat.GetGeoTransform()
    wkt = dat.GetProjection()

    out = np.zeros((x.shape[1], x.shape[2])) 
    print(x.shape)
    #out_ds = gdal.GetDriverByName("GTiff").Create(fbase + ".RGB_no_scale.tif", x.shape[2], x.shape[1], 1, gdal.GDT_Float32)
    #out_ds.SetGeoTransform(geoTransform)
    #out_ds.SetProjection(wkt)

    #out[np.where(np.max(x, axis = 0) <= 0.0)] = 1 
    for i in range(x.shape[0]):
        out = x[i,:,:] #np.zeros((x.shape[1], x.shape[2]))
        print(x.shape, i, out.min(), out.mean(), out.max(), out.std())
        out_ds = gdal.GetDriverByName("GTiff").Create(fbase + ".CHAN_" + str(i) + "_no_scale.tif", x.shape[2], x.shape[1], 1, gdal.GDT_Float32)
        out_ds.SetGeoTransform(geoTransform)
        out_ds.SetProjection(wkt)
 

        #tmp = x[i]
        #out[np.where(tmp < 0.0)] = 1

        out_ds.GetRasterBand(1).WriteArray(out)
        out_ds.FlushCache()
        out_ds = None



    #out_ds.GetRasterBand(1).WriteArray(x[30,:,:])
    #out_ds.GetRasterBand(2).WriteArray(x[20,:,:])
    #out_ds.GetRasterBand(3).WriteArray(x[10,:,:])
 
    #out_ds.FlushCache()
    #out_ds = None

