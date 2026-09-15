import cv2

from osgeo import gdal
import os
from netCDF4 import Dataset
 
#tiff = "/data/nlahaye/output/Learnergy/EMIT_WQ_TEST/EMIT_L2A_RFL_001_20240626T152155_2417810_031.nc.clust.data_79498clusters.tif"
#nc_data = "/data/nlahaye/remoteSensing/EMIT_WQ/EMIT_L2A_RFL_001_20240626T152155_2417810_031.nc"



tiffs = [
"/data/nlahaye/remoteSensing/RAD_L1_V04/TEMPO_RAD_L1_V04_20260717T180102Z_S019G08.tif",
]

for i in range(len(tiffs)):
    tiff = tiffs[i]

    fbase, _ = os.path.splitext(tiff)

    dat = gdal.Open(tiff)
 
    #x = dat.ReadAsArray()[[1823,1043,900],:,:]
    x = dat.ReadAsArray()[[606, 464, 313],:,:] #UV Bands
    print(x.shape)

    geoTransform = dat.GetGeoTransform()
    wkt = dat.GetProjection()

    print(x.min(), x.max(), x.mean(), x.std())
    print(wkt, geoTransform)
    
    out_ds = gdal.GetDriverByName("GTiff").Create(fbase + ".UV_RGB.tif", x.shape[2], x.shape[1], 3, gdal.GDT_Float32)
    out_ds.SetGeoTransform(geoTransform)
    out_ds.SetProjection(wkt)

 
    print(x.shape)
    print(fbase + ".UV_RGB.tif")

    for j in range(3):
        out_ds.GetRasterBand((j+1)).WriteArray(x[j,:,:])
    out_ds.FlushCache()
    out_ds = None

