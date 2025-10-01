

from osgeo import gdal
import os 
import numpy as np

files = [
"/data/nlahaye/remoteSensing/GK2A_FOG/TWILIGHT_FALSE/gk2a_ami_le1b_ir087_ea020lc_202405072130.tif",
"/data/nlahaye/remoteSensing/GK2A_FOG/RADI_FOG/gk2a_ami_le1b_ir087_ea020lc_202109302100.tif"
] 


for i in range(len(files)):
  
    dat = gdal.Open(files[i])
    imgData = dat.ReadAsArray()
    nx = imgData.shape[2]
    ny = imgData.shape[1]
    metadata=dat.GetMetadata()
    geoTransform = dat.GetGeoTransform()
    wkt = dat.GetProjection()
    dat.FlushCache()
    dat = None

    bands = []
    #bands.append(imgData[4] - imgData[2])
    #bands.append(imgData[2] - imgData[8])
    #bands.append(imgData[2]) 
    bands.append(imgData[10])
    bands.append(imgData[11])
    bands.append(imgData[9])


    fname, _ = os.path.splitext(files[i])
    fname = fname + "_RGB.tif"
    out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 3, gdal.GDT_Float32)
    out_ds.SetMetadata(metadata)
    out_ds.SetGeoTransform(geoTransform)
    out_ds.SetProjection(wkt)
    for i in range(len(bands)):
        print(imgData.shape, bands[i].shape, nx, ny)
        out_ds.GetRasterBand(i+1).WriteArray(bands[i])
    out_ds.FlushCache()
    out_ds = None


