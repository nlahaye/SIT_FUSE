import cv2

from osgeo import gdal
import os
from netCDF4 import Dataset
 
#tiff = "/data/nlahaye/output/Learnergy/EMIT_WQ_TEST/EMIT_L2A_RFL_001_20240626T152155_2417810_031.nc.clust.data_79498clusters.tif"
#nc_data = "/data/nlahaye/remoteSensing/EMIT_WQ/EMIT_L2A_RFL_001_20240626T152155_2417810_031.nc"


nc_dats= [
"/data/nlahaye/remoteSensing/EMIT_RED_SEA/EMIT_L2A_RFL_001_20250127T085626_2502706_018.nc",
"/data/nlahaye/remoteSensing/EMIT_RED_SEA/EMIT_L2A_RFL_001_20250127T085637_2502706_019.nc",
"/data/nlahaye/remoteSensing/EMIT_RED_SEA/EMIT_L2A_RFL_001_20250127T085614_2502706_017.nc"]

tiffs = [
"/data/nlahaye/output/Learnergy/EMIT_WQ_REDUCED/EMIT_L2A_RFL_001_20250127T085626_2502706_018.nc.clust.data_62698clusters.full_geo.tif",
"/data/nlahaye/output/Learnergy/EMIT_WQ_REDUCED/EMIT_L2A_RFL_001_20250127T085637_2502706_019.nc.clust.data_62699clusters.full_geo.tif",
"/data/nlahaye/output/Learnergy/EMIT_WQ_REDUCED/EMIT_L2A_RFL_001_20250127T085614_2502706_017.nc.clust.data_62698clusters.full_geo.tif"]

for i in range(len(nc_dats)):
    tiff = tiffs[i]
    nc_data = nc_dats[i]
 
    x = Dataset(nc_data)["reflectance"][:]
    x = x[:,:,[30,20,10]]

    fbase, _ = os.path.splitext(tiff)

    dat = gdal.Open(tiff)
    metadata = dat.GetMetadata()
    geoTransform = dat.GetGeoTransform()
    wkt = dat.GetProjection()

    fg = dat.ReadAsArray()
    
    print(x.shape, fg.shape)  
    #x = cv2.resize(x, dsize=(fg.shape[1], fg.shape[0]), interpolation=cv2.INTER_CUBIC)
 
    out_ds = gdal.GetDriverByName("GTiff").Create(fbase + ".RGB.no_geo.tif", x.shape[1], x.shape[0], 3, gdal.GDT_Float32)
    #out_ds.SetMetadata(metadata)
    #out_ds.SetGeoTransform(geoTransform)
    #out_ds.SetProjection(wkt)

 
    print(x.shape)
    print(fbase + ".RGB.no_geo.tif")

    for j in range(x.shape[2]):
        out_ds.GetRasterBand((j+1)).WriteArray(x[:,:,j])
    out_ds.FlushCache()
    out_ds = None

