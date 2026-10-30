import os
from osgeo import gdal
import numpy as np

from pyresample import area_config, bilinear, geometry, data_reduce, create_area_def, kd_tree
from pyresample.utils.rasterio import get_area_def_from_raster

fire_masks = [
"/data/nlahaye/SIT_FUSE_Geo/GOES_TEMPO_DATA/GOES_TEMPO_FIRE.tif",
"/data/nlahaye/SIT_FUSE_Geo/GOES_TEMPO_DATA/GOES_TEMPO_FIRE_2.tif",
"/data/nlahaye/SIT_FUSE_Geo/GOES_TEMPO_DATA/GOES_TEMPO_FIRE_3.tif",
"/data/nlahaye/SIT_FUSE_Geo/GOES_TEMPO_DATA/GOES_TEMPO_FIRE_4.tif",
"/data/nlahaye/SIT_FUSE_Geo/GOES_TEMPO_DATA/GOES_TEMPO_FIRE_5.tif",
"/data/nlahaye/SIT_FUSE_Geo/GOES_TEMPO_DATA/GOES_TEMPO_FIRE_6.tif"
]

fire_background_mask = "/data/nlahaye/SIT_FUSE_Geo/GOES_TEMPO_DATA/GOES_TEMPO_FIRE_BACKGROUND.tif"

bg = gdal.Open(fire_background_mask).ReadAsArray()
bg_inds = np.where(bg == 0)

print(np.unique(bg))

for i in range(len(fire_masks)):

    new_mask = np.zeros(bg.shape) - 1
    new_mask[bg_inds] = 0

    src = gdal.Open(fire_masks[i])
    fmask = src.ReadAsArray()

    area_def = get_area_def_from_raster(fire_masks[i])
    final_area_def = get_area_def_from_raster(fire_background_mask)
    fmask = np.squeeze(kd_tree.resample_nearest(area_def, fmask, final_area_def, radius_of_influence=150, fill_value = -1)) 

    fire_inds = np.where(fmask == 1)
    new_mask[fire_inds] = 1    

    nx = fmask.shape[1]
    ny = fmask.shape[0]
    metadata=src.GetMetadata()
    geoTransform = src.GetGeoTransform()
    wkt = src.GetProjection()

    print(nx, ny, fmask.shape, bg.shape, new_mask.shape)

    fname = os.path.splitext(fire_masks[i])[0] + "complete.tif"
    out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Float32)
    out_ds.SetMetadata(metadata)
    out_ds.SetGeoTransform(geoTransform)
    out_ds.SetProjection(wkt)
    out_ds.GetRasterBand(1).WriteArray(new_mask)
    out_ds.FlushCache()
    out_ds = None



