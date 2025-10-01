
import os
import argparse
import cv2
import re
import numpy as np
from osgeo import osr, gdal
from sit_fuse.preprocessing.misc_utils import clip_geotiff
from sit_fuse.utils import read_yaml





 
fname_pv_lab = "/data/nlahaye/remoteSensing/PV_Mapping/pv_polygons.tif"

fname_pv = "/data/nlahaye/remoteSensing/PV_Mapping/dice_lr1e-5.tif"

for i in range(len(fnames)):
    data = gdal.Open(fnames[i])
    geoTransform = data.GetGeoTransform()

    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * data.RasterXSize
    miny = maxy + geoTransform[5] * data.RasterYSize
    extents = [minx, miny, maxx, maxy]

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs = srs.ExportToWkt() 
  
    gdal.Warp("/data/nlahaye/remoteSensing/PV_Mapping/pv_polygons_" + str(i) + ".tif", fname_pv, outputBounds=extents, outputBoundsSRS=srs)



