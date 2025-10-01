
import numpy as np
import matplotlib
matplotlib.use('agg')
from pprint import pprint
from sklearn.metrics import confusion_matrix
from osgeo import gdal, osr
import argparse
import os
from sit_fuse.utils import numpy_to_torch, read_yaml, get_read_func
import zarr
from netCDF4 import Dataset
from pyhdf import SD
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def read_viirs_fire_mask(filename, kwargs):

        ds = Dataset(filename)
        fire_mask = ds.variables['fire mask'][:]


        if "bool_fire" in kwargs and kwargs["bool_fire"]:
            dat = np.zeros(fire_mask.shape)
            inds = np.where(fire_mask >= 9)
            dat[inds] = 1
        else:
            dat = fire_mask


        if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                            dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]


        return dat


def read_viirs_fire_mask_geo(filename, kwargs):

    ds = Dataset(filename)
    lon = ds.variables['FP_longitude']
    lat = ds.variables['FP_latitude']
    line = ds.variables['FP_line']
    sample = ds.variables['FP_sample']


    dat = np.array([lat, lon, line, sample])

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]

    return dat

def read_modis_fire_mask(filename, **kwargs):

    f = SD.SD(filename)
    sds_obj = f.select('fire mask')
    dat = sds_obj.get()
    f.end()

    if "bool_fire" in kwargs and kwargs["bool_fire"]:
        dat = np.zeros(fire_mask.shape)
        inds = np.where(fire_mask >= 9)
        dat[inds] = 1
    else:
        dat = fire_mask


    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]


    return dat


def read_modis_fire_mask_geo(filename, **kwargs):

    f = SD.SD(filename)
    sds_obj = f.select('FP_latitude')
    lat = sds_obj.get()
    sds_obj = f.select('FP_longitude')
    lon = sds_obj.get()
    sds_obj = f.select('FP_line')
    line = sds_obj.get()
    sds_obj = f.select('FP_sample')
    sample = sds_obj.get()

    dat = np.array([lat, lon, line, sample])

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
            dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat



def im_to_latlon(col, row, gt): # p: pixel coords to map coords
    c, a, b, f, d, e = gt
    x_geo = a * col + b * row + a * 0.5 + b * 0.5 + c
    y_geo = d * col + e * row + d * 0.5 + e * 0.5 + f
    return x_geo, y_geo  # map coordinates


def latlon_to_im(x_geo, y_geo, gt):
    c, a, b, f, d, e = gt
    col = int((x_geo - c) / a)
    row = int((y_geo - f) / e)
 
    return row, col


data = gdal.Open("/data/nlahaye/SIT_FUSE_Geo/VIIRS/noaa20_viirs_i01_20190806_191800_wgs84_fit.tif.clust.data_79769clusters.zarr.full_geo.smoke.tif.Contours.tif")
gt = data.GetGeoTransform()
ref = data.ReadAsArray()
wkt = data.GetProjection()

fp_files = [
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VJ114IMG.A2019220.0854.002.2024030202835.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VJ114IMG.A2019220.1036.002.2024030202901.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VJ114IMG.A2019220.2018.002.2024030202844.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VJ114IMG.A2019220.2200.002.2024030202837.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VJ114IMG.A2019221.0836.002.2024030211308.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VJ114IMG.A2019221.1012.002.2024030211308.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VJ114IMG.A2019221.2000.002.2024030211304.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VJ114IMG.A2019221.2142.002.2024030211305.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VJ114IMG.A2019222.0954.002.2024030214823.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VJ114IMG.A2019222.1942.002.2024030214805.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VJ114IMG.A2019222.2124.002.2024030214813.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VJ114IMG.A2019223.0936.002.2024030222348.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VJ114IMG.A2019223.1118.002.2024030222351.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VJ114IMG.A2019223.1924.002.2024030222345.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VJ114IMG.A2019223.2106.002.2024030222350.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VJ114IMG.A2019224.0918.002.2024030225507.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VJ114IMG.A2019224.1100.002.2024030225521.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VJ114IMG.A2019224.2042.002.2024030225507.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019218.0842.002.2024094221202.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019218.1024.002.2024094221204.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019218.2006.002.2024094221204.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019218.2148.002.2024094221206.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019219.1000.002.2024094224207.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019219.1006.002.2024094224206.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019219.1948.002.2024094224207.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019219.2130.002.2024094224205.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019220.0942.002.2024094230337.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019220.1124.002.2024094230335.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019220.1930.002.2024094230338.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019220.2112.002.2024094230336.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019221.0924.002.2024094235612.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019221.1106.002.2024094235611.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019221.1912.002.2024094235611.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019221.2054.002.2024094235612.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019222.0906.002.2024095002603.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019222.1048.002.2024095002606.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019222.2036.002.2024095002633.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019223.0848.002.2024095004850.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019223.1030.002.2024095004843.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019223.2012.002.2024095004842.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019223.2154.002.2024095004842.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019224.1012.002.2024095012041.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019224.1954.002.2024095012041.nc",
"/data/nlahaye/remoteSensing/VIIRS_FIRE/VNP14IMG.A2019224.2136.002.2024095012043.nc",
]

for i in range(len(fp_files)):
    mask = np.zeros(ref.shape)
    pixels = read_viirs_fire_mask_geo(fp_files[i], {"bool_fire": True})
    mp = read_viirs_fire_mask(fp_files[i], {"bool_fire": True})

    print(pixels.shape, mask.max())

    for j in range(pixels.shape[1]):
        lat = pixels[0,j]
        lon = pixels[1,j]
        ln = int(pixels[2,j])
        smp = int(pixels[3,j])

        print(ln, smp, mp.shape)

        line, sample = latlon_to_im(lon, lat, gt)
        line = int(line)
        sample = int(sample)
        if mp[ln, smp] > 0 and line >= 0 and line < mask.shape[0] and sample >= 0 and sample < mask.shape[1]:
            mask[line, sample] = 1
    driver = gdal.GetDriverByName('GTiff')
    dstds = driver.Create(fp_files[i] + ".tif", mask.shape[1], mask.shape[0], 1, gdal.GDT_Float32)

    dstds.SetGeoTransform(gt)
    dstds.SetProjection(wkt)
 

    dstds.GetRasterBand(1).WriteArray(mask)
    #dstds.GetRasterBand(1).SetNoDataValue(-9999.0)

    dstds.FlushCache()
    dstds=None




