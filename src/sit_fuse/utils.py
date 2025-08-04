"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
import torch
import yaml
import cv2
import os
import zarr
import re
import numpy as np
import rioxarray
from shapely.geometry import mapping
import xarray as xr
import dask.array as da
from netCDF4 import Dataset
from osgeo import osr, gdal
import pandas as pd
import geopandas as gpd
from geocube.api.core import make_geocube
from pprint import pprint
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sit_fuse.viz.CMAP import CMAP, CMAP_COLORS
from glob import glob
from scipy.interpolate import griddata
from pyhdf import SD
from datetime import datetime
import regionmask
import h5py
from collections import OrderedDict
ocean_basins_50 =  regionmask.defined_regions.natural_earth_v5_1_2.ocean_basins_50 

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from dask_ml.preprocessing import StandardScaler as DaskStandardScaler

from sit_fuse.preprocessing.misc_utils import lee_filter


def get_output_shape(model, image_dim):
        with torch.no_grad():
            tmp = torch.rand(*(image_dim)).to(next(model.parameters()).device)
            if isinstance(tmp, list) or isinstance(tmp, tuple):
                tmp = tmp[0] 
            tmp = model.forward(tmp)
            if isinstance(tmp, list) or isinstance(tmp, tuple):
                tmp = tmp[0]
            return tmp.data.shape


def concat_numpy_files(np_files, final_file):
    full = None
    for i in range(len(np_files)):
        print(i, np_files[i])
        tmp = np.load(np_files[i])
        if full is None:
            full = tmp
        else:
            np.concatenate((full, tmp), axis=0)

        del tmp
    np.save(final_file, full)


def torch_to_numpy(trch):
    dat = trch.numpy()

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]

    return dat
    
def numpy_to_torch(npy):
    dat = torch.from_numpy(npy)

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]

    return dat

def read_yaml(fpath_yaml):
    yml_conf = None
    with open(fpath_yaml) as f_yaml:
        yml_conf = yaml.load(f_yaml, Loader=yaml.FullLoader)
    return yml_conf


def torch_load(filename, **kwargs):
    dat = torch.load(filename)

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat

def numpy_load(filename, **kwargs):

    dat = np.load(filename)

    if "bands" in kwargs:
        bands = kwargs["bands"]
        chan_dim = kwargs["chan_dim"]        
        
        dat = np.moveaxis(dat, chan_dim, 2)
        dat = dat[:,:,bands]
        dat = np.moveaxis(dat, 2, chan_dim)

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]

    return dat

def zarr_load(filename, **kwargs):
    dat = da.from_zarr(filename)

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat

def numpy_from_zarr(filename, **kwargs):
    dat = np.array(zarr_load(filename).compute())
    
    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat = dat[kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat



def read_viirs_aerosol_type(filename, **kwargs):

    ds = Dataset(filename)
    aero_mask = ds.groups["geophysical_data"].variables["Optical_Depth_Land_And_Ocean"][:]

    if "bool_aero" in kwargs and kwargs["bool_aero"]:
        dat = np.zeros(aero_mask.shape)
        inds = np.where(aero_mask >= 0.2)
        dat[inds] = 1 
    else:
        dat = aero_mask

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]


    return dat 


def read_viirs_aerosol_type_geo(filename, **kwargs):

    ds = Dataset(filename)
    lon = ds.groups["geolocation_data"].variables["longitude"][:]
    lat = ds.groups["geolocation_data"].variables["latitude"][:]

    dat = np.array([lat,lon])            

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]

    print(dat.shape)

    return dat

def read_emit(filename, **kwargs):

    ds = Dataset(filename)
    dat = ds.variables['radiance'][:]

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat

def read_emit_l2(filename, **kwargs):

    ds = Dataset(filename)
    dat = ds.variables['reflectance'][:]
 
    print(dat.shape)

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat = dat[:,kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]

    print(dat.shape)

    if "start_wl" in kwargs and "end_wl" in kwargs:
        wls = ds.groups["sensor_band_parameters"]["wavelengths"][:]
        inds = np.where(((wls >= kwargs["start_wl"]) & (wls <= kwargs["end_wl"])))
        print(len(inds), inds)
        dat = dat[inds[0],:,:]

    print(dat.shape)

    if "mask_shp" in kwargs:

        loc = read_zarr_geo(filename[2], **kwargs)

        print(loc[0,:,0].shape, loc[:,0,1].shape, dat.shape)
        print(loc[:,0,0], loc[0,:,1])


        emit_xr = xr.Dataset(coords=dict(bands=(["band"],np.linspace(0,dat.shape[0],dat.shape[0])), lon=(["x"],loc[0,:,1]),\
        lat=(["y"],loc[:,0,0])))

       

        emit_xr["emit"] = (['band', 'y', 'x'],  dat)

        emit_xr = emit_xr.rename({'x': 'lon','y': 'lat'})
        #emit_xr["emit"] = emit_xr["emit"].rename({'x': 'lon','y': 'lat'})
        emit_xr["emit"].rio.set_spatial_dims("lat", "lon", inplace=True)
        emit_xr["emit"].rio.write_crs("epsg:4326", inplace=True)
        emit_xr.rio.write_crs("epsg:4326", inplace=True) 
        emit_xr.rio.set_spatial_dims("lat", "lon", inplace=True)
     

        mask = gpd.read_file(kwargs["mask_shp"], crs="epsg:4326")
        mask.crs = 'epsg:4326'
        print(mask.crs)
        
        geometries = mask.geometry.apply(mapping)

        emit_xr = xr.open_dataset(filename[1], engine='rasterio')
        print(emit_xr.rio)
        print(emit_xr["spatial_ref"])
        emit_temp = emit_xr.rio.clip(geometries, drop=False)
        print(np.unique(emit_temp["band_data"]))
        tmp2 = emit_temp["band_data"].isnull().to_numpy().astype(np.bool_)
        print(np.unique(tmp2))
        dat[tmp2] = -999999       
        print(np.unique(dat))

    return dat

def read_emit_geo(filename, **kwargs):

    ds = Dataset(filename)
    dat1 = ds['location']["lon"][:]
    dat2 = ds['location']["lat"][:]

    dat = np.array([dat1, dat2])

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]

    return dat

def read_zarr_geo(filename, **kwargs):

    dat = zarr.load(filename)
    print(filename, dat.shape)
    return dat
    

def read_pace_oc(filename, **kwargs):

    vrs = ["RRS.V3_0.Rrs"]
    kwrg = {}
    data = None
    if "nrt"  in kwargs and  kwargs["nrt"]:
        kwrg['nrt'] = kwargs["nrt"]
        flename = filename + vrs[0] + ".4km.NRT.nc"
    else:
        flename = filename + vrs[0] + ".4km.nc"
    start_ind = 9
    print(flename, vrs[0][start_ind:])
    f = Dataset(flename)
    f.set_auto_maskandscale(False)
    ref = f.variables[vrs[0][start_ind:]]
    data = ref[:].astype(np.float32)
    try:
        valid_data_ind = np.where((data >= ref.valid_min) & (data <= ref.valid_max))
        invalid_data_ind = np.where((data < ref.valid_min) | (data > ref.valid_max))
        try:
            data[valid_data_ind] = data[valid_data_ind] * ref.scale_factor
        except AttributeError:
            pass
        try:
             data[valid_data_ind] = data[valid_data_ind] + ref.add_offset
        except AttributeError:
            pass
        data[invalid_data_ind] = -999999.0
    except AttributeError:
        pass
    print(data.shape)
    data = np.moveaxis(data, 2,0)
    print(data.shape)
    dat = data.astype(np.float32)

    if "start_lat" in kwargs and "end_lat" in kwargs and "start_lon" in kwargs and "end_lon" in kwargs:
        loc = read_oc_geo(filename, **kwrg)
        lat = loc[0]
        lon = loc[1]
        print(lat.shape, lon.shape, dat.shape)
        inds1 = np.where((lat >= kwargs["start_lat"]) & (lat <= kwargs["end_lat"]))
        inds2 = np.where((lon >= kwargs["start_lon"]) & (lon <= kwargs["end_lon"]))
        lat = lat[inds1]
        lon = lon[inds2]

        nind2, nind1 = np.meshgrid(inds2, inds1)
        dat = dat[:, nind1,nind2]

    #TODO Mask via shapefile

    tmp1 = None
    tmp2 = None
    if "mask_oceans" in kwargs:
        land_temp = ocean_basins_50.mask(lon, lat)
        land_temp = land_temp.rename({'lon': 'x','lat': 'y'})
        tmp1 = land_temp.isnull().to_numpy().astype(np.bool_)

    final_mask = None
    if tmp1 is not None and tmp2 is not None:
        final_mask = xr.apply_ufunc(np.logical_and, tmp1, tmp2, vectorize=True, dask="parallelized",\
            input_core_dims=[[],[]], output_core_dims=[[],[]])
    elif tmp1 is not None:
        final_mask = tmp1
    elif tmp2 is not None:
        final_mask = tmp2

    if final_mask is not None:
        print(final_mask)
        dat[:,final_mask] = -999999

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat


def read_s3_oc(filename, **kwargs):
    vrs = ["RRS.Rrs_400","RRS.Rrs_412","RRS.Rrs_443","RRS.Rrs_490","RRS.Rrs_510","RRS.Rrs_560","RRS.Rrs_620","RRS.Rrs_665","RRS.Rrs_674","RRS.Rrs_681", "RRS.Rrs_709"]

    #vrs = ["CHL.chlor_a", "KD.Kd_490", "RRS.aot_865", "RRS.angstrom"]
    kwrg = {}
    data1 = None
    for i in range(len(vrs)):
        if "nrt" in kwargs and kwargs["nrt"]:
            kwrg['nrt'] = kwargs["nrt"]
            flename = filename + vrs[i] + ".4km.NRT.nc"
        else:
            flename = filename + vrs[i] + ".4km.nc"
        # flename = filename + vrs[i] + ".4km.nc"
        print(flename)
        f = Dataset(flename)
        f.set_auto_maskandscale(False)
        start_ind = 4
        if "KD" in vrs[i]:
            start_ind = 3
        ref = f.variables[vrs[i][start_ind:]]
        data = ref[:].astype(np.float32)
        try:
            valid_data_ind = np.where((data >= ref.valid_min) & (data <= ref.valid_max))
            invalid_data_ind = np.where((data < ref.valid_min) | (data > ref.valid_max))
            try:
                data[valid_data_ind] = data[valid_data_ind] * ref.scale_factor 
            except AttributeError:
                pass
            try:
                 data[valid_data_ind] = data[valid_data_ind] + ref.add_offset
            except AttributeError:
                pass
            data[invalid_data_ind] = -999999.0
        except AttributeError:
            pass
        print(data.shape)
        if data1 is None:
            data1 = np.zeros((len(vrs), data.shape[0], data.shape[1]))
        data1[i] = data
    dat = data1.astype(np.float32)

    if "start_lat" in kwargs and "end_lat" in kwargs and "start_lon" in kwargs and "end_lon" in kwargs:
        loc = read_oc_geo(filename, **kwrg)
        lat = loc[0]
        lon = loc[1]
        print(lat.shape, lon.shape, dat.shape)
        inds1 = np.where((lat >= kwargs["start_lat"]) & (lat <= kwargs["end_lat"]))
        inds2 = np.where((lon >= kwargs["start_lon"]) & (lon <= kwargs["end_lon"]))
        lat = lat[inds1]
        lon = lon[inds2]

        nind2, nind1 = np.meshgrid(inds2, inds1)
        dat = dat[:, nind1,nind2]

    #TODO Mask via shapefile

    tmp1 = None
    tmp2 = None 
    if "mask_oceans" in kwargs:
        land_temp = ocean_basins_50.mask(lon, lat)
        land_temp = land_temp.rename({'lon': 'x','lat': 'y'})
        tmp1 = land_temp.isnull().to_numpy().astype(np.bool_)

    final_mask = None
    if tmp1 is not None and tmp2 is not None:
        final_mask = xr.apply_ufunc(np.logical_and, tmp1, tmp2, vectorize=True, dask="parallelized",\
            input_core_dims=[[],[]], output_core_dims=[[],[]])
    elif tmp1 is not None:
        final_mask = tmp1
    elif tmp2 is not None:
        final_mask = tmp2

    if final_mask is not None:
        print(final_mask)
        dat[:,final_mask] = -999999

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat



def read_viirs_oc(filename, **kwargs):
    #vrs = ["CHL.chlor_a", "KD.Kd_490", "PAR.par", "PIC.pic", "POC.poc", "RRS.aot_862", "RRS.angstrom"]
    #vrs2 = ["CHL.chlor_a", "KD.Kd_490", "PAR.par", "PIC.pic", "POC.poc", "RRS.aot_868", "RRS.angstrom"]

    vrs = ["RRS.Rrs_410", "RRS.Rrs_443", "RRS.Rrs_486", "RRS.Rrs_551", "RRS.Rrs_671"]
    vrs2 = ["RRS.Rrs_411", "RRS.Rrs_445", "RRS.Rrs_489", "RRS.Rrs_556", "RRS.Rrs_667"]

    #"RRS.aot_862", "RRS.Rrs_410", "RRS.Rrs_443", "RRS.Rrs_486", "RRS.Rrs_551", "RRS.Rrs_671"]
    if 'JPSS' in filename:
        vrs = vrs2

    data1 = []
    kwrg = {}
    allow_nrt = kwargs.get("nrt", False)
    for i in range(len(vrs)):
        if 'JPSS' in filename:
            vrs = vrs2

    for i in range(len(vrs)):
        base = filename + vrs[i] + ".4km"
        flename = base + ".nc"
        if not os.path.exists(flename) and allow_nrt:
            flename = base + ".NRT.nc"
            kwrg["nrt"] = True
        f = Dataset(flename)
        f.set_auto_maskandscale(False)
        start_ind = 4
        if "KD" in vrs[i]:
            start_ind = 3
        ref = f.variables[vrs[i][start_ind:]]
        data = ref[:].astype(np.float32)
        try:
            valid_data_ind = np.where((data >= ref.valid_min) & (data <= ref.valid_max))
            invalid_data_ind = np.where((data < ref.valid_min) | (data > ref.valid_max))
            try:
                data[valid_data_ind] = data[valid_data_ind] * ref.scale_factor
            except AttributeError:
                pass
            try:
                 data[valid_data_ind] = data[valid_data_ind] + ref.add_offset
            except AttributeError:
                pass
        except AttributeError:
            pass
        data[invalid_data_ind] = -999999.0
        data1.append(data)
    dat = np.array(data1).astype(np.float32)

    if "start_lat" in kwargs and "end_lat" in kwargs and "start_lon" in kwargs and "end_lon" in kwargs: 
        loc = read_oc_geo(filename, **kwrg)
        lat = loc[0]
        lon = loc[1]
        print(lat.shape, lon.shape, dat.shape)
        inds1 = np.where((lat >= kwargs["start_lat"]) & (lat <= kwargs["end_lat"]))
        inds2 = np.where((lon >= kwargs["start_lon"]) & (lon <= kwargs["end_lon"]))
        lat = lat[inds1]
        lon = lon[inds2]
        nind1, nind2 = np.meshgrid(inds2, inds1)
        dat = dat[:, nind1,nind2]


    #TODO Mask via shapefile

    tmp1 = None
    tmp2 = None
    if "mask_oceans" in kwargs:
        land_temp = ocean_basins_50.mask(lon, lat)
        land_temp = land_temp.rename({'lon': 'x','lat': 'y'})
        tmp1 = land_temp.isnull().to_numpy().astype(np.bool_)

    final_mask = None
    if tmp1 is not None and tmp2 is not None:
        final_mask = xr.apply_ufunc(np.logical_and, tmp1, tmp2, vectorize=True, dask="parallelized",\
            input_core_dims=[[],[]], output_core_dims=[[],[]])
    elif tmp1 is not None:
        final_mask = tmp1
    elif tmp2 is not None:
        final_mask = tmp2

    if final_mask is not None:
        print(final_mask)
        dat[:,final_mask] = -999999

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat


def read_oc_geo(filename, **kwargs):

    vrs = ["lat", "lon"]
    if "PACE" not in filename:
        stem = "RRS.Rrs_445" if "JPSS" in filename else "RRS.Rrs_443"
        base = f"{filename}{stem}.4km"
    else:  # PACE has its own naming convention
        base = f"{filename}RRS.V3_0.Rrs.4km"

    flename = base + ".nc"
    if not os.path.exists(flename) and kwargs.get("nrt", False):
        flename = base + ".NRT.nc"

    f = Dataset(flename)
    f.set_auto_maskandscale(False)
    data1 = []
    for i in range(len(vrs)):
        ref = f.variables[vrs[i]]
        data = ref[:].astype(np.float32)
        valid_data_ind = np.where((data >= ref.valid_min) & (data <= ref.valid_max))
        invalid_data_ind = np.where((data < ref.valid_min) | (data > ref.valid_max))
        data[invalid_data_ind] = -999999.0
        data1.append(data)
    #longr, latgr = np.meshgrid(data1[1], data1[0])
    #dat = np.array([longr, latgr]).astype(np.float32)
    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        data1[0] = data1[0][kwargs["start_line"]:kwargs["end_line"]]
        data1[1] = data1[1][kwargs["start_sample"]:kwargs["end_sample"]]

    if "start_lat" in kwargs and "end_lat" in kwargs and "start_lon" in kwargs and "end_lon" in kwargs:
        inds1 = np.where((data1[0] >= kwargs["start_lat"]) & (data1[0] <= kwargs["end_lat"]))
        inds2 = np.where((data1[1] >= kwargs["start_lon"]) & (data1[1] <= kwargs["end_lon"]))
        lat = data1[0]
        lon = data1[1]


        nind2, nind1 = np.meshgrid(inds2, inds1)
        lon1, lat1 = np.meshgrid(lon, lat)    
    
        data1 = np.array([lat1,lon1]) 

        data1 = data1[:, nind1,nind2]


        #dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return data1


def read_modis_oc(filename, **kwargs):

    #vrs = ["CHL.chlor_a", "FLH.ipar", "FLH.nflh", "KD.Kd_490", "PAR.par", "PIC.pic", "POC.poc", "RRS.aot_869", "RRS.angstrom"]
    vrs = ["RRS.Rrs_412", "RRS.Rrs_443", "RRS.Rrs_469", "RRS.Rrs_488", "RRS.Rrs_531", "RRS.Rrs_547", "RRS.Rrs_555", "RRS.Rrs_645", "RRS.Rrs_667", "RRS.Rrs_678"]

    kwrg = {}
    data1 = []
    for i in range(len(vrs)):
        if "nrt" in kwargs and kwargs["nrt"]:
            kwrg['nrt'] = kwargs["nrt"]
            flename = filename + vrs[i] + ".4km.NRT.nc"
        else:
            flename = filename + vrs[i] + ".4km.nc"
        # flename = filename + vrs[i] + ".4km.nc"
        print(flename)
        f = Dataset(flename)
        f.set_auto_maskandscale(False)
        start_ind = 4
        if "KD" in vrs[i]:
            start_ind = 3
        ref = f.variables[vrs[i][start_ind:]]
        data = ref[:].astype(np.float32)
        try:
            valid_data_ind = np.where((data >= ref.valid_min) & (data <= ref.valid_max))
            invalid_data_ind = np.where((data < ref.valid_min) | (data > ref.valid_max))
            try:
                data[valid_data_ind] = data[valid_data_ind] * ref.scale_factor 
            except AttributeError:
                pass
            try:
                 data[valid_data_ind] = data[valid_data_ind] + ref.add_offset
            except AttributeError:
                pass
        except AttributeError:
            pass
        data[invalid_data_ind] = -999999.0
        data1.append(data)
        #if i == 0:
        #    plt.imshow(data)
        #    plt.savefig(filename + "CHLOR_FULL.png")
    dat = np.array(data1).astype(np.float32)

    loc = read_oc_geo(filename, **kwrg)
    lat = loc[0]
    lon = loc[1]
    print(lat.shape, lon.shape, dat.shape)
    if "start_lat" in kwargs and "end_lat" in kwargs and "start_lon" in kwargs and "end_lon" in kwargs: 
        inds1 = np.where((lat >= kwargs["start_lat"]) & (lat <= kwargs["end_lat"]))
        inds2 = np.where((lon >= kwargs["start_lon"]) & (lon <= kwargs["end_lon"]))
        lat = lat[inds1]
        lon = lon[inds2]
        nind1, nind2 = np.meshgrid(inds2, inds1)
        dat = dat[:, nind1,nind2]

    #TODO Mask via shapefile

    tmp1 = None
    tmp2 = None
    if "mask_oceans" in kwargs:
        land_temp = ocean_basins_50.mask(lon, lat)
        land_temp = land_temp.rename({'lon': 'x','lat': 'y'})
        tmp1 = land_temp.isnull().to_numpy().astype(np.bool_)

    final_mask = None
    if tmp1 is not None and tmp2 is not None:
        final_mask = xr.apply_ufunc(np.logical_and, tmp1, tmp2, vectorize=True, dask="parallelized",\
            input_core_dims=[[],[]], output_core_dims=[[],[]])
    elif tmp1 is not None:
        final_mask = tmp1
    elif tmp2 is not None:
        final_mask = tmp2

    if final_mask is not None:
        print(final_mask)
        dat[:,final_mask] = -999999



    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]

    #plt.imshow(np.squeeze(dat[0]))
    #plt.savefig(filename + "CHLOR_SUBSET.png")
    #plt.clf() 

    return dat


def read_modis_aero_mask(filename, **kwargs):

    f = SD.SD(filename)
    sds_obj = f.select('Optical_Depth_Land_And_Ocean')
    aero_mask = sds_obj.get()
    f.end()

    if "bool_aero" in kwargs and kwargs["bool_aero"]:
        dat = np.zeros(aero_mask.shape)
        inds = np.where(aero_mask >= 0.2)
        dat[inds] = 1
    else:
        dat = aero_mask


    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]


    return dat


def read_modis_aero_mask_geo(filename, **kwargs):

    f = SD.SD(filename)
    sds_obj = f.select('Latitude')
    lat = sds_obj.get()
    sds_obj = f.select('Longitude')
    lon = sds_obj.get()

    dat = np.array([lat, lon])

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
            dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat



    


def read_modis_sr(filename, **kwargs):

    data1 = []
    base_path = "" #grid1km/Data_Fields/"
    ds=SD.SD(filename)
    for i in range(1,13):
        data_path = base_path + "Sur_refl" + str(i)
        print(data_path)
        r = ds.select(data_path)
        attrs = r.attributes(full=1)
        valid_range = attrs["valid_range"][0]
        scale_factor = attrs["scale_factor"][0]
        fill = attrs["_FillValue"][0]
        dat = r.get()
        print(valid_range, scale_factor, fill, data_path)

        print(dat.shape, fill, valid_range[0], valid_range[1])
        dat[np.where(dat < valid_range[0])] = fill
        dat[np.where(dat > valid_range[1])] = fill
        inds = np.where(dat == fill)
        dat = dat.astype(np.float32)
        dat = dat * scale_factor
        dat[inds] = -999999
        data1.append(dat)
    data1 = np.array(data1).astype(np.float32)

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                data1 = data1[:,:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]


    data1 = np.swapaxes(data1, 0,1)
    print(data1.shape, "HERE")
    return data1


def read_misr_sim(filename, **kwargs):
 
    ds = Dataset(filename)
    dat = ds.variables['rad'][:]

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat    


def read_tempo_no2_netcdf(filename, **kwargs):

    f = Dataset(filename)
    f.set_always_mask(True)
    f.set_auto_mask(True)
 
    dat = f.groups["product"]["vertical_column_troposphere"][:]
    mask = f.groups["support_data"]["amf_cloud_fraction"][:]

    inds = np.where(mask > 0.75)
    dat[inds] = np.nan

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]

    return dat
 

def read_tempo_no2_netcdf_geo(filename, **kwargs):

    f = Dataset(filename)
    f.set_always_mask(True)
    f.set_auto_mask(True)

    lat = f.variables["latitude"][:]
    lon = f.variables["longitude"][:]

    longr, latgr = np.meshgrid(lon, lat)

    dat = np.array([latgr, longr])

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
            dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat

def read_tempo_netcdf(filename, **kwargs):

    f = Dataset(filename)
    f.set_always_mask(True)    
    f.set_auto_mask(True)

    dat = f.groups['band_290_490_nm']['radiance'][:]
    dat2 = f.groups['band_540_740_nm']['radiance'][:]

    dat = np.ma.concatenate((dat, dat2), axis=2) 

    for j in range(dat.shape[2]):
        print(dat[:,:,j].min(), dat[:,:,j].max(), dat[:,:,j].mean(), dat[:,:,j].std())

    print(dat.shape, dat.min(), dat.max())

    f = None

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    print(dat.shape)
    return dat


def read_tempo_netcdf_geo(filename, **kwargs):

    f = Dataset(filename)
    f.set_always_mask(True)
    f.set_auto_mask(True)

    lat = f.groups['band_290_490_nm']['latitude'][:] 
    lon = f.groups['band_290_490_nm']['longitude'][:]

    dat = np.array([lat, lon])

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
            dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat

def read_goes_netcdf(filenames, **kwargs):
    data1 = []
    for j in range(0, len(filenames)):
        f = Dataset(filenames[j])
        
        print(filenames[j], kwargs.keys())
        fire = False
        bool_fire = False
        if "fire_mask" in kwargs:
            fire = kwargs["fire_mask"]
        if fire and "bool_fire" in kwargs:
            bool_fire = kwargs["bool_fire"]

        if fire:
            rad = f.variables['Mask'][:]
            if bool_fire:
                tmp = np.zeros(rad.shape)
                tmp[np.where((rad > 10) & ((rad < 16) | ((rad > 29) & (rad < 36))))] = 1
                rad = tmp
        else:
            rad = f.variables['Rad'][:]
        #f.close()
        f = None
        print(rad.shape)
        data1.append(rad)
    if not fire:
        refShp = data1[3].shape
        for k in range(0, len(data1)):
            shp = data1[k].shape
            print(shp, refShp)
            if shp[0] != refShp[0] or shp[1] != refShp[1]:
                data1[k] = cv2.resize(data1[k], (refShp[1],refShp[0]), interpolation=cv2.INTER_CUBIC)
            print(data1[k].shape)
    dat = np.array(data1)
    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:	
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    print(dat.shape)
    return dat
 

def read_goes_netcdf_geo(filenames, **kwargs):
    find = min(len(filenames)-1, 3)
    f = Dataset(filenames[find])
    lat, lon = calculate_degrees(f)

    dat = np.array([lat, lon])

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
            dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat


#Please acknowledge the NOAA/NESDIS/STAR Aerosols and Atmospheric Composition Science Team if using any of this code in your work/research!
# Calculate latitude and longitude from GOES ABI fixed grid projection data
# GOES ABI fixed grid projection is a map projection relative to the GOES satellite
# Units: latitude in °N (°S < 0), longitude in °E (°W < 0)
# See GOES-R Product User Guide (PUG) Volume 5 (L2 products) Section 4.2.8 for details & example of calculations
# "file_id" is an ABI L1b or L2 .nc file opened using the netCDF4 library

def calculate_degrees(file_id):
    
    # Read in GOES ABI fixed grid projection variables and constants
    x_coordinate_1d = file_id.variables['x'][:]  # E/W scanning angle in radians
    y_coordinate_1d = file_id.variables['y'][:]  # N/S elevation angle in radians
    projection_info = file_id.variables['goes_imager_projection']
    lon_origin = projection_info.longitude_of_projection_origin
    H = projection_info.perspective_point_height+projection_info.semi_major_axis
    r_eq = projection_info.semi_major_axis
    r_pol = projection_info.semi_minor_axis
    
    # Create 2D coordinate matrices from 1D coordinate vectors
    x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)
    
    # Equations to calculate latitude and longitude
    lambda_0 = (lon_origin*np.pi)/180.0  
    a_var = np.power(np.sin(x_coordinate_2d),2.0) + (np.power(np.cos(x_coordinate_2d),2.0)*(np.power(np.cos(y_coordinate_2d),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(y_coordinate_2d),2.0))))
    b_var = -2.0*H*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    c_var = (H**2.0)-(r_eq**2.0)
    r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
    s_x = r_s*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    s_y = - r_s*np.sin(x_coordinate_2d)
    s_z = r_s*np.cos(x_coordinate_2d)*np.sin(y_coordinate_2d)
    
    # Ignore numpy errors for sqrt of negative number; occurs for GOES-16 ABI CONUS sector data
    np.seterr(all='ignore')
    
    abi_lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
    abi_lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)
    
    return abi_lat, abi_lon




def read_gk2a_netcdf(filenames, **kwargs):
    data1 = []
    for j in range(0, len(filenames)):
        f = Dataset(filenames[j])

        rad = f.variables['image_pixel_values'][:]
        f.close()
        f = None
        data1.append(rad)
    refShp = data1[0].shape
    for k in range(0, len(data1)):
            shp = data1[k].shape
            print(shp, refShp)
            if shp[0] != refShp[0] or shp[1] != refShp[1]:
                data1[k] = cv2.resize(data1[k], (refShp[1],refShp[0]), interpolation=cv2.INTER_CUBIC)
    dat = np.array(data1)

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    print(dat.shape)
    return dat

def read_gk2a_netcdf_geo(filename, **kwargs):

    f = Dataset(filename)
    lat = f.variables["lat"]
    lon = f.variables["lon"]
    dat = np.array([lat, lon])

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
            dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat





def read_hysplit_netcdf_geo(hysplit_fname, **kwargs):

        data1 = []
        if os.path.isfile(hysplit_fname):
                f = Dataset(hysplit_fname)
                data_keys = ["lat", "lon"]
                for i  in range(len(data_keys)):
                    data1.append(f.variables[data_keys[i]][:])

 
        dat = np.array(data1)        
        if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
        return dat


def read_hysplit_netcdf(hysplit_fname, **kwargs):
        if os.path.isfile(hysplit_fname):
                f = Dataset(hysplit_fname)
                data_key = "smoke-col"
                dat = f.variables[data_key][:]
        
        if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
        return dat


def read_s3_netcdf(s3_dir, **kwargs):
        data1 = []
        bands = None
        if "bands" in kwargs:
                bands = kwargs["bands"]
        if os.path.isdir(s3_dir):
                for i in range(1,22):
                        if bands is None or i in bands:
                                data_key = "Oa" + str(i).zfill(2)+ "_radiance"
                                fname = os.path.join(s3_dir, data_key + ".nc")
                                f = Dataset(fname)
                                f.set_auto_maskandscale(False)
                                rad = f.variables[data_key]
                                data = rad[:].astype(np.float32)
                                print(rad.valid_min,  rad.valid_max, rad.scale_factor, rad.add_offset, data.min(), data.max())
                                valid_data_ind = np.where((data >= rad.valid_min) & (data <= rad.valid_max))
                                invalid_data_ind = np.where((data < rad.valid_min) | (data > rad.valid_max))
                                data[valid_data_ind] = data[valid_data_ind] * rad.scale_factor + rad.add_offset
                                data[invalid_data_ind] = -999999.0
                                print(data[valid_data_ind].min(), data[valid_data_ind].max(), data[valid_data_ind].mean(), data[valid_data_ind].std())
                                print(data[valid_data_ind].min()/rad.scale_factor, data[valid_data_ind].max()/rad.scale_factor, data[valid_data_ind].mean()/rad.scale_factor, data[valid_data_ind].std()/rad.scale_factor)
                                print(data.min(), data.max(),"\n\n\n")
                                data1.append(data)
        dat = np.array(data1)
        if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
        return dat



def read_s3_netcdf_geo(s3_dir, **kwargs):
        data1 = []
        if os.path.isdir(s3_dir):
                fname = os.path.join(s3_dir, "geo_coordinates.nc")
                f = Dataset(fname)

                lat = f.variables["latitude"]
                data = lat[:]
                valid_data_ind = np.where((data >= lat.valid_min) & (data <= lat.valid_max))
                invalid_data_ind = np.where((data < lat.valid_min) & (data > lat.valid_max))
                #data[valid_data_ind] = data[valid_data_ind] * lat.scale_factor
                data[invalid_data_ind] = -9999.0
                data1.append(data)

                lon = f.variables["longitude"]
                data = lon[:]
                valid_data_ind = np.where((data >= lon.valid_min) & (data <= lon.valid_max))
                invalid_data_ind = np.where((data < lon.valid_min) & (data > lon.valid_max))
                #data[valid_data_ind] = data[valid_data_ind] * lon.scale_factor
                data[invalid_data_ind] = -9999.0
                data1.append(data)

        dat = np.array(data1)
        if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
        return dat



def read_s6_netcdf(filename, **kwargs):
	f = Dataset(filename)
	dat = f.variables["multilook_ffsar"]
	data = dat[:]
	#scale = dat.scale_factor
	#add_offset = dat.add_offset
	#data = data * scale + add_offset
	data = data.reshape((1, data.shape[0], data.shape[1]))
	if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
		data = data[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
	if "log" in kwargs and kwargs["log"]:
		data = np.log(data)
       
	return data

def read_s6_netcdf_geo(filename, **kwargs):
        data1 = []
        f = Dataset(filename)
        dat = f.variables["lat_ffsar"]
        lat = dat[:]
        data1.append(lat)
        dat2 = f.variables["lon_ffsar"]
        lon = dat2[:]
        data1.append(lon)
        dat = np.array(data1)
        if "start_line" in kwargs and "end_line" in kwargs:
                dat = dat[:, kwargs["start_line"]:kwargs["end_line"]]

        return dat


def read_s2_gtiff(files, **kwargs):
    print(files)
    data1 = []
    for j in range(0, len(files)):
        dat1 = gdal.Open(files[j]).ReadAsArray()
        print(dat1.shape, len(data1))
        if len(data1) > 0:
            print(dat1.shape, data1[0].shape)
        if len(data1) > 0:
            if dat1.shape[0] != data1[0].shape[0] or dat1.shape[1] != data1[0].shape[1]:
                print(dat1.shape, data1[0].shape)
                dat1 = cv2.resize(dat1, (data1[0].shape[1], data1[0].shape[0]), interpolation=cv2.INTER_CUBIC)

        if len(data1) == 0 or len(dat1.shape) == 2:
            data1.append(dat1)
        print(dat1.shape)
    dat = np.array(data1).astype(np.float32)
    if len(dat.shape) == 4:
        dat = np.squeeze(dat)
    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    print(dat.shape)
    return dat



def read_gtiff_multifile_generic(files, **kwargs):
    print(files)
    data1 = []
    for j in range(0, len(files)):
        dat1 = gdal.Open(files[j]).ReadAsArray()
        print(dat1.shape, len(data1), files[j])
        if "grayscale" and dat1.ndim > 2:
            dat1 = dat1[0,:,:]
        if len(data1) > 0:
            print(dat1.shape, data1[0].shape)
        if len(data1) > 0:
            if dat1.shape[0] != data1[0].shape[0] or dat1.shape[1] != data1[0].shape[1]:
                print(dat1.shape, data1[0].shape)
                dat1 = cv2.resize(dat1, (data1[0].shape[1], data1[0].shape[0]), interpolation=cv2.INTER_CUBIC)

        if len(data1) == 0 or len(dat1.shape) == 2:
            data1.append(dat1)
        else:
            data1[0] = np.concatenate((data1[0], dat1), axis=0)
        print(dat1.shape)
    dat = np.array(data1).astype(np.float32)
    if len(dat.shape) == 4:
        dat = np.squeeze(dat)
    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    print(dat.shape)
    return dat


def read_bps_benchmark(flename, **kwargs):
    dat = None
    dat2 = None
 
    if kwargs["RIF"]:
        dat = gdal.Open(flename, gdal.GA_ReadOnly).ReadAsArray()
 
    if kwargs["DAPI"]:
        fname2 = flename.replace("proj", "DAPI")
        dat2 = gdal.Open(fname2, gdal.GA_ReadOnly).ReadAsArray()

    if dat is not None:
        if dat2 is not None:
            dat = np.array([dat, dat2])
        else:
            dat = np.array(dat)
    else:
        dat = np.array(dat2)

    np.clip(dat, 400, 4000)

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                if len(dat.shape) == 3:
                        dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
                else:
                        dat = dat[kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat



def read_burn_severity_stacks(flename, **kwargs):
    init_dat = gdal.Open(flename, gdal.GA_ReadOnly).ReadAsArray()

    dat = init_dat[0:7]
    fire_mask = init_dat[7]
    urban_mask = init_dat[8]


    print(dat.shape, fire_mask.shape, urban_mask.shape)
    inds = np.where((~np.isfinite(dat)) | (np.isnan(dat)))
    dat[inds] = -99999.0
    inds2 = np.where(np.squeeze(urban_mask) > 0.0)
    dat[:,inds2[0],inds2[1]] = -99999.0

    inds2 = np.where(np.squeeze(fire_mask) < 0.5)
    dat[:,inds2[0],inds2[1]] = -99999.0


    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                if len(dat.shape) == 3:
                        dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
                else:
                        dat = dat[kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat



#TODO config for AVIRIS - scale 0.0001 valid_min = 0 and Fill = -9999
def read_gtiff_generic(flename, **kwargs):
    dat = gdal.Open(flename, gdal.GA_ReadOnly).ReadAsArray()
    print(dat.shape)
    dat[np.where(dat.max() <= 0.0)] = -9999.0

    tmp1 = None
    tmp2 = None  # TODO add in generic masking abilities
    if "mask_oceans" in kwargs:

        latlon = read_gtiff_generic_geo(flename, **kwargs)
        land_temp = ocean_basins_50.mask(latlon[:, :, 1], latlon[:, :, 0])
        land_temp = land_temp.rename({'lon': 'x', 'lat': 'y'})
        tmp1 = land_temp.isnull().to_numpy().astype(np.bool_)

        final_mask = None
        if tmp1 is not None and tmp2 is not None:
            final_mask = xr.apply_ufunc(np.logical_and, tmp1, tmp2, vectorize=True, dask="parallelized", \
                                        input_core_dims=[[], []], output_core_dims=[[], []])
        elif tmp1 is not None:
            final_mask = tmp1
        elif tmp2 is not None:
            final_mask = tmp2

        if final_mask is not None:
            print(final_mask)
            dat[:, final_mask] = -9999.0

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        if len(dat.shape) == 3:
            dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
        else:
            dat = dat[kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat


#TODO generalize pieces for other tasks
def insitu_hab_to_multi_hist(insitu_fname, start_date, end_date, clusters_dir, n_clusters, radius_degrees, ranges, global_max, input_file_type, karenia, discard_lower=False, use_key='Total_Phytoplankton', output_dir="."): #, lookup = {}):

    os.makedirs(output_dir, exist_ok=True)
    print(insitu_fname)
    insitu_df = None
    if 'xlsx' in insitu_fname:
        insitu_df = pd.read_excel(insitu_fname)
    elif 'csv' in insitu_fname:
        insitu_df = pd.read_csv(insitu_fname)
    if karenia:
        # Format Datetime Stamp
        insitu_df['Datetime'] = pd.to_datetime(insitu_df['Sample Date'])
        insitu_df.set_index('Datetime')

        # Shorten Karenia Column Name
        insitu_df.rename(columns={"Karenia brevis abundance (cells/L)":use_key}, inplace=True)
    else:
        # Format Datetime Stamp
        insitu_df['Datetime'] = pd.to_datetime(insitu_df['time'])
        insitu_df.set_index('Datetime')
        insitu_df.rename(columns={"latitude": "Latitude"}, inplace=True)
        insitu_df.rename(columns={"longitude": "Longitude"}, inplace=True)
    

    insitu_df = insitu_df[(insitu_df['Datetime'] >= start_date) & (insitu_df['Datetime'] <= end_date)] 

    uniques = sorted(np.unique(insitu_df['Datetime'].values))

    #TODO
    #subset by date - start and end day of SIF 
    #tie date to cluster dat

    final_hist_data = []
    ind = 1
    for dateind in range(len(uniques)):
        date = uniques[dateind]
        #    find associated cluster
        if "sif" in input_file_type:
            clust_fname = os.path.join(os.path.join(clusters_dir, "sif_finalday_" + str(ind) + ".tif"))
        elif "daily" in input_file_type:
            file_ext = "_DAY." #"DAY." #"_DAY." TODO HERE
            if "no_heir" in input_file_type:
                file_ext = file_ext  + "no_heir."
            if "PACE" in input_file_type:
                file_ext = ".RRS.V3_0.Rrs.4km."
            if "alexandrium" in input_file_type:
                file_ext = file_ext  + "alexandrium_bloom.tif"
            elif "seriata" in input_file_type:
                file_ext = file_ext  + "pseudo_nitzschia_seriata_bloom.tif"
            elif "delicatissima" in input_file_type:
                file_ext = file_ext  + "pseudo_nitzschia_delicatissima_bloom.tif"
            elif "karenia_brevis" in input_file_type: # CAN CHANGE BACK
                file_ext = file_ext + "karenia_brevis_bloom.tif"
            else:
                file_ext = file_ext + "total_phytoplankton.tif"
            #clust_fname = os.path.join(os.path.join(clusters_dir, "AQUA_MODIS." + pd.to_datetime(str(date)).strftime("%Y%m%d") + ".L3m." + file_ext))
            clust_fname = os.path.join(os.path.join(clusters_dir, pd.to_datetime(str(date)).strftime("%Y%m%d") +  file_ext))
        elif "alexandrium" in input_file_type:
            clust_fname = os.path.join(os.path.join(clusters_dir, pd.to_datetime(str(date)).strftime("%Y%m%d") + "_alexandrium_bloom" + ".tif"))
        elif "pseudo_nitzschia_seriata" in input_file_type:
            clust_fname = os.path.join(os.path.join(clusters_dir, pd.to_datetime(str(date)).strftime("%Y%m%d") + "_pseudo_nitzschia_seriata_bloom" + ".tif"))
        elif "pseudo_nitzschia_delicatissima" in input_file_type:
            clust_fname = os.path.join(os.path.join(clusters_dir, pd.to_datetime(str(date)).strftime("%Y%m%d") + "_pseudo_nitzschia_delicatissima_bloom" + ".tif"))
        else:
            file_ext = ".tif"
            if "no_heir" in input_file_type: 
                file_ext = ".no_heir.tif"
            if "S3B" in input_file_type:     
                clust_fname = os.path.join(os.path.join(clusters_dir, "S3B_OLCI_ERRNT." + pd.to_datetime(str(date)).strftime("%Y%m%d") + ".L3m.DAY" + file_ext))
            elif "S3A" in input_file_type:
                clust_fname = os.path.join(os.path.join(clusters_dir, "S3A_OLCI_ERRNT." + pd.to_datetime(str(date)).strftime("%Y%m%d") + ".L3m.DAY" + file_ext))
            elif "SNPP_VIIRS" in input_file_type:
                clust_fname = os.path.join(os.path.join(clusters_dir, "SNPP_VIIRS." + pd.to_datetime(str(date)).strftime("%Y%m%d") + ".L3m.DAY" + file_ext))
            elif "JPSS1_VIIRS" in input_file_type:
                clust_fname = os.path.join(os.path.join(clusters_dir, "JPSS1_VIIRS." + pd.to_datetime(str(date)).strftime("%Y%m%d") + ".L3m.DAY" + file_ext))
            elif "JPSS2_VIIRS" in input_file_type:
                clust_fname = os.path.join(os.path.join(clusters_dir, "JPSS2_VIIRS." + pd.to_datetime(str(date)).strftime("%Y%m%d") + ".L3m.DAY" + file_ext))
            elif "AQUA_MODIS" in input_file_type:
                clust_fname = os.path.join(os.path.join(clusters_dir, "AQUA_MODIS." + pd.to_datetime(str(date)).strftime("%Y%m%d") + ".L3m.DAY" + file_ext))
            elif "TERRA_MODIS" in input_file_type:
                clust_fname = os.path.join(os.path.join(clusters_dir, "TERRA_MODIS." + pd.to_datetime(str(date)).strftime("%Y%m%d") + ".L3m.DAY" + file_ext))
            elif "PACE_OCI" in input_file_type:
                file_ext ='.RRS.V3_0.Rrs.4km.tif'
                clust_fname = os.path.join(os.path.join(clusters_dir, "PACE_OCI." + pd.to_datetime(str(date)).strftime("%Y%m%d") + ".L3m.DAY" + file_ext))
            elif "GOES" in input_file_type:
                date_str = pd.to_datetime(str(date)).strftime("%Y%j")
                clust_fname = os.path.join(clusters_dir, f"OR_ABI-L1b-RadC-M6C01_G18_s{date_str}*_clusters.zarr.full_geo.cloud_mask.FullColor.tif")
        ind = ind + 1

        print(clust_fname)

        dat_train = False
        dat_test = False
        clust_fname = glob(clust_fname)
        print(clust_fname)
        
        if len(clust_fname) < 1:
            continue     
        clust_fname = clust_fname[0]
        if not os.path.exists(clust_fname):
            clust_fname = clust_fname + "f"
            if not os.path.exists(clust_fname):
                continue
        print("Opening cluster file:", clust_fname)
        clust = gdal.Open(clust_fname)
        latLon = get_lat_lon(clust_fname)
        clust = clust.ReadAsArray()
        clust = clust * 1000
        clust = clust.astype(np.int32)               
        #print(np.unique(clust))
 
        lat = latLon[:,:,0].reshape(clust.shape[0]*clust.shape[1])
        lon = latLon[:,:,1].reshape(clust.shape[0]*clust.shape[1])
        clust = clust.reshape(clust.shape[0]*clust.shape[1])
        inds_clust = np.where(clust >= 0)
        lat = lat[inds_clust]
        lon = lon[inds_clust]
        clust = clust[inds_clust]


        gdf = gpd.GeoDataFrame(clust, geometry=gpd.GeoSeries.from_xy(lon, lat), crs=4326)
        subset = insitu_df[(insitu_df['Datetime'] == date)]
        #print(len(subset), date)
        gdf_insitu = gpd.GeoDataFrame(subset[use_key], geometry=gpd.GeoSeries.from_xy(subset['Longitude'], subset['Latitude']), crs=4326)

        #gdf_proj = gdf.to_crs({"init": "EPSG:3857"})
        #gdf_insitu_proj = gdf_insitu.to_crs({"init": "EPSG:3857"})     
 
        dists = []
        count = -1
        hist_data = []
        #print("PRE-ITER", len(gdf_insitu))

        for index, poi in gdf_insitu.iterrows():
            count = count + 1
            neighbours = []
            for index2, poi2 in gdf.iterrows():
                #print(abs(poi2.geometry.distance(poi.geometry)) < radius_degrees, "DISTANCE")
                if abs(poi2.geometry.distance(poi.geometry)) < radius_degrees:
                    #print(poi.geometry, poi2.geometry, poi2.geometry.distance(poi.geometry))
                    clust_val = clust[index2] / 1000.0 
                    #if  discard_lower  and str(clust_val) in lookup and np.digitize(poi[use_key], ranges) < lookup[str(clust_val)]:
                    #    continue
                    neighbours.append(index2)
            #print(poi.geometry)
            #x = poi.geometry.buffer(0.011) #.unary_union
            #print(gdf["geometry"])
            #neighbours = gdf["geometry"].intersection(x)
            #inds = gdf[~neighbours.is_empty]
            #print(index, poi)
            #print(inds, len(inds), len(clust))
            #print(len(neighbours), "HERE")
            if len(neighbours) < 1:
                continue
            clusters = clust[neighbours]
            clusters_index = [np.nan]*n_clusters
            #print(np.unique(clusters))
            clusts = np.unique(clusters).astype(np.int32)
            #print(clusts, len(clusters_index))
            for c in clusts:
                clusters_index[c] = (c in clusters)
            hist_data.append(np.array(clusters_index))
            good_inds = np.where(np.isfinite(clusters_index))
            bad_inds = np.where(np.isnan(clusters_index))
            #print(good_inds, "HERE")
            #for j in range(len(clusters_index)):
            hist_data[-1][good_inds] = poi[use_key]
            hist_data[-1][bad_inds] = -1
            #for j in good_inds[0]:
            #    if clusters_index[j]:
            #        hist_data[-1][j] = poi[use_key]
            #    else:
            #        hist_data[-1][j] = -1

        final_hist_data.extend(hist_data)
    fnl = np.array(final_hist_data, dtype=np.float32)
    #print(fnl.shape)
    if  fnl.ndim < 2:
        return [[] for _ in range(len(ranges)-1)]
    fnl = np.swapaxes(fnl, 0,1)
    #print(fnl.shape, fnl.max())
    ranges[-1] = max(ranges[-1], global_max)
    algal = [[] for _ in range(len(ranges)-1)]
    for i in range(fnl.shape[0]):
        hist, bins = np.histogram(fnl[i], bins=ranges, density=False)
        if max(hist) < 1:
            continue
        #print(fnl[i], ranges)
        mx1 = np.argmax(hist)
        if mx1 == 0:
            sm = np.sum(hist[1:])
            if sm >= hist[0]:
                mx1 = np.argmax(hist[1:])
        algal[mx1].append(i / 1000.0)
        print(bins, hist,i)
        plt.ylim(0, 50)
        plt.bar([k*2 for k in range(len(bins[:-1]))],hist, width=1, linewidth=1, align="center")
        plt.show()
        plt.savefig(os.path.join(output_dir, "TEST_HIST_" + str(i) + ".png"))
        plt.clf()
    print("HERE FINAL", algal)
    return algal
    


    #for p in ra nge(len(ranges)):
    #    plt.hist(fnl, 5, density=True, histtype='bar', stacked=True, color = CMAP_COLORS[0:n_clusters+1], label=range(0,101), range=ranges[p]) 
    #    plt.savefig("TEST_HIST_" + str(p) + ".png")

def insitu_hab_to_tif(filename, **kwargs):

    print(filename)
    insitu_df = None
    if '.xlsx' in filename:
        insitu_df = pd.read_excel(filename)
    else:
        insitu_df = pd.read_csv(filename)
    # Format Datetime Stamp
    insitu_df['Datetime'] = pd.to_datetime(insitu_df['Sample Date'])
    insitu_df.set_index('Datetime')

    # Shorten Karenia Column Name
    insitu_df.rename(columns={"Karenia brevis abundance (cells/L)":'Karenia'}, inplace=True)

    insitu_df['Karenia'] = np.log(insitu_df['Karenia'])
    insitu_df.replace([np.inf, -np.inf], -10.0, inplace=True)
    ## Subset to Time Window and Depth  of Interest
    #insitu_subTime = insitu_df[(insitu_df['Sample Date'] > '2018-07-01') &
    #                       (insitu_df['Sample Date'] < '2019-03-01')]
    #insitu_subDepth = insitu_subTime[insitu_subTime['Sample Depth (m)'] < 1]
 
    uniques = np.unique(insitu_df['Datetime'].values)
    for date in uniques:
        subset = insitu_df[(insitu_df['Datetime'] == date)]
        print(subset.head)
        subset.drop(['Sample Date', 'Datetime', 'Sample Depth (m)', 'County'], inplace=True, axis=1)
        gdf = gpd.GeoDataFrame(subset, geometry=gpd.GeoSeries.from_xy(subset['Longitude'], subset['Latitude']), crs=4326)
        interp = "nearest"
        if subset.count()["Karenia"] >= 8:
            interp = "cubic"
        elif subset.count()["Karenia"] >= 4:
            interp = "linear"
        else:
            interp = "nearest"

        shp_fname = filename + np.datetime_as_string(date) + ".shp"
        gdf.to_file(shp_fname)

        rst_fname = filename + np.datetime_as_string(date) + ".tif"
         
        gdf.plot() # first image hereunder

        geotif_file = "/tmp/raster.tif"
 
        try:
            out_grd = make_geocube(
                vector_data=gdf,
                measurements=["Karenia"],
                output_crs=4326,
                fill = 0.0,
                interpolate_na_method=interp,
                resolution=(-0.01,0.01) #, interpolate_na_method="nearest", TODO parameterize
            )
        except:
            out_grd = make_geocube(
                vector_data=gdf,
                measurements=["Karenia"],
                output_crs=4326,
                fill = 0.0,
                interpolate_na_method="nearest",
                resolution=(-0.01,0.01)
            )
        out_grd["Karenia"].rio.to_raster(rst_fname)



# Define function to interpolate sampled data to grid.
def interp_to_grid(u, yc, xc, new_lats, new_lons):
    new_points = np.stack(np.meshgrid(new_lons, new_lats), axis = 2).reshape((new_lats.size * new_lons.size, 2))
    z = griddata((yc, xc), u, (new_points[:,1], new_points[:,0]), method = 'cubic', fill_value = np.nan)
    out = z.reshape((new_lats.size, new_lons.size))
    return out 


def sort_by_array(arr, arr_sort, dim):
    """
    Sort array(s) by the values of another array along a dimension
    
    Parameters
    ----------
    arr : xarray DataArray or Dataset
        The field(s) to be sorted
    dim : str
        Dimension in arr to sort along
    """

    SORT_DIM = "i"

    sort_axis = arr_sort.get_axis_num(dim)
    sort_inds = arr_sort.argsort(axis=sort_axis)
    # Replace dim with dummy dim and drop coordinates since
    # they're no longer correct
    print(sort_inds)
    print(sort_inds.rename({dim: SORT_DIM}))
    sort_inds = sort_inds.rename({dim: SORT_DIM}).drop(SORT_DIM)
    
    return arr.isel({dim: sort_inds})


def wrapped_log_and(x1,x2):
    return np.logical_and(x1, x2)

def apply_log_and_along_axis(x1,x2,axis=-1):
    return np.apply_along_axis(wrapped_log_and, x1, x2)


def read_sif(trop_fname, **kwargs):

    ds = xr.open_dataset(trop_fname, decode_times=False)

    if "ungridded" in trop_fname: 
        sif_df = ds.to_dataframe()
        sif_df[sif_df['sif'] < -900.0] = np.nan
        if "start_lat" in kwargs and "end_lat" in kwargs and "start_lon" in kwargs and "end_lon" in kwargs:
            sif_df = sif_df[(sif_df['lat'] >= kwargs["start_lat"]-2) & (sif_df['lat'] <= kwargs["end_lat"]+2) &\
                (sif_df['lon'] >= kwargs["start_lon"]-2) & (sif_df['lon'] <= kwargs["end_lon"]+2)]

        gdf = gpd.GeoDataFrame(sif_df, geometry=gpd.GeoSeries.from_xy(sif_df["lon"], sif_df["lat"]), crs=4326)
        if(gdf.empty):
            return None
        sif_raw = make_geocube(vector_data=gdf,
                measurements=["sif"],
                output_crs=4326,
                resolution=(-0.063,0.063)#,
                #fill = np.nan,
                #interpolate_na_method="cubic"
            )

        #plt.imshow(sif_raw["sif"])
        #plt.savefig(trop_fname + ".IMG.FIRST.png")
        #plt.clf()   
 
        sif_raw = sif_raw.rename({'x': 'lon','y': 'lat'})
        #sif_raw.rio.set_spatial_dims("lat", "lon", inplace=True)
        #sif_raw = sif_raw.sortby("lat").sortby("lon")
        #tmp =  np.flipud(sif_raw["sif"].to_numpy()) #np.flipud(np.fliplr(sif_raw["sif"].to_numpy()))
        #tmp = np.fliplr(tmp)
        tmp = sif_raw["sif"].to_numpy()
        sif_raw = xr.DataArray(tmp,\
            coords={'lat': sif_raw.lat.data,'lon': sif_raw.lon.data},\
            dims=["lat", "lon"]) 

        #plt.imshow(sif_raw)
        #plt.savefig(trop_fname + ".IMG.SECOND.png")
        #plt.clf()
        print("RAW", sif_raw) 

    return sif_raw


def clip_and_save_trop(fnames, **kwargs):
    
    for fname in fnames:
        print(fname[1])
        sif_raw = read_sif(fname[1], **kwargs)
        if sif_raw is None:
            continue
        out_fname = fname[1].replace("ungridded", "clipped_c_fla")
        print(out_fname)
        sif_raw.to_netcdf(out_fname)


def read_oc_and_trop(fnames, **kwargs):

    print(fnames)
    trop_fname = fnames[1]
    oc_fname = fnames[0]

    dat1 = None
    oc_lat = None
    oc_lon = None
    print(oc_fname, trop_fname)
    if "VIIRS" in oc_fname or "viirs" in oc_fname:
        dat1 = read_viirs_oc(oc_fname, **kwargs)
    elif "MODIS" in oc_fname or "modis" in oc_fname:
        dat1 = read_modis_oc(oc_fname, **kwargs)
    elif "S3" in oc_fname or "s3" in oc_fname:
        dat1 = read_s3_oc(oc_fname, **kwargs)
    print("HERE DAT", dat1.shape) 

    loc = read_oc_geo(oc_fname)
    lat = loc[0]
    lon = loc[1]
    if "start_lat" in kwargs and "end_lat" in kwargs and "start_lon" in kwargs and "end_lon" in kwargs:
        print(lat.shape, lon.shape, dat1.shape)
        inds1 = np.where((lat >= kwargs["start_lat"]) & (lat <= kwargs["end_lat"]))
        inds2 = np.where((lon >= kwargs["start_lon"]) & (lon <= kwargs["end_lon"]))
        lat = lat[inds1]
        lon = lon[inds2]
 
        nind1, nind2 = np.meshgrid(inds2, inds1)

    print("HERE DAT", dat1.shape)
    sif_raw = read_sif(trop_fname, **kwargs)
    if sif_raw is None:
        return None

    # Convert to xarray
    data_vrs = {}
    dat1[np.where(dat1 < -90)] = np.nan
    oc_xr = xr.Dataset(coords=dict(lon=(["x"],lon),\
        lat=(["y"],lat)))

    for i in range(dat1.shape[0]):
        oc_xr["oc_" + str(i)] = (['y', 'x'],  dat1[i])
        print("INIT OC STATS", i, np.nanmin(dat1[i]), np.nanmean(dat1[i]), np.nanmax(dat1[i]), np.nanstd(dat1[i]))


    oc_xr = oc_xr.rename({'x': 'lon','y': 'lat'})

    plt.clf()
    oc_xr2 = [] 
    for i in range(dat1.shape[0]):
        oc_xr2.append(oc_xr["oc_" + str(i)])

    #sif_aoi = sif_raw["sif"]
    #sif_aoi = sif_raw.sel(**{"lat":slice(lat.min(), lat.max()), "lon":slice(lon.min(), lon.max())})
    #sif_aoi = sif_aoi.interp_like(oc_xr2[0], method="nearest") #, assume_sorted=True, method_non_numeric="none")
    #plt.imshow(sif_aoi)
    #plt.savefig(trop_fname + ".LOW_RES.IMG.png")
    #plt.clf()
    #print(lat.shape, lon.shape)
    #plt.imshow(sif_raw)
    #plt.savefig(trop_fname + ".NO_MASK.IMG.png")
    #plt.clf()
    sif_aoi_2 = sif_raw.interp_like(oc_xr2[0], method="nearest") #, assume_sorted=True, method_non_numeric="none")
    sif_aoi_2 = sif_aoi_2.reindex(lat=lat, lon=lon, method="nearest", tolerance=0.063)
 
    #plt.imshow(sif_aoi_2)
    #plt.savefig(trop_fname + ".REINDEXED_NO_MASK.IMG.png")
    #plt.clf()

    #tmp =  np.flipud(sif_aoi.to_numpy()) #np.flipud(np.fliplr(sif_raw["sif"].to_numpy()))
    #tmp = np.fliplr(tmp)
    #sif_aoi = xr.DataArray(tmp,\
    #coords={'lat': sif_aoi.lat.data,'lon': sif_aoi.lon.data},\
    #dims=["lat", "lon"])
 
    print(sif_aoi_2)
    print(oc_xr)
    #sif_aoi = sif_aoi.dropna(dim="lat", how="all")
    #sif_aoi = sif_aoi.dropna(dim="lon", how="all")
    print(sif_aoi_2) 
    print(oc_xr2[0])

    sif_aoi_2.rio.set_spatial_dims("lat", "lon", inplace=True)
    sif_aoi_2.rio.write_crs("epsg:4326", inplace=True)
    tmp2 = None
    if "mask_shp" in kwargs:
        mask = gpd.read_file(kwargs["mask_shp"], crs="epsg:4326")
        sif_temp = sif_aoi_2.rio.clip(mask.geometry.apply(mapping), mask.crs, drop=False)
        print("HERE ", np.unique(sif_temp))
        tmp2 = sif_temp.isnull().to_numpy().astype(np.bool_)
    tmp1 = None

    if "mask_oceans" in kwargs:
        land_temp = ocean_basins_50.mask(sif_aoi_2)
        print(sif_aoi_2)
        print(land_temp, land_temp.min(), land_temp.max())
        land_temp = land_temp.rename({'lon': 'x','lat': 'y'})
        #final_mask = ((land_temp.isnull()) & (sif_temp.isnull())).to_numpy()
        tmp1 = land_temp.isnull().to_numpy().astype(np.bool_)

    final_mask = None
    if tmp1 is not None and tmp2 is not None:
        final_mask = xr.apply_ufunc(np.logical_and, tmp1, tmp2, vectorize=True, dask="parallelized", input_core_dims=[[],[]], output_core_dims=[[],[]])
    elif tmp1 is not None:
        final_mask = tmp1
    elif tmp2 is not None:
        final_mask = tmp2
                
    sif_aoi = np.array(sif_aoi_2.to_numpy())
    print(sif_aoi.shape)
    #sif_aoi = np.fliplr(np.rot90(sif_aoi, 1))
    #sif_aoi = np.rot90(sif_aoi, 3)
    #tmp = np.array(oc_xr2)
    #print(tmp.shape, sif_aoi.shape)
    #sif_aoi = cv2.resize(sif_aoi, (tmp.shape[2], tmp.shape[1]), interpolation=cv2.INTER_CUBIC)
    #print(sif_aoi.shape, (tmp.shape[1], tmp.shape[2]))
    if final_mask is not None:
        print(final_mask)
        sif_aoi[final_mask] = -999999

    #sif_aoi = np.swapaxes(sif_aoi, 0,1)
    print("HERE SHAPE", dat1.shape, sif_aoi)
    dat2 = oc_xr2
    #for i in range(dat1.shape[0]):
    #    dat2.append(oc_xr2["oc_" + str(i)].to_numpy())
    #    print("FINAL OC STATS", i, np.nanmin(dat2), np.nanmean(dat2), np.nanmax(dat2), np.nanstd(dat2))
    dat2 = np.array(dat2)
    dat2[np.where(np.isnan(dat2))] = -999999
    #sif_aoi = cv2.resize(sif_aoi, (dat2.shape[1], dat2.shape[2]), interpolation=cv2.INTER_CUBIC)
    sif_aoi = np.expand_dims(sif_aoi, axis=0)
    #sif_aoi = np.flip(sif_aoi, axis=1)
    #plt.imshow(np.squeeze(sif_aoi), vmin=-10, vmax=10)
    #plt.savefig(trop_fname + ".IMG.png")
    #plt.clf()
    #plt.imshow(np.squeeze(dat2[0,:,:]), vmin=-10, vmax=10)
    #plt.savefig(trop_fname + ".DAT.IMG.png")
    #plt.clf()
    print(sif_aoi.shape, dat2.shape)
    dat2 = np.concatenate((dat2, sif_aoi), axis=0)     #(dat2,np.swapaxes(sif_aoi, 1,2)), axis=0)
    dat2[np.where(np.isnan(dat2))] = -999999
    print("END OC READ", dat2.shape, dat2.min(), dat2.max(), dat2.mean(), dat2.std())
    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat2 = dat2[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat2

 

def read_trop_mod_xr(flename, **kwargs):

    print(flename)
    sif_raw = xr.open_dataset(flename, engine="netcdf4").sortby("time")
 
    sif_temp = sif_raw
    if "start_time" in  kwargs and "end_time" in kwargs:
        sif_temp = sif_raw.sel(time=slice(kwargs["start_time"], kwargs["end_time"]))
    if "start_lat" in kwargs and "end_lat" in kwargs and "start_lon" in kwargs and "end_lon" in kwargs:
        sif_temp = sif_temp.sel(**{'lon' : slice(kwargs["start_lon"], kwargs["end_lon"]), 'lat': slice(kwargs["start_lat"], kwargs["start_lat"])})
 
    vrs = ['nflh', 'aot_869', 'angstrom', 'sif', 'chlor_a', 'chl_ocx'] 
    print(kwargs.keys()) 
    if "vars" in  kwargs:
        vrs = kwargs["vars"] 



    data1  = []
    for i in range(len(vrs)):
        var = vrs[i]
        x = sif_temp.variables[var]
        if var == "sif":
           x = x.to_numpy()
           print(x.shape)
           x = np.moveaxis(x, 2, 0) 
           x[np.where(np.isnan(x))] = -999999
        else:
            valid_min = x.attrs["valid_min"]
            valid_max = x.attrs["valid_max"]
            x = x.to_numpy()
            x[np.where(np.isnan(x))] = -999999
            inds = np.where(x < valid_min - 0.00000000005)
            x[inds] = -999999
            inds = np.where(x > valid_max - 0.00000000005)
            x[inds] = -999999
        data1.append(x)
        print(x.min(), x.max(), var)
        print(x.shape)
    return np.array(data1)


def read_trop_mod_xr_geo(flename, **kwargs):

    print(flename)
    sif_raw = xr.open_dataset(flename, engine="netcdf4").sortby("time")

    sif_temp = sif_raw
    if "start_time" in  kwargs and "end_time" in kwargs:
        sif_temp = sif_raw.sel(time=slice(kwargs["start_time"], kwargs["end_time"]))
    if "start_lat" in kwargs and "end_lat" in kwargs and "start_lon" in kwargs and "end_lon" in kwargs:
        sif_temp = sif_temp.sel(**{'lon' : slice(kwargs["start_lon"], kwargs["end_lon"]), 'lat': slice(kwargs["start_lat"], kwargs["start_lat"])})


    print(sif_temp.variables["time"].min())
    print(sif_temp.variables["time"].max())
    vrs = ["lat", "lon"]
    data1  = []
    for i in range(len(vrs)):
        var = vrs[i]
        x = sif_temp.variables[var]
        valid_min = x.attrs["valid_min"]
        valid_max = x.attrs["valid_max"]
        x = x.to_numpy()
        x[np.where(np.isnan(x))] = -999999
        inds = np.where(x < valid_min - 0.00000000005)
        x[inds] = -999999
        inds = np.where(x > valid_max - 0.00000000005)
        x[inds] = -999999
        data1.append(x)
        print(x.min(), x.max(), var)
        print(x.shape)
    lat = data1[0]
    lon = data1[1]
    longr, latgr = np.meshgrid(lon, lat)
    print(longr, latgr)
    geo = np.array([latgr, longr])
    print(geo.shape)
    return geo



def read_trop_l1b(filenames, **kwargs):
    data1 = None
    bands = kwargs["bands"]
    for i in range(len(filenames)):
        x = Dataset(filenames[i])
        group_name = "BAND" + str(bands[i]) + "_RADIANCE"
        if data1 is None:
            data1 = x.groups[group_name].groups["STANDARD_MODE"].groups["OBSERVATIONS"].variables["radiance"][:] 
        else:
            np.concatenate((data1, x.groups[group_name].groups["STANDARD_MODE"].groups["OBSERVATIONS"].variables["radiance"][:]), axis=3)
        del x
    data1 = np.log(np.squeeze(data1[0,:,:,:]))
    print(data1.min(), data1.max())
    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                data1 = data1[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
  
    return data1

 
#TODO HERE We are only using BD5 and BD6, which have same footprint, will need to collocate/resample if using other bands
#Will hack to only use BD5 files here, for now
def read_trop_l1b_geo(filename, **kwargs):
    data1 = []
    vrs = ["latitude", "longitude"]
    print(filename)
    x = Dataset(filename)
    for i in range(len(vrs)):
        print(vrs[i])
        dat = x.groups["BAND5_RADIANCE"].groups["STANDARD_MODE"].groups["GEODATA"].variables[vrs[i]][:]
        dat = np.squeeze(dat[0,:,:])
        data1.append(dat)
    dat = np.array(data1)

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]

    return dat
 

def read_geo_nc_ungridded(fname, **kwargs):
    print(fname)
    dat = Dataset(fname)
    lat = dat.variables['lat'][:]
    lon = dat.variables['lon'][:]
    longr, latgr = np.meshgrid(lon, lat)
    geo = np.array([latgr, longr])
    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        geo = geo[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return geo


def get_scaler(scaler_name, cuda=True):
	if scaler_name == "standard":
		return StandardScaler(), True
	elif scaler_name == "standard_dask":
		return DaskStandardScaler(), True
	elif scaler_name == "maxabs":
		return MaxAbsScaler(), True
	elif scaler_name == "sparse_standard":
		return StandardScaler(with_mean=False), True
	elif scaler_name == "sparse_standard_dask":
		return DaskStandardScaler(with_mean=False), True
	else:
		return None, True

def read_gtiff_generic_geo(flename, **kwargs):
    latLon = get_lat_lon(flename)    

    print("HERE IN UTILS", latLon.shape)

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
            latLon = latLon[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]

    return latLon


def get_lat_lon(fname):
    # open the dataset and get the geo transform matrix
    ds = gdal.Open(fname)
    xoffset, px_w, rot1, yoffset, px_h, rot2 = ds.GetGeoTransform()
    dataArr = ds.ReadAsArray()
    #print(dataArr.shape, "LONLAT")
 
    ind = 1
    if dataArr.ndim == 3:
        latLon = np.zeros((dataArr.shape[1], dataArr.shape[2], 2))
        ind = 2
    else:
        latLon = np.zeros((dataArr.shape[0], dataArr.shape[1], 2))

    # get CRS from dataset 
    crs = osr.SpatialReference()
    crs.ImportFromWkt(ds.GetProjectionRef())

    # create lat/long crs with WGS84 datum
    crsGeo = osr.SpatialReference()
    crsGeo.ImportFromEPSG(4326) # 4326 is the EPSG id of lat/long crs 
    t = osr.CoordinateTransformation(crs, crsGeo)
    for j in range(dataArr.shape[ind]):
        for k in range(dataArr.shape[ind-1]):
            posX = px_w * j + rot1 * k + (px_w * 0.5) + (rot1 * 0.5) + xoffset
            posY = px_h * j + rot2 * k + (px_h * 0.5) + (rot2 * 0.5) + yoffset

            (lon, lat, z) = t.TransformPoint(posX, posY)
            latLon[k,j,1] = lon
            latLon[k,j,0] = lat
    return latLon

def genLatLon(fnames):

    for i in range(len(fnames)):
        fname = fnames[i]
        latLon = get_lat_lon(fname)

        outFname = fname + ".lonlat.zarr"
        print(outFname)
        zarr.save(outFname, latLon)


def combine_modis_gtiffs_laads(file_list):
    for i in range(len(file_list)):
        data1 = []
        for j in range(len(file_list[i])):
            fn = file_list[i][j]
            dat = gdal.Open(fn)
            band = dat.GetRasterBand(1).ReadAsArray()
            band[np.where(band > 65535)] = -9999
            band[np.where(band < -0.0000000005)] = -9999
            data1.append(band)
        dat = np.array(data1)
        fn = os.path.join(file_list[i][0] + "Full_Bands.zarr")
        zarr.save(fn, dat)
        genLatLon([file_list[i][0]])


def read_emas_master_hdf(fname, **kwargs):

    ds=SD.SD(fname)
    r = ds.select('CalibratedData')
    attrs = r.attributes(full=1)
    scale_factor = attrs["scale_factor"][0]
    fill = attrs["_FillValue"][0]
    dat = r.get()
 
    inds = np.where(dat == fill)
    dat = dat.astype(np.float32)
    dat[inds] = -999999
     
    inds = np.where(dat > -999999)
    for i in range(dat.shape[1]):
        dat[:,i,:] = dat[:,i,:] * scale_factor[i]

    dat = dat.astype(np.float32)

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]


    dat = np.swapaxes(dat, 0,1)
    print(dat.shape, "HERE")
    return dat


def read_emas_master_hdf_geo(fname, **kwargs):

    ds=SD.SD(fname)
    r = ds.select('PixelLatitude')
    attrs = r.attributes(full=1)
    fill = attrs["_FillValue"][0]
    dat = r.get()
    inds = np.where(dat == fill)
    dat = dat.astype(np.float32)
    dat[inds] = -999999

    r2 = ds.select('PixelLongitude')
    attrs2 = r2.attributes(full=1)
    fill2 = attrs2["_FillValue"][0]
    dat2 = r2.get()
    inds2 = np.where(dat2 == fill)
    dat2 = dat2.astype(np.float32)
    dat2[inds2] = -999999


    dat = np.array([dat2,dat], dtype=np.float32)

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat = dat[:,kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]


    print(dat.shape, "HERE")
    return dat




def read_mspi(fname, **kwargs):

        data_fields = OrderedDict([
           #("355nm_band" ,{"Data" : ["I"], "QC" : [["I.mask"]]}),
           #("380nm_band" , {"Data" : ["I"], "QC" : [["I.mask"]]}),
           ("445nm_band" , {"Data" : ["I"], "QC" : [["I.mask"]]}),
           #("470nm_band" , {"Data" : ["I", "IPOL"], "QC" : [["I.mask"], ["Q.mask", "U.mask"]]}),
           ("470nm_band" , {"Data" : ["I"], "QC" : [["I.mask"]]}),
           ("555nm_band" , {"Data" : ["I"], "QC" : [["I.mask"]]}),
           ("660nm_band" , {"Data" : ["I"], "QC" : [["I.mask"]]}),
           ("865nm_band" , {"Data" : ["I"], "QC" : [["I.mask"]]}),
           #("660nm_band" , {"Data" : ["I", "IPOL"], "QC" : [["I.mask"], ["Q.mask", "U.mask"]]}),
           #("865nm_band" , {"Data" : ["I", "IPOL"], "QC" : [["I.mask"], ["Q.mask", "U.mask"]]}),
           ("935nm_band" , {"Data" : ["I"], "QC" : [["I.mask"]]})])

        f = h5py.File(fname, 'r')
        data = []
        mask = []
        for band, band_data in data_fields.items():
                print(band)
                for i in range(len(band_data["Data"])):
                        dat = f['HDFEOS']['GRIDS'][band]['Data Fields'][band_data["Data"][i]][:]
                        for j in range(len(band_data["QC"][i])):
                                data.append(dat)
                                print(i, j, dat.shape)
                                mask.append(f['HDFEOS']['GRIDS'][band]['Data Fields'][band_data["QC"][i][j]][:])


        data = np.array(data)
        mask = np.array(mask)
        print(data.shape, mask.shape)
        inds = np.where(np.any(mask == 0, axis = 0) == True)
        print(mask[:,inds[0], inds[1]])
        fill = -999999
        data[:,inds[0], inds[1]] = fill
 
        if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
            data = data[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
        return data


def read_mspi_geo(fname, **kwargs):

        data_fields = OrderedDict([
           #("355nm_band" ,{"Data" : ["I"], "QC" : [["I.mask"]]}),
           #("380nm_band" , {"Data" : ["I"], "QC" : [["I.mask"]]}),
           ("445nm_band" , {"Data" : ["I"], "QC" : [["I.mask"]]}),
           #("470nm_band" , {"Data" : ["I", "IPOL"], "QC" : [["I.mask"], ["Q.mask", "U.mask"]]}),
           ("470nm_band" , {"Data" : ["I"], "QC" : [["I.mask"]]}),
           ("555nm_band" , {"Data" : ["I"], "QC" : [["I.mask"]]}),    
           ("660nm_band" , {"Data" : ["I"], "QC" : [["I.mask"]]}),
           ("865nm_band" , {"Data" : ["I"], "QC" : [["I.mask"]]}),
           #("660nm_band" , {"Data" : ["I", "IPOL"], "QC" : [["I.mask"], ["Q.mask", "U.mask"]]}),
           #("865nm_band" , {"Data" : ["I", "IPOL"], "QC" : [["I.mask"], ["Q.mask", "U.mask"]]}),
           ("935nm_band" , {"Data" : ["I"], "QC" : [["I.mask"]]})])

        f = h5py.File(fname, 'r')
        print(fname)
        geo = []
        #geo.append(f['HDFEOS']['GRIDS']['Ancillary']['Data Fields']['XDim'][:])
        x = f['HDFEOS']['GRIDS']['Ancillary']['Data Fields']['Longitude'][:].astype(np.float32)
        y = f['HDFEOS']['GRIDS']['Ancillary']['Data Fields']['Latitude'][:].astype(np.float32)
        geo = [x,y]
        #geo.append(f['HDFEOS']['GRIDS']['Ancillary']['Data Fields']['YDim'][:])
        print(geo[0].shape)
        geo = np.array(geo)
        print(geo.shape)

 
        if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
            geo = geo[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
        return geo


def read_s1(files, **kwargs):
 
    dat = read_gtiff_multifile_generic(files, **kwargs)
    dat = np.flipud(dat)

    for i in range(dat.shape[0]):
        #dat[i] = lee_filter(dat[i], 5)
        dat[i] = 10*np.log(dat[i])
        print(dat.shape, dat[i].min(), dat[i].max(), dat[i].mean(), dat[i].std())
     

    #if "clip" in kwargs and kwargs["clip"]:
    #    dat = np.clip(dat, 1e-3, 1)

    return dat

def read_uavsar(in_fps, desc_out=None, type_out=None, search_out=None, **kwargs):
    """
    Reads UAVSAR data. 

    Args:
        in_fps (list(string) or string):  list of strings (each file will be treated as a separate channel)
                                          or string of data file paths
        desc_out (optional): if specified, is set to the annotation description of the files converted 
        type_out (optional): if specified, is set to the filetype of the files converted 
        search_out (optional): if specified, is set to the search keyword used to search the .ann file
        kwargs:
            ann_fps (list(string) or string): list of or string of UAVSAR annotation file paths,
                                          ann files will be automatically matched to data files
            pol_modes (list(string)) (optional): list of allowed polarization modes 
                                                    to filter for (e.g. ['HHHH', 'HVHV', 'VVVV'])
            linear_to_dB (bool) (optional): convert linear amplitude units to decibels

    Returns:
        data: numpy array of shape (channels, lines, samples) 
              Complex-valued (unlike polarization) data will be split into separate phase and amplitude channels. 
    """


    if "ann_fps" in kwargs:
        ann_fps = kwargs["ann_fps"]
    else:
        raise Exception("No annotation files specified.")
    
    if "pol_modes" in kwargs:
        pol_modes = list(kwargs["pol_modes"])
    else:
        pol_modes = None
    if "linear_to_dB" in kwargs:
        linear_to_dB = kwargs["linear_to_dB"]
    else:
        linear_to_dB = False

    if isinstance(in_fps, str):
        in_fps = [in_fps]
    if isinstance(ann_fps, str):
        ann_fps = [ann_fps]
    
    data = []
    
    print("Reading UAVSAR files...")
    
    # Filter allowed polarization modes
    if pol_modes:
        tmp = []
        for fp in in_fps:
            if any(mode in os.path.basename(fp) for mode in pol_modes):
                tmp.append(fp)
        in_fps = tmp
    
    for fp in in_fps:
        
        # Locate file and matching annotation
        if not os.path.os.path.exists(fp):
            raise Exception(f"Failed to find file: {fp}")
        fname = os.path.basename(fp)
        id = "_".join(fname.split("_")[0:4])
        ann_fp = None
        for ann in ann_fps:
            if id in os.path.basename(ann):
                ann_fp = ann
        if not ann_fp:
            raise Exception(f"File {fname} does not have an associated annotation file.")
        
        print(f"file: {fp}")
        print(f"matching ann file: {ann_fp}")
    
        exts = fname.split('.')[1:]

        if len(exts) == 2:
            ext = exts[1]
            type = exts[0]
        elif len(exts) == 1:
            type = ext = exts[0]
        else:
            raise ValueError('Unable to parse extensions')
        
        # Check for compatible extensions
        if type == 'zip':
            raise Exception('Cannot convert zipped directories. Unzip first.')
        if type == 'dat' or type == 'kmz' or type == 'kml' or type == 'png' or type == 'tif':
            raise Exception(f"Cannot handle {type} products")
        if type == 'ann':
            raise Exception('Cannot convert annotation files.')
            
        # Check for slant range files and ancillary files
        anc = None
        if type == 'slope' or type == 'inc':
            anc = True

        # Read in annotation file
        desc = read_annotation(ann_fp)

        if 'start time of acquisition for pass 1' in desc.keys():
            mode = 'insar'
            raise Exception('INSAR data currently not supported.')
        else:
            mode = 'polsar'

        # Determine the correct file typing for searching data dictionary
        if not anc:
            if mode == 'polsar':
                if type == 'hgt':
                    search = type
                else:
                    polarization = os.path.os.path.basename(fp).split('_')[5][-4:]
                    if polarization == 'HHHH' or polarization == 'HVHV' or polarization == 'VVVV':
                            search = f'{type}_pwr'
                    else:
                        search = f'{type}_phase'
                    type = polarization

            elif mode == 'insar':
                if ext == 'grd':
                    if type == 'int':
                        search = f'grd_phs'
                    else:
                        search = 'grd'
                else:
                    if type == 'int':
                        search = 'slt_phs'
                    else:
                        search = 'slt'
                pass
        else:
            search = type

        # Pull the appropriate values from our annotation dictionary
        nrow = desc[f'{search}.set_rows']['value']
        ncol = desc[f'{search}.set_cols']['value']

        # Set up datatypes
        com_des = desc[f'{search}.val_frmt']['value']
        com = False
        if 'COMPLEX' in com_des:                                    
            com = True
        if com:
            dtype = np.complex64
        else:
            dtype = np.float32

        # Read in binary data
        dat = np.fromfile(fp, dtype = dtype)
        if com:
            dat = np.abs(dat)
            phase = np.angle(dat)
            
        # Change zeros and -10,000 to fillvalue and convert linear units to dB if specified
        fillvalue = -9999.0
        dat[dat==0] = fillvalue
        dat[dat==-10000] = fillvalue
                
        if linear_to_dB:
            dat = 10.0 * np.log10(dat)
            
        # Reshape it to match what the text file says the image is
        if type == 'slope':
            slopes = {}
            slopes['east'] = dat[::2].reshape(nrow, ncol)
            slopes['north'] = dat[1::2].reshape(nrow, ncol)
            dat = slopes
        else:
            slopes = None
            dat = dat.reshape(nrow, ncol)
            if com:
                phase = phase.reshape(nrow, ncol)
        
        # Apply 5x5 Lee Speckle Filter
        if not anc and type != 'hgt':
            if com:
                dat = lee_filter(np.real(dat), 5) + np.imag(dat)
            else:
                dat = lee_filter(dat, 5)
        data.append(dat)
        if com:
            data.append(phase)
            
        dat = None
        phase = None
    
    data = np.array(data)
    print(data.shape)
    if "clip" in kwargs and kwargs["clip"]:
        data = np.clip(data, 1e-3, 1)
    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        data = data[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    
    if search_out:
        search_out = search
    if desc_out:
        desc_out = desc
    if type_out:
        type_out = type
    
    return data


def read_annotation(ann_file):
    """
    Reads a UAVSAR annotation file.

    Args:
        ann_file: path to the annotation file

    Returns:
        data: a dictionary of the annotation's contents, 
              data[key] = {'value': value, 'units': units, 'comment': comment}
    """
    with open(ann_file) as fp:
        lines = fp.readlines()
        fp.close()
    data = {}

    # loop through the data and parse
    for line in lines:

        # Filter out all comments and remove any line returns
        info = line.strip().split(';')
        comment = info[-1].strip().lower()
        info = info[0]
        
        # Ignore empty strings
        if info and "=" in info:
            d = info.strip().split('=')
            name, value = d[0], d[1]
            name_split = name.split('(')
            key = name_split[0].strip().lower()
            
            # Isolate units encapsulated between '(' and ')'
            if len(name_split) > 1:
                lidx = name_split[-1].find('(') + 1
                ridx = name_split[-1].find(')')
                units = name_split[-1][lidx:ridx]
            else:
                units = None

            value = value.strip()

            # Cast the values that can be to numbers ###
            if value.strip('-').replace('.', '').isnumeric():
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)

            # Assign each entry as a dictionary with value and units
            data[key] = {'value': value, 'units': units, 'comment': comment}

    return data


def get_read_func(data_reader):
    if data_reader == "emit":
        return read_emit
    if data_reader == "emit_l2":
        return read_emit_l2
    if data_reader == "emit_geo":
        return read_emit_geo
    if data_reader == "misr_sim":
        return read_misr_sim
    if data_reader == "s2_gtiff":
        return read_s2_gtiff
    if data_reader == "goes_netcdf":
        return read_goes_netcdf
    if data_reader == "goes_netcdf_geo":
        return read_goes_netcdf_geo
    if data_reader == "s3_netcdf":
        return read_s3_netcdf     
    if data_reader == "s3_netcdf_geo":
        return read_s3_netcdf_geo
    if data_reader == "gtiff_multifile":
        return read_gtiff_multifile_generic   
    if data_reader == "landsat_gtiff":
        return read_gtiff_multifile_generic
    if data_reader == "s1_gtiff":
        return read_s1
    if data_reader == "gtiff":
        return read_gtiff_generic
    if data_reader == "aviris_gtiff":
        return read_gtiff_generic
    if data_reader == "gtiff_geo":
        return read_gtiff_generic_geo
    if data_reader == "numpy":
        return numpy_load
    if data_reader == "zarr_to_numpy":
        return numpy_from_zarr
    if data_reader == "torch":
        return torch_load
    if data_reader == "s6_netcdf":
        return read_s6_netcdf
    if data_reader == "s6_netcdf_geo":
        return read_s6_netcdf_geo
    if data_reader == "trop_mod_xr":
        return read_trop_mod_xr
    if data_reader == "trop_mod_xr_geo":
        return read_trop_mod_xr_geo
    if data_reader == "trop_l1b":
        return read_trop_l1b
    if data_reader == "trop_l1b_geo":
        return read_trop_l1b_geo
    if data_reader == "trop_nc":
        return read_trop_red_sif_nc
    if data_reader == "nc_ungrid_geo":
        return read_geo_nc_ungridded
    if data_reader == "uavsar":
        return read_uavsar
    if data_reader == "hysplit_netcdf":
        return read_hysplit_netcdf
    if data_reader == "hysplit_netcdf_geo":
        return read_hysplit_netcdf_geo
    if data_reader == "gk2a_netcdf":
        return read_gk2a_netcdf  
    if data_reader == "gk2a_netcdf_geo":
        return read_gk2a_netcdf_geo
    if data_reader == "modis_sr":
        return read_modis_sr
    if data_reader == "viirs_oc":
        return read_viirs_oc 
    if data_reader == "modis_oc":
        return read_modis_oc
    if data_reader == "s3_oc":
        return read_s3_oc
    if data_reader == "viirs_oc_geo":
        return read_oc_geo
    if data_reader == "modis_oc_geo":
        return read_oc_geo
    if data_reader == "s3_oc_geo":
        return read_oc_geo
    if data_reader == "pace_oc":
        return read_pace_oc
    if data_reader == "pace_oc_geo":
        return read_oc_geo
    if data_reader == "oc_and_trop":
        return read_oc_and_trop
    if data_reader == "mspi":
        return read_mspi
    if data_reader == "mspi_geo":
        return read_mspi_geo
    if data_reader == "emas_hdf":
        return read_emas_master_hdf
    if data_reader == "master_hdf":
        return read_emas_master_hdf
    if data_reader == "emas_master_hdf":
        return read_emas_master_hdf
    if data_reader == "emas_master_hdf_geo":
        return read_emas_master_hdf_geo
    if data_reader == "tempo_netcdf":
        return read_tempo_netcdf
    if data_reader == "tempo_netcdf_geo":
         return read_tempo_netcdf_geo
    if data_reader == "tempo_no2_netcdf":
        return read_tempo_no2_netcdf
    if data_reader == "tempo_no2_netcdf_geo":
         return read_tempo_no2_netcdf_geo
    if data_reader == "bps_benchmark":
         return read_bps_benchmark
    if data_reader == "burn_severity":
         return read_burn_severity_stacks
    if data_reader == "viirs_aero_mask":
        return read_viirs_aerosol_type
    if data_reader == "viirs_aero_mask_geo":
        return read_viirs_aerosol_type_geo
    if data_reader == "modis_aero_mask":
        return read_modis_aero_mask
    if data_reader == "modis_aero_mask_geo":
        return read_modis_aero_mask_geo


    return None
