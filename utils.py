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
import numpy as np
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
from CMAP import CMAP, CMAP_COLORS
from glob import glob

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from dask_ml.preprocessing import StandardScaler as DaskStandardScaler

def torch_to_numpy(trch):
        return trch.numpy()

def numpy_to_torch(npy):
        return torch.from_numpy(npy)


def read_yaml(fpath_yaml):
    yml_conf = None
    with open(fpath_yaml) as f_yaml:
        yml_conf = yaml.load(f_yaml, Loader=yaml.FullLoader)
    return yml_conf


def torch_load(filename, **kwargs):
    return torch.load(filename)

def numpy_load(filename, **kwargs):

    data = np.load(filename)

    if "bands" in kwargs:
        bands = kwargs["bands"]
        chan_dim = kwargs["chan_dim"]        
        
        data = np.moveaxis(data, chan_dim, 2)
        data = data[:,:,bands]
        data = np.moveaxis(data, 2, chan_dim)

    return data

def zarr_load(filename, **kwargs):
    return da.from_zarr(filename)

def numpy_from_zarr(filename, **kwargs):
    return np.array(zarr_load(filename).compute())


def read_emit(filename, **kwargs):

    ds = Dataset(filename)
    dat = ds.variables['radiance'][:]

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat


def read_misr_sim(filename, **kwargs):
 
    ds = Dataset(filename)
    dat = ds.variables['rad'][:]

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
                dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat    
 
 
def read_goes_netcdf(filenames, **kwargs):
    data1 = []
    for j in range(0, len(filenames)):
        f = Dataset(filenames[j])
        
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
        f.close()
        f = None
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
                                rad = f.variables[data_key]
                                data = rad[:]
                                valid_data_ind = np.where((data >= rad.valid_min) & (data <= rad.valid_max))
                                invalid_data_ind = np.where((data < rad.valid_min) & (data > rad.valid_max))
                                #data[valid_data_ind] = data[valid_data_ind] * rad.scale_factor + rad.add_offset
                                data[invalid_data_ind] = -9999.0
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


def read_gtiff_multifile_generic(files, **kwargs):
    print(files)
    data1 = []
    for j in range(0, len(files)):
        dat1 = gdal.Open(files[j]).ReadAsArray()
        if len(data1) == 0 or len(dat1.shape) == 2:
            data1.append(dat1)
        else:
            data1[0] = np.concatenate((data1[0], dat1), axis=0)
    dat = np.array(data1)
    if len(dat.shape) == 4:
        dat = np.squeeze(dat)
    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    print(dat.shape)
    return dat

 
#TODO config for AVIRIS - scale 0.0001 valid_min = 0 and Fill = -9999
def read_gtiff_generic(flename, **kwargs): 
	dat = gdal.Open(flename, gdal.GA_ReadOnly).ReadAsArray()
	if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
		if len(dat.shape) == 3:
			dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
		else:
			dat = dat[kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
	return dat



#TODO generalize pieces for other tasks
def insitu_hab_to_multi_hist(insitu_fname, start_date, end_date, clusters_dir, n_clusters, radius_degrees, ranges, global_max, files_test, files_train):
    print(insitu_fname)
    insitu_df = pd.read_excel(insitu_fname)
    # Format Datetime Stamp
    insitu_df['Datetime'] = pd.to_datetime(insitu_df['Sample Date'])
    insitu_df.set_index('Datetime')

    # Shorten Karenia Column Name
    insitu_df.rename(columns={"Karenia brevis abundance (cells/L)":'Karenia'}, inplace=True)
    
    insitu_df = insitu_df[(insitu_df['Sample Date'] >= start_date) & (insitu_df['Sample Date'] <= end_date)] 

    uniques = np.unique(insitu_df['Sample Date'].values)

    #TODO
    #subset by date - start and end day of SIF 
    #tie date to cluster dat

    final_hist_data = []
    ind = 1
    for date in uniques:
        #    find associated cluster
        input_fname = os.path.join(os.path.dirname(files_train[0]), "sif_finalday_" + str(ind))
        ind = ind + 1
        clust_fname = os.path.join(clusters_dir, "file")
        dat_ind = -1

        dat_train = False
        dat_test = False

        try:
            dat_ind = files_train.index(input_fname)
            dat_train = True
        except ValueError:
            dat_ind = -1

        if dat_ind == -1:
            try:
                dat_ind = files_test.index(input_fname)
            except ValueError:
                continue
       
        if dat_train:
            clust_fname = clust_fname + str(dat_ind) + "_output.data.clustering_" + str(n_clusters) + "clusters.zarr.tif"
        else:
            clust_fname = clust_fname + str(dat_ind) + "_output_test.data.clustering_" + str(n_clusters) + "clusters.zarr.tif"
        
        if not os.path.exists(clust_fname):
            clust_fname = clust_fname + "f"
            if not os.path.exists(clust_fname):
                continue
  
        clust = gdal.Open(clust_fname)
        lonLat = get_lat_lon(clust_fname)
        clust = clust.ReadAsArray()
               
 
        lat = lonLat[:,:,0].reshape(clust.shape[0]*clust.shape[1])
        lon = lonLat[:,:,1].reshape(clust.shape[0]*clust.shape[1])
        clust = clust.reshape(clust.shape[0]*clust.shape[1])
        inds_clust = np.where(clust >= 0)
        lat = lat[inds_clust]
        lon = lon[inds_clust]
        clust = clust[inds_clust]

        gdf = gpd.GeoDataFrame(clust, geometry=gpd.GeoSeries.from_xy(lon, lat), crs=4326)
        subset = insitu_df[(insitu_df['Sample Date'] == date)]
        gdf_insitu = gpd.GeoDataFrame(subset["Karenia"], geometry=gpd.GeoSeries.from_xy(subset['Longitude'], subset['Latitude']), crs=4326)

        #gdf_proj = gdf.to_crs({"init": "EPSG:3857"})
        #gdf_insitu_proj = gdf_insitu.to_crs({"init": "EPSG:3857"})     
 
        dists = []
        count = -1
        hist_data = []
        for index, poi in gdf_insitu.iterrows():
            count = count + 1
            neighbours = []
            for index2, poi2 in gdf.iterrows():
                if abs(poi2.geometry.distance(poi.geometry)) < radius_degrees:
                    neighbours.append(index2)
            #print(poi.geometry)
            #x = poi.geometry.buffer(0.011) #.unary_union
            #print(gdf["geometry"])
            #neighbours = gdf["geometry"].intersection(x)
            #inds = gdf[~neighbours.is_empty]
            #print(index, poi)
            #print(inds, len(inds), len(clust))
            if len(neighbours) < 1:
                continue
            clusters = clust[neighbours]
            clusters_index = []
            for c in range(n_clusters+1):
                clusters_index.append((c in clusters))
            hist_data.append(clusters_index)
            for j in range(len(clusters_index)):
                if clusters_index[j]:
                    hist_data[-1][j] = poi["Karenia"]
                else:
                    hist_data[-1][j] = -1

        final_hist_data.extend(hist_data)
    fnl = np.array(final_hist_data)
    fnl = np.swapaxes(fnl, 0,1)
    print(fnl.shape, fnl.max())
    ranges[-1] = max(ranges[-1], global_max)
    algal = [[] for _ in range(len(ranges)-1)]
    for i in range(fnl.shape[0]):
        hist, bins = np.histogram(fnl[i], bins=ranges, density=False)
        mx1 = np.argmax(hist)
        if mx1 == 0:
            sm = np.sum(hist[1:])
            if sm >= hist[0]:
                mx1 = np.argmax(hist[1:])
        algal[mx1].append(i)
        print(bins, hist,i)
        plt.ylim(0, 50)
        plt.bar([k*2 for k in range(len(bins[:-1]))],hist, width=1, linewidth=1, align="center")
        plt.show()
        plt.savefig("TEST_HIST_" + str(i) + ".png") 
        plt.clf()
    print(algal)    

    


    #for p in ra nge(len(ranges)):
    #    plt.hist(fnl, 5, density=True, histtype='bar', stacked=True, color = CMAP_COLORS[0:n_clusters+1], label=range(0,101), range=ranges[p]) 
    #    plt.savefig("TEST_HIST_" + str(p) + ".png")

def insitu_hab_to_tif(filename, **kwargs):

    print(filename)
    insitu_df = pd.read_excel(filename)
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
 
    uniques = np.unique(insitu_df['Sample Date'].values)
    for date in uniques:
        subset = insitu_df[(insitu_df['Sample Date'] == date)]
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
    longr, latgr = np.meshgrid(lat, lon)
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

def get_lat_lon(fname):
    # open the dataset and get the geo transform matrix
    ds = gdal.Open(fname)
    xoffset, px_w, rot1, yoffset, px_h, rot2 = ds.GetGeoTransform()
    dataArr = ds.ReadAsArray()

    lonLat = np.zeros((dataArr.shape[0], dataArr.shape[1], 2))

    # get CRS from dataset 
    crs = osr.SpatialReference()
    crs.ImportFromWkt(ds.GetProjectionRef())

    # create lat/long crs with WGS84 datum
    crsGeo = osr.SpatialReference()
    crsGeo.ImportFromEPSG(4326) # 4326 is the EPSG id of lat/long crs 
    t = osr.CoordinateTransformation(crs, crsGeo)
    for j in range(dataArr.shape[1]):
        for k in range(dataArr.shape[0]):
            posX = px_w * j + rot1 * k + (px_w * 0.5) + (rot1 * 0.5) + xoffset
            posY = px_h * j + rot2 * k + (px_h * 0.5) + (rot2 * 0.5) + yoffset

            (lon, lat, z) = t.TransformPoint(posX, posY)
            lonLat[k,j,1] = lon
            lonLat[k,j,0] = lat
    return lonLat


def read_uavsar(in_fps, **kwargs):
    """
    Reads UAVSAR data. 
    Complex-valued (unlike polarization) data will be split into separate phase and amplitude channels. 

    Args:
        in_fps (list(string) or string):  list of strings (each file will be treated as a separate channel)
                                          or string of data file paths
        ann_fps (list(string) or string): list of or string of UAVSAR annotation file paths,
                                          ann files will be automatically matched to data files
        pol_modes (list(string)) (optional): list of allowed polarization modes 
                                             to filter for (e.g. ['HHHH', 'HVHV', 'VVVV'])
        linear_to_dB (bool) (optional): convert linear amplitude units to decibels

    Returns:
        (data, desc, type, search): tuple containing information about the file

        data: image array of floats or complex values, or dict containing two arrays of slopes (north, east)
        desc: the dictionary generated from the annotation file
        type: the type of inputted file
        search: the search term for relevant values inside the annotation dictionary
    """

    from preprocessing.misc_utils import lee_filter

    if "ann_fps" in kwargs:
        ann_fps = kwargs["ann_fps"]
    
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
        
        # Apply 5x5 Lee Speckle Filter
        if not anc and type != 'hgt':
            if com:
                dat = lee_filter(np.real(dat), 5) + np.imag(dat)
            else:
                lee_filter(dat, 5)

        data.append(dat)
        if com:
            data.append(phase)
            
        dat = None
        phase = None
    
    data = np.array(data)
    if "clip" in kwargs and kwargs["clip"]:
        data = np.clip(data, 1e-3, 1)
    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        data = data[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
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
    if data_reader == "misr_sim":
        return read_misr_sim
    if data_reader == "goes_netcdf":
        return read_goes_netcdf
    if data_reader == "s3_netcdf":
        return read_s3_netcdf     
    if data_reader == "s3_netcdf_geo":
        return read_s3_netcdf_geo
    if data_reader == "gtiff_multifile":
        return read_gtiff_multifile_generic   
    if data_reader == "landsat_gtiff":
        return read_gtiff_multifile_generic
    if data_reader == "s1_gtiff":
        return read_gtiff_multifile_generic
    if data_reader == "gtiff":
        return read_gtiff_generic
    if data_reader == "aviris_gtiff":
        return read_gtiff_generic
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
    if data_reader == "nc_ungrid_geo":
        return read_geo_nc_ungridded
    if data_reader == "uavsar":
        return read_uavsar
    if data_reader == "hysplit_netcdf":
        return read_hysplit_netcdf
    if data_reader == "hysplit_netcdf_geo":
        return read_hysplit_netcdf_geo
  
 
    #TODO return BCDP reader
    return None
