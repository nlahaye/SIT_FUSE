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
import cupy as cp
import xarray as xr
import dask.array as da
from netCDF4 import Dataset
from osgeo import osr, gdal



def sliding_window_view(x, window_shape, axis=None, *,
                        subok=False):
    window_shape = (tuple(window_shape)
                    if np.iterable(window_shape)
                    else (window_shape,))
    # first convert input to array, possibly keeping subclass
    x = cp.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError('`window_shape` cannot contain negative values')

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(f'Since axis is `None`, must provide '
                             f'window_shape for all dimensions of `x`; '
                             f'got {len(window_shape)} window_shape elements '
                             f'and `x.ndim` is {x.ndim}.')
    else:
        axis = np.core.numeric.normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(f'Must provide matching length window_shape and '
                             f'axis; got {len(window_shape)} window_shape '
                             f'elements and {len(axis)} axes elements.')

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError(
                'window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return cp.lib.stride_tricks.as_strided(x, strides=out_strides, shape=out_shape)


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
        data = data[:, :, bands]
        data = np.moveaxis(data, 2, chan_dim)

    return data


def cupy_load(filename, **kwargs):

    data = cp.load(filename)

    if "bands" in kwargs:
        bands = kwargs["bands"]
        chan_dim = kwargs["chan_dim"]

        data = cp.moveaxis(data, chan_dim, 2)
        data = data[:, :, bands]
        data = cp.moveaxis(data, 2, chan_dim)

    return data


def zarr_load(filename, **kwargs):
    return da.from_zarr(filename)


def numpy_from_zarr(filename, **kwargs):
    return np.array(zarr_load(filename).compute())


def cupy_from_zarr(filename, **kwargs):
    return cp.array(zarr_load(filename).compute())


def read_goes_netcdf(filenames, **kwargs):
    data1 = []
    for j in range(0, len(filenames)):
        f = Dataset(filenames[j])
        rad = f.variables['Rad'][:]
        f.close()
        f = None
        data1.append(rad)
    refShp = data1[3].shape
    for k in range(0, len(data1)):
        shp = data1[k].shape
        print(shp, refShp)
        if shp[0] != refShp[0] or shp[1] != refShp[1]:
            data1[k] = cv2.resize(
                data1[k], (refShp[1], refShp[0]), interpolation=cv2.INTER_CUBIC)
        print(data1[k].shape)
    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = np.array(data1)[:, kwargs["start_line"]:kwargs["end_line"],
                              kwargs["start_sample"]:kwargs["end_sample"]]
    print(dat.shape)
    return dat


def read_s3_netcdf(s3_dir, **kwargs):
    data1 = []
    bands = None
    if "bands" in kwargs:
        bands = kwargs["bands"]
    if os.path.isdir(s3_dir):
        for i in range(1, 22):
            if bands is None or i in bands:
                data_key = "Oa" + str(i).zfill(2) + "_radiance"
                fname = os.path.join(s3_dir, data_key + ".nc")
                f = Dataset(fname)
                rad = f.variables[data_key]
                data = rad[:]
                valid_data_ind = np.where(
                    (data >= rad.valid_min) & (data <= rad.valid_max))
                invalid_data_ind = np.where(
                    (data < rad.valid_min) & (data > rad.valid_max))
                #data[valid_data_ind] = data[valid_data_ind] * rad.scale_factor + rad.add_offset
                data[invalid_data_ind] = -9999.0
                data1.append(data)
    dat = np.array(data1)
    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"],
                  kwargs["start_sample"]:kwargs["end_sample"]]
    return dat


def read_s3_netcdf_geo(s3_dir, **kwargs):
    data1 = []
    if os.path.isdir(s3_dir):
        fname = os.path.join(s3_dir, "geo_coordinates.nc")
        f = Dataset(fname)

        lat = f.variables["latitude"]
        data = lat[:]
        valid_data_ind = np.where(
            (data >= lat.valid_min) & (data <= lat.valid_max))
        invalid_data_ind = np.where(
            (data < lat.valid_min) & (data > lat.valid_max))
        #data[valid_data_ind] = data[valid_data_ind] * lat.scale_factor
        data[invalid_data_ind] = -9999.0
        data1.append(data)

        lon = f.variables["longitude"]
        data = lon[:]
        valid_data_ind = np.where(
            (data >= lon.valid_min) & (data <= lon.valid_max))
        invalid_data_ind = np.where(
            (data < lon.valid_min) & (data > lon.valid_max))
        #data[valid_data_ind] = data[valid_data_ind] * lon.scale_factor
        data[invalid_data_ind] = -9999.0
        data1.append(data)

    dat = np.array(data1)
    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"],
                  kwargs["start_sample"]:kwargs["end_sample"]]
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
        data = data[:, kwargs["start_line"]:kwargs["end_line"],
                    kwargs["start_sample"]:kwargs["end_sample"]]
    if "log" in kwargs and kwargs["log"]:
        data = np.log(data)

    print("HERE", data.min(), data.max())
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
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"],
                  kwargs["start_sample"]:kwargs["end_sample"]]
    print(dat.shape)
    return dat


# TODO config for AVIRIS - scale 0.0001 valid_min = 0 and Fill = -9999
def read_gtiff_generic(flename, **kwargs):
    dat = gdal.Open(flename, gdal.GA_ReadOnly).ReadAsArray()
    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"],
                  kwargs["start_sample"]:kwargs["end_sample"]]
    return dat


def read_trop_mod_xr(flename, **kwargs):

    print(flename)
    sif_raw = xr.open_dataset(flename, engine="netcdf4").sortby("time")

    sif_temp = sif_raw
    if "start_time" in kwargs and "end_time" in kwargs:
        sif_temp = sif_raw.sel(time=slice(
            kwargs["start_time"], kwargs["end_time"]))
    if "start_lat" in kwargs and "end_lat" in kwargs and "start_lon" in kwargs and "end_lon" in kwargs:
        sif_temp = sif_temp.sel(
            **{'lon': slice(kwargs["start_lon"], kwargs["end_lon"]), 'lat': slice(kwargs["start_lat"], kwargs["start_lat"])})

    vrs = ['nflh', 'aot_869', 'angstrom', 'sif', 'chlor_a', 'chl_ocx']
    print(kwargs.keys())
    if "vars" in kwargs:
        vrs = kwargs["vars"]

    print("HERE ", vrs)

    data1 = []
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
    if "start_time" in kwargs and "end_time" in kwargs:
        sif_temp = sif_raw.sel(time=slice(
            kwargs["start_time"], kwargs["end_time"]))
    if "start_lat" in kwargs and "end_lat" in kwargs and "start_lon" in kwargs and "end_lon" in kwargs:
        sif_temp = sif_temp.sel(
            **{'lon': slice(kwargs["start_lon"], kwargs["end_lon"]), 'lat': slice(kwargs["start_lat"], kwargs["start_lat"])})

    vrs = ["lat", "lon"]
    data1 = []
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
            np.concatenate(
                (data1, x.groups[group_name].groups["STANDARD_MODE"].groups["OBSERVATIONS"].variables["radiance"][:]), axis=3)
        del x
    data1 = np.log(np.squeeze(data1[0, :, :, :]))
    print(data1.min(), data1.max())
    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        data1 = data1[:, kwargs["start_line"]:kwargs["end_line"],
                      kwargs["start_sample"]:kwargs["end_sample"]]

    return data1


# TODO HERE We are only using BD5 and BD6, which have same footprint, will need to collocate/resample if using other bands
# Will hack to only use BD5 files here, for now
def read_trop_l1b_geo(filename, **kwargs):
    data1 = []
    vrs = ["latitude", "longitude"]
    print(filename)
    x = Dataset(filename)
    print("HERE")
    for i in range(len(vrs)):
        print(vrs[i])
        dat = x.groups["BAND5_RADIANCE"].groups["STANDARD_MODE"].groups["GEODATA"].variables[vrs[i]][:]
        dat = np.squeeze(dat[0, :, :])
        data1.append(dat)
    dat = np.array(data1)

    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"],
                  kwargs["start_sample"]:kwargs["end_sample"]]

    return dat


def get_scaler(scaler_name, cuda=True):
    if cuda:
        from cuml.preprocessing import StandardScaler, MaxAbsScaler
    else:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

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


# TODO worldview
def get_read_func(data_reader):
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
    if data_reader == "cupy":
        return cupy_load
    if data_reader == "zarr_to_cupy":
        return cupy_from_zarr
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
    # TODO return BCDP reader
    return None
