import torch
import yaml
import cv2
import os
import numpy as np
from netCDF4 import Dataset
import dask.array as da


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
    return np.load(filename)

def zarr_load(filename, **kwargs):
    return da.from_zarr(filename)

def numpy_from_zarr(filename, **kwargs):
    return np.array(zarr_load(filename).compute())
 
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
            data1[k] = cv2.resize(data1[k], (refShp[1],refShp[0]), interpolation=cv2.INTER_CUBIC)
        print(data1[k].shape)
    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:	
        dat = np.array(data1)[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    print(dat.shape)
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
		dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
	return data


def read_gtiff_multifile_generic(files, **kwargs):
    data1 = []
    for j in range(0, len(files)):
        dat = gdal.Open(files[j]).ReadAsArray()
        data1.append(dat)
    dat = np.array(data1)
    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        dat = dat[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    return dat

 
#TODO config for AVIRIS - scale 0.0001 valid_min = 0 and Fill = -9999
def read_gtiff_generic(file, **kwargs): 
	dat = gdal.Open(filenames[i], gdal.GA_ReadOnly).ReadAsArray()


#TODO worldview

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
	if data_reader == "zarr_to_numpy":
		return numpy_from_zarr
	if data_reader == "torch":
		return torch_load
	if data_reader == "s6_netcdf":
		return read_s6_netcdf
	#TODO return BCDP reader
	return None

