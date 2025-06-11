"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
import numpy as np
from pyresample.geometry import AreaDefinition
from pyresample import area_config, bilinear, geometry, data_reduce, create_area_def, kd_tree
from sit_fuse.utils import numpy_to_torch, read_yaml, get_read_func
from osgeo import gdal, osr
import argparse
import os

def generate_gtif(yml_conf):

    data_reader_lo =  yml_conf["low_res"]["data"]["reader_type"]
    data_reader_kwargs_lo = yml_conf["low_res"]["data"]["reader_kwargs"]
    geo_data_reader_lo =  yml_conf["low_res"]["data"]["geo_reader_type"]
    geo_data_reader_kwargs_lo = yml_conf["low_res"]["data"]["geo_reader_kwargs"]
    lo_filenames = yml_conf["low_res"]["data"]["filenames"]
    lo_geoloc = yml_conf["low_res"]["data"]["geo_filenames"]
    lo_channel_dim = yml_conf["low_res"]["data"]["chan_dim"]
    lo_coord_dim = yml_conf["low_res"]["data"]["geo_coord_dim"]
    lo_lat_index = yml_conf["low_res"]["data"]["geo_lat_index"]
    lo_lon_index = yml_conf["low_res"]["data"]["geo_lon_index"]
    valid_max_lo = yml_conf["low_res"]["data"]["valid_max"]
    valid_min_lo = yml_conf["low_res"]["data"]["valid_min"]

    read_func_lo = get_read_func(data_reader_lo)
    read_func_geo_lo = get_read_func(geo_data_reader_lo)

    proj_id = yml_conf["fusion"]["projection_id"]
    output_files = yml_conf["output_files"]

    print(len(lo_geoloc) , len(lo_filenames))
    for i in range(len(lo_filenames)):
        lo_channel_dim = yml_conf["low_res"]["data"]["chan_dim"]
        lo_coord_dim = yml_conf["low_res"]["data"]["geo_coord_dim"]


        lo_dat = read_func_lo(lo_filenames[i], **data_reader_kwargs_lo).astype(np.float64)
        if len(lo_dat.shape) < 3:
            lo_dat = np.expand_dims(lo_dat, lo_channel_dim)

        print(lo_dat.shape) 
        lo_geo = read_func_geo_lo(lo_geoloc[i], **geo_data_reader_kwargs_lo).astype(np.float64)
        print(lo_geo.shape, lo_geoloc[i])


        print(lo_dat.shape, lo_geo.shape)
        if lo_channel_dim != 2:
            lo_dat = np.moveaxis(lo_dat, lo_channel_dim, 2)
        if lo_coord_dim != 2:
            lo_geo = np.moveaxis(lo_geo, lo_coord_dim, 2)
        lo_channel_dim = 2
        lo_coord_dim = 2
        if lo_geo.shape[0] > lo_dat.shape[0] or lo_geo.shape[1] > lo_dat.shape[1]:
            tmp = np.zeros((lo_geo.shape[0], lo_geo.shape[1], lo_dat.shape[2])) - 9999.0
            min_shape_1 = min(lo_geo.shape[0], lo_dat.shape[0])
            min_shape_2 = min(lo_geo.shape[1], lo_dat.shape[1])
            tmp[:min_shape_1, :min_shape_2,:] = lo_dat[:min_shape_1, :min_shape_2,:]
            lo_dat = tmp
        print(lo_dat.shape, lo_geo.shape)
        print(lo_dat.min(), lo_dat.max(), valid_min_lo, valid_max_lo)

        slc_lat_lo = [slice(None)] * lo_geo.ndim
        slc_lat_lo[lo_coord_dim] = slice(lo_lat_index, lo_lat_index+1)
        slc_lon_lo = [slice(None)] * lo_geo.ndim
        slc_lon_lo[lo_coord_dim] = slice(lo_lon_index, lo_lon_index+1)
    
        inds = np.where((lo_dat < valid_min_lo) | (lo_dat > valid_max_lo))
        lo_dat[inds] = -9999.0 

        xmin, ymin, xmax, ymax = [lo_geo[tuple(slc_lon_lo)].min(), lo_geo[tuple(slc_lat_lo)].min(),
                lo_geo[tuple(slc_lon_lo)].max(), lo_geo[tuple(slc_lat_lo)].max()]

        
        nx = lo_dat.shape[1]
        ny = lo_dat.shape[0]
        xres = (xmax - xmin) / float(nx)
        yres = (ymax - ymin) / float(ny)
        geotransform = (xmin, xres, 0, ymax, 0, -yres)


        toGeotiff(lo_dat, geotransform, output_files[i], proj_id)

def toGeotiff(output_image, gt, out_fname, proj_id):
 
  # create GDAL driver for writing Geotiff
  driver = gdal.GetDriverByName('GTiff')
 
 
  print("HERE", output_image.shape)
  # create Geotiff destination dataset for output
  dstds = driver.Create(out_fname,output_image.shape[1],output_image.shape[0],output_image.shape[2],gdal.GDT_Float32)
 
  # seto geotransform of output geotiff
  dstds.SetGeoTransform(gt)
 
  # set-up projection of output Geotiff
  srs = osr.SpatialReference()
  srs = osr.SpatialReference()
  srs.ImportFromEPSG(proj_id)
  srs = srs.ExportToWkt()
  dstds.SetProjection(srs)
 
  # write array data (1 band) ... set NoData value at 0.0
  # in output Geotiff
  print(output_image.shape)
 
  for i in range(output_image.shape[2]):
      dstds.GetRasterBand((i+1)).WriteArray(np.squeeze(output_image[:,:,i]))
      dstds.GetRasterBand((i+1)).SetNoDataValue(-9999.0)

  dstds=None

 

def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    generate_gtif(yml_conf)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    main(args.yaml)






