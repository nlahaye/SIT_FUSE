import numpy as np
from pyresample.geometry import AreaDefinition
from pyresample import area_config, bilinear, geometry, data_reduce, create_area_def, kd_tree
from utils import numpy_to_torch, read_yaml, get_read_func
from osgeo import gdal, osr
import argparse
import os

def fuse_data(yml_conf):

    data_reader_hi =  yml_conf["high_res"]["data"]["reader_type"]
    data_reader_kwargs_hi = yml_conf["high_res"]["data"]["reader_kwargs"]
    geo_data_reader_hi =  yml_conf["high_res"]["data"]["geo_reader_type"]
    geo_data_reader_kwargs_hi = yml_conf["high_res"]["data"]["geo_reader_kwargs"]
    hi_filenames = yml_conf["high_res"]["data"]["filenames"]
    hi_geoloc = yml_conf["high_res"]["data"]["geo_filenames"]
    hi_channel_dim = yml_conf["high_res"]["data"]["chan_dim"]
    hi_coord_dim = yml_conf["high_res"]["data"]["geo_coord_dim"]
    hi_lat_index = yml_conf["high_res"]["data"]["geo_lat_index"]
    hi_lon_index = yml_conf["high_res"]["data"]["geo_lon_index"]
    valid_max_hi = yml_conf["high_res"]["data"]["valid_max"]
    valid_min_hi = yml_conf["high_res"]["data"]["valid_min"]

    read_func_hi = get_read_func(data_reader_hi)
    read_func_geo_hi = get_read_func(geo_data_reader_hi)    

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
    proj_description = yml_conf["fusion"]["description"]
    area_id = yml_conf["fusion"]["area_id"]
    projection = yml_conf["fusion"]["projection_proj4"]
 
    final_resolution = yml_conf["fusion"]["final_resolution"]
    projection_units = yml_conf["fusion"]["projection_units"]

    resample_radius = yml_conf["fusion"]["resample_radius"]
    resample_n_neighbors = yml_conf["fusion"]["resample_n_neighbors"]
    resample_n_procs = yml_conf["fusion"]["resample_n_procs"]
    resample_epsilon= yml_conf["fusion"]["resample_epsilon"]
    use_bilinear = yml_conf["fusion"]["use_bilinear"]
 
    output_files = yml_conf["output_files"]
 
    for i in range(len(lo_filenames)):
        hi_channel_dim = yml_conf["high_res"]["data"]["chan_dim"]
        hi_coord_dim = yml_conf["high_res"]["data"]["geo_coord_dim"]
        lo_channel_dim = yml_conf["low_res"]["data"]["chan_dim"]
        lo_coord_dim = yml_conf["low_res"]["data"]["geo_coord_dim"]


        if len(hi_filenames) > 0:
            hi_dat = read_func_hi(hi_filenames[i], **data_reader_kwargs_hi).astype(np.float64)
            if len(hi_dat.shape) < 3:
                hi_dat = np.expand_dims(hi_dat, hi_channel_dim)
                print(hi_dat.shape)

            hi_geo = read_func_geo_hi(hi_geoloc[i], **geo_data_reader_kwargs_hi).astype(np.float64)

            hi_dat = np.moveaxis(hi_dat, hi_channel_dim, 2)
            hi_geo = np.moveaxis(hi_geo, hi_coord_dim, 2)
            hi_channel_dim = 2
            hi_coord_dim = 2
            if hi_geo.shape[0] > hi_dat.shape[0] or hi_geo.shape[1] > hi_dat.shape[1]:
                tmp = np.zeros((hi_geo.shape[0], hi_geo.shape[1], hi_dat.shape[2])) - 9999.0
                tmp[:hi_dat.shape[0], :hi_dat.shape[1],:] = hi_dat
                hi_dat = tmp

            slc_lat_hi = [slice(None)] * hi_geo.ndim
            slc_lat_hi[hi_coord_dim] = slice(hi_lat_index, hi_lat_index+1)
            slc_lon_hi = [slice(None)] * hi_geo.ndim
            slc_lon_hi[hi_coord_dim] = slice(hi_lon_index, hi_lon_index+1)
            source_def_hi = geometry.SwathDefinition(lons=np.squeeze(hi_geo[tuple(slc_lon_hi)]), lats=np.squeeze(hi_geo[tuple(slc_lat_hi)]))
            hi_dat = np.ma.masked_where((hi_dat < valid_min_hi) | (hi_dat > valid_max_hi), hi_dat)

            #Assumes reader or preprocessor has defaulted bad values to -9999
            np.ma.set_fill_value(hi_dat, -9999.0)
 
        lo_dat = read_func_lo(lo_filenames[i], **data_reader_kwargs_lo).astype(np.float64)
        if len(lo_dat.shape) < 3:
            lo_dat = np.expand_dims(lo_dat, lo_channel_dim)

        print(lo_dat.shape) 
        lo_geo = read_func_geo_lo(lo_geoloc[i], **geo_data_reader_kwargs_lo).astype(np.float64)


        print(lo_dat.shape, lo_geo.shape)
        lo_dat = np.moveaxis(lo_dat, lo_channel_dim, 2)
        lo_geo = np.moveaxis(lo_geo, lo_coord_dim, 2)
        lo_channel_dim = 2
        lo_coord_dim = 2
        if lo_geo.shape[0] > lo_dat.shape[0] or lo_geo.shape[1] > lo_dat.shape[1]:
            tmp = np.zeros((lo_geo.shape[0], lo_geo.shape[1], lo_dat.shape[2])) - 9999.0
            tmp[:lo_dat.shape[0], :lo_dat.shape[1],:] = lo_dat
            lo_dat = tmp
        print(lo_dat.shape, lo_geo.shape)
        print(lo_dat.min(), lo_dat.max(), valid_min_lo, valid_max_lo)

        slc_lat_lo = [slice(None)] * lo_geo.ndim
        slc_lat_lo[lo_coord_dim] = slice(lo_lat_index, lo_lat_index+1)
        slc_lon_lo = [slice(None)] * lo_geo.ndim
        slc_lon_lo[lo_coord_dim] = slice(lo_lon_index, lo_lon_index+1)
        source_def_lo = geometry.SwathDefinition(lons=np.squeeze(lo_geo[tuple(slc_lon_lo)]), lats=np.squeeze(lo_geo[tuple(slc_lat_lo)]))


        if "lon_bounds" in yml_conf["fusion"].keys() and "lat_bounds" in yml_conf["fusion"].keys():
            area_extent = (yml_conf["fusion"]["lon_bounds"][0], yml_conf["fusion"]["lat_bounds"][0],
                yml_conf["fusion"]["lon_bounds"][1], yml_conf["fusion"]["lat_bounds"][1])
        else:
            area_extent = (lo_geo[tuple(slc_lon_lo)].min(), lo_geo[tuple(slc_lat_lo)].min(),
                lo_geo[tuple(slc_lon_lo)].max(), lo_geo[tuple(slc_lat_lo)].max())


        area_def = create_area_def(area_id, projection, area_extent=area_extent, resolution = final_resolution, units = projection_units)
        lonsa, latsa = area_def.get_lonlats()

        #Assumes reader or preprocessor has defaulted bad values to -9999
        lo_dat = np.ma.masked_where((lo_dat < valid_min_lo) | (lo_dat > valid_max_lo), lo_dat)
        np.ma.set_fill_value(lo_dat, -9999.0)

        if resample_n_procs > 1:
            from pyresample._spatial_mp import cKDTree_MP as kdtree_class
        else:
            kdtree_class = KDTree
 

        if len(hi_filenames) > 0:
            print("BEFORE RESAMPLING", hi_dat.shape, lo_dat.shape)
 
            if use_bilinear:
                resampler = bilinear.NumpyBilinearResampler(source_def_hi, area_def, resample_radius, neighbours=resample_n_neighbors) 

                resampler.get_bil_info(kdtree_class=kdtree_class, nprocs=resample_n_procs)
                result = resampler.get_sample_from_bil_info(hi_dat, fill_value=-9999.0, output_shape=None)

                #result = bilinear.resample_bilinear(hi_dat, source_def_hi, area_def, radius= resample_radius, neighbours= resample_n_neighbors, nprocs= resample_n_procs, fill_value = -9999.0)
            else:
                #radius_of_influence=5000, epsilon=1.5, nprocs=6, 
                result = kd_tree.resample_nearest(source_def_hi, hi_dat, area_def, radius_of_influence=resample_radius, epsilon=resample_epsilon, nprocs=resample_n_procs, fill_value = -9999)
            result = np.ma.masked_where((result < (min(valid_min_lo, valid_min_hi)-100.0)), result)
            np.ma.set_fill_value(result, -9999.0)
     
            print("BEFORE RESAMPLING2", hi_dat.shape, lo_dat.shape, result.shape)

        if use_bilinear: 
            resampler = bilinear.NumpyBilinearResampler(source_def_lo, area_def, resample_radius, neighbours=resample_n_neighbors)
            resampler.get_bil_info(kdtree_class=kdtree_class, nprocs=resample_n_procs)
            result2 = resampler.get_sample_from_bil_info(lo_dat, fill_value=-9999.0, output_shape=None)    
            #result2 = bilinear.resample_bilinear(lo_dat, source_def_lo, area_def, radius= resample_radius, neighbours= resample_n_neighbors, nprocs= resample_n_procs, fill_value = -9999.0)

        else:
            result2 = kd_tree.resample_nearest(source_def_lo, lo_dat, area_def, radius_of_influence=resample_radius, epsilon=resample_epsilon, nprocs=resample_n_procs, fill_value = -9999)
        result2 = np.ma.masked_where((result2 < (min(valid_min_lo, valid_min_hi)-100.0)), result2)
        np.ma.set_fill_value(result2, -9999.0)

        print("AFTER RESAMPLING2", lo_dat.shape, result2.shape, result2.min())
   
        datFinal = []
        print("MASKING", np.count_nonzero(result2.mask))
        if len(hi_filenames) > 0:
            print("MASKING", np.count_nonzero(result.mask))
 
            if len(result.shape) > 2:
                for q in range(result.shape[2]):
                        datFinal.append(result[:,:,q].filled())
            else:
                datFinal.append(result[:,:].filled())
            del result

        if len(result2.shape) > 2:
            for q in range(result2.shape[2]):
                    datFinal.append(result2[:,:,q].filled())
        else:
            datFinal.append(result2[:,:].filled())
        del result2
        datFinal = np.array(datFinal)

        print(datFinal.shape)
        if "tif" in os.path.splitext(output_files[i])[1]:
            toGeotiff(datFinal, area_def, output_files[i], proj_id)
        else:
            np.save(output_files[i], datFinal)
            locMapped = np.array([lonsa, latsa])
            locMapped = np.moveaxis(locMapped, 0, 2)
            np.save(output_files[i] + ".lonlat.npy", locMapped)

def toGeotiff(outputImage, areaDef, outFname, proj_id):
 
  # create GDAL driver for writing Geotiff
  driver = gdal.GetDriverByName('GTiff')
 
 
  # create Geotiff destination dataset for output
  dstds = driver.Create(outFname,outputImage.shape[2],outputImage.shape[1],outputImage.shape[0],gdal.GDT_Float32)
  gt = [ areaDef.area_extent[0],areaDef.pixel_size_x,0,\
  areaDef.area_extent[3],0,-areaDef.pixel_size_y]
 
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
  print(outputImage.shape)
 
  for i in range(outputImage.shape[0]):
      dstds.GetRasterBand((i+1)).WriteArray(np.squeeze(outputImage[i]))
      dstds.GetRasterBand((i+1)).SetNoDataValue(-9999.0)

  dstds=None

 

def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    fuse_data(yml_conf)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    main(args.yaml)






