import numpy as np
from pyresample.geometry import AreaDefinition
from pyresample import area_config, bilinear, geometry, data_reduce, create_area_def
from utils import numpy_to_torch, read_yaml, get_read_func
import argparse

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

    output_files = yml_conf["output_files"]

    for i in range(len(lo_filenames)):

        if len(hi_filenames) > 0:
            hi_dat = read_func_hi(hi_filenames[i], **read_func_kwargs_hi).astype(np.float64)
            hi_geo = read_func_geo_hi(hi_geoloc[i], **geo_data_reader_kwargs_hi).astype(np.float64)
            hi_dat = np.moveaxis(hi_dat, hi_channel_dim, 2)
            slc_lat_hi = [slice(None)] * hi_geo.ndim
            slc_lat_hi[hi_coord_dim] = slice(hi_lat_index, hi_lat_index+1)
            slc_lon_hi = [slice(None)] * hi_geo.ndim
            slc_lon_hi[hi_coord_dim] = slice(hi_lon_index, hi_lon_index+1)
            source_def_hi = geometry.SwathDefinition(lons=hi_geo[tuple(slc_lon_hi)], lats=hi_geo[tuple(slc_lat_hi)])
            hi_dat = np.ma.masked_where((hi_dat < valid_max_hi) | (hi_dat > valid_max_hi), hi_dat)

            #Assumes reader or preprocessor has defaulted bad values to -9999
            np.ma.set_fill_value(hi_dat, -9999.0)
 
        lo_dat = read_func_lo(lo_filenames[i], **data_reader_kwargs_lo).astype(np.float64)

        lo_geo = read_func_geo_lo(lo_geoloc[i], **geo_data_reader_kwargs_lo).astype(np.float64)

        lo_dat = np.moveaxis(lo_dat, lo_channel_dim, 2)

        slc_lat_lo = [slice(None)] * lo_geo.ndim
        slc_lat_lo[lo_coord_dim] = slice(lo_lat_index, lo_lat_index+1)
        slc_lon_lo = [slice(None)] * lo_geo.ndim
        slc_lon_lo[lo_coord_dim] = slice(lo_lon_index, lo_lon_index+1)
        print(lo_geo[tuple(slc_lon_lo)].shape)
        source_def_lo = geometry.SwathDefinition(lons=np.squeeze(lo_geo[tuple(slc_lon_lo)]), lats=np.squeeze(lo_geo[tuple(slc_lat_lo)]))

        area_extent = (lo_geo[tuple(slc_lon_lo)].min(), lo_geo[tuple(slc_lat_lo)].min(), lo_geo[tuple(slc_lon_lo)].max(), lo_geo[tuple(slc_lat_lo)].max())

        print(area_extent)

        area_def = create_area_def(area_id, projection, area_extent=area_extent, resolution = final_resolution, units = projection_units)
        lonsa, latsa = area_def.get_lonlats()

        #Assumes reader or preprocessor has defaulted bad values to -9999
        lo_dat = np.ma.masked_where((lo_dat < valid_max_lo) | (lo_dat > valid_max_lo), lo_dat)
        np.ma.set_fill_value(lo_dat, -9999.0)


        if len(hi_filenames) > 0:
            print("BEFORE RESAMPLING", hi_dat.shape, lo_dat.shape)
  
            result = bilinear.resample_bilinear(hi_dat, source_def_hi, area_def, radius= resample_radius, neighbours= resample_n_neighbors, nprocs= resample_n_procs, fill_value = -9999.0)
            result = np.ma.masked_where((result < 0.0000000005), result)
            np.ma.set_fill_value(result, -9999.0)
     
            print("BEFORE RESAMPLING2", hi_dat.shape, lo_dat.shape, result.shape)

        result2 = bilinear.resample_bilinear(lo_dat, source_def_lo, area_def, radius= resample_radius, neighbours= resample_n_neighbors, nprocs= resample_n_procs, fill_value = -9999.0)
        result2 = np.ma.masked_where(result2 < 0.0000000005, result2)
        np.ma.set_fill_value(result2, -9999.0)

        print("AFTER RESAMPLING2", lo_dat.shape, result2.shape)

        datFinal = []
        print("MASKING", np.count_nonzero(datLowRes))
        if len(hi_filenames) > 0:
            print("MASKING", np.count_nonzero(result.mask))
 
            for q in range(result.shape[2]):
                    datFinal.append(result[:,:,q].filled())
            del result

        for q in range(result2.shape[2]):
                datFinal.append(result2[:,:,q].filled())
        del result2
        datFinal = np.array(datFinal)

        print(datFinal.shape)
        if "tif" in os.path.splitext(output_files[i])[1]:
            toGeotiff(datFinal, area_def, output_files[i])
        else:
            np.save(output_files[i], datFinal)
            locMapped = np.array([lonsa, latsa])
            locMapped = np.moveaxis(locMapped, 0, 2)
            np.save(output_files[i] + ".lonlat.npy", locMapped)

def toGeotiff(outputImage, areaDef, outFname):
 
  # create GDAL driver for writing Geotiff
  driver = gdal.GetDriverByName('GTiff')
 
 
  # create Geotiff destination dataset for output
  dstds = driver.Create(outFname,outputImage.shape[1],outputImage.shape[0],1,gdal.GDT_Float32)
  gt = [ areaDef.area_extent[0],areaDef.pixel_size_x,0,\
  areaDef.area_extent[3],0,-areaDef.pixel_size_y]
 
  # seto geotransform of output geotiff
  dstds.SetGeoTransform(gt)
 
  # set-up projection of output Geotiff
  srs = osr.SpatialReference()
  srs.ImportFromProj4(areaDef.proj4_string)
  srs.SetProjCS(areaDef.proj_id)
  srs = srs.ExportToWkt()
  dstds.SetProjection(srs)
 
  # write array data (1 band) ... set NoData value at 0.0
  # in output Geotiff
  dstds.GetRasterBand(1).WriteArray(outputImage)
  dstds.GetRasterBand(1).SetNoDataValue(0.0)
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






