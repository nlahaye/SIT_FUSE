"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""

from osgeo import gdal, ogr
import os
import argparse
from sit_fuse.utils import numpy_to_torch, read_yaml, get_read_func

def rasterize(nodata, vector_fn, pixel_size, epsg):
    print(vector_fn)
    source_ex = gdal.OpenEx(vector_fn)
    
    raster_fn = os.path.splitext(vector_fn)[0] + ".tif"


    gdal.Rasterize(raster_fn, source_ex, format='GTIFF', outputType=gdal.GDT_Byte, creationOptions=["COMPRESS=DEFLATE"], noData=nodata, initValues=nodata, xRes=pixel_size, yRes=-pixel_size, allTouched=False, burnValues=[1], outputSRS="EPSG:"+str(epsg))    


def rasterize_to_extent(nodata, vector_fn, raster_extent_fns):

    vec_ds = gdal.Open(vector_fn)
    lyr = vec_ds.GetLayer(1)   
    #lyr = gdal.OpenEx(vector_fn)
 
    for i in range(len(raster_extent_fns)):
        ras_ds = gdal.Open(raster_extent_fns[i])
        geot = ras_ds.GetGeoTransform()

        out_raster_fn = os.path.splitext(vector_fn)[0] + "_" + str(i) + ".tif"        
 

        drv_tiff = gdal.GetDriverByName("GTiff")
        chn_ras_ds = drv_tiff.Create(out_raster_fn, ras_ds.RasterXSize, ras_ds.RasterYSize, 1, gdal.GDT_Float32)
        chn_ras_ds.SetGeoTransform(geot)
        gdal.RasterizeLayer(chn_ras_ds, [1], lyr, burn_values = [1])
        chn_ras_ds.GetRasterBand(1).SetNoDataValue(nodata)
        chn_ras_ds.FlushCache()
        chn_ras_ds = None
  

def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    for i in range(len(yml_conf["vector_fn"])):
        #Run
        if "raster_extent_fns" in yml_conf:
            rasterize_to_extent(yml_conf["nodata"], yml_conf["vector_fn"][i], yml_conf["raster_extent_fns"][i])
        else:
            epsg = str(4326)
            if "epsg" in yml_conf.keys():
                epsg = yml_conf["epsg"][i]
            rasterize(yml_conf["nodata"], yml_conf["vector_fn"][i], yml_conf["pixel_size"], epsg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    main(args.yaml)



