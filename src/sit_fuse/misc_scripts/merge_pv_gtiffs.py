import numpy as np
import dask.array as da
import xarray as xr
import glob

from pyresample.geometry import AreaDefinition
from pyresample import area_config, bilinear, geometry, data_reduce, create_area_def, kd_tree
from sit_fuse.utils import numpy_to_torch, read_yaml, get_read_func
from osgeo import gdal, osr
import argparse
import os



def get_extent(dataset):

    cols = dataset.RasterYSize
    rows = dataset.RasterXSize
    transform = dataset.GetGeoTransform()
    minx = transform[0]
    maxx = transform[0] + cols * transform[1] + rows * transform[2]

    miny = transform[3] + cols * transform[4] + rows * transform[5]
    maxy = transform[3]

    return {
            "minX": str(minx), "maxX": str(maxx),
            "minY": str(miny), "maxY": str(maxy),
            "cols": str(cols), "rows": str(rows)
            }

def create_tiles(minx, miny, maxx, maxy, n):
    width = maxx - minx
    height = maxy - miny

    matrix = []

    for j in range(n, 0, -1):
        for i in range(0, n):

            ulx = minx + (width/n) * i # 10/5 * 1
            uly = miny + (height/n) * j # 10/5 * 1

            lrx = minx + (width/n) * (i + 1)
            lry = miny + (height/n) * (j - 1)
            matrix.append([[ulx, uly], [lrx, lry]])

    return matrix



def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    fname = yml_conf["fname"]
    n_sq_tiles = yml_conf["n_sq_tiles"]
    full_extent_geotiff = yml_conf["extent_gtiff"]
    tile_basename = yml_conf["tile_basename"]
    tile_ext =  yml_conf["tile_ext"]
    
 
    dat = gdal.Open(full_extent_geotiff)
    geoTransform = dat.GetGeoTransform()
    metadata = dat.GetMetadata()
    wkt = dat.GetProjection()

    extent = get_extent(dat)

    rows =  np.zeros(n_sq_tiles).astype(np.int32)
    cols = np.zeros(n_sq_tiles).astype(np.int32)


    print(extent)

    outp = np.zeros((int(extent["cols"]),int(extent["rows"])), dtype=np.int16)
    nx = outp.shape[1]
    ny = outp.shape[0]

    cntr = 0
    for i in range(n_sq_tiles):
        for j in range(n_sq_tiles):
            print(tile_basename + str(cntr) + tile_ext)
            fglob = glob.glob(tile_basename + str(cntr) + tile_ext)
            print(fglob)
            if len(fglob) < 1:

                dat_tmp = gdal.Open("/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pv_polygons_" + str(cntr) + ".tif")
                arr_tmp =dat_tmp.ReadAsArray()

                extent = get_extent(dat_tmp)
                print(rows[j], rows[j]+int(extent["cols"]), arr_tmp.shape, outp.shape, i, j)
                print(cols[i], cols[i]+ int(extent["rows"]), arr_tmp.shape, outp.shape, i, j)
                rows[j] = int(rows[j]) + int(extent["cols"])
                cols[i] = int(cols[i]) + int(extent["rows"])

            
                del dat_tmp
                del arr_tmp
                cntr = cntr + 1
                continue

            fname_tmp = fglob[0]

            local_arr = gdal.Open(fname_tmp).ReadAsArray()
            print(rows[j], rows[j]+local_arr.shape[0], local_arr.shape, outp.shape, i, j) 
            print(cols[i], cols[i]+local_arr.shape[1], local_arr.shape, outp.shape, i, j)
            outp[rows[j]:rows[j]+local_arr.shape[0],cols[i]:cols[i]+local_arr.shape[1]] = local_arr

            rows[j] = rows[j] + local_arr.shape[0]
            cols[i] = cols[i] + local_arr.shape[1]

            del local_arr
            cntr = cntr + 1

    out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(geoTransform)
    out_ds.SetMetadata(metadata)
    out_ds.SetProjection(wkt)
    out_ds.GetRasterBand(1).WriteArray(outp)
    out_ds.FlushCache()
    out_ds = None
 
     


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    main(args.yaml)

