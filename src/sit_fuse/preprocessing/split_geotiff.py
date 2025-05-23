import numpy as np
import dask.array as da
import xarray as xr

from pyresample.geometry import AreaDefinition
from pyresample import area_config, bilinear, geometry, data_reduce, create_area_def, kd_tree
from sit_fuse.utils import numpy_to_torch, read_yaml, get_read_func
from osgeo import gdal, osr
import argparse
import os



def get_extent(dataset):

    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
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


def split(file_name, n):
    raw_file_name = os.path.splitext(os.path.basename(file_name))[0]
    driver = gdal.GetDriverByName('GTiff')
    dataset = gdal.Open(file_name)
   
    transform = dataset.GetGeoTransform()

    extent = get_extent(dataset)

    cols = int(extent["cols"])
    rows = int(extent["rows"])

    print("Columns: ", cols)
    print("Rows: ", rows)

    minx = float(extent["minX"])
    maxx = float(extent["maxX"])
    miny = float(extent["minY"])
    maxy = float(extent["maxY"])

    width = maxx - minx
    height = maxy - miny
 
    output_path = raw_file_name
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("Width", width)
    print("height", height)


    tiles = create_tiles(minx, miny, maxx, maxy, n)
    transform = dataset.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    print(xOrigin, yOrigin)

    tile_num = 0
    for tile in tiles:

        minx = tile[0][0]
        maxx = tile[1][0]
        miny = tile[1][1]
        maxy = tile[0][1]

        p1 = (minx, maxy)
        p2 = (maxx, miny)

        i1 = int((p1[0] - xOrigin) / pixelWidth)
        j1 = int((yOrigin - p1[1])  / pixelHeight)
        i2 = int((p2[0] - xOrigin) / pixelWidth)
        j2 = int((yOrigin - p2[1]) / pixelHeight)

        print(i1, j1)
        print(i2, j2)

        new_cols = i2-i1
        new_rows = j2-j1

        data = dataset.ReadAsArray(i1, j1, new_cols, new_rows)

        #print data

        new_x = xOrigin + i1*pixelWidth
        new_y = yOrigin - j1*pixelHeight

        print(new_x, new_y)

        new_transform = (new_x, transform[1], transform[2], new_y, transform[4], transform[5])

        output_file_base = raw_file_name + "_" + str(tile_num) + ".tif"
        output_file = output_file_base

        print(data.shape)
        dst_ds = driver.Create(output_file,
                               new_cols,
                               new_rows,
                               data.shape[0],
                               gdal.GDT_Float32)
 
        #writting output raster
        for chan in range(0, data.shape[0]):
            dst_ds.GetRasterBand((chan+1)).WriteArray( data[chan])

        tif_metadata = {
            "minX": str(minx), "maxX": str(maxx),
            "minY": str(miny), "maxY": str(maxy)
        }
        dst_ds.SetMetadata(tif_metadata)

        #setting extension of output raster
        # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
        dst_ds.SetGeoTransform(new_transform)

        wkt = dataset.GetProjection()

        # setting spatial reference of output raster
        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt)
        dst_ds.SetProjection( srs.ExportToWkt() )

        #Close output raster dataset
        dst_ds = None

        tile_num += 1

    dataset = None



def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    fnames = yml_conf["fnames"]
    n_sq_tiles = yml_conf["n_sq_tiles"]

    for i in range(len(fnames)):
        split(fnames[i], n_sq_tiles)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    main(args.yaml)

