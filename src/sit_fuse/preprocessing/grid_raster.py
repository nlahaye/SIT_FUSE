
from osgeo import gdal
import numpy as np

from utils import read_yaml


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

def create_tile_bounds(minx, miny, maxx, maxy, n):
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
    outputs = []
    file_base = os.path.splitext(file_name)[0]
    driver = gdal.GetDriverByName('GTiff')
    dataset = gdal.Open(file_name)
    #band = dataset.GetRasterBand(1)
    transform = dataset.GetGeoTransform()

    extent = get_extent(dataset)

    cols = int(extent["cols"])
    rows = int(extent["rows"])
    minx = float(extent["minX"])
    maxx = float(extent["maxX"])
    miny = float(extent["minY"])
    maxy = float(extent["maxY"])
    width = maxx - minx
    height = maxy - miny

    tiles = create_tile_bounds(minx, miny, maxx, maxy, n)
    transform = dataset.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

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

        new_cols = i2-i1
        new_rows = j2-j1

        data = dataset.ReadAsArray(xoff=i1, yoff=j1, xsize=new_cols, ysize=new_rows)

        #print data

        new_x = xOrigin + i1*pixelWidth
        new_y = yOrigin - j1*pixelHeight

        print new_x, new_y

        new_transform = (new_x, transform[1], transform[2], new_y, transform[4], transform[5])

        output_file = file_base + "_" + str(tile_num) + ".tif"
        print(output_file)
        outputs.append(output_file)

        nchan = 1
        if data.ndim > 2:
            ncham = data.shape[0]

        dst_ds = driver.Create(output_file,
                               new_cols,
                               new_rows,
                               data.shape[0],
                               gdal.GDT_Float32)

        #writting output raster
        if data.ndim < 3:
            dst_ds.GetRasterBand(1).WriteArray( data )
        else:
            for c in range(len(nchan)):
                dst_ds.GetRasterBand(c).WriteArray( data[c,:,:] )

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
    
    return outputs


def run_split(yml_conf):
    
    n_tiles = yml_conf["n_tiles"]
    fnames = yml_conf["fnames"]
 
    total_outputs = []

    for fname in fnames:
        total_outputs.extend(split(fname, n_tiles))

    return sorted(total_outputs)

 
def run_split_outside(yml_fpath):

    yml_conf = read_yaml(yml_fpath)
    run_split(yml_conf)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    run_split_outside(args.yaml)
