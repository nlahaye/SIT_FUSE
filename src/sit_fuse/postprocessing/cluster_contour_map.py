import numpy as np
from osgeo import gdal, osr
import argparse
import os
from sit_fuse.utils import numpy_to_torch, read_yaml, get_read_func







def contour_map(contour_fname, clust_fname):

    contour = gdal.Open(contour_fname)
    cdata = contour.ReadAsArray()
    clust = gdal.Open(clust_fname).ReadAsArray()

    inds = np.where(cdata > 0)
    out = np.zeros(clust.shape) - 1
    out[inds] = clust[inds]

    nx = clust.shape[1]
    ny = clust.shape[0]
    geoTransform = contour.GetGeoTransform()
    metadata = contour.GetMetadata()
    wkt = contour.GetProjection()
    gcpcount = contour.GetGCPCount()


    fname = contour_fname + ".FullColorContour.tif"
    out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(geoTransform)
    out_ds.SetMetadata(metadata)
    out_ds.SetProjection(wkt)
    out_ds.GetRasterBand(1).WriteArray(out)
    out_ds.FlushCache()
    out_ds = None




def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    contour_fnames = yml_conf["data"]["contour_fnames"]
    clust_fnames = yml_conf["data"]["clust_fnames"]

    for i in range(len(contour_fnames)):
        contour_map(contour_fnames[i], clust_fnames[i])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)




