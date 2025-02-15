



import numpy as np
import os
from osgeo import gdal
import argparse
from sit_fuse.utils import numpy_to_torch, read_yaml, get_read_func


def gen_zonal_histogram(zone_raster_path, value_raster_path, zonal_histogram = None):
    """
    Calculates the zonal histogram of a value raster, 
    based on the zones defined in a zone raster.

    Args:
        zone_raster_path (str): Path to the zone raster file.
        value_raster_path (str): Path to the value raster file.

    Returns:
        dict: A dictionary where keys are the unique zone values 
              and values are the histograms of the corresponding zones 
              in the value raster.
    """

    zone_array = gdal.Open(zone_raster_path).ReadAsArray()
    value_array = gdal.Open(value_raster_path).ReadAsArray()

    unique_zones = np.unique(zone_array)
    if zonal_histogram is None:
        zonal_histograms = {}
    else:
        zonal_histograms = zonal_histogram 

    for zone in unique_zones:
      if zone is not None:
        mask = zone_array == zone
        masked_values = value_array[mask]
        hist, bins = np.histogram(masked_values, bins=sorted(np.unique(masked_values)))
        if zone not in zonal_histograms.keys():
            zonal_histograms[zone] = {} 
        for b in range(len(bins)):
            if bins[b] not in zonal_histograms[zone].keys():
                zonal_histograms[zone][bins[b]] = hist[b] 
            else:
                zonal_histograms[zone][bins[b]] = zonal_histograms[zone][bins[b]] + hist[b] 

    return zonal_histograms


def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    clust_gtiffs = yml_conf["data"]["clust_gtiffs"]
    label_gtiffs = yml_conf["data"]["label_gtiffs"]
    out_dir = yml_conf["output"]["out_dir"]
    out_tag = yml_conf["output"]["class_name"]

    zonal_histogram = None
    for i in range(len(clust_gtiffs)):
        for j in range(len(label_gtiffs[i])):
            zonal_histogram = gen_zonal_histogram(label_gtiffs[i][j], clust_gtiffs[i], zonal_histogram)

    np.save(os.path.join(out_dir, out_tag + "_hist_ditc.npy"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)






