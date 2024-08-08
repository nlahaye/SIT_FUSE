import salem
from salem import get_demo_file, open_xr_dataset
from sklearn.metrics import classification_report

import argparse
 
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from osgeo import gdal, osr
import copy

from sit_fuse.utils import numpy_to_torch, read_yaml, get_read_func

def regrid_and_compare(config):

    sit_fuse_map_fnames = config["sit_fuse_maps"]
    truth_map_fnames = config["truth_maps"]

    for i in range(len(sit_fuse_map_fnames)):
 
        map_fle = truth_map_fnames[i]
        sfmd = sit_fuse_map_fnames[i]

        print(sfmd, map_fle)

        sfm = open_xr_dataset(sfmd)
 
        # prepare the figure and plot
        f, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(11, 7))
        sm = sfm.salem.get_map()
        #sm.set_lonlat_contours(interval=0)
 
        # absolute values
        sm.set_data(sfm.to_array())
        sm.set_cmap('Spectral')
        sm.set_plot_params(vmin=-3, vmax=3)
        sm.visualize(ax=ax1, title='SIT_FUSE_Map')

        sm2 = copy.deepcopy(sfm.salem.get_map())
        sm2.set_cmap('Spectral')
        sm2.set_plot_params(vmin=-3, vmax=3)
        sm2.visualize(ax=ax1, title='Truth_Map')

        full_data = None
        sm3 = copy.deepcopy(sfm.salem.get_map())
        sm3.set_cmap('Spectral')
        sm3.set_plot_params(vmin=-3, vmax=3)
        sm3.visualize(ax=ax1, title='Difference_Map')
 
        full_data = np.zeros(np.squeeze(sfm.to_array()).shape)

        print("Opening", map_fle)
        gm = open_xr_dataset(map_fle)
 
        print("Regridding")
        gm_on_sfm, lut = sfm.salem.lookup_transform(gm, return_lut=True)
        #gm_on_sfm = gm_on_sfm.fillna(-3)
        gm_on_sfm = np.array(gm_on_sfm.to_array())
        gm_on_sfm[np.where(gm_on_sfm  > 2)] = 0
        gm_on_sfm[np.where(gm_on_sfm  > 1)] = 1
        gm_on_sfm[np.where(gm_on_sfm  < 0)] = 0
 
        dif = np.squeeze(np.array(sfm.to_array())) - np.squeeze(np.array(gm_on_sfm))
        print(np.nanmean(dif), np.nanmin(dif), np.nanmax(dif))

        print("Diff Mapping")
 
        sm2.set_data(gm_on_sfm, overplot=True)
        sm2.set_cmap('Spectral')
        sm2.set_plot_params(vmin=-3, vmax=3)
        sm2.visualize(ax=ax2, title='Truth_Map')
        sm3.set_data(dif, overplot=True)
        inds = np.where(np.isfinite(dif))
        print(dif.shape, full_data.shape)
        full_data[inds] = dif[inds]
        sm3.set_cmap('Spectral')
        sm3.set_plot_params(vmin=-3, vmax=3)
        sm3.visualize(ax=ax3, title='Difference_Map')

        # make it nice
        plt.tight_layout()
        #plt.show()
        plt.savefig("DIFFERENCE_" + os.path.basename(map_fle) + ".png")


        out_dat = np.squeeze(full_data)
        print(np.nanmean(out_dat), np.nanmin(out_dat), np.nanmax(out_dat))
        dat = gdal.Open(sfmd)

        dat_tmp = dat.ReadAsArray()
        nx = out_dat.shape[1]
        ny = out_dat.shape[0]
        geoTransform = dat.GetGeoTransform()
        gt2 = [geoTransform[0], geoTransform[1], geoTransform[2], geoTransform[3], geoTransform[4], geoTransform[5]]
        gt2[1] = gt2[1] * dat_tmp.shape[1] / nx
        gt2[5] = gt2[5] * dat_tmp.shape[0] / ny
 
        metadata = dat.GetMetadata()
        wkt = dat.GetProjection()
        gcpcount = dat.GetGCPCount()
        gcp = None
        gcpproj = None

        if gcpcount > 0:
            gcp = dat.GetGCPs()
            gcpproj = dat.GetGCPProjection()
            dat.FlushCache()
            dat = None

        print(out_dat.shape, np.nanmean(out_dat), np.nanmin(out_dat), np.nanmax(out_dat), nx, ny, "HERE")
        print(geoTransform, gt2)
        fname = "SIT_FUSE_vs_Global_Map_" + os.path.basename(sfmd) + ".tif"
        print(fname)
        out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Int16)
        out_ds.SetGeoTransform(gt2)
        out_ds.SetMetadata(metadata)
        out_ds.SetProjection(wkt)
        if gcpcount > 0:
            out_ds.SetGCPs(gcp, gcpproj)
        out_ds.GetRasterBand(1).WriteArray(out_dat)
        out_ds.FlushCache()
        out_ds = None


        trth = np.squeeze(np.array(gm_on_sfm))
        finite_points = (np.isfinite(out_dat) & np.isfinite(trth)) 
        report = classification_report(out_dat[finite_points], trth[finite_points], labels=[0,1])
        print(report)


def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    regrid_and_compare(yml_conf)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)


