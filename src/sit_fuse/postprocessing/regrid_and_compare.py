import salem
from salem import get_demo_file, open_xr_dataset
from sklearn.metrics import classification_report, confusion_matrix

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
    truth_full = None
    sf_full = None

    for i in range(len(sit_fuse_map_fnames)):
 
        map_fle = truth_map_fnames[i]
        sfmd = sit_fuse_map_fnames[i]

        print(sfmd, map_fle)

        sfm = open_xr_dataset(sfmd)
        sfm = sfm.clip(min=0, max=1)

        # prepare the figure and plot
        f, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(11, 7))
        sm = sfm.salem.get_map()
        sm.set_lonlat_contours(interval=0)
 
        # absolute values
        sm.set_data(sfm.to_array())
        sm.set_cmap('Spectral')
        sm.set_plot_params(vmin=-3, vmax=3)
        sm.visualize(ax=ax1, title='SIT_FUSE_Map')

        sm2 = copy.deepcopy(sfm.salem.get_map())
        sm2.set_lonlat_contours(interval=0)
        sm2.set_cmap('Spectral')
        sm2.set_plot_params(vmin=-3, vmax=3)
        sm2.visualize(ax=ax1, title='Truth_Map')

        full_data = None
        sm3 = copy.deepcopy(sfm.salem.get_map())
        sm3.set_lonlat_contours(interval=0)
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
        #gm_on_sfm[np.where(gm_on_sfm  > 2)] = 0
        #gm_on_sfm[np.where(gm_on_sfm  > 1)] = 1
        #gm_on_sfm[np.where(gm_on_sfm  < 0)] = 0

        gm_on_sfm[np.where(gm_on_sfm  > 0)] = 1
        gm_on_sfm[np.where(gm_on_sfm  < 0)] = 0

        tmp = np.array(sfm.to_array()).astype(np.int32)
        tmp2 = np.array(gm_on_sfm).astype(np.int32)
        dif = np.squeeze(tmp - tmp2)
        print(tmp.min(), tmp2.min(), tmp.max(), tmp2.max(), tmp.shape, tmp2.shape)
        #dif = np.squeeze(np.array(sfm.to_array())) - np.squeeze(np.array(gm_on_sfm))
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

        fbase = os.path.join(os.path.dirname(sfmd), os.path.basename(map_fle) + "_vs_" + os.path.basename(sfmd))
        plt.savefig(fbase + ".DIFFERENCE.png")


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
        fname = fbase + ".tif"
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

 
        #tmp2 is truth 
        finite_points = (np.isfinite(tmp2) & np.isfinite(tmp)) 
        report = classification_report(tmp2.ravel(), tmp.ravel())
        cm = confusion_matrix(tmp2.ravel(), tmp.ravel())
        #report = classification_report(tmp[finite_points], tmp2[finite_points], labels=[0,1])
        #report =
        if sf_full is None:
            sf_full = tmp.ravel()
            truth_full = tmp2.ravel()
        else:
            sf_full = np.concatenate((sf_full, tmp.ravel()))
            truth_full = np.concatenate((truth_full, tmp2.ravel()))
        print(sf_full.shape, truth_full.shape, "LENGTHS")
        print(report)
        print(cm)

    full_report = classification_report(truth_full, sf_full)
    cm_full = confusion_matrix(truth_full, sf_full)
    print(full_report)
    print(cm_full)

def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    regrid_and_compare(yml_conf)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)


