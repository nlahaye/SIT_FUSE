import salem
from salem import get_demo_file, open_xr_dataset

import os
import numpy as np
import matplotlib.pyplot as plt

from osgeo import gdal, osr
import copy

sit_fuse_map_fnames = [
    "/Users/nlahaye/Downloads/Palm_Oil_Maps/ucayali_stacks_2020_h1v1.tif.heir_clust1output_test.data_79900clusters.zarr.palm_oil_low.tif.Contours.tif",
]

global_map_fnames = [
    "/Users/nlahaye/Downloads/High_resolution_global_industrial_and_smallholder_oil_palm_map_for_2019/oil_palm_map/Ucayali_Subset/L2_2019b_0279.tif",
]

for sfmd in sit_fuse_map_fnames:

    sfm = open_xr_dataset(sfmd)

    # prepare the figure and plot
    f, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(11, 7))
    sm = sfm.salem.get_map()
    # sm.set_lonlat_contours(interval=0)

    # absolute values
    sm.set_data(sfm.to_array())
    sm.set_cmap('Spectral')
    sm.set_plot_params(vmin=-3, vmax=3)
    sm.visualize(ax=ax1, title='Ucayali Map')

    sm2 = copy.deepcopy(sfm.salem.get_map())
    sm2.set_cmap('Spectral')
    sm2.set_plot_params(vmin=-3, vmax=3)
    sm2.visualize(ax=ax1, title='Ucayali Map')

    full_data = None
    sm3 = copy.deepcopy(sfm.salem.get_map())
    sm3.set_cmap('Spectral')
    sm3.set_plot_params(vmin=-3, vmax=3)
    sm3.visualize(ax=ax1, title='Ucayali Map')

    full_data = np.zeros(np.squeeze(sfm.to_array()).shape)
    for map_fle in global_map_fnames:
        print("Opening", map_fle)
        gm = open_xr_dataset(map_fle)

        print("Regridding")
        gm_on_sfm, lut = sfm.salem.lookup_transform(gm, return_lut=True)
        # gm_on_sfm = gm_on_sfm.fillna(-3)
        gm_on_sfm = np.array(gm_on_sfm.to_array())
        gm_on_sfm[np.where(gm_on_sfm > 2)] = 0
        gm_on_sfm[np.where(gm_on_sfm > 1)] = 1
        gm_on_sfm[np.where(gm_on_sfm < 0)] = 0

        dif = np.squeeze(np.array(sfm.to_array())) - np.squeeze(np.array(gm_on_sfm))
        print(np.nanmean(dif), np.nanmin(dif), np.nanmax(dif))

        print("Diff Mapping")

        sm2.set_data(gm_on_sfm, overplot=True)
        sm2.set_cmap('Spectral')
        sm2.set_plot_params(vmin=-3, vmax=3)
        sm2.visualize(ax=ax2, title='Global Map')
        sm3.set_data(dif, overplot=True)
        inds = np.where(np.isfinite(dif))
        print(dif.shape, full_data.shape)
        full_data[inds] = dif[inds]
        sm3.set_cmap('Spectral')
        sm3.set_plot_params(vmin=-3, vmax=3)
        sm3.visualize(ax=ax3, title='Difference Map')

    # make it nice
    plt.tight_layout()
    # plt.show()
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

# plt.clf()