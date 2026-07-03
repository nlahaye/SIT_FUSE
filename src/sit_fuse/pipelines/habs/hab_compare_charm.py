
import os
import re
import numpy as np
import copy


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from netCDF4 import Dataset

from datetime import datetime, timezone

import rasterio
from rasterio.mask import mask
from shapely.geometry import box

from pyresample import area_config, bilinear, geometry, data_reduce, create_area_def, kd_tree
from pyresample import geometry
from pyresample.utils.rasterio import get_area_def_from_raster

INST_SUBDIRS = {
"modis" : "AQUA_MODIS",
"jpss1" : "JPSS1_VIIRS",
"pace" : "PACE",
"s3a" : "S3A",
"s3b" : "S3B",
"snpp" : "SNPP_VIIRS"
}

PRODUCT_RE = {
"pnd" : "(\\d{8})_DAY.pseudo_nitzschia_delicatissima_bloom.tif",
"pns" : "(\\d{8})_DAY.pseudo_nitzschia_seriata_bloom.tif",
#"total" : "(\d{8})_DAY\.total_phytoplankton\.tif"
} 

#SF_BASE_DIR = "/mnt/data/HAB_Data_SIT_FUSE/MERGED_HAB_20250225_S_CA/"
SF_BASE_DIR = "Z:\\2026\\Summer\\Southern California Bight HABs\\validation_testrun/"

#CHARM_FILES = [
#"/mnt/data/CHARM/charmForecast0day_LonPM180.nc",
#"/mnt/data/CHARM/wvcharmV3_0day_LonPM180.nc"
#]
CHARM_FILES = [
"Z:\\2026\\Summer\\Southern California Bight HABs\\C-HARM data\\wvcharmV3_0day_LonPM180.nc"
]

CHARM_AREA_DEF = "Z:\\2026\\Summer\\Southern California Bight HABs\\C-HARM data\\charmForecast0day_LonPM180.tif"

DT_RANGE = [datetime.strptime("20240306", "%Y%m%d"), datetime.strptime("20250401", "%Y%m%d")]

LON_BOUNDS = [-128, -116.0]
LAT_BOUNDS = [30, 38.94]

PN_VAR = "pseudo_nitzschia"
TIME_VAR = "time"
PN_VAR_ADDON = ":pseudo_nitzschia"
TIME_VAR_ADDON = ":time"

CHARM_THRESH = 0.75
SF_THRESH = 2

def find_sit_fuse_products(fdir):

    products = {}
    for key in INST_SUBDIRS:
        products[key] = {}
        for key2 in PRODUCT_RE:
            products[key][key2] = {}
            wlk_dir = os.path.join(fdir, INST_SUBDIRS[key])
            for root, dirs, files in os.walk(wlk_dir):
                for fle in files:
                    mtch = re.search(PRODUCT_RE[key2], fle)
                    if mtch:
                        datestr = mtch.group(1)
                        dt = datetime.strptime(datestr, "%Y%m%d")
                        if dt >= DT_RANGE[0] and dt <= DT_RANGE[1]:
                            products[key][key2][datestr] = os.path.join(root, fle)

    return products



def regrid_and_bin_charm(charm_fname, sf_fname):

    data_arr = None
    time_arr = None

    #print("netcdf:" , charm_fname , PN_VAR_ADDON)
    #nc_file_path = "netcdf:" + charm_fname + PN_VAR_ADDON
    #with rasterio.open(nc_file_path, 'r') as src:
    #    data_arr = src.read(1)
    nc = Dataset(charm_fname)
    nc.set_auto_maskandscale(False)
    lat = nc.variables["latitude"][:]
    lon = nc.variables["longitude"][:]

    nc.set_auto_maskandscale(True)
    data_arr = nc.variables[PN_VAR][:]
    time_arr = nc.variables[TIME_VAR][:]

    #print(lon.shape)  
    data_arr = np.moveaxis(data_arr, 0,2)
    data_arr = np.swapaxes(data_arr, 0,1)
    data_arr = np.flipud(data_arr) 


    area_id = 'my_area'
    description = 'Custom geographic region'
    proj_id = 'EPSG:4326'
    projection = 'EPSG:4326' #{'proj': 'latlong', 'datum': 'WGS84'}
    width = lat.shape[0]
    height = lon.shape[0]
    #print(lon, lat.min(), lon.max(), lat.max())
    area_extent = (lon.min(), lat.min(), lon.max(), lat.max()) # (left, bottom, right, top)
 
    charm_area_def = geometry.AreaDefinition(
        area_id, description, proj_id, projection, width, height, area_extent
    )
 
 
    #print(data_arr.shape)
    #print(time_arr.shape)
    #nc_file_path = "netcdf:" + charm_fname + TIME_VAR_ADDON
    #with rasterio.open(nc_file_path, 'r') as src:
    #    time_arr = np.squeeze(src.read(1))

    inds = np.where(data_arr < 0.0)
    data_arr[np.where(data_arr < CHARM_THRESH)] = 0
    data_arr[np.where(data_arr >= CHARM_THRESH)] = 1
    data_arr[inds] = -1.0
    data_arr = data_arr.astype(np.int16)

    #minx, miny, maxx, maxy = [LON_BOUNDS[0], LAT_BOUNDS[0], LON_BOUNDS[1], LAT_BOUNDS[1]] 
    #bbox_polygon = box(minx, miny, maxx, maxy)
    #geometries = [bbox_polygon]
   
    #sf_transform = None
    #with rasterio.open(sf_fname) as src:
    #    sf_image, sf_transform = mask(dataset=src, 
    #                                shapes=geometries, 
    #                                crop=True)


    final_area_def = get_area_def_from_raster(sf_fname)
    #charm_area_def = get_area_def_from_raster(CHARM_AREA_DEF)
    #final_area_def = sf_transform

    #print(charm_area_def, final_area_def)
    #print(data_arr.min(), data_arr.max(), data_arr.mean()) 

    regrid_charm = np.squeeze(kd_tree.resample_nearest(charm_area_def, data_arr, final_area_def, radius_of_influence=10000, fill_value = -1))

    #print(regrid_charm.min(), regrid_charm.max(), regrid_charm.mean())
    #print(charm_area_def, final_area_def)

    return regrid_charm, data_arr, time_arr


def regrid_and_bin_sf(sf_fname):

    #minx, miny, maxx, maxy = [LON_BOUNDS[0], LAT_BOUNDS[0], LON_BOUNDS[1], LAT_BOUNDS[1]]
    #bbox_polygon = box(minx, miny, maxx, maxy)
    #geometries = [bbox_polygon]

    #sf_transform = None
    #sf_image = None
    #with rasterio.open(sf_fname) as src:
    #    sf_image, sf_transform = mask(dataset=src,
    #                                shapes=geometries,
    #                                crop=False, nodata=-1)   
    

    sf_image =  np.squeeze(rasterio.open(sf_fname).read(1))
    sf_image_binned = copy.deepcopy(sf_image)
    inds = np.where(sf_image < 1.0)
    sf_image_binned[np.where(sf_image < SF_THRESH)] = 0
    sf_image_binned[np.where(sf_image >= SF_THRESH)] = 1
    sf_image_binned[inds] = -1.0

    sf_image_binned = sf_image_binned.astype(np.int16)

    return sf_image_binned, sf_image




def run_comparison():
    fdir = SF_BASE_DIR
    sf_prods = find_sit_fuse_products(fdir)
 
    sf_fname = ""
    for key in sf_prods:
        for key2 in sf_prods[key]:
            for key3 in sf_prods[key][key2]:
                sf_fname = sf_prods[key][key2][key3]
                break

    charm_data, data_arr, time_arr = regrid_and_bin_charm(CHARM_FILES[0], sf_fname)
    charm_data_pace, da_pace, time_arr_pace = regrid_and_bin_charm(CHARM_FILES[1], sf_fname)
 
    hist_by_concentration = {}
    hist_by_inst = {}
    hist_total = {}
    instrument_order = ["modis", "snpp", "jpss1", "s3a", "s3b", "pace"]
    product_order = ["pnd", "pns"] #, "total"]


    diff_map = None
    total_diffs = None
    for i in range(len(time_arr)):
        for inst in range(len(instrument_order)):
            instrument = instrument_order[inst]
            if "pace" in instrument:
                if i >= time_arr_pace.shape[0]:
                    continue
                charm_dt = datetime.fromtimestamp(time_arr_pace[i], tz=timezone.utc)
                tm_str = charm_dt.strftime("%Y%m%d")
                charm_arr = np.squeeze(charm_data_pace[:,:,i])
            else:
                #print(time_arr.shape, charm_data.shape)
                charm_dt = datetime.fromtimestamp(time_arr[i], tz=timezone.utc)
                tm_str = charm_dt.strftime("%Y%m%d")
                charm_arr = np.squeeze(charm_data[:,:,i])
 
            #if instrument not in hist_by_concentration:
            #    hist_by_concentration[instrument] = {}
            if instrument not in hist_by_inst:
                hist_by_inst[instrument] = {}            

            for prod in range(len(product_order)):

                product = product_order[prod]                

                if product not in hist_by_concentration:
                    hist_by_concentration[product] = {1:{"f1":[], "count":[]},2:{"f1":[], "count":[]},\
                       3:{"f1":[], "count":[]},4:{"f1":[], "count":[]},5:{"f1":[], "count":[]},6:{"f1":[], "count":[]}}
                if product not in hist_by_inst[instrument]:
                    hist_by_inst[instrument][product] = {"f1":[], "count":[]}
                if product not in hist_total:
                    hist_total[product] = {"f1":[], "count":[]}

                if tm_str in sf_prods[instrument][product]:
                    sf_fname = sf_prods[instrument][product][tm_str]
                    #print(sf_fname, tm_str, time_arr[i])
                    sf_data_binned, sf_data = regrid_and_bin_sf(sf_fname)

                    sf_data_binned = np.squeeze(sf_data_binned)
                    sf_data = np.squeeze(sf_data)

                    #sf_data_comp = np.squeeze(copy.deepcopy(sf_data_binned))
                    #charm_arr_comp = np.squeeze(copy.deepcopy(charm_arr))
                    #print(sf_data.shape, sf_data_binned.shape, charm_arr_comp.shape)
                    inds = np.where((sf_data == 1) & (charm_arr > -1))
                    sf_data_comp = np.zeros(sf_data_binned.shape)-1
                    sf_data_comp[inds] = sf_data_binned[inds]
                    charm_arr_comp = np.zeros(sf_data_binned.shape)-1
                    charm_arr_comp[inds] = charm_arr[inds]

                    #inds2 = np.where((sf_data > 1) | (charm_arr_comp < 0))
                    #sf_data_comp[inds2] = -1
                    #charm_arr_comp[inds2] = -1
                    if diff_map is None:
                        diff_map = np.zeros(sf_data_binned.shape).astype(np.float32)
                        total_diffs = np.zeros(sf_data_binned.shape).astype(np.float32)

                    tp_count =  np.count_nonzero(((sf_data_comp == 0) & (charm_arr_comp == 0)))
                    fp_count = np.count_nonzero(((sf_data_comp == 0) & (charm_arr_comp == 1)))
                    fn_count = np.count_nonzero(((sf_data_comp == 1) & (charm_arr_comp == 0)))
                    total_count = tp_count + fn_count 
                    if total_count > 0 and tp_count > 0:
                        precision = tp_count  / (tp_count + fp_count)
                        recall = tp_count  / (tp_count + fn_count)
                        f1 = 2 * ((precision*recall) / (precision + recall))
                        hist_by_concentration[product][1]["f1"].append(f1)
                        hist_by_concentration[product][1]["count"].append(total_count)

                    for j in range(2,7): #concentration_levels
                        #sf_data_comp = copy.deepcopy(sf_data_binned)
                        #charm_arr_comp = copy.deepcopy(charm_arr)

                        inds = np.where(sf_data == j)
                        sf_data_comp = np.zeros(sf_data_binned.shape)-1
                        sf_data_comp[inds] = sf_data_binned[inds]
                        charm_arr_comp = np.zeros(sf_data_binned.shape)-1
                        charm_arr_comp[inds] = charm_arr[inds]
                        #inds2 = np.where((sf_data != j) | (charm_arr_comp < 0))
                        #sf_data_comp[inds2] = -1
                        #charm_arr_comp[inds2] = -1

                        tp_count =  np.count_nonzero(((sf_data_comp == 0) & (charm_arr_comp == 0)))
                        fp_count = np.count_nonzero(((sf_data_comp == 0) & (charm_arr_comp == 1)))
                        fn_count = np.count_nonzero(((sf_data_comp == 1) & (charm_arr_comp == 0)))
                        
                        total_count = tp_count + fn_count 
                        if total_count == 0 or tp_count == 0:
                            if fn_count > 0:
                                hist_by_concentration[product][j]["f1"].append(0.0)
                                hist_by_concentration[product][j]["count"].append(total_count)
                            continue
                        precision = tp_count  / (tp_count + fp_count)
                        recall = tp_count  / (tp_count + fn_count)
                        f1 = 2 * ((precision*recall) / (precision + recall))
                        hist_by_concentration[product][j]["f1"].append(f1)
                        hist_by_concentration[product][j]["count"].append(total_count)

                        
                        tp_count =  np.count_nonzero(((sf_data_comp == 1) & (charm_arr_comp == 1)))
                        fp_count = np.count_nonzero(((sf_data_comp == 1) & (charm_arr_comp == 0)))
                        fn_count = np.count_nonzero(((sf_data_comp == 0) & (charm_arr_comp == 1)))
                        total_count = tp_count + fn_count
                        if total_count == 0 or tp_count == 0:
                            if fn_count > 0:
                                hist_by_concentration[product][j]["f1"].append(0.0)
                                hist_by_concentration[product][j]["count"].append(total_count)
                            continue
                        precision = tp_count  / (tp_count + fp_count)
                        recall = tp_count  / (tp_count + fn_count)
                        f1 = 2 * ((precision*recall) / (precision + recall))
                        hist_by_concentration[product][j]["f1"].append(f1)
                        hist_by_concentration[product][j]["count"].append(total_count)

                    inds = np.where((sf_data > -1) & (charm_arr > -1))
                    sf_data_comp = np.zeros(sf_data_binned.shape)-1
                    sf_data_comp[inds] = sf_data_binned[inds]
                    charm_arr_comp = np.zeros(sf_data_binned.shape)-1
                    charm_arr_comp[inds] = charm_arr[inds]

                    #inds2 = np.where((sf_data_binned < 0) | (charm_arr < 0))
                    #sf_data_comp = copy.deepcopy(sf_data_binned)
                    #charm_arr_comp = copy.deepcopy(charm_arr)
                    #sf_data_comp[inds2] = -1
                    #charm_arr_comp[inds2] = -1
                    #plt.matshow(sf_data_comp, vmin=-1, vmax=1, cmap="jet")
                    #plt.savefig(instrument + "_" + product + "_" + tm_str + "_SF.png")
                    #plt.clf()
                    #plt.matshow(sf_data_comp-charm_arr_comp, vmin=0, vmax=1, cmap="jet")
                    #plt.savefig(instrument + "_" + product + "_" + tm_str + "_DIFF.png")
                    #plt.clf()
                    #plt.matshow(charm_arr_comp, vmin=-1, vmax=1, cmap="jet")
                    #plt.savefig(instrument + "_" + product + "_" + tm_str + "_CHARM.png")
                    #plt.matshow(data_arr[:,:,i], vmin=0, vmax=1, cmap="jet")
                    #plt.savefig(instrument + "_" + product + "_" + tm_str + "_CHARM_INIT.png")
                    #print(charm_arr_comp.max(),charm_arr.mean(), charm_arr.max())
                    tp_count =  np.count_nonzero(((sf_data_comp == 0) & (charm_arr_comp == 0)))
                    fp_count = np.count_nonzero(((sf_data_comp == 0) & (charm_arr_comp == 1)))
                    fn_count = np.count_nonzero(((sf_data_comp == 1) & (charm_arr_comp == 0)))
                    total_count = tp_count + fn_count 

                    #print(instrument, product, total_count)
                    if total_count > 0 and tp_count > 0:
                        precision = tp_count  / (tp_count + fp_count)
                        recall = tp_count  / (tp_count + fn_count)
                        f1 = 2 * ((precision*recall) / (precision + recall))
                        hist_by_inst[instrument][product]["f1"].append(f1)
                        hist_by_inst[instrument][product]["count"].append(total_count)
                        hist_total[product]["f1"].append(f1)
                        hist_total[product]["count"].append(total_count)
                    elif fp_count > 0:
                        hist_by_inst[instrument][product]["f1"].append(0.0)
                        hist_by_inst[instrument][product]["count"].append(total_count)
                        hist_total[product]["f1"].append(0.0)
                        hist_total[product]["count"].append(total_count)
                    
                    tp_count =  np.count_nonzero(((sf_data_comp == 1) & (charm_arr_comp == 1)))
                    fp_count = np.count_nonzero(((sf_data_comp == 1) & (charm_arr_comp == 0)))
                    fn_count = np.count_nonzero(((sf_data_comp == 0) & (charm_arr_comp == 1)))
                    total_count = tp_count + fn_count

                    #print(instrument, product, total_count)
                    if total_count > 0 and tp_count > 0:
                        precision = tp_count  / (tp_count + fp_count)
                        recall = tp_count  / (tp_count + fn_count)
                        f1 = 2 * ((precision*recall) / (precision + recall))
                        hist_by_inst[instrument][product]["f1"].append(f1)
                        hist_by_inst[instrument][product]["count"].append(total_count)
                        hist_total[product]["f1"].append(f1)
                        hist_total[product]["count"].append(total_count)
                    elif fp_count > 0:
                        hist_by_inst[instrument][product]["f1"].append(0.0)
                        hist_by_inst[instrument][product]["count"].append(total_count)
                        hist_total[product]["f1"].append(0.0)
                        hist_total[product]["count"].append(total_count)

                    diff_inds = np.where(((sf_data_comp == 0) & (charm_arr_comp == 1)) | ((sf_data_comp == 1) & (charm_arr_comp == 0)))
                    total_inds = np.where((sf_data_comp >= 0) & (charm_arr_comp >= 0))
                    diff_map[diff_inds] = diff_map[diff_inds] + 1
                    total_diffs[total_inds] = total_diffs[total_inds] + 1 #len(diff_inds[0])      
                    print(total_diffs, len(diff_inds[0]))
    print(diff_map.min(), diff_map.mean(),diff_map.max(), diff_map.std())
    diff_map = diff_map / total_diffs
    print(np.nanmean(diff_map), np.nanmin(diff_map),np.nanmax(diff_map), np.nanstd(diff_map))
    plt.matshow(diff_map, vmin=0, vmax=1, cmap="jet")  
    plt.colorbar() 
    plt.savefig("TOTAL_DIFF_MAP.png", dpi=400) 
    plt.clf()
    plt.close()
    plt.matshow(total_diffs, cmap="jet")
    plt.colorbar()
    plt.savefig("TOTAL_DIFFS.png", dpi=400)       
    print(instrument_order, product_order)
    for prod in product_order:
        print(prod,  np.average(hist_total[prod]["f1"], weights=hist_total[prod]["count"]))
        print(prod, np.average(hist_total[prod]["f1"], weights=hist_total[prod]["count"]))
        #print(hist_total[prod]["f1"])
        for instrument in instrument_order:
            print(instrument, prod, np.average(hist_by_inst[instrument][prod]["f1"], weights=hist_by_inst[instrument][prod]["count"]))
            #print(hist_by_inst[instrument][prod]["f1"])
        for n in range(1,7):
            if sum(hist_by_concentration[prod][n]["count"]) < 1:
                continue
            print(prod, n, np.average(hist_by_concentration[prod][n]["f1"], weights=hist_by_concentration[prod][n]["count"]))      
        



run_comparison()

