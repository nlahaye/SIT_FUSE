import geojson
import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import os
import re

from sit_fuse.preprocessing.grid_raster import run_split

import regionmask

ocean_basins_50 =  regionmask.defined_regions.natural_earth_v5_1_2.ocean_basins_50


def find_init_files(yml_conf):

    fnames = []
    fdir = yml_conf["input_dir"]
    #Find all files
    for root, dirs, files in os.walk(fdir):
        for fle in files:
            mtch = re.search(yml_conf["base_file_re"], fle)
            if mtch:
                fnames.append(os.path.join(root, fle))
    return fnames

def grid_data(fnames, yml_conf):
    
    new_conf = {"n_tiles" : yml_conf["n_tiles"], "fnames": fnames}
    gridded_fnames = run_split(new_conf)

    return gridded_fnames

def mask_tiles(gridded_fnames, yml_conf):

    savi_raster_path = yml_conf["savi_raster_path"]
    roads_geojson_path = yml_conf["roads_geojson_path"]

    roads = None
    with open(roads_geojson_path, 'r') as file:
        roads = geojson.load(file)

    road_geom = []
    for g in range(len(roads["features"])):
        road_geom.append(roads["features"][g]["geometry"])

    for i in range(len(gridded_fnames)):
        print(gridded_fnames[i])

        raster_data = None
        raster_transform = None
        raster_crs = None
        raster_meta = None
        lat = None
        lon = None
        # Load your raster data
        with rasterio.open(gridded_fnames[i]) as target_src:
            raster_transform = target_src.transform
            raster_crs = target_src.crs
            raster_meta = target_src.meta
            raster_profile = target_src.profile
            raster_profile['dtype'] = "float32"
            raster_data = target_src.read()  # Read a single band

            # Create a land mask using regionmask
            # Adjust grid parameters to match your raster's extent and resolution
            lon = np.arange(target_src.bounds.left, target_src.bounds.right, target_src.res[0])
            lat = np.arange(target_src.bounds.bottom, target_src.bounds.top, target_src.res[1])
           
  
            raster_data, transform_info = rasterio.mask.mask(target_src, road_geom, invert=True, nodata=0.0, filled=False)

            savi_data = None
            with rasterio.open(savi_raster_path) as savi_src:
                savi_data = savi_src.read(1) * 0.0001
                savi_meta = savi_src.meta.copy()

                # Check if mask and target have the same dimensions and transform
                if savi_data.shape != raster_data.shape[1:3] or savi_meta['transform'] != target_meta['transform']:
 
                    height = target_src.height
                    width = target_src.width
                    savi_resampled = np.empty((1, height, width), dtype=savi_data.dtype)

                    print("resampling")

                    reproject(
                        source=savi_data,
                        destination=savi_resampled,
                        src_transform=savi_src.transform,
                        src_crs=savi_src.crs,
                        dst_transform=target_src.transform,
                        dst_crs=target_src.crs,
                        resampling=Resampling.cubic
                    ) 
                    print("resampled")

                    savi_data = savi_resampled[0] #.astype(bool)
                else:
                    savi_data = savi_data #.astype(bool)



        inds = np.where(((savi_data > -100) & (savi_data < yml_conf["savi_min"]))) #Account for lower bound, but don't filter areas with no data (fill == -19999)
        savi_data[inds] = 0.0
        inds = np.where(((savi_data > yml_conf["savi_max"])))
        savi_data[inds] = 0.0
        inds = np.where((savi_data >= yml_conf["savi_min"]))
        savi_data[inds] = 1.0
        savi_data = savi_data.astype(np.bool)

        # Apply the mask
        inds = np.where(savi_data == False)
        raster_data[:,inds[0], inds[1]] = 0.0
 
        # Use a predefined landmask like 'natural_earth_v5_0_0.land_110'
        land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_10.mask(lon, lat)
 

        #tmp1 = (~land_mask.isnull().to_numpy()).astype(np.bool_)
        tmp1 = (land_mask.isnull().to_numpy()).astype(np.bool_)

        # Reshape the land mask to match the raster dimensions
        land_mask_reshaped = np.squeeze(tmp1)
        land_mask_reshaped = np.flipud(land_mask_reshaped)  # Flip if necessary
 
        # Apply the mask to the raster data
        max_y = raster_data.shape[-2]
        max_x = raster_data.shape[-1]
        land_mask_reshaped = land_mask_reshaped[:max_y, :max_x]
 
        inds = np.where(land_mask_reshaped == True)
        for c in range(raster_data.shape[0]):
            raster_data[c,inds[0], inds[1]] = 0.0
 
        raster_meta.update({
            "driver": "GTiff",
            "height": raster_data.shape[1],
            "count": raster_data.shape[0],
            "width": raster_data.shape[2],
            "transform": raster_transform,
            "crs": raster_crs,
            "dtype": "float32", #"float32",
            "nodata": 0
        })

        with rasterio.open(gridded_fnames[i], "w", **raster_meta) as dest:
            dest.profile['dtype'] = "float32"
            dest.write(raster_data)


def run_pv_grid_and_mask(yml_conf):

    fnames = find_init_files(yml_conf)
    gridded_fnames = grid_data(fnames, yml_conf)
    mask_tiles(gridded_fnames, yml_conf)
  
    #gridded_fnames = find_init_files(yml_conf)

    return gridded_fnames



