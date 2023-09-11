"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
import os
import re
import zarr
import numpy as np
from osgeo import osr, gdal
from subprocess import DEVNULL, run, Popen, PIPE
from scipy.ndimage import uniform_filter
from scipy.ndimage import variance

from utils import numpy_to_torch, read_yaml, get_read_func, get_lat_lon

TIF_RE = "(\w+_\w+_)\w+(_\d+_\d+)_wgs84_fit.tif"
MODIS_BAND_ORDER = ["vis01", "vis02", "vis03", "vis04", "vis05", "vis06", "vis07",  "bt20", "bt21", "bt22", "bt23", "bt24", "bt25", "vis26", "bt27", "bt28", "bt29", "bt30", "bt31", "bt32", "bt33", "bt34", "bt35", "bt36"]


def run_cmd(cmd):
    print(f"Running: {cmd}")
    p = Popen(cmd,
             stdout=PIPE,
             stderr=PIPE,
             shell=True)

    (out, err) = p.communicate()
    print(err.decode(), end=" ")
    print(out.decode(), end=" ")




def goes_to_geotiff(data_file):

    out_fname = os.path.splitext(data_file)[0] + ".tif"
    cmd = "gdal_translate NETCDF:\"" + data_file + "\":Rad tmp.tif;" 
    cmd += " gdalwarp -t_srs EPSG:4326 -wo SOURCE_EXTRA=100 tmp.tif " + out_fname
    cmd += "; rm tmp.tif" 

    run_cmd(cmd)



def lee_filter(img, size):
    """
        Lee Speckle Filter for synthetic aperature radar data.
        
        img: image data
        size: size of Lee Speckle Filter window (optimal size is usually 5)
    """
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output



def uavsar_to_geotiff(in_fps, out_dir, **kwargs):
    """
    Converts UAVSAR file(s) to geotiff.
    Args:
        in_fps (list(string) or string):  list of strings (each file will be treated as a separate channel)
                                          or string of data file paths
        out_dir (string): directory to which the geotiffs will be saved
        kwargs:
            ann_fps (list(string) or string): list of or string of UAVSAR annotation file paths,
                                            ann files will be automatically matched to data files

    Returns:
        data: numpy array of shape (channels, lines, samples) 
              Complex-valued (unlike polarization) data will be split into separate phase and amplitude channels. 
    """
    
    from utils import read_uavsar
    
    if "ann_fps" in kwargs:
        ann_fps = kwargs["ann_fps"]

    if not out_dir:
        out_dir = os.path.dirname(in_fps)
    if os.path.isfile(out_dir):
        raise Exception('Provide a directory, not a filepath.')
    
    desc = None
    type = None
    search = None
    
    data = read_uavsar(in_fps, desc, type, search, **kwargs)
    out_fps = []
    for dat, fp in zip(data, in_fps):
        
        fname = os.path.os.path.basename(fp)
        out_fp = os.path.join(out_dir, fname) + '.tiff'
        exts = fname.split('.')[1:]
        dtype = dat.dtype
        if dtype == np.complex64:
            bands = 2
        else:
            bands = 1

        driver = gdal.GetDriverByName("GTiff")
            
        # If ground projected image, north up...
        if type in {'grd', 'slope', 'inc'}: 
            # Delta latitude and longitude
            dlat = float(desc[f'{search}.row_mult']['value'])
            dlon = float(desc[f'{search}.col_mult']['value'])
            # Upper left corner coordinates
            lat1 = float(desc[f'{search}.row_addr']['value'])
            lon1 = float(desc[f'{search}.col_addr']['value'])
            # Set up geotransform for gdal
            srs = osr.SpatialReference()
            # WGS84 Projection, spatial reference using the EPSG code (4326)
            srs.ImportFromEPSG(4326)
            t = [lon1, dlon, 0.0, lat1, 0.0, dlat]

        if type == 'slope':
                out_fps = []
                for direction, arr in data.items():
                    slope_fp = out_fp.replace('.tiff',f'.{direction}.tiff')
                    print(f"saving to {slope_fp}.")
                    ds = driver.Create(slope_fp, 
                                        ysize=arr.shape[0], 
                                        xsize=arr.shape[1], 
                                        bands=bands, 
                                        eType=gdal.GDT_Float32)
                    ds.SetProjection(srs.ExportToWkt())
                    ds.SetGeoTransform(t)
                    ds.GetRasterBand(1).WriteArray(np.abs(dat))
                    ds.GetRasterBand(1).SetNoDataValue(np.nan)
                    ds.FlushCache()
                    ds = None
                    out_fps.append(slope_fp)
        else:
            ds = driver.Create(out_fp, 
                                    ysize=dat.shape[0], 
                                    xsize=dat.shape[1], 
                                    bands=bands, 
                                    eType=gdal.GDT_Float32)
            if type in {'grd', 'inc'}:
                ds.SetProjection(srs.ExportToWkt())
                ds.SetGeoTransform(t)
            if bands == 2:
                ds.GetRasterBand(1).WriteArray(np.abs(dat))
                ds.GetRasterBand(1).SetNoDataValue(np.nan)
                ds.GetRasterBand(2).WriteArray(np.angle(dat))
                ds.GetRasterBand(2).SetNoDataValue(np.nan)
            else:
                ds.GetRasterBand(1).WriteArray(dat)
                ds.GetRasterBand(1).SetNoDataValue(np.nan)
            out_fps.append(out_fp)
            
        ds.FlushCache() # save tiffs
        ds = None  # close the dataset
    
    print("Saved geotiffs to:")
    print(out_fps, sep='\n')
    return out_fps



def gtiff_to_gtiff_multfile(fname, n_channels, **kwargs):

    if os.path.isfile(fname):
        dat = gdal.Open(fname)
        root, ext = os.path.splitext(fname)
        geoTransform = dat.GetGeoTransform()
        wkt = dat.GetProjection()
        dat.FlushCache()
        nx = 0
        ny = 0
        for i in range(n_channels):
            dat_arr = dat.GetRasterBand((i+1)).ReadAsArray()
            if i == 0:
                nx = dat_arr.shape[1]
                ny = dat_arr.shape[0]
            dat.FlushCache()
            out_fname = None
            if "band_key" in kwargs:
                out_fname = root + "_" + kwargs["band_key"] + str(i+1) + ext
            else:
                out_fname = root + "_BAND" + str(i+1) + ext

            
            out_ds = gdal.GetDriverByName("GTiff").Create(out_fname, nx, ny, 1, gdal.GDT_Float32)
            out_ds.SetGeoTransform(geoTransform)
            out_ds.SetProjection(wkt)
            out_ds.GetRasterBand(1).WriteArray(dat_arr)
            out_ds.FlushCache()
            out_ds = None
            del dat_arr
        dat.FlushCache()
        dat = None            


#Assumes each set of tiffs will be moved to a separate directory
def gen_polar_2_grid_cmds(exe_location, data_files, location_files, instruments, out_dirs):

    cmds = []
    cmd = "rm *dat; rm *tif; "
    for i in range(len(data_files)):
        os.makedirs(out_dirs[i], exist_ok = True)
        cmd_str = ""
        if instruments[i] == "viirs":
            cmd_str = cmd + "./viirs_l1b2gtiff.sh --debug -f "
        elif instruments[i] == "modis":
            cmd_str = cmd + "./modis2gtiff.sh --debug -f "
        cmd_str += f"{data_files[i]} {location_files[i]}; mv *tif {out_dirs[i]}"
        cmds.append(cmd_str)
    return cmds


#Assumes one set of tiffs per directory
#Generates a single zarr file from Polar2Grid generated tiffs
def combine_viirs_gtiffs(tiff_dirs):
    for i in range(len(tiff_dirs)):
        mtch = re.match(os.path.listdir(tiff_dirs[i])[0])
        data = None
        for j in range(1,17):
            fn = os.path.join(tiff_dirs[i], mtch.group(1) + "m" + str(j).zfill(2) + mtch.group(2) + "_wgs84_fit.tif")
            dat = gdal.Open(fn)
            band = dat.GetRasterBand(1).ReadAsArray()
            band[np.where(band < 0.0000000005)] = -9999
            if data is None:
                data = np.zeros((16, band.shape[0], band.shape[1]))
            data[j-1,:,:] = band
        fn = os.path.join(tiff_dirs[i], mtch.group(1) + "m" + str(i).zfill(2) + mtch.group(2) + ".zarr", data)
        zarr.save(fn, data)

        data = None 
        for j in range(1,6):
            fn = os.path.join(tiff_dirs[i], mtch.group(1) + "i" + str(j).zfill(2) + mtch.group(2) + "_wgs84_fit.tif")   
            dat = gdal.Open(fn)
            band = dat.GetRasterBand(1).ReadAsArray()
            band[np.where(band < 0.0000000005)] = -9999
            if data is None:
                data = np.zeros((5, band.shape[0], band.shape[1]))
            data[j-1,:,:] = band[:data.shape[1], :data.shape[2]]
        fn = os.path.join(tiff_dirs[i], mtch.group(1) + "i" + str(j).zfill(2) + mtch.group(2) + ".zarr", data)
        zarr.save(fn, data)        


#Assumes one set of tiffs per directory
#Generates a single zarr file from Polar2Grid generated tiffs
def combine_modis_gtiffs(tiff_dirs):
    for i in range(len(tiff_dirs)):
        mtch = re.match(os.path.listdir(tiff_dirs[i])[0])
        data = None
        for j in range(len(MODIS_BAND_ORDER)):
            fn = os.path.join(tiff_dirs[i], mtch.group(1) + MODIS_BAND_ORDER[j] + mtch.group(2) + "_wgs84_fit.tif")
            dat = gdal.Open(fn)
            band = dat.GetRasterBand(1).ReadAsArray()
            band[np.where(band < 0.0000000005)] = -9999
            if data is None:
                data = np.zeros((5, band.shape[0], band.shape[1]))
            data[j-1,:,:] = band[:data.shape[1], :data.shape[2]]
        fn = os.path.join(tiff_dirs[i], mtch.group(1) + MODIS_BAND_ORDER[j] + mtch.group(2) + ".zarr", data)
        zarr.save(fn, data)


def combine_modis_gtiffs_laads(file_list):
    for i in range(len(file_list)):
        data1 = []
        for j in range(len(file_list[i])):
            fn = file_list[i][j]
            dat = gdal.Open(fn)
            band = dat.GetRasterBand(1).ReadAsArray()
            band[np.where(band > 65535)] = -9999
            band[np.where(band < -0.0000000005)] = -9999 
            data1.append(band)
        dat = np.array(data1)
        fn = os.path.join(file_list[i][0] + "Full_Bands.zarr")
        zarr.save(fn, dat)     
        genLatLon([file_list[i][0]])

 
def dummyLatLonS6(fnames, from_data=False):

    data_read = get_read_func("s6_netcdf")
    geo_read = get_read_func("s6_netcdf_geo")
 
    for i in range(len(fnames)):
        fname = fnames[i]

        geo = geo_read(fname)
        dat = data_read(fname)

        print(dat.shape, geo[1].min(), geo[1].max())

        new_geo = np.zeros((2,dat.shape[1],dat.shape[2]))

        if from_data:        
            new_geo[0,:,0] = geo[0]
            new_geo[1,:,0] = geo[1]
            for j in range(dat.shape[1]):
                for k in range(1,dat.shape[2]):
                    if j == 0:
                        lat_spacing = abs(new_geo[0,0,0] - new_geo[0,1,0])
                        if k == 1:
                            lon_spacing = abs(new_geo[1,0,0] - new_geo[1,1,0])
                        else:
                            lon_spacing = abs(new_geo[1,j,k-1] - new_geo[1,j,k-2])

                        new_geo[0,j,k] = new_geo[0,j,k-1] + lat_spacing 
    
                    else:
                        lat_spacing = abs(new_geo[0,j,0] - new_geo[0,j-1,0])
                        if k == 1:
                            lon_spacing = abs(new_geo[1,j-1,k-1] - new_geo[1,j-1,k])
                        else:
                            lon_spacing = abs(new_geo[1,j,k-1] - new_geo[1,j,k-2])
 
                        new_geo[0,j,k] = new_geo[0,j-1,k] + lat_spacing

                    new_geo[1,j,k] = new_geo[1,j,k-1] + lon_spacing
        else:
            dt_lon = np.linspace(geo[1,0], geo[1,0] + (dat.shape[2]*0.002), dat.shape[2]) 
            dt_lat = np.linspace(geo[0,0], geo[0,0] + (dat.shape[1]*0.002), dat.shape[1])
          
            for j in range(dat.shape[1]):
                new_geo[1,j,:] = dt_lon
            for j in range(dat.shape[2]):
                new_geo[0,:,j] = dt_lat

        new_geo[1,:,:] = (new_geo[1,:,:] + 180) % 360 - 180
        print(new_geo[0].min(), new_geo[0].max(), new_geo[1].min(), new_geo[1].max())

        outFname = fname + ".lonlat.zarr"
        print(outFname)
        zarr.save(outFname, new_geo)


def genLatLon(fnames):

    for i in range(len(fnames)):
        fname = fnames[i]
        lonLat = get_lat_lon(fname)
 
        outFname = fname + ".lonlat.zarr"
        print(outFname)
        zarr.save(outFname, lonLat)



