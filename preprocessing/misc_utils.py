import os
import re
import zarr
import numpy as np
from osgeo import osr, gdal
from subprocess import DEVNULL, run, Popen, PIPE

from utils import numpy_to_torch, read_yaml, get_read_func

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
 
        # open the dataset and get the geo transform matrix
        ds = gdal.Open(fname)
        xoffset, px_w, rot1, yoffset, px_h, rot2 = ds.GetGeoTransform()
        dataArr = ds.ReadAsArray()

        lonLat = np.zeros((dataArr.shape[0], dataArr.shape[1], 2))

        # get CRS from dataset 
        crs = osr.SpatialReference()
        crs.ImportFromWkt(ds.GetProjectionRef())

        # create lat/long crs with WGS84 datum
        crsGeo = osr.SpatialReference()
        crsGeo.ImportFromEPSG(4326) # 4326 is the EPSG id of lat/long crs 
        t = osr.CoordinateTransformation(crs, crsGeo)
        for j in range(dataArr.shape[1]):
            for k in range(dataArr.shape[0]):
                posX = px_w * j + rot1 * k + (px_w * 0.5) + (rot1 * 0.5) + xoffset
                posY = px_h * j + rot2 * k + (px_h * 0.5) + (rot2 * 0.5) + yoffset
 
                (lon, lat, z) = t.TransformPoint(posX, posY)
                lonLat[k,j,1] = lon
                lonLat[k,j,0] = lat

 
        outFname = fname + ".lonlat.zarr"
        print(outFname)
        zarr.save(outFname, lonLat)



