import os
import re
import numpy as np
from osgeo import osr, gdal
from subprocess import DEVNULL, run, Popen, PIPE

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


#Assumes each set of tiffs will be moved to a seperate directory
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
#Generates a single npy file from Polar2Grid generated tiffs
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

        fn = os.path.join(tiff_dirs[i], mtch.group(1) + "m" + str(i).zfill(2) + mtch.group(2) + ".npy", data)
 
       data = None 
       for j in range(1,6):
           fn = os.path.join(tiff_dirs[i], mtch.group(1) + "i" + str(j).zfill(2) + mtch.group(2) + "_wgs84_fit.tif")   
           dat = gdal.Open(fn)
           band = dat.GetRasterBand(1).ReadAsArray()
           band[np.where(band < 0.0000000005)] = -9999
           if data is None:
               data = np.zeros((5, band.shape[0], band.shape[1]))
           data[j-1,:,:] = band[:data.shape[1], :data.shape[2]]
       fn = os.path.join(tiff_dirs[i], mtch.group(1) + "i" + str(j).zfill(2) + mtch.group(2) + ".npy", data)
       

#Assumes one set of tiffs per directory
#Generates a single npy file from Polar2Grid generated tiffs
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
       fn = os.path.join(tiff_dirs[i], mtch.group(1) + MODIS_BAND_ORDER[j] + mtch.group(2) + ".npy", data)




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



 
        outFname = fname + ".lonlat.npy"
        np.save(outFname, lonLat)



