"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""

import numpy as np
from sit_fuse.utils import numpy_to_torch, read_yaml, insitu_hab_to_tif
from osgeo import gdal, osr
import argparse
import os
from pandas import DataFrame as df
from skimage.util import view_as_windows
from copy import deepcopy
import re
import datetime
from pprint import pprint

DATE_RE = ".*(\d{8}).*"



fname_res = ["(sif_finalday_\d+).*karenia_brevis_bloom.tif", ".*(\d{8}).*karenia_brevis_bloom.tif", ".*(\d{8}).*no_heir.*karenia_brevis_bloom.tif", ".*(\d{8}).*pseudo_nitzschia_seriata_bloom.tif", ".*(\d{8}).*pseudo_nitzschia_delicatissima_bloom.tif", ".*(\d{8}).*alexandrium_bloom.tif", ".*(\d{8}).*total_phytoplankton.tif"]
 
def merge_datasets(paths, fname_str, out_dir, re_index = 0, base_index = 0): 
    for root, dirs, files in os.walk(paths[base_index]):
        for fle in files:
            if fname_str in fle:
                if "OR_ABI" in fle or "_G18_" in fle:
                    goes_match = re.search(r"s(20\d{2})(\d{3})\d{6}", fle)
                    year = goes_match.group(1)
                    doy = goes_match.group(2)
                    date_obj = datetime.datetime.strptime(year + doy, "%Y%j")
                    new_fname_root = date_obj.strftime("%Y%m%d")
                else:
                    new_fname_root = re.search(fname_res[re_index], fle).group(1)
                fname = os.path.join(out_dir, new_fname_root + "_" + fname_str)
                dqi_fname = os.path.splitext(fname)[0] + ".DQI.tif"
                data = None
                qual = None
                if os.path.exists(fname):
                    data = gdal.Open(fname).ReadAsArray()
                    qual = gdal.Open(dqi_fname).ReadAsArray()
                fle1 = os.path.join(root, fle)
                dat1 = gdal.Open(fle1)
                tmp = dat1.ReadAsArray()  
                if data is None:
                    imgData1 = np.zeros(tmp.shape) - 1
                else:
                    imgData1 = data

                if qual is None:
                    dqi = np.zeros(imgData1.shape) - 1
                else:
                    dqi = qual

                inds = np.where((imgData1 < 0) & (tmp >= 0))
                imgData1[inds] = tmp[inds] 

                dqi[inds] = base_index                
                for j in range(base_index+1, len(paths)):
                    fle2 = os.path.join(paths[j], fle)
                    if os.path.exists(fle2):
                        dat2 = gdal.Open(fle2)
                        imgData2 = dat2.ReadAsArray()
                        inds = np.where((imgData1 < 0) & (imgData2 >= 0))
                        imgData1[inds] = imgData2[inds]
                        dqi[inds] = j
                        #imgData1[inds] = 0
                        #dat2.FlushCache()
                        dat2 = None
                
                nx = imgData1.shape[1]
                ny = imgData1.shape[0]
                geoTransform = dat1.GetGeoTransform()
                wkt = dat1.GetProjection()
                dat1.FlushCache()
                dat1 = None
          
                out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Int16)
                out_ds.SetGeoTransform(geoTransform)
                out_ds.SetProjection(wkt)
                out_ds.GetRasterBand(1).WriteArray(imgData1)
                out_ds.FlushCache()
                out_ds = None
  
                out_ds = gdal.GetDriverByName("GTiff").Create(dqi_fname, nx, ny, 1, gdal.GDT_Int16)
                out_ds.SetGeoTransform(geoTransform)
                out_ds.SetProjection(wkt)
                out_ds.GetRasterBand(1).WriteArray(dqi)
                out_ds.FlushCache()
                out_ds = None




#assumes rename to having date in filename
def merge_monthly(dirname, fname_str):

    monthlies = {}

    for root, dirs, files in os.walk(dirname):
        for fle in files:
            if "DQI" in fle:
                continue
            if fname_str not in fle:
                continue
            mtch = re.search(DATE_RE, fle)
            if mtch:
                dte = datetime.datetime.strptime(mtch.group(1), "%Y%m%d")
                if dte.year in monthlies.keys():
                    if dte.month in monthlies[dte.year].keys():
                        monthlies[dte.year][dte.month].append([dte, os.path.join(root, fle)])
                    else:
                        monthlies[dte.year][dte.month] = [[dte, os.path.join(root, fle)]]
                else:
                    monthlies[dte.year] = {dte.month : [[dte, os.path.join(root, fle)]]}
 
    pprint(monthlies.keys())
    for yr in monthlies.keys():
        for mnth in monthlies[yr].keys():
            newImgData = None
            newDqi = None
            dat = None
            pixCnt = None
            for j in range(0, len(monthlies[yr][mnth])):
                      
                dqi_fname = os.path.splitext(monthlies[yr][mnth][j][1])[0] + ".DQI.tif"
 
                dat = gdal.Open(monthlies[yr][mnth][j][1]) 
                imgData = dat.ReadAsArray()

                dqi = gdal.Open(dqi_fname).ReadAsArray()
 
                if newImgData is None:
                    newImgData = np.zeros(imgData.shape)
                    newDqi = np.zeros(imgData.shape)
                    pixCnt = np.zeros(imgData.shape) 
                
                inds = np.where((dqi >= 0) & (imgData >= 0))
                    
                newImgData[inds] += imgData[inds]
                newDqi[inds] += dqi[inds]
                pixCnt[inds] += 1

        
            inds = np.where(pixCnt < 1)
            pixCnt[inds] = 1

            newImgData = np.round(np.divide(newImgData, pixCnt)).astype(np.int32)
            newDqi = np.round(np.divide(newDqi, pixCnt)).astype(np.int32)  

            newImgData[inds] = -1
            newDqi[inds] = -1
            pixCnt[inds] = 0
            print("HERE ", mnth, yr)


            nx = newImgData.shape[1]
            ny = newImgData.shape[0]
            geoTransform = dat.GetGeoTransform()
            wkt = dat.GetProjection()
            dat.FlushCache()
            dat = None
    
            out_ds = gdal.GetDriverByName("GTiff").Create(os.path.join(dirname, monthlies[yr][mnth][0][0].strftime("%Y%m") + "_" + fname_str + ".Monthly.tif"), nx, ny, 1, gdal.GDT_Int32)
            out_ds.SetGeoTransform(geoTransform)
            out_ds.SetProjection(wkt)
            out_ds.GetRasterBand(1).WriteArray(newImgData)
            out_ds.FlushCache()
            out_ds = None
            
            out_ds = gdal.GetDriverByName("GTiff").Create(os.path.join(dirname, monthlies[yr][mnth][0][0].strftime("%Y%m") + "_" + fname_str + ".Monthly.DQI.tif"), nx, ny, 1, gdal.GDT_Int32)
            out_ds.SetGeoTransform(geoTransform)
            out_ds.SetProjection(wkt)
            out_ds.GetRasterBand(1).WriteArray(newDqi)
            out_ds.FlushCache()
            out_ds = None    



def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    data = None
    qual = None
 
    if yml_conf["gen_daily"]:
        for i in range(len(yml_conf['input_paths'])):
            merge_datasets(yml_conf['input_paths'], 
                yml_conf['fname_str'], yml_conf['out_dir'], yml_conf['re_index'], i) 
    if yml_conf["gen_monthly"]:
        merge_monthly(yml_conf['dirname'], yml_conf['fname_str'])  
 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    main(args.yaml)
