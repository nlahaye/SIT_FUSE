import os
import re
import zarr
import glob
import cv2
import numpy as np
from pprint import pprint
from osgeo import osr, gdal
from subprocess import DEVNULL, run, Popen, PIPE
from scipy.ndimage import uniform_filter
from scipy.ndimage import variance

import sys


hours = ["13","14","15","16", "17", "18", "19", "20", "21", "22"]

cm_thresh = [0.2, 0.3]


days = ["26",]

#"25", "25", "25", "25", "25", "25", "25", "25", "26", "26", "26", "26", "26", "26", "26", "26", "26", "26"] 

dat_fire = [
 
#"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242071406175_e20242071408548_c20242071408582.fire.resampled.tif",
#"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242071501175_e20242071503548_c20242071504006.fire.resampled.tif",
#"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242071601175_e20242071603548_c20242071603595.fire.resampled.tif",
#"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242071701175_e20242071703548_c20242071703583.fire.resampled.tif",
#"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242071801175_e20242071803548_c20242071804022.fire.resampled.tif",
#"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242071901175_e20242071903548_c20242071903582.fire.resampled.tif",
#"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242072001175_e20242072003548_c20242072003587.fire.resampled.tif",
#"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242072101176_e20242072103549_c20242072104027.fire.resampled.tif",
#"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242072201176_e20242072203549_c20242072203597.fire.resampled.tif",
#"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242072301176_e20242072303549_c20242072303587.fire.resampled.tif",


"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242081401177_e20242081403550_c20242081403597.fire.resampled.tif",
"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242081501177_e20242081503550_c20242081503592.fire.resampled.tif",
"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242081601177_e20242081603550_c20242081603591.fire.resampled.tif",
"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242081701177_e20242081703550_c20242081703592.fire.resampled.tif",
"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242081801177_e20242081803550_c20242081803594.fire.resampled.tif",
"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242081901177_e20242081903550_c20242081904000.fire.resampled.tif",
"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242082001177_e20242082003550_c20242082003591.fire.resampled.tif",
"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242082101177_e20242082103550_c20242082103590.fire.resampled.tif",
"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242082201177_e20242082203550_c20242082204028.fire.resampled.tif",
"/home/nlahaye/SIT_FUSE/src/sit_fuse/pipelines/context_assign/OR_ABI-L1b-RadC-M6C01_G18_s20242082301177_e20242082303550_c20242082303596.fire.resampled.tif",
]


dat_smoke = [

#"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242071406175_e20242071408548_c20242071408582.tif.clust.data_63392clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",
#"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242071501175_e20242071503548_c20242071504006.tif.clust.data_63397clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",
#"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242071601175_e20242071603548_c20242071603595.tif.clust.data_64799clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",
#"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242071701175_e20242071703548_c20242071703583.tif.clust.data_64799clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",
#"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242071801175_e20242071803548_c20242071804022.tif.clust.data_64799clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",
#"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242071901175_e20242071903548_c20242071903582.tif.clust.data_64799clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",
#"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242072001175_e20242072003548_c20242072003587.tif.clust.data_64799clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",
#"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242072101176_e20242072103549_c20242072104027.tif.clust.data_64799clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",
#"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242072201176_e20242072203549_c20242072203597.tif.clust.data_64799clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",
#"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242072301176_e20242072303549_c20242072303587.tif.clust.data_64799clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",


"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242081401177_e20242081403550_c20242081403597.tif.clust.data_63392clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",
"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242081501177_e20242081503550_c20242081503592.tif.clust.data_63398clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",
"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242081601177_e20242081603550_c20242081603591.tif.clust.data_64799clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",
"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242081701177_e20242081703550_c20242081703592.tif.clust.data_64799clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",
"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242081801177_e20242081803550_c20242081803594.tif.clust.data_64799clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",
"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242081901177_e20242081903550_c20242081904000.tif.clust.data_64799clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",
"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242082001177_e20242082003550_c20242082003591.tif.clust.data_64799clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",
"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242082101177_e20242082103550_c20242082103590.tif.clust.data_64799clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",
"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242082201177_e20242082203550_c20242082204028.tif.clust.data_64799clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",
"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/OR_ABI-L1b-RadC-M6C01_G18_s20242082301177_e20242082303550_c20242082303596.tif.clust.data_64799clusters.zarr.smoke.HOURLY_MOE.tif.Contours.resampled.tif",

]



ind = -1 
for k in range(len(days)):
    for i in range(len(hours)):
        ind = ind + 1
        for j in range(len(cm_thresh)):
 
            print("/data/nlahaye/remoteSensing/NO2_L3_V03/TEMPO_NO2_L3_V03_202407" + days[k] + "T" + hours[i] + "*_subsetted.cloud.tif")
            data_tempo = sorted(glob.glob("/data/nlahaye/remoteSensing/NO2_L3_V03/TEMPO_NO2_L3_V03_202407" + days[k] + "T" + hours[i] + "*_subsetted.tif"))[-1]
            tempo_cm = sorted(glob.glob("/data/nlahaye/remoteSensing/NO2_L3_V03/TEMPO_NO2_L3_V03_202407" + days[k] + "T" + hours[i] + "*_subsetted.cloud.tif"))[-1]

  
            data_fire = dat_fire[ind] #"/data/nlahaye/SIT_FUSE_Geo/GOES_TEMPO_DATA/GOES_TEMPO_FIRE_2.resampled.tif"
            data_smoke = dat_smoke[ind] #"/data/nlahaye/output/Learnergy/GOES_TEMPO_UPDATED/smoke/OR_ABI-L1b-RadC-M6C01_G18_s20242081601177_e20242081603550_c20242081603591.smoke.resampled.tif"
         
            cm = gdal.Open(tempo_cm).ReadAsArray()


            dat1 = gdal.Open(data_smoke).ReadAsArray()
            dat2 = gdal.Open(data_fire).ReadAsArray()

            dat_t = gdal.Open(data_tempo).ReadAsArray()

            print(data_tempo , tempo_cm, data_fire, data_smoke)
            print("SMOKE", "FIRE", "DATA")
            print(dat1.shape, dat2.shape, dat_t.shape)

            #sys.exit(0)

            mask = np.where(((cm > cm_thresh[j]) & ((dat1 <= 0) & (dat2 <= 0))))
            #mask = np.where(((cm > cm_thresh[j])))
            dat_t[mask] = -999999
 

            dat = gdal.Open(data_smoke)
            geoTransform = dat.GetGeoTransform()
            wkt = dat.GetProjection()
            dat.FlushCache()

            out_fname = os.path.splitext(data_tempo)[0] + "cm_" + str(cm_thresh[j]) + ".masked.tif"
            #out_fname = os.path.splitext(data_tempo)[0] + "cm_" + str(cm_thresh[j]) + ".op_masked.tif"
            print(out_fname)
    
            nx = dat1.shape[1]
            ny = dat1.shape[0]

            out_ds = gdal.GetDriverByName("GTiff").Create(out_fname, nx, ny, 1, gdal.GDT_Float32)
            out_ds.SetGeoTransform(geoTransform)
            out_ds.SetProjection(wkt)
            out_ds.GetRasterBand(1).WriteArray(dat_t)

out_ds.FlushCache()
out_ds = None
del dat1
dat.FlushCache()
dat = None





