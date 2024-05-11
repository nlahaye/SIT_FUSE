from osgeo import osr, gdal
import numpy as np
import os





data = [
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192181816196_e20192181818569_c20192181819111.tif",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192181926196_e20192181928569_c20192181929117.tif",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192182036196_e20192182038569_c20192182039106.tif",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192182151196_e20192182153569_c20192182154117.tif",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192191821196_e20192191823569_c20192191824120.tif",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192191931196_e20192191933569_c20192191934133.tif",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192191956196_e20192191958569_c20192191959111.tif",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192192006196_e20192192008569_c20192192009118.tif",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192192026196_e20192192028569_c20192192029115.tif",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192192041196_e20192192043569_c20192192044115.tif",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192200056197_e20192200058570_c20192200059114.tif",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192200236197_e20192200238569_c20192200239114.tif",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192201851196_e20192201853569_c20192201854128.tif",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192201931196_e20192201933569_c20192201934124.tif",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192210116197_e20192210118570_c20192210119119.tif",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192210121197_e20192210123570_c20192210124125.tif",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192210221197_e20192210223570_c20192210224139.tif"
]


for fle in data:
    
    dat = gdal.Open(fle)
    rad = dat.ReadAsArray()

    tmp = np.zeros(rad.shape)
    tmp[np.where((rad > 10) & ((rad < 16) | ((rad > 29) & (rad < 36))))] = 1
    print(tmp.min(), tmp.max())

    sizex = tmp.shape[1] 
    sizey = tmp.shape[0]

    geoTransform = dat.GetGeoTransform()
    wkt = dat.GetProjection()
    dat.FlushCache()
    dat = None


    out_ds = gdal.GetDriverByName("GTiff").Create(os.path.splitext(fle)[0] + ".bin.tif", sizex, sizey, 1, gdal.GDT_Int32)

    out_ds.SetGeoTransform(geoTransform)
    out_ds.SetProjection(wkt)
    out_ds.GetRasterBand(1).WriteArray(tmp)
    out_ds.FlushCache()
    out_ds = None
 
 

