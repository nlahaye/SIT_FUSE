from osgeo import osr, gdal
import numpy as np
import os





data = ["/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192181816196_e20192181818569_c20192181819111.nc",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192181926196_e20192181928569_c20192181929117.nc",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192182036196_e20192182038569_c20192182039106.nc",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192182151196_e20192182153569_c20192182154117.nc",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192191821196_e20192191823569_c20192191824120.nc",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192191931196_e20192191933569_c20192191934133.nc",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192191956196_e20192191958569_c20192191959111.nc",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192192006196_e20192192008569_c20192192009118.nc",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192192026196_e20192192028569_c20192192029115.nc",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192192041196_e20192192043569_c20192192044115.nc",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192200056197_e20192200058570_c20192200059114.nc",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192200236197_e20192200238569_c20192200239114.nc",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192201851196_e20192201853569_c20192201854128.nc",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192201931196_e20192201933569_c20192201934124.nc",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192210116197_e20192210118570_c20192210119119.nc",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192210121197_e20192210123570_c20192210124125.nc",
"/data/nlahaye/remoteSensing/GOES_FIRE/OR_ABI-L2-FDCC-M6_G17_s20192210221197_e20192210223570_c20192210224139.nc"]

def getGeoT(extent, nlines, ncols):
    # Compute resolution based on data dimension
    resx = (extent[2] - extent[0]) / ncols
    resy = (extent[3] - extent[1]) / nlines
    return [extent[0], resx, 0, extent[3] , 0, -resy]
 


for fle in data:
    
    dat = gdal.Open("NETCDF:\"" + fle + "\":Mask")


    # Define KM_PER_DEGREE
    KM_PER_DEGREE = 111.32

    # GOES-16 Extent (satellite projection) [llx, lly, urx, ury]
    GOES16_EXTENT = [-5434894.885056, -5434894.885056, 5434894.885056, 5434894.885056]

    # GOES-16 Spatial Reference System
    sourcePrj = osr.SpatialReference()
    sourcePrj.ImportFromProj4('+proj=geos +h=35786023.0 +a=6378137.0 +b=6356752.31414 +f=0.00335281068119356027 +lat_0=0.0 +lon_0=-89.5 +sweep=x +no_defs')


    # Setup projection and geo-transformation
    dat.SetProjection(sourcePrj.ExportToWkt())
    dat.SetGeoTransform(getGeoT(GOES16_EXTENT, dat.RasterYSize, dat.RasterXSize))
 
    rad = dat.ReadAsArray()

    tmp = np.zeros(rad.shape)
    tmp[np.where((rad > 10) & ((rad < 16) | ((rad > 29) & (rad < 36))))] = 1
    print(tmp.min(), tmp.max())

    sizex = tmp.shape[1] 
    sizey = tmp.shape[0]

    ## Lat/lon WSG84 Spatial Reference System
    #targetPrj = osr.SpatialReference()
    #targetPrj.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
 

    # Get memory driver
    memDriver = gdal.GetDriverByName('MEM')
   
    # Create grid
    grid = memDriver.Create('grid', dat.RasterXSize, dat.RasterYSize, 1, gdal.GDT_Float32)
    grid.SetProjection(sourcePrj.ExportToWkt())
    grid.SetGeoTransform(dat.GetGeoTransform())
    grid.GetRasterBand(1).WriteArray(tmp)


    driver = gdal.GetDriverByName('GTiff')
    driver.CreateCopy(os.path.splitext(fle)[0] + ".tif", grid, 0)
 

