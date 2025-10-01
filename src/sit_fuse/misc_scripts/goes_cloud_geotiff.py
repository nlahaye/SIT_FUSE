import numpy as np
from osgeo import gdal 
 
lst = [
        "/mnt/data/GOES_18_CA_COASTAL_FINAL/GOES_CLOUD/OR_ABI-L1b-RadC-M6C01_G18_s20250062201170_e20250062203543_c20250062203580.tif.clust.data_71199clusters.zarr.full_geo.cloud_mask.FullColor.tif",
        "/mnt/data/GOES_18_CA_COASTAL_FINAL/GOES_CLOUD/OR_ABI-L1b-RadC-M6C01_G18_s20250072301172_e20250072303545_c20250072303584.tif.clust.data_71199clusters.zarr.full_geo.cloud_mask.FullColor.tif",
        "/mnt/data/GOES_18_CA_COASTAL_FINAL/GOES_CLOUD/OR_ABI-L1b-RadC-M6C01_G18_s20250082201173_e20250082203546_c20250082203585.tif.clust.data_71198clusters.zarr.full_geo.cloud_mask.FullColor.tif",
        "/mnt/data/GOES_18_CA_COASTAL_FINAL/GOES_CLOUD/OR_ABI-L1b-RadC-M6C01_G18_s20250092201175_e20250092203548_c20250092203584.tif.clust.data_71199clusters.zarr.full_geo.cloud_mask.FullColor.tif",
        "/mnt/data/GOES_18_CA_COASTAL_FINAL/GOES_CLOUD/OR_ABI-L1b-RadC-M6C01_G18_s20250101601176_e20250101603549_c20250101603589.tif.clust.data_55296clusters.zarr.full_geo.cloud_mask.FullColor.tif",
        "/mnt/data/GOES_18_CA_COASTAL_FINAL/GOES_CLOUD/OR_ABI-L1b-RadC-M6C01_G18_s20250112301178_e20250112303551_c20250112303597.tif.clust.data_71197clusters.zarr.full_geo.cloud_mask.FullColor.tif",
        "/mnt/data/GOES_18_CA_COASTAL_FINAL/GOES_CLOUD/OR_ABI-L1b-RadC-M6C01_G18_s20250121901180_e20250121903553_c20250121904000.tif.clust.data_71199clusters.zarr.full_geo.cloud_mask.FullColor.tif",
        "/mnt/data/GOES_18_CA_COASTAL_FINAL/GOES_CLOUD/OR_ABI-L1b-RadC-M6C01_G18_s20250131801181_e20250131803554_c20250131803593.tif.clust.data_71191clusters.zarr.full_geo.cloud_mask.FullColor.tif",
        ]



for fle in lst:
    dat = gdal.Open(fle)
    x = dat.ReadAsArray()
    x[np.where(x < 0.0000000001)] = -1

    nx = x.shape[1] 
    ny = x.shape[0]
    metadata=dat.GetMetadata()
    geoTransform = dat.GetGeoTransform()
    wkt = dat.GetProjection()


    out_ds = gdal.GetDriverByName("GTiff").Create(fle, nx, ny, 1, gdal.GDT_Float32)
    out_ds.SetMetadata(metadata)
    out_ds.SetGeoTransform(geoTransform)
    out_ds.SetProjection(wkt)
    out_ds.GetRasterBand(1).WriteArray(x)
    out_ds.FlushCache() 
    out_ds = None  
 

