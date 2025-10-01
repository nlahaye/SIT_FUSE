
import numpy as np
import dask
import dask.array as da
from sit_fuse.utils import read_oc_geo

#start_lon = -97.8985
#end_lon = -80.5301
#start_lat = 18.1599
#end_lat = 30.4159
 
#start_lon = -128.00
#end_lon = -116.00
#start_lat = 30.00
#end_lat = 38.94


start_lon = 32.0
end_lon = 44.0
start_lat = 12.0
end_lat = 29.0

tmp = read_oc_geo("/mnt/data/REFLECTANCES/S3A_OLCI/S3A_OLCI_ERRNT.20250112.L3m.DAY.")


print(tmp[0].shape)
print(tmp[1].shape)
print(tmp[0])
print(tmp[1])


loc = tmp
#tmp[0].shape, tmp[1].shape)

lat = np.array(loc[0])
lon = np.array(loc[1])
inds1 = np.where((lat >= start_lat) & (lat <= end_lat))
inds2 = np.where((lon >= start_lon) & (lon <= end_lon))

print(inds2[0])
nind1, nind2 = np.meshgrid(inds2[0], inds1[0])
lat = lat[inds1]
lon = lon[inds2]
loc = np.array(np.meshgrid(lat, lon))

loc = np.moveaxis(loc, 0,2)
loc = np.swapaxes(loc,0,1)
print(loc.shape)

print(min(lat), min(lon), max(lat), max(lon))

data2 = da.from_array(loc)
#da.to_zarr(data2, "/data/nlahaye/remoteSensing/TROPOMI_MODIS_HAB/gulf_of_mex_geo.zarr")
da.to_zarr(data2, "/mnt/data/red_sea.zarr")


