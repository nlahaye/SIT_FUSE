from osgeo import osr, gdal
import numpy as np
import os





data = ["/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19915_22_20190815_2231_2241_20201001_1745.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19916_20_20190817_0057_0108_20201001_1804.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19911_10_20190807_2004_2016_20201001_1655.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19919_13_20190821_2324_2339_20201001_1837.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19916_14_20190816_2313_2327_20201001_1759.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19919_14_20190821_2343_2356_20201001_1838.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19910_20_20190806_2052_2106_20201001_1636.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19916_18_20190817_0023_0036_20201001_1802.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19910_06_20190806_1815_1824_20201001_1621.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19919_17_20190822_0036_0104_20201001_1841.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19911_09_20190807_1947_2002_20201001_1654.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19916_17_20190817_0006_0020_20201001_1801.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19916_10_20190816_2201_2214_20201001_1755.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19910_10_20190806_1858_1910_20201001_1626.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19915_17_20190815_2122_2143_20201001_1741.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19916_15_20190816_2330_2345_20201001_1759.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19915_23_20190815_2247_2256_20201001_1746.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19916_11_20190816_2219_2234_20201001_1756.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19916_16_20190816_2349_0002_20201001_1800.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19916_19_20190817_0040_0054_20201001_1803.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19912_07_20190808_1806_1821_20201001_1708.nc","/Users/stasyaeasley/Desktop/ncfiles/eMASL2AER_19915_21_20190815_2216_2225_20201001_1744.nc"]

# ← use the mask, not the geolocation variable!
VAR = "/geophysical_data/Aerosol_Cldmask_Land_Ocean"
GEO_VAR = "/geophysical_data/Aerosol_Cloud_Fraction_Ocean"  # any geophysical var that had the GEOLOCATION metadata

for fle in data:
    # 1) grab GEOLOCATION metadata from a “known-good” band
    info = gdal.Open(f'NETCDF:"{fle}":{GEO_VAR}')
    geo_md = info.GetMetadata(domain="GEOLOCATION")
    info = None

    # 2) open your mask band
    src = gdal.Open(f'NETCDF:"{fle}":{VAR}')
    src.SetMetadata(geo_md, domain="GEOLOCATION")

    # 3) warp it into EPSG:4326, pulling in the lat/lon arrays
    mem = gdal.Warp(
        "", src,
        format="MEM",
        geoloc=True,
        dstSRS="EPSG:4326"
    )
    src = None

    # 4) read into numpy & threshold—check what codes your mask actually uses!
    arr = mem.GetRasterBand(1).ReadAsArray().astype(np.int16)
    print("mask codes:", np.unique(arr))   # e.g. [0,1] or maybe [0,1,2,…]

    # build a binary fire/smoke mask (1 where aerosol/cloud detected)
    fire_mask = (arr == 1).astype(np.uint8)

    # 5) write out the georeferenced GeoTIFF
    out_tif = os.path.splitext(fle)[0] + "_firemask.tif"
    gt = gdal.GetDriverByName("GTiff").CreateCopy(out_tif, mem, 0)
    gt.GetRasterBand(1).WriteArray(fire_mask)
    gt = None

    print("wrote:", out_tif)