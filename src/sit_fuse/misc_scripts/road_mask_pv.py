import numpy as np
import rasterio
import rasterio.mask
import fiona
import regionmask

ocean_basins_50 =  regionmask.defined_regions.natural_earth_v5_1_2.ocean_basins_50


fnames = [
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_0.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_100.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_101.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_102.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_103.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_104.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_105.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_106.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_107.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_108.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_109.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_10.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_110.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_111.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_112.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_113.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_114.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_115.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_116.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_117.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_118.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_119.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_11.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_120.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_121.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_122.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_123.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_124.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_125.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_126.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_127.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_128.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_129.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_12.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_130.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_131.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_132.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_133.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_134.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_135.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_136.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_137.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_138.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_139.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_13.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_140.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_141.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_142.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_143.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_14.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_15.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_16.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_17.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_18.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_19.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_1.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_20.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_21.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_22.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_23.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_24.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_25.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_26.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_27.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_28.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_29.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_2.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_30.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_31.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_32.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_33.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_34.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_35.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_36.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_37.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_38.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_39.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_3.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_40.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_41.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_42.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_43.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_44.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_45.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_46.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_47.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_48.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_49.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_4.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_50.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_51.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_52.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_53.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_54.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_55.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_56.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_57.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_58.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_59.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_5.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_60.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_61.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_62.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_63.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_64.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_65.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_66.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_67.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_68.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_69.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_6.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_70.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_71.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_72.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_73.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_74.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_75.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_76.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_77.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_78.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_79.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_7.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_80.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_81.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_82.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_83.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_84.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_85.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_86.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_87.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_88.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_89.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_8.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_90.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_91.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_92.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_93.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_94.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_95.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_96.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_97.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_98.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_99.tif",
#"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_9.tif",
#"/data/nlahaye/remoteSensing/S1_Gambia/s1a_iw_grd_vv_gambia_20240224_clipped_ln.tif",
#"/data/nlahaye/remoteSensing/S1_Gambia/s1a_iw_grd_vh_gambia_20240224_clipped_ln.tif",
##"/data/nlahaye/remoteSensing/S2_PV/HLS.S30.T28PCV.2024057T113319.v2.0.B11.lonlat.tif",
##"/data/nlahaye/remoteSensing/S2_PV/HLS.S30.T28PCV.2024057T113319.v2.0.B10.lonlat.tif",
##"/data/nlahaye/remoteSensing/S2_PV/HLS.S30.T28PCV.2024057T113319.v2.0.B12.lonlat.tif",
##"/data/nlahaye/remoteSensing/PV_S2/HLS.L30.T28PCV.2024041T112810.v2.0.B10.lonlat.tif",
##"/data/nlahaye/remoteSensing/PV_S2/HLS.L30.T28PCV.2024041T112810.v2.0.B11.lonlat.tif",
#"/data/nlahaye/remoteSensing/PV_S2/HLS.L30.T28PCV.2024041T112810.v2.0.B10.lonlat.tif",
#"/data/nlahaye/remoteSensing/PV_S2/HLS.L30.T28PCV.2024041T112810.v2.0.B11.lonlat.tif",
#"/data/nlahaye/remoteSensing/PV_S2/HLS.S30.T28PCV.2024052T113321.v2.0.B10.lonlat.tif",
#"/data/nlahaye/remoteSensing/PV_S2/HLS.S30.T28PCV.2024052T113321.v2.0.B11.lonlat.tif",
#"/data/nlahaye/remoteSensing/PV_S2/HLS.S30.T28PCV.2024052T113321.v2.0.B12.lonlat.tif",
"/data/nlahaye/remoteSensing/S1_Gambia/s1a_iw_grd_vh_gambia_20240224.tif",
"/data/nlahaye/remoteSensing/S1_Gambia/s1a_iw_grd_vv_gambia_20240224.tif"
] 


out_fnames = [
"/data/nlahaye/remoteSensing/S1_Gambia/s1a_iw_grd_vh_gambia_20240224_ln.tif",
"/data/nlahaye/remoteSensing/S1_Gambia/s1a_iw_grd_vv_gambia_20240224_ln.tif"
]

road_mask = "/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/roads.geojson"

for i in range(len(fnames)):
 

    raster_data = None
    raster_transform = None
    raster_crs = None
    raster_meta = None
    lat = None
    lon = None
    # Load your raster data
    print(fnames[i])
    with rasterio.open(fnames[i]) as src:
        raster_transform = src.transform
        raster_crs = src.crs
        raster_meta = src.meta
        raster_profile = src.profile
        raster_profile['dtype'] = "float32"
        raster_data = src.read()  # Read a single band

        # Create a land mask using regionmask
        # Adjust grid parameters to match your raster's extent and resolution
        lon = np.arange(src.bounds.left, src.bounds.right, src.res[0])
        lat = np.arange(src.bounds.bottom, src.bounds.top, src.res[1])

    
    print(lon)
    print(lat)
    # Use a predefined landmask like 'natural_earth_v5_0_0.land_110'
    land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_10.mask(lon, lat) 
    tmp1 = (~land_mask.isnull().to_numpy()).astype(np.bool_)
    print(land_mask.max(), land_mask.min())
    print(tmp1.min(), tmp1.max(), tmp1.shape)

    # Reshape the land mask to match the raster dimensions
    land_mask_reshaped = np.squeeze(tmp1)
    land_mask_reshaped = np.flipud(land_mask_reshaped)  # Flip if necessary

    # Apply the mask to the raster data
    max_y = raster_data.shape[-2]
    max_x = raster_data.shape[-1]
    land_mask_reshaped = land_mask_reshaped[:max_y, :max_x]   

    raster_data = raster_data.astype(np.float32) 
    print(raster_data.min(), raster_data.max())
    masked_data = np.where(land_mask_reshaped == True, raster_data, 0) #-9999) # Assuming land_mask==1 for land
    print(masked_data.min(), masked_data.max())
    inds = np.where(masked_data > 0)
    masked_data[inds] = np.log(masked_data[inds])
    print(masked_data.min(), masked_data.max())

    print(masked_data.min(), masked_data.max(), masked_data.shape, fnames[i])

    raster_meta.update({
        "driver": "GTiff",
        "height": masked_data.shape[1],
        "count": 1
        "width": masked_data.shape[2],
        "transform": raster_transform,
        "crs": raster_crs,
        "dtype": "int16", #"float32",
        "nodata": 0 #-9999
    })

    with rasterio.open(out_fnames[i], "w", **raster_meta) as dest:
        dest.profile['dtype'] = "float32"
        dest.write(masked_data)



