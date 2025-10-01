import sys
import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling

fnames = [
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_100.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_101.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_102.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_103.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_104.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_105.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_106.tif.clust.data_79197clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_109.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_10.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_110.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_111.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_112.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_113.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_114.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_115.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_116.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_117.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_118.tif.clust.data_79197clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_121.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_122.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_123.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_124.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_125.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_126.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_127.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_128.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_129.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_130.tif.clust.data_79198clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_133.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_134.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_135.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_136.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_137.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_138.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_139.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_140.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_141.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_142.tif.clust.data_79197clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_16.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_17.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_18.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_19.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_20.tif.clust.data_79198clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_22.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_23.tif.clust.data_79197clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_27.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_28.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_29.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_30.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_31.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_32.tif.clust.data_79195clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_37.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_38.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_39.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_40.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_41.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_42.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_43.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_44.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_45.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_46.tif.clust.data_79190clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_48.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_49.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_50.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_51.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_52.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_53.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_54.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_55.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_56.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_57.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_58.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_59.tif.clust.data_79131clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_5.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_60.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_61.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_62.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_63.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_64.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_65.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_66.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_67.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_68.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_69.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_6.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_70.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_72.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_73.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_74.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_75.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_76.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_77.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_78.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_79.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_7.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_80.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_81.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_82.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_83.tif.clust.data_79198clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_84.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_85.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_86.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_87.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_88.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_89.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_8.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_90.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_91.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_92.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_93.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_94.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_96.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_97.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_98.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_99.tif.clust.data_79199clusters.zarr.full_geo.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_9.tif.clust.data_79199clusters.zarr.full_geo.tif",
]
 
#road_mask = "/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pv_polygons.tif"
road_mask = "/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/HLS-VI.L30.T28PCV.2025139T112723.v2.0.SAVI.tiff"



def mask_raster_with_raster(target_raster_path, mask_raster_path, output_path):
    """Masks a target GeoTIFF using a mask GeoTIFF."""

    with rasterio.open(target_raster_path) as target_src:
        target_data = target_src.read()
        target_meta = target_src.meta.copy()

        with rasterio.open(mask_raster_path) as mask_src:
            mask_data = mask_src.read(1) * 0.0001
            mask_meta = mask_src.meta.copy()
            print(mask_data.max(), mask_data.min())

            # Check if mask and target have the same dimensions and transform
            if mask_data.shape != target_data.shape[1:3] or mask_meta['transform'] != target_meta['transform']:
                # Resample mask to target's geometry
                #transform, width, height = calculate_default_transform(
                #    mask_src.crs, target_src.crs, target_src.width, target_src.height, *target_src.bounds)

                height = target_src.height
                width = target_src.width
                mask_resampled = np.empty((1, height, width), dtype=mask_data.dtype)
                print(mask_resampled.shape, height, width)

                print("resampling")

                reproject(
                    source=mask_data,
                    destination=mask_resampled,
                    src_transform=mask_src.transform,
                    src_crs=mask_src.crs,
                    dst_transform=target_src.transform,
                    dst_crs=target_src.crs,
                    resampling=Resampling.cubic
                )
                print("resampled")
                print(mask_data.shape,  mask_resampled.shape, target_data.shape, target_src.width, target_src.height)
                print( mask_resampled.min(), mask_resampled.max(), mask_resampled.shape)

                mask_data = mask_resampled[0] #.astype(bool)
            else:
                mask_data = mask_data #.astype(bool)


            
            print(mask_data.min(), mask_data.max())
            inds = np.where((mask_data < 0.1) | (mask_data > 0.4))
            mask_data[inds] = 0
            inds = np.where(mask_data >= 0.1)
            mask_data[inds] = 1

            mask_data = mask_data.astype(bool)

            print(mask_data.min(), mask_data.max())
            # Apply the mask
            mask_data = np.expand_dims(mask_data, 0)
            target_data[np.where(mask_data == 0)] = 0
            masked_data = target_data
            print(masked_data.min(), masked_data.max())
            print(mask_data.min(), mask_data.max(), mask_data.shape)
            #mask_data = np.expand_dims(mask_data, 0)
            #masked_data = np.squeeze(masked_data)
 

            """
            # Write the output
            target_meta.update({
                "driver": "GTiff",
                "height": mask_data.shape[0], #masked_data.shape[1],
                "width": mask_data.shape[1], #masked_data.shape[2],
                "count": 1,
                "dtype": "int16", #"float32",
                "nodata": 0 #-9999
            })

            with rasterio.open(output_path + "PV_POLYS.tif", 'w', **target_meta) as dest_src:
                dest_src.write(mask_data) #masked_data)
            """

            # Write the output
            target_meta.update({
                "driver": "GTiff",
                "height": masked_data.shape[1], #masked_data.shape[1],
                "width": masked_data.shape[2], #masked_data.shape[2],
                "count": 1,
                "dtype": "float32",
                "nodata": -999
            })

            with rasterio.open(output_path + ".masked.tif", 'w', **target_meta) as dest_src:
                dest_src.write(masked_data) #masked_data)


for i in range(len(fnames)):
    target_raster = fnames[i]
    mask_raster = road_mask
    output_raster = fnames[i]

    mask_raster_with_raster(target_raster, mask_raster, output_raster)
    print(f"Masked raster saved to {output_raster}.masked.tif")
 


