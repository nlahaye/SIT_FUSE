import os
from pprint import pprint



fnames = [
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_6.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_77.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_78.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_100.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_101.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_102.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_103.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_104.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_105.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_106.tif.clust.data_79197clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_109.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_10.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_110.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_111.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_112.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_113.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_114.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_115.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_116.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_117.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_118.tif.clust.data_79197clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_121.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_122.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_123.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_124.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_125.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_126.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_127.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_128.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_129.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_130.tif.clust.data_79198clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_133.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_134.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_135.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_136.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_137.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_138.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_139.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_140.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_141.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_142.tif.clust.data_79197clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_16.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_17.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_18.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_19.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_20.tif.clust.data_79198clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_22.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_23.tif.clust.data_79197clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_27.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_28.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_29.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_30.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_31.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_32.tif.clust.data_79195clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_37.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_38.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_39.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_40.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_41.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_42.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_43.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_44.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_45.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_46.tif.clust.data_79190clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_48.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_49.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_50.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_51.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_52.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_53.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_54.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_55.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_56.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_57.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_58.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_59.tif.clust.data_79131clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_5.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_60.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_61.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_62.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_63.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_64.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_65.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_66.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_67.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_68.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_69.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_70.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_72.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_73.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_74.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_75.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_76.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_79.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_7.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_80.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_81.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_82.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_83.tif.clust.data_79198clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_84.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_85.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_86.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_87.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_88.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_89.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_8.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_90.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_91.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_92.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_93.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_94.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_96.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_97.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_98.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_99.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
#"/data/nlahaye/output/Learnergy/DBN_PV_MAPPING_WIDE_ROAD_CLIP/pleides_RGBNED_9.tif.clust.data_79199clusters.zarr.full_geo.tif.masked.tif.PV_Map_Tiered_Mask.tif.Contours.tif.FullColorContour.tif",
]


ext_1 = ".tile_cluster."
ext_2 = ".tif"
tiles = [1024, 512, 256, 128]

classes = {
1024: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
512: [20.0, 23.0, 25.0, 26.0, 29.0, 36.0, 37.0, 55.0, 56.0, 59.0, 66.0, 69.0, 72.0, 76.0, 78.0, 79.0, 80.0, 87.0, 96.0, 100.0, 108.0, 112.0, 115.0, 127.0, 129.0, 138.0, 147.0],
256: [9.0, 14.0, 20.0, 30.0, 33.0, 38.0, 46.0, 47.0, 51.0, 59.0, 90.0, 95.0, 120.0, 145.0, 165.0, 186.0, 195.0, 196.0, 211.0, 236.0, 241.0, 245.0, 259.0, 273.0, 295.0, 349.0, 366.0, 368.0, 378.0, 381.0, 402.0],
128: [776.0, 1912.0, 1434.0, 992.0, 1087.0, 1844.0, 683.0, 1719.0, 2076.0, 1121.0, 2120.0, 2306.0, 312.0, 409.0, 486.0, 171.0, 2318.0, 1232.0, 396.0, 1119.0, 917.0, 217.0, 340.0, 301.0, 2664.0, 580.0, 355.0, 1191.0, 1992.0, 1365.0, 674.0, 333.0, 129.0, 1382.0, 1553.0, 1635.0, 603.0, 967.0, 1926.0, 202.0, 249.0, 1821.0, 604.0, 2394.0, 1489.0, 2205.0, 1637.0, 2360.0, 1611.0, 872.0, 2537.0, 1639.0, 978.0, 2248.0, 2124.0, 1629.0, 1809.0, 2025.0, 1899.0, 1148.0, 2691.0, 751.0, 453.0, 1756.0, 932.0, 797.0, 1973.0, 1485.0, 1582.0, 2768.0, 2767.0, 2443.0, 2478.0, 87.0, 481.0, 2764.0, 1073.0, 1092.0, 1953.0, 1757.0, 571.0, 212.0, 2629.0, 294.0, 574.0, 1195.0, 1619.0, 877.0, 1857.0, 1287.0, 233.0, 1233.0, 710.0, 1296.0, 1016.0, 1128, 959, 1033,1034, 227, 1178, 1254, 1763, 2632, 2733, 1129, 12, 1368, 1410, 1332, 1408, 1410, 1368, 1254, 993, 1076, 1077]
}

class_dicts = []
tile_fnames = []

fnme_final = []
for i in range(len(fnames)):
    class_dict = []
    tile_fname = []
    for j in range(len(tiles)):
        fname = fnames[i] + ext_1 + str(tiles[j]) + ext_2
        if os.path.exists(fname):
           tile_fname.append(fname)
           class_dict.append(classes[tiles[j]])
    if len(class_dict) < 4:
        continue
    fnme_final.append(fnames[i])
    class_dicts.append(class_dict)
    tile_fnames.append(tile_fname)


pprint(fnme_final)

pprint(tile_fnames)

for i in range(len(class_dicts)):
    print("[")
    for j in range(len(class_dicts[i])):
        print(class_dicts[i][j], ",")
    print("],") 

