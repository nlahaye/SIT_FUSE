from osgeo import gdal
import re

import sys

out_dat_fname = "/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/background"

annotation_fname = "/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/background.tif"

end_fname_re = "(_\d+\.tif)"

fnames = [
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_0.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_100.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_101.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_102.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_103.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_104.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_105.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_106.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_107.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_108.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_109.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_10.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_110.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_111.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_112.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_113.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_114.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_115.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_116.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_117.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_118.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_119.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_11.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_120.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_121.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_122.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_123.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_124.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_125.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_126.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_127.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_128.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_129.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_12.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_130.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_131.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_132.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_133.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_134.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_135.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_136.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_137.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_138.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_139.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_13.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_140.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_141.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_142.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_143.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_14.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_15.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_16.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_17.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_18.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_19.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_1.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_20.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_21.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_22.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_23.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_24.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_25.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_26.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_27.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_28.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_29.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_2.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_30.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_31.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_32.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_33.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_34.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_35.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_36.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_37.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_38.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_39.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_3.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_40.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_41.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_42.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_43.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_44.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_45.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_46.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_47.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_48.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_49.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_4.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_50.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_51.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_52.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_53.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_54.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_55.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_56.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_57.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_58.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_59.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_5.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_60.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_61.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_62.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_63.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_64.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_65.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_66.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_67.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_68.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_69.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_6.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_70.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_71.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_72.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_73.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_74.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_75.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_76.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_77.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_78.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_79.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_7.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_80.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_81.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_82.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_83.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_84.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_85.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_86.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_87.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_88.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_89.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_8.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_90.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_91.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_92.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_93.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_94.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_95.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_96.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_97.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_98.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_99.tif",
"/data/nlahaye/remoteSensing/PV_Mapping/Full_Area/pleides_RGBNED_9.tif",
]


for fname in fnames:
    ref = gdal.Open(fname,0)
    tform = ref.GetGeoTransform()
    x_res = tform[1]
    y_res = tform[5]

    xsize = ref.RasterXSize
    ysize = ref.RasterYSize

    xmin = tform[0]
    ymax = tform[3]
    xmax = xmin + x_res * xsize
    ymin = ymax + y_res * ysize

    end_fname = str(re.search(end_fname_re, fname).group(1))
    
    out_fname = out_dat_fname + end_fname

    print(out_fname)
    source_ds = gdal.Open(annotation_fname)

    ds = gdal.Warp( out_fname,  source_ds, format = "GTiff", 
                      srcSRS=source_ds.GetProjection(),
                      dstSRS = ref.GetProjection(),
                      resampleAlg = gdal.GRA_NearestNeighbour,
                      outputBounds = [xmin, ymin, xmax, ymax],
                      width = xsize,
                      xRes = x_res,
                      yRes = y_res,
                      height = ysize,
                      targetAlignedPixels = True )


    ref = None
    source_ds = None
