from osgeo import gdal, osr
import numpy as np

data = ["/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TQL.2022247T181930.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TPL.2022253T183134.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TNL.2022245T183151.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TPK.2022246T182546.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TPN.2022253T183110.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TQL.2022246T182522.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TPM.2022246T182458.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TPL.2022246T182522.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TNN.2022253T183110.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TQK.2022246T182546.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TNK.2022246T182546.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TPL.2022245T183151.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TNM.2022245T183127.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TNL.2022253T183134.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TQM.2022247T181907.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TPM.2022247T181930.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TNN.2022245T183127.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TNN.2022246T182458.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TNN.2022252T183742.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TNM.2022246T182458.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TPL.2022247T181930.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TPM.2022245T183127.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TQM.2022246T182458.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TPN.2022245T183127.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TNL.2022246T182522.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TNN.2022244T183720.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TPK.2022253T183134.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TPM.2022253T183110.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TPN.2022246T182458.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TPK.2022245T183151.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TQK.2022247T181954.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TPK.2022247T181954.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TNM.2022253T183110.v2.0.B01.fire.tif",
"/data/nlahaye/Fire_Labels/LandSat_Subset_Fire_Labels/HLS.L30.T11TNK.2022253T183158.v2.0.B01.fire.tif"]

print("HERE ", len(data))

sums = [0,0,0]

for i in range(len(data)):

    dat = gdal.Open(data[i]).ReadAsArray()
    

    cnt = 0
    for j in range(-1,2):
        inds = np.where(dat == j)[0]
        print("HERE", j, cnt, inds.shape, sums)
        sums[cnt] = sums[cnt] + len(inds)
        cnt = cnt + 1


print("TOTALS", sums, sum(sums))




