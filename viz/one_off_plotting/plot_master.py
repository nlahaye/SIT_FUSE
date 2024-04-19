
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

#data_files = ["/data/nlahaye/remoteSensing/MASTER/MASTERL1B_1981719_07_20190806_2226_2231_V01_georeferenced.tif.npy",
#"/data/nlahaye/remoteSensing/MASTER/MASTERL1B_1981719_10_20190807_0042_0050_V01_georeferenced.tif.npy",
#"/data/nlahaye/remoteSensing/MASTER/MASTERL1B_1981719_06_20190806_2153_2156_V01_georeferenced.tif.npy",
#"/data/nlahaye/remoteSensing/MASTER/MASTERL1B_1981720_07_20190808_0055_0103_V01_georeferenced.tif.npy",
#"/data/nlahaye/remoteSensing/MASTER/MASTERL1B_1981720_09_20190808_0237_0238_V01_georeferenced.tif.npy",
#"/data/nlahaye/remoteSensing/MASTER/MASTERL1B_1981721_05_20190809_0219_0226_V01_georeferenced.tif.npy",
#"/data/nlahaye/remoteSensing/MASTER/MASTERL1B_1981721_02_20190809_0120_0121_V01_georeferenced.tif.npy",
#"/data/nlahaye/remoteSensing/MASTER/MASTERL1B_1981721_01_20190809_0113_0117_V01_georeferenced.tif.npy",
#"/data/nlahaye/remoteSensing/MASTER/MASTERL1B_1981719_02_20190806_1851_1900_V01_georeferenced.tif.npy",
#"/data/nlahaye/remoteSensing/MASTER/MASTERL1B_1981719_08_20190806_2249_2259_V01_georeferenced.tif.npy",
#"/data/nlahaye/remoteSensing/MASTER/MASTERL1B_1981719_05_20190806_2033_2039_V01_georeferenced.tif.npy"]


data_files =  ["/data/nlahaye/remoteSensing//MisrModisFuse/Fuse1_9Cam.npy",
"/data/nlahaye/remoteSensing//MisrModisFuse/Fuse3_9Cam.npy",
"/data/nlahaye/remoteSensing//MisrModisFuse/Fuse4_9Cam.npy",
"/data/nlahaye/remoteSensing//MisrModisFuse/Fuse2_9Cam.npy",
"/data/nlahaye/remoteSensing//MisrModisFuse/Fuse5_9Cam.npy",
"/data/nlahaye/remoteSensing//MisrModisFuse/Fuse9_9Cam.npy"]
  

 
for i in range(len(data_files)):
    filename = data_files[i]
    data = np.load(filename)

    print(data.shape)
    for c in range(data.shape[0]):
        mask_gray = cv.normalize(src=data[c,100:-100,100:-100], dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        equ = cv.equalizeHist(mask_gray)
        plt.imshow(equ)
        plt.savefig("MISR_IMAGE_" + str(i) + "C" + str(c) + ".png")



