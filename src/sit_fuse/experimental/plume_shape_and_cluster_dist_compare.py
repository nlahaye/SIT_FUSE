"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
from swd import swd
import gdal
import torch
import numpy as np

import cv2 

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swd import swd

#TODO initial test. Need to generalize and make configuration-based. swd code is in thirdparty/

data = [
"/home/nlahaye/RBM_and_GAN_eMAS_Smoke/0Predicted Image.png.bin.orig.tif",
"/home/nlahaye/RBM_and_GAN_eMAS_Smoke/1Predicted Image.png.bin.orig.tif", 
"/home/nlahaye/RBM_and_GAN_eMAS_Smoke/3Predicted Image.png.bin.orig.tif",
"/home/nlahaye/RBM_and_GAN_eMAS_Smoke/4Predicted Image.png.bin.orig.tif"]#,
#"/home/nlahaye/RBM_and_GAN_eMAS_Smoke/5Predicted Image.png.bin.orig.tif"]
  
#"/home/nlahaye/IM-NET-INPUT/plumes/eMASL1B_19910_21_20190806_2111_2125_V02_Georeferenced_scaled_SmokeMask_v13_Clipped.jpg",
#"/home/nlahaye/IM-NET-INPUT/plumes/eMASL1B_19911_11_20190807_2021_2035_V02_georeferenced_scaled_SmokeMask_v13_Clipped.jpg",
#"/home/nlahaye/IM-NET-INPUT/plumes/eMASL1B_19910_19_20190806_2035_2048_V02_Georeferenced_scaled_SmokeMask_v13_Clipped.jpg",
#"/home/nlahaye/IM-NET-INPUT/plumes/eMASL1B_19911_09_20190807_1947_2002_V02_georeferenced_scaled_SmokeMask_v13_Clipped.jpg",
#"/home/nlahaye/IM-NET-INPUT/plumes/eMASL1B_19911_10_20190807_2004_2016_V02_georeferenced_scaled_SmokeMask_v13_Clipped.jpg",
#"/home/nlahaye/IM-NET-INPUT/plumes/eMASL1B_19910_20_20190806_2052_2106_V02_Georeferenced_scaled_SmokeMask_v13_Clipped.jpg",
#"/home/nlahaye/IM-NET-INPUT/plumes/eMASL1B_19911_05_20190807_1817_1833_V02_georeferenced_scaled_SmokeMask_v13_Clipped.jpg",
#"/home/nlahaye/IM-NET-INPUT/plumes/eMASL1B_19911_12_20190807_2039_2051_V02_georeferenced_scaled_SmokeMask_v13_Clipped.jpg",
#"/home/nlahaye/IM-NET-INPUT/plumes/eMASL1B_19910_18_20190806_2018_2031_V02_Georeferenced_scaled_SmokeMask_v13_Clipped.jpg",
#"/home/nlahaye/IM-NET-INPUT/plumes/eMASL1B_19911_08_20190807_1928_1942_V02_georeferenced_scaled_SmokeMask_v13_Clipped.jpg",]
 
data2 = ["/home/nlahaye/RBM_and_GAN_eMAS_Smoke/EMAS_SMOKE_FULLCOLOR/eMASL1B_19911_08_20190807_1928_1942_V02_georeferenced_scaled_SmokeMask_v13_FullColor.tif",
"/home/nlahaye/RBM_and_GAN_eMAS_Smoke/EMAS_SMOKE_FULLCOLOR/eMASL1B_19911_09_20190807_1947_2002_V02_georeferenced_scaled_SmokeMask_v13_FullColor.tif",
"/home/nlahaye/RBM_and_GAN_eMAS_Smoke/EMAS_SMOKE_FULLCOLOR/eMASL1B_19911_05_20190807_1817_1833_V02_georeferenced_scaled_SmokeMask_v13_FullColor.tif",
"/home/nlahaye/RBM_and_GAN_eMAS_Smoke/EMAS_SMOKE_FULLCOLOR/eMASL1B_19910_10_20190806_1858_1910_V02_Georeferenced_scaled_SmokeMask_v13_FullColor.tif"]


def normalize_filled(dat, binarize = True):
    print("HERE1", dat)
    img = gdal.Open(dat).ReadAsArray().astype(np.uint8)
    if binarize:
        img[np.where(img > 0)] = 255
    print(img.shape)
    return cv2.resize(img, (512, 512))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    #return
    #cnt,_= cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print("HERE2", len(cnt))
    #bounding_rect = cv2.boundingRect(cnt[0])
    #print("HERE3", bounding_rect, img.shape)
    #img_cropped_bounding_rect = img[:bounding_rect[1] + bounding_rect[3],:]#, :bounding_rect[0] + bounding_rect[2]]
    # resize all to same size
    #print("HERE4")
    #img_resized = cv2.resize(img_cropped_bounding_rect, (300, 300))
    return img_cropped_bounding_rect#_resized


imgs = [normalize_filled(i) for i in data]
imgs2 = [normalize_filled(i, False) for i in data2]
print("HERE")
#for i in range(1, len(imgs)+1):
for i in range(len(data)):
  plt.subplot(2, 3, i+1), plt.imshow(imgs[i], cmap='gray')
  for j in range(len(data)):
    #plt.subplot(2, 3, i), plt.imshow(imgs[i - 1], cmap='gray')
    arr = torch.from_numpy(imgs[i].astype(np.float32))
    arr2 = torch.from_numpy(imgs[j].astype(np.float32))
    arr = torch.reshape(arr, (1,1,arr.shape[0], arr.shape[1]))
    arr2 = torch.reshape(arr2, (1,1,arr2.shape[0], arr2.shape[1]))
    print(arr.shape, arr2.shape) 
    #print(i, j, np.mean([cv2.matchShapes(imgs[i], imgs[j], 1, 0.0), cv2.matchShapes(imgs[i], imgs[j], 2, 0.0), cv2.matchShapes(imgs[i], imgs[j], 3, 0.0)]), swd(arr, arr2, device="cuda").numpy())
    swd1 = 10000*np.mean([cv2.matchShapes(imgs[i], imgs[j], 1, 0.0), cv2.matchShapes(imgs[i], imgs[j], 2, 0.0), cv2.matchShapes(imgs[i], imgs[j], 3, 0.0)])
    #swd1 = swd(arr, arr2, device="cuda").numpy()
    arr = torch.from_numpy(imgs2[i].astype(np.float32))
    arr2 = torch.from_numpy(imgs2[j].astype(np.float32))
    arr = torch.reshape(arr, (1,1,arr.shape[0], arr.shape[1]))
    arr2 = torch.reshape(arr2, (1,1,arr2.shape[0], arr2.shape[1]))
    #print(i, j, swd(arr, arr2, device="cuda").numpy())    
    swd2 = swd(arr, arr2, device="cuda").numpy()
    print(i,j,np.mean([swd1,swd2]), swd1, swd2)

plt.savefig("TEST.jpg")






