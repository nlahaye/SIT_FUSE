import re
import os
import numpy as np
import glob

dirpth = "/data/nlahaye/remoteSensing/MADOS/MADOS/"

valpth = dirpth + "splits/val_X.txt"
testpth = dirpth  + "splits/test_X.txt"
trainpth = dirpth  + "splits/train_X.txt"

dir1_re = "(Scene_\d+)_(\d+)"

lnes = []
with open(trainpth, "r") as f:
    lnes = f.readlines()

ten_ms = ["4*", "5*", "6*", "8*"]
twenty_ms = ["704", "740", "783", "865", "1614", "2202"]
sixty_ms =  ["443"]

print("files_train: [")

#Scene_12_L2R_rhorc_443_45.tif

"""
for lne in lnes:
    pth = []
    mtch = re.search(dir1_re, lne)
    print("[")
    for i in range(len(ten_ms)):
        pth = dirpth + str(mtch.group(1)) + "/10/" + str(mtch.group(1)) +  "_L2R_rhorc_" + ten_ms[i] + "_" + str(mtch.group(2)) + ".tif"
        fglob = glob.glob(pth)
        fname_tmp = fglob[0]
        print("\"" +fname_tmp + "\",")
    print("],")
print("]")
   
"""

print("final_labels: [")
lnes = []
with open(testpth, "r") as f:
    lnes = f.readlines()


for lne in lnes:
    pth = []
    mtch = re.search(dir1_re, lne)
    #print("[")
    #for i in range(len(ten_ms)):
    pth = dirpth + str(mtch.group(1)) + "/10/" +  str(mtch.group(1)) + "_L2R_cl_" + str(mtch.group(2)) + ".tif"
    fglob = glob.glob(pth)
    fname_tmp = fglob[0]
    print("\"" +fname_tmp + "\",")
    #print("],")
print("]")


