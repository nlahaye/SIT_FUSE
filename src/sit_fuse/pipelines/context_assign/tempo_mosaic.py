import re
import os
from rasterio.merge import merge
import rasterio
import glob
from scipy import ndimage

from pprint import pprint

import numpy as np


TEMPO_RE = "*Contours.tif"
TEMPO_DIR = "/data/nlahaye/output/Learnergy/SF_TEMPO_V04/"


TEMPO_SWATH_ID_RE = "S0[0-9][0-9]G0([0-9])"

fnames = sorted(glob.glob(TEMPO_DIR + "TEMPO_RAD_L1_V04_*Contours.tif"))





last_id = -1
current_ind = 0

mosaic_fnames = []
while current_ind < len(fnames):
    mosaic_list = []

    for i in range(current_ind, len(fnames)):
        current_id = int(re.search(TEMPO_SWATH_ID_RE, fnames[i]).group(1))
        print(current_id, last_id) 
        if current_id > last_id:
            mosaic_list.append(fnames[i])
            current_ind = current_ind + 1
            last_id = current_id
        else:
            last_id = -1
            break


    mosaic_fnames.append(mosaic_list)
    print(current_ind, mosaic_list[0]) 
    if len(mosaic_list) > 0:

        dest, output_transform = merge(mosaic_list, method="max")

    print(dest.shape) 
    dest = np.squeeze(dest)
    H, W = dest.shape
    # Mask of pixels that currently belong to some region
    occupied = dest != 0

    close_radius = 1
    if close_radius > 0:
        # Per-label closing to remove tiny cracks before EDT
        y, x = np.ogrid[-close_radius:close_radius + 1,
                        -close_radius:close_radius + 1]
        structure = (x * x + y * y) <= close_radius * close_radius

        unique_labels = np.unique(dest[occupied])
        for lab in unique_labels:
            mask = dest == lab
            if np.any(mask):
                closed = ndimage.binary_closing(mask, structure=structure)
                dest[closed] = lab
        occupied = dest != 0

    # Pixels we want to fill: currently background
    seam = ~occupied



    dest = np.expand_dims(dest, axis=0) 
    with rasterio.open(mosaic_list[0]) as src:
            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": dest.shape[1],
                    "width": dest.shape[2],
                    "transform": output_transform
                }
            )
  
    with rasterio.open(mosaic_list[0] + ".merged.tif", "w", **out_meta) as fp:
        fp.write(dest)

 
  

print(len(mosaic_fnames))
pprint(mosaic_fnames)
