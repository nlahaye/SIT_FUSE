import csv
import os

import fiftyone as fo
import fiftyone.utils.data as foud
import fiftyone.brain as fob
import numpy as np

from PIL import Image
from osgeo import gdal
import glob
 
#images_patt = "/home/nlahaye/eMAS_DBN_Embeddings/*.tif"

#fire_label_dict = {
#    0: "Nothing",
#    1: "Fire",
#    2: "Smoke",
#    3: "Fire_Smoke",
        }


def gtif_to_png(filepath):

    ds = gdal.Open(filepath).ReadAsArray()
    r = ds[0]
    g = ds[1]
    b = ds[2]

    # Get rid of the pesky -3.4e+38 marker for out-of-bounds pixels
    r[r < 0.00001]     = 0
    b[b < 0.00001]   = 0
    g[g < 0.00001] = 0

    # Find maximum across all channels for scaling
    max = np.max([r,g,b])
 
    # Scale all channels equally to range 0..255 to fit in a PNG (could use 65,535 and np.uint16 instead)
    R = (r * 255/max).astype(np.uint8)
    G = (g * 255/max).astype(np.uint8)
    B = (b * 255/max).astype(np.uint8)

    # Build a PNG
    RGB = np.dstack((R,G,B))
    Image.fromarray(RGB).save(filepath + ".png")


def import_dataset(label_extension, images_patt, label_dict, dataset_name):
 
    samples = []
    #labels_path = "/home/nlahaye/eMAS_DBN_Embeddings/eMASL1B_19910_20_20190806_2052_2106_V03.dbn_2_layer_2000.viz_dict.npy"
    for filepath in glob.glob(images_patt):
        gtif_to_png(filepath)
        sample = fo.Sample(filepath=filepath+".png")
  
        my_dict = np.load(os.path.join(os.path.dirname(filepath), os.path.basename(filepath) \
                + "." + dataset_name + ".viz_dict.npy", allow_pickle=True).item()
 
        detections = []
        #TODO
        dat = gdal.Open(filepath).ReadAsArray()
        h = dat.shape[0] #1379
        w = dat.shape[1] #5217
        for i in range(len(my_dict["bb_x"])):
            x = (my_dict["bb_x"][i]-1) / w #Make TL index (from center) and normalize in [0,1]
            y = (my_dict["bb_y"][i]-1) / h
            w_bb = my_dict["bb_width"][i] / w
            h_bb = my_dict["bb_height"][i] / h
            rel_box = [x, y, w_bb, h_bb]

            #fire_label = fire_label_dict[my_dict["final_label"][i]]
            final_label = label_dict[my_dict["final_label"][i]]
            heir_label = str(my_dict["heir_label"][i])
            no_heir_label = str(my_dict["no_heir_label"][i])

            detections.append(fo.Detection(label=fire_label, bounding_box=rel_box, no_heir=no_heir_label, heir=heir_label,
                tags=[no_heir_label, heir_label, fire_label]))

        print("HERE", len(detections))
        sample["detections"] = fo.Detections(detections=detections)
        samples.append(sample)

    # Create dataset
    dataset = fo.Dataset(name=dataset_name, overwrite=True) #"eMAS_dbn_2_layer_2000", overwrite=True)
    dataset.persistent = True
    dataset.add_samples(samples) 

    return dataset




def comp_viz(dataset, dataset_name, labels_path):
 
    for sample in dataset.samples:
 
        ind = np.where[os.path.basename(sample.filepath) in label_paths] 
        labels_path = labels_paths[ind][0]   #"/home/nlahaye/eMAS_DBN_Embeddings/eMASL1B_19910_20_20190806_2052_2106_V03.dbn_2_layer_2000.viz_dict.npy"
        my_dict = np.load(labels_path, allow_pickle=True).item()

        sample = dataset.first()
        num_detections = len(sample.detections.detections)
        points = np.zeros((num_detections, 3)) #, 3))
        print(points.shape)
        for i in range(len(my_dict["bb_x"])):
            points[i,1] = my_dict["proj_x"][i]
            points[i,0] = my_dict["proj_y"][i]
        results = fob.compute_visualization(dataset, patches_field="detections", points=points, \
                brain_key=dataset_name, progress=True, verbose=True, num_dims=2)

    






