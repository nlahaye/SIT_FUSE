import csv
import os

import fiftyone as fo
import fiftyone.utils.data as foud
import fiftyone.brain as fob
import numpy as np

import glob

images_patt = "/data/nlahaye/SIT_FUSE_Geo/eMAS_DBN_Embeddings/*.tif"

fire_label_dict = {
    0: "Nothing",
    1: "Fire",
    2: "Smoke",
    3: "Fire_Smoke",
        }
 
def import_dataset(filepath):
 
    samples = []
    labels_path = "/data/nlahaye/SIT_FUSE_Geo/eMAS_DBN_Embeddings/eMASL1B_19910_20_20190806_2052_2106_V03.dbn_2_layer_2000.viz_dict.npy"
    for filepath in glob.glob(images_patt):
        sample = fo.Sample(filepath=filepath)
  
        my_dict = np.load(labels_path, allow_pickle=True).item()
 
        detections = []
        #TODO
        h = 1379
        w = 5217
        for i in range(len(my_dict["bb_x"])):
            x = (my_dict["bb_x"][i]-1) / w #Make TL index (from center) and normalize in [0,1]
            y = (my_dict["bb_y"][i]-1) / h
            w_bb = my_dict["bb_width"][i] / w
            h_bb = my_dict["bb_height"][i] / h
            rel_box = [x, y, w_bb, h_bb]

            fire_label = fire_label_dict[my_dict["final_label"][i]]
            heir_label = str(my_dict["heir_label"][i])
            no_heir_label = str(my_dict["no_heir_label"][i])

            detections.append(fo.Detection(label=fire_label, bounding_box=rel_box, no_heir=no_heir_label, heir=heir_label,
                tags=[no_heir_label, heir_label, fire_label]))

        print("HERE", len(detections))
        sample["detections"] = fo.Detections(detections=detections)
        samples.append(sample)

    # Create dataset
    dataset = fo.Dataset(name="eMAS_dbn_2_layer_2000")
    dataset.persistent = True
    dataset.add_samples(samples) 

    return dataset




def comp_viz(dataset):

    labels_path = "/data/nlahaye/SIT_FUSE_Geo/eMAS_DBN_Embeddings/eMASL1B_19910_20_20190806_2052_2106_V03.dbn_2_layer_2000.viz_dict.npy"
    my_dict = np.load(labels_path, allow_pickle=True).item()

    sample = dataset.first()
    num_detections = len(sample.detections.detections)
    points = np.zeros((num_detections, 2))
    print(points.shape)
    for i in range(len(my_dict["bb_x"])):
        points[i,1] = my_dict["proj_x"][i]
        points[i,0] = my_dict["proj_y"][i]
    results = fob.compute_visualization(dataset, patches_field="detections", points=points, brain_key="eMAS_dbn_2_layer_2000", progress=True, verbose=True)

    






