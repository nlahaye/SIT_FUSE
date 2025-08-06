import xarray
import rasterio
import os
from pyproj import Transformer
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import yaml
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
from prithvi_mae import PrithviViT

import zarr
import presto

# Load model directly
from transformers import AutoModel

# this is to silence the xarray deprecation warning.
# Our version of xarray is pinned, but we'll need to fix this
# when we upgrade
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 



treesat_folder = "/data/nlahaye/remoteSensing/PRESTO_Trees/s2/200m/"
assert os.path.exists(treesat_folder)

# this folder should exist once the s2 file from zenodo has been unzipped
s2_data_60m = "/data/nlahaye/remoteSensing/PRESTO_Trees/s2/200m/"
assert os.path.exists(s2_data_60m)



TREESATAI_S2_BANDS = ["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12", "B1", "B9"]

SPECIES = {"Pinus":0, "Picea":1, "Abies":2, "Quercus": 3 , "Tilia": 4, "Prunus": 5, "Acer": 6, "Betula": 7, "Fraxinus": 8, "Alnus": 9, "Cleared": 10, "Fagus": 11, "Larix": 12, "Populus":13, "Pseudotsuga": 14}

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
PERCENTILES = (0.1, 99.9)


def preprocess_image(image, means, stds):
    # normalize image
    normalized = image.copy()
    normalized = ((image - means) / stds)
    normalized = torch.from_numpy(normalized.reshape(normalized.shape[0], 1, *normalized.shape[-2:])).to(torch.float32)
    return normalized


def load_raster(path, crop=None):
    with rasterio.open(path) as src:
        img = src.read()

        # load first 6 bands
        img = img[:6]

        img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)
        if crop:
            img = img[:, -crop[0]:, -crop[1]:]
    return img



 
def process_images(filenames, means, stds):
    arrays, labels, image_names  = [], [], []
    
    for filename in tqdm(filenames):
        input_data = load_raster(os.path.join(s2_data_60m,filename.strip()), crop=(16, 16))        
        normalized = preprocess_image(input_data, means, stds)


        arrays.append(normalized)
               
        ind = -1 
        for key in SPECIES.keys():
             if key in filename:
                 ind = SPECIES[key]
                 break
                
             labels.append(ind)
             image_names.append(filename)
 
    return (torch.stack(arrays, axis=0),
            torch.tensor(labels),
            image_names,
        )



# takes a (6, 6) treesat tif file, and returns a
# (9,1,18) cropharvest eo-style file (with all bands "masked"
# except for S1 and S2)


 
with open(os.path.join(treesat_folder,"train_filenames.lst"), "r") as f:
    train_files = [line for line in f]
with open(os.path.join(treesat_folder, "test_filenames.lst"), "r") as f:
    test_files = [line for line in f]

print(f"{len(train_files)} train files and {len(test_files)} test files")



# load weights
weights_path = "./prithvi/Prithvi_EO_V1_100M.pt"
checkpoint = torch.load(weights_path, map_location="cpu")
#model = AutoModel.from_pretrained("ibm-nasa-geospatial/Prithvi-EO-1.0-100M")


# read model config
model_cfg_path = "./prithvi/Prithvi_100M_config.yaml"
with open(model_cfg_path) as f:
    model_config = yaml.safe_load(f)

model_args, train_args = model_config["model_args"], model_config["train_params"]

# let us use only 1 frame for now (the model was trained on 3 frames)
model_args["num_frames"] = 1

# instantiate model
model = PrithviViT(**model_args)
model.eval()

# load weights into model
# strict=false since we are loading with only 1 frame, but the warning is expected
_ = model.load_state_dict(checkpoint, strict=False)


# statistics used to normalize images before passing to the model
means = np.array(train_args["data_mean"]).reshape(-1, 1, 1)
stds = np.array(train_args["data_std"]).reshape(-1, 1, 1)





train_data, train_labels, train_fnames  = process_images(train_files, means, stds)
test_data, test_labels, test_fnames  = process_images(test_files, means, stds)

print(test_data.shape, train_data.shape)
batch_size = 64


features, _, _ = model.forward(train_data, mask_ratio=0)
features = features.detach().numpy()

zarr.save('/data/nlahaye/remoteSensing/PRESTO_Trees/prithvi_treesat_embeddings.zarr', features)

