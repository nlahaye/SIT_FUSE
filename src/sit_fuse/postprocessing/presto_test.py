import matplotlib.pyplot as plt
import rasterio
import yaml
from prithvi.Prithvi import MaskedAutoencoderViT

import xarray
import os
from pyproj import Transformer
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset


import zarr
import presto

# this is to silence the xarray deprecation warning.
# Our version of xarray is pinned, but we'll need to fix this
# when we upgrade
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 



treesat_folder = "/data/nlahaye/remoteSensing/PRESTO_Trees/s2/60m/"
assert os.path.exists(treesat_folder)

# this folder should exist once the s2 file from zenodo has been unzipped
s2_data_60m = "/data/nlahaye/remoteSensing/PRESTO_Trees/s2/60m"
assert os.path.exists(s2_data_60m)

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
PERCENTILES = (0.1, 99.9)


TREESATAI_S2_BANDS = ["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12", "B1", "B9"]

SPECIES = {"Pinus":0, "Picea":1, "Abies":2, "Quercus": 3 , "Tilia": 4, "Prunus": 5, "Acer": 6, "Betula": 7, "Fraxinus": 8, "Alnus": 9, "Cleared": 10, "Fagus": 11, "Larix": 12, "Populus":13, "Pseudotsuga": 14}
 
def process_images(filenames):
    arrays, masks, latlons, image_names, labels, dynamic_worlds = [], [], [], [], [], []
    
    for filename in tqdm(filenames):
        tif_file = xarray.open_dataset(os.path.join(s2_data_60m,filename.strip()),  engine='rasterio')
        crs = tif_file.rio.crs.to_proj4().split("=")[-1]
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        
        for x_idx in INDICES_IN_TIF_FILE:
            for y_idx in INDICES_IN_TIF_FILE:
                
                # firstly, get the latitudes and longitudes
                x, y = tif_file.x[x_idx], tif_file.y[y_idx]
                lon, lat = transformer.transform(x, y) 
                latlons.append(torch.tensor([lat, lon]))
                
                # then, get the eo_data, mask and dynamic world
                s2_data_for_pixel = torch.from_numpy((tif_file["band_data"].to_numpy())[:, x_idx, y_idx].astype(int)).float()
                s2_data_with_time_dimension = s2_data_for_pixel.unsqueeze(0)
                x, mask, dynamic_world = presto.construct_single_presto_input(
                    s2=s2_data_with_time_dimension, s2_bands=TREESATAI_S2_BANDS
                )
                arrays.append(x)
                masks.append(mask)
                dynamic_worlds.append(dynamic_world)
                
                for key in SPECIES.keys():
                    if key in filename:
                        ind = SPECIES[key]
                        break
                
                labels.append(ind)
                image_names.append(filename)

    return (torch.stack(arrays, axis=0),
            torch.stack(masks, axis=0),
            torch.stack(dynamic_worlds, axis=0),
            torch.stack(latlons, axis=0),
            torch.tensor(labels),
            image_names,
        )




with open("/home/nlahaye/prithvi/__init__.py", "w") as f:
    f.write("")


 
with open(os.path.join(treesat_folder,"train_filenames.lst"), "r") as f:
    train_files = [line for line in f]
with open(os.path.join(treesat_folder, "test_filenames.lst"), "r") as f:
    test_files = [line for line in f]

print(f"{len(train_files)} train files and {len(test_files)} test files")

train_data = process_images(train_files)
test_data = process_images(test_files)


batch_size = 64

pretrained_model = presto.Presto.load_pretrained()
pretrained_model.eval()
pretrained_model = pretrained_model.cuda()

# the treesat AI data was collected during the summer,
# so we estimate the month to be 6 (July)
month = torch.tensor([6] * train_data[0].shape[0]).long()

dl = DataLoader(
    TensorDataset(
        train_data[0].float(),  # x
        train_data[1].bool(),  # mask
        train_data[2].long(),  # dynamic world
        train_data[3].float(),  # latlons
        month
    ),
    batch_size=batch_size,
    shuffle=False,
)


features_list = []
for (x, mask, dw, latlons, month) in tqdm(dl):

    x = x.cuda()
    mask = mask.cuda()
    dw = dw.cuda()
    latlons = latlons.cuda()
    month = month.cuda()

    with torch.no_grad():
        encodings = (
            pretrained_model.encoder(
                x, dynamic_world=dw, mask=mask, latlons=latlons, month=month
            )
            .cpu()
            .numpy()
        )
        features_list.append(encodings)
features_np = np.concatenate(features_list)

zarr.save('/data/nlahaye/remoteSensing/PRESTO_Trees/presto_treesat_embeddings.zarr', features_np)

