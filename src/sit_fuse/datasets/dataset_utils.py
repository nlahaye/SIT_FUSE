"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
#General Imports
import os
import numpy as np

#ML imports
import torch
import torch.nn as nn
from torchvision import transforms


#Serialization
from joblib import dump, load

#Data
from sit_fuse.datasets.sf_dataset import SFDataset
from sit_fuse.datasets.sf_dataset_conv import SFDatasetConv
from sit_fuse.utils import get_read_func, get_scaler


#Serialization
import pickle



def get_train_dataset_sf(yml_conf):
    #Get config values 
    data_train = yml_conf["data"]["files_train"]

    pixel_padding = yml_conf["data"]["pixel_padding"]
    number_channel = yml_conf["data"]["number_channels"]
    data_reader =  yml_conf["data"]["reader_type"]
    data_reader_kwargs = yml_conf["data"]["reader_kwargs"]
    fill = yml_conf["data"]["fill_value"]
    chan_dim = yml_conf["data"]["chan_dim"]
    valid_min = yml_conf["data"]["valid_min"]
    valid_max = yml_conf["data"]["valid_max"]
    delete_chans = yml_conf["data"]["delete_chans"]
    scale_data = yml_conf["data"]["scale_data"]

    transform_chans = yml_conf["data"]["transform_default"]["chans"]
    transform_values =  yml_conf["data"]["transform_default"]["transform"]

    out_dir = yml_conf["output"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)


    tiled = yml_conf["data"]["tile"]
    print("TILED", tiled)

    tiled = yml_conf["data"]["tile"]
    tile_size = None
    tile_step = None
    if tiled:
        tile_size = yml_conf["data"]["tile_size"]
        tile_step = yml_conf["data"]["tile_step"]

    tune_scaler = False
    subset_training = -1
    if "encoder" in yml_conf:
        tune_scaler = yml_conf["encoder"]["tune_scaler"]
        subset_training = yml_conf["encoder"]["subset_training"]

    stratify_data = None
    if "encoder" in yml_conf and "stratify_data" in yml_conf["encoder"]["training"]:
        stratify_data = yml_conf["encoder"]["training"]["stratify_data"]


    scaler = None
    scaler_train = True
    scaler_fname = os.path.join(out_dir, "encoder_scaler.pkl")

    preprocess_train = True
    targets_fname = os.path.join(out_dir, "train_data.indices.npy")
    data_fname = os.path.join(out_dir, "train_data.npy")

    if os.path.exists(targets_fname) and os.path.exists(data_fname):
        preprocess_train = False



    scaler_tune = True
    if not os.path.exists(scaler_fname) or preprocess_train == True:
        scaler_type = yml_conf["scaler"]["name"]
        scaler, scaler_train = get_scaler(scaler_type)
    else:
        scaler = load(scaler_fname)
        scaler_train = False
        if tune_scaler:
            scaler_train = True


    read_func = get_read_func(data_reader)

    data = None
    if not tiled:
        #TODO stratify conv data

        if preprocess_train:
            if stratify_data is not None and "kmeans" not in stratify_data:
                strat_read_func = get_read_func(stratify_data["reader"])
                stratify_data["reader"] = strat_read_func

            data = SFDataset()
            data.read_and_preprocess_data(data_train, read_func, data_reader_kwargs, pixel_padding, delete_chans=delete_chans, \
                valid_min=valid_min, valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, \
                transform_values=transform_values, scaler = scaler, train_scaler = scaler_train, scale = scale_data, \
                transform=None, subset_training = subset_training, stratify_data=stratify_data)
        else:
            data = SFDataset()
            data.read_data_preprocessed(data_fname, targets_fname, scaler, subset_training = subset_training, stratify_data=stratify_data)
    else:
        if preprocess_train:
            data = SFDatasetConv()
            data.read_and_preprocess_data(data_train, read_func, data_reader_kwargs, delete_chans=delete_chans, \
                 valid_min=valid_min, valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, \
                 transform_values=transform_values, transform=None, tile=tiled, tile_size=tile_size, tile_step=tile_step,
                 subset_training = subset_training)
        else:
           transform = None
           if os.path.exists(os.path.join(out_dir, "encoder_data_transform.ckpt")):
               state_dict = torch.load(os.path.join(out_dir, "encoder_data_transform.ckpt"))
               transform = torch.nn.Sequential(
                                transforms.Normalize(state_dict["mean_per_channel"], state_dict["std_per_channel"])
                        )
           data = SFDatasetConv()
           data.read_data_preprocessed(data_fname, targets_fname, transform=transform, subset_training = subset_training, stratify_data=stratify_data)

    print(data.data_full.shape)
    #if data.train_indices is not None:
    #    np.save(os.path.join(out_dir, "train_indices"), data.train_indices)


    #Save scaler
    if hasattr(data, "scaler") and data.scaler is not None and not os.path.exists(os.path.join(out_dir, "encoder_scaler.pkl")):
        with open(os.path.join(out_dir, "encoder_scaler.pkl"), "wb") as f:
            dump(data.scaler, f, True, pickle.HIGHEST_PROTOCOL)
    else:
        data.scaler = None

    return data    

    


def get_prediction_dataset(yml_conf, fname):

    #Get config values 
    pixel_padding = yml_conf["data"]["pixel_padding"]
    number_channel = yml_conf["data"]["number_channels"]
    data_reader =  yml_conf["data"]["reader_type"]
    data_reader_kwargs = yml_conf["data"]["reader_kwargs"]
    fill = yml_conf["data"]["fill_value"]
    chan_dim = yml_conf["data"]["chan_dim"]
    valid_min = yml_conf["data"]["valid_min"]
    valid_max = yml_conf["data"]["valid_max"]
    delete_chans = yml_conf["data"]["delete_chans"]
    scale_data = yml_conf["data"]["scale_data"]

    transform_chans = yml_conf["data"]["transform_default"]["chans"]
    transform_values =  yml_conf["data"]["transform_default"]["transform"]

    out_dir = yml_conf["output"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)


    
    tiled = yml_conf["data"]["tile"]
    tile_size = None
    tile_step = None
    if tiled:
        tile_size = yml_conf["data"]["tile_size"]
        tile_step = yml_conf["data"]["tile_step"]  

    tune_scaler = False
    if "encoder" in yml_conf:
        tune_scaler = yml_conf["encoder"]["tune_scaler"]

    scaler = None
    scaler_train = True
    scaler_fname = os.path.join(out_dir, "encoder_scaler.pkl")

    preprocess_train = True
    targets_fname = os.path.join(out_dir, "train_data.indices.npy")
    data_fname = os.path.join(out_dir, "train_data.npy")

    if os.path.exists(targets_fname) and os.path.exists(data_fname):
        preprocess_train = False

    scaler_tune = False
    if not os.path.exists(scaler_fname) or preprocess_train == True:
        scaler_type = yml_conf["scaler"]["name"]
        scaler, scaler_train = get_scaler(scaler_type)
    else:
        scaler = load(scaler_fname)
        scaler_train = False
        if tune_scaler:
            scaler_train = True

    transform = None
    if os.path.exists(os.path.join(out_dir, "encoder_data_transform.ckpt")):
        state_dict = torch.load(os.path.join(out_dir, "encoder_data_transform.ckpt"))
        transform = torch.nn.Sequential(
            transforms.Normalize(state_dict["mean_per_channel"], state_dict["std_per_channel"])
        )


    read_func = get_read_func(data_reader)

    fbase = fname
    while isinstance(fbase, list):
        fbase = fbase[0]
    fbase = os.path.basename(fbase)
    
    fname_begin = os.path.basename(fbase) + ".clust"
    if not tiled:
        data = SFDataset()
        data.read_and_preprocess_data([fname], read_func, data_reader_kwargs, pixel_padding, delete_chans=delete_chans, valid_min=valid_min, valid_max=valid_max, \
            fill_value = fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, scaler=scaler, scale = scale_data, \
            transform=transform, do_shuffle=False)
    else:
        data = SFDatasetConv()
        data.read_and_preprocess_data([fname], read_func, data_reader_kwargs,  delete_chans=delete_chans, valid_min=valid_min, valid_max=valid_max, \
            fill_value = fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, transform = transform, \
            tile=tiled, tile_size=tile_size, tile_step=tile_step, do_shuffle=False)

    return data, fname_begin





