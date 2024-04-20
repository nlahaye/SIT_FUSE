"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
from GPUtil import showUtilization as gpu_usage

#General Imports
import os
import numpy as np
import random

from tqdm import tqdm

import sys
import resource
try:
    max_rec = 10**6
    # May segfault without this line. 100 is a guess at the size of each stack frame.
    resource.setrlimit(resource.RLIMIT_STACK, [100 * max_rec, resource.RLIM_INFINITY])
    sys.setrecursionlimit(max_rec)
except Exception as e:
    pass

#ML imports
import torch
import torch.nn as nn
import torch.optim as opt 
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from learnergy.models.deep import DBN, ResidualDBN, ConvDBN
from torch.utils.data.distributed import DistributedSampler

from pixel_level_contrastive_learning import PixelCL
from torchvision import models
from torchsummary import summary

from vit_pytorch.regionvit import RegionViT
import multichannel_resnet



from rbm_models.clust_dbn import ClustDBN
#Visualization
import learnergy.visual.convergence as converge
import matplotlib 
matplotlib.use("Agg")
import  matplotlib.pyplot as plt

#Serialization
from joblib import dump, load

#Data
#from dbn_datasets_cupy import DBNDataset
from dbn_datasets import DBNDataset
from dbn_datasets_conv import DBNDatasetConv 
#from utils_cupy import numpy_to_torch, read_yaml, get_read_func, get_scaler
from utils import numpy_to_torch, read_yaml, get_read_func, get_scaler

#Input Parsing
import yaml
import argparse
from datetime import timedelta

#Serialization
import pickle

SEED = 42

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True



def get_model_dbn(yml_conf, visible_shape):
    #Get config values 
    data_test = yml_conf["data"]["files_test"]
    data_train = yml_conf["data"]["files_train"]

    #num_loader_workers = int(yml_conf["data"]["num_loader_workers"])

    out_dir = yml_conf["output"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)


    temp = None
    nesterov_accel = None
    model_type = None
    encoder_arch = None
    gibbs_steps = None
    learning_rate = None
    momentum = None
    decay = None
    normalize_learnergy = None
    batch_normalize = None
    tiled = False
    padding = None
    stride = None
    tile = False
    tile_size = None 
    tile_step = None
    auto_clust = -1
    use_gpu = False
    batch_size = 0
    cluster_batch_size = 0
    cluster_epochs = 0
    cluster_gauss_noise_stdev = 1
    cluster_lambda = 1.0
    epochs = 0
    model_type = yml_conf["encoder"]["params"]["model_type"]
    dbn_arch = tuple(yml_conf["encoder"]["params"]["dbn_arch"])
    gibbs_steps = tuple(yml_conf["encoder"]["params"]["gibbs_steps"])
    learning_rate = tuple(yml_conf["encoder"]["params"]["learning_rate"])
    momentum = tuple(yml_conf["encoder"]["params"]["momentum"])
    decay = tuple(yml_conf["encoder"]["params"]["decay"])
    normalize_learnergy = tuple(yml_conf["encoder"]["params"]["normalize_learnergy"])
    batch_normalize = tuple(yml_conf["encoder"]["params"]["batch_normalize"])
 
    subset_training = yml_conf["encoder"]["subset_training"]

    generate_train_output = yml_conf["output"]["generate_train_output"]

    tiled = yml_conf["encoder"]["tiled"]
    if tiled:
        tile = yml_conf["data"]["tile"]
        tile_size = yml_conf["data"]["tile_size"]
        tile_step = yml_conf["data"]["tile_step"]
        temp = tuple(yml_conf["encoder"]["params"]["temp"])
        stride = yml_conf["encoder"]["params"]["stride"]
        padding = yml_conf["encoder"]["params"]["padding"]
        tiled = True
    else:
        temp = tuple(yml_conf["encoder"]["params"]["temp"])
    nesterov_accel = tuple(yml_conf["encoder"]["params"]["nesterov_accel"])
 
    use_gpu = yml_conf["encoder"]["training"]["use_gpu"]
    batch_size = yml_conf["encoder"]["training"]["batch_size"]
    cluster_batch_size = yml_conf["encoder"]["training"]["cluster_batch_size"]
    cluster_epochs =  yml_conf["encoder"]["training"]["cluster_epochs"]
    cluster_gauss_noise_stdev = yml_conf["encoder"]["training"]["cluster_gauss_noise_stdev"]
    cluster_lambda = yml_conf["encoder"]["training"]["cluster_lambda"]
    epochs = yml_conf["encoder"]["training"]["epochs"]
 
    #gen_output = yml_conf["output"]["generate_output"]
 
    #stratify_data = None
    #if "stratify_data" in yml_conf["dbn"]["training"]:
    #    stratify_data = yml_conf["dbn"]["training"]["stratify_data"]

    local_rank = 0
    if "LOCAL_RANK" in os.environ.keys():
        setup_ddp(device_ids, use_gpu)
        local_rank = int(os.environ["LOCAL_RANK"])

    if not tiled:
        #ResidualDBN?
        new_dbn = DBN(model=model_type, n_visible=visible_shape, n_hidden=dbn_arch, steps=gibbs_steps, \
            learning_rate=learning_rate, momentum=momentum, decay=decay, temperature=temp, use_gpu=use_gpu)
    else:
        mp = [False]*len(dbn_arch[1])
        new_dbn = ConvDBN(model=model_type, visible_shape=visible_shape, filter_shape = dbn_arch[1], n_filters = dbn_arch[0], \
        n_channels=number_channel, stride=stride, padding=padding, steps=gibbs_steps, learning_rate=learning_rate, momentum=momentum, \
        decay=decay, use_gpu=use_gpu, maxpooling=mp)
 
    if use_gpu:
        torch.cuda.manual_seed_all(SEED)
        device = torch.device("cuda:{}".format(local_rank))
    else:
        device = torch.device("cpu:{}".format(local_rank))

    for i in range(len(new_dbn.models)):
        if not isinstance(new_dbn.models[i], torch.nn.MaxPool2d):
           new_dbn.models[i]._optimizer = opt.SGD(new_dbn.models[i].parameters(), lr=learning_rate[i], momentum=momentum[i], weight_decay=decay[i], nesterov=nesterov_accel[i])
           new_dbn.models[i].normalize = normalize_learnergy[i]
           new_dbn.models[i].batch_normalize = batch_normalize[i]

    return new_dbn



"""TODO
def get_dbn_opt(yml_conf, module, trainer):
for i in range(len(new_dbn.models)):
        if not isinstance(new_dbn.models[i], torch.nn.MaxPool2d):
           new_dbn.models[i]._optimizer = opt.SGD(new_dbn.models[i].parameters(), lr=learning_rate[i], momentum=momentum[i], weight_decay=decay[i], nesterov=nesterov_accel[i])
           new_dbn.models[i].normalize = normalize_learnergy[i]
           new_dbn.models[i].batch_normalize = batch_normalize[i]
        if "LOCAL_RANK" in os.environ.keys():
            if not isinstance(new_dbn.models[i], torch.nn.MaxPool2d):
                new_dbn.models[i] = DDP(new_dbn.models[i], device_ids=[new_dbn.models[i].torch_device], output_device=new_dbn.models[i].torch_device) #, device_ids=device_ids)
            else:
                new_dbn.models[i] = new_dbn.models[i]
"""


#TODO Gen Output

        if not gen_output:
            return
        #TODO: For now set all subsetting to 1 - will remove subsetting later. 
        #Maintain output_subset_count - is/will be used by DataLoader in generate_output
        #Generate test datasets
        x3 = None
        scaler = None
        fname_begin = "file"
        if auto_clust > 0:
            fname_begin = fname_begin +"_clust"
        scaler = x2.scaler
        transform = x2.transform
        del x2
        for t in range(0, len(data_test)):
	
            fbase = data_test[t]
            while isinstance(fbase, list):
                fbase = fbase[0]

            fname_begin = os.path.basename(fbase) + ".clust"
            if not conv:
                x3 = DBNDataset()
                x3.read_and_preprocess_data([data_test[t]], read_func, data_reader_kwargs, pixel_padding, delete_chans=delete_chans, valid_min=valid_min, valid_max=valid_max, \
                    fill_value = fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, scaler=scaler, scale = scale_data, \
				transform=transform,  subset=output_subset_count)
            else:
                x3 = DBNDatasetConv()
                x3.read_and_preprocess_data([data_test[t]], read_func, data_reader_kwargs,  delete_chans=delete_chans, valid_min=valid_min, valid_max=valid_max, \
                fill_value = fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, transform = transform, \
                subset=output_subset_count, tile=tile, tile_size=tile_size, tile_step=tile_step)
                x3.scaler = None

            if x3.data_full is None or x3.data_full.shape[0] == 0:
                continue
 
            generate_output(x3, final_model, use_gpu, out_dir, fname_begin + testing_output, testing_mse, output_subset_count, True, conv)
    
 
        x2 = None
        #Generate output from training datasets. Doing it this way, because generating all at once creates memory issues
        if generate_train_output:
            for t in range(0, len(data_train)):

                fbase = data_train[t]
                while isinstance(fbase, list):
                    fbase = fbase[0]

                fname_begin = os.path.basename(fbase) + ".clust"
                if not conv:
                    x2 = DBNDataset()
                    x2.read_and_preprocess_data([data_train[t]], read_func, data_reader_kwargs, pixel_padding, delete_chans=delete_chans, valid_min=valid_min, \
                       valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, \
                       scaler = scaler, scale = scale_data, transform=None, subset=output_subset_count)
                else:
                    x2 = DBNDatasetConv()
                    x2.read_and_preprocess_data([data_train[t]], read_func, data_reader_kwargs,  delete_chans=delete_chans, valid_min=valid_min, valid_max=valid_max, \
                        fill_value = fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, transform = transform, \
                        subset=output_subset_count, tile=tile, tile_size=tile_size, tile_step=tile_step)
                    x2.scaler = None 
 
                if x2.data_full is None or x2.data_full.shape[0] == 0:
                    continue
                generate_output(x2, final_model, use_gpu, out_dir, fname_begin + training_output, training_mse, output_subset_count, True, conv)

    if "LOCAL_RANK" in os.environ.keys():
        cleanup_ddp() 


