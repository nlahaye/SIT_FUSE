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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
#from torch.nn import DataParallel as DDP
from learnergy.models.deep import DBN, ResidualDBN, ConvDBN


#from rbm_models.fcn_dbn import DBNUnet
from rbm_models.clust_dbn import ClustDBN
from rbm_models.heirarchichal_deep_clust import HeirClust
#from rbm_models.clust_dbn_2d import ClustDBN2D
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
from utils import read_yaml, get_read_func, get_scaler

#Input Parsing
import yaml
import argparse
from datetime import timedelta

#Serialization
import pickle

SEED = 42

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def setup_ddp(device_ids, use_gpu=True):
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = port
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(device_ids)
    #os.environ['NCCL_SOCKET_IFNAME'] = "lo"

    driver = "nccl"
    if not use_gpu or not torch.cuda.is_available():
        driver = "gloo"

    # initialize the process group
    dist.init_process_group(driver, timeout=timedelta(seconds=5400))

def cleanup_ddp():
    dist.destroy_process_group()


def run_dbn(yml_conf):

    #Get config values 
    data_test = yml_conf["data"]["files_test"]
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
    subset_count = yml_conf["data"]["subset_count"]
    output_subset_count = yml_conf["data"]["output_subset_count"]
    scale_data = yml_conf["data"]["scale_data"]

    transform_chans = yml_conf["data"]["transform_default"]["chans"]
    transform_values = 	yml_conf["data"]["transform_default"]["transform"]

    num_loader_workers = int(yml_conf["data"]["num_loader_workers"])

    out_dir = yml_conf["output"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    model_fname = yml_conf["output"]["model"]
    model_file = os.path.join(out_dir, model_fname)
    heir_model_file = os.path.join(out_dir, "heir_" + model_fname)
    heir_model_tiers = yml_conf["dbn"]["heir_tiers"]
       
 
    training_output = yml_conf["output"]["training_output"]
    training_mse = yml_conf["output"]["training_mse"]
    testing_output = yml_conf["output"]["testing_output"]
    testing_mse = yml_conf["output"]["testing_mse"]

    overwrite_model = yml_conf["dbn"]["overwrite_model"] 
    tune_scaler = yml_conf["dbn"]["tune_scaler"]
    tune_dbn = yml_conf["dbn"]["tune_dbn"]
    tune_clust = yml_conf["dbn"]["tune_clust"]
    use_gpu_pre = yml_conf["dbn"]["training"]["use_gpu_preprocessing"]
    device_ids = yml_conf["dbn"]["training"]["device_ids"] 

    conv = yml_conf["dbn"]["conv"]
 
    temp = None
    nesterov_accel = None
    model_type = None
    dbn_arch = None
    gibbs_steps = None
    learning_rate = None
    momentum = None
    decay = None
    normalize_learnergy = None
    batch_normalize = None
    fcn = False
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
    model_type = yml_conf["dbn"]["params"]["model_type"]
    dbn_arch = tuple(yml_conf["dbn"]["params"]["dbn_arch"])
    gibbs_steps = tuple(yml_conf["dbn"]["params"]["gibbs_steps"])
    learning_rate = tuple(yml_conf["dbn"]["params"]["learning_rate"])
    momentum = tuple(yml_conf["dbn"]["params"]["momentum"])
    decay = tuple(yml_conf["dbn"]["params"]["decay"])
    normalize_learnergy = tuple(yml_conf["dbn"]["params"]["normalize_learnergy"])
    batch_normalize = tuple(yml_conf["dbn"]["params"]["batch_normalize"])
 
    subset_training = yml_conf["dbn"]["subset_training"]

    generate_train_output = yml_conf["output"]["generate_train_output"]

    padding = None
    stride = None
    if conv:
        tile = yml_conf["data"]["tile"]
        tile_size = yml_conf["data"]["tile_size"]
        tile_step = yml_conf["data"]["tile_step"]
        temp = tuple(yml_conf["dbn"]["params"]["temp"])
        stride = yml_conf["dbn"]["params"]["stride"]
        padding = yml_conf["dbn"]["params"]["padding"]
        conv = True
    else:
        temp = tuple(yml_conf["dbn"]["params"]["temp"])
    nesterov_accel = tuple(yml_conf["dbn"]["params"]["nesterov_accel"])
 
    auto_clust = yml_conf["dbn"]["deep_cluster"]
    use_gpu = yml_conf["dbn"]["training"]["use_gpu"]
    batch_size = yml_conf["dbn"]["training"]["batch_size"]
    cluster_batch_size = yml_conf["dbn"]["training"]["cluster_batch_size"]
    cluster_epochs =  yml_conf["dbn"]["training"]["cluster_epochs"]
    cluster_gauss_noise_stdev = yml_conf["dbn"]["training"]["cluster_gauss_noise_stdev"]
    cluster_lambda = yml_conf["dbn"]["training"]["cluster_lambda"]
    epochs = yml_conf["dbn"]["training"]["epochs"]
 
    gen_output = yml_conf["output"]["generate_output"]

 
    stratify_data = None
    if "stratify_data" in yml_conf["dbn"]["training"]:
        stratify_data = yml_conf["dbn"]["training"]["stratify_data"]

    scaler = None
    scaler_train = True 
    scaler_fname = os.path.join(out_dir, "dbn_scaler.pkl")

    preprocess_train = True
    targets_fname = os.path.join(out_dir, "train_data.indices.npy")
    data_fname = os.path.join(out_dir, "train_data.npy")


    heir_min_samples = yml_conf["dbn"]["training"]["heir_cluster_min_samples"]
    heir_gauss_stdevs = yml_conf["dbn"]["training"]["heir_cluster_gauss_noise_stdev"]
    heir_epochs = yml_conf["dbn"]["training"]["heir_epochs"]
    heir_tune_subtrees = yml_conf["dbn"]["training"]["heir_tune_subtrees"]
    heir_tune_subtree_list = yml_conf["dbn"]["training"]["heir_tune_subtree_list"]
    n_heir_classes = yml_conf["dbn"]["training"]["heir_deep_cluster"]

    if os.path.exists(targets_fname) and os.path.exists(data_fname):
        preprocess_train = False

 
    if os.path.exists(scaler_fname): 
        scaler = load(scaler_fname)
    scaler_train = False

    os.environ['PREPROCESS_GPU'] = str(int(use_gpu_pre))

    local_rank = 0
    if "LOCAL_RANK" in os.environ.keys():
        setup_ddp(device_ids, use_gpu)
        local_rank = int(os.environ["LOCAL_RANK"])

    read_func = get_read_func(data_reader)

    if subset_count > 1: 
        print("WARNING: Making subset count > 1 for training data may lead to suboptimal results")


    if preprocess_train: 
        if stratify_data is not None:
            strat_read_func = get_read_func(stratify_data["reader"]) 
            stratify_data["reader"] = strat_read_func

 
        if not conv:
            x2 = DBNDataset()
            x2.read_and_preprocess_data(data_train, read_func, data_reader_kwargs, pixel_padding, delete_chans=delete_chans, \
                valid_min=valid_min, valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, \
                transform_values=transform_values, scaler = scaler, train_scaler = scaler_train, scale = scale_data, \
                transform=None, subset=subset_count, subset_training = subset_training, stratify_data=stratify_data)
        else:
            x2 = DBNDatasetConv()
            x2.read_and_preprocess_data(data_train, read_func, data_reader_kwargs, delete_chans=delete_chans, \
                 valid_min=valid_min, valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, \
                 transform_values=transform_values, transform=None, subset=subset_count, tile=tile, tile_size=tile_size, tile_step=tile_step,
                 subset_training = subset_training)
    else:
        if not conv:
            x2 = DBNDataset()
            x2.read_data_preprocessed(data_fname, targets_fname, scaler)
        else:
            x2 = DBNDatasetConv()
            x2.read_data_preprocessed(data_fname, targets_fname, scaler)
 
    if x2.train_indices is not None:
        np.save(os.path.join(out_dir, "train_indices"), x2.train_indices)
 
    #Generate model
    if not conv:
        chunk_size = 1
        for i in range(1,pixel_padding+1):
            chunk_size = chunk_size + (8*i)
    else:
        chunk_size = (x2.data_full.shape[2],x2.data_full.shape[3]) #TODO generalize
        #chunk_size = x2.data_full.shape[x2.chan_dim+1] 


    #TODO random sample in test, fix for generic case for commit
    if not conv:
        new_dbn = DBN(model=model_type, n_visible=x2.data_full.shape[1], n_hidden=dbn_arch, steps=gibbs_steps, \
            learning_rate=learning_rate, momentum=momentum, decay=decay, temperature=temp, use_gpu=use_gpu)
    else:
        new_dbn = ConvDBN(model=model_type, visible_shape=chunk_size, filter_shape = dbn_arch[1], n_filters = dbn_arch[0], \
        n_channels=number_channel, stride=stride, padding=padding, steps=gibbs_steps, learning_rate=learning_rate, momentum=momentum, \
        decay=decay, use_gpu=use_gpu, maxpooling=[False]) 

    if use_gpu:
        torch.cuda.manual_seed_all(SEED)
        device = torch.device("cuda:{}".format(local_rank))
    else:
        device = torch.device("cpu:{}".format(local_rank))

    for i in range(len(new_dbn.models)):
        if "LOCAL_RANK" in os.environ.keys():
            if not isinstance(new_dbn.models[i], torch.nn.MaxPool2d):
                new_dbn.models[i] = DDP(new_dbn.models[i], device_ids=[local_rank], output_device=local_rank) #, device_ids=device_ids)
            else:
                new_dbn.models[i] = new_dbn.models[i]
 
    clust_scaler = None
    if os.path.exists(os.path.join(out_dir, "fc_clust_scaler.pkl")) and not overwrite_model:
            with open(os.path.join(out_dir, "fc_clust_scaler.pkl"), "rb") as f:
                clust_scaler = load(f)

    if not conv:
            clust_dbn = ClustDBN(new_dbn, dbn_arch[-1], auto_clust, True, clust_scaler) #TODO parameterize
    else:
            if "LOCAL_RANK" in os.environ.keys():
                dbn_hidden = new_dbn.models[-1].module.hidden_shape
                dbn_filters = new_dbn.models[-1].module.n_filters
            else:
                #dbn_hidden = new_dbn.models[-1].hidden_shape
                dbn_filters = new_dbn.models[-1].n_filters
            visible_shape  = dbn_filters
            clust_dbn = ClustDBN(new_dbn, visible_shape, auto_clust, True, clust_scaler)

    clust_dbn.fc = DDP(clust_dbn.fc, device_ids=[local_rank], output_device=local_rank)
    final_model = clust_dbn
    final_model.eval()
    final_model.fc.eval()
    final_model.dbn_trunk.eval()
    print("Loading pre-existing model")
    final_model.dbn_trunk.load_state_dict(torch.load(model_file + ".ckpt"))

    if os.path.exists(model_file + "_fc_clust.ckpt") and not overwrite_model:
        final_model.fc.load_state_dict(torch.load(model_file + "_fc_clust.ckpt"))

    dist.barrier()


    heir_clust = None 

    for tiers in range(0,heir_model_tiers):

        print("HEIRARCHICAL TIER ", str(tiers + 1))

        heir_mdl_file = heir_model_file + ""
        if tiers > 0:
            heir_mdl_file = heir_model_file + "_" + str(tiers)
 
        print(heir_mdl_file)
        if not os.path.exists(heir_mdl_file + ".ckpt") or overwrite_model: 
            heir_clust = HeirClust(final_model, x2, n_heir_classes, use_gpu=use_gpu, min_samples=heir_min_samples, gauss_stdevs = heir_gauss_stdevs)
            heir_clust.fit(x2, epochs=heir_epochs)

            state_dict = heir_clust.get_state_dict(out_dir) #TODO
            torch.save(state_dict, heir_mdl_file + ".ckpt")
        else:
            heir_clust = HeirClust(final_model, x2, n_heir_classes, use_gpu=use_gpu, min_samples=heir_min_samples, gauss_stdevs = heir_gauss_stdevs)
            heir_dict = torch.load(heir_mdl_file + ".ckpt")
            heir_clust.load_model(heir_dict)

            if heir_tune_subtrees:
                heir_clust.fit(x2, epochs = heir_epochs, tune_subtrees =  heir_tune_subtree_list)        
            
        final_model = heir_clust

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

        #TODO update arguments/signatures

        fname_begin = os.path.basename(fbase) + ".heir_clust" + str(heir_model_tiers)
        if not conv:
            x3 = DBNDataset()
            x3.read_and_preprocess_data([data_test[t]], read_func, data_reader_kwargs, pixel_padding, delete_chans=delete_chans, valid_min=valid_min, valid_max=valid_max, \
                fill_value = fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, scaler=scaler, scale = scale_data, \
                transform=transform,  subset=subset_count)
        else:
           x3 = DBNDatasetConv()
           x3.read_and_preprocess_data([data_test[t]], read_func, data_reader_kwargs,  delete_chans=delete_chans, valid_min=valid_min, valid_max=valid_max, \
               fill_value = fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, transform = transform, \
               subset=output_subset_count, tile=tile, tile_size=tile_size, tile_step=tile_step)
           x3.scaler = None 

        if x3.data_full is None or x3.data_full.shape[0] == 0:
                continue

        generate_output(x3, heir_clust, use_gpu, out_dir, fname_begin + testing_output, testing_mse, output_subset_count, (not use_gpu_pre))
    

 
    x2 = None
    #Generate output from training datasets. Doing it this way, because generating all at once creates memory issues
    if generate_train_output:
        for t in range(0, len(data_train)):
            fbase = data_train[t]
            while isinstance(fbase, list):
                fbase = fbase[0]

            fname_begin = os.path.basename(fbase) + ".heir_clust"  + str(heir_model_tiers)
            if not conv:
                x2 = DBNDataset()
                x2.read_and_preprocess_data([data_train[t]], read_func, data_reader_kwargs, pixel_padding, delete_chans=delete_chans, valid_min=valid_min, \
                   valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, \
                   scaler = scaler, scale = scale_data, transform=None, subset=subset_count)
            else:
                x2 = DBNDatasetConv()
                x2.read_and_preprocess_data([data_train[t]], read_func, data_reader_kwargs,  delete_chans=delete_chans, valid_min=valid_min, valid_max=valid_max, \
                    fill_value = fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, transform = transform, \
                    subset=output_subset_count, tile=tile, tile_size=tile_size, tile_step=tile_step)
                x2.scaler = None
 
            if x2.data_full is None or x2.data_full.shape[0] == 0:
                continue

            generate_output(x2, heir_clust, use_gpu, out_dir, fname_begin + training_output, training_mse, output_subset_count, (not use_gpu_pre)) 

    if "LOCAL_RANK" in os.environ.keys():
        cleanup_ddp() 




def generate_output(dat, mdl, use_gpu, out_dir, output_fle, mse_fle, output_subset_count, pin_mem = False):
    output_full = None
    count = 0
    dat.current_subset = -1
    dat.next_subset()

    local_rank = 0
    if "LOCAL_RANK" in os.environ.keys():
        local_rank = int(os.environ["LOCAL_RANK"])
    if use_gpu:
        device = torch.device("cuda:{}".format(local_rank))
    else:
        device = torch.device("cpu:{}".format(local_rank))

    ind = 0
    while(count == 0 or dat.has_next_subset() or (dat.subset > 1 and dat.current_subset > (dat.subset-2))):
        output_batch_size = min(5000, max(int(dat.data_full.shape[0] / 5), dat.data_full.shape[0]))

        output_sze = dat.data_full.shape[0]
        append_remainder = int(output_batch_size - (output_sze % output_batch_size))

        if isinstance(dat.data_full,torch.Tensor):
            dat.data_full = torch.cat((dat.data_full,dat.data_full[0:append_remainder]))
            dat.targets_full = torch.cat((dat.targets_full,dat.targets_full[0:append_remainder]))
        else:
            dat.data_full = np.concatenate((dat.data_full,dat.data_full[0:append_remainder]))
            dat.targets_full = np.concatenate((dat.targets_full,dat.targets_full[0:append_remainder]))

        dat.current_subset = -1
        dat.next_subset()

        test_loader = DataLoader(dat, batch_size=output_batch_size, shuffle=False, \
        num_workers = 0, drop_last = False, pin_memory = pin_mem) 
        ind = 0
        ind2 = 0 
        for data in tqdm(test_loader):
            dat_dev, lab_dev = data[0].to(device=device, non_blocking=True), data[1].to(device=device, non_blocking=True)
            dev_ds = TensorDataset(dat_dev, lab_dev)

            output = mdl.forward(dat_dev)  
            if isinstance(output, list):
                output = output[0] #TODO improve usage uf multi-headed output after single-headed approach validated

            if use_gpu == True:
                output = output.detach().cpu()
            loader = DataLoader(dev_ds, batch_size=output_batch_size, shuffle=False, \
                num_workers = 0, drop_last = False, pin_memory = False)
            dat_dev = dat_dev.detach().cpu()
            lab_dev = lab_dev.detach().cpu()
            del dev_ds

            if output_full is None:
                #if not fcn:
                print(dat.data_full.shape, output.shape)
                output_full = torch.zeros(dat.data_full.shape[0], 1, dtype=torch.float32)
                #else:
                #    output_full = torch.zeros(dat.data_full.shape[0], dat.data_full.shape[1], output.shape[2], dtype=torch.float32)       
            ind1 = ind2 
            ind2 += dat_dev.shape[0]
            if ind2 > output_full.shape[0]:
                ind2 = output_full.shape[0]
            output_full[ind1:ind2,:] = output
            ind = ind + 1
            del output
            del dat_dev
            del lab_dev
            del loader
        count = count + 1
        if dat.has_next_subset():
            dat.next_subset()
        else:
            break 
    #Save training output
    print("FILLING OUTPUT", ind1, ind2, torch.unique(output_full), torch.unique(output_full).shape)
    print("SAVING", os.path.join(out_dir, output_fle))
    torch.save(output_full, os.path.join(out_dir, output_fle), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(dat.targets_full, os.path.join(out_dir, output_fle + ".indices"), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(dat.data_full, os.path.join(out_dir, output_fle + ".input"), pickle_protocol=pickle.HIGHEST_PROTOCOL)


def main(yml_fpath):
	#Translate config to dictionary 
	yml_conf = read_yaml(yml_fpath)
	#Run 
	run_dbn(yml_conf)


if __name__ == '__main__':
	 
	parser = argparse.ArgumentParser()
	parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
	args = parser.parse_args()
	from timeit import default_timer as timer
	start = timer()
	main(args.yaml)
	end = timer()
	print(end - start) # Time in seconds, e.g. 5.38091952400282


