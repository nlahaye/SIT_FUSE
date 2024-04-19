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
from torchvision import transforms
#from torch.nn import DataParallel as DDP
from learnergy.models.deep import DBN, ResidualDBN, ConvDBN
from torch.utils.data.distributed import DistributedSampler
import multichannel_resnet


#from rbm_models.fcn_dbn import DBNUnet
from rbm_models.clust_dbn import ClustDBN
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

def setup_ddp(device_ids, use_gpu=True):
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = port
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(device_ids)
    #os.environ['NCCL_SOCKET_IFNAME'] = "lo"

    print("HERE CUDA_VISIBLE_DEVICES", os.environ['CUDA_VISIBLE_DEVICES'])

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
    conv = False
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

    conv = yml_conf["dbn"]["conv"]
    if conv:
        tile = yml_conf["data"]["tile"]
        tile_size = yml_conf["data"]["tile_size"]
        tile_step = yml_conf["data"]["tile_step"]
        temp = tuple(yml_conf["dbn"]["params"]["temp"])
        #stride = yml_conf["dbn"]["params"]["stride"]
        #padding = yml_conf["dbn"]["params"]["padding"]
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

    if os.path.exists(targets_fname) and os.path.exists(data_fname):
        preprocess_train = False


   
    scaler_tune = True #TODO configurable
    if not os.path.exists(scaler_fname) or (preprocess_train == True and overwrite_model):    
        scaler_type = yml_conf["scaler"]["name"]
        scaler, scaler_train = get_scaler(scaler_type, cuda = use_gpu_pre)
    else:
        scaler = load(scaler_fname)
        scaler_train = False
        if tune_scaler:
            scaler_train = True

    os.environ['PREPROCESS_GPU'] = str(int(use_gpu_pre))

    local_rank = 0
    if "LOCAL_RANK" in os.environ.keys():
        setup_ddp(device_ids, use_gpu)
        local_rank = int(os.environ["LOCAL_RANK"])

    read_func = get_read_func(data_reader)

    if subset_count > 1: 
        print("WARNING: Making subset count > 1 for training data may lead to suboptimal results")

    if not conv:
        #TODO stratify conv data

        if preprocess_train: 
            if stratify_data is not None and "kmeans" not in stratify_data:
                strat_read_func = get_read_func(stratify_data["reader"]) 
                stratify_data["reader"] = strat_read_func

            x2 = DBNDataset()
            x2.read_and_preprocess_data(data_train, read_func, data_reader_kwargs, pixel_padding, delete_chans=delete_chans, \
                valid_min=valid_min, valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, \
                transform_values=transform_values, scaler = scaler, train_scaler = scaler_train, scale = scale_data, \
                transform=None, subset=subset_count, subset_training = subset_training, stratify_data=stratify_data)
        else:
            x2 = DBNDataset()
            x2.read_data_preprocessed(data_fname, targets_fname, scaler)
    else:
        if preprocess_train:
            x2 = DBNDatasetConv()
            x2.read_and_preprocess_data(data_train, read_func, data_reader_kwargs, delete_chans=delete_chans, \
                 valid_min=valid_min, valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, \
                 transform_values=transform_values, transform=None, subset=subset_count, tile=tile, tile_size=tile_size, tile_step=tile_step,
                 subset_training = subset_training)
        else:
           transform = None
           if os.path.exists(os.path.join(out_dir, "dbn_data_transform.ckpt")):
               state_dict = torch.load(os.path.join(out_dir, "dbn_data_transform.ckpt"))
               transform = torch.nn.Sequential(
                                transforms.Normalize(state_dict["mean_per_channel"], state_dict["std_per_channel"])
                        )            
           x2 = DBNDatasetConv()
           x2.read_data_preprocessed(data_fname, targets_fname, transform=transform)

    print(x2.data_full.shape)
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
    #if not conv:
    #    #ResidualDBN?
    #    new_dbn = DBN(model=model_type, n_visible=x2.data_full.shape[1], n_hidden=dbn_arch, steps=gibbs_steps, \
    #        learning_rate=learning_rate, momentum=momentum, decay=decay, temperature=temp, use_gpu=use_gpu)
    #else:
    #    mp = [False]*len(dbn_arch[1])
    #    new_dbn = ConvDBN(model=model_type, visible_shape=chunk_size, filter_shape = dbn_arch[1], n_filters = dbn_arch[0], \
    #    n_channels=number_channel, stride=stride, padding=padding, steps=gibbs_steps, learning_rate=learning_rate, momentum=momentum, \
    #    decay=decay, use_gpu=use_gpu, maxpooling=mp)

    new_dbn = multichannel_resnet.Resnet_multichannel(pretrained=False, encoder_depth=152, num_in_channels=34).cuda()
    #new_dbn.load_state_dict(torch.load("/data/nlahaye/output/Learnergy/DBN_eMAS_FULL_STRAT_RESNET_BACKBONE_RGB/segmentation_backbone_resnet50_rgb_state_dict_no_pretrain.pt"))
    new_dbn.load_state_dict(torch.load("/data/nlahaye/output/Learnergy/DBN_eMAS_FULL_STRAT_RESNET_BACKBONE/segmentation_backbone_resnet50_state_dict_no_pretrain.pt"))

    print("MODULE LEN", len(list(new_dbn.children())) )
    modules = list(new_dbn.children())[:-4]
    new_dbn = nn.Sequential(*modules)
  
    new_dbn = DDP(new_dbn, device_ids=[0])  
 
    #new_dbn.load_state_dict(torch.load("/data/nlahaye/output/Learnergy/DBN_eMAS_FULL_STRAT_RESNET_BACKBONE_RGB/segmentation_backbone_resnet50_rgb_state_dict.pt"))

    #new_dbn = torch.load("segmentation_backbone_resnet50.pt") 
    #torch.save(new_dbn.state_dict(), "segmentation_backbone_resnet50_state_dict.pt")


    if use_gpu:
        torch.cuda.manual_seed_all(SEED)
        device = torch.device("cuda:{}".format(local_rank))
    else:
        device = torch.device("cpu:{}".format(local_rank))

    #for i in range(len(new_dbn.models)):
    #    if not isinstance(new_dbn.models[i], torch.nn.MaxPool2d):
    #       new_dbn.models[i]._optimizer = opt.SGD(new_dbn.models[i].parameters(), lr=learning_rate[i], momentum=momentum[i], weight_decay=decay[i], nesterov=nesterov_accel[i])
    #       new_dbn.models[i].normalize = normalize_learnergy[i]
    #       new_dbn.models[i].batch_normalize = batch_normalize[i]
    #    if "LOCAL_RANK" in os.environ.keys():
    #        if not isinstance(new_dbn.models[i], torch.nn.MaxPool2d):
    #            new_dbn.models[i] = DDP(new_dbn.models[i], device_ids=[new_dbn.models[i].torch_device], output_device=new_dbn.models[i].torch_device) #, device_ids=device_ids)
     #       else:
     #           new_dbn.models[i] = new_dbn.models[i]
 

    #if not os.path.exists(model_file + ".ckpt") or overwrite_model: 
    #    #Train model
    #    count = 0
    #    pl = None
    #    while(count == 0 or x2.has_next_subset()):
    #        if conv:
    #            mse = \
    #                new_dbn.fit(x2, batch_size=batch_size, epochs=epochs,
    #                    is_distributed = True, num_loader_workers = num_loader_workers, pin_memory=(not use_gpu_pre)) #int(os.cpu_count() / 3))
    #        else:
    #            mse, pl = \
    #                new_dbn.fit(x2, batch_size=batch_size, epochs=epochs,
    #                    is_distributed = True, num_loader_workers = num_loader_workers, pin_memory=(not use_gpu_pre)) #int(os.cpu_count() / 3))
    #        count = count + 1
    #        x2.next_subset()
    #    dist.barrier()
    #    if local_rank == 0:
    #        for i in range(len(new_dbn.models)):	
    #            if not isinstance(new_dbn.models[i], torch.nn.MaxPool2d):
    #                converge.plot(new_dbn.models[i].module._history['mse'],
    #                    labels=['MSE'], title='convergence', subtitle='Model: Restricted Boltzmann Machine')
    #                plt.savefig(os.path.join(out_dir, "mse_plot_layer" + str(i) + ".png"))
    #                plt.clf()
    #                if pl is not None:
    #                    if 'pl' in new_dbn.models[i].module._history:
    #                        converge.plot( new_dbn.models[i].module._history['pl'],
    #                            labels=['log-PL'], title='Log-PL', subtitle='Model: Restricted Boltzmann Machine')
    #                        plt.savefig(os.path.join(out_dir, "log-pl_plot_layer" + str(i) + ".png"))
    #                        plt.clf()
    #                    if 'time' in new_dbn.models[i].module._history:
    #                        converge.plot( new_dbn.models[i].module._history['time'],
    #                            labels=['time (s)'], title='Training Time', subtitle='Model: Restricted Boltzmann Machine')
    #                        plt.savefig(os.path.join(out_dir, "time_plot_layer" + str(i) + ".png"))
    #                        plt.clf()

    final_model = new_dbn
    if auto_clust > 0:
        #sze = 0
        #if not fcn:
        #    sze = dbn_arch[-1]
        #else:
        #    #viz = 9*9*2 #new_dbn.models[-1].module.visible_shape
        #    #filts = new_dbn.models[-1].module.n_filters
        #    #sze = viz #viz[0]*viz[1]*filts
 
        clust_scaler = None
        if os.path.exists(os.path.join(out_dir, "fc_clust_scaler.pkl")) and not overwrite_model:
                with open(os.path.join(out_dir, "fc_clust_scaler.pkl"), "rb") as f:
                    clust_scaler = load(f)
 
        #if not conv:
        #    clust_dbn = ClustDBN(new_dbn, dbn_arch[-1], auto_clust, True, clust_scaler) #TODO parameterize
        #else:
            #if "LOCAL_RANK" in os.environ.keys():
            #    dbn_hidden = new_dbn.models[-1].module.hidden_shape
            #    dbn_filters = new_dbn.models[-1].module.n_filters
            #else:
            #    dbn_hidden = new_dbn.models[-1].hidden_shape
            #    dbn_filters = new_dbn.models[-1].out_channels #.n_filters
        visible_shape  = int(512*2*2*16) #dbn_filters * dbn_hidden[0] * dbn_hidden[1]
        clust_dbn = ClustDBN(new_dbn, visible_shape, auto_clust, True, clust_scaler)
        clust_dbn.fc = DDP(clust_dbn.fc, device_ids=[local_rank], output_device=local_rank)
        final_model = clust_dbn
        if not os.path.exists(model_file + "_fc_clust.ckpt") or overwrite_model:
           dataset2 = TensorDataset(x2.data, x2.targets)
           loader = None
           is_distributed = True 
           if is_distributed:
               sampler = DistributedSampler(dataset2, shuffle=True)
               loader = DataLoader(dataset2, batch_size=cluster_batch_size, shuffle=False,
                    sampler=sampler, num_workers = num_loader_workers, pin_memory = (not use_gpu_pre),
                    drop_last=True)

           count = 0
           while(count == 0 or x2.has_next_subset()):
               final_model.fit(dataset2, cluster_batch_size, cluster_epochs, loader, sampler, cluster_gauss_noise_stdev, cluster_lambda)
               count = count + 1
               x2.next_subset()
        final_model.eval()
        final_model.fc.eval()
        final_model.dbn_trunk.eval()
    """
    if os.path.exists(model_file + ".ckpt") and not overwrite_model:
        print("Loading pre-existing model")
        print("HERE INIT PARAMS", final_model.dbn_trunk.state_dict())
        if auto_clust > 0:
            #final_model.dbn_trunk.load_state_dict(torch.load(model_file + ".ckpt"))
            print("FINAL PARAMS", final_model.dbn_trunk.state_dict())
            if tune_dbn:
                count = 0
                while(count == 0 or x2.has_next_subset()):
                    mse, pl = \
                        final_model.dbn_trunk.fit(x2, batch_size=batch_size, epochs=epochs,
                            is_distributed = True, num_loader_workers = num_loader_workers, pin_memory=(not use_gpu_pre)) #int(os.cpu_count() / 3))
                    count = count + 1
                    x2.next_subset()
                torch.save(final_model.dbn_trunk.state_dict(), model_file + ".ckpt") 

            if os.path.exists(model_file + "_fc_clust.ckpt") and not overwrite_model:
                final_model.fc.load_state_dict(torch.load(model_file + "_fc_clust.ckpt"))
                if tune_clust:
                    print("Tuning pre-existing Deep Clustering layers")
                    dataset2 = TensorDataset(x2.data, x2.targets)
                    loader = None
                    is_distributed = True
                    if is_distributed:
                        sampler = DistributedSampler(dataset2, shuffle=True)
                        loader = DataLoader(dataset2, batch_size=cluster_batch_size, shuffle=False,
                             sampler=sampler, num_workers = num_loader_workers, pin_memory = (not use_gpu_pre),
                             drop_last=True)

                    count = 0
                    final_model.fit_scaler = True
                    while(count == 0 or x2.has_next_subset()):
                        final_model.fit(dataset2, cluster_batch_size, cluster_epochs, loader, sampler, cluster_gauss_noise_stdev, cluster_lambda)
                        count = count + 1
                        x2.next_subset()
                    final_model.eval()
                    final_model.dbn_trunk.eval()
                    final_model.fc.eval() 
                    torch.save(final_model.fc.state_dict(), model_file + "_fc_clust.ckpt")
            else:
                dataset2 = TensorDataset(x2.data, x2.targets)
                loader = None
                is_distributed = True
                if is_distributed:
                    sampler = DistributedSampler(dataset2, shuffle=True)
                    loader = DataLoader(dataset2, batch_size=cluster_batch_size, shuffle=False,
                        sampler=sampler, num_workers = num_loader_workers, pin_memory = (not use_gpu_pre),
                        drop_last=True)


                count = 0
                while(count == 0 or x2.has_next_subset()):
                    final_model.fit(dataset2, cluster_batch_size, cluster_epochs, loader, sampler, cluster_gauss_noise_stdev, cluster_lambda)
                    count = count + 1
                    x2.next_subset()
                final_model.eval()
                final_model.fc.eval()
                final_model.dbn_trunk.eval()
        else:
            final_model.load_state_dict(torch.load(model_file + ".ckpt"))

            if tune_dbn:
                count = 0
                while(count == 0 or x2.has_next_subset()):
                    mse, pl = \
                            final_model.fit(x2, batch_size=batch_size, epochs=epochs,
                                is_distributed = True, num_loader_workers = num_loader_workers, pin_memory=(not use_gpu_pre)) #int(os.cpu_count() / 3))
                    count = count + 1
                    x2.next_subset()
                torch.save(final_model.state_dict(), model_file + ".ckpt")
        """

        #for m in range(len(new_dbn._models)):
        #    new_dbn._models[m].load_state_dict(model_file + "_sub_model_" + str(m) + ".ckpt") 


    #dist.barrier()
    if local_rank == 0:
        #if not os.path.exists(model_file + ".ckpt") or overwrite_model:
        #    #for i in range(len(new_dbn.models)):
        #    #    if not isinstance(new_dbn.models[i], torch.nn.MaxPool2d):
        #    #        if 'pl' in new_dbn.models[i].module._history:
        #    #            converge.plot(new_dbn.models[i].module._history['mse'], new_dbn.models[i].module._history['pl'],
        #    #                    new_dbn.models[i].module._history['time'], labels=['MSE', 'log-PL', 'time (s)'],
        #    #                    title='convergence over dataset', subtitle='Model: Restricted Boltzmann Machine')
        #    #        else:
        #    #            converge.plot(new_dbn.models[i].module._history['mse'], new_dbn.models[i].module._history['time'],
        #    #                    labels=['MSE', 'time (s)'],
        #    #                    title='convergence over dataset', subtitle='Model: Restricted Boltzmann Machine')
        #    #        plt.savefig(os.path.join(out_dir, "convergence_plot_layer" + str(i) + ".png"))
    

        if not os.path.exists(model_file + ".ckpt") or (auto_clust > 0 & os.path.exists(model_file + "_fc_clust.ckpt")) or overwrite_model:
            #Save model
            if auto_clust > 0:
                torch.save(final_model.dbn_trunk.state_dict(), model_file + ".ckpt")
                torch.save(final_model.fc.state_dict(), model_file + "_fc_clust.ckpt")
            else:
                torch.save(final_model.state_dict(), model_file + ".ckpt")
 
            if conv and not os.path.exists(os.path.join(out_dir, "dbn_data_transform.ckpt")):
                torch.save(x2.transform.state_dict(), os.path.join(out_dir, "dbn_data_transform.ckpt"))

 
            #Save scaler
            if hasattr(x2, "scaler") and x2.scaler is not None and not os.path.exists(os.path.join(out_dir, "dbn_scaler.pkl")):
                with open(os.path.join(out_dir, "dbn_scaler.pkl"), "wb") as f:
                    dump(x2.scaler, f, True, pickle.HIGHEST_PROTOCOL)
            else:
                x2.scaler = None             

            if hasattr(final_model, "scaler") and final_model.scaler is not None:
                with open(os.path.join(out_dir, "fc_clust_scaler.pkl"), "wb") as f:
                    dump(final_model.scaler, f, True, pickle.HIGHEST_PROTOCOL)
            else:
                final_model.scaler = None


        if not gen_output:
            return
 
        with torch.no_grad():
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

                for frac_ind in range(0,30):
                    fname_begin = os.path.basename(fbase) + ".clust_subset_" + str(frac_ind+1)
                    x3 = DBNDatasetConv()
                    x3.read_and_preprocess_data([data_test[t]], read_func, data_reader_kwargs,  delete_chans=delete_chans, valid_min=valid_min, valid_max=valid_max, \
                    fill_value = fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, transform = transform, \
                    subset=output_subset_count, tile=tile, tile_size=tile_size, tile_step=tile_step, data_fraction=30, data_fraction_index=frac_ind)
                    x3.scaler = None

                    if x3.data_full is None or x3.data_full.shape[0] == 0:
                        continue
 
                    generate_output(x3, final_model, use_gpu, out_dir, fname_begin + testing_output, testing_mse, output_subset_count, (not use_gpu_pre), conv)
    
 
            x2 = None
            #Generate output from training datasets. Doing it this way, because generating all at once creates memory issues
            if generate_train_output:
                for t in range(0, len(data_train)):

                    fbase = data_train[t]
                    while isinstance(fbase, list):
                        fbase = fbase[0]
     
                    for frac_ind in range(0,30):
                        fname_begin = os.path.basename(fbase) + ".clust_subset" + str(frac_ind+1)
                        x2 = DBNDatasetConv()
                        x2.read_and_preprocess_data([data_train[t]], read_func, data_reader_kwargs,  delete_chans=delete_chans, valid_min=valid_min, valid_max=valid_max, \
                            fill_value = fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, transform = transform, \
                            subset=output_subset_count, tile=tile, tile_size=tile_size, tile_step=tile_step, data_fraction=30, data_fraction_index=frac_ind)
                        x2.scaler = None 
  
                        if x2.data_full is None or x2.data_full.shape[0] == 0:
                            continue
                        generate_output(x2, final_model, use_gpu, out_dir, fname_begin + training_output, training_mse, output_subset_count, (not use_gpu_pre), conv)

    if "LOCAL_RANK" in os.environ.keys():
        cleanup_ddp() 
 
def generate_output(dat, mdl, use_gpu, out_dir, output_fle, mse_fle, output_subset_count, pin_mem = False, conv = False):
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
        output_batch_size = min(50, max(int(dat.data_full.shape[0] / 5), dat.data_full.shape[0]))

        print("HERE GENERATING OUTPUT", output_batch_size, output_subset_count, dat.current_subset, dat.subset)
  
        if count == 0:
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
            output = torch.unsqueeze(torch.argmax(output, axis = 1), axis=1) 

            if use_gpu == True:
                output = output.detach().cpu()
            loader = DataLoader(dev_ds, batch_size=output_batch_size, shuffle=False, \
                num_workers = 0, drop_last = False, pin_memory = False)
            dat_dev = dat_dev.detach().cpu()
            lab_dev = lab_dev.detach().cpu()
            del dev_ds

            if output_full is None:
                output_full = torch.zeros(dat.data_full.shape[0], output.shape[1], dtype=torch.float32)
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
    print("SAVING", os.path.join(out_dir, output_fle))
    torch.save(output_full, os.path.join(out_dir, output_fle), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(dat.targets_full, os.path.join(out_dir, output_fle + ".indices"), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    #torch.save(dat.data_full, os.path.join(out_dir, output_fle + ".input"), pickle_protocol=pickle.HIGHEST_PROTOCOL)





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


