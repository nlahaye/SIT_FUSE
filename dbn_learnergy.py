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
import random

import sys
sys.setrecursionlimit(4500)

#ML imports
import torch
import torch.nn as nn
import torch.optim as opt 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
#from torch.nn import DataParallel as DDP
from learnergy.models.deep import DBN, ResidualDBN, FCDBN

#Visualization
import learnergy.visual.convergence as converge
import matplotlib 
matplotlib.use("Agg")
import  matplotlib.pyplot as plt

#Data
#from dbn_datasets_cupy import DBNDataset
from dbn_datasets import DBNDataset
#from utils_cupy import numpy_to_torch, read_yaml, get_read_func, get_scaler
from utils import numpy_to_torch, read_yaml, get_read_func, get_scaler

#Input Parsing
import yaml
import argparse
from datetime import timedelta

#Serialization
import pickle

SEED = 42

def setup_ddp(device_ids, use_gpu=True):
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = port
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(device_ids)
    #os.environ['NCCL_SOCKET_IFNAME'] = "lo"

    driver = "nccl"
    if not use_gpu or not torch.cuda.is_available():
        driver = "gloo"

    # initialize the process group
    dist.init_process_group(driver)#, rank=rank, world_size=world_size, init_method="env://", timeout=timedelta(seconds=10))

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
    training_output = yml_conf["output"]["training_output"]
    training_mse = yml_conf["output"]["training_mse"]
    testing_output = yml_conf["output"]["testing_output"]
    testing_mse = yml_conf["output"]["testing_mse"]

    model_type = yml_conf["dbn"]["params"]["model_type"]
    dbn_arch = tuple(yml_conf["dbn"]["params"]["dbn_arch"])
    gibbs_steps = tuple(yml_conf["dbn"]["params"]["gibbs_steps"])
    learning_rate = tuple(yml_conf["dbn"]["params"]["learning_rate"])
    momentum = tuple(yml_conf["dbn"]["params"]["momentum"])
    decay = tuple(yml_conf["dbn"]["params"]["decay"])
    normalize_learnergy = tuple(yml_conf["dbn"]["params"]["normalize_learnergy"])
    batch_normalize = tuple(yml_conf["dbn"]["params"]["batch_normalize"])

    temp = None
    nesterov_accel = None

    visible_shape = None
    filter_shape = None
    stride = None
    n_filters = None
    if "conv" in model_type[0]:
        visible_shape = yml_conf["dbn"]["params"]["visible_shape"]
        filter_shape = yml_conf["dbn"]["params"]["filter_shape"]
        stride = yml_conf["dbn"]["params"]["stride"]
        n_filters = yml_conf["dbn"]["params"]["n_filters"]
    else:
        temp = tuple(yml_conf["dbn"]["params"]["temp"])
        nesterov_accel = tuple(yml_conf["dbn"]["params"]["nesterov_accel"])
 
    use_gpu = yml_conf["dbn"]["training"]["use_gpu"]
    use_gpu_pre = yml_conf["dbn"]["training"]["use_gpu_preprocessing"]
    world_size = yml_conf["dbn"]["training"]["world_size"]
    rank = yml_conf["dbn"]["training"]["rank"]
    device_ids = yml_conf["dbn"]["training"]["device_ids"]
    batch_size = yml_conf["dbn"]["training"]["batch_size"]
    epochs = yml_conf["dbn"]["training"]["epochs"]

    overwrite_model = yml_conf["dbn"]["overwrite_model"]

    scaler_type = yml_conf["scaler"]["name"]
    scaler, scaler_train = get_scaler(scaler_type, cuda = use_gpu_pre)

    os.environ['PREPROCESS_GPU'] = str(int(use_gpu_pre))

    local_rank = 0
    if "LOCAL_RANK" in os.environ.keys():
        setup_ddp(device_ids, use_gpu)
        local_rank = int(os.environ["LOCAL_RANK"])

    read_func = get_read_func(data_reader)

    #Generate model
    chunk_size = 1
    for i in range(1,pixel_padding+1):
        chunk_size = chunk_size + (8*i)

    #Generate training dataset object
    #Unsupervised, so targets are not used. Currently, I use this to store original image indices for each point 

 

    if subset_count > 1: 
        print("WARNING: Making subset count > 1 for training data may lead to suboptimal results")
    x2 = DBNDataset(data_train, read_func, data_reader_kwargs, pixel_padding, delete_chans=delete_chans, \
        valid_min=valid_min, valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, \
        transform_values=transform_values, scalers = [scaler], train_scalers = scaler_train, scale = scale_data, \
        transform=numpy_to_torch, subset=subset_count)

 
    fcn = False ##TODO fix

    model_file = os.path.join(out_dir, model_fname)
    if not os.path.exists(model_file) or overwrite_model:
        if not fcn:
            #ResidualDBN?
            new_dbn = DBN(model=model_type, n_visible=chunk_size*number_channel, n_hidden=dbn_arch, steps=gibbs_steps, \
                learning_rate=learning_rate, momentum=momentum, decay=decay, temperature=temp, use_gpu=use_gpu)
        else:
            new_dbn = FCDBN(model=model_type,visible_shape=(chunk_size, chunk_size, number_channel), n_channels=number_channel, steps=gibbs_steps, \
                learning_rate=learning_rate, momentum=momentum, decay=decay, use_gpu=use_gpu)

        if use_gpu:
            torch.cuda.manual_seed_all(SEED)
            device = torch.device("cuda:{}".format(local_rank))
        else:
            device = torch.device("cpu:{}".format(local_rank))

        for i in range(len(new_dbn.models)):
            #new_dbn.models[i] = new_dbn.models[i].local_rank = local_rank
            #new_dbn.models[i] = new_dbn.models[i].torch_device = device
            #new_dbn.models[i] = new_dbn.models[i].to(device=device)
            if "LOCAL_RANK" in os.environ.keys():
                new_dbn.models[i].ddp_model = DDP(new_dbn.models[i], device_ids=[local_rank], output_device=local_rank) #, device_ids=device_ids)
            new_dbn.models[i]._optimizer = opt.SGD(new_dbn.models[i].parameters(), lr=learning_rate[i], momentum=momentum[i], weight_decay=decay[i], nesterov=nesterov_accel[i])
            new_dbn.models[i].normalize = normalize_learnergy[i]
            new_dbn.models[i].batch_normalize = batch_normalize[i]
 
        #Train model
        count = 0
        while(count == 0 or x2.has_next_subset()):
            mse, pl = new_dbn.fit(x2, batch_size=batch_size, epochs=epochs,
               is_distributed = True, num_loader_workers = num_loader_workers, pin_memory=~use_gpu_pre) #int(os.cpu_count() / 3))
            count = count + 1
            x2.next_subset()


        dist.barrier()
        if local_rank == 0:            
            for i in range(len(new_dbn.models)):	
                converge.plot(new_dbn.models[i]._history['mse'],
                    labels=['MSE'], title='convergence', subtitle='Model: Restricted Boltzmann Machine')
                plt.savefig(os.path.join(out_dir, "mse_plot_layer" + str(i) + ".png"))
                plt.clf()
                converge.plot( new_dbn.models[i]._history['pl'],
                    labels=['log-PL'], title='Log-PL', subtitle='Model: Restricted Boltzmann Machine')
                plt.savefig(os.path.join(out_dir, "log-pl_plot_layer" + str(i) + ".png"))
                plt.clf()
                converge.plot( new_dbn.models[i]._history['time'],
                    labels=['time (s)'], title='Training Time', subtitle='Model: Restricted Boltzmann Machine')
                plt.savefig(os.path.join(out_dir, "time_plot_layer" + str(i) + ".png"))
                plt.clf()

    else:
        print("Loading pre-existing model")
        new_dbn = torch.load(model_file)   

    for i in range(len(new_dbn.models)):
        converge.plot(new_dbn.models[i]._history['mse'], new_dbn.models[i]._history['pl'],
            new_dbn.models[i]._history['time'], labels=['MSE', 'log-PL', 'time (s)'],
            title='convergence over MNIST dataset', subtitle='Model: Restricted Boltzmann Machine')
        plt.savefig(os.path.join(out_dir, "convergence_plot_layer" + str(i) + ".png"))

    if not os.path.exists(model_file) or overwrite_model:
        #Save model
        torch.save(new_dbn, os.path.join(out_dir, model_fname))

    #TODO: For now set all subsetting to 1 - will remove subsetting later. 
    #Maintain output_subset_count - is/will be used by DataLoader in generate_output
    if local_rank == 0:
        #Generate test datasets
        x3 = None
        for t in range(0, len(data_test)):
            if t == 0:
                scaler = x2.scalers
                del x2
            else:
                scaler = x3.scalers
            x3 = DBNDataset([data_test[t]], read_func, data_reader_kwargs, pixel_padding, delete_chans=delete_chans, valid_min=valid_min, valid_max=valid_max, fill_value = fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, scalers=scaler, scale = scale_data, transform=numpy_to_torch, subset=subset_count)

            generate_output(x3, new_dbn, use_gpu, out_dir, "file" + str(t) + "_" +  testing_output, testing_mse, output_subset_count, ~use_gpu_pre)
    
        #Generate test datasets. Doing it this way, because generating all at once creates memory issues
        x2 = None
        for t in range(0, len(data_train)):
            if t == 0:
                scaler = x3.scalers
                del x3
            else:
                scaler = x2.scalers
            x2 = DBNDataset([data_train[t]], read_func, data_reader_kwargs, pixel_padding, delete_chans=delete_chans, valid_min=valid_min, valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, scalers = scaler, scale = scale_data, transform=numpy_to_torch, subset=subset_count)

            generate_output(x2, new_dbn, use_gpu, out_dir, "file" + str(t) + "_" +  training_output, training_mse, output_subset_count, ~use_gpu_pre)

    if "LOCAL_RANK" in os.environ.keys():
        cleanup_ddp() 
 
def generate_output(dat, mdl, use_gpu, out_dir, output_fle, mse_fle, output_subset_count, pin_mem = False):
    output_full = None
    rec_mse_full = []
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
    while(count == 0 or dat.has_next_subset() or dat.current_subset > (dat.subset-2)):
        test_loader = DataLoader(dat, batch_size=1000, shuffle=False,
                    num_workers = 0, drop_last = False, pin_memory = pin_mem) #int(os.cpu_count() / 3), pin_memory = True,
        #            drop_last=True)

        
        for data in test_loader:
            dat_dev, lab_dev = data[0].to(device=device, non_blocking=True), data[1].to(device=device, non_blocking=True)
            dev_ds = TensorDataset(dat_dev, lab_dev)

            output = mdl.forward(dat_dev)  
            if use_gpu == True:
                output = output.detach().cpu()

            loader = DataLoader(dev_ds, batch_size=len(dat_dev), shuffle=False, \
                num_workers = 0, drop_last = True, pin_memory = False)
            rec_mse, _ = mdl.reconstruct(dat_dev, loader)

            rec_mse = rec_mse.detach().cpu()

            rec_mse_full.append(torch.unsqueeze(rec_mse, 0)) 
            if output_full is None:
                output_full = torch.zeros(dat.data.shape[0], output.shape[1], dtype=torch.float32)
            output_full[1000*ind:1000*(ind+1),:] = output
            print("CONSTRUCTING OUTPUT", dat.data.shape, dat.data_full.shape, output.shape, output_full.shape, output.get_device(), rec_mse.get_device())
            ind = ind + 1
            del output
            #del rec_mse
            del dat_dev
            del lab_dev
            del dev_ds
            del loader
        count = count + 1
        if dat.has_next_subset():
            dat.next_subset()
        else:
            break 
    #Save training output
    torch.save(output_full, os.path.join(out_dir, output_fle), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(dat.targets_full, os.path.join(out_dir, output_fle + ".indices"), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(dat.data_full, os.path.join(out_dir, output_fle + ".input"), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    #torch.save(torch.cat(rec_mse_full, dim=0), os.path.join(out_dir, mse_fle), pickle_protocol=pickle.HIGHEST_PROTOCOL)





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


