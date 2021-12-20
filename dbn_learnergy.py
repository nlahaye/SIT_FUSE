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
#from torch.nn import DataParallel as DDP
from learnergy.models.deep import DBN
 
#Data
from dbnDatasets import DBNDataset
from utils import numpy_to_torch, read_yaml, get_read_func

#Input Parsing
import yaml
import argparse
from datetime import timedelta

import learnergy.visual.convergence as converge

def setup_ddp(rank, world_size, use_gpu=True):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
    os.environ['NCCL_SOCKET_IFNAME'] = "lo"

    driver = "nccl"
    if not use_gpu or not torch.cuda.is_available():
        driver = "gloo"

    # initialize the process group
    dist.init_process_group(driver, rank=rank, world_size=world_size, init_method="env://", timeout=timedelta(seconds=10))

def cleanup_ddp():
    dist.destroy_process_group()


def run_dbn(yml_conf):

    #Get config values 
    data_test = yml_conf["data"]["files_test"]
    data_train = yml_conf["data"]["files_train"]  	

    pixel_padding = yml_conf["data"]["pixel_padding"]
    number_channel = yml_conf["data"]["number_channels"]
    data_reader =  yml_conf["data"]["reader_type"]
    fill = yml_conf["data"]["fill_value"]
    chan_dim = yml_conf["data"]["chan_dim"]
    valid_min = yml_conf["data"]["valid_min"]
    valid_max = yml_conf["data"]["valid_max"]
    delete_chans = yml_conf["data"]["delete_chans"]
    subset_count = yml_conf["data"]["subset_count"]

    transform_chans = yml_conf["data"]["transform_default"]["chans"]
    transform_values = 	yml_conf["data"]["transform_default"]["transform"]

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
    temp = tuple(yml_conf["dbn"]["params"]["temp"])
    
    use_gpu = yml_conf["dbn"]["training"]["use_gpu"]
    world_size = yml_conf["dbn"]["training"]["world_size"]
    rank = yml_conf["dbn"]["training"]["rank"]
    device_ids = yml_conf["dbn"]["training"]["device_ids"]
    batch_size = yml_conf["dbn"]["training"]["batch_size"]
    epochs = yml_conf["dbn"]["training"]["epochs"]

    overwrite_model = yml_conf["dbn"]["overwrite_model"]
 
    setup_ddp(rank, world_size, use_gpu)

    read_func = get_read_func(data_reader)

    #Generate model
    chunk_size = 1
    for i in range(1,pixel_padding+1):
        chunk_size = chunk_size + (8*i)

    #Generate training dataset object
    #Unsupervised, so targets are not used. Currently, I use this to store original image indices for each point 
    x2 = DBNDataset(data_train, read_func, pixel_padding, delete_chans=delete_chans, valid_min=valid_min, valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, scalers = None, scale = True, transform=numpy_to_torch, subset=subset_count)
 
    model_file = os.path.join(out_dir, model_fname)
    if(not os.path.exists(model_file) or overwrite_model):
        new_dbn = DBN(model=model_type, n_visible=chunk_size*number_channel, n_hidden=dbn_arch, steps=gibbs_steps, learning_rate=learning_rate, momentum=momentum, decay=decay, temperature=temp, use_gpu=use_gpu)

        for i in range(len(new_dbn.models)):
            new_dbn.models[i].ddp_model = DDP(new_dbn.models[i], device_ids=device_ids).cuda()
            new_dbn.models[i].optimizer_ = opt.SGD(new_dbn.models[i].ddp_model.parameters(),
                                                 lr=learning_rate[i], momentum=momentum[i], weight_decay=decay[i])

        #Train model
        count = 0
        while(count == 0 or x2.has_next_subset()):
            new_dbn.fit(x2, batch_size=batch_size, epochs=epochs)
            count = count + 1
            x2.next_subset()            	
        #reset to first subset for output generation
        x2.next_subset()

    else:
        print("Loading pre-existing model")
        new_dbn = torch.load(model_file)   

    generate_output(x2, new_dbn, use_gpu, out_dir, training_output, training_mse)

    #Generate test dataset object
    x3 = DBNDataset(data_test, read_func, pixel_padding, delete_chans=delete_chans, valid_min=valid_min, valid_max=valid_max, fill_value = fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, scalers=x2.scalers, scale = True, transform=numpy_to_torch)

    generate_output(x3, new_dbn, use_gpu, out_dir, testing_output, testing_mse)

    converge.plot(new_dbn.history['mse'], new_dbn.history['pl'], 
        new_dbn.history['time'], labels=['MSE', 'log-PL', 'time (s)'],
        title='convergence over MNIST dataset', subtitle='Model: Restricted Boltzmann Machine')

    if not os.path.exists(model_file) or overwrite_model:
        #Save model
        torch.save(new_dbn, os.path.join(out_dir, model_fname))
    
    cleanup_ddp()

def generate_output(dat, mdl, use_gpu, out_dir, output_fle, mse_fle):
 
    output_full = None
    count = 0
    while(count == 0 or dat.has_next_subset()):
        if torch.cuda.is_available() and use_gpu:
            output = mdl.models[0].ddp_model(dat.data.cuda()).cpu()
            for i in range(1, len(mdl.models)):
                output = mdl.models[i].ddp_model(output)
            output = output.cpu()
            dat.transform = None
            rec_mse, v = mdl.reconstruct(dat)
        else:
            output = mdl.forward(dat.data)
            dat.transform = None
            rec_mse, v = mdl.reconstruct(dat)
        
        if output_full is None:
            output_full = output
        else:
            np.concatenate(output_full,output)
        del output
        count = count + 1
        dat.next_subset()

    #Save training output
    torch.save(output_full, os.path.join(out_dir, output_fle))
    torch.save(dat.targets_full, os.path.join(out_dir, output_fle + ".indices"))
    torch.save(dat.data_full, os.path.join(out_dir, output_fle + ".input"))
    torch.save(rec_mse, os.path.join(out_dir, mse_fle))


def main(yml_fpath):
   
    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    run_dbn(yml_conf)


if __name__ == '__main__':
	
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)



