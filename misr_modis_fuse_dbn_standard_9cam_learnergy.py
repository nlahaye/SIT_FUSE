#General Imports
import os
import numpy as np
import random

import sys
sys.setrecursionlimit(4500)

#ML imports
import torch
from learnergy.models import dbn
 
#Data
from MISR_MODIS_FUSE_DATA_FIREX_9CAM_9 import data_fn3, data_fn3_test, NUMBER_CHANNELS, CHUNK_SIZE
from dbnDatasets import DBNDataset
from utils import numpy_to_torch, read_yaml, get_read_func

#Input Parsing
import yaml
import argparse

def run_dbn(yml_conf, data_train, data_test):

    #Get config values 
    pixel_padding = yml_conf["data"]["pixel_padding"]
    number_channel = yml_conf["data"]["number_channels"]
    data_reader =  yml_conf["data"]["reader_type"]
    fill = yml_conf["data"]["fill_value"]
    chan_dim = yml_conf["data"]["chan_dim"]
    valid_min = yml_conf["data"]["valid_min"]
    valid_max = yml_conf["data"]["valid_max"]
    delete_chans = yml_conf["data"]["delete_chans"]

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
    dbn_arch = yml_conf["dbn"]["params"]["dbn_arch"]
    gibbs_steps = yml_conf["dbn"]["params"]["gibbs_steps"]
    learning_rate = yml_conf["dbn"]["params"]["learning_rate"]
    momentum = yml_conf["dbn"]["params"]["momentum"]
    decay = yml_conf["dbn"]["params"]["decay"]
    temp = yml_conf["dbn"]["params"]["temp"]
    
    use_gpu = yml_conf["dbn"]["training"]["use_gpu"]
    batch_size = yml_conf["dbn"]["training"]["batch_size"]
    epochs = yml_conf["dbn"]["training"]["epochs"]

    read_func = get_read_func(data_reader)

    #Generate training dataset object
    #Unsupervised, so targets are not used. Currently, I use this to store original image indices for each point 
    x2 = DBNDataset(data_train, read_func, pixel_padding, delete_chans=delete_chans, valid_min=valid_min, valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, scalers = None, transform=numpy_to_torch)

    #Generate model
    chunk_size = 2*pixel_padding + 1
    new_dbn = dbn.DBN(model=model_type, n_visible=chunk_size*chunk_size*number_channel, n_hidden=dbn_arch, steps=gibbs_steps, learning_rate=learning_rate, momentum=momentum, decay=decay, temperature=temp, use_gpu=use_gpu)

    #Train model
    new_dbn.fit(x2, batch_size=batch_size, epochs=epochs)	

    #Generate training data output 
    if torch.cuda.is_available() and use_gpu:	
        output = new_dbn.forward(x2.data.cuda()).cpu()
        x2.transform = None
        rec_mse, v = new_dbn.reconstruct(x2)
    else:
        output = new_dbn.forward(x2.data)
        x2.transform = None
        rec_mse, v = new_dbn.reconstruct(x2) 
   
    #Save training output
    torch.save(output, os.path.join(out_dir, training_output))
    torch.save(rec_mse, os.path.join(out_dir, training_mse))

    #Generate test dataset object
    x3 = DBNDataset(data_test, read_func, chunk_size, delete_chans=delete_chans, valid_min=valid_min, valid_max=valid_max, fill_value = fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, scalers=x2.scalers, transform=numpy_to_torch)
 
    #Generate test data output
    if torch.cuda.is_available() and use_gpu:
        output = new_dbn.forward(x3.data.cuda()).cpu()
        x3.transform = None
        rec_mse, v = new_dbn.reconstruct(x3) 
    else:
        output = new_dbn.forward(x3.data)
        x3.transform = None
        rec_mse, v = new_dbn.reconstruct(x3) 
   
    #Save test data output
    torch.save(output, os.path.join(out_dir, testing_output))
    torch.save(rec_mse, os.path.join(out_dir, testing_mse))
 
    #Save model
    torch.save(new_dbn, os.path.join(out_dir, model_fname))
    

def main(yml_fpath, data_train, data_test):
   
    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    run_dbn(yml_conf, data_train, data_test)


if __name__ == '__main__':
	
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml, data_fn3, data_fn3_test)



