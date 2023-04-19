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

import sys
import resource
max_rec = 10**6
# May segfault without this line. 100 is a guess at the size of each stack frame.
resource.setrlimit(resource.RLIMIT_STACK, [100 * max_rec, resource.RLIM_INFINITY])
sys.setrecursionlimit(max_rec)

#ML imports
import torch
import torch.nn as nn
import torch.optim as opt 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
#from torch.nn import DataParallel as DDP
from learnergy.models.deep import DBN, ResidualDBN
from torch.utils.data.distributed import DistributedSampler

from rbm_models.fcn_dbn import FCDBN
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

def setup_ddp(device_ids, use_gpu=True):
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = port
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(device_ids)
    #os.environ['NCCL_SOCKET_IFNAME'] = "lo"

    driver = "nccl"
    if not use_gpu or not torch.cuda.is_available():
        driver = "gloo"

    # initialize the process group
    dist.init_process_group(driver)

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
    use_gpu_pre = yml_conf["dbn"]["training"]["use_gpu_preprocessing"]
    use_gpu = yml_conf["dbn"]["training"]["use_gpu"]
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
    fcn = False
    tile = False
    tile_size = None 
    tile_step = None
    nesterov_accel = None
    auto_clust = False
    use_gpu = False
    batch_size = 0
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

    if "FCDBN" in model_type[0]:
        tile = yml_conf["data"]["tile"]
        tile_size = yml_conf["data"]["tile_size"]
        tile_step = yml_conf["data"]["tile_step"]
        fcn = True
    else:
        temp = tuple(yml_conf["dbn"]["params"]["temp"])
    nesterov_accel = tuple(yml_conf["dbn"]["params"]["nesterov_accel"])
 
    auto_clust = yml_conf["dbn"]["deep_cluster"]
    use_gpu = yml_conf["dbn"]["training"]["use_gpu"]
    batch_size = yml_conf["dbn"]["training"]["batch_size"]
    epochs = yml_conf["dbn"]["training"]["epochs"]
 
 
    stratify_data = None
    if "stratify_data" in yml_conf["dbn"]["training"]:
        stratify_data = yml_conf["dbn"]["training"]["stratify_data"]

    scaler = None
    scaler_train = True 
    scaler_fname = os.path.join(out_dir, "dbn_scaler.pkl")
    if not os.path.exists(scaler_fname) or overwrite_model:    
        scaler_type = yml_conf["scaler"]["name"]
        scaler, scaler_train = get_scaler(scaler_type, cuda = use_gpu_pre)
    else:
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
    if not fcn:
        #TODO stratify conv data

 
        if stratify_data is not None:
            strat_read_func = get_read_func(stratify_data["stratify_data"]["reader"]) 
            stratify_data["stratify_data"]["reader"] = strat_read_func

        x2 = DBNDataset(data_train, read_func, data_reader_kwargs, pixel_padding, delete_chans=delete_chans, \
            valid_min=valid_min, valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, \
            transform_values=transform_values, scaler = scaler, train_scaler = scaler_train, scale = scale_data, \
            transform=numpy_to_torch, subset=subset_count, subset_training = subset_training, stratify_data=stratify_data)
    else:
        x2 = DBNDatasetConv(data_train, read_func, data_reader_kwargs, delete_chans=delete_chans, \
             valid_min=valid_min, valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, \
             transform_values=transform_values, transform=None, subset=subset_count, tile=tile, tile_size=tile_size, tile_step=tile_step,
             subset_training = subset_training)

    if x2.train_indices is not None:
        np.save(os.path.join(out_dir, "train_indices"), x2.train_indices)
 
    #Generate model
    if not fcn:
        chunk_size = 1
        for i in range(1,pixel_padding+1):
            chunk_size = chunk_size + (8*i)
    else:
        print(x2.data_full.shape)
        chunk_size = (x2.data_full.shape[2],x2.data_full.shape[3]) #TODO generalize
        #chunk_size = x2.data_full.shape[x2.chan_dim+1] 

    #TODO random sample in test, fix for generic case for commit
    #x2.data_full = x2.data_full[100,:,:,:]
    print(x2.data_full.shape)
 
    if not fcn:
        #ResidualDBN?
        new_dbn = DBN(model=model_type, n_visible=chunk_size*number_channel, n_hidden=dbn_arch, steps=gibbs_steps, \
            learning_rate=learning_rate, momentum=momentum, decay=decay, temperature=temp, use_gpu=use_gpu)
    else:
        new_dbn = FCDBN(model=model_type,visible_shape=chunk_size, n_channels=number_channel, steps=gibbs_steps, \
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
            if not isinstance(new_dbn.models[i], torch.nn.MaxPool2d):
                new_dbn.models[i] = DDP(new_dbn.models[i], device_ids=[local_rank], output_device=local_rank) #, device_ids=device_ids)
            else:
                new_dbn.models[i] = new_dbn.models[i]
        if not isinstance(new_dbn.models[i], torch.nn.MaxPool2d):
            new_dbn.models[i]._optimizer = opt.SGD(new_dbn.models[i].parameters(), lr=learning_rate[i], momentum=momentum[i], weight_decay=decay[i], nesterov=nesterov_accel[i])
            new_dbn.models[i].normalize = normalize_learnergy[i]
            new_dbn.models[i].batch_normalize = batch_normalize[i]

    if not os.path.exists(model_file + ".ckpt") or overwrite_model: 
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
                if not isinstance(new_dbn.models[i], torch.nn.MaxPool2d):
                    converge.plot(new_dbn.models[i].module._history['mse'],
                        labels=['MSE'], title='convergence', subtitle='Model: Restricted Boltzmann Machine')
                    plt.savefig(os.path.join(out_dir, "mse_plot_layer" + str(i) + ".png"))
                    plt.clf()
                    converge.plot( new_dbn.models[i].module._history['pl'],
                        labels=['log-PL'], title='Log-PL', subtitle='Model: Restricted Boltzmann Machine')
                    plt.savefig(os.path.join(out_dir, "log-pl_plot_layer" + str(i) + ".png"))
                    plt.clf()
                    converge.plot( new_dbn.models[i].module._history['time'],
                        labels=['time (s)'], title='Training Time', subtitle='Model: Restricted Boltzmann Machine')
                    plt.savefig(os.path.join(out_dir, "time_plot_layer" + str(i) + ".png"))
                    plt.clf()


    final_model = new_dbn
    if auto_clust == True:
        clust_dbn = ClustDBN(new_dbn, dbn_arch[-1], 500, True) #TODO parameterize
        clust_dbn.fc = DDP(clust_dbn.fc, device_ids=[local_rank], output_device=local_rank)
        final_model = clust_dbn
        if not os.path.exists(model_file + ".ckpt") or overwrite_model:
           dataset2 = TensorDataset(x2.data, x2.targets)
           loader = None
           is_distributed = True 
           if is_distributed:
               sampler = DistributedSampler(dataset2, shuffle=True)
               loader = DataLoader(dataset2, batch_size=batch_size, shuffle=False,
                    sampler=sampler, num_workers = num_loader_workers, pin_memory = ~use_gpu_pre,
                    drop_last=False)

           count = 0
           while(count == 0 or x2.has_next_subset()):
               clust_dbn.fit(dataset2, batch_size, epochs[-1], loader, sampler)
               count = count + 1
               x2.next_subset()

    if os.path.exists(model_file + ".ckpt") and not overwrite_model:
        print("Loading pre-existing model")
        if auto_clust == True:
            final_model.dbn_trunk.load_state_dict(torch.load(model_file + ".ckpt"))
            final_model.fc.load_state_dict(torch.load(model_file + "_fc_clust.ckpt"))
        else:
            final_model.load_state_dict(torch.load(model_file + ".ckpt"))
        #for m in range(len(new_dbn._models)):
        #    new_dbn._models[m].load_state_dict(model_file + "_sub_model_" + str(m) + ".ckpt") 


 
    if not os.path.exists(model_file + ".ckpt") or overwrite_model:
        for i in range(len(new_dbn.models)):
            if not isinstance(new_dbn.models[i], torch.nn.MaxPool2d):
                converge.plot(new_dbn.models[i].module._history['mse'], new_dbn.models[i].module._history['pl'],
                    new_dbn.models[i].module._history['time'], labels=['MSE', 'log-PL', 'time (s)'],
                    title='convergence over dataset', subtitle='Model: Restricted Boltzmann Machine')
                plt.savefig(os.path.join(out_dir, "convergence_plot_layer" + str(i) + ".png"))

    dist.barrier()
    if local_rank == 0:
        if not os.path.exists(model_file + ".ckpt") or overwrite_model:
            #Save model
            if auto_clust == True:
                torch.save(final_model.dbn_trunk.state_dict(), model_file + ".ckpt")
                torch.save(final_model.fc.state_dict(), model_file + "_fc_clust.ckpt")
            else:
                torch.save(final_model.state_dict(), model_file + ".ckpt")

            #Save scaler
            with open(os.path.join(out_dir, "dbn_scaler.pkl"), "wb") as f:
                dump(x2.scaler, f, True, pickle.HIGHEST_PROTOCOL)

        #TODO: For now set all subsetting to 1 - will remove subsetting later. 
        #Maintain output_subset_count - is/will be used by DataLoader in generate_output
        #Generate test datasets
        x3 = None
        fname_begin = "file"
        if auto_clust == True:
            fname_begin = fname_begin +"_clust"
        for t in range(0, len(data_test)):
            if t == 0:
                scaler = x2.scaler
                transform = x2.transform
                del x2
            else:
                scaler = x3.scaler
                transform = x3.transform
	
            if not fcn:
                x3 = DBNDataset([data_test[t]], read_func, data_reader_kwargs, pixel_padding, delete_chans=delete_chans, valid_min=valid_min, valid_max=valid_max, \
                    fill_value = fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, scaler=scaler, scale = scale_data, \
				transform=transform,  subset=subset_count)
            else:
                x3 = DBNDatasetConv([data_test[t]], read_func, data_reader_kwargs,  delete_chans=delete_chans, valid_min=valid_min, valid_max=valid_max, \
                fill_value = fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, transform = transform, \
                subset=subset_count, tile=tile, tile_size=tile_size, tile_step=tile_step)

            generate_output(x3, final_model, use_gpu, out_dir, fname_begin + str(t) + "_" +  testing_output, testing_mse, output_subset_count, ~use_gpu_pre, fcn)
    
	#Generate test datasets. Doing it this way, because generating all at once creates memory issues
        x2 = None
        for t in range(0, len(data_train)):
            if t == 0:
                scaler = x3.scaler
                transform = x3.transform
                del x3
            else:
                scaler = x2.scaler
                transform = x2.transform

            if not fcn:
                x2 = DBNDataset([data_train[t]], read_func, data_reader_kwargs, pixel_padding, delete_chans=delete_chans, valid_min=valid_min, \
                   valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, \
                   scaler = scaler, scale = scale_data, transform=numpy_to_torch, subset=subset_count)
            else:
                x2 = DBNDatasetConv([data_train[t]], read_func, data_reader_kwargs,  delete_chans=delete_chans, valid_min=valid_min, valid_max=valid_max, \
                    fill_value = fill, chan_dim = chan_dim, transform_chans=transform_chans, transform_values=transform_values, transform = transform, \
                    subset=subset_count, tile=tile, tile_size=tile_size, tile_step=tile_step)
 
            generate_output(x2, final_model, use_gpu, out_dir, fname_begin + str(t) + "_" +  training_output, training_mse, output_subset_count, ~use_gpu_pre, fcn)

    if "LOCAL_RANK" in os.environ.keys():
        cleanup_ddp() 
 
def generate_output(dat, mdl, use_gpu, out_dir, output_fle, mse_fle, output_subset_count, pin_mem = False, fcn = False):
    output_full = None
    #rec_mse_full = []
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
        gpu_usage()
        output_batch_size = min(100000, int(dat.data_full.shape[0] / 5))
        test_loader = DataLoader(dat, batch_size=output_batch_size, shuffle=False, \
        num_workers = 0, drop_last = False, pin_memory = pin_mem) #int(os.cpu_count() / 3), pin_memory = True,
        #            drop_last=False)

        gpu_usage()
        ind = 0
        ind2 = 0 
        for data in test_loader:
            dat_dev, lab_dev = data[0].to(device=device, non_blocking=True), data[1].to(device=device, non_blocking=True)
            dev_ds = TensorDataset(dat_dev, lab_dev)

            gpu_usage()
            output = mdl.forward(dat_dev)  
            if isinstance(output, list):
                output = output[0] #TODO improve usage uf multi-headed output after single-headed approach validated

            if use_gpu == True:
                output = output.detach().cpu()
            gpu_usage()
            loader = DataLoader(dev_ds, batch_size=output_batch_size, shuffle=False, \
                num_workers = 0, drop_last = False, pin_memory = False)
            #TODO rec_mse, visible_probs = mdl.reconstruct(dat_dev, loader)
            dat_dev = dat_dev.detach().cpu()
            lab_dev = lab_dev.detach().cpu()
            del dev_ds
            #visible_probs = visible_probs.detach().cpu()
            #del visible_probs
            #if use_gpu == True:
            #    rec_mse = rec_mse.detach().cpu() 

            #gpu_usage()
            #rec_mse_full.append(rec_mse) 
            if output_full is None:
                if not fcn:
                    output_full = torch.zeros(dat.data_full.shape[0], output.shape[1], dtype=torch.float32)
                else:
                    output_full = torch.zeros(dat.data_full.shape[0], dat.data_full.shape[1], output.shape[2], dtype=torch.float32)       
            ind1 = ind2 
            ind2 += dat_dev.shape[0]
            if ind2 > output_full.shape[0]:
                ind2 = output_full.shape[0]
            output_full[ind1:ind2,:] = output
            print("CONSTRUCTING OUTPUT", dat.data.shape, dat.data_full.shape, output.shape, output_full.shape, output.get_device(), ind1, ind2) #, rec_mse.get_device())
            ind = ind + 1
            del output
            #del rec_mse
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


