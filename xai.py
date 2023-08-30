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
from glob import glob
import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use("Agg")
import  matplotlib.pyplot as plt

#Serialization
import pickle
from joblib import dump, load

#Data
import pandas
from utils import numpy_to_torch, read_yaml, get_read_func, get_scaler

#ML Imports
import torch
import torch.optim as opt 
from torch.nn.parallel import DistributedDataParallel as DDP
from rbm_models.clust_dbn import ClustDBN
from learnergy.models.deep import DBN
from dbn_learnergy import setup_ddp, cleanup_ddp, run_dbn
import shap

#Input Parsing
import argparse
import yaml


SEED = 42

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def save_shap(shap_values, filename, Class=0):
    # Save all SHAP values
    with open(filename, 'wb') as f:
        pickle.dump(shap_values, f)


def load_model(yml_conf, n_visible = None):
    # TODO: Fix bugs
    # FCDBN not supported
    
    out_dir = yml_conf["output"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    model_fname = yml_conf["output"]["model"]
    model_file = os.path.join(out_dir, model_fname)
    auto_clust = yml_conf["dbn"]["deep_cluster"]
    device_ids = yml_conf["dbn"]["training"]["device_ids"] 
    use_gpu = yml_conf["dbn"]["training"]["use_gpu"]
    model_type = yml_conf["dbn"]["params"]["model_type"]
    dbn_arch = tuple(yml_conf["dbn"]["params"]["dbn_arch"])
    temp = tuple(yml_conf["dbn"]["params"]["temp"])
    gibbs_steps = tuple(yml_conf["dbn"]["params"]["gibbs_steps"])
    learning_rate = tuple(yml_conf["dbn"]["params"]["learning_rate"])
    momentum = tuple(yml_conf["dbn"]["params"]["momentum"])
    decay = tuple(yml_conf["dbn"]["params"]["decay"])
    normalize_learnergy = tuple(yml_conf["dbn"]["params"]["normalize_learnergy"])
    batch_normalize = tuple(yml_conf["dbn"]["params"]["batch_normalize"])
    nesterov_accel = tuple(yml_conf["dbn"]["params"]["nesterov_accel"])

    local_rank = 0
    if "LOCAL_RANK" in os.environ.keys():
        setup_ddp(device_ids, use_gpu)
        local_rank = int(os.environ["LOCAL_RANK"])
    
    if n_visible is None:
        input_fp = glob(os.path.join(out_dir, "*.clustoutput.data.input"))[0]
        data_full = torch.load(input_fp)
        n_visible = data_full.shape[1]
    
    new_dbn = DBN(model=model_type, n_visible=n_visible, n_hidden=dbn_arch, steps=gibbs_steps, \
        learning_rate=learning_rate, momentum=momentum, decay=decay, temperature=temp, use_gpu=use_gpu)
    
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
    
    clust_scaler = None
    with open(os.path.join(out_dir, "fc_clust_scaler.pkl"), "rb") as f:
        clust_scaler = load(f)
    clust_dbn = ClustDBN(new_dbn, dbn_arch[-1], auto_clust, True, clust_scaler)
    clust_dbn.fc = DDP(clust_dbn.fc, device_ids=[local_rank], output_device=local_rank)
    model = clust_dbn
    model.dbn_trunk.load_state_dict(torch.load(model_file + ".ckpt"))
    model.fc.load_state_dict(torch.load(model_file + "_fc_clust.ckpt"))
    
    cleanup_ddp()
    
    return model


def wrap(x):
    # Reshapes from (channel, row, col) to (# pixels, channel) or (N_samples, N_features)
    x = np.array(x)
    orig_shape =  x.shape
    x = np.moveaxis(x, 0, 2)
    x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
    print(f"Wrapped {orig_shape} to {x.shape}.")
    return x

def unwrap(x, shape):
    # shape: desired shape (# channels, # rows, # cols)
    # Reshapes from (# pixels, channel) or (N_samples, N_features) to (channel, row, col)
    orig_shape = x.shape
    x = np.reshape(x, (shape[1], shape[2], shape[0]))
    x = np.moveaxis(x, 2, 0)
    print(f"Unwrapped {orig_shape} to {x.shape}.")
    return x
    
def background(x, fill=0.):
    if isinstance(x, np.ndarray) or isinstance(x, list):
        return (np.zeros_like(x, dtype=x.dtype) + fill)
    elif isinstance(x, tuple):
        return (np.zeros(x, dtype=np.float32) + fill)
    else:
        return None


     
# def get_masker(masker, shape, fill):
#     """
#         args:
#             masker: string id of desired masker
#             shape: int tuple dimensions (bands, rows, cols) of data 
#             fill: fill/no-data value for model  
#     """
    
#     img_dims = shape
#     print("Image masker shape: ", img_dims)
    
#     if shape[0] == 1:
#         partition_scheme = 0
#     else:
#         partition_scheme = 1
        
#     print("Partition scheme: ", partition_scheme)
    
#     if masker == 'masker_inpaint_telea':
#         return shap.maskers.Image("inpaint_telea", img_dims, partition_scheme=partition_scheme)
#     if masker == 'masker_inpaint_ns':
#         return shap.maskers.Image("inpaint_ns", img_dims, partition_scheme=partition_scheme)
#     # TODO: Add support for maskers with custom blur kernel
#     if masker == 'masker_blur_3x3':
#         return shap.maskers.Image("blur(3, 3)", img_dims, partition_scheme=partition_scheme)
#     if masker == 'masker_blur_10x10':
#         return shap.maskers.Image("blur(10, 10)", img_dims, partition_scheme=partition_scheme)
#     if masker == 'masker_blur_100x100':
#         return shap.maskers.Image("blur(100, 100)", img_dims, partition_scheme=partition_scheme)
#     if masker == 'masker_uniform_black':
#         return shap.maskers.Image(np.zeros(img_dims), img_dims, partition_scheme=partition_scheme)
#     if masker == 'masker_uniform_gray':
#         return shap.maskers.Image(np.zeros(img_dims) + 128, img_dims, partition_scheme=partition_scheme)
#     if masker == 'masker_uniform_white':
#         return shap.maskers.Image(np.zeros(img_dims) + 255, img_dims, partition_scheme=partition_scheme)
#     if masker == 'uniform_fill':
#         return shap.maskers.Image(np.zeros(img_dims) + fill, img_dims, partition_scheme=partition_scheme)


def explain(f, masker, data, max_evals, batch_size = None, class_names = None):
        
        # create an explainer with model and image masker
        explainer = shap.Explainer(f, masker, output_names=class_names)
        shap_values = explainer(data, max_evals=max_evals, batch_size=batch_size, outputs=shap.Explanation.argsort.flip[:4])
        # output with shap values
        shap.image_plot(shap_values, show=False)
        
        return shap_values
        

def main(**kwargs):

    if 'yml' in kwargs:
        yml_conf = read_yaml(kwargs['yml'])
    else:
        raise Exception('No yaml path specified.')
    if 'masker' in kwargs:
        masker = kwargs['masker']
    else: 
        masker = 'uniform_fill'
    if 'max_evals' in kwargs:
        max_evals = kwargs['max_evals']
    else:
        max_evals =  2 * 100**2 + 1
    if 'batch_size' in kwargs:
        batch_size = kwargs['batch_size']
    else:
        batch_size = 50
    
    
    out_dir = yml_conf['output']['out_dir']
    model = yml_conf['output']['model']
    
    fill_value = yml_conf['data']['fill_value']
    reader_type = yml_conf['data']['reader_type']
    reader_kwargs = yml_conf['data']['reader_kwargs']
    files_train = yml_conf['data']['files_train']
    files_test = yml_conf['data']['files_test']
    
    
    # TODO: improve class names
    clusters = yml_conf['dbn']['deep_cluster']
    class_names = []
    for i in range(clusters):
        class_names.append(str(i))
    
    d1 = torch.load('/work/09562/nleet/ls6/output2/caldor_08200_21049_026_210831_L090HHHH_CX_01.grd.clustoutput.data.input')
    print(".input shape", d1.shape)
    i = torch.load('/work/09562/nleet/ls6/output2/caldor_08200_21049_026_210831_L090HHHH_CX_01.grd.clustoutput.data.indices')
    print(".indices shape", i.shape)
    d2 = torch.load('/work/09562/nleet/ls6/output2/caldor_08200_21049_026_210831_L090HHHH_CX_01.grd.clustoutput.data').numpy()
    print(".data shape", d2.shape)
    
    # Initialize model
    model = load_model(yml_conf)
    model.cuda()
    print("Loaded model and scaler")
    
    # Load test data
    read_func = get_read_func(reader_type)
    
    data_train = None
    for file in files_train:
        if isinstance(file, list):
            fname = os.path.basename(file[0])
        else:
            fname = os.path.basename(file)
        data_train = read_func(file, **reader_kwargs)
    
    data_test = None
    for file in files_test:
        if isinstance(file, list):
            fname = os.path.basename(file[0])
        else:
            fname = os.path.basename(file)
        data_test = read_func(file, **reader_kwargs)
        
    # mask_func = get_masker(masker, dat.shape, fill_value)

    data_train = wrap(data_train)
    data_test = wrap(data_test)

    explainer = shap.KernelExplainer(model.forward_numpy, background(data_test.shape, fill_value), link='logit', output_names=class_names)
    shap_values = explainer.shap_values(data_test)
    save_shap(shap_values, os.path.join(out_dir, "shap_values.npz"))
    
    shap_values = np.load(os.path.join(out_dir, 'shap_values.npz'), allow_pickle=True)
    shap_values = np.array(shap_values)
    print(shap_values)
    print(shap_values.shape)
    print(len(shap_values))
    feature_names = ['HHHH', 'HVHVH', 'VVVV']
    shap_values = shap.Explanation(shap_values, feature_names=feature_names)
    shap_out_dir = os.path.join(out_dir, "shap plots")
    os.makedirs(os.path.join(out_dir, "shap plots"), exist_ok=True)
    for label in range(len(shap_values) + 1):
        print(f"saving plot {label}...")    
        p = shap.plots.beeswarm(shap_values[label], show=False, color_bar=True)
        plt.savefig(os.path.join(shap_out_dir, f"shap_image_plot_{label}.png"), bbox_inches='tight', dpi=100)
    
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: remove nargs='?' from -y switch
    parser.add_argument("-y", "--yaml", nargs='?', help="YAML file for cluster discretization.")
    parser.add_argument("-m", "--masker", nargs='?', help="Masker type.")
    parser.add_argument("-e", "--max-evals", nargs='?', help="Number of evaluations of underlying model.")
    parser.add_argument("-b", "--batch-size", nargs='?', help="Batch size for explanation.")
    args = parser.parse_args()    
    from timeit import default_timer as timer
    start = timer()
    # TODO: update params
    main(yml="config/dbn/uavsar_dbn.yaml", masker="masker_blur_3x3")
    end = timer()
    print(end - start) # Time in seconds, e.g. 5.38091952400282
