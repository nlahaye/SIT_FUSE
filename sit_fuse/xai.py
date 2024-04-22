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
import matplotlib 
matplotlib.use("Agg")
import  matplotlib.pyplot as plt

#Serialization
import pickle
from joblib import load


#ML Imports
import torch
import torch.optim as opt 

from sit_fuse.models.deep_cluster.dc import DeepCluster
from sit_fuse.models.deep_cluster.ijepa_dc import IJEPA_DC
from sit_fuse.models.deep_cluster.dbn_dc import DBN_DC
from sit_fuse.datasets.dataset_utils import get_prediction_dataset

from torch.nn.parallel import DistributedDataParallel as DDP
import shap

#Input Parsing
import argparse

SEED = 42

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


# TODO: Image plotting features
# TODO: Waterfall plot for average of observations over a label
# TODO: Parallel process shap values


def load_model(yml_conf, n_visible = None):
    
    num_loader_workers = int(yml_conf["data"]["num_loader_workers"])
    val_percent = int(yml_conf["data"]["val_percent"])
    batch_size = yml_conf["cluster"]["training"]["batch_size"]
    use_gpu = yml_conf["encoder"]["training"]["use_gpu"]
    out_dir = yml_conf["output"]["out_dir"]

    save_dir = yml_conf["output"]["out_dir"]
    use_wandb_logger = yml_conf["logger"]["use_wandb"]
    if use_wandb_logger:
        save_dir = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
    ckpt_path = os.path.join(os.path.join(save_dir, "full_model"), "checkpoint.ckpt")
    ckpt_path_heir = os.path.join(os.path.join(save_dir, "full_model_heir"), "checkpoint.ckpt")
   

    model = None
    
    if os.path.exists(ckpt_path_heir):
         model = Heir_DC.load_from_checkpoint(ckpt_path_heir)
    else:
        if "encoder_type" in yml_conf:
            if yml_conf["encoder_type"] == "dbn":
                model = DBN_DC.load_from_checkpoint(ckpt_path)
            elif yml_conf["encoder_type"] == "ijepa":
                model = IJEPA_DC.load_from_checkpoint(ckpt_path)
        else:
             model = DeepCluster.load_from_checkpoint(ckpt_path)


    return final_model



def wrap(x):
    """
    Reshapes from (channel, row, col) to (# pixels, channel) or (N_samples, N_features)
    """
    x = np.array(x)
    orig_shape =  x.shape
    x = np.moveaxis(x, 0, 2)
    x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
    print(f"Wrapped {orig_shape} to {x.shape}.")
    return x


def unwrap(x, shape):
    """
    Reshapes from (N_samples, N_features) to (N_channels, N_rows, N_cols)
    Args:
        x: dataset (N_samples, N_features)
        shape: desired output shape (N_channels, N_rows, N_cols)
    Returns:
        Reshaped array.
    """
    # TODO: Support chan_dim != 0 
    orig_shape = x.shape
    print(orig_shape)
    x = np.reshape(x, (shape[1], shape[2], shape[0]))
    x = np.moveaxis(x, 2, 0)
    print(f"Unwrapped {orig_shape} to {x.shape}.")
    return x


def filled_array(x, fill=0.):
    if isinstance(x, np.ndarray) or isinstance(x, list):
        return (np.zeros_like(x, dtype=x.data_full.dtype) + fill)
    elif isinstance(x, tuple):
        return (np.zeros(x, dtype=np.float32) + fill)
    else:
        return None


def forward_numpy(x, model):
        """Performs a forward pass over the data.

        Args:
            x: An input np.ndarray that will be converted to a tensor for computing the forward pass.

        Returns:
            (np.ndarray): An array containing the DBN's outputs.
        """
        numpy_to_torch_dtype_dict = {
            np.dtype(np.bool)        : torch.bool,
            np.dtype(np.uint8)       : torch.uint8,
            np.dtype(np.int8)        : torch.int8,
            np.dtype(np.int16)       : torch.int16,
            np.dtype(np.int32)       : torch.int32,
            np.dtype(np.int64)       : torch.int64,
            np.dtype(np.float16)     : torch.float16,
            np.dtype(np.float32)     : torch.float32,
            np.dtype(np.float64)     : torch.float64,
            np.dtype(np.complex64)   : torch.complex64,
            np.dtype(np.complex128)  : torch.complex128
        }

        dt = numpy_to_torch_dtype_dict[x.dtype]
        t = torch.from_numpy(x).cuda()
        y = model.forward(t)
        y = y.detach().cpu().numpy()

        return y



def most_frequent_labels(data):
    """
    Args: 
        data: (# samples, # labels)
    Returns: 
        Array of labels (# labels) sorted by frequency of occurence
    """
   

    if data.shape[1] > 1:
        labels = np.argmax(data, axis = 1)
    else:
        labels = data.astype(np.int32)
    unique_labels, frequency = np.unique(labels, return_counts=True)
    sorted_indexes = np.argsort(frequency)[::-1]
    sorted_by_freq = unique_labels[sorted_indexes]

    return sorted_by_freq
        


def dims_from_indices(indices, n_channels, chan_dim = 0):
    t = np.moveaxis(np.transpose(indices), chan_dim, 0)
    dims = (n_channels, np.max(t[1]) + 1, np.max(t[2]) + 1)
    np.moveaxis(dims, 0, chan_dim)
    return dims



def get_background(background_type, data: np.ndarray, n_samples=None):
    if background_type == "kmeans":
        background = shap.kmeans(data, n_samples)
    elif background_type == "sample":
        background = shap.sample(data, n_samples)
    elif background_type == "zero":
        background = np.zeros((1,data.shape[1]), dtype=data.dtype)
    else:
        background = data
    return background



def save_summary_plot(out_dir, shap_values, label, feature_names): 
    filename = f"shap_summary_plot_{label}.png"  
    print(f"Saving {os.path.join(out_dir, filename)}...")
    p = shap.summary_plot(shap_values[:,:,label], feature_names=feature_names, show=False, color_bar=True)
    plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight', pad_inches=0.2, dpi=100)



def save_shap(shap_values, filename):
    # Save all SHAP values
    with open(filename, 'wb') as f:
        pickle.dump(shap_values, f)



def explain(f, dataset: np.ndarray, background: np.ndarray, link, output_names, out_dir, explanation_fname="explanation.pkl"):
        
    print("Calculating shap values...")
    if link not in ['identity', 'logit']:
        link = 'identity'
    explainer = shap.KernelExplainer(f, background, link=link, output_names=output_names)
    print(dataset.shape)
    explanation = explainer(dataset)
    save_shap(explanation, os.path.join(out_dir, explanation_fname))
        
    return explanation



def main(**kwargs):
    
    # TODO: Finish argparse params
    if 'yaml' in kwargs:
        yml_conf = read_yaml(kwargs['yaml'])
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
    clusters = yml_conf['cluster']['num_classes']
    fill_value = yml_conf['data']['fill_value']
    n_channels = yml_conf['data']['number_channels']
    chan_dim = yml_conf['data']['chan_dim']
    
    #loop over multiple files + subsample
    # Load train and test data
    data = get_prediction_dataset((yml_conf, yml_conf["data"]["files_test"][0])

    # Load model and scaler
    model = load_model(yml_conf, n_channels).cuda()
    
    # Subset? (yes: (0:200, 0:600))
    #row_min, row_max = 0, 200
    #col_min, col_max = 0, 600
    #new_dims = np.moveaxis(np.array([n_channels, row_max, col_max]), 0, chan_dim)
    
    #if row_min and col_min and row_max and col_max:
    #    data_train = data_train[:,row_min:row_max,col_min:col_max]
    #    data_test = data_test[:,row_min:row_max,col_min:col_max]
    
    #print("Data: ", data_test.shape)
        
    # Preprocess data 
    #data_train = scaler.transform(wrap(data_train))
    #data_test = scaler.transform(wrap(data_test))
    #print("Model input: ", data_test.shape)
    
    # Set explain params
    overwrite = True
    link = 'identity' # 'logit'
    background_type = 'zero'
    background = get_background(background_type, data_train.data_full)
    
    # ============================== EXPLANATION ===============================
    
    labels = [x for x in range(clusters)]
    label_names = map(str, labels)
   
    f = lambda x: (model.forward(torch.from_numpy(x).to(model.dbn_trunk.models[0].module.torch_device))).detach().cpu().numpy() 
    if not overwrite and os.path.exists(os.path.join(out_dir, "explanation.pkl")) and os.path.exists(os.path.join(out_dir, "shap_values.npz")):
        
        explanation = np.load(os.path.join(out_dir, "explanation.pkl"), allow_pickle=True)
        shap_values = np.array(np.load(os.path.join(out_dir, "shap_values.npz"), allow_pickle=True))   
    
    else:
        print("HERE", data_test.data_full.shape, data_test.data_full[0:1000].shape)
        # Compute with zero background
        explanation = explain(f, data_test.data_full[0:1000], background, link=link, 
                            output_names=label_names, out_dir=out_dir, 
                            explanation_fname="explanation_zero_background.pkl")
        shap_values = explanation.values
        save_shap(shap_values, os.path.join(out_dir, "shap_values_zero_background.npz"))
    
        # Also compute with kmeans background 
        #background = get_background('kmeans', data_train.data_full[0:1000], n_samples=50)
        explanation = explain(f, data_train.data_full[0:1000], background, link=link, 
                            output_names=label_names, out_dir=out_dir, 
                            explanation_fname="explanation_kmeans_background.pkl")
        shap_values = explanation.values
        save_shap(shap_values, os.path.join(out_dir, "shap_values_kmeans_background.npz"))

    
    print("Shap values shape: ", np.array(shap_values).shape)
    print("Shap explanation shape: ", np.array(explanation).shape)  
    
    # ================================ PLOTTING ================================
    
    shap_out_dir = os.path.join(out_dir, "shap plots")
    os.makedirs(shap_out_dir, exist_ok=True)
    print(shap_out_dir)
    
    feature_names = ["chan_" + str(i) for i in range(data_test.data_full.shape[1])]
    
    plot_by_freq = True
    labels_by_freq = most_frequent_labels(f(data_test.data_full[0:1000]))
    print("Most significant labels: ", labels_by_freq)
    
    if plot_by_freq:
        for label in labels_by_freq:
            save_summary_plot(out_dir, shap_values, label, feature_names)
    else:
        for label in labels:
            save_summary_plot(out_dir, shap_values, label, feature_names)
            


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
    main(yaml=args.yaml)
    end = timer()
    print(end - start) # Time in seconds, e.g. 5.38091952400282

    cleanup_ddp()


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
