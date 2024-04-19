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

#Data
from utils import read_yaml, get_read_func
from dbn_datasets import DBNDataset

#ML Imports
import torch
import torch.optim as opt 
from torch.nn.parallel import DistributedDataParallel as DDP
from rbm_models.clust_dbn import ClustDBN
from rbm_models.heirarchichal_deep_clust import HeirClust
from learnergy.models.deep import DBN
from dbn_learnergy import setup_ddp, cleanup_ddp
import shap

#Input Parsing
import argparse

SEED = 42

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


# TODO: Image plotting features
# TODO: Waterfall plot for average of observations over a label
# TODO: Parallel process shap values


def load_model(yml_conf, n_visible = None, data_train = None):
    
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


    model_fname = yml_conf["output"]["model"]
    model_file = os.path.join(out_dir, model_fname)
    heir_model_file = os.path.join(out_dir, "heir_" + model_fname)
    heir_model_tiers = yml_conf["dbn"]["heir_tiers"]

    heir_min_samples = yml_conf["dbn"]["training"]["heir_cluster_min_samples"]
    heir_gauss_stdevs = yml_conf["dbn"]["training"]["heir_cluster_gauss_noise_stdev"]
    heir_epochs = yml_conf["dbn"]["training"]["heir_epochs"]
    n_heir_classes = yml_conf["dbn"]["training"]["heir_deep_cluster"]

    local_rank = 0
    device = "cpu"
    if "LOCAL_RANK" in os.environ.keys():
        setup_ddp(device_ids, use_gpu)
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda:{}".format(local_rank))   
 
    if n_visible is None:
        input_fp = glob(os.path.join(out_dir, "*.clustoutput.data.input"))[0]
        data_full = torch.load(input_fp)
        n_visible = data_full.shape[1]
    
    n_visible = data_train.data_full.shape[1]
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
    final_model = model  
 
    heir_clust = None

    for tiers in range(0,heir_model_tiers):

        print("HEIRARCHICAL TIER ", str(tiers + 1))

        heir_mdl_file = heir_model_file + ""
        if tiers > 0:
            heir_mdl_file = heir_model_file + "_" + str(tiers)

        print(heir_mdl_file)
        if os.path.exists(heir_mdl_file + ".ckpt"):
            heir_clust = HeirClust(final_model, data_train, n_heir_classes, use_gpu=use_gpu, min_samples=heir_min_samples, gauss_stdevs = heir_gauss_stdevs)
            heir_dict = torch.load(heir_mdl_file + ".ckpt")
            heir_clust.load_model(heir_dict)


        final_model = heir_clust





 
    return final_model



def read_train_test(yml_conf, read_from_input_file = False):
    
    # TODO: Fix reading from .input and .indices files.
    
    out_dir = yml_conf['output']['out_dir']
    n_channels = yml_conf['data']['number_channels']
    chan_dim = yml_conf['data']['chan_dim']
    train_fp = yml_conf['data']['files_train']
    test_fp = yml_conf['data']['files_test']
    data_train = None
    data_test = None

    while isinstance(train_fp, list):
        train_fp = train_fp[0] 
    while isinstance(test_fp, list):
        test_fp = test_fp[0] 
        
    train_fp = os.path.join(out_dir, os.path.basename(train_fp) + ".clustoutput.data.input")
    test_fp = os.path.join(out_dir, os.path.basename(test_fp) + ".clustoutput_test.data.input")
    train_idx_fp = '.'.join(train_fp.split('.')[0:-1]) + ".indices"
    test_idx_fp = '.'.join(test_fp.split('.')[0:-1]) + ".indices"
    
    # Try to load data from .input files (seems to be a bit faster)
    if read_from_input_file and os.path.exists(train_fp) and os.path.exists(test_fp):
        data_train = np.array(torch.load(train_fp))
        data_test = np.array(torch.load(test_fp))
        
        test_idx = np.array(torch.load(test_idx_fp))
        dims = dims_from_indices(test_idx, n_channels, chan_dim)
        print(data_train.shape)
        data_train = unwrap(data_train, dims)
        del test_idx
        
        # If train and test datasets appear to the have the same shape, we can reuse the dimensions
        if data_train.shape == data_test.shape:
            data_test = unwrap(data_test, dims)
        else:
            train_idx = np.array(torch.load(train_idx_fp))
            dims = dims_from_indices(train_idx, n_channels, chan_dim)
            data_train = unwrap(data_train, dims)
            del train_idx
    else:
        # Load data using reader
        reader_type = yml_conf['data']['reader_type']
        reader_kwargs = yml_conf['data']['reader_kwargs']
        files_train = yml_conf['data']['files_train']
        files_test = yml_conf['data']['files_test']
        data_reader_kwargs = yml_conf["data"]["reader_kwargs"]
        pixel_padding = yml_conf["data"]["pixel_padding"]
        number_channel = yml_conf["data"]["number_channels"]
        fill = yml_conf["data"]["fill_value"]
        chan_dim = yml_conf["data"]["chan_dim"]
        valid_min = yml_conf["data"]["valid_min"]
        valid_max = yml_conf["data"]["valid_max"]
        delete_chans = yml_conf["data"]["delete_chans"]
        subset_count = yml_conf["data"]["subset_count"]
        output_subset_count = yml_conf["data"]["output_subset_count"]
        scale_data = yml_conf["data"]["scale_data"]
        transform_chans = yml_conf["data"]["transform_default"]["chans"]
        transform_values =  yml_conf["data"]["transform_default"]["transform"]
        scale_data = yml_conf["data"]["scale_data"]
        stratify_data = None
        if "stratify_data" in yml_conf["dbn"]["training"]:
            stratify_data = yml_conf["dbn"]["training"]["stratify_data"]

        subset_training = yml_conf["dbn"]["subset_training"]
        read_func = get_read_func(reader_type)

        scaler = None
        scaler_train = False
        scaler_fname = os.path.join(out_dir, "dbn_scaler.pkl")
 
        if not os.path.exists(scaler_fname):
            scaler_type = yml_conf["scaler"]["name"]
            scaler, scaler_train = get_scaler(scaler_type, cuda = use_gpu_pre)
        else:
            scaler = load(scaler_fname)

 
        data_train = DBNDataset()
        data_train.read_and_preprocess_data(files_train, read_func, data_reader_kwargs, pixel_padding, delete_chans=delete_chans, \
                valid_min=valid_min, valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, \
                transform_values=transform_values, scaler = scaler, train_scaler = scaler_train, scale = scale_data, \
                transform=None, subset=subset_count, subset_training = subset_training, stratify_data=stratify_data)


        data_test = DBNDataset()
        data_test.read_and_preprocess_data(files_test, read_func, data_reader_kwargs, pixel_padding, delete_chans=delete_chans, \
                valid_min=valid_min, valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, \
                transform_values=transform_values, scaler = scaler, train_scaler = scaler_train, scale = scale_data, \
                transform=None, subset=subset_count, subset_training = subset_training, stratify_data=stratify_data)
    
    return data_train, data_test



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
    model = yml_conf['output']['model']
    clusters = yml_conf['dbn']['deep_cluster']
    fill_value = yml_conf['data']['fill_value']
    n_channels = yml_conf['data']['number_channels']
    chan_dim = yml_conf['data']['chan_dim']
    
    # Load train and test data
    data_train, data_test = read_train_test(yml_conf)

    # Load model and scaler
    model = load_model(yml_conf, n_channels, data_train).cuda()
    scaler = load(os.path.join(out_dir, "dbn_scaler.pkl"))
    
    # Load train and test data
    data_train, data_test = read_train_test(yml_conf)
    
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
