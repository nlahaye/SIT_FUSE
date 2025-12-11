from resource import *
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

import time

import joblib

import pickle

import sys
from learnergy.models.deep import DBN, ConvDBN

from segmentation.models.gcn import GCN
from segmentation.models.deeplabv3_plus_xception import DeepLab
from segmentation.models.unet import UNetEncoder, UNetDecoder

from sit_fuse.models.deep_cluster.dc import DeepCluster
from sit_fuse.models.encoders.cnn_encoder import DeepConvEncoder
from sit_fuse.models.encoders.pca_encoder import PCAEncoder
from sit_fuse.models.deep_cluster.ijepa_dc import IJEPA_DC
from sit_fuse.models.deep_cluster.dbn_dc import DBN_DC
from sit_fuse.models.deep_cluster.pca_dc import PCA_DC
from sit_fuse.datasets.dataset_utils import get_prediction_dataset
from sit_fuse.models.deep_cluster.heir_dc import Heir_DC
from sit_fuse.utils import read_yaml
from sit_fuse.models.deep_cluster.multi_prototypes import MultiPrototypes

from tqdm import tqdm

import argparse
import os
import numpy as np
import sys

import dask
import dask.array as da

import matplotlib
matplotlib.use('agg')

from osgeo import gdal, osr

from torchinfo import summary

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sit_fuse.viz.CMAP import CMAP, CMAP_COLORS



def run_inference(dat, mdl, use_gpu, out_dir, output_fle, pin_mem = True, tiled = False, return_embed = False):
    output_full = None
    embed_full = None
    count = 0

    ind = 0
    #output_batch_size = min(5000, max(int(dat.data_full.shape[0] / 5), dat.data_full.shape[0]))
    output_batch_size = 10000 #10000 #10 #0 #1
    if tiled:
        output_batch_size = 5 #50

    output_sze = dat.data_full.shape[0]
    append_remainder = int(output_batch_size - (output_sze % output_batch_size))


    if isinstance(dat.data_full,torch.Tensor):
        dat.data_full = torch.cat((dat.data_full,dat.data_full[0:append_remainder]))
        dat.targets_full = torch.cat((dat.targets_full,dat.targets_full[0:append_remainder]))
    else:
        dat.data_full = np.concatenate((dat.data_full,dat.data_full[0:append_remainder]))
        dat.targets_full = np.concatenate((dat.targets_full,dat.targets_full[0:append_remainder]))

    output = None
    output_prob = None
    #print(output_batch_size)
    test_loader = DataLoader(dat, batch_size=output_batch_size, shuffle=False, \
    num_workers = 0, drop_last = False, pin_memory = pin_mem)
    ind = 0
    ind2 = 0
    cntr = 0
    inference_times = []
    for data in tqdm(test_loader):
        embed = None
        cntr = cntr + 1
        if use_gpu:
            dat_dev, lab_dev = data[0].cuda(), data[1].cuda()
        else:
            dat_dev, lab_dev = data[0].cuda(), data[1].cuda()

        with torch.no_grad():
            start_time = time.time() 
            if hasattr(mdl, 'clust_tree'):

                print(dat.data_full.shape)
                input_shape = (1,dat.data_full.shape[1])
                if return_embed:
                    _, output, embed, _, output_prob = mdl.forward(dat_dev, return_embed=return_embed)
                else:
                    _, output, _, output_prob = mdl.forward(dat_dev, return_embed=return_embed)
            else:
                if return_embed:
                    output, embed  = mdl.forward(dat_dev, return_embed=return_embed)
                else:
                    output = mdl.forward(dat_dev)

                output_prob = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1).values 
                output = torch.argmax(torch.nn.functional.softmax(output, dim=1), dim=1)
                #print("HERE", output.shape, torch.unique(output))
            end_time = time.time()
            inference_times.append(end_time - start_time)

        if isinstance(output, list) or isinstance(output, tuple):
            output = output[0] #TODO improve usage uf multi-headed output after single-headed approach validated
            if output_prob is not None:
                output_prob = output_prob[0]
        #output = torch.unsqueeze(torch.argmax(output, axis = 1), axis=1)
 
        if use_gpu == True:
            output = output.detach().cpu()
            output_prob = output_prob.detach().cpu()

        dat_dev = dat_dev.detach().cpu()
        lab_dev = lab_dev.detach().cpu()


        #output = torch.squeeze(output)

        if output.ndim == 1:
            output = torch.unsqueeze(output, dim=1)
            output_prob = torch.unsqueeze(output_prob, dim=1)
        if output_full is None:
            if dat.data_full.ndim > 2:
                out_d1 = 1
                out_d2 = 2
                if output.ndim > 3:
                    out_d1 = 2
                    out_d2 = 3
                output_prob_full = torch.zeros((dat.data_full.shape[0], output.shape[out_d1], output.shape[out_d2]), dtype=torch.float32)
                output_full = torch.zeros((dat.data_full.shape[0], output.shape[out_d1], output.shape[out_d2]), dtype=torch.float32)
                #print("INIT DATA FULL", dat.init_data_shape, dat.init_data_shape[2:], output_full.shape)
            else:
                output_full = torch.zeros(dat.data_full.shape[0], output.shape[1], dtype=torch.float32)
                output_prob_full = torch.zeros(dat.data_full.shape[0], output.shape[1], dtype=torch.float32)
        ind1 = ind2
        if dat_dev.ndim > 2:
            ind2 += dat_dev.shape[0]
            if ind2 > output_full.shape[0]:
                ind2 = output_full.shape[0]
            #print("HERE", output.shape)
            output_full[ind1:ind2,:,:] = torch.squeeze(output)
            if output_prob is not None:
                output_prob_full[ind1:ind2,:,:] = torch.squeeze(output_prob)
        else:
            ind2 += dat_dev.shape[0]
            if ind2 > output_full.shape[0]:
                ind2 = output_full.shape[0]
            output_full[ind1:ind2,:] = output
            if output_prob is not None:
                output_prob_full[ind1:ind2,:] = output_prob

        if embed is not None:
  
            if embed.ndim == 1:
                embed = torch.unsqueeze(embed, dim=1)
            if embed_full is None:
                #print(dat.data_full.shape,  embed.shape)
                if dat.data_full.ndim > 2:
                    #print(dat.data_full.shape[0], embed.shape[1], embed.shape[2], embed.shape[3], embed.shape)
                    embed_full = torch.zeros((dat.data_full.shape[0], embed.shape[1], embed.shape[2], embed.shape[3]), dtype=torch.float32)
                    #pri    nt("INIT DATA FULL", dat.init_data_shape, dat.init_data_shape[2:], output_full.shape)
                else:
                    embed_full = torch.zeros(dat.data_full.shape[0], embed.shape[1], dtype=torch.float32)
            if dat_dev.ndim > 2:
                embed_full[ind1:ind2,:,:,:] = torch.squeeze(embed)
            else:
                embed_full[ind1:ind2,:] = embed



        #img = plt.imshow(output)
        #cmap = ListedColormap(CMAP_COLORS[0:int((500*200)- (-1) + 1)])
        #img.set_cmap(cmap)
        #plt.savefig(output_fle + "_tile" + str(count) + "_clusters.png", dpi=400, bbox_inches='tight') 

        ind = ind + 1
        del output
        del dat_dev
        del lab_dev
        del output_prob
        count = count + 1

    print("AVERAGE INFERENCE", np.mean(inference_times))
    return output_full, embed_full, output_prob_full



def generate_output(dat, mdl, use_gpu, out_dir, output_fle, pin_mem = True, tiled = False, return_embed = False):

    output_full, embed_full, output_prob = run_inference(dat, mdl, use_gpu, out_dir, output_fle, pin_mem, tiled, return_embed)

    print("SAVING", os.path.join(out_dir, output_fle), dat.targets_full.shape, output_full.shape)
    plot_clusters(dat.targets_full, output_full.numpy(), os.path.join(out_dir, output_fle), output_prob, pixel_padding=1) 


def plot_clusters(coord, output_data, output_basename, output_prob = None, pixel_padding=1):

        max_cluster = output_data.shape[1]
        min_cluster = 0
        labels = None
        #if output_data.shape[1] > 1:
        #    max_cluster = output_data.shape[1]
        #    labels = np.argmax(output_data, axis = 1)
        #else:
        labels = output_data.astype(np.int32)
        prob = output_prob
        max_cluster = labels.max() #TODO this better!!!

        #print(np.unique(labels).shape, "UNIQUE LABELS", np.unique(labels), coord.shape, output_data.shape)

        n_clusters_local = max_cluster - min_cluster

        data = []
        line_ind = 0
        samp_ind = 1
        if coord.shape[1] >  2:
            line_ind = 1
            samp_ind = 2
        max_dim1 = max(coord[:,line_ind])
        max_dim2 = max(coord[:,samp_ind])
        strt_dim1 = 0
        strt_dim2 = 0

        #print(labels.shape, max_dim1, max_dim2, labels.ndim)
        #1 subtracted to separate No Data from areas that have cluster value 0.
        if labels.ndim <= 2:
            data = np.zeros((((int)(max_dim1)+1+pixel_padding), ((int)(max_dim2)+pixel_padding+1))) - 1
            prob_data =  np.zeros((((int)(max_dim1)+1+pixel_padding), ((int)(max_dim2)+pixel_padding+1)))
            labels = np.array(labels)
            print("ASSIGNING LABELS", min_cluster, max_cluster)
            #print(data.shape, labels.shape, coord.shape)
            for i in range(labels.shape[0]):
                if prob is not None:
                    prob_data[coord[i,line_ind], coord[i,samp_ind]] = prob[i]
                data[coord[i,line_ind], coord[i,samp_ind]] = labels[i]
                #print(data.shape, coord[i,1], coord[i,2], labels[i], max_dim1, max_dim2)
    
        else:
            prob_data =  np.zeros((((int)(max_dim1)+1+pixel_padding), ((int)(max_dim2)+pixel_padding+1)))
            data = np.zeros((((int)(max_dim1)+1+pixel_padding), ((int)(max_dim2)+pixel_padding+1))) - 1
            labels = np.array(labels)
            #print("ASSIGNING LABELS", min_cluster, max_cluster)
            #print(data.shape, labels.shape, coord.shape)
            #print(data.shape, labels.shape, coord.shape)
            for i in range(labels.shape[0]):
                #print(coord[i,line_ind], coord[i,samp_ind], data.shape, labels.shape)
                #print(coord[i,line_ind], coord[i,samp_ind], coord[i], labels[i].shape, labels.shape)

                max_line = min(data.shape[0],   coord[i,line_ind]+labels[i].shape[0])
                max_samp = min(data.shape[1],   coord[i,samp_ind]+labels[i].shape[1])

                if prob is not None:
                    prob_data[coord[i,line_ind]:max_line, coord[i,samp_ind]:max_samp] = prob[i, :max_line - coord[i,line_ind], :max_samp - coord[i,samp_ind]]

                data[coord[i,line_ind]:max_line, coord[i,samp_ind]:max_samp] = labels[i, :max_line - coord[i,line_ind], :max_samp - coord[i,samp_ind]] 

        print("FINISHED WITH LABEL ASSIGNMENT")
        print("FINAL DATA TO DASK")
        data = data.astype(np.float32)
        #print(data)
        data = (data/1000.0).astype(np.float32)
        #print(data)
        data2_prob = da.from_array(prob_data)
        data2 = da.from_array(data)
        #del data

        #print(data.shape, data2.shape, "HERE TEST")
        da.to_zarr(data2,output_basename + "_" + str(n_clusters_local) + "clusters.zarr", overwrite=True)
        img = plt.imshow(data, vmin=-1, vmax=max_cluster)
        #print("HERE CLUSTERS MIN MAX MEAN STD", data.min(), data.max(), data.mean(), data.std(), data.shape)
        cmap = ListedColormap(CMAP_COLORS[0:int(max_cluster - (-1) + 1)])
        img.set_cmap(cmap)
        plt.colorbar()
        plt.savefig(output_basename + "_" + str(n_clusters_local) + "clusters.png", dpi=400, bbox_inches='tight')
        plt.clf()
        if prob is not None:
            da.to_zarr(data2_prob,output_basename + "_proba.zarr", overwrite=True)
            img = plt.imshow(data, vmin=0, vmax=100, cmap='plasma')
            plt.colorbar()
            plt.savefig(output_basename + "_proba.png", dpi=400, bbox_inches='tight')
            plt.clf()

            file_ext = ".no_geo.proba"
            fname = output_basename + file_ext + ".tif"
            out_ds = gdal.GetDriverByName("GTiff").Create(fname, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
            out_ds.GetRasterBand(1).WriteArray(prob_data)
            out_ds.FlushCache()
            out_ds = None

        file_ext = ".no_geo"
        fname = output_basename + "_" + str(n_clusters_local) + "clusters" + file_ext + ".tif"
        #print("HERE", data.min(), data.max(), data.mean(), data.std(), data.shape)
        out_ds = gdal.GetDriverByName("GTiff").Create(fname, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
        out_ds.GetRasterBand(1).WriteArray(data)
        out_ds.FlushCache()
        out_ds = None



def get_model(yml_conf, n_visible):

    use_wandb_logger = yml_conf["logger"]["use_wandb"]
    log_model = None
    save_dir = yml_conf["output"]["out_dir"]
    project = None
    if use_wandb_logger:
        log_model = yml_conf["logger"]["log_model"]
        save_dir = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
        project = yml_conf["logger"]["project"]
 
    heir_model_dir = os.path.join(save_dir, "full_model_heir")
    full_model_dir = os.path.join(save_dir, "full_model")

    accelerator = yml_conf["cluster"]["heir"]["training"]["accelerator"]
    devices = yml_conf["cluster"]["heir"]["training"]["devices"]
    precision = yml_conf["cluster"]["heir"]["training"]["precision"]
    max_epochs = yml_conf["cluster"]["heir"]["training"]["epochs"]
    gradient_clip_val = yml_conf["encoder"]["training"]["gradient_clip_val"]
    heir_model_tiers = yml_conf["cluster"]["heir"]["tiers"] #TODO incorporate
    heir_gauss_stdev = yml_conf["cluster"]["heir"]["gauss_noise_stdev"] #TODO incorporate
    lambda_iid = yml_conf["cluster"]["heir"]["lambda"] #TODO incorporate 

    num_classes = yml_conf["cluster"]["num_classes"]
    num_classes_heir = yml_conf["cluster"]["heir"]["num_classes"]
    min_samples = yml_conf["cluster"]["heir"]["training"]["min_samples"]

    heir_ckpt_path = os.path.join(heir_model_dir, "heir_fc.ckpt")
    ckpt_path = os.path.join(full_model_dir, "deep_cluster.ckpt")

    encoder_type=None
    if "encoder_type" in yml_conf:
        encoder_type=yml_conf["encoder_type"]

    encoder = None
    if encoder_type is not None and  "dbn" in encoder_type:
        model_type = tuple(yml_conf["dbn"]["model_type"])
        dbn_arch = tuple(yml_conf["dbn"]["dbn_arch"])
        gibbs_steps = tuple(yml_conf["dbn"]["gibbs_steps"])
        normalize_learnergy = tuple(yml_conf["dbn"]["normalize_learnergy"])
        batch_normalize = tuple(yml_conf["dbn"]["batch_normalize"])
        temp = tuple(yml_conf["dbn"]["temp"])

        learning_rate = tuple(yml_conf["encoder"]["training"]["learning_rate"])
        momentum = tuple(yml_conf["encoder"]["training"]["momentum"])
        decay = tuple(yml_conf["encoder"]["training"]["weight_decay"])
        nesterov_accel = tuple(yml_conf["encoder"]["training"]["nesterov_accel"])


        save_dir_dbn = yml_conf["output"]["out_dir"]
        use_wandb_logger = yml_conf["logger"]["use_wandb"]
        if use_wandb_logger:
            save_dir_dbn = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
        encoder_dir = os.path.join(save_dir_dbn, "encoder")

        enc_ckpt_path = os.path.join(encoder_dir, "dbn.ckpt")

        conv = ("conv" in encoder_type)
        if not conv:
            model_type = tuple(yml_conf["dbn"]["model_type"])
            encoder = DBN(model=model_type, n_visible=n_visible, n_hidden=dbn_arch, steps=gibbs_steps,
                 learning_rate=learning_rate, momentum=momentum, decay=decay, temperature=temp, use_gpu=True)
        else:
            model_type = yml_conf["dbn"]["model_type"]
            visible_shape = yml_conf["data"]["tile_size"]
            number_channel = yml_conf["data"]["number_channels"]
            #stride = yml_conf["dbn"]["stride"]
            #padding = yml_conf["dbn"]["padding"]
            encoder = ConvDBN(model=model_type, visible_shape=visible_shape, filter_shape = dbn_arch[1], n_filters = dbn_arch[0], \
                n_channels=number_channel, steps=gibbs_steps, learning_rate=learning_rate, momentum=momentum, \
                decay=decay, use_gpu=True, maxpooling=[False]*len(gibbs_steps)) #, maxpooling=mp)


        encoder.load_state_dict(torch.load(enc_ckpt_path))

        encoder.eval()

    elif encoder_type is not None and encoder_type == "byol":
        save_dir_byol = yml_conf["output"]["out_dir"]

        if use_wandb_logger:
            log_model = yml_conf["logger"]["log_model"]
            save_dir_byol = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
            project = yml_conf["logger"]["project"]

        encoder_dir = os.path.join(save_dir_byol, "encoder")
        in_chans = yml_conf["data"]["tile_size"][2]
        tile_size = yml_conf["data"]["tile_size"][0]
        encoder_ckpt_path = os.path.join(encoder_dir, "byol.ckpt")


        model_type = cutout_ratio_range = yml_conf["byol"]["model_type"]
        if model_type == "GCN":
            encoder = GCN(num_classes, in_chans, pretrained = False, use_deconv=True, use_resnet_gcn=True)
            encoder.load_state_dict(torch.load(encoder_ckpt_path))
        elif model_type == "DeepLab":
            encoder = DeepLab(num_classes, in_chans, backbone='resnet', pretrained=True, checkpoint_path=encoder_dir)
            encoder.load_state_dict(torch.load(encoder_ckpt_path)) 
        elif model_type == "Unet":
            m1 = UNetEncoder(num_classes, in_channels=in_chans)
            m1.load_state_dict(torch.load(encoder_ckpt_path))
            m2 = UNetDecoder(num_classes, in_channels=in_chans)
            encoder = torch.nn.Sequential(m1, m2)
        elif model_type == "DCE":
            m1 = DeepConvEncoder(in_chans)
            m1.load_state_dict(torch.load(encoder_ckpt_path))
            m1 = m1.eval()
            output_dim = in_chans*8*tile_size*tile_size
            m2 =  MultiPrototypes(output_dim, num_classes, 1)
            m2.eval()
            encoder = m1 #torch.nn.Sequential(m1, m2)
 

        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()

    elif encoder_type is not None and encoder_type == "pca":
        encoder = PCAEncoder()
 
        use_wandb_logger = yml_conf["logger"]["use_wandb"]
        if use_wandb_logger:
            encoder_dir = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
        encoder_dir = os.path.join(encoder_dir, "encoder")

        encoder.pca = joblib.load(os.path.join(encoder_dir, "pca.pkl"))


    generate_labels = False
    if not os.path.exists(heir_ckpt_path):
        heir_ckpt_path = None 
    model = Heir_DC(None, pretrained_model_path=ckpt_path, num_classes=num_classes_heir, yml_conf=yml_conf, \
        encoder_type=encoder_type, encoder=encoder, clust_tree_ckpt = heir_ckpt_path, generate_labels = generate_labels)
    #summary(model=encoder, input_size=(1, 34, 64, 64), col_names=['input_size', 'output_size', 'num_params', 'trainable'])
    model.eval()

    for key in model.clust_tree["1"].keys():
        if model.clust_tree["1"][key] is not None:
            model.clust_tree["1"][key].eval()
          
    return model



def predict_outside(yml_fpath):

    yml_conf = read_yaml(yml_fpath)

    predict(yml_conf)


def prep_model(yml_conf):
    test_fnames = yml_conf["data"]["files_test"]
    train_fnames = yml_conf["data"]["files_train"]

    data = None
    cntr = 0
    while data is None or data.data_full is None:
        print("HERE", data is None, cntr, len(train_fnames))
        data, _  = get_prediction_dataset(yml_conf, train_fnames[cntr])
        cntr = cntr + 1


    model = get_model(yml_conf, data.data_full.shape[1])

    if torch.cuda.is_available():
 
        model = model.cuda()

        model.pretrained_model = model.pretrained_model.cuda()
        if hasattr(model.pretrained_model, "pretrained_model"):
            #model.pretrained_model.mlp_head = model.pretrained_model.mlp_head.cuda()
            model.pretrained_model.pretrained_model = model.pretrained_model.pretrained_model.cuda()

        for lab1 in model.clust_tree.keys():
            if lab1 == "0":
                continue
            for lab2 in model.lab_full.keys():
                if lab2 in model.clust_tree[lab1].keys():
                    if model.clust_tree[lab1][lab2] is not None:
                        model.clust_tree[lab1][lab2] = model.clust_tree[lab1][lab2].cuda()

    return model

def predict(yml_conf):

    test_fnames = yml_conf["data"]["files_test"]
    train_fnames = yml_conf["data"]["files_train"]

    model = prep_model(yml_conf)
 
    out_dir = yml_conf["output"]["out_dir"]
    generate_intermediate_output = yml_conf["output"]["generate_intermediate_output"]
    generate_train_output =  yml_conf["output"]["generate_train_output"]

    tiled = yml_conf["data"]["tile"]

    #TODO make use_gpu configurable
    for i in range(len(test_fnames)):
        if isinstance(test_fnames[i], list):
            output_fle = os.path.basename(test_fnames[i][0])
        else:
            output_fle = os.path.basename(test_fnames[i])
        data, output_file  = get_prediction_dataset(yml_conf, test_fnames[i])
        if data.data_full is None:
            print("SKIPPING", test_fnames[i], " No valid samples")
            continue
        if model.clust_tree_ckpt is not None:
            generate_output(data, model, True, out_dir, output_fle + ".clust.data", tiled = tiled)
        if generate_intermediate_output:
            generate_output(data, model.pretrained_model, True, out_dir, output_fle + ".no_heir.clust.data", tiled = tiled)
 
    if generate_train_output:
        for i in range(len(train_fnames)):
            data, output_file = get_prediction_dataset(yml_conf, train_fnames[i])
            if data.data_full is None:
                print("SKIPPING", train_fnames[i], " No valid samples")
                continue
            if isinstance(train_fnames[i], list):
                output_fle = os.path.basename(train_fnames[i][0])
            else:
                output_fle = os.path.basename(train_fnames[i])
            if model.clust_tree_ckpt is not None:
                generate_output(data, model, True, out_dir, output_fle + ".clust.data", tiled = tiled)
            if generate_intermediate_output:
                generate_output(data, model.pretrained_model, True, out_dir, output_fle + ".no_heir.clust.data", tiled = tiled)

def gen_embeddings_from_arr(yml_conf, data_arr, init_shape, gen_image_shaped = True, strat_inds = None):

    model = prep_model(yml_conf)

    data, output_file  = get_prediction_dataset_from_scene_arr(yml_conf, data_arr, init_shape, strat_inds)

def gen_embeddings(yml_conf, fname, gen_image_shaped = True):

    model = prep_model(yml_conf)

    data, output_file  = get_prediction_dataset(yml_conf, fname)
    if data.data_full is None:
        print("SKIPPING", fname, " No valid samples")
        return None, None
    output = None
    embed = None
    recon_arr = None
    recon_lab = None
    if model.pretrained_model is not None:
        print("GENERATING EMBEDDING")
        output, embed, _ = run_inference(data, model, torch.cuda.is_available(), yml_conf["output"]["out_dir"], output_file + ".clust.data", \
                tiled = yml_conf["encoder"]["tiled"], return_embed =  True)

        if gen_image_shaped:
            recon_arr, recon_lab = conscruct_embed_image(embed, output, data.targets_full, init_shape = data.init_shape)
            return recon_arr, recon_lab

    return embed, output

def conscruct_embed_image(embed, labels, coord, init_shape = []):

    coord_coord_1 = 1
    coord_coord_2 = 2
    if coord.ndim < 3 and coord.shape[1] < 3:
        coord_coord_1 = 0
        coord_coord_2 = 1

    reconstructed_arr = None
    reconstructed_labels = None
    if len(init_shape) > 0:
        original_shape = init_shape[0]
    else:
        original_shape = (max(coord[:,coord_coord_1])+1, max(coord[:,coord_coord_2])+1)
    if embed.ndim  == 4:
        embed = np.transpose(embed, axes=(0,2,3,1))
        reconstructed_arr = np.zeros((original_shape[0], original_shape[1], embed.shape[3]), dtype=np.float32)
        reconstructed_labels = np.zeros((original_shape[0], original_shape[1]), dtype=np.float32) - 1
    else:
        reconstructed_arr = np.zeros((original_shape[0], original_shape[1], embed.shape[1]), dtype=np.float32)
        reconstructed_labels = np.zeros((original_shape[0], original_shape[1]), dtype=np.float32) - 1

    if embed.ndim  == 4:
        for i in range(embed.shape[0]):
            if coord[i,coord_coord_2]+embed.shape[2] > reconstructed_arr.shape[1] or \
                    coord[i,coord_coord_1]+embed.shape[1] > reconstructed_arr.shape[0]:
                continue
            reconstructed_arr[coord[i,coord_coord_1]:coord[i,coord_coord_1]+embed.shape[1], \
                    coord[i,coord_coord_2]:coord[i,coord_coord_2]+embed.shape[2], :] = embed[i,:,:,:]
            reconstructed_labels[coord[i,coord_coord_1]:coord[i,coord_coord_1]+labels.shape[1], \
                    coord[i,coord_coord_2]:coord[i,coord_coord_2]+labels.shape[2]] = labels[i,:,:]
        else:
            for i in range(embed.shape[0]):
                reconstructed_arr[coord[i,coord_coord_1], coord[i,coord_coord_2]] = embed[i]
                reconstructed_labels[coord[i,coord_coord_1], coord[i,coord_coord_2]] = labels[i]

    return reconstructed_arr, reconstructed_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    predict_outside(args.yaml)

    print(getrusage(RUSAGE_SELF))


