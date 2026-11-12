from resource import *

# Cap native thread pools (OpenMP/BLAS/MKL) *before* numpy/torch are imported
# when run as a script. Library callers get the same limits via prep_model().
if __name__ == '__main__':
    from sit_fuse.compute_resources import limit_env_from_argv
    limit_env_from_argv()

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
from sit_fuse.datasets.dataset_utils import get_prediction_dataset, get_prediction_dataset_from_scene_arr
from sit_fuse.datasets.sf_dataset import SFDataset
from sit_fuse.datasets.sf_dataset_conv import SFDatasetConv
from sit_fuse.inference.geo_output import geo_output_enabled, resolve_reference, write_geotiff
from sit_fuse.models.deep_cluster.heir_dc import Heir_DC
from sit_fuse.utils import read_yaml
from sit_fuse.compute_resources import apply_compute_limits, resolve_use_gpu, get_compute_conf
from sit_fuse.models.deep_cluster.multi_prototypes import MultiPrototypes

from tqdm import tqdm

import argparse
import csv
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



def _direct_slicing(dat):
    """True when dat[i:j] is just (data_full[i:j], targets_full[i:j]) - the SFDataset/SFDatasetConv case."""
    return type(dat).__getitem__ in (SFDataset.__getitem__, SFDatasetConv.__getitem__)


def _inference_batches(dat, output_batch_size, use_gpu, pin_mem):
    """
    Yields input batches in order. The last partial batch is padded with rows from the
    start of the dataset to a full batch (as the old in-place padding of dat.data_full did),
    so every forward pass sees the same batch shape; the extra rows are dropped afterwards.
    dat.data_full / dat.targets_full are no longer modified.
    """
    n = dat.data_full.shape[0]
    if _direct_slicing(dat):
        full = dat.data_full
        for start in range(0, n, output_batch_size):
            batch = full[start:start + output_batch_size]
            n_real = batch.shape[0]
            if n_real < output_batch_size:
                pad = full[0:output_batch_size - n_real]
                batch = torch.cat((torch.as_tensor(batch), torch.as_tensor(pad))) if torch.is_tensor(full) \
                    else np.concatenate((batch, pad))
            batch = torch.as_tensor(batch)
            if use_gpu:
                if pin_mem:
                    batch = batch.pin_memory()
                batch = batch.cuda(non_blocking=pin_mem)
            yield batch, n_real
    else:
        n_pad = (output_batch_size - (n % output_batch_size)) % output_batch_size
        idx = np.concatenate((np.arange(n), np.arange(min(n_pad, n))))
        loader = DataLoader(torch.utils.data.Subset(dat, idx.tolist()), batch_size=output_batch_size, shuffle=False, \
            num_workers = 0, drop_last = False, pin_memory = pin_mem)
        seen = 0
        for data in loader:
            batch = data[0].cuda(non_blocking=pin_mem) if use_gpu else data[0]
            n_real = max(0, min(batch.shape[0], n - seen))
            seen += batch.shape[0]
            yield batch, n_real


def run_inference(dat, mdl, use_gpu, out_dir, output_fle, pin_mem = True, tiled = False, return_embed = False, batch_size = None):
    output_full = None
    embed_full = None
    output_prob_full = None

    #output_batch_size = min(5000, max(int(dat.data_full.shape[0] / 5), dat.data_full.shape[0]))
    output_batch_size = 10000 #10000 #10 #0 #1
    if tiled:
        output_batch_size = 5 #50
    if batch_size is not None:
        output_batch_size = int(batch_size)
    pin_mem = pin_mem and use_gpu  # pinned host memory only helps GPU transfers

    n_out = dat.data_full.shape[0]
    data_ndim = dat.data_full.ndim
    n_batches = (n_out + output_batch_size - 1) // output_batch_size

    ind2 = 0
    inference_times = []
    for dat_dev, n_real in tqdm(_inference_batches(dat, output_batch_size, use_gpu, pin_mem), total=n_batches):
        embed = None
        output_prob = None

        with torch.no_grad():
            start_time = time.time()
            if hasattr(mdl, 'clust_tree'):
                result = mdl.forward(dat_dev, return_embed=return_embed)
                if result is None:
                    raise RuntimeError("Hierarchical model returned no output for a batch in %s" % output_fle)
                if return_embed:
                    _, output, embed, _, output_prob = result
                else:
                    _, output, _, output_prob = result
            else:
                if return_embed:
                    output, embed  = mdl.forward(dat_dev, return_embed=return_embed)
                else:
                    output = mdl.forward(dat_dev)
                if isinstance(output, list) or isinstance(output, tuple):
                    output = output[0]
                # Cluster heads end in softmax: keep the max probability alongside the label
                # (previously computed and discarded, and the None prob crashed below).
                output_prob = torch.max(output, dim=1).values
                output = torch.argmax(output, dim=1)
            end_time = time.time()
            inference_times.append(end_time - start_time)

        if isinstance(output, list) or isinstance(output, tuple):
            output = output[0] #TODO improve usage uf multi-headed output after single-headed approach validated
            if output_prob is not None:
                output_prob = output_prob[0]

        if use_gpu == True:
            output = output.detach().cpu()
            output_prob = output_prob.detach().cpu() if output_prob is not None else None
            embed = embed.detach().cpu() if embed is not None else None

        batch_rows = dat_dev.shape[0]
        batch_ndim = dat_dev.ndim
        del dat_dev

        if output.ndim == 1:
            output = torch.unsqueeze(output, dim=1)
            if output_prob is not None:
                output_prob = torch.unsqueeze(output_prob, dim=1)
        if output_full is None:
            if data_ndim > 2:
                out_d1 = 1
                out_d2 = 2
                if output.ndim > 3:
                    out_d1 = 2
                    out_d2 = 3
                output_prob_full = torch.zeros((n_out, output.shape[out_d1], output.shape[out_d2]), dtype=torch.float32)
                output_full = torch.zeros((n_out, output.shape[out_d1], output.shape[out_d2]), dtype=torch.float32)
            else:
                output_full = torch.zeros(n_out, output.shape[1], dtype=torch.float32)
                output_prob_full = torch.zeros(n_out, output.shape[1], dtype=torch.float32)

        # Keep only the real rows of a padded last batch.
        if n_real < batch_rows:
            output = output[:n_real]
            output_prob = output_prob[:n_real] if output_prob is not None else None
            embed = embed[:n_real] if embed is not None else None
        ind1 = ind2
        ind2 = min(ind1 + n_real, n_out)
        if batch_ndim > 2:
            output_full[ind1:ind2,:,:] = torch.squeeze(output)
            if output_prob is not None:
                output_prob_full[ind1:ind2,:,:] = torch.squeeze(output_prob)
        else:
            output_full[ind1:ind2,:] = output
            if output_prob is not None:
                output_prob_full[ind1:ind2,:] = output_prob

        if embed is not None:
            if embed.ndim == 1:
                embed = torch.unsqueeze(embed, dim=1)
            if embed_full is None:
                if data_ndim > 2:
                    embed_full = torch.zeros((n_out, embed.shape[1], embed.shape[2], embed.shape[3]), dtype=torch.float32)
                else:
                    embed_full = torch.zeros(n_out, embed.shape[1], dtype=torch.float32)
            if batch_ndim > 2:
                embed_full[ind1:ind2,:,:,:] = torch.squeeze(embed)
            else:
                embed_full[ind1:ind2,:] = embed

        del output
        del output_prob

    print("AVERAGE INFERENCE", np.mean(inference_times) if len(inference_times) > 0 else 0.0)
    return output_full, embed_full, output_prob_full


def _output_options(yml_conf):
    """PNG / GeoTIFF options from the output: YAML section (defaults match the previous behaviour)."""
    out_conf = (yml_conf or {}).get("output") or {}
    return {"save_png": bool(out_conf.get("save_png", True)),
            "png_dpi": int(out_conf.get("png_dpi", 400)),
            "gtiff_compress": out_conf.get("gtiff_compress", None)}


def generate_output(dat, mdl, use_gpu, out_dir, output_fle, pin_mem = True, tiled = False, return_embed = False, batch_size = None,
        geo_ref = None, subset_inds = None, output_opts = None):

    output_full, embed_full, output_prob = run_inference(dat, mdl, use_gpu, out_dir, output_fle, pin_mem, tiled, return_embed, batch_size)

    print("SAVING", os.path.join(out_dir, output_fle), dat.targets_full.shape, output_full.shape)
    return plot_clusters(dat.targets_full, output_full.numpy(), os.path.join(out_dir, output_fle), output_prob, pixel_padding=1,
        geo_ref=geo_ref, subset_inds=subset_inds, **(output_opts or {}))


def _to_numpy(arr):
    if arr is None:
        return None
    if torch.is_tensor(arr):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _per_pixel_values(vals, n_pix, name):
    """(N,), (N,1), (N,1,1)... -> (N,). Mirrors the old loop, which assigned vals[i] to one pixel."""
    vals = vals[:n_pix]
    if vals.shape[0] < n_pix:
        raise IndexError("%s has %d rows but %d pixels were requested" % (name, vals.shape[0], n_pix))
    vals = vals.reshape(n_pix, -1)
    if vals.shape[1] != 1:
        raise ValueError("%s must have one value per pixel for non-tiled output, got shape %s"
                         % (name, str(vals.shape)))
    return vals[:, 0]


def _assign_pixels(coord, labels, prob, line_ind, samp_ind, pixel_padding):
    """
    Build the (label, probability) rasters from per-pixel predictions.

    Vectorized equivalent of the original per-pixel loop:
        data[coord[i,line_ind], coord[i,samp_ind]] = labels[i]
    Duplicate coordinates (e.g. the rows run_inference appends to pad the last
    batch) resolve to the *last* occurrence, exactly like the sequential loop.
    NumPy does not guarantee an order for repeated fancy-index writes, so the
    last occurrence per pixel is selected explicitly.
    """
    coord = _to_numpy(coord)
    labels = _to_numpy(labels)
    prob = _to_numpy(prob)

    # targets_full is int16; widen so pixel arithmetic can't overflow.
    lines = coord[:, line_ind].astype(np.int64)
    samps = coord[:, samp_ind].astype(np.int64)

    shape = (int(lines.max()) + 1 + pixel_padding, int(samps.max()) + 1 + pixel_padding)
    data = np.zeros(shape) - 1   # -1 separates No Data from cluster 0
    prob_data = np.zeros(shape)

    n_pix = labels.shape[0]
    if lines.shape[0] < n_pix:
        raise IndexError("coord has %d rows but labels has %d" % (lines.shape[0], n_pix))
    lines = lines[:n_pix]
    samps = samps[:n_pix]

    lab_vals = _per_pixel_values(labels, n_pix, "labels")
    prob_vals = _per_pixel_values(prob, n_pix, "output_prob") if prob is not None else None

    if n_pix == 0:
        return data, prob_data

    # Same negative-index wraparound / bounds behaviour as scalar indexing.
    lines = np.where(lines < 0, lines + shape[0], lines)
    samps = np.where(samps < 0, samps + shape[1], samps)
    if lines.min() < 0 or samps.min() < 0:
        raise IndexError("negative pixel coordinate out of range for raster of shape %s" % (shape,))

    flat = lines * shape[1] + samps
    _, first_in_reversed = np.unique(flat[::-1], return_index=True)
    keep = (n_pix - 1) - first_in_reversed   # last occurrence of each pixel

    data[lines[keep], samps[keep]] = lab_vals[keep]
    if prob_vals is not None:
        prob_data[lines[keep], samps[keep]] = prob_vals[keep]
    return data, prob_data


def plot_clusters(coord, output_data, output_basename, output_prob = None, pixel_padding=1,
        geo_ref = None, subset_inds = None, save_png = True, png_dpi = 400, gtiff_compress = None):
        """
        Writes <basename>_<n>clusters.zarr / .png / .no_geo.tif (values = label / 1000) and, with
        probabilities, <basename>_proba.zarr / _proba.png / .no_geo.proba.tif.

        geo_ref (from sit_fuse.inference.geo_output.resolve_reference) additionally writes
        <basename>_<n>clusters.zarr.full_geo.tif and <basename>_proba.zarr.full_geo.tif, matching
        postprocessing/generate_cluster_geotiffs.py output naming and layout.
        save_png=False skips the matplotlib renders; gtiff_compress (e.g. "DEFLATE") compresses GeoTIFFs.

        Returns (cluster raster as written, probability raster, file basename with cluster count).
        """

        max_cluster = output_data.shape[1]
        min_cluster = 0
        labels = None
        #if output_data.shape[1] > 1:
        #    max_cluster = output_data.shape[1]
        #    labels = np.argmax(output_data, axis = 1)
        #else:
        labels = output_data.astype(np.int32)
        prob = _to_numpy(output_prob)
        max_cluster = labels.max() #TODO this better!!!

        #print(np.unique(labels).shape, "UNIQUE LABELS", np.unique(labels), coord.shape, output_data.shape)

        n_clusters_local = max_cluster - min_cluster

        coord = _to_numpy(coord)
        data = []
        line_ind = 0
        samp_ind = 1
        if coord.shape[1] >  2:
            line_ind = 1
            samp_ind = 2

        #1 subtracted to separate No Data from areas that have cluster value 0.
        if labels.ndim <= 2:
            labels = np.array(labels)
            print("ASSIGNING LABELS", min_cluster, max_cluster)
            data, prob_data = _assign_pixels(coord, labels, prob, line_ind, samp_ind, pixel_padding)
    
        else:
            # Tiled output: one patch per row. Overlapping patches must be written
            # in order (later tiles overwrite earlier ones), so this stays a loop
            # over tiles (few iterations) rather than pixels.
            lines = coord[:, line_ind].astype(np.int64)
            samps = coord[:, samp_ind].astype(np.int64)
            max_dim1 = int(lines.max())
            max_dim2 = int(samps.max())
            prob_data =  np.zeros(((max_dim1+1+pixel_padding), (max_dim2+pixel_padding+1)))
            data = np.zeros(((max_dim1+1+pixel_padding), (max_dim2+pixel_padding+1))) - 1
            labels = np.array(labels)
            for i in range(labels.shape[0]):
                l0 = lines[i]
                s0 = samps[i]
                max_line = min(data.shape[0],   l0+labels[i].shape[0])
                max_samp = min(data.shape[1],   s0+labels[i].shape[1])

                if prob is not None:
                    prob_data[l0:max_line, s0:max_samp] = prob[i, :max_line - l0, :max_samp - s0]

                data[l0:max_line, s0:max_samp] = labels[i, :max_line - l0, :max_samp - s0] 

        print("FINISHED WITH LABEL ASSIGNMENT")
        data = data.astype(np.float32)
        data = (data/1000.0).astype(np.float32)
        prob_data = np.asarray(prob_data)

        clust_base = output_basename + "_" + str(n_clusters_local) + "clusters"
        tif_opts = ["COMPRESS=%s" % gtiff_compress, "TILED=YES"] if gtiff_compress else []

        da.to_zarr(da.from_array(data), clust_base + ".zarr", overwrite=True)
        if save_png:
            img = plt.imshow(data, vmin=-1, vmax=max_cluster)
            cmap = ListedColormap(CMAP_COLORS[0:int(max_cluster - (-1) + 1)])
            img.set_cmap(cmap)
            plt.colorbar()
            plt.savefig(clust_base + ".png", dpi=png_dpi, bbox_inches='tight')
            plt.clf()
        if prob is not None:
            da.to_zarr(da.from_array(prob_data), output_basename + "_proba.zarr", overwrite=True)
            if save_png:
                # Previously plotted the cluster raster here instead of the probabilities.
                vmax = 1.0 if prob_data.size == 0 or float(np.nanmax(prob_data)) <= 1.0 else 100
                img = plt.imshow(prob_data, vmin=0, vmax=vmax, cmap='plasma')
                plt.colorbar()
                plt.savefig(output_basename + "_proba.png", dpi=png_dpi, bbox_inches='tight')
                plt.clf()

            fname = output_basename + ".no_geo.proba.tif"
            out_ds = gdal.GetDriverByName("GTiff").Create(fname, data.shape[1], data.shape[0], 1, gdal.GDT_Float32, options=tif_opts)
            out_ds.GetRasterBand(1).WriteArray(prob_data)
            out_ds.FlushCache()
            out_ds = None

        fname = clust_base + ".no_geo.tif"
        out_ds = gdal.GetDriverByName("GTiff").Create(fname, data.shape[1], data.shape[0], 1, gdal.GDT_Float32, options=tif_opts)
        out_ds.GetRasterBand(1).WriteArray(data)
        out_ds.FlushCache()
        out_ds = None

        if geo_ref is not None:
            write_geotiff(clust_base + ".zarr.full_geo.tif", data, geo_ref, subset_inds, gtiff_compress)
            if prob is not None:
                write_geotiff(output_basename + "_proba.zarr.full_geo.tif", prob_data, geo_ref, subset_inds, gtiff_compress)

        return data, prob_data, clust_base



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
    if encoder_type is not None and  "dbn" in encoder_type and encoder_type != "rtdbn":
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
                 learning_rate=learning_rate, momentum=momentum, decay=decay, temperature=temp, use_gpu=resolve_use_gpu(yml_conf))
        else:
            model_type = yml_conf["dbn"]["model_type"]
            visible_shape = yml_conf["data"]["tile_size"]
            number_channel = yml_conf["data"]["number_channels"]
            #stride = yml_conf["dbn"]["stride"]
            #padding = yml_conf["dbn"]["padding"]
            encoder = ConvDBN(model=model_type, visible_shape=visible_shape, filter_shape = dbn_arch[1], n_filters = dbn_arch[0], \
                n_channels=number_channel, steps=gibbs_steps, learning_rate=learning_rate, momentum=momentum, \
                decay=decay, use_gpu=resolve_use_gpu(yml_conf), maxpooling=[False]*len(gibbs_steps)) #, maxpooling=mp)


        encoder.load_state_dict(torch.load(enc_ckpt_path, map_location="cpu"))

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
            encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location="cpu"))
        elif model_type == "DeepLab":
            encoder = DeepLab(num_classes, in_chans, backbone='resnet', pretrained=True, checkpoint_path=encoder_dir)
            encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location="cpu")) 
        elif model_type == "Unet":
            m1 = UNetEncoder(num_classes, in_channels=in_chans)
            m1.load_state_dict(torch.load(encoder_ckpt_path, map_location="cpu"))
            m2 = UNetDecoder(num_classes, in_channels=in_chans)
            encoder = torch.nn.Sequential(m1, m2)
        elif model_type == "DCE":
            m1 = DeepConvEncoder(in_chans)
            m1.load_state_dict(torch.load(encoder_ckpt_path, map_location="cpu"))
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


def _encoder_dir(yml_conf):
    save_dir = yml_conf["output"]["out_dir"]
    if yml_conf["logger"]["use_wandb"]:
        save_dir = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
    return os.path.join(save_dir, "encoder")


def _input_feature_size(yml_conf):
    """
    Per-sample input size used to build the model (only the non-conv DBN encoder needs it).

    Previously a full training scene was read and preprocessed just to get data_full.shape[1].
    It is now taken from the saved DBN checkpoint (first-layer weights), or from
    encoder_scaler.pkl (n_features_in_ x (2 * pixel_padding + 1)^2) for non-tiled data,
    falling back to reading a training scene as before.
    """
    encoder_type = yml_conf.get("encoder_type")
    if encoder_type is None or "dbn" not in encoder_type or encoder_type == "rtdbn" or "conv" in encoder_type:
        return None  # n_visible is unused by get_model for these encoders
    try:
        state = torch.load(os.path.join(_encoder_dir(yml_conf), "dbn.ckpt"), map_location="cpu")
        for key in ("models.0.W", "models.0.weight"):
            if key in state:
                return int(state[key].shape[0])
    except Exception:
        pass
    if not yml_conf["data"]["tile"]:
        try:
            scaler = joblib.load(os.path.join(yml_conf["output"]["out_dir"], "encoder_scaler.pkl"))
            pad = int(yml_conf["data"]["pixel_padding"])
            return int(scaler.n_features_in_) * (2 * pad + 1) ** 2
        except Exception:
            pass
    train_fnames = yml_conf["data"]["files_train"]
    for fname in train_fnames:
        data, _  = get_prediction_dataset(yml_conf, fname)
        if data is not None and data.data_full is not None:
            return data.data_full.shape[1]
    raise RuntimeError("Could not determine the model input size from checkpoints or training files.")


def prep_model(yml_conf):
    apply_compute_limits(yml_conf)  # thread/CPU throttling from the `compute:` YAML section

    model = get_model(yml_conf, _input_feature_size(yml_conf))

    if resolve_use_gpu(yml_conf):
 
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

def _iter_prediction_datasets(yml_conf, fnames, prefetch = 0):
    """
    Yields (index, fname, dataset) in order. With compute.prefetch_files > 0 the next file(s)
    are read/preprocessed in a background thread while the current one runs inference
    (costs one extra scene in memory per prefetched file).
    """
    prefetch = int(prefetch or 0)
    if prefetch <= 0:
        for i, fname in enumerate(fnames):
            yield i, fname, get_prediction_dataset(yml_conf, fname)[0]
        return
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = {}
        for i in range(min(prefetch, len(fnames))):
            futures[i] = pool.submit(get_prediction_dataset, yml_conf, fnames[i])
        for i, fname in enumerate(fnames):
            data = futures.pop(i).result()[0]
            nxt = i + prefetch
            if nxt < len(fnames):
                futures[nxt] = pool.submit(get_prediction_dataset, yml_conf, fnames[nxt])
            yield i, fname, data


def _predict_files(yml_conf, model, fnames, train = False):
    out_dir = yml_conf["output"]["out_dir"]
    generate_intermediate_output = yml_conf["output"]["generate_intermediate_output"]
    tiled = yml_conf["data"]["tile"]
    use_gpu = resolve_use_gpu(yml_conf)
    compute_conf = get_compute_conf(yml_conf)
    batch_size = compute_conf.get("inference_batch_size")
    output_opts = _output_options(yml_conf)
    geo = geo_output_enabled(yml_conf)

    for i, fname, data in _iter_prediction_datasets(yml_conf, fnames, compute_conf.get("prefetch_files", 0)):
        if isinstance(fname, list):
            output_fle = os.path.basename(fname[0])
        else:
            output_fle = os.path.basename(fname)
        if data is None or data.data_full is None:
            print("SKIPPING", fname, " No valid samples")
            continue
        geo_ref, subset_inds = None, None
        if geo:
            geo_ref, subset_inds = resolve_reference(yml_conf, fname, i, train=train)
            if geo_ref is None:
                print("WARNING: no georeferencing found for", fname, "- writing .no_geo outputs only")
        kwargs = dict(tiled = tiled, batch_size = batch_size, geo_ref = geo_ref, subset_inds = subset_inds, output_opts = output_opts)
        if model.clust_tree_ckpt is not None:
            generate_output(data, model, use_gpu, out_dir, output_fle + ".clust.data", **kwargs)
        if generate_intermediate_output:
            generate_output(data, model.pretrained_model, use_gpu, out_dir, output_fle + ".no_heir.clust.data", **kwargs)
        del data


def predict(yml_conf):

    test_fnames = yml_conf["data"]["files_test"]
    train_fnames = yml_conf["data"]["files_train"]

    model = prep_model(yml_conf)

    _predict_files(yml_conf, model, test_fnames, train = False)
    if yml_conf["output"]["generate_train_output"]:
        _predict_files(yml_conf, model, train_fnames, train = True)

def gen_embeddings_from_arr(yml_conf, data_arr, init_shape, gen_image_shaped = True, strat_inds = None):

    model = prep_model(yml_conf)

    data, output_file  = get_prediction_dataset_from_scene_arr(yml_conf, data_arr, init_shape, strat_inds)
    if data is None or data.data_full is None:
        print("SKIPPING scene array: no valid samples")
        return None, None
    return _embed_dataset(yml_conf, model, data, output_file, gen_image_shaped)


def _embed_dataset(yml_conf, model, data, output_file, gen_image_shaped):
    output = None
    embed = None
    if model.pretrained_model is not None:
        print("GENERATING EMBEDDING")
        output, embed, _ = run_inference(data, model, resolve_use_gpu(yml_conf), yml_conf["output"]["out_dir"], output_file + ".clust.data", \
                tiled = yml_conf["encoder"]["tiled"], return_embed =  True, \
                batch_size = get_compute_conf(yml_conf).get("inference_batch_size"))

        if gen_image_shaped:
            recon_arr, recon_lab = conscruct_embed_image(embed, output, data.targets_full, init_shape = data.init_shape)
            return recon_arr, recon_lab

    return embed, output

def gen_embeddings(yml_conf, fname, gen_image_shaped = True):

    model = prep_model(yml_conf)

    data, output_file  = get_prediction_dataset(yml_conf, fname)
    if data.data_full is None:
        print("SKIPPING", fname, " No valid samples")
        return None, None
    return _embed_dataset(yml_conf, model, data, output_file, gen_image_shaped)

def conscruct_embed_image(embed, labels, coord, init_shape = []):
    embed = _to_numpy(embed)
    labels = _to_numpy(labels)
    coord = _to_numpy(coord)

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
        # Was indented under the for-loop above (a for/else), so per-pixel embeddings were never written.
        rows = coord[:embed.shape[0], coord_coord_1].astype(np.int64)
        cols = coord[:embed.shape[0], coord_coord_2].astype(np.int64)
        reconstructed_arr[rows, cols] = embed.reshape(embed.shape[0], -1)
        reconstructed_labels[rows, cols] = labels.reshape(labels.shape[0], -1)[:, 0]

    return reconstructed_arr, reconstructed_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    parser.add_argument("-n", "--num-threads", default=None,
        help="CPU thread cap: int, negative int (all but N), fraction (0-1], or low/medium/high/max. "
             "Overrides compute.num_threads in the YAML.")
    args = parser.parse_args()
    if args.num_threads is not None:
        os.environ["SIT_FUSE_NUM_THREADS"] = str(args.num_threads)  # highest-priority override
    predict_outside(args.yaml)

    print(getrusage(RUSAGE_SELF))


