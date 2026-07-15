import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger

from learnergy.models.deep import DBN, ConvDBN

from sit_fuse.models.deep_cluster.dbn_dc import DBN_DC
from sit_fuse.datasets.sf_dataset_module import SFDataModule
from sit_fuse.utils import read_yaml

import joblib
import argparse
import os


def add_layers(yml_conf, dataset, conv=False):

    dataset.setup()

    dbn_arch = tuple(yml_conf["dbn"]["dbn_arch"])
    gibbs_steps = tuple(yml_conf["dbn"]["gibbs_steps"])
    normalize_learnergy = tuple(yml_conf["dbn"]["normalize_learnergy"])
    batch_normalize = tuple(yml_conf["dbn"]["batch_normalize"])
    temp = tuple(yml_conf["dbn"]["temp"])

    learning_rate = tuple(yml_conf["encoder"]["training"]["learning_rate"])
    momentum = tuple(yml_conf["encoder"]["training"]["momentum"])
    decay = tuple(yml_conf["encoder"]["training"]["weight_decay"])
    nesterov_accel = tuple(yml_conf["encoder"]["training"]["nesterov_accel"])

    use_wandb_logger = yml_conf["logger"]["use_wandb"]
    log_model = None
    save_dir = yml_conf["output"]["out_dir"]
    project = None
    if use_wandb_logger:
        log_model = yml_conf["logger"]["log_model"]
        save_dir = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
        project = yml_conf["logger"]["project"]

    encoder_dir = os.path.join(save_dir, "encoder")
    save_dir = os.path.join(save_dir, "encoder")

    accelerator = yml_conf["cluster"]["training"]["accelerator"]
    devices = yml_conf["cluster"]["training"]["devices"]
    precision = yml_conf["cluster"]["training"]["precision"]
    max_epochs = yml_conf["cluster"]["training"]["epochs"]
    gradient_clip_val = yml_conf["cluster"]["training"]["gradient_clip_val"]
    gauss_stdev = yml_conf["cluster"]["gauss_noise_stdev"] #TODO incorporate
    lambda_iid = yml_conf["cluster"]["lambda"] #TODO incorporate

    num_classes = yml_conf["cluster"]["num_classes"]

    ckpt_path = os.path.join(encoder_dir, "dbn.ckpt")

    if not conv:
        model_type = tuple(yml_conf["dbn"]["model_type"])
        dbn = DBN(model=model_type, n_visible=dataset.n_visible, n_hidden=dbn_arch, steps=gibbs_steps,
             learning_rate=learning_rate, momentum=momentum, decay=decay, temperature=temp, use_gpu=True)
    else:
        model_type = yml_conf["dbn"]["model_type"]
        visible_shape = yml_conf["data"]["tile_size"]
        number_channel = yml_conf["data"]["number_channels"]
        #stride = yml_conf["dbn"]["stride"]
        #padding = yml_conf["dbn"]["padding"]
        dbn = ConvDBN(model=model_type, visible_shape=visible_shape, filter_shape = dbn_arch[1], n_filters = dbn_arch[0], \
            n_channels=number_channel, steps=gibbs_steps, learning_rate=learning_rate, momentum=momentum, \
            decay=decay, use_gpu=True, maxpooling=[False]*len(gibbs_steps)) #, maxpooling=mp)

    print(ckpt_path, conv, model_type)
    dbn.load_state_dict(torch.load(ckpt_path))

    model_type = list(model_type)
    new_model_type = list(yml_conf["dbn"]["new_layers"]["model_type"])
    new_dbn_arch = list(yml_conf["dbn"]["new_layers"]["dbn_arch"])
    new_gibbs_steps = list(yml_conf["dbn"]["new_layers"]["gibbs_steps"])
    new_temp = list(yml_conf["dbn"]["new_layers"]["temp"])
    new_norm = list(yml_conf["dbn"]["new_layers"]["normalize_learnergy"])
    new_bn = list(yml_conf["dbn"]["new_layers"]["batch_normalize"])
    new_lr =  list(yml_conf["dbn"]["new_layers"]["learning_rate"])
    new_momentum =  list(yml_conf["dbn"]["new_layers"]["momentum"])
    new_nest_accel = list(yml_conf["dbn"]["new_layers"]["nesterov_accel"])
    new_decay = list(yml_conf["dbn"]["new_layers"]["weight_decay"])


    decay = list(decay)
    dbn_arch = list(dbn_arch)
    gibbs_steps = list(gibbs_steps)
    temp = list(temp)
    learning_rate = list(learning_rate)
    momentum = list(momentum)

    decay.extend(new_decay)
    model_type.extend(new_model_type)
    dbn_arch.extend(new_dbn_arch)
    gibbs_steps.extend(new_gibbs_steps)
    temp.extend(new_temp)
    learning_rate.extend(new_lr)
    momentum.extend(new_momentum)
   


    decay = tuple(decay)
    dbn_arch = tuple(dbn_arch)
    gibbs_steps = tuple(gibbs_steps)
    temp = tuple(temp)
    learning_rate = tuple(learning_rate)
    momentum = tuple(momentum)

    model_type = tuple(model_type) 
    new_dbn = DBN(model=model_type, n_visible=dataset.n_visible, n_hidden=dbn_arch, steps=gibbs_steps,
             learning_rate=learning_rate, momentum=momentum, decay=decay, temperature=temp, use_gpu=True)
 
    for i in range(len(dbn.models)):
        new_dbn.models[i] = dbn.models[i] 
 
    torch.save(new_dbn.state_dict(), os.path.join(save_dir, "dbn.ckpt")) 
 

def add_dbn_layers_outside(yml_fpath):

    yml_conf = read_yaml(yml_fpath)

    add_dbn_layers(yml_conf)

def add_dbn_layers(yml_conf):

    num_loader_workers = int(yml_conf["data"]["num_loader_workers"])
    val_percent = float(yml_conf["data"]["val_percent"])
    batch_size = yml_conf["encoder"]["training"]["batch_size"]

    dataset = SFDataModule(yml_conf, batch_size, num_loader_workers, val_percent=val_percent)

    add_layers(yml_conf, dataset, ("conv" in yml_conf["encoder_type"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    add_dbn_layers_outside(args.yaml)

