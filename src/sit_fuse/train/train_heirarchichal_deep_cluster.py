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

from segmentation.models.gcn import GCN
from segmentation.models.deeplabv3_plus_xception import DeepLab
from sit_fuse.models.encoders.cnn_encoder import DeepConvEncoder

from sit_fuse.models.encoders.pca_encoder import PCAEncoder
from sit_fuse.models.deep_cluster.multi_prototypes import MultiPrototypes, JEPA_Seg
from sit_fuse.models.deep_cluster.heir_dc import Heir_DC, get_state_dict
from sit_fuse.datasets.sf_heir_dataset_module import SFHeirDataModule
from sit_fuse.datasets.sf_dataset import SFDataset
from sit_fuse.datasets.sf_dataset_conv import SFDatasetConv
from sit_fuse.datasets.dataset_utils import get_train_dataset_sf
from sit_fuse.utils import read_yaml, get_output_shape

import wandb

import joblib

import uuid
import argparse
import os

def heir_dc(yml_conf, dataset, ckpt_path):

    use_wandb_logger = yml_conf["logger"]["use_wandb"]
    log_model = None
    save_dir = yml_conf["output"]["out_dir"]
    project = None
    if use_wandb_logger:
        log_model = yml_conf["logger"]["log_model"]
        save_dir = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
        project = yml_conf["logger"]["project"]

    full_model_dir = os.path.join(save_dir, "full_model")
    save_dir = os.path.join(save_dir, "full_model_heir")
    encoder_dir = os.path.join(save_dir, "encoder")

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


    ckpt_path = os.path.join(full_model_dir, "deep_cluster.ckpt")

    encoder_type=None
    if "encoder_type" in yml_conf:
        encoder_type=yml_conf["encoder_type"]

    in_chans = None
    tile_size = None
    encoder = None
    if encoder_type is not None and  "pca" in encoder_type:
        encoder = PCAEncoder()

        use_wandb_logger = yml_conf["logger"]["use_wandb"]
        if use_wandb_logger:
            encoder_dir = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
        encoder_dir = os.path.join(encoder_dir, "encoder")

        encoder.pca = joblib.load(os.path.join(encoder_dir, "pca.pkl"))
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
            encoder = DBN(model=model_type, n_visible=dataset.data_full.shape[1], n_hidden=dbn_arch, steps=gibbs_steps,
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
 
        for param in encoder.parameters():
            param.requires_grad = False
        for model in encoder.models:
            for param in model.parameters():
                param.requires_grad = False
        encoder.eval()

    elif encoder_type is not None and encoder_type == "byol":
        save_dir_byol = yml_conf["output"]["out_dir"]
 
        if use_wandb_logger:
            log_model = yml_conf["logger"]["log_model"]
            save_dir_byol = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
            project = yml_conf["logger"]["project"]

        encoder_dir = os.path.join(save_dir_byol, "encoder")
        in_chans = yml_conf["data"]["tile_size"][2]
        tile_size = yml_conf["data"]["tile_size"][0]*yml_conf["data"]["tile_size"][1]
        encoder_ckpt_path = os.path.join(encoder_dir, "byol.ckpt")


        model_type = cutout_ratio_range = yml_conf["byol"]["model_type"]
        if model_type == "GCN":
            #encoder = GCN(num_classes, in_chans)
            encoder = GCN(num_classes, in_chans, pretrained = False, use_deconv=True, use_resnet_gcn=True)
        elif model_type == "DeepLab":
            encoder = DeepLab(num_classes, in_chans, backbone='resnet', pretrained=True, checkpoint_path=encoder_dir)
        elif model_type == "DCE":
            encoder = DeepConvEncoder(in_chans)
        encoder.load_state_dict(torch.load(encoder_ckpt_path))
            #encoder.eval()
            #output_dim = in_chans*8*tile_size*tile_size
            #m2 =  MultiPrototypes(output_dim, num_classes, 1)
            #model_2 = torch.nn.Sequential(encoder, m2)


        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()

    elif encoder_type is not None and encoder_type == "ijepa" or encoder_type == "clay" or encoder_type == "mae":
        in_chans = yml_conf["data"]["tile_size"][2]
        tile_size = yml_conf["data"]["tile_size"][0]

 
    lr = yml_conf["cluster"]["training"]["learning_rate"]

    heir_ckpt_path = os.path.join(save_dir, "heir_fc.ckpt")
    if os.path.exists(heir_ckpt_path): #TODO make optional
        model = Heir_DC(None, pretrained_model_path=ckpt_path, num_classes=num_classes_heir, yml_conf=yml_conf, \
            encoder_type=encoder_type, encoder=encoder, clust_tree_ckpt = heir_ckpt_path, lr = lr)
    else: 
        model = Heir_DC(dataset, pretrained_model_path=ckpt_path, num_classes=num_classes_heir, yml_conf=yml_conf, \
            encoder_type=encoder_type, encoder=encoder, lr = lr)

    for param in model.pretrained_model.parameters():
        param.requires_grad = False

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)
 
    os.makedirs(save_dir, exist_ok=True)

    count = 0 
    num_loader_workers = int(yml_conf["data"]["num_loader_workers"])
    val_percent = float(yml_conf["data"]["val_percent"])
    batch_size = yml_conf["cluster"]["heir"]["training"]["batch_size"]

    for key in model.lab_full.keys():

        count = count + 1
        if len(model.lab_full[key]) < model.min_samples:
            model.clust_tree["1"][key] = None
            continue

        print("TRAINING MODEL ", str(count), " / ", str(len(model.lab_full.keys())))

        encoder_output_size = None
        n_visible = 0
        if "encoder_type" not in yml_conf.keys():
            n_visible = yml_conf["data"]["tile_size"][2] * yml_conf["data"]["tile_size"][0] * yml_conf["data"]["tile_size"][1]
        elif yml_conf["encoder_type"] == "ijepa":
            #TODO encoder_output_size = get_output_shape(model.pretrained_model.pretrained_model.model, (2, in_chans,model.pretrained_model.pretrained_model.img_size,model.pretrained_model.pretrained_model.img_size))
            #n_visible = encoder_output_size[2] #[1]*encoder_output_size[2]
             n_visible = 512 #1024
        elif yml_conf["encoder_type"] == "dbn":
            encoder_output_size = (1, model.pretrained_model.pretrained_model.models[-1].n_hidden)
            n_visible = encoder_output_size[1]
        elif yml_conf["encoder_type"] == "conv_dbn":
            encoder_output_size = get_output_shape(encoder, (1, yml_conf["data"]["tile_size"][2], yml_conf["data"]["tile_size"][0], yml_conf["data"]["tile_size"][1]))
            n_visible = 900 #TODO encoder_output_size[1]
        elif yml_conf["encoder_type"] == "byol":
            encoder_output_size = get_output_shape(encoder, (1, yml_conf["data"]["tile_size"][2], yml_conf["data"]["tile_size"][0], yml_conf["data"]["tile_size"][1]))
            n_visible = encoder_output_size[1]
        elif yml_conf["encoder_type"] == "clay":
            #TODO print((2, in_chans, yml_conf["data"]["tile_size"][0], yml_conf["data"]["tile_size"][1]))
            #encoder_output_size = get_output_shape(model.pretrained_model.pretrained_model.encoder, (2, in_chans, yml_conf["data"]["tile_size"][0], yml_conf["data"]["tile_size"][1]))
            n_visible = 768 #encoder_output_size[2]
        elif yml_conf["encoder_type"] == "mae":
            #TODO print((2, in_chans, yml_conf["data"]["tile_size"][0], yml_conf["data"]["tile_size"][1]))
            #encoder_output_size = get_output_shape(model.pretrained_model.pretrained_model.encoder, (2, in_chans, yml_conf["data"]["tile_size"][0], yml_conf["data"]["tile_size"][1]))
            n_visible = 1024 #encoder_output_size[2]
        elif yml_conf["encoder_type"] == "pca":
            encoder_output_size = (1, model.pretrained_model.pretrained_model.pca.components_.shape[0])
            n_visible = encoder_output_size[1]
        print("HERE ENCODER SHAPE", encoder_output_size)

        if key in model.clust_tree["1"] and model.clust_tree["1"][key] is not None:
            continue #TODO make optional

        #TODO num_channels from uppper modeli

        #if yml_conf["encoder_type"] == "ijepa":
        #    model.clust_tree["1"][key] = JEPA_Seg(num_classes_heir)
        #else:
        model.clust_tree["1"][key] = MultiPrototypes(n_visible, model.num_classes, model.number_heads)

        model.clust_tree["1"][key].train() 
        for param in model.clust_tree["1"][key].parameters():
            param.requires_grad = True 

        model.module_list.append(model.clust_tree["1"][key])

        #if "tile" not in yml_conf["data"] or yml_conf["data"]["tile"] == False:
        #    train_subset = SFDataset()
        #    train_subset.init_from_array(dataset.data_full[model.lab_full[key]], 
        #        dataset.targets_full[model.lab_full[key]], scaler = dataset.scaler)
        #else:
        #    train_subset = SFDatasetConv()
        #    train_subset.init_from_array(dataset.data_full[model.lab_full[key]], 
        #        dataset.targets_full[model.lab_full[key]], transform = dataset.transform)

        final_dataset = SFHeirDataModule(dataset, batch_size=batch_size, num_workers=num_loader_workers, val_percent=val_percent)
        model.key = key

        if use_wandb_logger:

            wandb_logger = WandbLogger(project=project, log_model=log_model, save_dir = save_dir)

            trainer = pl.Trainer(
                limit_train_batches=100,
                default_root_dir=save_dir,
                accelerator=accelerator,
                devices=devices,
                strategy=DDPStrategy(find_unused_parameters=True),
                precision=precision,
                max_epochs=max_epochs,
                callbacks=[lr_monitor, model_summary],
                gradient_clip_val=gradient_clip_val,
                logger=wandb_logger
            )
        else:
            trainer = pl.Trainer(
                limit_train_batches=100,
                default_root_dir=save_dir,
                accelerator=accelerator,
                devices=devices,
                strategy=DDPStrategy(find_unused_parameters=True),
                precision=precision,
                max_epochs=max_epochs,
                callbacks=[lr_monitor, model_summary],
                gradient_clip_val=gradient_clip_val
            )



        state_dict = get_state_dict(model.clust_tree, model.lab_full)
        torch.save(state_dict, os.path.join(save_dir, "heir_fc.ckpt"))
        trainer.fit(model, final_dataset)

        if use_wandb_logger:
            wandb.finish()
        state_dict = get_state_dict(model.clust_tree, model.lab_full)
        torch.save(state_dict, os.path.join(save_dir, "heir_fc.ckpt"))


    state_dict = get_state_dict(model.clust_tree, model.lab_full)
    torch.save(state_dict, os.path.join(save_dir, "heir_fc.ckpt"))         

        #for param in model.clust_tree["1"][key].parameters():
        #    param.requires_grad = False
        #model.clust_tree["1"][key].eval()
        #model.module_list[-1] = model.clust_tree["1"][key]

def run_heir_training_outside(yml_fpath):

    yml_conf = read_yaml(yml_fpath)
    run_heir_training(yml_conf)

def run_heir_training(yml_conf):

    dataset = get_train_dataset_sf(yml_conf)

    yml_conf = read_yaml(yml_fpath)

    save_dir = yml_conf["output"]["out_dir"]
    use_wandb_logger = yml_conf["logger"]["use_wandb"]
    if use_wandb_logger:
        save_dir = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
    ckpt_path = os.path.join(os.path.join(save_dir, "full_model"), "checkpoint.ckpt")

    heir_dc(yml_conf, dataset, ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    run_heir_training_outside(args.yaml)

