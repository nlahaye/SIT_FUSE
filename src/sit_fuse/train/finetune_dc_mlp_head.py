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

from sit_fuse.models.deep_cluster.dc import DeepCluster
from sit_fuse.models.encoders.cnn_encoder import DeepConvEncoder
from sit_fuse.models.deep_cluster.ijepa_dc import IJEPA_DC
from sit_fuse.models.deep_cluster.dbn_dc import DBN_DC
from sit_fuse.models.deep_cluster.byol_dc import BYOL_DC
from sit_fuse.models.encoders.byol_pl import BYOL_Learner
from sit_fuse.models.deep_cluster.multi_prototypes import MultiPrototypes
from sit_fuse.datasets.sf_dataset_module import SFDataModule
from sit_fuse.utils import read_yaml
 
from segmentation.models.deeplabv3_plus_xception import DeepLab
from segmentation.models.gcn import GCN
from segmentation.models.unet import UNetEncoder, UNetDecoder

import argparse
import os

def train_dc_no_pt(yml_conf, dataset, conv=False):

    use_wandb_logger = yml_conf["logger"]["use_wandb"]
    log_model = None
    save_dir = yml_conf["output"]["out_dir"]
    project = None
    if use_wandb_logger:
        log_model = yml_conf["logger"]["log_model"]
        save_dir = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
        project = yml_conf["logger"]["project"]

    encoder_dir = os.path.join(save_dir, "encoder")
    save_dir = os.path.join(save_dir, "full_model")

    accelerator = yml_conf["cluster"]["training"]["accelerator"]
    devices = yml_conf["cluster"]["training"]["devices"]
    precision = yml_conf["cluster"]["training"]["precision"]
    max_epochs = yml_conf["cluster"]["training"]["epochs"]
    gradient_clip_val = yml_conf["cluster"]["training"]["gradient_clip_val"]
    gauss_stdev = yml_conf["cluster"]["gauss_noise_stdev"] #TODO incorporate
    lambda_iid = yml_conf["cluster"]["lambda"] #TODO incorporate

    num_classes = yml_conf["cluster"]["num_classes"]

    img_size = yml_conf["data"]["tile_size"][0]
    in_chans = yml_conf["data"]["tile_size"][2]

    model = DeepCluster(num_classes=num_classes, conv=yml_conf["conv"], img_size=img_size, in_chans=in_chans)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir, filename="deep_cluster", every_n_epochs=1, save_on_train_epoch_end=False)

    os.makedirs(save_dir, exist_ok=True)
    if use_wandb_logger:

        wandb_logger = WandbLogger(project=project, log_model=log_model, save_dir = save_dir)

        trainer = pl.Trainer(
            default_root_dir=save_dir,
            accelerator=accelerator,
            devices=devices,
            strategy=DDPStrategy(find_unused_parameters=True),
            precision=precision,
            max_epochs=max_epochs,
            callbacks=[lr_monitor, model_summary, checkpoint_callback],
            gradient_clip_val=gradient_clip_val,
            logger=wandb_logger
        )
    else:
        trainer = pl.Trainer(
            default_root_dir=save_dir,
            accelerator=accelerator,
            devices=devices,
            strategy=DDPStrategy(find_unused_parameters=True),
            precision=precision,
            max_epochs=max_epochs,
            callbacks=[lr_monitor, model_summary, checkpoint_callback],
            gradient_clip_val=gradient_clip_val
        )

    trainer.fit(model, dataset)



def dc_DBN(yml_conf, dataset, conv=False):

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
    save_dir = os.path.join(save_dir, "full_model")

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
            decay=decay, use_gpu=True) #, maxpooling=mp)

    dbn.load_state_dict(torch.load(ckpt_path))

    for param in dbn.parameters():
        param.requires_grad = False
    for model in dbn.models:
        for param in model.parameters():
            param.requires_grad = False
    dbn.eval() 
    dbn.models.eval()

    model = DBN_DC(dbn, num_classes=num_classes, conv=conv)
  
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir, filename="deep_cluster", every_n_epochs=1, save_on_train_epoch_end=False)

    os.makedirs(save_dir, exist_ok=True)
    if use_wandb_logger:

        wandb_logger = WandbLogger(project=project, log_model=log_model, save_dir = save_dir)

        trainer = pl.Trainer(
            default_root_dir=save_dir,
            accelerator=accelerator,
            devices=devices,
            strategy=DDPStrategy(find_unused_parameters=True),
            precision=precision,
            max_epochs=max_epochs,
            callbacks=[lr_monitor, model_summary, checkpoint_callback],
            gradient_clip_val=gradient_clip_val,
            logger=wandb_logger
        )
    else:
        trainer = pl.Trainer(
            default_root_dir=save_dir,
            accelerator=accelerator,
            devices=devices,
            strategy=DDPStrategy(find_unused_parameters=True),
            precision=precision,
            max_epochs=max_epochs,
            callbacks=[lr_monitor, model_summary, checkpoint_callback],
            gradient_clip_val=gradient_clip_val
        )

    trainer.fit(model, dataset)


def dc_IJEPA(yml_conf, dataset):

    use_wandb_logger = yml_conf["logger"]["use_wandb"]
    log_model = None
    save_dir = yml_conf["output"]["out_dir"]
    project = None
    if use_wandb_logger:
        log_model = yml_conf["logger"]["log_model"]
        save_dir = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
        project = yml_conf["logger"]["project"]

    encoder_dir = os.path.join(save_dir, "encoder")
    save_dir = os.path.join(save_dir, "full_model")

    accelerator = yml_conf["cluster"]["training"]["accelerator"]
    devices = yml_conf["cluster"]["training"]["devices"]
    precision = yml_conf["cluster"]["training"]["precision"]
    max_epochs = yml_conf["cluster"]["training"]["epochs"]
    gradient_clip_val = yml_conf["cluster"]["training"]["gradient_clip_val"]
    gauss_stdev = yml_conf["cluster"]["gauss_noise_stdev"] #TODO incorporate
    lambda_iid = yml_conf["cluster"]["lambda"] #TODO incorporate

    num_classes = yml_conf["cluster"]["num_classes"]

    finetune_encoder = yml_conf["cluster"]["training"]["finetune_encoder"]

    ckpt_path = os.path.join(encoder_dir, "encoder.ckpt")

    model = IJEPA_DC(ckpt_path, num_classes)
    for param in model.pretrained_model.parameters():
        param.requires_grad = finetune_encoder
    for param in model.mlp_head.parameters():
        param.requires_grad = True

    model.mlp_head.train()
    model.pretrained_model.train()
    model.pretrained_model.model.train()
    model.pretrained_model.model.mode = "test"
 
    if not finetune_encoder:
        model.pretrained_model.eval()
        model.pretrained_model.model.eval()

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)
 
    os.makedirs(save_dir, exist_ok=True)
    if use_wandb_logger:

        wandb_logger = WandbLogger(project=project, log_model=log_model, save_dir = save_dir)

        trainer = pl.Trainer(
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
            default_root_dir=save_dir,
            accelerator=accelerator,
            devices=devices,
            strategy=DDPStrategy(find_unused_parameters=True),
            precision=precision,
            max_epochs=max_epochs,
            callbacks=[lr_monitor, model_summary],
            gradient_clip_val=gradient_clip_val
        )


    trainer.fit(model, dataset)


def dc_BYOL(yml_conf, dataset):

    use_wandb_logger = yml_conf["logger"]["use_wandb"]
    log_model = None
    save_dir = yml_conf["output"]["out_dir"]
    project = None
    if use_wandb_logger:
        log_model = yml_conf["logger"]["log_model"]
        save_dir = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
        project = yml_conf["logger"]["project"]

    encoder_dir = os.path.join(save_dir, "encoder")
    save_dir = os.path.join(save_dir, "full_model")

    accelerator = yml_conf["cluster"]["training"]["accelerator"]
    devices = yml_conf["cluster"]["training"]["devices"]
    precision = yml_conf["cluster"]["training"]["precision"]
    max_epochs = yml_conf["cluster"]["training"]["epochs"]
    gradient_clip_val = yml_conf["cluster"]["training"]["gradient_clip_val"]
    gauss_stdev = yml_conf["cluster"]["gauss_noise_stdev"] #TODO incorporate
    lambda_iid = yml_conf["cluster"]["lambda"] #TODO incorporate

    num_classes = yml_conf["cluster"]["num_classes"]

    in_chans = yml_conf["data"]["tile_size"][2]
    img_size = yml_conf["data"]["tile_size"][0]
    ckpt_path = os.path.join(encoder_dir, "byol.ckpt")
    #model_init = DeepConvEncoder(in_chans=in_chans, flatten=True)


    model_type = cutout_ratio_range = yml_conf["byol"]["model_type"]
    model_2 = None
    if model_type == "GCN":
        model_2 = GCN(num_classes, in_chans)
        model_2.load_state_dict(torch.load(ckpt_path))
    elif model_type == "DeepLab":
        model_2 = DeepLab(num_classes, in_chans, backbone='resnet', pretrained=True, checkpoint_path=encoder_dir)
        model_2.load_state_dict(torch.load(ckpt_path))
    elif model_type == "Unet":
        m1 = UNetEncoder(num_classes, in_channels=in_chans) 
        m1.load_state_dict(torch.load(ckpt_path))
        m2 = UNetDecoder(num_classes, in_channels=in_chans)  
        model_2 = torch.nn.Sequential(m1, m2)
    elif model_type == "DCE":
        m1 = DeepConvEncoder(in_chans)
        m1 = m1.eval()
        output_dim = in_chans*8*img_size*img_size
        m2 =  MultiPrototypes(output_dim, num_classes, 1)
        model_2 = torch.nn.Sequential(m1, m2)
    if hasattr(model_2, "backbone"):
        for param in model_2.backbone.parameters():
                param.requires_grad = False
        model_2.backbone.eval()
    else:
        for param in model_2[0].parameters():
                param.requires_grad = False
        model_2[0].eval()  


    model = BYOL_DC(pretrained_model=model_2, num_classes=num_classes, lr=1e-3, weight_decay=0, number_heads=1, tile_size =img_size, in_chans = in_chans, model_type = model_type)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir, filename="deep_cluster", every_n_epochs=1, save_on_train_epoch_end=False)



    os.makedirs(save_dir, exist_ok=True) 
    if use_wandb_logger:
        wandb_logger = WandbLogger(project=project, log_model=log_model, save_dir=save_dir)
 
        trainer = pl.Trainer(
            default_root_dir=save_dir,
            accelerator=accelerator,
            devices=devices,
            strategy=DDPStrategy(find_unused_parameters=True),
            precision=precision,
            max_epochs=max_epochs,
            callbacks=[lr_monitor, model_summary, checkpoint_callback],
            gradient_clip_val=gradient_clip_val,
            logger=wandb_logger
        )
    else:
        trainer = pl.Trainer(
            default_root_dir=save_dir,
            accelerator=accelerator,
            devices=devices,
            strategy=DDPStrategy(find_unused_parameters=True),
            precision=precision,
            max_epochs=max_epochs,
            callbacks=[lr_monitor, model_summary, checkpoint_callback],
            gradient_clip_val=gradient_clip_val,
        )

    trainer.fit(model, dataset)



def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)

    num_loader_workers = int(yml_conf["data"]["num_loader_workers"])
    val_percent = float(yml_conf["data"]["val_percent"])
    batch_size = yml_conf["cluster"]["training"]["batch_size"]

    dataset = SFDataModule(yml_conf, batch_size, num_loader_workers, val_percent=val_percent)

    if "encoder_type" in yml_conf:
        if "dbn" in yml_conf["encoder_type"]:
            dc_DBN(yml_conf, dataset, ("conv" in yml_conf["encoder_type"]))
        elif yml_conf["encoder_type"] == "ijepa":
            dc_IJEPA(yml_conf, dataset)
        elif yml_conf["encoder_type"] == "byol":
            dc_BYOL(yml_conf, dataset)
    else:
        train_dc_no_pt(yml_conf, dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)

