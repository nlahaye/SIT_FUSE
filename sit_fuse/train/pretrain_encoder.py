import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger

from learnergy.models.deep import DBN

from sit_fuse.models.encoders.ijepa_pl import IJEPA_PL
from sit_fuse.models.encoders.dbn_pl import DBN_PL
from sit_fuse.models.encoders.byol_pl import BYOL_Learner
from sit_fuse.models.encoders.cnn_encoder import DeepConvEncoder
from sit_fuse.datasets.sf_dataset_module import SFDataModule
from sit_fuse.utils import read_yaml

import argparse
import os


def pretrain_DBN(yml_conf, dataset):

    dataset.setup()

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

    dbn = DBN(model=model_type, n_visible=dataset.n_visible, n_hidden=dbn_arch, steps=gibbs_steps,
        learning_rate=learning_rate, momentum=momentum, decay=decay, temperature=temp, use_gpu=True)


    use_wandb_logger = yml_conf["logger"]["use_wandb"]
    log_model = None
    save_dir = None
    project = None
    if use_wandb_logger:
        log_model = yml_conf["logger"]["log_model"]
        save_dir = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
        project = yml_conf["logger"]["project"]

    accelerator = yml_conf["encoder"]["training"]["accelerator"]
    devices = yml_conf["encoder"]["training"]["devices"]
    precision = yml_conf["encoder"]["training"]["precision"]
    max_epochs = yml_conf["encoder"]["training"]["epochs"]
    gradient_clip_val = yml_conf["encoder"]["training"]["gradient_clip_val"]

    for i, model in enumerate(dbn.models):
        current_rbm = model
        previous_layers = None
        if i > 0:
            previous_layers = dbn.models[:i]
 
        dbn.models[i].normalize = normalize_learnergy[i]
        dbn.models[i].batch_normalize = batch_normalize[i]

        model = DBN_PL(current_rbm, previous_layers, learning_rate[i], momentum[i], nesterov_accel[i], decay[i])

        lr_monitor = LearningRateMonitor(logging_interval="step")
        model_summary = ModelSummary(max_depth=2)
 
        if use_wandb_logger:
            os.makedirs(save_dir, exist_ok=True)
            wandb_logger = WandbLogger(project=project, log_model=log_model, save_dir = save_dir)

            trainer = pl.Trainer(
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
                accelerator=accelerator,
                devices=devices,
                strategy=DDPStrategy(find_unused_parameters=True),
                precision=precision,
                max_epochs=max_epochs,
                callbacks=[lr_monitor, model_summary],
                gradient_clip_val=gradient_clip_val
            )

        trainer.fit(model, dataset)
    
    torch.save(dbn.state_dict(), os.path.join(save_dir, "dbn.ckpt"))

 
def pretrain_IJEPA(yml_conf, dataset):

    patch_size = int(yml_conf["ijepa"]["patch_size"])
    embed_dim = int(yml_conf["ijepa"]["embed_dim"])
    enc_heads = int(yml_conf["ijepa"]["encoder_heads"])
    enc_depth = int(yml_conf["ijepa"]["encoder_depth"])
    decoder_depth =  int(yml_conf["ijepa"]["decoder_depth"])
    
    weight_decay = yml_conf["encoder"]["training"]["weight_decay"]
    momentum = yml_conf["encoder"]["training"]["momentum"]
    m_start_end = tuple(yml_conf["encoder"]["training"]["momentum_start_end"])
    lr = yml_conf["encoder"]["training"]["learning_rate"]

    target_aspect_ratio = tuple(yml_conf["ijepa"]["target_aspect_ratio"])
    target_scale  = tuple(yml_conf["ijepa"]["target_scale"])
    context_aspect_ratio = yml_conf["ijepa"]["context_aspect_ratio"]
    context_scale = tuple(yml_conf["ijepa"]["context_scale"])
    M = yml_conf["ijepa"]["number_target_blocks"]        
    
    img_size = yml_conf["data"]["tile_size"][0]
    in_chans = yml_conf["data"]["tile_size"][2]

    use_wandb_logger = yml_conf["logger"]["use_wandb"]
    log_model = None
    save_dir = None
    project = None
    if use_wandb_logger:
        log_model = yml_conf["logger"]["log_model"]
        save_dir = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
        project = yml_conf["logger"]["project"]

    accelerator = yml_conf["encoder"]["training"]["accelerator"]
    devices = yml_conf["encoder"]["training"]["devices"]
    precision = yml_conf["encoder"]["training"]["precision"]
    max_epochs = yml_conf["encoder"]["training"]["epochs"] 
    gradient_clip_val = yml_conf["encoder"]["training"]["gradient_clip_val"]


    model = IJEPA_PL(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
        embed_dim=embed_dim, enc_heads=enc_heads, enc_depth=enc_depth, decoder_depth=decoder_depth,
        lr=lr, weight_decay=weight_decay, m=momentum, m_start_end=m_start_end, M=M, 
        target_aspect_ratio=target_aspect_ratio, target_scale=target_scale, context_aspect_ratio=context_aspect_ratio,
        context_scale=context_scale)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)
 
    if use_wandb_logger:
        os.makedirs(out_dir, exist_ok=True)
        wandb_logger = WandbLogger(project=project, log_model=log_model, save_dir = save_dir)
        #id="a02ltqei", resume="must")

        trainer = pl.Trainer(
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
            accelerator=accelerator,
            devices=devices,
            strategy=DDPStrategy(find_unused_parameters=True),
            precision=precision,
            max_epochs=max_epochs,
            callbacks=[lr_monitor, model_summary],
            gradient_clip_val=gradient_clip_val
        )

    trainer.fit(model, dataset)


def pretrain_BYOL(yml_conf, dataset):

    img_size = yml_conf["data"]["tile_size"][0]
    in_chans = yml_conf["data"]["tile_size"][2]

    hidden_layer = yml_conf["byol"]["hidden_layer"]
    projection_size = yml_conf["byol"]["projection_size"]
    projection_hidden_size = yml_conf["byol"]["projection_hidden_size"]
    moving_average_decay = yml_conf["byol"]["moving_average_decay"]

    use_wandb_logger = yml_conf["logger"]["use_wandb"]
    log_model = None
    save_dir = None
    project = None
    if use_wandb_logger:
        log_model = yml_conf["logger"]["log_model"]
        save_dir = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
        project = yml_conf["logger"]["project"]

    accelerator = yml_conf["encoder"]["training"]["accelerator"]
    devices = yml_conf["encoder"]["training"]["devices"]
    precision = yml_conf["encoder"]["training"]["precision"]
    max_epochs = yml_conf["encoder"]["training"]["epochs"]
    gradient_clip_val = yml_conf["encoder"]["training"]["gradient_clip_val"]

    model = DeepConvEncoder(in_chans=in_chans, flatten=True)
 
    learner = BYOL_Learner(
        model,
        image_size = img_size,
        hidden_layer = hidden_layer,
        projection_size = projection_size,
        projection_hidden_size = projection_hidden_size,
        moving_average_decay = moving_average_decay
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)
 
    if use_wandb_logger:
        os.makedirs(save_dir, exist_ok=True)
 
        wandb_logger = WandbLogger(project=project, log_model=log_model, save_dir = save_dir)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=precision,
        max_epochs=max_epochs,
        callbacks=[lr_monitor, model_summary],
        gradient_clip_val=gradient_clip_val,
        logger=wandb_logger
    )

    trainer.fit(learner, dataset)
    torch.save(model.state_dict(), os.path.join(save_dir, "byol.ckpt")


def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)

    num_loader_workers = int(yml_conf["data"]["num_loader_workers"])
    val_percent = float(yml_conf["data"]["val_percent"])
    batch_size = yml_conf["encoder"]["training"]["batch_size"]    

    dataset = SFDataModule(yml_conf, batch_size, num_loader_workers, val_percent=val_percent)

    if "encoder_type" in yml_conf:
        if yml_conf["encoder_type"] == "dbn":
            pretrain_DBN(yml_conf, dataset)
        elif yml_conf["encoder_type"] == "ijepa":
            pretrain_IJEPA(yml_conf, dataset)
        elif yml_conf["encoder_type"] == "byol":
            pretrain_BYOL(yml_conf, dataset)
        #TODO Pixel CL?

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)



