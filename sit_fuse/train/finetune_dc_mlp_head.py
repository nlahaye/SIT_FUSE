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

from sit_fuse.models.deep_cluster.dc import DeepCluster
from sit_fuse.models.deep_cluster.ijepa_dc import IJEPA_DC
from sit_fuse.models.deep_cluster.dbn_dc import DBN_DC
from sit_fuse.datasets.sf_dataset_module import SFDataModule
from sit_fuse.utils import read_yaml

import argparse

def train_dc_no_pt(yml_conf, dataset):

    model = DeepCluster(num_classes=800, conv=yml_conf["conv"])

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    wandb_logger = WandbLogger(project="SIT-FUSE", log_model=True, save_dir = "/home/nlahaye/SIT_FUSE_DEV/wandb_cnn/")

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision="16-mixed",
        max_epochs=50,
        callbacks=[lr_monitor, model_summary],
        gradient_clip_val=.1,
        logger=wandb_logger
    )

    trainer.fit(model, dataset)


def dc_DBN(yml_conf, dataset):

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

    use_wandb_logger = yml_conf["logger"]["use_wandb"]
    log_model = None
    save_dir = yml_conf["output"]["out_dir"]
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

    num_classes = yml_conf["cluster"]["num_classes"]

    ckpt_path = os.path.join(save_dir, "dbn.ckpt")

    dbn = DBN(model=model_type, n_visible=dataset.n_visible, n_hidden=dbn_arch, steps=gibbs_steps,
        learning_rate=learning_rate, momentum=momentum, decay=decay, temperature=temp, use_gpu=True)
    dbn.load_state_dict(torch.load(ckpt_path))

    for param in dbn.parameters():
        param.requires_grad = False
    dbn.eval() 

    model = DBN_DC(dbn, num_classes=num_classes)
  
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    os.makedirs(save_dir, exist_ok=True)
    if use_wandb_logger:

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


def dc_IJEPA(yml_conf, dataset):

    use_wandb_logger = yml_conf["logger"]["use_wandb"]
    log_model = None
    save_dir = yml_conf["output"]["out_dir"]
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

    num_classes = yml_conf["cluster"]["num_classes"]

    ckpt_path = os.path.join(save_dir, "checkpoint.ckpt")

    model = IJEPA_DC(pretrained_model_path=ckpt_path, num_classes=num_classes)
    for param in model.pretrained_model.parameters():
        param.requires_grad = False
    for param in model.mlp_head.parameters():
        param.requires_grad = True

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)
 
    os.makedirs(save_dir, exist_ok=True)
    if use_wandb_logger:

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


def dc_BYOL(yml_conf, dataset):

    use_wandb_logger = yml_conf["logger"]["use_wandb"]
    log_model = None
    save_dir = yml_conf["output"]["out_dir"]
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

    num_classes = yml_conf["cluster"]["num_classes"]

    ckpt_path = os.path.join(save_dir, "byol.ckpt")

    model = BYOL_DC(pretrained_model_path=ckpt_path, num_classes=num_classes)
    for param in model.pretrained_model.parameters():
        param.requires_grad = False
    for param in model.mlp_head.parameters():
        param.requires_grad = True

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    os.makedirs(save_dir, exist_ok=True) 
    if use_wandb_logger:
        wandb_logger = WandbLogger(project=project, log_model=log_model, save_dir=save_dir)
 
        trainer = pl.Trainer(
            default_root_dir=save_dir
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
            default_root_dir=save_dir
            accelerator=accelerator,
            devices=devices,
            strategy=DDPStrategy(find_unused_parameters=True),
            precision=precision,
            max_epochs=max_epochs,
            callbacks=[lr_monitor, model_summary],
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
        if yml_conf["encoder_type"] == "dbn":
            dc_DBN(yml_conf, dataset)
        elif yml_conf["encoder_type"] == "ijepa":
            dc_IJEPA(yml_conf, dataset)
    else:
        train_dc_no_pt(yml_conf, dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)

