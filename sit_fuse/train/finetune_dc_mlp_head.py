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

    model_type = tuple(yml_conf["dbn"]["params"]["model_type"])
    dbn_arch = tuple(yml_conf["dbn"]["params"]["dbn_arch"])
    gibbs_steps = tuple(yml_conf["dbn"]["params"]["gibbs_steps"])
    normalize_learnergy = tuple(yml_conf["dbn"]["params"]["normalize_learnergy"])
    batch_normalize = tuple(yml_conf["dbn"]["params"]["batch_normalize"])

    learning_rate = tuple(yml_conf["dbn"]["params"]["learning_rate"])
    momentum = tuple(yml_conf["dbn"]["params"]["momentum"])
    decay = tuple(yml_conf["dbn"]["params"]["decay"])
    nesterov_accel = tuple(yml_conf["dbn"]["params"]["nesterov_accel"])
    temp = tuple(yml_conf["dbn"]["params"]["temp"])

    dbn = DBN(model=model_type, n_visible=dataset.n_visible, n_hidden=dbn_arch, steps=gibbs_steps,
        learning_rate=learning_rate, momentum=momentum, decay=decay, temperature=temp, use_gpu=True)
    dbn.load_state_dict(torch.load("/home/nlahaye/SIT_FUSE_DEV/wandb_dbn/dbn.ckpt"))

    for param in dbn.parameters():
        param.requires_grad = False
    dbn.eval() 

    model = DBN_DC(dbn, num_classes=800)
  
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    wandb_logger = WandbLogger(project="SIT-FUSE", log_model=True, save_dir = "/home/nlahaye/SIT_FUSE_DEV/wandb_dbn_finetune/")

    trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            strategy=DDPStrategy(find_unused_parameters=True),
            precision="16-mixed",
            max_epochs=100,
            callbacks=[lr_monitor, model_summary],
            gradient_clip_val=.1,
            logger=wandb_logger
    )

    trainer.fit(model, dataset)



def dc_IJEPA(yml_conf, dataset):

    model = IJEPA_DC(pretrained_model_path="/data/nlahaye/output/Learnergy/IJEPA_TEST_FULL/ijepa.ckpt", num_classes=800)
    for param in model.pretrained_model.parameters():
        param.requires_grad = False

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    wandb_logger = WandbLogger(project="SIT-FUSE", log_model=True, save_dir = "/home/nlahaye/SIT_FUSE_DEV/wandb_finetune_ijepa_small_full/")

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=2,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="16-mixed",
        max_epochs=50,
        callbacks=[lr_monitor, model_summary],
        gradient_clip_val=.1,
        logger=wandb_logger
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

