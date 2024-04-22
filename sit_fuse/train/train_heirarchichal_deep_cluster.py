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

from sit_fuse.models.deep_cluster.multi_prototypes import MultiPrototypes
from sit_fuse.models.deep_cluster.heir_dc import Heir_DC
from sit_fuse.datasets.sf_heir_dataset_module import SFHeirDataModule
from sit_fuse.datasets.sf_dataset import SFDataset
from sit_fuse.datasets.sf_dataset_conv import SFDatasetConv
from sit_fuse.datasets.dataset_utils import get_train_dataset_sf
from sit_fuse.utils import read_yaml

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

    accelerator = yml_conf["cluster"]["heir"]["training"]["accelerator"]
    devices = yml_conf["cluster"]["heir"]["training"]["devices"]
    precision = yml_conf["cluster"]["heir"]["training"]["precision"]
    max_epochs = yml_conf["cluster"]["heir"]["training"]["epochs"]
    gradient_clip_val = yml_conf["encoder"]["training"]["gradient_clip_val"]
    heir_model_tiers = yml_conf["cluster"]["heir"]["tiers"] #TODO incorporate
    heir_gauss_stdev = yml_conf["cluster"]["heir"]["gauss_noise_stdev"] #TODO incorporate
    lambda_iid = yml_conf["cluster"]["heir"]["lambda"] #TODO incorporate 

    num_classes = yml_conf["cluster"]["heir"]["num_classes"]
    min_samples = yml_conf["cluster"]["heir"]["training"]["min_samples"]


    ckpt_path = os.path.join(full_model_dir, "checkpoint.ckpt")

    model = Heir_DC(dataset, pretrained_model_path=ckpt_path, num_classes=num_classes, encoder_type=yml_conf["encoder_type"])
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
        print("LABEL", key, len(model.lab_full[key]))
        if len(model.lab_full[key]) < model.min_samples:
            model.clust_tree["1"][key] = None
            continue

        print("TRAINING MODEL ", str(count), " / ", str(len(model.lab_full.keys())))
 
        model.clust_tree["1"][key] = MultiPrototypes(model.pretrained_model.num_classes, model.num_classes, model.number_heads)
       
        for param in model.clust_tree["1"][key].parameters():
            param.requires_grad = True 
  

        if "tile" not in yml_conf["data"] or yml_conf["data"]["tile"] == False:
            train_subset = SFDataset()
            train_subset.init_from_array(dataset.data_full[model.lab_full[key]], 
                dataset.targets_full[model.lab_full[key]], scaler = dataset.scaler)
        else:
            train_subset = SFDatasetConv()
            train_subset.init_from_array(dataset.data_full[model.lab_full[key]], 
                dataset.targets_full[model.lab_full[key]], transform = dataset.transform)

        final_dataset = SFHeirDataModule(train_subset, batch_size=batch_size, num_workers=num_loader_workers, val_percent=val_percent)

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


        trainer.fit(model, final_dataset)

        for param in model.clust_tree["1"][key].parameters():
            param.requires_grad = False
        model.clust_tree["1"][key].eval()

def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)

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
    main(args.yaml)

