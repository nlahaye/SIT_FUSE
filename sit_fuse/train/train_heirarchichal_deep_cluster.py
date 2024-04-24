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

import wandb

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

    encoder_type=None
    if "encoder_type" in yml_conf:
        encoder_type=yml_conf["encoder_type"]

    encoder = None
    if encoder_type is not None and  encoder_type == "dbn":
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


        save_dir = yml_conf["output"]["out_dir"]
        use_wandb_logger = yml_conf["logger"]["use_wandb"]
        if use_wandb_logger:
            save_dir = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
        encoder_dir = os.path.join(save_dir, "encoder")

        enc_ckpt_path = os.path.join(encoder_dir, "dbn.ckpt")

        encoder = DBN(model=model_type, n_visible=dataset.data_full.shape[1], n_hidden=dbn_arch, steps=gibbs_steps,
            learning_rate=learning_rate, momentum=momentum, decay=decay, temperature=temp, use_gpu=True)

        encoder.load_state_dict(torch.load(enc_ckpt_path))
 
        for param in encoder.parameters():
            param.requires_grad = False
        for model in encoder.models:
            for param in model.parameters():
                param.requires_grad = False
        encoder.eval()


    model = Heir_DC(dataset, pretrained_model_path=ckpt_path, num_classes=num_classes, encoder_type=encoder_type, encoder=encoder)
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
 

        n_visible = list(model.pretrained_model.mlp_head.children())[1].num_features
        model.clust_tree["1"][key] = MultiPrototypes(n_visible, model.num_classes, model.number_heads)

        model.clust_tree["1"][key].train() 
        for param in model.clust_tree["1"][key].parameters():
            param.requires_grad = True 

        model.module_list.append(model.clust_tree["1"][key])

        if "tile" not in yml_conf["data"] or yml_conf["data"]["tile"] == False:
            train_subset = SFDataset()
            train_subset.init_from_array(dataset.data_full[model.lab_full[key]], 
                dataset.targets_full[model.lab_full[key]], scaler = dataset.scaler)
        else:
            train_subset = SFDatasetConv()
            train_subset.init_from_array(dataset.data_full[model.lab_full[key]], 
                dataset.targets_full[model.lab_full[key]], transform = dataset.transform)

        final_dataset = SFHeirDataModule(train_subset, batch_size=batch_size, num_workers=num_loader_workers, val_percent=val_percent)
        model.key = key

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

        if use_wandb_logger:
            wandb.finish()

    state_dict = get_state_dict(model.clust_tree, model.lab_full)
    torch.save(model.state_dict(), os.path.join(save_dir, "heir_fc.ckpt"))         

        #for param in model.clust_tree["1"][key].parameters():
        #    param.requires_grad = False
        #model.clust_tree["1"][key].eval()
        #model.module_list[-1] = model.clust_tree["1"][key]


def get_state_dict(clust_tree, lab_full):

    state_dict = {}
    for lab1 in clust_tree.keys():
        if lab1 == "0":
            continue
        if lab1 not in state_dict:
            state_dict[lab1] = {}
            for lab2 in clust_tree[lab1].keys():
                if lab2 not in state_dict[lab1].keys():
                    if clust_tree[lab1][lab2] is not None:
                        if lab2 not in state_dict[lab1].keys():
                            state_dict[lab1][lab2] = {}
                        state_dict[lab1][lab2]["model"] = clust_tree[lab1][lab2].state_dict()
                        uid = str(uuid.uuid1())
    state_dict["labels"] = lab_full
    return state_dict

def load_model(clust_tree, n_visible, model, state_dict):
        lab_full = state_dict["labels"]
        for lab1 in clust_tree.keys():
            if lab1 == "0":
                continue
            for lab2 in lab_full.keys():
                clust_tree[lab1][lab2] = None
                if lab2 in state_dict[lab1].keys():
                    clust_tree[lab1][lab2] = MultiPrototypes(n_visible, model.num_classes, model.number_heads)
                    clust_tree[lab1][lab2].load_state_dict(state_dict[lab1][lab2]["model"])
        return clust_tree, lab_full 


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

