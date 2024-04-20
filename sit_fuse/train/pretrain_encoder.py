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



def pretrain_DBN(yml_conf, dataset):

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
    for i, model in enumerate(dbn.models):
        current_rbm = model
        previous_layers = None
        if i > 0:
            previous_layers = dbn.models[:i]

        model = DBN_PL(current_rbm, previous_layers, learning_rate[i], momentum[i], nesterov_accel[i], decay[i])

        lr_monitor = LearningRateMonitor(logging_interval="step")
        model_summary = ModelSummary(max_depth=2)

        wandb_logger = WandbLogger(project="SIT-FUSE", log_model=True, save_dir = "/home/nlahaye/SIT_FUSE_DEV/wandb_dbn/")

        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            strategy=DDPStrategy(),
            precision="16-mixed",
            max_epochs=30,
            callbacks=[lr_monitor, model_summary],
            gradient_clip_val=.1,
            logger=wandb_logger
        )

        trainer.fit(model, dataset)

    torch.save(dbn.state_dict(), "/home/nlahaye/SIT_FUSE_DEV/wandb_dbn/dbn_2.ckpt")

 
def pretrain_IJEPA(yml_conf, dataset):

    model = IJEPA_PL(img_size=3, patch_size=1, in_chans=34, embed_dim=128, enc_heads=8, enc_depth=8, decoder_depth=6, lr=1e-3)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    wandb_logger = WandbLogger(project="SIT-FUSE", log_model=True, save_dir = "/home/nlahaye/SIT_FUSE_DEV/wandb_small_full/", id="a02ltqei", resume="must")

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=6,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="16-mixed",
        max_epochs=100,
        callbacks=[lr_monitor, model_summary],
        gradient_clip_val=.1,
        logger=wandb_logger
    )

    trainer.fit(model, dataset)


def pretrain_BYOL(yml_conf, dataset):

    model = DeepConvEncoder(in_chans=34, flatten=True)

    learner = BYOL_Learner(
        model,
        image_size = 3,
        hidden_layer = -1,
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)
 
    wandb_logger = WandbLogger(project="SIT-FUSE", log_model=True, save_dir = "/home/nlahaye/SIT_FUSE_DEV/wandb_byol/")

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=4,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="16-mixed",
        max_epochs=100,
        callbacks=[lr_monitor, model_summary],
        gradient_clip_val=.1,
        logger=wandb_logger
    )

    trainer.fit(learner, dataset)



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



