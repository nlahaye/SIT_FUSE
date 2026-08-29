import argparse
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.strategies import DDPStrategy

from learnergy.models.temporal.rt_variance_gaussian_rbm import RTVarianceGaussianRBM
from learnergy.models.temporal.rtdbn import RTDBN

from sit_fuse.models.encoders.rtdbn_pl import RTDBN_PL
from sit_fuse.datasets.rtdbn_data_utils import set_all_seeds, build_rtdbn_splits
import yaml


def read_yaml(fpath):
    with open(fpath, "r") as f:
        return yaml.safe_load(f)


def pretrain_RTDBN(yml_conf):

    num_workers = int(yml_conf["data"]["num_loader_workers"])

    n_visible = int(yml_conf["rtdbn"]["n_visible"])
    n_hidden = tuple(yml_conf["rtdbn"]["n_hidden"])
    gibbs_steps = tuple(yml_conf["rtdbn"]["gibbs_steps"])
    temp = tuple(yml_conf["rtdbn"]["temp"])
    model_type = tuple(yml_conf["rtdbn"]["model_type"])

    learning_rate = tuple(yml_conf["encoder"]["training"]["learning_rate"])
    momentum = tuple(yml_conf["encoder"]["training"]["momentum"])
    decay = tuple(yml_conf["encoder"]["training"]["weight_decay"])
    nesterov_accel = tuple(yml_conf["encoder"]["training"]["nesterov_accel"])
    warmup_epochs = int(yml_conf["encoder"]["training"]["warmup_epochs"])
    batch_size = yml_conf["encoder"]["training"]["batch_size"]
    epochs = yml_conf["encoder"]["training"]["epochs"]
    accelerator = yml_conf["encoder"]["training"]["accelerator"]
    devices = yml_conf["encoder"]["training"]["devices"]
    precision = yml_conf["encoder"]["training"]["precision"]
    gradient_clip = yml_conf["encoder"]["training"]["gradient_clip_val"]

    save_dir = os.path.join(yml_conf["output"]["out_dir"], "encoder")
    os.makedirs(save_dir, exist_ok=True)

    set_all_seeds(42)

    # Fits and persists the scaler for downstream stages; test_ds unused here.
    train_ds, val_ds, test_ds = build_rtdbn_splits(
        yml_conf, scaler_out_path=os.path.join(save_dir, "scaler.pkl")
    )

    # Fail fast on a config/data n_visible mismatch (e.g. env version change).
    actual_n_visible = train_ds.data_full.shape[-1]
    if actual_n_visible != n_visible:
        raise ValueError(
            f"Config n_visible={n_visible} does not match the actual data "
            f"feature dimension ({actual_n_visible}) found in "
            f"{yml_conf['data']['data_dir']}. If this data was regenerated "
            f"(e.g. a different myosuite/gymnasium version), update "
            f"rtdbn.n_visible in the config to match."
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader = DataLoader(val_ds,   batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    rtdbn = RTDBN(
        model=model_type,
        n_visible=n_visible,
        n_hidden=n_hidden,
        steps=gibbs_steps,
        learning_rate=learning_rate,
        momentum=momentum,
        decay=decay,
        temperature=temp,
        use_gpu=(accelerator == "gpu"),
    )

    for i, model in enumerate(rtdbn.models):

        max_epochs = epochs[i] if isinstance(epochs, list) else epochs
        previous_layers = rtdbn.models[:i] if i > 0 else None

        pl_model = RTDBN_PL(
            model=model,
            save_dir=save_dir,
            previous_layers=previous_layers,
            learning_rate=learning_rate[i],
            momentum=momentum[i],
            nesterov_accel=nesterov_accel[i],
            decay=decay[i],
            warmup_epochs=warmup_epochs,
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        model_summary = ModelSummary(max_depth=2)
        checkpoint_steps = 100
        if max_epochs < checkpoint_steps:
            checkpoint_steps = 1
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir,
            filename="rtdbn_encoder",
            enable_version_counter=False,
            every_n_train_steps=checkpoint_steps,
            save_on_train_epoch_end=False
        )

        trainer = pl.Trainer(
            default_root_dir=save_dir,
            accelerator=accelerator,
            devices=devices,
            # find_unused_parameters=True: sigma's requires_grad toggles mid-training.
            strategy=DDPStrategy(find_unused_parameters=True),
            precision=precision,
            max_epochs=max_epochs,
            callbacks=[lr_monitor, model_summary, checkpoint_callback],
            gradient_clip_val=gradient_clip,
        )

        trainer.fit(pl_model, train_loader, val_loader)

        rtdbn.models[i] = pl_model.model
        torch.save(rtdbn.state_dict(), os.path.join(save_dir, "rtdbn.ckpt"))

    torch.save(rtdbn.state_dict(), os.path.join(save_dir, "rtdbn.ckpt"))
    print(f"RTDBN encoder saved to {save_dir}/rtdbn.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml",
                        help="YAML config file for RTDBN training.")
    args = parser.parse_args()

    yml_conf = read_yaml(args.yaml)
    pretrain_RTDBN(yml_conf)
