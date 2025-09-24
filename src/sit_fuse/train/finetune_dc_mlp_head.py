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
from sit_fuse.models.encoders.pca_encoder import PCAEncoder
from sit_fuse.models.deep_cluster.ijepa_dc import IJEPA_DC
from sit_fuse.models.deep_cluster.mae_dc import MAE_DC
from sit_fuse.models.deep_cluster.dbn_dc import DBN_DC
from sit_fuse.models.deep_cluster.byol_dc import BYOL_DC
from sit_fuse.models.encoders.byol_pl import BYOL_Learner
from sit_fuse.models.deep_cluster.cdbn_dc import CDBN_DC
from sit_fuse.models.deep_cluster.clay_dc import Clay_DC
from sit_fuse.models.deep_cluster.pca_dc import PCA_DC
from sit_fuse.models.deep_cluster.multi_prototypes import MultiPrototypes, DeepConvMultiPrototypes
from sit_fuse.datasets.sf_dataset_module import SFDataModule
from sit_fuse.utils import read_yaml
 
from segmentation.models.deeplabv3_plus_xception import DeepLab
from segmentation.models.gcn import GCN
from segmentation.models.unet import UNetEncoder, UNetDecoder

import joblib
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

    img_size = yml_conf["data"]["tile_size"][0]*yml_conf["data"]["tile_size"][1]
    in_chans = yml_conf["data"]["tile_size"][2]

    model = DeepCluster(num_classes=num_classes, conv=yml_conf["conv"], img_size=img_size, in_chans=in_chans)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir, filename="deep_cluster", enable_version_counter=False, every_n_train_steps = 100, save_on_train_epoch_end=False)

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

def dc_PCA(yml_conf, dataset, conv=False):
 
    dataset.setup()
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

    ckpt_path = os.path.join(encoder_dir, "pca.pkl")
    lr = yml_conf["cluster"]["training"]["learning_rate"]

    pretrained_model = PCAEncoder()
    pretrained_model.pca = joblib.load(ckpt_path)


    model = PCA_DC(pretrained_model, num_classes, weight_decay=0.95, lr = lr)
    for param in model.mlp_head.parameters():
        param.requires_grad = True

    model.mlp_head.train()

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir, filename="deep_cluster", enable_version_counter=False, every_n_train_steps = 100, save_on_train_epoch_end=False)

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
            decay=decay, use_gpu=True, maxpooling=[False]*len(gibbs_steps)) #, maxpooling=mp)

    dbn.load_state_dict(torch.load(ckpt_path))

    for param in dbn.parameters():
        param.requires_grad = False
    for model in dbn.models:
        for param in model.parameters():
            param.requires_grad = False
    dbn.eval() 
    dbn.models.eval()

    if not conv:
        model = DBN_DC(dbn, num_classes=num_classes, conv=conv)
    else:
        lr = yml_conf["cluster"]["training"]["learning_rate"]
        model = CDBN_DC(dbn, num_classes=num_classes, weight_decay=0.95, lr = lr)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir, filename="deep_cluster", enable_version_counter=False, every_n_train_steps = 100, save_on_train_epoch_end=False)

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


def dc_Clay(yml_conf, dataset):

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

    ckpt_path = os.path.join(encoder_dir, "clay-v1-base.ckpt")

    gsd = yml_conf["cluster"]["gsd"]
    waves = yml_conf["cluster"]["waves"] #TODO
    feature_maps = [3,5,7,11]#["cluster"]["feature_maps"]
    


    model = Clay_DC(ckpt_path, num_classes, feature_maps, waves, gsd, lr = 1e-5, weight_decay=0.05)
    #for param in model.pretrained_model.parameters():
    #    param.requires_grad = finetune_encoder
    #for param in model.mlp_head.parameters():
    #    param.requires_grad = True

    model.pretrained_model.train()
    model.pretrained_model.encoder.train()

    #if not finetune_encoder:
    #    model.pretrained_model.eval()
    #    model.pretrained_model.encoder.eval()

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir, filename="deep_cluster", enable_version_counter=False, every_n_train_steps = 100, save_on_train_epoch_end=False)

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

    #trainer.save_checkpoint(os.path.join(save_dir, "deep_cluster.ckpt"))
    trainer.fit(model, dataset)


def dc_MAE(yml_conf, dataset):

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
    lr = yml_conf["cluster"]["training"]["learning_rate"]

    model = MAE_DC(ckpt_path, num_classes, weight_decay=0.95, lr = lr)
    for param in model.pretrained_model.parameters():
        param.requires_grad = finetune_encoder
    for param in model.mlp_head.parameters():
        param.requires_grad = True

    model.mlp_head.train()
    model.pretrained_model.train()
    model.pretrained_model.vit.train()

    if not finetune_encoder:
        model.pretrained_model.eval()
        model.pretrained_model.vit.eval()

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir, filename="deep_cluster", enable_version_counter=False, every_n_train_steps = 100, save_on_train_epoch_end=False)

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
    lr = yml_conf["cluster"]["training"]["learning_rate"]

    model = IJEPA_DC(ckpt_path, num_classes, weight_decay=0.95, lr = lr)
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
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir, filename="deep_cluster", enable_version_counter=False, every_n_train_steps = 100, save_on_train_epoch_end=False) 

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
        model_2 = GCN(num_classes, in_chans, pretrained = False, use_deconv=True, use_resnet_gcn=True)
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
        m1.load_state_dict(torch.load(ckpt_path))
        m1 = m1.eval()
        #output_dim = in_chans*8*img_size*img_size
        m2 = DeepConvMultiPrototypes(in_chans*8, num_classes, 1)
        #m2 =  MultiPrototypes(output_dim, num_classes, 1)
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
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir, filename="deep_cluster", enable_version_counter=False, every_n_train_steps = 100, save_on_train_epoch_end=False)



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



def run_tuning_outside(yml_fpath):

    yml_conf = read_yaml(yml_fpath)
    run_tuning(yml_conf)


def run_tuning(yml_conf):

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
        elif yml_conf["encoder_type"] == "clay":
            dc_Clay(yml_conf, dataset)
        elif yml_conf["encoder_type"] == "mae":
            dc_MAE(yml_conf, dataset)
        elif yml_conf["encoder_type"] == "pca":
            dc_PCA(yml_conf, dataset)
    else:
        train_dc_no_pt(yml_conf, dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    run_tuning_outside(args.yaml)

