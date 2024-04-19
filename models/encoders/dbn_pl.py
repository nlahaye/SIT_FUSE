import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)

from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger

from learnergy.models.deep import DBN

import argparse
from ...utils import numpy_to_torch, read_yaml, get_read_func, get_scaler

'''
pytorch lightning model
'''
class DBN_PL(pl.LightningModule):
    def __init__(
            self,
            model,
            previous_layers = None,
            learning_rate = 1e-5,
            momentum = 0.95,
            nesterov_accel = True,
            decay = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'previous_layers'])
        
        #define models
        self.model = model
        self.previous_layers = previous_layers
        self.lr = learning_rate
        self.momentum = momentum
        self.nesterov_accel = nesterov_accel
        self.decay = decay
    
        self.register_module("current_rbm", self.model)
        if self.previous_layers is not None:
            for i in range(len(self.previous_layers)):
                self.register_module("previous_layer_rbm_" + str(i), self.previous_layers[i])

    def forward(self, x):
        return self.model(x)
    

    def training_step(self, batch, batch_idx):
  
        if self.previous_layers is not None:
            for mod in range(len(previous_layers)):
                batch = previous_layers[mod](batch)
 
        if self.model.normalize:
            samples = (
                (batch - torch.mean(batch, 0, True))
                    / (torch.std(batch, 0, True) + 1e-6)
                )

        samples = samples.reshape(len(samples), self.model.n_visible)

        # Performs the Gibbs sampling procedure
        _, _, _, _, visible_states = self.model.gibbs_sampling(samples)
        visible_states = visible_states

        loss = torch.mean(self.model.energy(samples)) - torch.mean(
            self.model.energy(visible_states)
        )

        batch_mse = torch.div(
            torch.sum(torch.pow(samples - visible_states, 2)), batch.shape[0] 
        )
        batch_pl = self.model.pseudo_likelihood(samples)
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_batch_mse',  batch_mse, sync_dist=True)
        self.log('train_batch_pl', batch_pl, sync_dist=True)        
          
        return loss
    
   
    def validation_step(self, batch, batch_idx):

        if self.previous_layers is not None:
            for mod in range(len(previous_layers)):
                batch = previous_layers[mod](batch)

        if self.model.normalize:
            samples = (
                (batch - torch.mean(batch, 0, True))
                    / (torch.std(batch, 0, True) + 1e-6)
                )

        samples = samples.reshape(len(samples), self.model.n_visible)

        # Performs the Gibbs sampling procedure
        _, _, _, _, visible_states = self.model.gibbs_sampling(samples)
        visible_states = visible_states

        loss = torch.mean(self.model.energy(samples)) - torch.mean(
            self.model.energy(visible_states)
        )

        batch_mse = torch.div(
            torch.sum(torch.pow(samples - visible_states, 2)), batch.shape[0]
        )
        print(self.model.device, samples.device)
        batch_pl = self.model.pseudo_likelihood(samples)
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_batch_mse',  batch_mse, sync_dist=True)
        self.log('val_batch_pl', batch_pl, sync_dist=True) 

        return loss

    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        return self(batch)


    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.decay, nesterov=self.nesterov_accel)


def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)

    dataset = D2VDataModule(dataset_path='data')
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
            strategy=DDPStrategy(find_unused_parameters=True),
            precision="16-mixed",
            max_epochs=100,
            callbacks=[lr_monitor, model_summary],
            gradient_clip_val=.1,
            logger=wandb_logger
        )

        trainer.fit(model, dataset)



if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)
