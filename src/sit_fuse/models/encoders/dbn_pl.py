import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch

from learnergy.models.deep import DBN

import argparse
from sit_fuse.utils import read_yaml

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
            for mod in range(len(self.previous_layers)):
                batch = self.previous_layers[mod](batch)
 
        if self.model.normalize:
            samples = (
                (batch - torch.mean(batch, 0, True))
                    / (torch.std(batch, 0, True) + 1e-6)
                ).detach()
        else:
            samples = batch

        samples = samples.reshape(len(samples), self.model.n_visible)

        # Performs the Gibbs sampling procedure
        _, _, _, _, visible_states = self.model.gibbs_sampling(samples)
        visible_states = visible_states.detach()

        loss = torch.mean(self.model.energy(samples)) - torch.mean(
            self.model.energy(visible_states)
        )

        batch_mse = torch.div(
            torch.sum(torch.pow(samples - visible_states, 2)), batch.shape[0] 
        ).detach()
        batch_pl = self.model.pseudo_likelihood(samples).detach()
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_batch_mse',  batch_mse, sync_dist=True)
        self.log('train_batch_pl', batch_pl, sync_dist=True)        
          
        return loss
    
   
    def validation_step(self, batch, batch_idx):

        if self.previous_layers is not None:
            for mod in range(len(self.previous_layers)):
                batch = self.previous_layers[mod](batch)

        if self.model.normalize:
            samples = (
                (batch - torch.mean(batch, 0, True))
                    / (torch.std(batch, 0, True) + 1e-6)
                ).detach()
        else:
            samples = batch

        samples = samples.reshape(len(samples), self.model.n_visible)

        # Performs the Gibbs sampling procedure
        _, _, _, _, visible_states = self.model.gibbs_sampling(samples)
        visible_states = visible_states.detach()

        loss = torch.mean(self.model.energy(samples)) - torch.mean(
            self.model.energy(visible_states)
        )

        batch_mse = torch.div(
            torch.sum(torch.pow(samples - visible_states, 2)), batch.shape[0]
        ).detach()
        print(self.model.device, samples.device)
        batch_pl = self.model.pseudo_likelihood(samples).detach()
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_batch_mse',  batch_mse, sync_dist=True)
        self.log('val_batch_pl', batch_pl, sync_dist=True) 

        return loss

    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        return self(batch)


    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.decay, nesterov=self.nesterov_accel)


