import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F

from sit_fuse.models.encoders.pca_encoder import PCAEncoder

from sit_fuse.losses.iid import IID_loss
from sit_fuse.models.deep_cluster.multi_prototypes import MultiPrototypes, DeepConvMultiPrototypes
import numpy as np

import joblib

import os

class PCA_DC(pl.LightningModule):
    #take pretrained model path, number of classes, learning rate, weight decay, and drop path as input
    def __init__(self, pretrained_model, num_classes, lr=1e-3, weight_decay=0, number_heads=1, conv=False):

        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_model'])
        self.num_classes = num_classes
        self.number_heads = number_heads
        self.conv = conv

        #set parameters
        self.lr = lr
        self.weight_decay = weight_decay
 
        self.pretrained_model = pretrained_model
 
        if not self.conv:
            self.mlp_head = MultiPrototypes(self.pretrained_model.pca.components_.shape[0], self.num_classes, self.number_heads)
        else:
            self.mlp_head = DeepConvMultiPrototypes(self.pretrained_model.pca.components_.shape[0], self.num_classes, self.number_heads)

        #define loss
        self.criterion = IID_loss
        self.rng = np.random.default_rng(None)
 
    def forward(self, x):
        x = torch.from_numpy(self.pretrained_model(x.cpu().numpy())).type(x.dtype).to(x.device)
        x = self.mlp_head(x)[0] #pass through mlp head
        return x
    
    def training_step(self, batch, batch_idx):
        x = batch
        y = torch.from_numpy(self.pretrained_model(x.cpu().numpy())).type(x.dtype).to(x.device)
        y2 = y.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                y.shape)).type(y.dtype).to(y.device)
        y = self.mlp_head(y)[0] 
        y2 = self.mlp_head(y2)[0]
        if self.conv:
            y = y.flatten(start_dim=2).permute(0,2,1).flatten(start_dim=0,end_dim=1)
            y2 = y2.flatten(start_dim=2).permute(0,2,1).flatten(start_dim=0,end_dim=1) 
  
        loss = self.criterion(y,y2)[0] #calculate loss
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y = torch.from_numpy(self.pretrained_model(x.cpu().numpy())).type(x.dtype).to(x.device)
        y2 = y.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                y.shape)).type(y.dtype).to(y.device)
        y = self.mlp_head(y)[0]
        y2 = self.mlp_head(y2)[0]

        if self.conv:
            y = y.flatten(start_dim=2).permute(0,2,1).flatten(start_dim=0,end_dim=1)
            y2 = y2.flatten(start_dim=2).permute(0,2,1).flatten(start_dim=0,end_dim=1)

        loss = self.criterion(y,y2)[0] #calculate loss
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx):
        return self(batch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

