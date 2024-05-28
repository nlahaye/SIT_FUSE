import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch

from sit_fuse.losses.iid import IID_loss
from sit_fuse.models.deep_cluster.multi_prototypes import MultiPrototypes

import numpy as np


class BYOL_DC(pl.LightningModule):
    #take pretrained model path, number of classes, learning rate, weight decay, and drop path as input
    def __init__(self, pretrained_model, num_classes, lr=1e-3, weight_decay=0, number_heads=1):

        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes   
        self.number_heads = number_heads

        #set parameters
        self.lr = lr
        self.weight_decay = weight_decay

        self.pretrained_model = pretrained_model

        #define model layers
        #TODO compute
        self.mlp_head =  MultiPrototypes(4896, self.num_classes, self.number_heads)


        #define loss
        self.criterion = IID_loss
        self.rng = np.random.default_rng(None)
 
    def forward(self, x):
        x = self.pretrained_model(x)
        x = x.flatten(start_dim=1)
        #x = self.average_pool(x) #conduct average pool like in paper
        x = x.squeeze(-1)
        x = self.mlp_head(x)[0] #pass through mlp head
        return x
    
    def training_step(self, batch, batch_idx):
        x = batch
        y = self.pretrained_model(x)
        y2 = y.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                y.shape[1]*y.shape[0]).reshape(y.shape[0],\
                                y.shape[1])).type(y.dtype).to(y.device)
        y = self.mlp_head(y)[0] 
        y2 = self.mlp_head(y2)[0]
        loss = self.criterion(y,y2)[0] #calculate loss
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y = self.pretrained_model(x)
        y2 = y.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                y.shape[1]*y.shape[0]).reshape(y.shape[0],\
                                y.shape[1])).type(y.dtype).to(y.device)
        y = self.mlp_head(y)[0]
        y2 = self.mlp_head(y2)[0]
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

