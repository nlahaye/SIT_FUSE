import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger

from sit_fuse.losses.iid import IID_loss
from sit_fuse.models.deep_cluster.multi_prototypes import DeepConvMultiPrototypes
import numpy as np

class DeepCluster(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-3, weight_decay=0, drop_path=0.1, conv=False):

        super().__init__()
        self.save_hyperparameters()

        #set parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.drop_path = drop_path

        self.conv = conv

        if conv:
            self.mlp_head =  DeepConvMultiPrototypes(34, 300, 1)

        else:
            self.mlp_head =  DeepMultiPrototypes(34*9, 800, 1)

     

        #define loss
        self.criterion = IID_loss
        self.rng = np.random.default_rng(None)
 
    def forward(self, x):
        #x = self.average_pool(x) #conduct average pool like in paper
        x = x.squeeze(-1)
        x = self.mlp_head(x)[0] #pass through mlp head
        return x
   
    def training_step(self, batch, batch_idx):
        x = batch

        if self.conv:
            x2 = x.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                x.shape[3]*x.shape[2]).reshape(x.shape[2],\
                                x.shape[3])).type(x.dtype).to(x.device)
            y = self(x).flatten(start_dim=2).permute(1,0,2).flatten(start_dim=1)
            y2 = self(x2).flatten(start_dim=2).permute(1,0,2).flatten(start_dim=1)
        else:
            x2 = x.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                x.shape[1]*x.shape[0]).reshape(x.shape[0],\
                                x.shape[1])).type(x.dtype).to(x.device)
            y = self(x).flatten(start_dim=1)
            y2 = self(x2).flatten(start_dim=1)
        loss = self.criterion(y,y2)[0] #calculate loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        if self.conv:
            x2 = x.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                x.shape[3]*x.shape[2]).reshape(x.shape[2],\
                                x.shape[3])).type(x.dtype).to(x.device)
            y = self(x).flatten(start_dim=2).permute(1,0,2).flatten(start_dim=1)
            y2 = self(x2).flatten(start_dim=2).permute(1,0,2).flatten(start_dim=1)
        else:
            x2 = x.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                x.shape[1]*x.shape[0]).reshape(x.shape[0],\
                                x.shape[1])).type(x.dtype).to(x.device)
            y = self(x).flatten(start_dim=1)
            y2 = self(x2).flatten(start_dim=1)
        loss = self.criterion(y,y2)[0] #calculate loss
        self.log('val_loss', loss)
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





