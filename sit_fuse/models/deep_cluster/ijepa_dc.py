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
from ijepa_model import IJEPA_base

from sit_fuse.models.encoders.ijepa_pl import IJEPA_PL
from sit_fuse.losses.iid import IID_loss
from sit_fuse.models.deep_cluster.multi_prototypes import MultiPrototypes

import numpy as np


class IJEPA_DC(pl.LightningModule):
    #take pretrained model path, number of classes, learning rate, weight decay, and drop path as input
    def __init__(self, pretrained_model_path, num_classes, lr=1e-3, weight_decay=0, drop_path=0.1):

        super().__init__()
        self.save_hyperparameters()

        #set parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.drop_path = drop_path

        #define model layers
        self.pretrained_model = IJEPA.load_from_checkpoint(pretrained_model_path)
        self.pretrained_model.model.mode = "test"
        self.pretrained_model.model.layer_dropout = self.drop_path
        #self.average_pool = nn.AvgPool1d((self.pretrained_model.embed_dim), stride=1)
        #mlp head
        
        #self.mlp_head =  MultiPrototypes(self.pretrained_model.num_tokens, 800, 1)
        self.mlp_head =  MultiPrototypes(196*704, 800, 1)

        #nn.Sequential(
        #    nn.LayerNorm(self.pretrained_model.num_tokens),
        #    nn.Linear(self.pretrained_model.num_tokens, num_classes),
        #)

        #define loss
        self.criterion = IID_loss
        self.rng = np.random.default_rng(None)
 
    def forward(self, x):
        x = self.pretrained_model.model(x)
        x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))
        #x = self.average_pool(x) #conduct average pool like in paper
        x = x.squeeze(-1)
        x = self.mlp_head(x)[0] #pass through mlp head
        return x
    
    def training_step(self, batch, batch_idx):
        x = batch
        y = self.pretrained_model.model(x)
        y2 = y.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                y.shape[1]*y.shape[0]).reshape(y.shape[0],\
                                y.shape[1])).type(y.dtype).to(y.device)
        y = self.mlp_head(y)[0] 
        y2 = self.mlp_head(y2)[0]
        loss = self.criterion(y,y2)[0] #calculate loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y = self.pretrained_model.model(x)
        y2 = y.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                y.shape[1]*y.shape[0]).reshape(y.shape[0],\
                                y.shape[1])).type(y.dtype).to(y.device)
        y = self.mlp_head(y)[0]
        y2 = self.mlp_head(y2)[0]
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

