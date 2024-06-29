import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch

from sit_fuse.models.encoders.ijepa_pl import IJEPA_PL
from sit_fuse.losses.iid import IID_loss
from sit_fuse.models.deep_cluster.multi_prototypes import MultiPrototypes, OutputProjection, JEPA_Seg

from torchmetrics.clustering import MutualInfoScore

import sys
import numpy as np


class IJEPA_DC(pl.LightningModule):
    #take pretrained model path, number of classes, learning rate, weight decay, and drop path as input
    def __init__(self, pretrained_model_path, num_classes, lr=1e-3, weight_decay=0, drop_path=0.1, number_heads=1):

        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.number_heads = number_heads

        #set parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.drop_path = drop_path

        #define model layers
        self.pretrained_model = IJEPA_PL.load_from_checkpoint(pretrained_model_path)
        self.pretrained_model.model.layer_dropout = 0.0
 
        #self.average_pool = nn.AvgPool1d((self.pretrained_model.embed_dim), stride=1)
        #mlp head
       
        #self.mlp_head =  MultiPrototypes(self.pretrained_model.num_tokens, 800, 1)
        #self.mlp_head =  MultiPrototypes(self.pretrained_model.num_tokens*self.pretrained_model.embed_dim, self.num_classes, self.number_heads)
        print("HERE", self.pretrained_model.patch_size, self.pretrained_model.embed_dim, self.num_classes, self.pretrained_model.num_tokens, self.pretrained_model.img_size)
        #self.mlp_head = OutputProjection(self.pretrained_model.img_size, self.pretrained_model.patch_size, self.pretrained_model.embed_dim, self.num_classes)

        self.mlp_head = JEPA_Seg()
  

        #nn.Sequential(
        #    nn.LayerNorm(self.pretrained_model.num_tokens),
        #    nn.Linear(self.pretrained_model.num_tokens, num_classes),
        #)

        #define loss
        self.criterion = IID_loss #IIDLoss(1.0, sys.float_info.epsilon)  #IID_loss
        self.rng = np.random.default_rng(None)
 
    def forward(self, x):
        print("HERE WAHT", x.mean(), x.min(), x.max(), x.std())
        x = self.pretrained_model.model(x)
        print("HERE WAHT2", x.mean(), x.min(), x.max(), x.std())
        ###x = x.flatten(start_dim=1)
        #x = self.average_pool(x) #conduct average pool like in paper
        ###x = x.squeeze(-1)
        x = self.mlp_head(x) #pass through mlp head
        return x
    
    def training_step(self, batch, batch_idx):
        y = batch
        y = self.pretrained_model.model(y)
        ###y = self.pretrained_model.model(x).flatten(start_dim=1)
        ###y2 = y.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
        ###                        y.shape[1]*y.shape[0]).reshape(y.shape[0],\
        ###                        y.shape[1])).type(y.dtype).to(y.device)
        y2 = y.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (y.shape))).type(y.dtype).to(y.device)
        y = torch.flatten(torch.moveaxis(self.mlp_head(y),1,3), start_dim=0, end_dim=-2)
        y2 = torch.flatten(torch.moveaxis(self.mlp_head(y2),1,3), start_dim=0, end_dim=-2)
        print(torch.unique(torch.argmax(y, dim=1)), "Y labels")
        print(torch.unique(torch.argmax(y2, dim=1)), "Y2 labels")
        loss = self.criterion(y,y2, lamb=1.0)[0] #calculate loss
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch
        y = self.pretrained_model.model(y)
        ###y = self.pretrained_model.model(x).flatten(start_dim=1)
        ###y2 = y.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
        ###                        y.shape[1]*y.shape[0]).reshape(y.shape[0],\
        ###                        y.shape[1])).type(y.dtype).to(y.device)
        y2 = y.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (y.shape))).type(y.dtype).to(y.device)
 
        y = torch.flatten(torch.moveaxis(self.mlp_head(y),1,3), start_dim=0, end_dim=-2)
        y2 = torch.flatten(torch.moveaxis(self.mlp_head(y2),1,3), start_dim=0, end_dim=-2)
        loss = self.criterion(y,y2, lamb=1.0)[0] #calculate loss
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

