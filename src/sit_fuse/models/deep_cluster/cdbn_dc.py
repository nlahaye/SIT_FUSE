import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch

import math

from sit_fuse.models.encoders.dbn_pl import DBN_PL
from sit_fuse.losses.iid import IID_loss
from sit_fuse.models.deep_cluster.multi_prototypes import MultiPrototypes, OutputProjection, JEPA_Seg
from sit_fuse.models.deep_cluster.cdbn_segmentor import CDBNSegmentor

from torchmetrics.clustering import MutualInfoScore

import sys
import numpy as np


class CDBN_DC(pl.LightningModule):
    #take pretrained model path, number of classes, learning rate, weight decay, and drop path as input
    def __init__(self, pretrained_model, num_classes, lr=1e-3, weight_decay=0, drop_path=0.1, number_heads=1):

        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.number_heads = number_heads

        #set parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.drop_path = drop_path

        #define model layers
        self.pretrained_model = pretrained_model

        feature_maps = [i for i in range(len(pretrained_model.model.feature_maps))] + 1
        self.segmentor = CDBNSegmentor(num_classes, feature_maps, self.pretrained_model)
        self.mlp_head = self.segmentor.seg_head
  
        #self.average_pool = nn.AvgPool1d((self.pretrained_model.embed_dim), stride=1)
        #mlp head
       
        #self.mlp_head =  MultiPrototypes(self.pretrained_model.num_tokens, 800, 1)
        #self.mlp_head =  MultiPrototypes(self.pretrained_model.num_tokens*self.pretrained_model.embed_dim, self.num_classes, self.number_heads)
        #self.mlp_head = OutputProjection(self.pretrained_model.img_size, self.pretrained_model.patch_size, self.pretrained_model.embed_dim, self.num_classes)
        #self.mlp_head = MultiPrototypes(3072*16, 800, 1, single=True)  
        #self.mlp_head = JEPA_Seg(num_classes)
          


        #nn.Sequential(
        #    nn.LayerNorm(self.pretrained_model.num_tokens),
        #    nn.Linear(self.pretrained_model.num_tokens, num_classes),
        #)

        #define loss
        self.criterion = IID_loss #IIDLoss(1.0, sys.float_info.epsilon)  #IID_loss
        self.rng = np.random.default_rng(None)
 
    def forward(self, x):
        self.segmentor(x)
        x = F.interpolate(
            x,
            size=(15, 15),
            mode="bilinear",
            align_corners=False,
        )  # Resize to match labels size
        x = F.softmax(x)
        return x
    
    def training_step(self, batch, batch_idx):
        y = batch

        y = self.segmentor.encoder(y)

        y2 = []
        for i in range(len(y)):


            y2.append(y[i].clone())
            y2[i] = y2[i] + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (y[i].shape))).type(y[i].dtype).to(y[i].device)

            print("INITIAL SIZES", y[i].shape, y2[i].shape, i)
            y[i] = self.segmentor.upsamples[i](y[i])
            y2[i] = self.segmentor.upsamples[i](y2[i])

            print("UPSAMPLE SIZES", y[i].shape, y2[i].shape, i)


        y = torch.cat(y, dim=1)
        y2 = torch.cat(y2, dim=1)

        print(y2.shape)

        y = self.segmentor.fusion(y)
        y2 = self.segmentor.fusion(y2)

        print(y2.shape)

        y = F.interpolate(
            y,
            size=(15, 15),
            mode="bilinear",
            align_corners=False,
        )  # Resize to match labels size

        y2 = F.interpolate(
            y2,
            size=(15, 15),
            mode="bilinear",
            align_corners=False,
        )  # Resize to match labels size


        print(y2.shape)

        y = F.softmax(self.segmentor.seg_head(y), dim=1)
        y2 = F.softmax(self.segmentor.seg_head(y2), dim=1)

        print(y2.shape)

        print(torch.unique(torch.argmax(y, dim=1)), "Y labels")
        print(torch.unique(torch.argmax(y2, dim=1)), "Y2 labels")

        y = torch.flatten(y.permute(0,2,3,1), start_dim=0, end_dim=2)
        y2 = torch.flatten(y2.permute(0,2,3,1), start_dim=0, end_dim=2) 


        print(y2.shape)

        loss = 0
        for i in range(0, y.shape[0], 100):
            i2 = i + 100
            if i2 > y.shape[0]:
                i2 = y.shape[0]
            loss = loss + self.criterion(y[i:i2],y2[i:i2], lamb=1.0)[0] #calculate loss
        loss = loss / int(math.ceil(y.shape[0] / 100))
        self.log('train_loss', loss, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        y = batch
    
        y = self.segmentor.encoder(y)

        y2 = []
        for i in range(len(y)):


            y2.append(y[i].clone())
            y2[i] = y2[i] + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (y[i].shape))).type(y[i].dtype).to(y[i].device)

            print("INITIAL SIZES", y[i].shape, y2[i].shape, i)
            y[i] = self.segmentor.upsamples[i](y[i])
            y2[i] = self.segmentor.upsamples[i](y2[i])

            print("UPSAMPLE SIZES", y[i].shape, y2[i].shape, i)


        y = torch.cat(y, dim=1)
        y2 = torch.cat(y2, dim=1)

        y = self.segmentor.fusion(y)
        y2 = self.segmentor.fusion(y2)

        y = F.interpolate(
            y,
            size=(15, 15),
            mode="bilinear",
            align_corners=False,
        )  # Resize to match labels size

        y2 = F.interpolate(
            y2,
            size=(15, 15),
            mode="bilinear",
            align_corners=False,
        )  # Resize to match labels size

        y = F.softmax(self.segmentor.seg_head(y), dim=1)
        y2 = F.softmax(self.segmentor.seg_head(y2), dim=1)

        print(torch.unique(torch.argmax(y, dim=1)), "Y labels")
        print(torch.unique(torch.argmax(y2, dim=1)), "Y2 labels")

        y = torch.flatten(y.permute(0,2,3,1), start_dim=0, end_dim=2)
        y2 = torch.flatten(y2.permute(0,2,3,1), start_dim=0, end_dim=2)

        loss = 0
        for i in range(0, y.shape[0], 100):
            i2 = i + 100
            if i2 > y.shape[0]:
                i2 = y.shape[0]
            loss = loss + self.criterion(y[i:i2],y2[i:i2], lamb=1.0)[0] #calculate loss
        loss = loss / int(math.ceil(y.shape[0] / 100))
        self.log('train_loss', loss, sync_dist=True)


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

