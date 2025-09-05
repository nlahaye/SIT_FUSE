import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch

import math

import torch.nn.functional as F

from sit_fuse.losses.iid import IID_loss
from sit_fuse.models.deep_cluster.clay_segmentor import Segmentor

import sys
import numpy as np


class Clay_DC(pl.LightningModule):
    #take pretrained model path, number of classes, learning rate, weight decay, and drop path as input
    def __init__(self, pretrained_model_path, num_classes, feature_maps, waves, gsd, lr=1e-3, weight_decay=0):

        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes

        self.gsd = gsd
        self.waves = torch.FloatTensor(waves)

        #set parameters
        self.lr = lr
        self.weight_decay = weight_decay

        #define model layers
        print(pretrained_model_path)
        self.pretrained_model = Segmentor(num_classes, feature_maps, pretrained_model_path)
        #self.pretrained_model.model.layer_dropout = 0.0
        self.mlp_head = self.pretrained_model.seg_head 

        #define loss
        self.criterion = IID_loss #IIDLoss(1.0, sys.float_info.epsilon)  #IID_loss
        self.rng = np.random.default_rng(None)
 
    def forward(self, x):

        waves = self.waves
        gsd = self.gsd
        tile_size = x.shape

        #print(x.shape, "HERE Clay")
        dat_final = {
            "pixels": x,
            "latlon": torch.zeros((x.shape[0], 4)),
            "time": torch.zeros((x.shape[0], 4)),
            "gsd": self.gsd,
            "waves": self.waves}
            

        y = self.pretrained_model.encoder(dat_final)
        y2 = []

        #print(len(y), len(self.pretrained_model.upsamples), x.shape)
        mn_tile_size = 99999
        mx_tile_size = -1
        for i in range(len(y)):
            #print(y[i].shape)
            y2.append(self.pretrained_model.upsamples[i](y[i]))
            mn_tile_size = min(mn_tile_size, y2[i].shape[-1])
            mx_tile_size = max(mx_tile_size, y2[i].shape[-1])

        if mx_tile_size > mn_tile_size:
            for i in range(len(y)):

                    y2[i] = F.interpolate(
                        y2[i],
                        size=(mx_tile_size, mx_tile_size),
                        mode="bilinear",
                        align_corners=False,
                    )



        y2 = torch.cat(y2, dim=1)
        y2 = self.pretrained_model.fusion(y2)
         
        y2 = F.interpolate(
            y2,
            size=(tile_size[-2], tile_size[-1]),
            mode="bilinear",
            align_corners=False,
        )  # Resize to match labels size

        y2 = F.softmax(self.pretrained_model.seg_head(y2), dim=1)

        return y2
    
    def training_step(self, batch, batch_idx):
        y = batch
        tile_size = y.shape

        #print(y[0].shape, "HERE Clay")
        dat_final = {
            "pixels": y, #[0],
            #"indices": y[1],
            "latlon": torch.zeros((y.shape[0], 4)),
            "time": torch.zeros((y.shape[0], 4)),
            "gsd": self.gsd,
            "waves": self.waves}
        y = self.pretrained_model.encoder(dat_final)

        y2 = []
        mn_tile_size = 99999
        mx_tile_size = -1
        for i in range(len(y)):

            y2.append(y[i].clone())
            y2[i] = y2[i] + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (y[i].shape))).type(y[i].dtype).to(y[i].device)
            print("INITIAL SIZES", y[i].shape, i)
            y[i] = self.pretrained_model.upsamples[i](y[i])
            y2[i] = self.pretrained_model.upsamples[i](y2[i])
            print("UPSAMPLE SIZES", y[i].shape, y2[i].shape, i)
            mn_tile_size = min(mn_tile_size, y[i].shape[-1])
            mx_tile_size = max(mx_tile_size, y[i].shape[-1])

        if mx_tile_size > mn_tile_size:
            for i in range(len(y)):
                if y[i].shape[-1] < mx_tile_size:
                    y[i] = F.interpolate(
                        y[i],
                        size=(mx_tile_size, mx_tile_size),
                        mode="bilinear",
                        align_corners=False,
                    )

                    y2[i] = F.interpolate(
                        y2[i],
                        size=(mx_tile_size, mx_tile_size),
                        mode="bilinear",
                        align_corners=False,
                    )

        y = torch.cat(y, dim=1)
        y2 = torch.cat(y2, dim=1)

        y = self.pretrained_model.fusion(y)
        y2 = self.pretrained_model.fusion(y2)

        y = F.interpolate(
            y,
            size=(tile_size[-2], tile_size[-1]),
            mode="bilinear",
            align_corners=False,
        )  # Resize to match labels size

        y2 = F.interpolate(
            y2,
            size=(tile_size[-2], tile_size[-1]),
            mode="bilinear",
            align_corners=False,
        )  # Resize to match labels size

        y = F.softmax(self.pretrained_model.seg_head(y), dim=1)
        y2 = F.softmax(self.pretrained_model.seg_head(y2), dim=1)

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


 
    def validation_step(self, batch, batch_idx):
        y = batch

        tile_size = y.shape

        print(y[0].shape, "HERE Clay")
        dat_final = {
            "pixels": y, #[0],
            #"indices": y[1],
            "latlon": torch.zeros((y.shape[0],4)),
            "time": torch.zeros((y.shape[0],4)),
            "gsd": self.gsd,   
            "waves": self.waves}

        y = self.pretrained_model.encoder(dat_final)

        y2 = [] 
        mn_tile_size = 99999
        mx_tile_size = -1
        for i in range(len(y)):

            y2.append(y[i].clone())
            y2[i] = y2[i] + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (y[i].shape))).type(y[i].dtype).to(y[i].device)
            print("INITIAL SIZES", y[i].shape, i)
            y[i] = self.pretrained_model.upsamples[i](y[i])
            y2[i] = self.pretrained_model.upsamples[i](y2[i])
            print("UPSAMPLE SIZES", y[i].shape, y2[i].shape, i)
            mn_tile_size = min(mn_tile_size, y[i].shape[-1])
            mx_tile_size = max(mx_tile_size, y[i].shape[-1])

        if mx_tile_size > mn_tile_size:
            for i in range(len(y)):
                if y[i].shape[-1] < mx_tile_size:
                    y[i] = F.interpolate(
                        y[i],
                        size=(mx_tile_size, mx_tile_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    
                    y2[i] = F.interpolate(
                        y2[i],
                        size=(mx_tile_size, mx_tile_size),
                        mode="bilinear",
                        align_corners=False,
                    )


        y = torch.cat(y, dim=1)
        y2 = torch.cat(y2, dim=1)


        print(y.shape, y2.shape, "PRE FUSION")
        y = self.pretrained_model.fusion(y)
        y2 = self.pretrained_model.fusion(y2)

        print(y.shape, y2.shape, "POST FUSION")

        y = F.interpolate(
            y,
            size=(tile_size[-2], tile_size[-1]),
            mode="bilinear",
            align_corners=False,
        )  # Resize to match labels size

        y2 = F.interpolate(
            y2,
            size=(tile_size[-2], tile_size[-1]),
            mode="bilinear",
            align_corners=False,
        )  # Resize to match labels size

        y = F.softmax(self.pretrained_model.seg_head(y), dim=1)
        y2 = F.softmax(self.pretrained_model.seg_head(y2), dim=1) 

        y = torch.flatten(y.permute(0,2,3,1), start_dim=0, end_dim=2)
        y2 = torch.flatten(y2.permute(0,2,3,1), start_dim=0, end_dim=2)

        loss = 0
        for i in range(0, y.shape[0], 100):
            i2 = i + 100
            if i2 > y.shape[0]:
                i2 = y.shape[0]
            loss = loss + self.criterion(y[i:i2],y2[i:i2], lamb=1.0)[0] #calculate loss
        loss = loss / int(math.ceil(y.shape[0] / 100))
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

