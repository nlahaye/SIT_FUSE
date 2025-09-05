import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F

from sit_fuse.losses.iid import IID_loss
from sit_fuse.models.deep_cluster.multi_prototypes import MultiPrototypes, OutputProjection
from sit_fuse.utils import get_output_shape

import numpy as np


class BYOL_DC(pl.LightningModule):
    #take pretrained model path, number of classes, learning rate, weight decay, and drop path as input
    def __init__(self, pretrained_model, num_classes, lr=1e-3, weight_decay=0, number_heads=1, tile_size =5, in_chans=3, model_type = "GCN", save_dir = "."):

        super().__init__()
        self.save_dir = save_dir
        self.save_hyperparameters()
        self.num_classes = num_classes   
        self.number_heads = number_heads
        self.in_chans = in_chans
 
        #set parameters
        self.lr = lr
        self.weight_decay = weight_decay

        self.pretrained_model = pretrained_model

        #define loss
        self.criterion = IID_loss
        self.rng = np.random.default_rng(None)
 
        self.model_type = model_type

    def forward(self, x):
        if self.model_type == "Unet":
            encoder = self.pretrained_model[0]
            decoder = self.pretrained_model[1]
            x, x1, x2, x3, x4 = encoder.full_forward(x)
            x = decoder.forward(x, x1, x2, x3, x4)
            x = F.softmax(x, dim=1) #.flatten(start_dim=2).permute(0,2,1).flatten(start_dim=0,end_dim=1)
        elif self.model_type == "DCE":
            print(self.pretrained_model(x)[0].shape, x.shape, "DCE DIMS")
            x = self.pretrained_model(x)[0]
            print(x.shape)
            print(torch.unique(torch.argmax(x, dim=1)), "Y labels")
        else:
            x = F.softmax(self.pretrained_model(x), dim=1)
        return x

    def dce_training_val_step(self, batch, batch_idx):
        encoder = self.pretrained_model[0]
        decoder = self.pretrained_model[1]
 
        x = batch
        x = encoder(x)
        x2 = x.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (x.shape))).type(x.dtype).to(x.device)

        x = decoder(x)[0]
        x2 = decoder(x2)[0]

        #x = F.softmax(x, dim=1) #.flatten(start_dim=2).permute(0,2,1).flatten(start_dim=0,end_dim=1)
        #x2 = F.softmax(x2, dim=1) #.flatten(start_dim=2).permute(0,2,1).flatten(start_dim=0,end_dim=1)

        print(x.shape, x2.shape, "BYOL DC SHAPE")
        x = x.permute(0,2,3,1).flatten(start_dim=0,end_dim=2)
        x2 = x2.permute(0,2,3,1).flatten(start_dim=0,end_dim=2)


        print(torch.unique(torch.argmax(x, dim=1)), "Y labels")
        print(torch.unique(torch.argmax(x2, dim=1)), "Y2 labels")

        loss = self.criterion(x,x2, lamb=2.0)[0] #calculate loss
        return loss

    def unet_training_val_step(self, batch, batch_idx):

        encoder = self.pretrained_model[0]
        decoder = self.pretrained_model[1]

        x = batch
        x, x1, x2, x3, x4 = encoder.full_forward(x)

        x_2 = x.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (x.shape))).type(x.dtype).to(x.device)    
        x2_2 = x2.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (x2.shape))).type(x2.dtype).to(x2.device)
        x1_2 = x1.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (x1.shape))).type(x1.dtype).to(x1.device)
        x3_2 = x3.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (x3.shape))).type(x3.dtype).to(x3.device)
        x4_2 = x4.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (x4.shape))).type(x4.dtype).to(x4.device)


        x = decoder.forward(x, x1, x2, x3, x4)
        x_2 = decoder.forward(x_2, x1_2, x2_2, x3_2, x4_2)

        x = F.softmax(x, dim=1).flatten(start_dim=2).permute(0,2,1).flatten(start_dim=0,end_dim=1)
        x_2 = F.softmax(x_2, dim=1).flatten(start_dim=2).permute(0,2,1).flatten(start_dim=0,end_dim=1)

        

        print(torch.unique(torch.argmax(x, dim=1)), "Y labels")
        print(torch.unique(torch.argmax(x_2, dim=1)), "Y2 labels")

        loss = self.criterion(x,x_2, lamb=1.0)[0] #calculate loss
        return loss


    #TODO update fork of pytorch-segmentation with a uniform interface to use here. Until then, this.
    def deeplab_training_val_step(self, batch, batch_idx):
        x = batch
        H, W = x.size(2), x.size(3)
        x, low_level_features = self.pretrained_model.backbone(x)
 

        x2 = x.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (x.shape))).type(x.dtype).to(x.device)
        x2_l = low_level_features.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (low_level_features.shape))).type(low_level_features.dtype).to(low_level_features.device)

        x = self.pretrained_model.ASSP(x)
        x = self.pretrained_model.decoder(x, low_level_features)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        x2 = self.pretrained_model.ASSP(x2)
        x2 = self.pretrained_model.decoder(x2, x2_l)
        x2 = F.interpolate(x2, size=(H, W), mode='bilinear', align_corners=True)


        x = F.softmax(x, dim=1).flatten(start_dim=2).permute(0,2,1).flatten(start_dim=0,end_dim=1)
        x2 = F.softmax(x2, dim=1).flatten(start_dim=2).permute(0,2,1).flatten(start_dim=0,end_dim=1)

        print(torch.unique(torch.argmax(x, dim=1)), "Y labels")
        print(torch.unique(torch.argmax(x2, dim=1)), "Y2 labels")
 
        loss = self.criterion(x,x2, lamb=1.0)[0] #calculate loss
        return loss


    def gcn_training_val_step(self, batch, batch_idx):
        x = batch
        H, W = x.size(2), x.size(3)

        x1, x2, x3, x4, conv1_sz = self.pretrained_model.backbone(x)
        
        x1_2 = x1.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (x1.shape))).type(x1.dtype).to(x1.device)
        x2_2 = x2.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (x2.shape))).type(x2.dtype).to(x2.device)
        x3_2 = x3.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (x3.shape))).type(x3.dtype).to(x3.device)
        x4_2 = x4.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (x4.shape))).type(x4.dtype).to(x4.device)

        x1 = self.pretrained_model.br1(self.pretrained_model.gcn1(x1))
        x2 = self.pretrained_model.br2(self.pretrained_model.gcn2(x2))
        x3 = self.pretrained_model.br3(self.pretrained_model.gcn3(x3))
        x4 = self.pretrained_model.br4(self.pretrained_model.gcn4(x4))
    
        x1_2 = self.pretrained_model.br1(self.pretrained_model.gcn1(x1_2))
        x2_2 = self.pretrained_model.br2(self.pretrained_model.gcn2(x2_2))
        x3_2 = self.pretrained_model.br3(self.pretrained_model.gcn3(x3_2))
        x4_2 = self.pretrained_model.br4(self.pretrained_model.gcn4(x4_2))



        if self.pretrained_model.use_deconv:
            # Padding because when using deconv, if the size is odd, we'll have an alignment error
            x4 = self.pretrained_model.decon4(x4)
            if x4.size() != x3.size(): x4 = self.pretrained_model._pad(x4, x3)
            x3 = self.pretrained_model.decon3(self.pretrained_model.br5(x3 + x4))
            if x3.size() != x2.size(): x3 = self.pretrained_model._pad(x3, x2)
            x2 = self.pretrained_model.decon2(self.pretrained_model.br6(x2 + x3))
            if x2.size() != x1.size(): x2 = self.pretrained_model._pad(x2, x1)
            x1 = self.pretrained_model.decon1(self.pretrained_model.br7(x1 + x2))

            x = self.pretrained_model.br9(self.pretrained_model.decon5(self.pretrained_model.br8(x1)))


            x4_2 = self.pretrained_model.decon4(x4_2)
            if x4_2.size() != x3_2.size(): x4_2 = self.pretrained_model._pad(x4_2, x3_2)
            x3_2 = self.pretrained_model.decon3(self.pretrained_model.br5(x3_2 + x4_2))
            if x3_2.size() != x2_2.size(): x3_2 = self.pretrained_model._pad(x3_2, x2_2)
            x2_2 = self.pretrained_model.decon2(self.pretrained_model.br6(x2_2 + x3_2))
            if x2_2.size() != x1_2.size(): x2_2 = self.pretrained_model._pad(x2_2, x1_2)
            x1_2 = self.pretrained_model.decon1(self.pretrained_model.br7(x1_2 + x2_2))

            x_2 = self.pretrained_model.br9(self.pretrained_model.decon5(self.pretrained_model.br8(x1_2)))


        else:
 
            x4 = F.interpolate(x4, size=x3.size()[2:], mode='bilinear', align_corners=True)
            x3 = F.interpolate(self.pretrained_model.br5(x3 + x4), size=x2.size()[2:], mode='bilinear', align_corners=True)
            x2 = F.interpolate(self.pretrained_model.br6(x2 + x3), size=x1.size()[2:], mode='bilinear', align_corners=True)
            x1 = F.interpolate(self.pretrained_model.br7(x1 + x2), size=conv1_sz, mode='bilinear', align_corners=True)

            x = self.pretrained_model.br9(F.interpolate(self.pretrained_model.br8(x1), size=x.size()[2:], mode='bilinear', align_corners=True))
 
            x4_2 = F.interpolate(x4_2, size=x3_2.size()[2:], mode='bilinear', align_corners=True)
            x3_2 = F.interpolate(self.pretrained_model.br5(x3_2 + x4_2), size=x2_2.size()[2:], mode='bilinear', align_corners=True)
            x2_2 = F.interpolate(self.pretrained_model.br6(x2_2 + x3_2), size=x1_2.size()[2:], mode='bilinear', align_corners=True)
            x1_2 = F.interpolate(self.pretrained_model.br7(x1_2 + x2_2), size=conv1_sz, mode='bilinear', align_corners=True)

            x_2 = self.pretrained_model.br9(F.interpolate(self.pretrained_model.br8(x1_2), \
                size=x.size()[2:], mode='bilinear', align_corners=True))
 
        x = F.softmax(self.pretrained_model.final_conv(x), dim=1).flatten(start_dim=2).permute(0,2,1).flatten(start_dim=0,end_dim=1)
        x_2 = F.softmax(self.pretrained_model.final_conv(x_2), dim=1).flatten(start_dim=2).permute(0,2,1).flatten(start_dim=0,end_dim=1)



        loss = self.criterion(x,x_2, lamb=1.0)[0] #calculate loss
        return loss


    def training_step(self, batch, batch_idx):

        loss = None
        if self.model_type == "GCN":
            loss = self.gcn_training_val_step(batch, batch_idx)
        elif self.model_type == "DeepLab":
            loss = self.deeplab_training_val_step(batch, batch_idx)
        elif self.model_type == "Unet":
            loss = self.unet_training_val_step(batch, batch_idx)
        elif self.model_type == "DCE":
            loss = self.dce_training_val_step(batch, batch_idx)

        self.log('train_loss', loss, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):

        loss = None
        if self.model_type == "GCN":
            loss = self.gcn_training_val_step(batch, batch_idx)
        elif self.model_type == "DeepLab":
            loss = self.deeplab_training_val_step(batch, batch_idx)
        elif self.model_type == "Unet":
            loss = self.unet_training_val_step(batch, batch_idx)
        elif self.model_type == "DCE":
            loss = self.dce_training_val_step(batch, batch_idx)

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

