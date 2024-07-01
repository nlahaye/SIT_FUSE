import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch

from vit_pytorch import ViT, MAE

'''
pytorch lightning model
'''
class MAE_PL(pl.LightningModule):
    def __init__(
            self,
            img_size=64,
            in_chans = 34,
            patch_size=4,
            dim=1024,
            enc_heads=8,
            enc_depth=8,
            lr=1e-6,
            weight_decay=0.05,
            masking_ratio = 0.75, 
            decoder_dim = 512,
            decoder_depth = 6):
        super().__init__()
        self.save_hyperparameters()
    

        self.vit = ViT(image_size=img_size, channels = in_chans, patch_size=patch_size, dim=dim, depth=enc_depth, heads=enc_heads, num_classes = 1000, mlp_dim=2048) 
        self.learner = MAE(encoder=self.vit, masking_ratio=masking_ratio, decoder_dim=decoder_dim, decoder_depth=decoder_depth)    

        #define hyperparameters
        self.masking_ratio = masking_ratio
        self.lr = lr
        self.weight_decay = weight_decay
        self.decoder_dim = decoder_dim
        self.dim = dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.decoder_depth = decoder_depth

        #define loss
        self.criterion = self.learner
    
    def forward(self, x, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale):
        return self.learner(x)
    

    def training_step(self, batch, batch_idx):
        x = batch
        loss = self.learner(batch)
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        print(x.shape)
        loss = self.learner(batch)
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

