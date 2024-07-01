import torch

import pytorch_lightning as pl
from pixel_level_contrastive_learning import PixelCL

import os

class BYOL_Learner(pl.LightningModule):
    def __init__(self, save_dir, net, lr=1e-6,
            weight_decay=0.05, **kwargs):
        super().__init__()
        self.lr = lr
        self.save_dir = save_dir
        self.weight_decay = weight_decay
        self.model = net
        self.learner = PixelCL(self.model, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, images, _):
        loss = self.forward(images)
        self.log('val_loss', loss, sync_dist=True)
        return loss

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


    def on_before_zero_grad(self, _):
        #if self.learner.use_momentum:
        self.learner.update_moving_average()



    def on_validation_epoch_end(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "byol.ckpt"))

