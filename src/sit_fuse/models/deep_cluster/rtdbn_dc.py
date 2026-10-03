import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
from sit_fuse.losses.iid import IID_loss
from sit_fuse.models.deep_cluster.multi_prototypes import MultiPrototypes

'''
pytorch lightning model for temporal RTDBN + IIC clustering head.
'''
class RTDBN_DC(pl.LightningModule):
    def __init__(self, pretrained_model, num_classes, lr=1e-3,
                 weight_decay=0, number_heads=1, noise_stdev=0.01, lamb=1.0):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_model'])
        self.num_classes = num_classes
        self.number_heads = number_heads
        self.lr = lr
        self.weight_decay = weight_decay
        self.pretrained_model = pretrained_model
        self.noise_stdev = noise_stdev
        self.lamb = lamb

        n_hidden = self.pretrained_model.n_hidden[-1]
        self.mlp_head = MultiPrototypes(n_hidden, self.num_classes, self.number_heads)

        self.criterion = IID_loss
        # Seeded rng for the IIC augmented-pair noise, for reproducibility.
        self.rng = np.random.default_rng(42)

    def _encode(self, x):
        """Encodes temporal sequences to fixed-length embeddings."""
        h = x
        for model in self.pretrained_model.models:
            h = model.forward(h)  # (batch, seq_len, n_hidden_i)

        return h.mean(dim=1)

    def forward(self, x):
        x = self._encode(x)
        x = self.mlp_head(x)[0]
        return x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        y = self._encode(x)

        y2 = y.clone() + torch.from_numpy(
            self.rng.normal(0.0, self.noise_stdev, y.shape)
        ).type(y.dtype).to(y.device)

        y = self.mlp_head(y)[0]
        y2 = self.mlp_head(y2)[0]

        loss = self.criterion(y, y2, lamb=self.lamb)[0]
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        y = self._encode(x)

        y2 = y.clone() + torch.from_numpy(
            self.rng.normal(0.0, self.noise_stdev, y.shape)
        ).type(y.dtype).to(y.device)

        y = self.mlp_head(y)[0]
        y2 = self.mlp_head(y2)[0]

        loss = self.criterion(y, y2, lamb=self.lamb)[0]
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
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