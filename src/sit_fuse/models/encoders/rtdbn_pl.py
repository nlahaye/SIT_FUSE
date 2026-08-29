import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch
import os
from learnergy.models.temporal.rt_variance_gaussian_rbm import RTVarianceGaussianRBM

'''
pytorch lightning model for temporal RTRBM encoder.
'''


class RTDBN_PL(pl.LightningModule):
    def __init__(
            self,
            model,
            save_dir,
            previous_layers=None,
            learning_rate=1e-3,
            momentum=0.95,
            nesterov_accel=True,
            decay=1e-4,
            warmup_epochs=15,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'previous_layers'])

        self.model = model
        self.save_dir = save_dir
        self.previous_layers = previous_layers
        self.lr = learning_rate
        self.momentum = momentum
        self.nesterov_accel = nesterov_accel
        self.decay = decay
        self.warmup_epochs = warmup_epochs
        self._epoch_count = 0

        self.register_module("current_rbm", self.model)
        if self.previous_layers is not None:
            for i in range(len(self.previous_layers)):
                self.register_module(
                    "previous_layer_rbm_" + str(i), self.previous_layers[i])

    def forward(self, x):
        # x: (batch, seq_len, n_visible)
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        # batch: (batch, seq_len, n_visible)
        if self.previous_layers is not None:
            for mod in range(len(self.previous_layers)):
                batch = self.previous_layers[mod](batch)

        batch, _ = batch  # unpack (sequence, target) tuple
        batch_size, seq_len, n_visible = batch.shape

        total_loss = torch.tensor(0.0, device=batch.device)
        total_mse = torch.tensor(0.0, device=batch.device)

        # Initialize recurrent hidden state from learned h0
        h_prev = self.model.h0.unsqueeze(0).expand(batch_size, -1)

        for t in range(seq_len):
            v_t = batch[:, t, :]

            _, _, _, _, visible_states = self.model.gibbs_sampling(v_t, h_prev)
            visible_states = visible_states.detach()

            loss_t = torch.mean(self.model.energy(v_t, h_prev)) - torch.mean(
                self.model.energy(visible_states, h_prev)
            )
            total_loss = total_loss + loss_t

            mse_t = torch.div(
                torch.sum(torch.pow(v_t - visible_states, 2)), batch_size
            ).detach()
            total_mse = total_mse + mse_t

            h_prev, _ = self.model.hidden_sampling(v_t, h_prev)
            h_prev = torch.nan_to_num(h_prev, nan=0.5)
            h_prev = torch.clamp(h_prev, 0.0, 1.0)

        self.log('train_loss', total_loss, sync_dist=True)
        self.log('train_batch_mse', total_mse, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        if self.previous_layers is not None:
            for mod in range(len(self.previous_layers)):
                batch = self.previous_layers[mod](batch)

        batch, _ = batch  # unpack (sequence, target) tuple
        batch_size, seq_len, n_visible = batch.shape

        total_loss = torch.tensor(0.0, device=batch.device)
        total_mse = torch.tensor(0.0, device=batch.device)

        h_prev = self.model.h0.unsqueeze(0).expand(batch_size, -1)

        for t in range(seq_len):
            v_t = batch[:, t, :]

            _, _, _, _, visible_states = self.model.gibbs_sampling(v_t, h_prev)
            visible_states = visible_states.detach()

            loss_t = torch.mean(self.model.energy(v_t, h_prev)) - torch.mean(
                self.model.energy(visible_states, h_prev)
            )
            total_loss = total_loss + loss_t

            mse_t = torch.div(
                torch.sum(torch.pow(v_t - visible_states, 2)), batch_size
            ).detach()
            total_mse = total_mse + mse_t

            h_prev, _ = self.model.hidden_sampling(v_t, h_prev)
            h_prev = torch.nan_to_num(h_prev, nan=0.5)
            h_prev = torch.clamp(h_prev, 0.0, 1.0)

        self.log('val_loss', total_loss, sync_dist=True)
        self.log('val_batch_mse', total_mse, sync_dist=True)

        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def on_train_epoch_end(self):
        self._epoch_count += 1
        if self._epoch_count == self.warmup_epochs:
            if hasattr(self.model, 'sigma'):
                self.model.sigma.requires_grad_(True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Sigma clamp: this class runs its own loop, not fit_subseries.
        if hasattr(self.model, 'sigma'):
            with torch.no_grad():
                self.model.sigma.data.clamp_(min=0.1, max=10.0)

    def on_validation_epoch_end(self):
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(
            self.save_dir, "rtdbn.ckpt"))

    def configure_optimizers(self):
        if hasattr(self.model, 'sigma'):
            self.model.sigma.requires_grad_(False)
        # Pass all parameters -- optimizer snapshots param_groups at
        # construction, so sigma must be included even while frozen.
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.decay,
            nesterov=self.nesterov_accel
        )
