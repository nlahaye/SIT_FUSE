import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from sit_fuse.datasets.simple_dataset import SimpleDataset

class SFHeirDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset,
                 batch_size=16,
                 num_workers=10,
                 pin_memory=True,
                 shuffle=False,
                 val_percent=0.1
                 ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.val_percent = val_percent
        self.dataset = dataset

    def setup(self, stage=None):
        self.n_visible = self.dataset.data_full.shape[1]
        train_max = int(self.dataset.data_full.shape[0]*(1.0-self.val_percent))
        self.train_dataset = SimpleDataset(torch.from_numpy(self.dataset.data_full[:train_max]))
        self.val_dataset = SimpleDataset(torch.from_numpy(self.dataset.data_full[train_max:]))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


