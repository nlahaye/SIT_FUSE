import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from sit_fuse.datasets.simple_dataset import SimpleDataset
from sit_fuse.datasets.dataset_utils import get_train_dataset_sf

class SFDataModule(pl.LightningDataModule):
    def __init__(self,
                 yml_conf,
                 batch_size=16,
                 num_workers=10,
                 pin_memory=True,
                 shuffle=True,
                 val_percent=0.1
                 ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.val_percent = val_percent
        self.yml_conf = yml_conf

    def setup(self, stage=None):
        dataset = get_train_dataset_sf(self.yml_conf)        
        self.n_visible = dataset.data_full.shape[1]
        train_max = int(dataset.data_full.shape[0]*(1.0-self.val_percent))
        print(train_max, "HERE TRAIN MAX")

         
        if not torch.is_tensor(dataset.data_full[:train_max]):
            self.train_dataset = SimpleDataset(torch.from_numpy(dataset.data_full[:train_max]))
        else:
            self.train_dataset = SimpleDataset(dataset.data_full[:train_max])
   
        if not torch.is_tensor(dataset.data_full[train_max:]):
            self.val_dataset = SimpleDataset(torch.from_numpy(dataset.data_full[train_max:]))
        else:
            self.val_dataset = SimpleDataset(dataset.data_full[train_max:])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True
        )


