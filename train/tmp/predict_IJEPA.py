import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
from ijepa_model import IJEPA_base

from pretrain_IJEPA import IJEPA

from losses.iid import IID_loss
from models.deep_cluster.multi_prototypes import MultiPrototypes
import numpy as np

class IJEPADataset(Dataset):
    def __init__(self,
                 data,
                 ):
        super().__init__()
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


'''Placeholder for datamodule in pytorch lightning'''
'''
Placeholder for datamodule in pytorch lightning
'''
class D2VDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_path,
                 batch_size=16,
                 num_workers=4,
                 pin_memory=True,
                 shuffle=True
                 ):
        super().__init__()
        
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        
    def setup(self, stage=None):
        data = np.load("/data/nlahaye/output/Learnergy/DBN_eMAS_FULL_STRAT_Segformer_DINO/train_data.npy")
        train_max = int(data.shape[0]*0.9)
        print(train_max, data.shape)
        self.train_dataset = IJEPADataset(torch.from_numpy(data[:train_max]))
        self.val_dataset = None #IJEPADataset(torch.from_numpy(data[train_max:]))
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )
    
    def val_dataloader(self):
        return None #DataLoader(
        #    self.val_dataset,
        #    batch_size=self.batch_size,
        #    num_workers=self.num_workers,
        #    shuffle=False,
        #)




'''
Finetune IJEPA
'''
class IJEPA_FT(pl.LightningModule):
    #take pretrained model path, number of classes, learning rate, weight decay, and drop path as input
    def __init__(self, pretrained_model_path, num_classes, lr=1e-3, weight_decay=0, drop_path=0.1):

        super().__init__()
        self.save_hyperparameters()

        #set parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.drop_path = drop_path

        #define model layers
        self.pretrained_model = IJEPA.load_from_checkpoint(pretrained_model_path)
        self.pretrained_model.model.mode = "test"
        self.pretrained_model.model.layer_dropout = self.drop_path
        #self.average_pool = nn.AvgPool1d((self.pretrained_model.embed_dim), stride=1)
        #mlp head
        
        #self.mlp_head =  MultiPrototypes(self.pretrained_model.num_tokens, 800, 1)
        self.mlp_head =  MultiPrototypes(196*64, 800, 1)

        #nn.Sequential(
        #    nn.LayerNorm(self.pretrained_model.num_tokens),
        #    nn.Linear(self.pretrained_model.num_tokens, num_classes),
        #)

        #define loss
        self.criterion = IID_loss
        self.rng = np.random.default_rng(None)
 
    def forward(self, x):
        x = self.pretrained_model.model(x)
        x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))
        #x = self.average_pool(x) #conduct average pool like in paper
        x = x.squeeze(-1)
        x = self.mlp_head(x)[0] #pass through mlp head
        return x
    
    def training_step(self, batch, batch_idx):
        x = batch
        y = self(x)
        y2 = y.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                y.shape[1]*y.shape[0]).reshape(y.shape[0],\
                                y.shape[1])).type(y.dtype).to(y.device)
        loss = self.criterion(y,y2)[0] #calculate loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y = self(x)
        y2 = y.clone() + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                y.shape[1]*y.shape[0]).reshape(y.shape[0],\
                                y.shape[1])).type(y.dtype).to(y.device)
        loss = self.criterion(y,y2)[0]
        self.log('val_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        return self(batch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

if __name__ == '__main__':

    dataset = D2VDataModule(dataset_path='data')

    model = IJEPA_FT(pretrained_model_path="/data/nlahaye/output/Learnergy/IJEPA_TEST/ijepa_and_decoder.ckpt", num_classes=800)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    #wandb_logger = WandbLogger(project="SIT-FUSE", log_model=True, save_dir = "/home/nlahaye/SIT_FUSE_DEV/wandb_finetune_xavier_batch_norm/")
 
    trainer = pl.Trainer()


    out = trainer.predict(model, dataset.train_dataloader())
    print(out.shape)
    out = out.detach().cpu().numpy()
    disc_data = np.argmax(out, axis = 1)
    print("UNIQUE", np.unique(disc_data).shape)

    np.save("/data/nlahaye/output/Learnergy/IJEPA_TEST/train_out.npy")
 



