import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch

from sit_fuse.losses.iid import IID_loss

from sit_fuse.models.deep_cluster.ijepa_dc import IJEPA_DC
from sit_fuse.models.deep_cluster.dbn_dc import DBN_DC
from sit_fuse.models.deep_cluster.dc import DeepCluster

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


import numpy as np


#TODO add back in multi-layer heir
class Heir_DC(pl.LightningModule):
    #take pretrained model path, number of classes, learning rate, weight decay, and drop path as input
    def __init__(self, data, pretrained_model_path, num_classes, lr=1e-3, weight_decay=0, encoder_type="dbn", number_heads=1, min_samples=1000):

        super().__init__()
        self.save_hyperparameters(ignore=['data'])

        self.min_samples = min_samples

        #set parameters
        self.lr = lr
        self.weight_decay = weight_decay


        self.number_heads = number_heads
        #define model layers
        self.pretrained_model = None
        if encoder_type == "dbn":
            self.pretrained_model = DBN_DC.load_from_checkpoint(pretrained_model_path)
        elif encoder_type == "ijepa":
            self.pretrained_model = IJEPA_DC.load_from_checkpoint(pretrained_model_path)
        else:
            self.pretrained_model = DeepCluster.load_from_checkpoint(pretrained_model_path)

        self.encoder_type = encoder_type
        self.pretrained_model.pretrained_model.model.mode = "test"
        self.num_classes = num_classes       

        self.clust_tree = {"0": {"-1": self.pretrained_model}, "1": {}}

        #define loss
        self.criterion = IID_loss
        self.rng = np.random.default_rng(None)

        self.lab_full = {}

        self.generate_label_set(data)
        del data

    def generate_label_set(self, data):

        count = 0
        self.lab_full = {}
        batch_size = max(700, self.min_samples)

        output_sze = data.data_full.shape[0]
        append_remainder = int(batch_size - (output_sze % batch_size))

        if isinstance(data.data_full,torch.Tensor):
            data.data_full = torch.cat((data.data_full,data.data_full[0:append_remainder]))
            data.targets_full = torch.cat((data.targets_full,data.targets_full[0:append_remainder]))
        else:
            data.data_full = np.concatenate((data.data_full,data.data_full[0:append_remainder]))
            data.targets_full = np.concatenate((data.targets_full,data.targets_full[0:append_remainder]))

        test_loader = DataLoader(data, batch_size=batch_size, shuffle=False, \
        num_workers = 0, drop_last = False, pin_memory = False)
        ind = 0
        ind2 = 0

        for data2 in tqdm(test_loader):
            dat_dev, lab_dev = data2[0].to(device=self.pretrained_model.device, non_blocking=True), data2[1].to(device=self.pretrained_model.device, non_blocking=True)
            dev_ds = TensorDataset(dat_dev, lab_dev)

            lab = self.pretrained_model.forward(dat_dev)
            if isinstance(lab, list):
                lab = lab[0]
            #If previous layer is top layer / otherwise argmax happens in forward function
            if lab.shape[1] > 1:
                lab = torch.argmax(lab, axis = 1)
                lab = lab.detach().cpu()
            dat_dev = dat_dev.detach().cpu()
            lab_dev = lab_dev.detach().cpu()
            del dev_ds


            ind1 = ind2
            ind2 += dat_dev.shape[0]
            if ind2 > data.data.shape[0]:
                ind2 = data.data.shape[0]

            lab_unq = torch.unique(lab)
            for l in lab_unq:
                inds = torch.where(lab == l)
                key = str(l.detach().cpu().numpy())
                if key in self.lab_full.keys():
                    self.lab_full[key] = torch.cat((self.lab_full[key],(inds[0] + ind1)))
                else:
                    self.lab_full[key] = inds[0] + ind1

            ind = ind + 1
            count = count + 1
            del dat_dev
            del lab_dev

 
    def forward(self, x, perturb = False):
        #TODO fix for multi-head
        dt = x.dtype
        y = self.pretrained_model.forward(x)

        if hasattr(self.pretrained_model.pretrained_model, 'model'):
            x = self.pretrained_model.pretrained_model.model(x).flatten(start_dim=1)
        else:
            x = self.pretrained_model.pretrained_model(x)

        if isinstance(y,tuple):
            y = y[0]

        if isinstance(y,list):
            y = y[0]

 
        if perturb:
            y = y + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                y.shape[1]*y.shape[0]).reshape(y.shape[0],\
                                y.shape[1])).type(y.dtype).to(y.device)
 
        tmp_full = torch.zeros((y.shape[0], 1), device=y.device, dtype=torch.int64)
        tmp = y
        if y.ndim > 1 and y.shape[1] > 1:
            tmp = torch.argmax(y, dim=1)
        f = lambda x: str(x)
        tmp2 = np.vectorize(f)(tmp.detach().cpu())
        tmp3 = tmp
        keys = np.unique(tmp2)
        for key in keys:
            inds = np.where(tmp2 == key)
            if key in self.clust_tree["1"].keys() and self.clust_tree["1"][key] is not None:
                tmp = self.clust_tree["1"][key].forward(x[inds]) #torch.unsqueeze(x[inds],dim=0))
                if isinstance(tmp,tuple):
                    tmp = tmp[0]
                if isinstance(tmp,list):
                    tmp = tmp[0]

                tmp = torch.unsqueeze(torch.argmax(tmp, dim=1), dim=1)
                tmp[:,0] = tmp[:,0] + (self.n_classes*tmp3[inds[0]])
            else:
                tmp = torch.unsqueeze((self.n_classes*tmp3[inds[0]]), dim=1)
            tmp_full[inds] = tmp

        return tmp_full

    
    def training_step(self, batch, batch_idx):
        x = batch
        y = self.forward(x)[0]
        y2 = self.forward(x.clone(), perturb=True)[0]
        loss = self.criterion(y,y2)[0] #calculate loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch 
        y = self.forward(x)[0]
        y2 = self.forward(x.clone(), perturb=True)[0]
        loss = self.criterion(y,y2)[0] #calculate loss
        self.log('val_loss', loss)
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

