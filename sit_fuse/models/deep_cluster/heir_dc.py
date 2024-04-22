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
    def __init__(self, data, pretrained_model_path, num_classes, lr=1e-3, weight_decay=0, encoder_type=None, number_heads=1, min_samples=1000):

        super().__init__()
        self.save_hyperparameters(ignore=['data'])

        self.min_samples = min_samples

        #set parameters
        self.lr = lr
        self.weight_decay = weight_decay


        self.number_heads = number_heads
        #define model layers
        self.pretrained_model = None
        if encoder_type is None:
            self.pretrained_model = DeepCluster.load_from_checkpoint(pretrained_model_path, img_size=3, in_chans=34) #Why arent these being saved
        elif encoder_type == "dbn":
            self.pretrained_model = DBN_DC.load_from_checkpoint(pretrained_model_path)
        elif encoder_type == "ijepa":
            self.pretrained_model = IJEPA_DC.load_from_checkpoint(pretrained_model_path)
            self.pretrained_model.pretrained_model.model.mode = "test"
        else:
            self.pretrained_model = DeepCluster.load_from_checkpoint(pretrained_model_path) #Why arent these being saved

        self.encoder_type = encoder_type
        self.num_classes = num_classes       

        self.clust_tree = {"0": {"-1": self.pretrained_model}, "1": {}}

        #define loss
        self.criterion = IID_loss
        self.rng = np.random.default_rng(None)

        self.lab_full = {}

        self.module_list = nn.ModuleList([self.pretrained_model])
 
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
                print("LABS", len(inds), inds[0])
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
 
        if hasattr(self.pretrained_model, 'pretrained_model'):
            if hasattr(self.pretrained_model.pretrained_model, 'model'):
                x = self.pretrained_model.pretrained_model.model(x).flatten(start_dim=1)
            else:
                x = self.pretrained_model.pretrained_model.forward(x).flatten(start_dim=1) 
        else:
            if hasattr(self, 'encoder') and self.encoder is not None:
                x = self.encoder(x).flatten(start_dim=1)
            else:
                modules = list(self.pretrained_model.model.children())[:-2]
                self.encoder = nn.Sequential(*modules)
                self.encoder.eval()
                x = self.encoder(x).flatten(start_dim=1)

        if isinstance(y,tuple):
            y = y[0]

        if isinstance(y,list):
            y = y[0]

 
        if perturb:
            x = x + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                x.shape[1]*x.shape[0]).reshape(x.shape[0],\
                                x.shape[1])).type(x.dtype).to(x.device)
 
        tmp_full = torch.zeros((y.shape[0], 1), device=y.device, dtype=torch.int64)
        tmp_subset = None
        tmp = y
        if y.ndim > 1 and y.shape[1] > 1:
            tmp = torch.argmax(y, dim=1)
        f = lambda z: str(z)
        tmp2 = np.vectorize(f)(tmp.detach().cpu())
        tmp3 = tmp
        keys = np.unique(tmp2)
        x.requires_grad = True
        print("HERE KEYS", keys, y.shape) 
        for key in keys:
            inds = np.where(tmp2 == key)
            input_tmp = x[inds]
            if key in self.clust_tree["1"].keys() and self.clust_tree["1"][key] is not None:
                tmp = self.clust_tree["1"][key].forward(input_tmp) #torch.unsqueeze(x[inds],dim=0))
                if isinstance(tmp,tuple):
                    tmp = tmp[0]
                if isinstance(tmp,list):
                    tmp = tmp[0]

                tmp = torch.unsqueeze(torch.argmax(tmp, dim=1), dim=1)
                tmp[:,0] = tmp[:,0] + (self.num_classes*tmp3[inds[0]])
            else:
                tmp = torch.unsqueeze((self.num_classes*tmp3[inds[0]]), dim=1)
            if tmp_subset is None:
                tmp_subset = tmp
            else:
                tmp_subset = torch.cat((tmp_subset, tmp), dim=0)
            tmp_full[inds] = tmp

        print("HERE", tmp_subset.requires_grad, tmp_full.requires_grad, tmp.requires_grad, input_tmp.requires_grad, input_tmp.shape, x.requires_grad, x.shape)
        return tmp_subset, tmp_full

    
    def training_step(self, batch, batch_idx):
        x = batch
        y, _  = self.forward(x)
        y2, _ = self.forward(x.clone(), perturb=True)
        loss = self.criterion(y,y2)[0] #calculate loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch 
        y, _ = self.forward(x)
        y2, _ = self.forward(x.clone(), perturb=True)
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

