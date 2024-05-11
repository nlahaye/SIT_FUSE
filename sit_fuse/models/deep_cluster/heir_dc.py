import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch

from sit_fuse.losses.iid import IID_loss


from sit_fuse.models.deep_cluster.byol_dc import BYOL_DC
from sit_fuse.models.deep_cluster.ijepa_dc import IJEPA_DC
from sit_fuse.models.deep_cluster.dbn_dc import DBN_DC
from sit_fuse.models.deep_cluster.dc import DeepCluster
from sit_fuse.models.deep_cluster.multi_prototypes import MultiPrototypes

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import uuid

#TODO add back in multi-layer heir
class Heir_DC(pl.LightningModule):
    #take pretrained model path, number of classes, learning rate, weight decay, and drop path as input
    def __init__(self, data, pretrained_model_path, num_classes, lr=1e-3, weight_decay=0, encoder_type=None, number_heads=1, min_samples=1000, encoder=None, clust_tree_ckpt = None):

        super().__init__()
        self.save_hyperparameters(ignore=['data'])

        self.min_samples = min_samples

        #set parameters
        self.lr = lr
        self.weight_decay = weight_decay

        self.key = -1


        self.number_heads = number_heads
        #define model layers
        self.pretrained_model = None
        if encoder_type is None:
            self.pretrained_model = DeepCluster.load_from_checkpoint(pretrained_model_path, img_size=3, in_chans=34) #Why arent these being saved
        elif encoder_type == "dbn":
            self.pretrained_model = DBN_DC.load_from_checkpoint(pretrained_model_path, pretrained_model=encoder)
            self.pretrained_model.eval()
            self.pretrained_model.pretrained_model.eval()
            self.pretrained_model.mlp_head.eval()

            for param in self.pretrained_model.pretrained_model.parameters():
                param.requires_grad = False
            for param in self.pretrained_model.mlp_head.parameters():
                param.requires_grad = False
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        elif encoder_type == "ijepa":
            self.pretrained_model = IJEPA_DC.load_from_checkpoint(pretrained_model_path)
            self.pretrained_model.pretrained_model.model.mode = "test"
            self.pretrained_model.eval()
            self.pretrained_model.pretrained_model.eval()
            self.pretrained_model.mlp_head.eval()
            self.pretrained_model.pretrained_model.model.eval()

            for param in self.pretrained_model.pretrained_model.model.parameters():
                param.requires_grad = False
            for param in self.pretrained_model.pretrained_model.parameters():
                param.requires_grad = False
            for param in self.pretrained_model.mlp_head.parameters():
                param.requires_grad = False
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

            #getattr(self.pretrained_model.mlp_head, "batch_norm0").track_running_stats = True
            #self.pretrained_model.pretrained_model.model.layer_dropout = 0.0
        elif encoder_type == "byol":
            self.pretrained_model = BYOL_DC.load_from_checkpoint(pretrained_model_path)
            self.pretrained_model.eval()
            self.pretrained_model.pretrained_model.eval()
            self.pretrained_model.mlp_head.eval()
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

 
        if clust_tree_ckpt is None: 
            self.generate_label_set(data)
        else:
            state_dict = torch.load(clust_tree_ckpt)
            self.clust_tree, self.lab_full = \
                load_model(self.clust_tree, list(self.pretrained_model.mlp_head.children())[1].num_features, self, state_dict)        
 
        del data

    def generate_label_set(self, data):
        count = 0
        self.lab_full = {}
        batch_size = max(700, self.min_samples)

        output_sze = data.data_full.shape[0]

        test_loader = DataLoader(data, batch_size=batch_size, shuffle=False, \
        num_workers = 0, drop_last = False, pin_memory = True)
        ind = 0
        ind2 = 0

        for data2 in tqdm(test_loader):
            dat_dev = data2[0].to(device=self.pretrained_model.device, non_blocking=True, dtype=torch.float32)

            lab = self.pretrained_model.forward(dat_dev)
            lab = torch.argmax(lab, axis = 1)
            lab = lab.detach().cpu()
            dat_dev = dat_dev.detach().cpu()

            ind1 = ind2
            ind2 += dat_dev.shape[0]
            if ind2 > data.data_full.shape[0]:
                ind2 = data.data_full.shape[0]

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

 
    def forward(self, x, perturb = False, train=False):
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
 
        if train == False: 
            tmp_full = torch.zeros((y.shape[0], 1), device=y.device, dtype=torch.int64)
        else:
            tmp_full = torch.zeros((y.shape[0], self.num_classes), device=y.device, dtype=torch.float32)
        tmp_subset = None
        tmp = y
        if y.ndim > 1 and y.shape[1] > 1:
            tmp = torch.argmax(y, dim=1)
        f = lambda z: str(z)
        tmp2 = np.vectorize(f)(tmp.detach().cpu())
        tmp3 = tmp
        keys = np.unique(tmp2)
        x.requires_grad = True
        #print(keys)
        for key in keys:
            if train and key != self.key:
                continue
            inds = np.where(tmp2 == key)
            input_tmp = x[inds]
            #print(input_tmp.shape)
            if key in self.clust_tree["1"].keys() and self.clust_tree["1"][key] is not None:
                tmp = self.clust_tree["1"][key].forward(input_tmp) #torch.unsqueeze(x[inds],dim=0))
                if isinstance(tmp,tuple):
                    tmp = tmp[0]
                if isinstance(tmp,list):
                    tmp = tmp[0]
 
                if train == False:
                    tmp = torch.unsqueeze(torch.argmax(tmp, dim=1), dim=1)
                    tmp[:,0] = tmp[:,0] + (self.num_classes*tmp3[inds[0]])
            else:
                tmp = torch.unsqueeze((self.num_classes*tmp3[inds[0]]), dim=1)
            if tmp_subset is None:
                tmp_subset = tmp
            else:
                tmp_subset = torch.cat((tmp_subset, tmp), dim=0)
            tmp_full[inds] = tmp

        return tmp_subset, tmp_full

    
    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        x = batch
        y, y1  = self.forward(x, train=True)
        y2, y21 = self.forward(x.clone(), perturb=True, train=True)
        loss, loss2 = self.criterion(y,y2) #calculate loss
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch 
        y, _ = self.forward(x, train=False)
        y2, _ = self.forward(x.clone(), perturb=True, train=False)
        loss = self.criterion(y,y2)[0] #calculate loss
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx):
        
        y, y1  = self.forward(x, train=False)
        return y
    
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



def get_state_dict(clust_tree, lab_full):

    state_dict = {}
    for lab1 in clust_tree.keys():
        if lab1 == "0":
            continue
        if lab1 not in state_dict:
            state_dict[lab1] = {}
            for lab2 in clust_tree[lab1].keys():
                if lab2 not in state_dict[lab1].keys():
                    if clust_tree[lab1][lab2] is not None:
                        if lab2 not in state_dict[lab1].keys():
                            state_dict[lab1][lab2] = {}
                        state_dict[lab1][lab2]["model"] = clust_tree[lab1][lab2].state_dict()
                        uid = str(uuid.uuid1())
    state_dict["labels"] = lab_full
    return state_dict

def load_model(clust_tree, n_visible, model, state_dict):
        lab_full = state_dict["labels"]
        for lab1 in clust_tree.keys():
            if lab1 == "0":
                continue
            for lab2 in lab_full.keys():
                clust_tree[lab1][lab2] = None
                if lab2 in state_dict[lab1].keys():
                    clust_tree[lab1][lab2] = MultiPrototypes(n_visible, model.num_classes, model.number_heads)
                    clust_tree[lab1][lab2].load_state_dict(state_dict[lab1][lab2]["model"])
        return clust_tree, lab_full





