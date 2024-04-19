import time
from typing import Optional, Tuple
import sys

import numpy as np
import copy
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from tqdm import tqdm

import pickle

from learnergy.core import Model
import learnergy.utils.constants as c
import learnergy.utils.exception as e
from learnergy.models.bernoulli import RBM
from learnergy.utils import logging

import importlib
cuml_loader = importlib.util.find_spec('cuml')
cuml_avail = cuml_loader is not None

if cuml_avail:
    from cuml.preprocessing import MinMaxScaler, StandardScaler
else:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler


from dbn_datasets import DBNDataset
from rbm_models.clust_dbn import ClustDBN
 
import scipy
from sys import float_info

import time
from datetime import timedelta

import os

from GPUtil import showUtilization as gpu_usage

logger = logging.get_logger(__name__)


from joblib import dump, load


#Take base cluster
#Build Tree
#Auto heirarchy - per cluster - per layer
#Specify clusters you do and dont want to expand on
#Save and load tree
#Predict/figure out integration of various layers for a single image with single label set


class HeirClust(Model):
  
    def __init__(self, base_clust, train_data, n_classes, use_gpu=True, min_samples = 1000, gauss_stdevs = [0.01,0.001,0.0], layered = False):
        super(HeirClust, self).__init__(use_gpu=use_gpu)

        self.base_clust = base_clust

        self.layered = layered

        #For Nesting
        self.dbn_trunk = self.base_clust.dbn_trunk
        self.input_fc = self.base_clust.input_fc
        self.scaler  = None
     
        self.local_rank = 0
        if "LOCAL_RANK" in os.environ.keys():
                self.local_rank = int(os.environ["LOCAL_RANK"])


        self.n_classes = n_classes
        self.min_samples = min_samples
        self.gauss_stdevs = gauss_stdevs
 

        self.generate_label_set(train_data)

        self.clust_tree = {"0": {"-1": self.base_clust}, "1": {}}

        
    def fit(self, train_data, epochs = 15, tune_subtrees = None):
        count = 0

        if tune_subtrees is not None and len(tune_subtrees)  > 0:
            for i in range(len(tune_subtrees)):
                tune_subtrees[i] = str(float(tune_subtrees[i]))


        print("TUNE_SUBTREES", tune_subtrees)
        for key in self.lab_full.keys():
            count = count + 1
            print("LABEL", key, len(self.lab_full[key])) 
            if len(self.lab_full[key]) < self.min_samples:
                self.clust_tree["1"][key] = None
                continue
            elif tune_subtrees is not None and len(tune_subtrees) > 0 and key not in tune_subtrees:
                continue 

            print("TRAINING MODEL ", str(count), " / ", str(len(self.lab_full.keys())))

            use_gpu = True
            self.clust_tree["1"][key] = ClustDBN(self.base_clust.dbn_trunk, self.base_clust.input_fc, self.n_classes,
               use_gpu, None, self.layered)
            self.clust_tree["1"][key].fc.train()
            self.clust_tree["1"][key].dbn_trunk.eval()
            self.clust_tree["1"][key].fc = DDP(self.clust_tree["1"][key].fc, device_ids=[self.local_rank], output_device=self.local_rank)

            train_subset = DBNDataset()
            print("HERE SUBSET STATS", train_data.data_full[self.lab_full[key]].min(), train_data.data_full[self.lab_full[key]].max(), train_data.data_full[self.lab_full[key]].mean(), train_data.data_full[self.lab_full[key]].std())
            train_subset.init_from_array(train_data.data_full[self.lab_full[key]], train_data.targets_full[self.lab_full[key]], train_data.scaler)
            sampler = DistributedSampler(train_subset, shuffle=True)
            batch_size = max(700, self.min_samples)
            loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False,
                sampler=sampler, num_workers = 10, pin_memory = False, drop_last=True)
            gpu_usage()
            div = round((10 *  (0.5* (np.log(train_subset.data_full.shape[0]) - np.log(self.min_samples)))) / 10)
            if div < 1:
                div = 1
            n_epochs = int(epochs / div)
            self.clust_tree["1"][key].fit(train_subset, batch_size, n_epochs, loader, sampler, self.gauss_stdevs)#TODO
            self.clust_tree["1"][key].eval()
            self.clust_tree["1"][key].fc.eval()
            #self.clust_tree["1"][key] = self.clust_tree["1"][key].cpu()
            #self.clust_tree["1"][key].fc = self.clust_tree["1"][key].fc.cpu()
            torch.cuda.empty_cache() 
             
 
    def generate_label_set(self, data, use_gpu=True):
         
        count = 0 
        self.lab_full = {}
        while(count == 0 or data.has_next_subset() or (data.subset > 1 and data.current_subset > (data.subset-2))):
            batch_size = max(700, self.min_samples)
 
            output_sze = data.data_full.shape[0]
            append_remainder = int(batch_size - (output_sze % batch_size))

            if isinstance(data.data_full,torch.Tensor):
                data.data_full = torch.cat((data.data_full,data.data_full[0:append_remainder]))
                data.targets_full = torch.cat((data.targets_full,data.targets_full[0:append_remainder]))
            else:
                data.data_full = np.concatenate((data.data_full,data.data_full[0:append_remainder]))
                data.targets_full = np.concatenate((data.targets_full,data.targets_full[0:append_remainder]))
 
            data.current_subset = -1
            data.next_subset()

            test_loader = DataLoader(data, batch_size=batch_size, shuffle=False, \
            num_workers = 0, drop_last = False, pin_memory = False)
            ind = 0
            ind2 = 0

 

            if use_gpu:
                device = torch.device("cuda:{}".format(self.local_rank))
            else:
                device = torch.device("cpu:{}".format(self.local_rank))
 
            for data2 in tqdm(test_loader):
                dat_dev, lab_dev = data2[0].to(device=device, non_blocking=True), data2[1].to(device=device, non_blocking=True)
                dev_ds = TensorDataset(dat_dev, lab_dev)
 
                lab = self.base_clust.forward(dat_dev)
                if isinstance(lab, list):
                    lab = lab[0]
                #If previous layer is top layer / otherwise argmax happens in forward function
                if lab.shape[1] > 1:
                    lab = torch.argmax(lab, axis = 1)
                if use_gpu == True:
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
             
            for lab in self.lab_full.keys():
                print("LABEL INIT", lab, self.lab_full[lab].shape)
            #dist.barrier()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass over the data.

        Args:
            x: An input tensor for computing the forward pass.

        Returns:
            (torch.Tensor): A tensor containing the DBN's outputs.

        """
        #TODO fix for multi-head
        dt = x.dtype
        y = self.base_clust.forward(x)

        if isinstance(y,tuple):
            y = y[0]

        if isinstance(y,list):
            y = y[0]


        tmp_full = torch.zeros((y.shape[0], 1), device=y.device, dtype=torch.int64)
        #for i in range(y.shape[0]):
        tmp = y
        if y.ndim > 1 and y.shape[1] > 1:
            tmp = torch.argmax(y, dim=1)
        #else:
        #    tmp = y[i]
        f = lambda x: str(x)
        tmp2 = np.vectorize(f)(tmp.detach().cpu())
        tmp3 = tmp
        #tmp2 = str(tmp.detach().cpu())
        keys = np.unique(tmp2)
        for key in keys:
            inds = np.where(tmp2 == key)
            if key in self.clust_tree["1"].keys() and self.clust_tree["1"][key] is not None:
                tmp = self.clust_tree["1"][key].forward(x[inds]) #torch.unsqueeze(x[inds],dim=0))
                if isinstance(tmp,tuple):
                    tmp = tmp[0]
                if isinstance(tmp,list):
                    tmp = tmp[0]

                #tmp = np.asarray(tmp)
                #if isinstance(tmp,tuple):
                #    tmp = tmp[0]
                #if isinstance(tmp,list):
                #    tmp = tmp[0]
                tmp = torch.unsqueeze(torch.argmax(tmp, dim=1), dim=1) 
                tmp[:,0] = tmp[:,0] + (self.n_classes*tmp3[inds[0]])
            else:
                tmp = torch.unsqueeze((self.n_classes*tmp3[inds[0]]), dim=1)
            #print(tmp.shape, tmp_full.shape, inds)
            tmp_full[inds] = tmp

        #print("HERE OUTPUT ", torch.unique(tmp_full), tmp_full.shape)
        return tmp_full



    def get_state_dict(self, output_dir="."):

        state_dict = {}
        for lab1 in self.clust_tree.keys():
            if lab1 == "0":
                continue
            if lab1 not in state_dict:
                state_dict[lab1] = {}
                for lab2 in self.clust_tree[lab1].keys(): 
                    if lab2 not in state_dict[lab1].keys():
                        if self.clust_tree[lab1][lab2] is not None:
                            if lab2 not in state_dict[lab1].keys():
                                state_dict[lab1][lab2] = {}
                            state_dict[lab1][lab2]["model"] = self.clust_tree[lab1][lab2].state_dict()
                            uid = str(uuid.uuid1())
                            state_dict[lab1][lab2]["scaler"] = os.path.join(output_dir, "fc_scaler_heir_" + uid + ".pkl")
                            with open(os.path.join(output_dir, state_dict[lab1][lab2]["scaler"]), "wb") as f:
                                dump(self.clust_tree[lab1][lab2].scaler, f, True, pickle.HIGHEST_PROTOCOL) 
        return state_dict  

    def load_model(self, state_dict):
        use_gpu = True
        print("LOADING MODEL")
        for lab1 in self.clust_tree.keys():
            if lab1 == "0":
                continue
            for lab2 in self.lab_full.keys():
                self.clust_tree[lab1][lab2] = None
                if lab2 in state_dict[lab1].keys():
                    self.clust_tree[lab1][lab2] = ClustDBN(self.base_clust.dbn_trunk, self.base_clust.input_fc, self.n_classes,
                        use_gpu, self.base_clust.scaler, self.layered)
                    self.clust_tree[lab1][lab2].fc = DDP(self.clust_tree[lab1][lab2].fc, device_ids=[self.local_rank], output_device=self.local_rank)
                    self.clust_tree[lab1][lab2].load_state_dict(state_dict[lab1][lab2]["model"])
                    self.clust_tree[lab1][lab2].scaler = load(state_dict[lab1][lab2]["scaler"])
        print(self.clust_tree["1"].keys(), self.lab_full.keys(), "KEYS") 




