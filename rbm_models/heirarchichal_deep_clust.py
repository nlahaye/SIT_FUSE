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

from cuml.preprocessing import MinMaxScaler, StandardScaler


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
  
    def __init__(self, base_clust, train_data, n_classes, use_gpu=True):
        super(HeirClust, self).__init__(use_gpu=use_gpu)

        self.base_clust = base_clust
    
        self.local_rank = 0
        if "LOCAL_RANK" in os.environ.keys():
                self.local_rank = int(os.environ["LOCAL_RANK"])
 
        self.generate_label_set(train_data)


        self.clust_tree = {"0": {"-1": self.base_clust}, "1": {}}

        
    def fit(self, train_data):
        count = 0
        for key in self.lab_full.keys():
            count = count + 1
            print("LABEL", key, len(self.lab_full[key])) 
            if len(self.lab_full[key]) < 5000:
                self.clust_tree["1"][key] = None
                continue

            print("TRAINING MODEL ", str(count), " / ", str(len(self.lab_full.keys())))

            use_gpu = True
            n_classes = 15
            self.clust_tree["1"][key] = ClustDBN(self.base_clust.dbn_trunk, self.base_clust.input_fc, n_classes,
               use_gpu, None)
            self.clust_tree["1"][key].fc.train()
            self.clust_tree["1"][key].dbn_trunk.eval()
            self.clust_tree["1"][key].fc = DDP(self.clust_tree["1"][key].fc, device_ids=[self.local_rank], output_device=self.local_rank)

            train_subset = DBNDataset()
            print("HERE SUBSET STATS", train_data.data_full[self.lab_full[key]].min(), train_data.data_full[self.lab_full[key]].max(), train_data.data_full[self.lab_full[key]].mean(), train_data.data_full[self.lab_full[key]].std())
            train_subset.init_from_array(train_data.data_full[self.lab_full[key]], train_data.targets_full[self.lab_full[key]], train_data.scaler)
            sampler = DistributedSampler(train_subset, shuffle=True)
            batch_size = min(100, int(train_subset.data_full.shape[0] / 15))
            loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False,
                sampler=sampler, num_workers = 10, pin_memory = False, drop_last=True)
            gpu_usage()
            self.clust_tree["1"][key].fit(train_subset, batch_size, 15, loader, sampler, [0.001, 0.0001, 0.00001, 0.000001, 0.0], 1.0) #TODO
            self.clust_tree["1"][key].eval()
            self.clust_tree["1"][key].fc.eval()
            #self.clust_tree["1"][key] = self.clust_tree["1"][key].cpu()
            #self.clust_tree["1"][key].fc = self.clust_tree["1"][key].fc.cpu()
            torch.cuda.empty_cache() 
             
 
    def generate_label_set(self, data, use_gpu=True):
         
        count = 0 
        self.lab_full = {}
        while(count == 0 or data.has_next_subset() or (data.subset > 1 and data.current_subset > (data.subset-2))):
            batch_size = min(100, int(data.data_full.shape[0] / 15))
 
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
                    if str(l) in self.lab_full.keys():
                        self.lab_full[str(l)] = torch.cat((self.lab_full[str(l)],(inds[0] + ind1)))
                    else:
                        self.lab_full[str(l)] = inds[0] + ind1

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

        tmp_full = torch.zeros((y.shape[0], 1), device=y.device)
        for i in range(y.shape[0]):
            tmp = torch.argmax(y[i])
            tmp2 = str(tmp.detach().cpu())
            tmp3 = tmp.detach().cpu()
            #print(tmp2, self.clust_tree["1"].keys())
            if tmp2 in self.clust_tree["1"].keys() and self.clust_tree["1"][tmp2] is not None:
                tmp = self.clust_tree["1"][tmp2].forward(torch.unsqueeze(x[i],dim=0))
                if isinstance(tmp,tuple):
                    tmp = tmp[0]
                if isinstance(tmp,list):
                    tmp = tmp[0]
                #print("HERE ARGMAX", torch.argmax(tmp, dim=1), (5*tmp))
                tmp = torch.argmax(tmp, dim=1) 
                #print("HERE TMP", tmp2, tmp)
                tmp = tmp + (15*tmp3)
            else:
                tmp = (15*tmp3)
            tmp_full[i] = tmp

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
        n_classes = 15
        use_gpu = True
        print("LOADING MODEL")
        for lab1 in self.clust_tree.keys():
            if lab1 == "0":
                continue
            for lab2 in self.lab_full.keys():
                self.clust_tree[lab1][lab2] = None
                if lab2 in state_dict[lab1].keys():
                    self.clust_tree[lab1][lab2] = ClustDBN(self.base_clust.dbn_trunk, self.base_clust.input_fc, n_classes,
                        use_gpu, self.base_clust.scaler)
                    self.clust_tree[lab1][lab2].fc = DDP(self.clust_tree[lab1][lab2].fc, device_ids=[self.local_rank], output_device=self.local_rank)
                    self.clust_tree[lab1][lab2].load_state_dict(state_dict[lab1][lab2]["model"])
                    self.clust_tree[lab1][lab2].scaler = load(state_dict[lab1][lab2]["scaler"])
        print(self.clust_tree["1"].keys(), self.lab_full.keys(), "KEYS") 




