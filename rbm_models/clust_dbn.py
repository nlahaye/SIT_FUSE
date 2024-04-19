import time
from typing import Optional, Tuple
import sys

import numpy as np
import pandas as pd
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from tqdm import tqdm
from pprint import pprint

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

import scipy
from sys import float_info

import time
from datetime import timedelta

logger = logging.get_logger(__name__)




class ClustDBN(Model):

    def __init__(self, dbn_trunk, input_fc , n_classes, use_gpu=True, scaler = None, layered = False):

        super(ClustDBN, self).__init__(use_gpu=use_gpu)

        self.layered = layered

        self.dbn_trunk = dbn_trunk
        self.input_fc = input_fc
        self.n_classes = n_classes

        self.number_heads = 1 #TODO try out multi
        self.fc = MultiPrototypes(self.input_fc, self.n_classes, self.number_heads)
        if hasattr(self.dbn_trunk, "torch_device"):
            self.fc = self.fc.to(self.dbn_trunk.torch_device, non_blocking = True)
            for m in self.fc.modules():
                m = m.to(self.dbn_trunk.torch_device, non_blocking = True)
            self.to(self.dbn_trunk.torch_device, non_blocking = True)
            self.dbn_trunk = self.dbn_trunk.to(self.dbn_trunk.torch_device, non_blocking = True)
        else:
            self.fc = self.fc.cuda()
            for m in self.fc.modules():
                m = m.cuda()
            self.cuda()
            self.dbn_trunk = self.dbn_trunk.cuda() 

        #TODO configurable? What does FaceBook and IID paper do, arch-wise?
        # Creating the optimzers
        self.optimizer = [
            torch.optim.Adam(self.fc.parameters(), lr=0.0001),
            #torch.optim.Adam(self.dbn_trunk.parameters(), lr=0.0001),
            #torch.optim.Adam(self.dbn_trunk.parameters(), lr=1e-4) #TODO Test altering all layers? Last DBN Layer? Only Head?
            #torch.optim.SGD(self.dbn_trunk.parameters(), lr=0.00001, momentum=0.95, weight_decay=0.0001, nesterov=True),
        ]

        self.fit_scaler = False
        if scaler is None:
            self.scaler = StandardScaler()
            self.fit_scaler = True
        else:
            self.scaler = scaler        

        self.initialize_weights()

    def train_scaler(self, batches):
        for x_batch, _ in tqdm(batches):
            global cuml_avail
            if cuml_avail:
                if hasattr(self.dbn_trunk, "torch_device"):
                    x_batch = x_batch.to(self.dbn_trunk.torch_device, non_blocking = True)
                else:
                    x_batch = x_batch.cuda()
            with torch.no_grad():
                tmp = self.dbn_trunk(x_batch)
                #if len(tmp.shape) == 4:
                #    tmp = torch.flatten(nn.functional.adaptive_avg_pool2d(tmp, (1,1)), start_dim=1)
                #print(tmp.shape)
                tmp = torch.flatten(tmp, start_dim=1)
                self.scaler.partial_fit(tmp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass over the data.

        Args:
            x: An input tensor for computing the forward pass.

        Returns:
            (torch.Tensor): A tensor containing the DBN's outputs.

        """
        #TODO fix for multi-head
        dt = x.dtype
        #If Clustering is single layered or first layer in heirarchy
        if not self.layered:
            y = self.dbn_trunk.forward(x)
        else:
            y = x
        if isinstance(y,tuple):
            y = y[0]
        #if len(y.shape) == 4:
        #            y = nn.functional.adaptive_avg_pool2d(y, (1,1))
        print(y.shape, y.device, y.dtype, self.scaler)
        y = torch.flatten(y, start_dim=1)
 
        global cuml_avail
        if cuml_avail:
            y = torch.as_tensor(self.scaler.transform(y), dtype = dt)
        else:
           y = torch.from_numpy(self.scaler.transform(y.cpu().numpy()), dtype = dt) 
           if hasattr(self.dbn_trunk, "torch_device"):
               y = y.to(self.dbn_trunk.torch_device, non_blocking = True)
           else:
               y = y.cuda()
        y = self.fc.forward(y)

        return y

    
 
    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: Optional[int] = 128,
        epochs: Optional[int] = 100,
        batches: Optional[torch.utils.data.DataLoader] = None,
        sampler: Optional[torch.utils.data.distributed.DistributedSampler] = None,
        cluster_gauss_noise_stdev: Optional[int] = 1,
        cluster_lambda: Optional[float] = 1.0,
    ) -> Tuple[float, float]:

        #self.dbn_trunk.eval()
 
        # Transforming the dataset into training batches
        if batches is None:
           batches = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

        scaler = None
        if self.device == "cuda":
            scaler = GradScaler()

 
        if self.fit_scaler:
            self.train_scaler(batches)

        #unique = None
        for e in range(epochs):
            #unique_tmp = None
            print(f"Epoch {e+1}/{epochs}")

            noise_stdev = cluster_gauss_noise_stdev[int(e % len(cluster_gauss_noise_stdev))]

            if sampler is not None:
                sampler.set_epoch(e)

            # Resetting metrics
            train_loss, val_acc = 0, 0

            ind = 0
            # For every possible batch
            loss = 0
            dist.barrier()
            rng = np.random.default_rng(None)
            for x_batch, _ in tqdm(batches): 
                start_time = time.monotonic()
                x2 = copy.deepcopy(x_batch)
                       
                loss = 0
                dt = torch.float16
                if self.device == "cpu":
                    dt = torch.bfloat16 
                with torch.autocast(device_type=self.device, dtype=dt):
                    if hasattr(self.dbn_trunk, "torch_device"):
                        x_batch = x_batch.to(self.dbn_trunk.torch_device, non_blocking = True)
                        x2 = x2.to(self.dbn_trunk.torch_device, non_blocking = True)
                    else:
                        x_batch = x_batch.cuda()
                        x2 = x2.cuda()            
                    # Passing the batch down the model
                    y = None
                    y2 = None
                    with torch.no_grad():
                        y = self.dbn_trunk(x_batch)
                        y2 = self.dbn_trunk(x2)
                        if isinstance(y,tuple):
                            y = y[0]
                            y2 = y2[0]
                        #if len(y.shape) == 4:
                        #    y = nn.functional.adaptive_avg_pool2d(y, (1,1))
                        #    y2 = nn.functional.adaptive_avg_pool2d(y2, (1,1))

                        y = torch.flatten(y, start_dim = 1)
                        y2 = torch.flatten(y2, start_dim = 1)
                        #print("HERE BATCH MEAN", y.mean(axis=1))
                        #print("HERE BATCH STD", y.std(axis=1))
    
                        global cuml_avail
                        if cuml_avail:
                            y = torch.flatten(torch.as_tensor(self.scaler.transform(y), dtype=dt), start_dim = 1)
                            y2 = torch.flatten(torch.as_tensor(self.scaler.transform(y2), dtype=dt), start_dim = 1)
                        else:
                            y = torch.flatten(torch.from_numpy(self.scaler.transform(y.cpu().numpy()), dtype=dt), start_dim = 1)
                            y2 = torch.flatten(torch.from_numpy(self.scaler.transform(y2.cpu().numpy()), dtype=dt), start_dim = 1)
 
                        if hasattr(self.dbn_trunk, "torch_device"):
                            y = y.to(self.dbn_trunk.torch_device, non_blocking = True)
                        else:
                            y = y.cuda()
                        if noise_stdev > 0.0:
                            #print(f"Epoch {e+1}/{epochs}", "MEAN STDEV " , torch.min(torch.abs(torch.mean(y2, axis=1))).cpu().numpy(), torch.min(torch.abs(torch.std(y2, axis=1))).cpu().numpy())
                            #y2 = y2 + torch.from_numpy(rng.normal(
                            #    torch.min(torch.abs(torch.mean(y2, axis=1))).cpu().numpy(), \
                            #    torch.min(torch.abs(torch.std(y2, axis=1))).cpu().numpy(),\
                            #    y2.shape[1]*y2.shape[0]).reshape(y2.shape[0],\
                            #    y2.shape[1])).type(y2.dtype).to(y2.device)
                            y2 = y2 + torch.from_numpy(rng.normal(0.0, noise_stdev, \
                                y2.shape[1]*y2.shape[0]).reshape(y2.shape[0],\
                                y2.shape[1])).type(y2.dtype).to(y2.device)
                        if hasattr(self.dbn_trunk, "torch_device"): 
                            y2 = y2.to(self.dbn_trunk.torch_device, non_blocking = True)
                        else:
                            y2 = y2.cuda()
                    ##NO GRAD

                    #print("HERE BATCH SCALED MEAN1 ", y.mean(axis=1))
                    #print("HERE BATCH SCALED STD1 ", y.std(axis=1))

 
                    #print("HERE BATCH SCALED MEAN2 ", y2.mean(axis=1))
                    #print("HERE BATCH SCALED STD2 ", y2.std(axis=1))


                    # Calculating the fully-connected outputs
                    y = self.fc(y)
                    y2 = self.fc(y2)
 
                    end_time = time.monotonic()
                    temperature = 1 #TODO toggle
                    # Calculating loss
                    for h in range(self.number_heads):
                        loss = loss + IID_loss(y[h], y2[h], cluster_lambda)[0] 
                    loss = loss / self.number_heads
                    temperature = 1 #TODO toggle                    

                    #if unique_tmp is None:
                    #    unique_tmp = np.unique(np.concatenate((torch.unique(torch.argmax(y[0], axis = 1)).detach().cpu().numpy(), torch.unique(torch.argmax(y2[0], axis = 1)).detach().cpu().numpy())))
                    #else:
                    #    unique_tmp = np.unique(np.concatenate((unique_tmp, np.unique(np.concatenate((torch.unique(torch.argmax(y[0], axis = 1)).detach().cpu().numpy(), torch.unique(torch.argmax(y2[0], axis = 1)).detach().cpu().numpy()))))))    

                    #print("UNIQUE CLUSTERS", unique_tmp, unique_tmp.shape)
                    if "cuda" in self.device:
                        x_batch = x_batch.detach().cpu()
                        x2 = x2.detach().cpu()
                        for i in range(len(y)):
                            y[i] = y[i].detach().cpu()
                            y2[i] = y2[i].detach().cpu()
                del x_batch
                del x2
                del y
                del y2
                end_time = time.monotonic()
                for param in self.fc.parameters():
                    param.grad = None
                    #TODO if optimizing DBN layers, zero out grad

                if scaler is not None:
                    # Computing the gradients
                    scaler.scale(loss).backward()

                    # Updating the parameters
                    for opt in self.optimizer:
                        scaler.step(opt)
                        scaler.update()
                else:
                    loss.backward()
                    for opt in self.optimizer:
                        opt.step() 

                if "cuda" in self.device:
                    loss = loss.detach().cpu()
                    torch.cuda.empty_cache()
                ind = ind + 1
        
                #self.print_weights_and_grad()
                #Adding current batch loss
                train_loss = train_loss + loss.item()
                end_time = time.monotonic()

            #if unique is None:
            #    unique = unique_tmp
            #else:
            #    unique = np.unique(np.concatenate((unique, unique_tmp)))
            #print("CLUSTERS", unique)
            logger.info("LOSS: %f", (train_loss/len(batches)))


    def initialize_weights(self):
        for m in self.modules():
          if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()



    def print_weights_and_grad(self):
      print("---------------")
      for n, p in self.fc.named_parameters():
        print("%s abs: min %f max %f max grad %f" %
              (n, torch.abs(p.data).min(), torch.abs(p.data).max(), \
               torch.abs(p.grad).max()))
      print("---------------")



#From SWAV
class MultiPrototypes(nn.Module):
    #I dont allow for variation of n_clusters in each prototype, as SWAV does
    def __init__(self, output_dim, n_classes, nmb_heads):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = nmb_heads
        for i in range(nmb_heads):
            self.n_layers = 0
            self.add_module("flatten" + str(i), nn.Flatten())
            #for j in range(0,3):
            #if output_dim <= n_classes*2.5:
            self.n_layers =  self.n_layers + 1
            self.add_module("prototypes" + str(i) + "_0", nn.Linear(output_dim, n_classes))
            #else:
            #    tmp = output_dim
            #    while tmp > n_classes*2.5:
            #        self.n_layers =  self.n_layers + 1
            #        self.add_module("prototypes" + str(i) + "_" + str(self.n_layers-1), nn.Linear(tmp, int(tmp/2)))
            #        tmp = int(tmp/2)
            #self.add_module("prototypes" + str(i) + "_0", nn.Linear(output_dim, output_dim*2)) 
            ##self.add_module("prototypes" + str(i) + "_1", nn.Linear(output_dim, n_classes))
            #self.add_module("prototypes" + str(i) + "_2", nn.Linear(n_classes*2, n_classes))
            self.n_layers =  self.n_layers + 1
            self.add_module("prototypes" + str(i) + "_" + str(self.n_layers-1), nn.Softmax(dim=1)) #n_classes, n_classes, bias=False))
         

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            x = getattr(self, "flatten" + str(i))(x)
            for j in range(0,self.n_layers):
                x = getattr(self, "prototypes" + str(i) + "_" + str(j))(x)
            out.append(x)
        return out 






#From IIC
def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
  # has had softmax applied
  _, k = x_out.size()



  start_time = time.monotonic()
  p_i_j = compute_joint(x_out, x_tf_out)
  end_time = time.monotonic()
 
  start_time = time.monotonic()

  p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
  p_j = p_i_j.sum(dim=0).view(1, k).expand(k,
                                           k)  # but should be same, symmetric

  p_i = p_i.contiguous()
  p_j = p_j.contiguous()
  p_i_j = p_i_j.contiguous()
 
  # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
  #lamb = 0.1
  
  p_i_j[(p_i_j < EPS).data] = EPS
  p_j[(p_j < EPS).data] = EPS
  p_i[(p_i < EPS).data] = EPS
 
  loss = - p_i_j * (torch.log(p_i_j) \
                    - lamb * torch.log(p_j) \
                    - lamb * torch.log(p_i))

  loss = loss.sum()

  loss_no_lamb = - p_i_j * (torch.log(p_i_j) \
                            - torch.log(p_j) \
                            - torch.log(p_i))

  p_i = p_i.detach()
  del p_i
  p_j = p_j.detach()
  del p_j
  p_i_j = p_i_j.detach()
  del p_i_j

  loss_no_lamb = loss_no_lamb.sum()
  end_time = time.monotonic()

  return loss, loss_no_lamb


def compute_joint(x_out, x_tf_out):
  # produces variable that requires grad (since args require grad)

  bn, k = x_out.size()

  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
  p_i_j = p_i_j.sum(dim=0)  # k, k
  p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
  p_i_j = p_i_j / p_i_j.sum()  # normalise

  return p_i_j



 


