import time
from typing import Optional, Tuple
import sys

import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from tqdm import tqdm

from learnergy.core import Model
import learnergy.utils.constants as c
import learnergy.utils.exception as e
from learnergy.models.bernoulli import RBM
from learnergy.utils import logging

from cuml.preprocessing import MinMaxScaler, StandardScaler
 
import scipy
from sys import float_info

import time
from datetime import timedelta

logger = logging.get_logger(__name__)




class ClustDBN(Model):

    def __init__(self, dbn_trunk, input_fc , n_classes, use_gpu=True, scaler = None):

        super(ClustDBN, self).__init__(use_gpu=use_gpu)

        self.dbn_trunk = dbn_trunk
        self.input_fc = input_fc
        self.n_classes = n_classes

        self.number_heads = 1 #TODO try out multi
        self.fc = MultiPrototypes(self.input_fc, self.n_classes, self.number_heads)
        self.fc = self.fc.to(self.dbn_trunk.torch_device, non_blocking = True)
        self.to(self.dbn_trunk.torch_device, non_blocking = True)
        self.dbn_trunk = self.dbn_trunk.to(self.dbn_trunk.torch_device, non_blocking = True)
 
        #TODO configurable? What does FaceBook and IID paper do, arch-wise?
        # Creating the optimzers
        self.optimizer = [
            torch.optim.Adam(self.fc.parameters(), lr=0.0001), #TODO Test altering all layers? Last DBN Layer? Only Head?
            #torch.optim.SGD(self.fc.parameters(), lr=0.0001, momentum=0.5, weight_decay=0.0001, nesterov=True),
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
            x_batch = x_batch.to(self.dbn_trunk.torch_device, non_blocking = True)
            with torch.no_grad():
                self.scaler.partial_fit(self.dbn_trunk(x_batch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass over the data.

        Args:
            x: An input tensor for computing the forward pass.

        Returns:
            (torch.Tensor): A tensor containing the DBN's outputs.

        """
        #TODO fix for multi-head
        dt = x.dtype
        y = self.dbn_trunk.forward(x)
        if isinstance(y,tuple):
            y = y[0]
 
        y = torch.as_tensor(self.scaler.transform(y), dtype = dt)
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

 
        # Transforming the dataset into training batches
        if batches is None:
           batches = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

        scaler = None
        if self.device == "cuda":
            scaler = GradScaler()

 
        if self.fit_scaler:
            scaler_loader = DataLoader(
                dataset, batch_size=1500, shuffle=False, num_workers=0
            )
            self.train_scaler(scaler_loader)

        # For amount of fine-tuning epochs
        #stdevs = [0.01,0.001,0.0]
        for e in range(epochs):

            noise_stdev = cluster_gauss_noise_stdev[int(e % len(cluster_gauss_noise_stdev))]
            #if self.fit_scaler and e == 0:
            #    self.train_scaler(batches)            

            if sampler is not None:
                sampler.set_epoch(e)
            print(f"Epoch {e+1}/{epochs}", "STDEV " , str(noise_stdev))

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
                if noise_stdev > 0.0:
                    x2 = x2 + torch.from_numpy(rng.normal(0,noise_stdev,\
                      x2.shape[1]*x2.shape[0]).reshape(x2.shape[0],\
                                x2.shape[1])).type(x2.dtype)
                       
                loss = 0
                dt = torch.float32
                if self.device == "cpu":
                    dt = torch.bfloat16 
                with torch.autocast(device_type=self.device, dtype=dt):
                    x_batch = x_batch.to(self.dbn_trunk.torch_device, non_blocking = True)
                    x2 = x2.to(self.dbn_trunk.torch_device, non_blocking = True)
                                   
                    # Passing the batch down the model
                    y = None
                    y2 = None
                    with torch.no_grad():
                        y = self.dbn_trunk(x_batch)
                        y2 = self.dbn_trunk(x2)
                        if isinstance(y,tuple):
                            y = y[0]
                            y2 = y2[0]
                        y = torch.flatten(torch.as_tensor(self.scaler.transform(y), dtype=dt), start_dim = 1)
                        #y = torch.flatten(y, start_dim = 1)
                        y = y.to(self.dbn_trunk.torch_device, non_blocking = True)
                        y2 = torch.flatten(torch.as_tensor(self.scaler.transform(y2), dtype=dt), start_dim = 1)
                        #y2 = torch.flatten(y2, start_dim = 1)
                        if noise_stdev > 0.0:
                            y2 = y2 + torch.from_numpy(rng.normal(0,noise_stdev,\
                                y2.shape[1]*y2.shape[0]).reshape(y2.shape[0],\
                                y2.shape[1])).type(y2.dtype)
                        y2 = y2.to(self.dbn_trunk.torch_device, non_blocking = True)
                    #x2 = np.clip(x2, pre_min, pre_max)


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
            self.add_module("flatten" + str(i), nn.Flatten())
            #for j in range(0,3):
            self.add_module("prototypes" + str(i) + "_0", nn.Linear(output_dim, n_classes))
            #self.add_module("prototypes" + str(i) + "_0", nn.Linear(output_dim, output_dim*2)) 
            ##self.add_module("prototypes" + str(i) + "_1", nn.Linear(output_dim, n_classes))
            #self.add_module("prototypes" + str(i) + "_2", nn.Linear(n_classes*2, n_classes))
            self.add_module("prototypes" + str(i) + "_1", nn.Softmax(dim=1)) #n_classes, n_classes, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            x = getattr(self, "flatten" + str(i))(x)
            for j in range(0,2):
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
  lamb = 0.1
  
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



 


