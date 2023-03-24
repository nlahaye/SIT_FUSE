import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from learnergy.core import Model
import learnergy.utils.constants as c
import learnergy.utils.exception as e
from learnergy.models.bernoulli import RBM
from learnergy.utils import logging

import scipy
from sys import float_info

logger = logging.get_logger(__name__)




class ClustDBN(Model):

    def __init__(self, dbn_trunk, input_fc , n_classes, use_gpu=True):

        super(ClustDBN, self).__init__(use_gpu=use_gpu)

        self.dbn_trunk = dbn_trunk
        self.input_fc = input_fc
        self.n_classes = n_classes

        #fc = nn.Linear(input_fc , n_classes)
        self.number_heads = 1 #TODO try out multi
        self.fc = MultiPrototypes(self.input_fc, self.n_classes, self.number_heads)
        self.fc.to(self.dbn_trunk.torch_device)

        # Cross-Entropy loss is used for the discriminative fine-tuning
        #criterion = nn.CrossEntropyLoss()
 

        #TODO configurable? What does FaceBook and IID paper do, arch-wise?
        # Creating the optimzers
        self.optimizer = [
            #torch.optim.Adam(dbn_trunk.models[-1].parameters(), lr=0.00001), #TODO Test altering all layers? Last DBN Layer? Only Head?
            torch.optim.Adam(self.fc.parameters(), lr=0.001),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass over the data.

        Args:
            x: An input tensor for computing the forward pass.

        Returns:
            (torch.Tensor): A tensor containing the DBN's outputs.

        """
        x = self.fc.forward(self.dbn_trunk.forward(x))

        return x



 
    def fit(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: Optional[int] = 128,
        epochs: Optional[int] = 10,
        batches: Optional[torch.utils.data.DataLoader] = None,
        sampler: Optional[torch.utils.data.distributed.DistributedSampler] = None,
    ) -> Tuple[float, float]:

 
        # Creating training and validation batches
        #train_batch = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0)
        #val_batch = DataLoader(test, batch_size=10000, shuffle=False, num_workers=0)

        # Transforming the dataset into training batches
        if batches is None:
            batches = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

        scaler = None
        if self.device == "cuda":
            scaler = GradScaler()


        # For amount of fine-tuning epochs
        for e in range(epochs):
            if sampler is not None:
                sampler.set_epoch(e)
            print(f"Epoch {e+1}/{epochs}")

            # Resetting metrics
            train_loss, val_acc = 0, 0

            # For every possible batch
            for x_batch, _ in tqdm(batches):

                loss = 0
                dt = torch.float16
                if self.device == "cpu":
                    dt = torch.bfloat16 
                with torch.autocast(device_type=self.device, dtype=dt):
                    x2 = scipy.ndimage.gaussian_filter1d(x_batch,3)
                    x3 = scipy.ndimage.gaussian_filter1d(x_batch,6)

                    # Checking whether GPU is avaliable and if it should be used
                    if "cuda" in self.device:
                        # Applies the GPU usage to the data and labels
                        x_batch = x_batch.to(self.dbn_trunk.torch_device, non_blocking = True)
                        x2 = torch.from_numpy(x2).to(self.dbn_trunk.torch_device, non_blocking = True)
                        x3 = torch.from_numpy(x3).to(self.dbn_trunk.torch_device, non_blocking = True)
                                   

                    # Passing the batch down the model
                    y = self.dbn_trunk(x_batch)
                    y2 = self.dbn_trunk(x2)
                    y3 = self.dbn_trunk(x3)

                    # Reshaping the outputs
                    y = y.reshape(
                        x_batch.size(0), self.input_fc)
                    y2 = y2.reshape(
                        x_batch.size(0), self.input_fc)
                    y3 = y3.reshape(
                        x_batch.size(0), self.input_fc)

                    # Calculating the fully-connected outputs
                    y = self.fc(y)
                    y2 = self.fc(y2)
                    y3 = self.fc(y3)
          
                    temperature = 1 #TODO toggle
                    # Calculating loss
                    for h in range(self.number_heads):
                        loss = loss + IID_segmentation_loss(y[h], y2[h], y3[h])[0] #criterion(y, y_batch)
                    loss = loss / self.number_heads
                    
                    if "cuda" in self.device:
                        x_batch.detach().cpu()
                        x2.detach().cpu()
                        x3.detach().cpu() 
                        for i in range(len(y)):
                            y[i].detach().cpu()
                            y2[i].detach().cpu()
                            y3[i].detach().cpu()

                # Initializing the gradient
                for param in self.parameters():
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
 

                # Propagating the loss to calculate the gradients
                #loss.backward()

                # For every possible optimizer
                #for opt in optimizer:
                #    # Performs the gradient update
                #    opt.step()
 
                #Adding current batch loss
                train_loss = train_loss + loss.item()
                logger.info("LOSS: %f", train_loss)
            """
            # Calculate the test accuracy for the model:
            for x_batch, y_batch in tqdm(val_batch):
                x2 = scipy.ndimage.gaussian_filter1d(x_batch,3)
                x3 = scipy.ndimage.gaussian_filter1d(x_batch,6)

                # Checking whether GPU is avaliable and if it should be used
                if model.device == "cuda":
                    # Applies the GPU usage to the data and labels
                    x_batch = x_batch.to(self.dbn_trunk.torch_device, non_blocking = True)
                    x2 = x2.to(self.dbn_trunk.torch_device, non_blocking = True)
                    x3 = x3.to(self.dbn_trunk.torch_device, non_blocking = True)




                # Passing the batch down the model
                y = model(x_batch)
                y2 = model(x2)
                y3 = model(x3)

                # Reshaping the outputs
                y = y.reshape(
                    x_batch.size(0), input_fc)
                y2 = y2.reshape(
                    x_batch.size(0), input_fc)
                y3 = y3.reshape(
                    x_batch.size(0), input_fc)

                # Calculating the fully-connected outputs
                y = fc(y)
                y2 = fc(y2)
                y3 = fc(y3)
                  

                # Calculating predictions
                _, preds = torch.max(y, 1)

                # Calculating validation set accuracy
                val_acc = torch.mean((torch.sum(preds == y_batch).float()) / x_batch.size(0))

            print(f"Loss: {train_loss / len(train_batch)} | Val Accuracy: {val_acc}")
            """

## Saving the fine-tuned model
#torch.save(model, "tuned_model.pth")

## Checking the model's history
#print(model.history)


#From SWAV
class MultiPrototypes(nn.Module):
    #I dont allow for variation of n_clusters in each prototype, as SWAV does
    def __init__(self, output_dim, n_classes, nmb_heads):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = nmb_heads
        for i in range(nmb_heads):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, n_classes, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out 






#From IIC
def IID_segmentation_loss(x1_outs, x2_outs, x3_outs, half_T_side_dense=0, half_T_side_sparse_min=0,half_T_side_sparse_max=0, lamb=1.0): # all_affine2_to_1=None,
                          #all_mask_img1=None, lamb=1.0,
                          #half_T_side_dense=None,
                          #half_T_side_sparse_min=None,
                          #half_T_side_sparse_max=None):
  #assert (x1_outs.requires_grad)
  #assert (x2_outs.requires_grad)
  #assert (not all_affine2_to_1.requires_grad)
  #assert (not all_mask_img1.requires_grad)

  #assert (x1_outs.shape == x2_outs.shape)

  # bring x2 back into x1's spatial frame
  #x2_outs_inv = perform_affine_tf(x2_outs, all_affine2_to_1)

  #if (half_T_side_sparse_min != 0) or (half_T_side_sparse_max != 0):
  #  x2_outs_inv = random_translation_multiple(x2_outs_inv,
  #                                            half_side_min=half_T_side_sparse_min,
  #                                            half_side_max=half_T_side_sparse_max)

  #if RENDER:
  #  # indices added to each name by render()
  #  render(x1_outs, mode="image_as_feat", name="invert_img1_")
  #  render(x2_outs, mode="image_as_feat", name="invert_img2_pre_")
  #  render(x2_outs_inv, mode="image_as_feat", name="invert_img2_post_")
  #  render(all_mask_img1, mode="mask", name="invert_mask_")

  # zero out all irrelevant patches
  #bn, k, h, w = x1_outs.shape
  #all_mask_img1 = all_mask_img1.view(bn, 1, h, w)  # mult, already float32
  #x1_outs = x1_outs * all_mask_img1  # broadcasts
  #x2_outs_inv = x2_outs_inv * all_mask_img1

  # sum over everything except classes, by convolving x1_outs with x2_outs_inv
  # which is symmetric, so doesn't matter which one is the filter
  #x1_outs = x1_outs.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w
  #x2_outs_inv = x2_outs_inv.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w

  # k, k, 2 * half_T_side_dense + 1,2 * half_T_side_dense + 1

  #TODO conv2d vs cond 1d based on input type
  #p_i_j = F.conv2d(x1_outs, weight=x2_outs, padding=(half_T_side_dense, half_T_side_dense))
  x1_outs = x1_outs.reshape(x1_outs.shape[0], 1, x1_outs.shape[1])
  x2_outs = x2_outs.reshape(x2_outs.shape[0], 1,x2_outs.shape[1])
  x3_outs = x3_outs.reshape(x3_outs.shape[0], 1,x3_outs.shape[1])

  #print(x1_outs.shape, x2_outs.shape, x3_outs.shape) 
  print(x1_outs.max(), x2_outs.max(), x3_outs.max(), x1_outs.mean(), x2_outs.mean(), x3_outs.mean())
  p_i_j = F.conv1d(x1_outs, weight=x2_outs)
  p_i_j_2 = F.conv1d(x1_outs, weight=x3_outs)
  p_i_j = p_i_j / p_i_j.sum()
  p_i_j_2 = p_i_j_2 / p_i_j_2.sum()
  #print(x1_outs.shape, x2_outs.shape, p_i_j.shape, x3_outs.shape)
  p_i_j = p_i_j + p_i_j_2
  p_i_j = p_i_j / 2.0
  p_i_j = p_i_j + abs(p_i_j.min()) + 1.0
  #print(x1_outs.shape, x2_outs.shape, p_i_j.shape, x3_outs.shape)
  #p_i_j = p_i_j.sum(dim=2, keepdim=False).sum(dim=2, keepdim=False)  # k, k
  p_i_j = p_i_j.squeeze()

  # normalise, use sum, not bn * h * w * T_side * T_side, because we use a mask
  # also, some pixels did not have a completely unmasked box neighbourhood,
  # but it's fine - just less samples from that pixel
  #current_norm = p_i_j.sum())
  #p_i_j = p_i_j / current_norm
 
  # symmetrise
  p_i_j = (p_i_j + p_i_j.t()) / 2.

  # compute marginals
  p_i_mat = p_i_j.sum(dim=1).unsqueeze(1)  # k, 1
  p_j_mat = p_i_j.sum(dim=0).unsqueeze(0)  # 1, k

  EPS = float_info.epsilon
  # for log stability; tiny values cancelled out by mult with p_i_j anyway
  p_i_j[(p_i_j < EPS).data] = EPS
  p_i_mat[(p_i_mat < EPS).data] = EPS
  p_j_mat[(p_j_mat < EPS).data] = EPS

  lamb = 0.001
  print(p_i_j.min(), p_i_mat.min(), p_j_mat.min(), EPS)
  #print(p_i_j, p_i_j.min(), p_i_j.max())
  # maximise information
  loss = (-p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_i_mat) -
                    lamb * torch.log(p_j_mat))).sum()
  

# for analysis only
  loss_no_lamb = (-p_i_j * (torch.log(p_i_j) - torch.log(p_i_mat) -
                            torch.log(p_j_mat))).sum()
  return loss, loss_no_lamb




