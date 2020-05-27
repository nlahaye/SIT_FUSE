
import os
import numpy as np
import random

import sys
sys.setrecursionlimit(4500)

#ML imports
import torch
from learnergy.core.dataset import Dataset
from learnergy.models import dbn
from sklearn.preprocessing import StandardScaler
 
#Data
from MISR_MODIS_FUSE_DATA_FIREX_9CAM_9 import data_fn3, data_fn3_test, NUMBER_CHANNELS, CHUNK_SIZE
from dbnDatasets import DBNDataset
from utils import numpy_to_torch


def test_misr(chunk_size, number_channel, data_train, data_test, out_dir, use_gpu = True):
 

    #Unsupervised, so targets are not used. Currently, I use this to store original image indices for each point 
    x2 = DBNDataset(data_train, np.load, chunk_size, delete_chans=[66], valid_min=0, valid_max=None, fill_value = -9999, chan_dim = 0, scalers = None, transform=numpy_to_torch)

    new_dbn = dbn.DBN(model='variance_gaussian', n_visible=chunk_size*chunk_size*number_channel, n_hidden=[2000], steps=[10], learning_rate=[0.01], momentum=[0.95], decay=[0.0001], temperature=[0.5], use_gpu=use_gpu)
    new_dbn.fit(x2, batch_size=10, epochs=[1])	

  
    if torch.cuda.is_available() and use_gpu:	
        output = new_dbn.forward(x2.data.cuda()).cpu()
        x2.transform = None
        rec_mse, v = new_dbn.reconstruct(x2)
    else:
        output = new_dbn.forward(x2.data)
        x2.transform = None
        rec_mse, v = new_dbn.reconstruct(x2) 
  
    torch.save(output, out_dir + "/output.data")
    torch.save(rec_mse, out_dir + "/rec_mse.data")


    x3 = DBNDataset(data_test, np.load, chunk_size, delete_chans=[66], valid_min=0, valid_max=None, fill_value = -9999, chan_dim = 0, scalers=x2.scalers, transform=numpy_to_torch)
 
    if torch.cuda.is_available() and use_gpu:
        output = new_dbn.forward(x3.data.cuda()).cpu()
        x3.transform = None
        rec_mse, v = new_dbn.reconstruct(x3) 
    else:
        output = new_dbn.forward(x3.data)
        x3.transform = None
        rec_mse, v = new_dbn.reconstruct(x3) 
   
    torch.save(output, out_dir + "/output_test.data")
    torch.save(rec_mse, out_dir + "/rec_mse_test.data")
 
    torch.save(new_dbn, out_dir + "/dbn.pth")
    

def main(chunk_size, number_channel, data_train, data_test, out_dir):
    test_misr(chunk_size, number_channel, data_train, data_test, out_dir)


if __name__ == '__main__':
    out_dir = 'output/Learnergy/DBN_MISR_MODIS_1/'
    os.makedirs(out_dir, exist_ok=True)
    main(CHUNK_SIZE, NUMBER_CHANNELS, data_fn3, data_fn3_test, out_dir)



