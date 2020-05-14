
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

def trainScalers(data):

	scalers = []
	for r in range(len(data)):
		for n in range(data[r].shape[0]):
				if r == 0:
					scalers.append(StandardScaler())
				subd = data[r][n,:,:]
				scalers[n].partial_fit(subd[np.where(subd > -9999)].reshape(-1, 1))
   
	return scalers

#TODO Remove once interface is implemented
def hardcoded_loader(filenames, chunk_size=5, scalers = None):
        data = []
        for i in range(0, len(filenames)):
                print(filenames[i])
                if os.path.exists(filenames[i]):
            
                        dat = np.load(filenames[i]).astype(np.float64)
                        dat = dat[:,:dat.shape[1]-2,:dat.shape[2]-2]
                        dat = np.delete(dat, 66, 0)
						#Out of range pixels set to 0.0 by MisrToolkit, and I followed convention with bad pixels
						#Need to change it before StandardScale
                        dat[np.where(dat < 0.000000005)] = -9999
            
                        print(dat.shape)
                        data.append(dat)

        if scalers is None:
            scalers = trainScalers(data) 

        data_final = []
        inds = []
        for r in range(len(data)):
            for n in range(data[r].shape[0]):
                subd = data[r][n, :, :]
                subd[np.where(subd > -9999)] = scalers[n].transform(subd[np.where(subd > -9999)].reshape(-1, 1)).reshape(-1)
                data[r][n, :, :] = subd
	            
        cont_count = 0
        data_final = []
        inds = []	
        for r in range(len(data)):
                    for j in range(0, data[r].shape[1], chunk_size):
                            for k in range(0, data[r].shape[2], chunk_size):
                                sub_data_total = []
                                for c in range(0, data[r].shape[0]): 
                                    end_j = j + chunk_size
                                    end_k = k + chunk_size
                                    sub_data = data[r][c, j:end_j, k:end_k]
                                    sub_data2 = sub_data.reshape(sub_data.shape[0] * sub_data.shape[1])
                                    sub_data_total.append(sub_data2)
 
                                sub_data_total = np.array(sub_data_total)
                                if(sub_data_total.min() < -1000):
                                    cont_count = cont_count + 1
                                    continue
                                if(sub_data_total.shape[1] != chunk_size*chunk_size):
                                    print("ERROR:", sub_data_total.shape)
                                data_final.append(sub_data_total.ravel())
                                inds.append([r,j,k])
                    print("NUMBER DEFAULT PIXELS SKIPPED:", cont_count)                 
        c = list(zip(data_final, inds))
        random.shuffle(c)
        data_final, inds = zip(*c)

        return np.array(data_final), np.array(inds), scalers

def test_misr(chunk_size, number_channel, data_train, data_test):

    data, inds, scalers = hardcoded_loader(data_train, chunk_size)     
    x = torch.from_numpy(data)
    print(data.shape, inds.shape, x.shape)

	#Unsupervised, so targets are not used. Currently, I use this to store original image indices for each point  
    x2 = Dataset(torch.from_numpy(data.astype(np.float32)), torch.from_numpy(inds))
    print(data.shape, inds.shape)
    new_dbn = dbn.DBN(model='variance_gaussian', n_visible=chunk_size*chunk_size*number_channel, n_hidden=[2000, 600], steps=[10, 10], learning_rate=[0.01, 0.01], momentum=[0.95, 0.95], decay=[0.0001, 0.0001], temperature=[0.5, 0.5], use_gpu=True)
    new_dbn.fit(x2, batch_size=10, epochs=[100, 100])	

 
    output = new_dbn.forward(x)
    rec_mse, v = new_dbn.reconstruct(x)   
    np.save(output.numpy(), "output/Learnergy/DBN_MISR_MODIS_1/output.npy")
    np.save(rec_mse.numpy(), "output/Learnergy/DBN_MISR_MODIS_1/rec_mse.npy")

    test, inds2, _ = hardcoded_loader(data_test, chunk_size, scalers)
    x2 = torch.from_numpy(test)
    output = new_dbn.forward(x2)
    rec_mse, v = new_dbn.reconstruct(x2) 
    np.save(output.numpy(), "output/Learnergy/DBN_MISR_MODIS_1/output_test.npy")
    np.save(rec_mse.numpy(), "output/Learnergy/DBN_MISR_MODIS_1/rec_mse_test.npy")
 
    torch.save(new_dbn, "output/Learnergy/DBN_MISR_MODIS_1/dbn.pth")
    

def main(chunk_size, number_channel, data_train, data_test):
    test_misr(chunk_size, number_channel, data_train, data_test)


if __name__ == '__main__':
    outDir = 'output/Learnergy/DBN_MISR_MODIS_1/'
    os.makedirs(outDir, exist_ok=True)
    main(CHUNK_SIZE, NUMBER_CHANNELS, data_fn3, data_fn3_test, outDir)



