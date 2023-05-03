"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
import os
import numpy as np
import random
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.setrecursionlimit(4500)

#ML imports
import torch
from torchvision import transforms
from dbn_datasets import DBNDataset

from utils import numpy_to_torch, read_yaml, get_read_func, get_scaler

import argparse

from skimage.util import view_as_windows

class DBNDatasetConv(DBNDataset):

	def __init__(self, data_filename, indices_filename, transform = None, subset=None):

		self.data_full = torch.load(data_filename)
		self.targets_full = torch.load(indices_filename)

		self.scale = False
		self.transform = transform

		self.subset = subset

		if self.subset is None:
			self.subset = 1
		self.current_subset = -1

		self.next_subset()



	def __init__(self, filenames, read_func, read_func_kwargs, delete_chans, valid_min, valid_max, fill_value = -9999, chan_dim = 0, transform_chans = [], transform_values = [], transform=None, subset=None, tile = False, tile_size = None, tile_step = None, subset_training = -1):
		#Scaler info isnt used here, but keeping same interface as DBNDataset

                #TODO Employ stratification
		self.train_indices = None
 
		self.filenames = filenames
		self.transform = transform
		self.delete_chans = delete_chans
		self.valid_min = valid_min
		self.valid_max = valid_max
		self.fill_value = fill_value
		self.chan_dim = chan_dim
		self.transform_chans = transform_chans
		self.transform_value = transform_values
		self.transform = transform
		self.read_func = read_func
		self.read_func_kwargs = read_func_kwargs
		self.subset = subset
		self.tile = tile
		self.tile_size = tile_size
		self.tile_step = tile_step
		if self.subset is None:
			self.subset = 1		
		self.current_subset = -1
		self.subset_training = subset_training
	

		self.__loaddata__()

	def __loaddata__(self):
		
		data_local = []
		for i in range(0, len(self.filenames)):
			if (type(self.filenames[i]) == str and os.path.exists(self.filenames[i])) or (type(self.filenames[i]) is list and os.path.exists(self.filenames[i][0])):
				print(self.filenames[i])
				dat = self.read_func(self.filenames[i], **self.read_func_kwargs).astype(np.float64)
				print(dat.shape, dat[np.where(dat > -99999)].min(), dat[np.where(dat > -99999)].max())
				for t in range(len(self.transform_chans)):
					slc = [slice(None)] * dat.ndim
					slc[self.chan_dim] = slice(self.transform_chans[t], self.transform_chans[t]+1)
					tmp = dat[tuple(slc)]
					if self.valid_min is not None:
						inds = np.where(tmp < self.valid_min - 0.00000000005) 
						tmp[inds] = self.transform_value[t]
					if self.valid_max is not None:
						inds = np.where(tmp > self.valid_max - 0.00000000005)
						tmp[inds] = self.transform_value[t]
				if len(self.transform_chans) > 0:
					del slc
					del tmp
										
				dat = np.delete(dat, self.delete_chans, self.chan_dim)

				if self.valid_min is not None:
					dat[np.where(dat < self.valid_min - 0.00000000005)] = -9999
				if self.valid_max is not None:	
					dat[np.where(dat > self.valid_max - 0.00000000005)] = -9999
				if self.fill_value is not None:
					dat[np.where(dat == self.fill_value)] = -9999
				dat = np.moveaxis(dat, self.chan_dim, 0)
				data_local.append(dat)

		del dat 
		dim1 = 1
		dim2 = 2
		self.chan_dim = 0

		self.data = []
		self.targets = []
		if self.tile:
			window_size = [0,0,0]
			window_size[dim1] = self.tile_size[0]
			window_size[dim2] = self.tile_size[1]
			tile_step = self.tile_step
			window_size[self.chan_dim] = data_local[0].shape[self.chan_dim]
			window_size = tuple(window_size)
		for r in range(len(data_local)):
			count = 0
			last_count = len(self.data)
			sub_data_total = []

			if self.tile:
				windw_dat = np.squeeze(view_as_windows(data_local[r], window_size, step=tile_step))
				print("WINDOW SHAPE", windw_dat.shape)
				#windw_dat = windw_dat.reshape(windw_dat.shape[0]*windw_dat.shape[1], windw_dat.shape[2],windw_dat.shape[3],windw_dat.shape[4])
				#print("WINDOW SHAPE2", windw_dat.shape)
				for w in range(0,windw_dat.shape[0]):
					if len(windw_dat.shape) == 5:
						for w2 in range(0,windw_dat.shape[1]):
							if(windw_dat[w][w2].max() > -9999):
								self.data.append(windw_dat[w][w2])
								self.targets.append([r,w,w2])
						
						else:
							count = count + 1
							continue
					else:
						if(windw_dat[w].max() > -9999):
							self.data.append(windw_dat[w])
							self.targets.append([r,w,0])
 
			else:
				if(data_local[r].max() > -9999):
					self.data.append(data_local[r])	
					self.targets.append([r,0,0])			
				else:
					count = count + 1
					continue
			if last_count >= len(self.data):
				print("ERROR NO DATA RECEIVED FROM", self.filenames[r]) 
			print("SKIPPED", count, "SAMPLES OUT OF", len(self.data), data_local[r].shape, dim1, dim2, self.chan_dim)
		c = list(zip(self.data, self.targets))
		random.shuffle(c)
		self.data, self.targets = zip(*c)
		del c
		self.data_full = np.array(self.data).astype(np.float32) #float32
		self.targets_full = np.array(self.targets).astype(np.int16)
		self.data_full = torch.from_numpy(self.data_full)
		self.targets_full = torch.from_numpy(self.targets_full)
		del self.data
		del self.targets

		print("ERROR", self.data_full.shape)
		self.chan_dim = 1
		if self.transform == None:
			mean_per_channel = []
			std_per_channel = []

 
			if self.subset_training > 0:
				self.data_full = self.dat_full[:self.subset_training,:,:,:]
				self.targets_full = self.dat_full[:self.subset_training,:,:]
			for chan in range(self.data_full.shape[self.chan_dim]):
				#TODO slice to make more generic
				subd = self.data_full[:,chan,:,:]
				inds = np.where(subd <= -9999)
				inds2 = np.where(subd > -9999)
				mean_per_channel.append(np.squeeze(subd[inds2].mean()))
				std_per_channel.append(np.squeeze(subd[inds2].std()))
				subd[inds] = mean_per_channel[chan]
				self.data_full[:,chan,:,:] = subd

			transform_norm = torch.nn.Sequential(
				#transforms.ToTensor(),
				transforms.Normalize(mean_per_channel, std_per_channel)
			)  

			self.transform = transform_norm 

		#print("HERE", self.data_full.shape, len(mean_per_channel), len(std_per_channel))
		self.data_full = self.transform(self.data_full)


		self.next_subset()
 


 
	def __train_scalers__(self, data):
		pass



def main(yml_fpath):
    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)

    #Get config values 
    data_train = yml_conf["data"]["files_train"]

    tile = False
    tile_size = None
    tile_step = None
    tile = yml_conf["data"]["tile"]
    tile_size = yml_conf["data"]["tile_size"]
    tile_step = yml_conf["data"]["tile_step"]

    data_reader =  yml_conf["data"]["reader_type"]
    data_reader_kwargs = yml_conf["data"]["reader_kwargs"]
    fill = yml_conf["data"]["fill_value"]
    chan_dim = yml_conf["data"]["chan_dim"]
    valid_min = yml_conf["data"]["valid_min"]
    valid_max = yml_conf["data"]["valid_max"]
    delete_chans = yml_conf["data"]["delete_chans"]
    subset_count = yml_conf["data"]["subset_count"]
    output_subset_count = yml_conf["data"]["output_subset_count"]
    scale_data = yml_conf["data"]["scale_data"]

    transform_chans = yml_conf["data"]["transform_default"]["chans"]
    transform_values =  yml_conf["data"]["transform_default"]["transform"]

    out_dir = yml_conf["output"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    scaler_fname = os.path.join(out_dir, "dbn_scaler.pkl")
    scaler_type = yml_conf["scaler"]["name"]
    scaler, scaler_train = get_scaler(scaler_type, cuda = use_gpu_pre)

    subset_training = yml_conf["dbn"]["subset_training"]

    os.environ['PREPROCESS_GPU'] = str(int(use_gpu_pre))

    read_func = get_read_func(data_reader)

    stratify_data = None
    if "stratify_data" in yml_conf["dbn"]["training"]:
        stratify_data = yml_conf["dbn"]["training"]["stratify_data"]

    if stratify_data is not None:
        strat_read_func = get_read_func(stratify_data["reader"])
        stratify_data["reader"] = strat_read_func

    x2 = DBNDatasetConv(data_train, read_func, data_reader_kwargs, delete_chans=delete_chans, \
            valid_min=valid_min, valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, \
            transform_values=transform_values, transform=None, subset=subset_count, tile=tile, tile_size=tile_size, tile_step=tile_step,
            subset_training = subset_training)

    if x2.train_indices is not None:
        np.save(os.path.join(out_dir, "train_indices"), x2.train_indices)


    np.save(os.path.join(out_dir, "train_data.indices"), x2.targets_full)
    np.save(os.path.join(out_dir, "train_data"), x2.data_full)

    torch.save(x2.transform.state_dict(), os.path.join(out_dir, "dbn_data_transform.ckpt"))



if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument("-y", "--yaml", help="YAML file for data config.")
        args = parser.parse_args()
        from timeit import default_timer as timer
        start = timer()
        main(args.yaml)
        end = timer()
        print(end - start) # Time in seconds, e.g. 5.38091952400282




