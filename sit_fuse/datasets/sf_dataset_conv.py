"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
from timeit import default_timer as timer
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
from sit_fuse.datasets.sf_dataset import SFDataset

from sit_fuse.utils import numpy_to_torch, read_yaml, get_read_func, get_scaler

import argparse

from skimage.util import view_as_windows

class SFDatasetConv(SFDataset):

	def __init__(self):
		pass

	def read_data_preprocessed(self, data_filename, indices_filename, transform = None):

		self.data_full = np.load(data_filename, allow_pickle=True)
		self.targets_full = np.load(indices_filename, allow_pickle=True)

		self.scale = False
		self.scaler = None
		self.train_indices = None
		self.transform = transform
		self.data = self.data_full
		self.targets = self.targets_full


	def read_and_preprocess_data(self, filenames, read_func, read_func_kwargs, delete_chans, valid_min, valid_max, fill_value = -999999, chan_dim = 0, transform_chans = [], transform_values = [], transform=None, tile = False, tile_size = None, tile_step = None, subset_training = -1, stratify_data = None, data_fraction = 1, data_fraction_index = 1):
		#Scaler info isnt used here, but keeping same interface as SFDataset

                #TODO Employ stratification
		self.train_indices = None
 
		self.scaler = None
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
		self.tile = tile
		self.tile_size = tile_size
		self.tile_step = tile_step
		self.stratify_data = stratify_data
		self.subset_training = subset_training
	
		self.data_fraction = data_fraction
		self.data_fraction_index = data_fraction_index

		self.__loaddata__()

	def __loaddata__(self):
		
		data_local = []
		strat_local = []
		dat_begin = []
		for i in range(0, len(self.filenames)):
			if (type(self.filenames[i]) == str and os.path.exists(self.filenames[i])) or (type(self.filenames[i]) is list and os.path.exists(self.filenames[i][1])):
				print(self.filenames[i])
				strat_data = None
				dat = self.read_func(self.filenames[i], **self.read_func_kwargs).astype(np.float32)
				if self.stratify_data is not None and "kmeans" not in self.stratify_data:
					strat_data = self.stratify_data["reader"](self.stratify_data["filename"][i], \
						**self.stratify_data["reader_kwargs"])
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
					dat[np.where(dat < self.valid_min - 0.00000000005)] = -999999
				if self.valid_max is not None:	
					dat[np.where(dat > self.valid_max - 0.00000000005)] = -999999
				if self.fill_value is not None:
					dat[np.where(dat == self.fill_value)] = -999999
				dat[np.where(np.logical_not(np.isfinite(dat)))] = -999999
				dat = np.moveaxis(dat, self.chan_dim, 0)
				if self.data_fraction > 1:
					dat_index_1_1 = self.data_fraction_index * int(dat.shape[1]//self.data_fraction)
					dat_index_1_2 = (self.data_fraction_index+1) * int(dat.shape[1]//self.data_fraction)
					#dat_index_2_2 = (self.data_fraction_index+1) * int(dat.shape[2]//self.data_fraction)
					if self.data_fraction_index == self.data_fraction -1:
						dat_index_1_2 = dat.shape[1]
						#dat_index_2_2 = dat.shape[2]
					#dat_index_2_1 = self.data_fraction_index * int(dat.shape[2]//self.data_fraction)
					dat_begin.append([dat_index_1_1, 0])
					dat = dat[:,dat_index_1_1:dat_index_1_2, :]
				else:
					dat_begin.append([0,0])  

				data_local.append(dat)
				
				if strat_data is not None:
					#TODO Generalize to multi-class
					strat_data = strat_data.astype(np.int32)
					strat_data[np.where(strat_data < 0)] = 0
					strat_data[np.where(strat_data > 0)] = 1
					strat_local.append(strat_data)


		del dat 
		dim1 = 1
		dim2 = 2
		self.chan_dim = 0
		self.n_chans = data_local[0].shape[self.chan_dim]

		self.data = []
		self.targets = []
		if self.tile:
			window_size = [0,0,0]
			tile_step_final = [0,0,0]
			window_size[dim1] = self.tile_size[0]
			window_size[dim2] = self.tile_size[1]
			tile_step_final[dim1] = self.tile_step[0]
			tile_step_final[dim2] = self.tile_step[1]
			tile_step_final[self.chan_dim] = data_local[0].shape[self.chan_dim]#*2
			window_size[self.chan_dim] = data_local[0].shape[self.chan_dim]#*2
			window_size = tuple(window_size)
		self.stratify_training = []
		for r in range(len(data_local)):
			count = 0
			last_count = len(self.data)
			sub_data_total = []
			
			#TODO - 0 imputation - fix with mean later
			data_local[r][np.where(data_local[r] <= -999999)] = 0.0	

			pixel_padding = (window_size[dim1] - 1) //2

			tgts = np.indices(data_local[r].shape[1:])
			tgts[0,:,:] = tgts[0,:,:] + dat_begin[r][0]
			tgts[1,:,:] = tgts[1,:,:] + dat_begin[r][1]
			#tgts = tgts[:,pixel_padding:tgts.shape[1] - pixel_padding,pixel_padding:tgts.shape[2] - pixel_padding]
			tgts = np.concatenate((np.full((1,tgts.shape[1], tgts.shape[2]),r, dtype=np.int16), tgts), axis=0)
	
			
			if self.tile:
				tmp = np.squeeze(view_as_windows(data_local[r], window_size, step=tile_step_final))
				tmp = tmp.reshape((tmp.shape[0]*tmp.shape[1], tmp.shape[2], tmp.shape[3], tmp.shape[4]))			


				window_size_tgt = [3,window_size[1],window_size[2]]
				tile_step_tgt = [3,tile_step_final[1],tile_step_final[2]]
				print(tgts.shape, tmp.shape, data_local[r].shape, window_size_tgt, tile_step_tgt)
				tgts2 = np.squeeze(view_as_windows(tgts, window_size_tgt, step=tile_step_tgt))
				cntr = int(tile_step_final[1]/2)
				print(tgts2.shape)
				tgts2 = tgts2[:,:,:,cntr,cntr]
				tgts2 = tgts2.reshape(tgts2.shape[0]*tgts2.shape[1], tgts2.shape[2])
				#tgts = tgts.reshape((3,tgts.shape[1]*tgts.shape[2])).astype(np.int16)
				#tgts = np.swapaxes(tgts, 0, 1)     


				if isinstance(self.data, list):
					self.data = tmp
					self.targets = tgts2
				else:
					self.data = np.append(self.data, tmp, axis=0)
					self.targets = np.append(self.targets, tgts2, axis=0)		
			else:
				if(data_local[r].max() > -999999):
					np.append(self.data,data_local[r], axis = 0)	
					np.append(self.targets,[r,0,0], axis = 0)			
				else:
					count = count + 1
					continue
			if last_count >= len(self.data):
				print("ERROR NO DATA RECEIVED FROM", self.filenames[r]) 
			elif len(strat_local) > 0:
				pass
				#TODO FIX
				#Preprocess stratification data in same manner
				#if len(strat_local) > 0:
				#	strat_local[r] = strat_local[r][self.pixel_padding:strat_local[r].shape[1] - \
				#	self.pixel_padding,self.pixel_padding:strat_local[r].shape[1] - self.pixel_padding]
				#print(strat_local[r].shape)
				#sub_data_strat = np.squeeze(strat_local[r].flatten())
				#self.stratify_training.append(sub_data_strat)
			print("SKIPPED", count, "SAMPLES OUT OF", len(self.data), data_local[r].shape, dim1, dim2, self.chan_dim)
		print("SHUFFLING", self.data.shape, self.targets.shape)
		p = np.random.permutation(self.data.shape[0])
		self.data_full = torch.from_numpy(self.data[p]) #np.array(self.data).astype(np.float32) #float32
		self.targets_full = torch.from_numpy(self.targets[p])  #np.array(self.targets).astype(np.int16)
		if len(self.stratify_training) > 0:
			self.stratify_training = torch.from_numpy(self.stratify_training[p])
 
		del self.data
		del self.targets


		#Subset data for training and/or stratify
		if self.subset_training > 0 or (self.stratify_data and self.stratify_data["kmeans"]):
			if len(self.stratify_training) > 0:
				self.stratify_training = np.array(self.stratify_training)
				self.stratify_training = self.stratify_training.reshape((-1))
				self.__stratify_training__()
			elif self.stratify_data and self.stratify_data["kmeans"]:
				self.__stratify_k_means__(flatten=True)
			else:
				self.data_full = self.data_full[:self.subset_training,:,:,:]
				self.targets_full = self.data_full[:self.subset_training,:,:]

		self.chan_dim = 1
		self.mean_per_channel = []
		self.std_per_channel = []
		if self.transform == None:
			mean_per_channel = []
			std_per_channel = []

 
			for chan in range(0, self.data_full.shape[self.chan_dim]):
				#TODO slice to make more generic
				subd = self.data_full[:,chan,:,:]
				inds = np.where(subd <= -999999)
				inds2 = np.where(subd > -999999)
				mean_per_channel.append(np.squeeze(subd[inds2].mean()))
				std_per_channel.append(np.squeeze(subd[inds2].std()))
				subd[inds] = mean_per_channel[chan]
				self.data_full[:,chan,:,:] = subd

			transform_norm = torch.nn.Sequential(
				#transforms.ToTensor(),
				transforms.Normalize(mean_per_channel, std_per_channel)
			)  
                       
			self.mean_per_channel = mean_per_channel
			self.std_per_channel = std_per_channel
			self.transform = transform_norm 

		self.data_full = self.transform(self.data_full)
		self.data = self.data_full
		self.targets = self.targets_full


	def __len__(self):
		"""
		Overriding of Dataset internal function __len__.
        
		:return: Number of samples.
		"""
		return len(self.data)

	def __getitem__(self, idx):
		"""
		Overriding of Dataset internal function __getitem__.
		
		:param idx: Index of sample to be returned.

		:return: Sample and associated idx.
		"""
		if torch.is_tensor(idx):
			idx = idx.tolist()

		sample = self.data[idx]
		#print("HERE STATS:", sample.shape, sample.min(), sample.max(), sample.mean(), sample.std())
		#if self.transform is not None:
		#	print("TRANSFORMING")
		#	sample = self.transform(sample)
		#print("HERE STATS_2:", sample.shape, sample.min(), sample.max(), sample.mean(), sample.std())

		#sample = sample * 1e10 
		#sample = sample.astype(np.int32)

		return sample, self.targets[idx]


 
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
    scale_data = yml_conf["data"]["scale_data"]

    transform_chans = yml_conf["data"]["transform_default"]["chans"]
    transform_values =  yml_conf["data"]["transform_default"]["transform"]

    out_dir = yml_conf["output"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    scaler_fname = os.path.join(out_dir, "encoder_scaler.pkl")
    scaler_type = yml_conf["scaler"]["name"]

    scaler, scaler_train = get_scaler(scaler_type)

    subset_training = yml_conf["encoder"]["subset_training"]

    read_func = get_read_func(data_reader)

    stratify_data = None
    if "stratify_data" in yml_conf["encoder"]["training"]:
        stratify_data = yml_conf["encoder"]["training"]["stratify_data"]
 
    if stratify_data is not None and "kmeans" not in stratify_data:
        strat_read_func = get_read_func(stratify_data["reader"])
        stratify_data["reader"] = strat_read_func

    x2 = SFDatasetConv()
    x2.read_and_preprocess_data(data_train, read_func, data_reader_kwargs, delete_chans=delete_chans, \
            valid_min=valid_min, valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, \
            transform_values=transform_values, transform=None, tile=tile, tile_size=tile_size, tile_step=tile_step,
            subset_training = subset_training, stratify_data=stratify_data)

    if x2.train_indices is not None:
        np.save(os.path.join(out_dir, "train_indices"), x2.train_indices)


    np.save(os.path.join(out_dir, "train_data.indices"), x2.targets_full)
    np.save(os.path.join(out_dir, "train_data"), x2.data_full)

    state_dict = {"mean_per_channel" : x2.mean_per_channel, "std_per_channel" : x2.std_per_channel}
    torch.save(state_dict, os.path.join(out_dir, "encoder_data_transform.ckpt"))



if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument("-y", "--yaml", help="YAML file for data config.")
        args = parser.parse_args()
        from timeit import default_timer as timer
        start = timer()
        main(args.yaml)
        end = timer()
        print(end - start) # Time in seconds, e.g. 5.38091952400282




