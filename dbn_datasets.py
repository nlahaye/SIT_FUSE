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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

class DBNDataset(torch.utils.data.Dataset):

	def __init__(self, filenames, read_func, read_func_kwargs, pixel_padding, delete_chans, valid_min, valid_max, fill_value = -9999, chan_dim = 0, transform_chans = [], transform_values = [], scaler = None, scale=False, transform=None, subset=None, train_scaler = False, subset_training = -1):

		self.training = False
		self.filenames = filenames
		self.transform = transform
		self.pixel_padding = pixel_padding
		self.delete_chans = delete_chans
		self.valid_min = valid_min
		self.valid_max = valid_max
		self.fill_value = fill_value
		self.chan_dim = chan_dim
		self.transform_chans = transform_chans
		self.transform_value = transform_values
		self.scaler = scaler
		self.train_scaler = train_scaler
		self.scale = scale
		self.transform = transform
		self.read_func = read_func
		self.read_func_kwargs = read_func_kwargs
		self.subset = subset
		self.subset_training = subset_training
		if self.subset is None:
			self.subset = 1		
		self.current_subset = -1


		self.__loaddata__()

	def __loaddata__(self):
		
		data_local = []
		for i in range(0, len(self.filenames)):
			if (type(self.filenames[i]) == str and os.path.exists(self.filenames[i])) or (type(self.filenames[i]) is list and os.path.exists(self.filenames[i][0])):
				print(self.filenames[i])
				dat = self.read_func(self.filenames[i], **self.read_func_kwargs).astype(np.float64)
				print(dat.shape)
				#dat = dat[:,2000:2100,2000:2100] #TODO REMOVE
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
				dat = np.moveaxis(dat, self.chan_dim, 2)
				data_local.append(dat)

		#del dat
		self.chan_dim = 2
		if self.scale:
			if self.scaler is None or self.train_scaler:
				self.training = True
				self.__train_scaler__(data_local)

			for r in range(len(data_local)):	
					subd = data_local[r]
					subd[np.where(subd > -9999)] = self.scaler.transform(subd[np.where(subd > -9999)].reshape(-1, 1)).reshape(-1)
					data_local[r] = subd
		dim1 = 0
		dim2 = 1
		if self.chan_dim == 0:
			dim1 = 1
			dim2 = 2
		elif self.chan_dim == 1:
			dim2 = 2

		self.data = []
		self.targets = []
		for r in range(len(data_local)):
			count = 0
			print("HERE",  data_local[r].shape)
			last_count = len(self.data)
			for j in range(self.pixel_padding, data_local[r].shape[dim1] - self.pixel_padding):
				for k in range(self.pixel_padding, data_local[r].shape[dim2] - self.pixel_padding):
					sub_data_total = []
					for c in range(0, data_local[r].shape[self.chan_dim]):

						slc = [slice(None)] * data_local[r].ndim
						slc[self.chan_dim] = slice(c, c+1)
						slc[dim1] = slice(j-self.pixel_padding,j+self.pixel_padding+1)	
						slc[dim2] = slice(k-self.pixel_padding, k+self.pixel_padding+1)
						sub_data = data_local[r][tuple(slc)]
						sub_data_total.append(sub_data)

					sub_data_total = np.array(sub_data_total)
					if(sub_data_total.min() <= -9999):
						count = count + 1
						continue
					self.data.append(sub_data_total.ravel())
					self.targets.append([r,j,k])
			if last_count >= len(self.data):
				print("ERROR NO DATA RECEIVED FROM", self.filenames[r]) 
			print("SKIPPED", count, "SAMPLES OUT OF", len(self.data), data_local[r].shape, dim1, dim2, self.chan_dim)
		c = list(zip(self.data, self.targets))
		random.shuffle(c)
		self.data, self.targets = zip(*c)
		self.data_full = np.array(self.data).astype(np.float32)
		#self.data_full = self.data_full * 1e10
		#self.data_full = self.data_full.astype(np.int32)
		self.targets_full = np.array(self.targets).astype(np.int16)
		if self.training and self.subset_training > 0:
			self.data_full = self.data_full[:self.subset_training,:]
			self.targets_full = self.targets_full[:self.subset_training,:]
		del self.data
		del self.targets

		subd = self.data_full[np.where(self.data_full > -9999)]
		print("STATS", subd.min(), subd.max(), subd.mean(), subd.std(), self.data_full.min(), self.data_full.max(), self.data_full.mean(), self.data_full.std())

		self.next_subset()
 

	def next_subset(self):
		self.__set_subset__(1)

	def prev_subset(self):
		self.__set_subset__(-1)

	def has_next_subset(self):
		return self.current_subset <= self.subset-2

	def has_prev_subset(self):
		return self.current_subset > 0

	def __set_subset__(self,increment):
		#TODO: optimize to minimize data duplication - lazy loading & Dask
		if self.subset is not None:
			if (increment < 0 and self.current_subset >= -1*increment) or \
				 (increment > 0 and self.current_subset <= self.subset-increment-1):
					self.current_subset = int(self.current_subset + increment)
			else:
				self.current_subset = 0
			self.subset_inds = sorted([self.current_subset*int(self.data_full.shape[0]/self.subset), \
				(self.current_subset+1)*int(self.data_full.shape[0]/self.subset)])   
			if self.current_subset == self.subset-1:
				self.subset_inds[1] = self.data_full.shape[0]
		else:
			self.subset_inds = [0,self.data_full.shape[0]]		
 
		if not torch.is_tensor(self.data_full): 
			self.data = torch.from_numpy(self.data_full[self.subset_inds[0]:self.subset_inds[1],:])
			self.targets = torch.from_numpy(self.targets_full[self.subset_inds[0]:self.subset_inds[1],:])		
		else:
			self.data = self.data_full[self.subset_inds[0]:self.subset_inds[1],:]
			self.targets = self.targets_full[self.subset_inds[0]:self.subset_inds[1],:]	


	def __train_scaler__(self, data):
		for r in range(len(data)):
                        subd = data[r]
                        self.scaler.partial_fit(subd[np.where(subd > -9999)].reshape(-1, 1))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		if torch.is_tensor(index):
			index = index.tolist()

		sample = self.data[index]
		#if self.transform:
		#	sample = self.transform(sample)

		#sample = sample * 1e10
		#sample = sample.astype(np.int32)

		return sample, self.targets[index]



