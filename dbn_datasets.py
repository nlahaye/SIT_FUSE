import os
import numpy as np
import random

import sys
sys.setrecursionlimit(4500)

#ML imports
import torch
from learnergy.models import dbn
from sklearn.preprocessing import StandardScaler

class DBNDataset(torch.utils.data.Dataset):

	def __init__(self, filenames, read_func, chunk_size, delete_chans, valid_min, valid_max, fill_value = -9999, chan_dim = 0, scalers = None, transform=None):

		self.filenames = filenames
		self.transform = transform
		self.chunk_size = chunk_size
		self.delete_chans = delete_chans
		self.valid_min = valid_min
		self.valid_max = valid_max
		self.fill_value = fill_value
		self.chan_dim = chan_dim
		self.scalers = scalers
		self.read_func = read_func		


		self.__loaddata__()

	def __loaddata__(self):
		
		data = []
		for i in range(0, len(self.filenames)):
			if os.path.exists(self.filenames[i]):
				dat = self.read_func(self.filenames[i]).astype(np.float64)
				for i in range(dat.ndim):
					if i == self.chan_dim:
						continue
					slc = [slice(None)] * dat.ndim
					slc[i] = slice(0, dat.shape[i] - dat.shape[i] % self.chunk_size)
					dat = dat[tuple(slc)]
				dat = np.delete(dat, self.delete_chans, self.chan_dim)

				if self.valid_min is not None:
					dat[np.where(dat < self.valid_min - 0.00000000005)] = -9999
				if self.valid_max is not None:	
					dat[np.where(dat < self.valid_max - 0.00000000005)] = -9999
				if self.fill_value is not None:
					dat[np.where(dat == self.fill_value)] = -9999
				data.append(dat)

		if self.scalers is None:
			self.__trainscalers__(data)

		for r in range(len(data)):	
			for n in range(data[r].shape[self.chan_dim]):
				slc = [slice(None)] * data[r].ndim
				slc[self.chan_dim] = slice(n, n+1)
				subd = data[r][tuple(slc)]
				subd[np.where(subd > -9999)] = self.scalers[n].transform(subd[np.where(subd > -9999)].reshape(-1, 1)).reshape(-1)
				data[r][tuple(slc)] = subd

		dim1 = 0
		dim2 = 1
		if self.chan_dim == 0:
			dim1 = 1
			dim2 = 2
		elif self.chan_dim == 1:
			dim2 = 2

		self.data = []
		self.targets = []
		for r in range(len(data)):
			for j in range(0, data[r].shape[dim1], self.chunk_size):
				for k in range(0, data[r].shape[dim2], self.chunk_size):
					sub_data_total = []
					for c in range(0, data[r].shape[self.chan_dim]):
						end_j = j + self.chunk_size
						end_k = k + self.chunk_size

						slc = [slice(None)] * data[r].ndim
						slc[self.chan_dim] = slice(c, c+1)
						slc[dim1] = slice(j,end_j)	
						slc[dim2] = slice(k, end_k)
						sub_data = data[r][tuple(slc)]

						sub_data_total.append(sub_data)

					sub_data_total = np.array(sub_data_total)
					if(sub_data_total.min() < -1000):
						continue
					self.data.append(sub_data_total.ravel())
					self.targets.append([r,j,k])

		c = list(zip(self.data, self.targets))
		random.shuffle(c)
		self.data, self.targets = zip(*c)
		self.data = torch.from_numpy(np.array(self.data).astype(np.float32))
		self.targets = torch.from_numpy(np.array(self.targets).astype(np.float32))



	def __trainscalers__(self, data):
		self.scalers = []
		for r in range(len(data)):
			for n in range(data[r].shape[self.chan_dim]):
				if r == 0:
					self.scalers.append(StandardScaler())
					slc = [slice(None)] * data[r].ndim
					slc[self.chan_dim] = slice(n, n+1)
					subd = data[r][tuple(slc)]
					self.scalers[n].partial_fit(subd[np.where(subd > -9999)].reshape(-1, 1))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		sample = self.data[idx]
		if self.transform:
			sample = self.transform(sample)

		return sample, self.targets[idx]



