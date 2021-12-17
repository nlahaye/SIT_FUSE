import os
import numpy as np
import random

import sys
sys.setrecursionlimit(4500)

#ML imports
import torch
from sklearn.preprocessing import StandardScaler

class DBNDataset(torch.utils.data.Dataset):

	def __init__(self, filenames, read_func, pixel_padding, delete_chans, valid_min, valid_max, fill_value = -9999, chan_dim = 0, transform_chans = [], transform_values = [], scalers = None, scale=False, transform=None):

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
		self.scalers = scalers
		self.scale = scale
		self.transform = transform
		self.read_func = read_func		


		self.__loaddata__()

	def __loaddata__(self):
		
		data_local = []
		for i in range(0, len(self.filenames)):
			if os.path.exists(self.filenames[i]):
				print(self.filenames[i])
				dat = self.read_func(self.filenames[i]).astype(np.float64)
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
									

				dat = np.delete(dat, self.delete_chans, self.chan_dim)

				if self.valid_min is not None:
					dat[np.where(dat < self.valid_min - 0.00000000005)] = -9999
				if self.valid_max is not None:	
					dat[np.where(dat > self.valid_max - 0.00000000005)] = -9999
				if self.fill_value is not None:
					dat[np.where(dat == self.fill_value)] = -9999
				data_local.append(dat)


		if self.scale:
			if self.scalers is None:
				self.__train_scalers__(data_local)

			for r in range(len(data_local)):	
				for n in range(data_local[r].shape[self.chan_dim]):
					slc = [slice(None)] * data_local[r].ndim
					slc[self.chan_dim] = slice(n, n+1)
					subd = data_local[r][tuple(slc)]
					print("HERE SUBD SCALING SIZE", subd[np.where(subd > -9999)].shape, subd.shape)	
					subd[np.where(subd > -9999)] = self.scalers[n].transform(subd[np.where(subd > -9999)].reshape(-1, 1)).reshape(-1)
					data_local[r][tuple(slc)] = subd

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
			print("SKIPPED", count, "SAMPLES OUT OF", len(self.data), data_local[r].shape, dim1, dim2, self.chan_dim)
		c = list(zip(self.data, self.targets))
		random.shuffle(c)
		self.data, self.targets = zip(*c)
		self.data = torch.from_numpy(np.array(self.data).astype(np.float32))
		self.targets = torch.from_numpy(np.array(self.targets).astype(np.float32))



	def __train_scalers__(self, data):
		self.scalers = []
		print(len(data))
		for r in range(len(data)):
			print(r, data[r].shape, self.chan_dim)
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



