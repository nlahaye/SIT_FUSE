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

import argparse

import sys
sys.setrecursionlimit(4500)

from utils import numpy_to_torch, read_yaml, get_read_func, get_scaler

import pickle
from joblib import load, dump

#ML imports
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.utils import shuffle

from skimage.filters import sobel
from skimage.util import view_as_windows

class DBNDataset(torch.utils.data.Dataset):
	"""
	This class is an extension of the PyTorch Dataset class. It is a specialization built for 2-D datasets used in SIT-FUSE.
	"""

	def __init__(self):
		"""
		Constructor for DBNDataset. Initialization of actual datasets are done in 
		init_from_array, read_data_preprocessed, or read_and_preprocess_data.
		"""
		pass	

	def init_from_array(self, data_full, targets_full, scaler = None, subset=None):
		"""
		Initializes Dataset from pre-existing external array(s).

		:param data_full: Dataset with N_samples x N_features dimensionality.
			:param targets_full: Since unsupervised, targets array consists of indices per sample. The indices are representative of (File_Index, Line_Index, Sample_index) and full shape is thus N_samples X 3.
		:param scaler: Optional The per-feature scaler to train and use with the dataset. If set to None, no scaling will be applied. Default value is None.
			:param subset: Number of samples to subset from full dataset. If set to None, no subsetting is applied. Default is None. 
		"""		

		self.data_full = data_full
		self.targets_full = targets_full

		self.train_indices = None
		self.scaler = scaler
		self.transform = None
		if scaler is not None:
			self.scale = True
		
		self.subset = subset

		if self.subset is None:
			self.subset = 1
		self.current_subset = -1
		
		self.next_subset()	


	def read_data_preprocessed(self, data_filename, indices_filename, scaler = None, subset=None):
		"""
		Initializes Dataset from files that contain preprocessed samples. Data should have N_samples x N_features dimensionality.
	
		:param data_filename: The path to the file that contains data to be loaded.
		:param indices_filename: The path to the filename that contains per-sample (File_Index, Line_Index, Sample_index) indices.
		:param scaler: Optional The per-feature scaler to train and use with the dataset. If set to None, no scaling will be applied. Default value is None.
			:param subset: Number of samples to subset from full dataset. If set to None, no subsetting is applied. Default is None.
		"""		

		self.data_full = np.load(data_filename)
		self.targets_full = np.load(indices_filename)

		self.train_indices = None
		self.scale = False
		self.scaler = scaler
		self.transform = None
		if scaler is not None:
			self.scale = True

		self.subset = subset
	  
		if self.subset is None:
   			self.subset = 1
		self.current_subset = -1
	
		self.next_subset()


	def read_and_preprocess_data(self, filenames, read_func, read_func_kwargs, pixel_padding, delete_chans, valid_min, valid_max, fill_value = -9999, chan_dim = 0, transform_chans = [], transform_values = [], scaler = None, scale=False, transform=None, subset=None, train_scaler = False, subset_training = -1, stratify_data = None):
		"""
		High level initialization function for data ingestion, preprocessesing, and Dataset initialization. Data gets read in in file x channel x line x sample dimensionality and gets preprocessed/changed into n_samples x n_features dimensionality. 
	
		:param filenames: list of paths to files that will be ingested.
		:param read_func: function used to read in data. Common interfaces for readers have been developed and can be seen within utils
		:param read_func_kwargs: keyword args to be passes to read_func.
			:param pixel_padding: Number of pixels to extend per-pixel/per-sample 'neighborhood' away from center sample of focus. Can be 0.
		:param delete_chans: list of channels to be deleted pror to preprocessing. Can be empty.
		:param valid_min: Minimum valid value in data. Anything less will be set to a fill value and not used.
		:param valid_max: Maximum valid value in data. Anything greater  will be set to a fill value and not used.
			:param fill_value: Optional fill value to be used for bad/unusample samples/pixeld. Default value is -9999.
			:param chan_dim: Optional dimension of index that represents channels/bands. Default value is 0.
		:param transform_chans: Optional channels to have special transforms applied to pixels out of expected ranges prior to filling. Default value is empty list ([]).
		:param transform_values: Optional values associated with transform_chans. Values to be used for out of range samples in each of the channels specified in transform_chans. Default value is empty list ([]).
			:param scaler: Optional per-feature scaler to train and use with the dataset. If set to None, no scaling will be applied. Default value is None.
			:param scale: Optional boolean value specifying whether or not to use scaler to scale data. Default value is False.
			:param transform: Optional unused currently, but will be updated to be a transform function to be applied in preprocessing. Used in child classes. Default is None.
			:param subset: Optional number of subsets to break data into. This addition was made to account for memory concerns, but does cause issues if Dataset is being used for training, so should be set to 1 for a Dataset being used for training. Default is 1.
			:param train_scaler: Optional boolean value indicating whether or not to train scaler with data in Dataset. Default is False.
			:param subset_training: Optional number of samples to subset and extract out of full preprocessed set. Typically used for training Datasets. If set to -1, full set of samples is kept. Associated stratification and oversampling techniques being developed. Default is -1. 
			:param stratify_data: Optional dictionary describing data and techniques for stratification of subset. Subset size specified via subset_training. Currently under development and should be left unset/set to None. If set to None, no stratification is done. Default value is None.
		"""

		self.train_indices = None
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
		self.stratify_data = stratify_data
		if self.subset is None:
			self.subset = 1		
		self.current_subset = -1


		self.__loaddata__()

	def __loaddata__(self):
		"""
		Internal function for data ingestion and preprocessing. Should not be interfaced with directly. Use read_and_preprocess_data to properly interface.
		"""

		strat_local = []
		data_local = []
		for i in range(0, len(self.filenames)):
			if (type(self.filenames[i]) == str and os.path.exists(self.filenames[i])) or (type(self.filenames[i]) is list and os.path.exists(self.filenames[i][0])):

				print(self.filenames[i])
				dat = self.read_func(self.filenames[i], **self.read_func_kwargs).astype(np.float64)
				print(dat.shape)
				strat_data = None
				if dat.ndim == 2:
					dat = np.expand_dims(dat, self.chan_dim)
				if self.stratify_data is not None:
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
					dat[tuple(slc)] = tmp
				inds = np.where(dat < self.valid_min - 0.00000000005)
				dat[inds] = -9999
				inds = np.where(dat > self.valid_max - 0.00000000005)
				dat[inds] = -9999
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
				if strat_data is not None:
					#TODO Generalize to multi-class
					strat_data = strat_data.astype(np.int32)
					strat_data[np.where(strat_data < 0)] = 0
					strat_data[np.where(strat_data > 0)] = 1	
					strat_local.append(strat_data)

		#del dat
		self.chan_dim = 2
		if self.scale:
			if self.scaler is None or self.train_scaler:
				self.training = True
				self.__train_scaler__(data_local)

			for r in range(len(data_local)):	
				subd = data_local[r]
				shape = copy.deepcopy(subd.shape)
				subd = subd.reshape(-1, shape[self.chan_dim])
				inds = np.where(subd <= -9998)
				subd = self.scaler.transform(subd)
				subd[inds] = -9999
				subd = subd.reshape(shape)
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
		self.stratify_training = []
		for r in range(len(data_local)):

			size_wind = 1 + 2 * self.pixel_padding
			tgts = np.indices(data_local[r].shape[0:2])
			tgts = tgts[:,self.pixel_padding:tgts.shape[1] - self.pixel_padding,self.pixel_padding:tgts.shape[2] - self.pixel_padding]
			tgts = np.concatenate((np.full((1,tgts.shape[1], tgts.shape[2]),r, dtype=np.int16), tgts), axis=0)
			tgts = tgts.reshape((3,tgts.shape[1]*tgts.shape[2])).astype(np.int16)
			sub_data_total = view_as_windows(data_local[r], [size_wind, size_wind, data_local[r].shape[2]], step=1)
			sub_data_total = sub_data_total.reshape((sub_data_total.shape[0]*sub_data_total.shape[1], -1))		

			del_inds = np.where(sub_data_total <= -9998)[0]
			sub_data_total = np.delete(sub_data_total, del_inds, 0)
			tgts = np.delete(tgts, del_inds, 1)

			if len(strat_local) > 0:
				strat_local[r] = strat_local[r][self.pixel_padding:strat_local[r].shape[1] - \
					self.pixel_padding,self.pixel_padding:strat_local[r].shape[1] - self.pixel_padding]
				print(strat_local[r].shape)
				sub_data_strat = np.squeeze(strat_local[r].flatten()) 
				self.stratify_training.append(sub_data_strat)
					   
			if len(self.data) == 0: 
				self.data.append(sub_data_total)
				self.targets.append(tgts)
			else:
				self.data[0] = np.concatenate((self.data[0],sub_data_total),axis=0)
				self.targets[0] = np.concatenate((self.targets[0],tgts),axis=1)
		self.targets[0] = np.swapaxes(self.targets[0], 0, 1)

		if len(self.stratify_training) > 0:
			c = list(zip(self.data[0], self.targets[0], self.stratify_training[0]))
		else:
			c = list(zip(self.data[0], self.targets[0]))
		random.shuffle(c)
		
		if len(self.stratify_training) > 0:
			self.data, self.targets, self.stratify_training = zip(*c)
		else:
			self.data, self.targets = zip(*c)
		self.data_full = np.array(self.data).astype(np.float32)
		self.targets_full = np.copy(self.targets).astype(np.int16)


		if self.training and self.subset_training > 0:
			if len(self.stratify_training) > 0:
				self.stratify_training = np.array(self.stratify_training)
				self.stratify_training = self.stratify_training.reshape((-1))
				self.__stratify_training__()
			else:
				self.data_full = self.data_full[:self.subset_training,:]
				self.targets_full = self.targets_full[:self.subset_training,:]
		del self.data
		del self.targets

		print("STATS", self.data_full.min(), self.data_full.max(), self.data_full.mean(), self.data_full.std())

		self.next_subset()
 

	def next_subset(self):
		"""
		Shift to next subset within data.
		"""
		self.__set_subset__(1)

	def prev_subset(self):
		"""
		Shift to previous subset within data.
		"""
		self.__set_subset__(-1)

	def has_next_subset(self):
		"""
		Checks is there is a subsequent subset of data
		
		:return: Whether or not there is a subsequent subset of data
		"""
		return self.current_subset <= self.subset-2

	def has_prev_subset(self):
		"""
		Checks is there is a previous subset of data
		
		:return: Whether or not there is a previous subset of data
		"""
		return self.current_subset > 0

		
	def __stratify_training__(self):
		"""
		Internal function to implement stratification. Currently under development and should not be used.
		"""

		num_train_exs = self.subset_training
		counts = []
		type_inds = []
		train_indices = []	
		train_inds_by_value = [] 
		dataset_size = self.stratify_training.shape[0]
				#TEST OVERSAMPLING
 

		#TODO Allow for oversampling, actual stratification, and only selction of a subset of labels
		for mask_val in range(self.stratify_training.max()):
			type_inds.append(np.where(self.stratify_training == mask_val)[0])
			percentage_of_dataset = len(type_inds[mask_val]) / dataset_size
			# calculate how many test & val exaples to take from the given water type
			num_train_exs_of_type = round(percentage_of_dataset * num_train_exs)
			# randomly sample examples from the givenlaprint(len(type_inds[mask_val]), num_train_exs_of_type)
			tmp = np.random.choice(type_inds[mask_val], size=num_train_exs_of_type, replace=False)
			train_indices.extend(tmp)
			train_inds_by_value.append(tmp) 
			

		self.data_full = self.data_full[train_indices]
		self.targets_full = self.targets_full[train_indices]
		self.train_indices = train_inds_by_value


	def __set_subset__(self,increment):
		"""
		Internal function to set the current subset of data. Use next_subset and previous_subset to interface externally.
	
		:param increment: The increment, positive or negative, to be used to identify the new current subset.
		"""
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
		"""
		Internal function to train scaler.
	
		:param data: Data to use to train scaler. partial_fit function used so this process can be done multiple separate times.
		"""
		for r in range(len(data)):
						subd = data[r]
						shape = subd.shape
						self.scaler.partial_fit(subd[np.where(subd > -9999)].reshape(-1, shape[self.chan_dim]))

	def __len__(self):
		"""
		Overriding of Dataset internal function __len__.
	
		:return: Number of samples.
		"""
		return len(self.data)

	def __getitem__(self, index):
		"""
		Overriding of Dataset internal function __getitem__.
	
		:param index: Index of sample to be returned.

		:return: Sample and associated index.
		"""
		if torch.is_tensor(index):
			index = index.tolist()

		sample = self.data[index]
		#if self.transform:
		#	sample = self.transform(sample)

		#sample = sample * 1e10
		#sample = sample.astype(np.int32)

		return sample, self.targets[index]



def main(yml_fpath):
	"""
	Function used if code is called as executable. Generates data and indices in preprocessed format and 
	saves to files. Can be reaccessed via read_data_preprocessed.

	:param yml_fpath: Path to YAML configuration.
	
	Values required to be in YAML configuration file:

	/data : Sub-dictionary that contains parameters about dataset.
	/data/files_train : List of files tp be used for training.
	/data/pixel_padding : Number of pixels to extend per-pixel/per-sample 'neighborhood' away from center sample of focus. Can be 0.
	/data/number_channels : Number of channels to be used from dataset.
	/data/reader_type : Name of reader key (see utils documentation) to get the appropriate data reader function.
	/data/reader_kwargs : Kwargs for reader function.
	/data/fill_value : Fill value to use for unusable pixels/samples.

	"""
	#Translate config to dictionary 
	yml_conf = read_yaml(yml_fpath)

	#Get config values 
	data_train = yml_conf["data"]["files_train"]

	pixel_padding = yml_conf["data"]["pixel_padding"]
	number_channel = yml_conf["data"]["number_channels"]
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

	use_gpu_pre = subset_training = yml_conf["dbn"]["training"]["use_gpu_preprocessing"]

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

	x2 = DBNDataset()
	x2.read_and_preprocess_data(data_train, read_func, data_reader_kwargs, pixel_padding, delete_chans=delete_chans, \
			valid_min=valid_min, valid_max=valid_max, fill_value =fill, chan_dim = chan_dim, transform_chans=transform_chans, \
			transform_values=transform_values, scaler = scaler, train_scaler = scaler_train, scale = scale_data, \
			transform=numpy_to_torch, subset=subset_count, subset_training = subset_training, stratify_data=stratify_data)
 
	if x2.train_indices is not None:
		np.save(os.path.join(out_dir, "train_indices"), x2.train_indices)
		

	np.save(os.path.join(out_dir, "train_data.indices"), x2.targets_full)
	np.save(os.path.join(out_dir, "train_data"), x2.data_full) 
 
	#Save scaler
	with open(os.path.join(out_dir, "dbn_scaler.pkl"), "wb") as f:
		dump(x2.scaler, f, True, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-y", "--yaml", help="YAML file for data config.")
	args = parser.parse_args()
	from timeit import default_timer as timer
	start = timer()
	main(args.yaml)
	end = timer()
	print(end - start) # Time in seconds, e.g. 5.38091952400282







