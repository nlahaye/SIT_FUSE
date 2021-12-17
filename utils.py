import torch
import yaml
import numpy as np

def torch_to_numpy(trch):
        return trch.numpy()

def numpy_to_torch(npy):
        return torch.from_numpy(npy)


def read_yaml(fpath_yaml):
    yml_conf = None
    with open(fpath_yaml) as f_yaml:
        yml_conf = yaml.load(f_yaml, Loader=yaml.FullLoader)
    return yml_conf


def get_read_func(data_reader):
	if data_reader == "numpy":
		return np.load
	if data_reader == "torch":
		return torch.load
	#TODO return BCDP reader
	return None

