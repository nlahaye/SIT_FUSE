"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
import torch
import os
import numba
import random
import copy
import sys

import numpy as arrop
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


sys.setrecursionlimit(4500)


def filter_samples_cupy(data_local, pixel_padding, chan_dim, filenames):
   
    import cupy as arrop
    from utils_cupy import sliding_window_view
    from cuml.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

    dpix = 2*pixel_padding + 1
    dim1 = 0
    dim2 = 1
    if chan_dim == 0:
        dim1 = 1
        dim2 = 2
    elif chan_dim == 1:
        dim2 = 2
    data = []
    targets = []
    for r in range(len(data_local)):
        sub_data = data_local[r]
        slc = [slice(None)] * sub_data.ndim
        sub_data_win = sliding_window_view(sub_data, [dpix, dpix], axis=[dim1, dim2])
        sub_data_min = sub_data_win.min(axis=(chan_dim, -1, -2))
        sub_data_min = arrop.expand_dims(sub_data_min, 0)
        idx = arrop.argwhere(sub_data_min > -9999)
        if len(idx) == 0:
            print("ERROR NO DATA RECEIVED FROM", filenames[r])
            continue
        slc[dim1], slc[dim2] = idx.T[1:]
        npix = sub_data.shape[chan_dim] * dpix**2
        sub_data_total = sub_data_win[tuple(slc)]
        if chan_dim == 0:
            sub_data_total = arrop.swapaxes(sub_data_total, 0, 1)
        sub_data_total = sub_data_total.reshape(-1, npix)
        data.append(sub_data_total)
        idx[:, 1:] += 1
        idx[:, 0] = r
        targets.append(idx)
        nsamples = sub_data.shape[dim1] * sub_data.shape[dim2]
        count = nsamples - sub_data_total.shape[0]
        print("SKIPPED", count, "SAMPLES OUT OF", nsamples,
              sub_data.shape, dim1, dim2, chan_dim)
    return data, targets

 
@numba.njit
def filter_samples_numba(data_local, pixel_padding, chan_dim, filenames):

    dim1 = 0
    dim2 = 1
    if chan_dim == 0:
        dim1 = 1
        dim2 = 2
    elif chan_dim == 1:
        dim2 = 2
    data = []
    targets = []
    for r in range(len(data_local)):
        count = 0
        last_count = len(data)
        for j in range(pixel_padding, data_local[r].shape[dim1] - pixel_padding):
            for k in range(pixel_padding, data_local[r].shape[dim2] - pixel_padding):
                nchan = data_local[r].shape[chan_dim]
                dpix = 2*pixel_padding + 1
                sub_data_total = arrop.zeros((nchan, dpix, dpix))
                for c in range(0, nchan):
                    slc = [slice(None)] * data_local[r].ndim
                    slc[chan_dim] = slice(c, c+1)
                    slc[dim1] = slice(j-pixel_padding,
                                      j+pixel_padding+1)
                    slc[dim2] = slice(k-pixel_padding,
                                      k+pixel_padding+1)
                    sub_data = data_local[r][slc[0], slc[1], slc[2]]

                    sub_data_total[c] = sub_data

                if (sub_data_total.min() <= -9999):
                    count = count + 1
                    continue
                data.append(sub_data_total.ravel())
                targets.append([r, j, k])
        if last_count >= len(data):
            print("ERROR NO DATA RECEIVED FROM", filenames[r])
        print("SKIPPED", count, "SAMPLES OUT OF", len(data),
              data_local[r].shape, dim1, dim2, chan_dim)
    return data, targets




class DBNDataset(torch.utils.data.Dataset):

    def __init__(self, filenames, read_func, read_func_kwargs, pixel_padding, delete_chans, valid_min, valid_max, fill_value=-9999, chan_dim=0, transform_chans=[], transform_values=[], scalers=None, scale=False, transform=None, subset=None, train_scalers=False):

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
        self.train_scalers = train_scalers
        self.scale = scale
        self.transform = transform
        self.read_func = read_func
        self.read_func_kwargs = read_func_kwargs
        self.subset = subset
        if self.subset is None:
            self.subset = 1
        self.current_subset = -1

        self.device = 'cpu'
        if "PREPROCESS_GPU" in os.environ and os.environ["PREPROCESS_GPU"] == "1": 
            self.device = 'cuda'

        self.__loaddata__()

    def __loaddata__(self):

        if "PREPROCESS_GPU" in os.environ and os.environ["PREPROCESS_GPU"] == "1":
            import cupy as arrop
            from utils_cupy import sliding_window_view
            from cuml.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
        else:
            import numpy as arrop
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
 


        data_local = []
        for i in range(0, len(self.filenames)):
            if (type(self.filenames[i]) == str and os.path.exists(self.filenames[i])) or (type(self.filenames[i]) is list and os.path.exists(self.filenames[i][0])):
                print(self.filenames[i])
                dat = self.read_func(
                    self.filenames[i], **self.read_func_kwargs)#.astype(np.float64)
                #print(dat.shape, dat[arrop.where(dat > -99999)
                #                     ].min(), dat[arrop.where(dat > -99999)].max())
                for t in range(len(self.transform_chans)):
                    slc = [slice(None)] * dat.ndim
                    slc[self.chan_dim] = slice(
                        self.transform_chans[t], self.transform_chans[t]+1)
                    tmp = dat[tuple(slc)]
                    if self.valid_min is not None:
                        inds = arrop.where(tmp < self.valid_min - 0.00000000005)
                        tmp[inds] = self.transform_value[t]
                    if self.valid_max is not None:
                        inds = arrop.where(tmp > self.valid_max - 0.00000000005)
                        tmp[inds] = self.transform_value[t]
                if len(self.transform_chans) > 0:
                    del slc
                    del tmp
                
                keep_chans = arrop.array(list(set(range(dat.shape[self.chan_dim])).difference(set(self.delete_chans))))
                slc = [slice(None)] * dat.ndim
                slc[self.chan_dim] = keep_chans
                dat = dat[tuple(slc)]

                if self.valid_min is not None:
                    dat[arrop.where(dat < self.valid_min - 0.00000000005)] = -9999
                if self.valid_max is not None:
                    dat[arrop.where(dat > self.valid_max - 0.00000000005)] = -9999
                if self.fill_value is not None:
                    dat[arrop.where(dat == self.fill_value)] = -9999
                data_local.append(dat)

        if self.scale:
            if self.scalers is None or self.train_scalers:
                self.__train_scalers__(data_local)

            for r in range(len(data_local)):
                for n in range(data_local[r].shape[self.chan_dim]):
                    slc = [slice(None)] * data_local[r].ndim
                    slc[self.chan_dim] = slice(n, n+1)
                    subd = data_local[r][tuple(slc)]
                    subd[arrop.where(subd > -9999)] = self.scalers[n].transform(
                        subd[arrop.where(subd > -9999)].reshape(-1, 1)).reshape(-1)
                    data_local[r][tuple(slc)] = subd

        f = filter_samples_numba
        if self.device == 'cuda':
            f = filter_samples_cupy

        self.data, self.targets = f(data_local, self.pixel_padding, self.chan_dim, self.filenames)
        self.data = arrop.array(self.data)
        self.targets = arrop.array(self.targets)
        if self.data.shape[0] == 1 and self.data.ndim == 3:
            self.data = self.data[0]
            self.targets = self.targets[0]
        i = arrop.arange(self.data.shape[0])
        arrop.random.shuffle(i)
        self.data, self.targets = self.data[i], self.targets[i]
        self.data_full = self.data.astype(arrop.float32)
        #self.data_full = self.data_full * 1e10
        #self.data_full = self.data_full.astype(np.int32)
        self.targets_full = self.targets.astype(arrop.int16)
        del self.data
        del self.targets

        subd = self.data_full[arrop.where(self.data_full > -9999)]
        print("STATS", subd.min(), subd.max(), subd.mean(), subd.std(), self.data_full.min(
        ), self.data_full.max(), self.data_full.mean(), self.data_full.std())

        self.next_subset()

    def next_subset(self):
        self.__set_subset__(1)

    def prev_subset(self):
        self.__set_subset__(-1)

    def has_next_subset(self):
        return self.current_subset <= self.subset-2

    def has_prev_subset(self):
        return self.current_subset > 0

    def __set_subset__(self, increment):
        # TODO: optimize to minimize data duplication - lazy loading & Dask
        if self.subset is not None:
            if (increment < 0 and self.current_subset >= -1*increment) or \
                    (increment > 0 and self.current_subset <= self.subset-increment-1):
                self.current_subset = int(self.current_subset + increment)
            else:
                self.current_subset = 0
            self.subset_inds = sorted([self.current_subset*int(self.data_full.shape[0]/self.subset),
                                       (self.current_subset+1)*int(self.data_full.shape[0]/self.subset)])
            if self.current_subset == self.subset-1:
                self.subset_inds[1] = self.data_full.shape[0]
        else:
            self.subset_inds = [0, self.data_full.shape[0]]


        self.data = torch.as_tensor(
            self.data_full[self.subset_inds[0]:self.subset_inds[1], :], device=self.device)
        self.targets = torch.as_tensor(
            self.targets_full[self.subset_inds[0]:self.subset_inds[1], :], device=self.device)

    # TODO - ensure that channel dimension is always last and only use one StandardScaler

    def __train_scalers__(self, data):

        if "PREPROCESS_GPU" in os.environ and os.environ["PREPROCESS_GPU"] == "1":
            import cupy as arrop
            from utils_cupy import sliding_window_view
            from cuml.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
        else:
            import numpy as arrop
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

        copy_first_scaler = False
        if self.scalers is None:
            self.scalers = []
        else:
            copy_first_scaler = True
        for r in range(len(data)):
            for n in range(data[r].shape[self.chan_dim]):
                if r == 0:
                    if n > 0 and copy_first_scaler:
                        self.scalers.append(copy.deepcopy(self.scalers[n-1]))
                    elif not copy_first_scaler:
                        self.scalers.append(StandardScaler())
                    # self.scalers.append(MaxAbsScaler())
                slc = [slice(None)] * data[r].ndim
                slc[self.chan_dim] = slice(n, n+1)
                subd = data[r][tuple(slc)]
                #print("CHANNELS", n, r, self.filenames[r], subd.min(
                #), subd.max(), arrop.where(subd > -9999)[0].shape)
                self.scalers[n].partial_fit(
                    subd[arrop.where(subd > -9999)].reshape(-1, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        sample = self.data[index]
        # if self.transform:
        #	sample = self.transform(sample)

        #sample = sample * 1e10
        #sample = sample.astype(np.int32)

        return sample, self.targets[index]
