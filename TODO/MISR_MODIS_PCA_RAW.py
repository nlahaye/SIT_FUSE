import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
 
from MISR_MODIS_FUSE_OUTPUT_RAW_PCA import TRAIN_DATA_OUTPUT, TEST_DATA_OUTPUT, DROP_CHANS
from sklearn import metrics, manifold
from pprint import pprint
import numpy as np

from sklearn.decomposition import PCA

import os
import logging
import argparse

import scipy

import sys
from matplotlib.colors import ListedColormap
from CMAP import CMAP, CMAP_COLORS
sys.setrecursionlimit(4500)

#PERPLEXITY = [750, 1000, 1500]
PERPLEXITY = [2200] 

#PERPLEXITY = [5, 10, 20, 50, 2500, 500, 100]
N_ITER = 5000
STOP_NO_PROGRESS = 1000
N_JOBS = 50
LEARNING_RATE = 200
LANDSAT_FILL = -99999

logging.basicConfig()
LOGGER = logging.getLogger("lrn2")
LOGGER.setLevel(logging.DEBUG)

from sklearn.preprocessing import StandardScaler
scalers = [None]*73
TRAIN = True

import openTSNE


def trainScalers(filenames, chunk_size=5):

    data = []
    for i in range(0, len(filenames)):
        if os.path.exists(filenames[i]):
            dat = np.load(filenames[i]).astype(np.float64)
            dat = dat[:,:dat.shape[1]-2,:dat.shape[2]-2]
            for t in range(36, dat.shape[0]):
                inds = np.where(dat[t] < -100.0)
                tmp = dat[t]
                tmp[inds] = -2.0
            for t in range(len(DROP_CHANS)):
                dat = np.delete(dat, DROP_CHANS[t]-t, 0)
            #dat = np.delete(dat, 66, 0)
            #dat[np.where(dat < 0.000000005)] = -9999
            contCount = 0
            #print(dat.shape)
            data.append(dat)

    global scalers
    for r in range(len(data)):
        for n in range(data[r].shape[0]):
            if scalers[n] is None:
                if n > 3:
                    scalers[n] = StandardScaler()
                else:
                    scalers[n] = StandardScaler()
            subd = data[r][n,:,:]
            scalers[n].partial_fit(subd[np.where(subd > -9999)].reshape(-1, 1))




def test_generic():

	fns = [] #fnames
	td = [[]]
	
	td = [TEST_DATA_OUTPUT]
	#td.append(TRAIN_DATA_OUTPUT)
	trn = [TRAIN_DATA_OUTPUT]

	for l in range(len(td)):
		fns.append([])
		for i in range(len(td[l]["Output"])):
			fns[l].append(td[l]["Output"][i]) 
			



	trainScalers(trn[0]["Output"])

	trainData = None
	global scalers
	for i in range(len(trn[0]["Output"])):
		tmp, coord = format_data(trn[0]["Output"][i], None)
		print(tmp.shape, coord.shape, "HERE")
		if i == 0:
			trainData = tmp
		else:   
			print(trainData.shape, tmp.shape, "CONCAT")
			trainData = np.concatenate((trainData,tmp))
			print(trainData.shape, "TD SHAPE")


	print(trainData.shape)
	pca = PCA(n_components=0.99)
	X_pca = pca.fit(trainData)

	del trainData
	del tmp

	for l in range(len(td)-1, -1, -1):
		for j in range(len(td[l]["Output"])):	
				test_data, coord = format_data(td[l]["Output"][j], td[l]["Clust"][j])
				print(test_data.shape)
				clust_data = np.load(td[l]["Clust"][j])
				test_chunks = [3]

				sys.stdout.flush()

				print(test_data.shape)
				print(test_data.shape)

				print(test_data.shape, clust_data.shape)
	
					

				tsne_data = pca.transform(test_data)
				final = plot_tsne(tsne_data, clust_data, coord)
				if final is None:
					print("BAD TSNE, MOVING ON")
					continue

				fname = td[l]["Clust"][j] + ".PCA99" + ".LR500.npy"
				np.save(fname, tsne_data)
				cmap = ListedColormap(CMAP_COLORS[0:500])
				img = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=final.astype(np.int32), alpha=0.2)
				img.set_cmap(cmap)
				plt.savefig(fname + ".png", dpi=400)
				plt.clf()
				                


def plot_tsne(tsne_data, clust_data, coord):


	fnl = max(tsne_data[:,0])
	fnl2 = max(tsne_data[:,1])
	strt = 0
	strt2 = 0


	clust_flat = np.zeros((coord.shape[0], 1))
	for pt in range(coord.shape[0]):
		clust_flat[pt] = clust_data[coord[pt,1], coord[pt,2]]

	return clust_flat


def format_data(fname, clust):

    dataRet = None
    if os.path.exists(fname):
        dat = np.load(fname).astype(np.float64)
        if clust is not None:
            clust_data = np.load(clust)
            clust_data = clust_data[1:clust_data.shape[0]-1, 1:clust_data.shape[1]-1]
            dat = dat[:,:clust_data.shape[0], :clust_data.shape[1]]
        else:
            dat = dat[:,:dat.shape[1]-2,:dat.shape[2]-2]

        for t in range(36, dat.shape[0]):
            inds = np.where(dat[t] < -100.0)
            tmp = dat[t]
            tmp[inds] = -2.0
        for t in range(len(DROP_CHANS)):
            dat = np.delete(dat, DROP_CHANS[t]-t, 0)

        global scalers
        for n in range(dat.shape[0]):
            subd = dat[n, :, :]
            subd[np.where(subd > -9999)] = scalers[n].transform(subd[np.where(subd > -9999)].reshape(-1, 1)).reshape(-1)
            dat[n, :, :] = subd



        contCount = 0

        dataRet = []
        dataCoord = []
        for j in range(1, dat.shape[1]-1):
            for k in range(1, dat.shape[2]-1):
                sub_data_total = []
                for c in range(0, dat.shape[0]):
                    sub_data = dat[c, j-1:j+2, k-1:k+2]
                    sub_data2 = sub_data.reshape(9)
                    sub_data_total.append(sub_data2)

                sub_data_total = np.array(sub_data_total)
                #if(sub_data_total[36:].min() < -1000):
                if(sub_data_total.min() < -1000):
                    contCount = contCount + 1
                    continue

                if(sub_data_total.shape[1] != 9):
                    print("ERROR:", sub_data_total.shape)

                #dataRet.append(sub_data_total[:36].ravel())
                dataRet.append(sub_data_total.ravel())
                dataCoord.append([0,j,k])
    return np.array(dataRet).astype(np.float32), np.array(dataCoord).astype(np.int8)


def main():
	global scalers
	parser = argparse.ArgumentParser(description = "Run a complete lrn2 work flow")

	test_generic()
	




if __name__ == '__main__':
    main()


	




