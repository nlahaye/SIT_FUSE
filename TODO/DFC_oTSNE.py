import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
 
from RAW_FUSE_DATA_DFC_2017 import TRAIN_DATA_OUTPUT, TEST_DATA_OUTPUT
from sklearn import metrics, manifold
from pprint import pprint
import numpy as np


import os
import logging
import argparse

import scipy

import sys
from matplotlib.colors import ListedColormap
from CMAP import CMAP, CMAP_COLORS
sys.setrecursionlimit(4500)

#PERPLEXITY = [750, 1000, 1500]
PERPLEXITY = [10000] 

#PERPLEXITY = [5, 10, 20, 50, 2500, 500, 100]
N_ITER = 5000
STOP_NO_PROGRESS = 1000
N_JOBS = 50
LEARNING_RATE = 200

logging.basicConfig()
LOGGER = logging.getLogger("lrn2")
LOGGER.setLevel(logging.DEBUG)

from sklearn.preprocessing import StandardScaler
scalers = [None]*2000
TRAIN = True

import openTSNE

def trainScalers(filenames):

    data = []
    for i in range(0, len(filenames)):
        if os.path.exists(filenames[i]):
            dat = np.load(filenames[i]).astype(np.float32)
            data.append(dat)

    global scalers
    for r in range(len(data)):
        for n in range(data[r].shape[1]):
            if scalers[n] is None:
                scalers[n] = StandardScaler()
            subd = data[r][:,n]
            scalers[n].partial_fit(subd.reshape(-1,1))


def test_generic():

	fns = [] #fnames
	td = [[]]
	
	td = [TEST_DATA_OUTPUT]
	td.append(TRAIN_DATA_OUTPUT)
	trn = [TRAIN_DATA_OUTPUT]

	for l in range(len(td)):
		fns.append([])
		for i in range(len(td[l]["Output"])):
			fns[l].append(td[l]["Output"][i]) 
			



	trainScalers(trn[0]["Output"])

	trainData = None
	global scalers
	for i in range(len(trn[0]["Output"])):
		tmp = np.load(trn[0]["Output"][i]).astype(np.float32)
		for n in range(tmp.shape[1]):
			subd = tmp[:,n]
			subd = scalers[n].transform(subd.reshape(-1,1)).reshape(-1)
			tmp[:,n] = subd

	for n in range(len(PERPLEXITY) -1 , -1, -1):
		p = PERPLEXITY[n]	
		print("PERPLEXITY ", p)
		#tsne = manifold.TSNE(perplexity = p, n_iter = N_ITER, learning_rate = LEARNING_RATE, n_iter_without_progress = STOP_NO_PROGRESS, n_jobs = N_JOBS)

		#aff50 = openTSNE.affinity.PerplexityBasedNN(
		
		for l in range(len(td)-1, -1, -1):
			for j in range(len(td[l]["Output"])):	
				test_data = np.load(td[l]["Output"][j]).astype(np.float32)
				for n in range(test_data.shape[1]):
					subd = test_data[:,n]
					subd = scalers[n].transform(subd.reshape(-1,1)).reshape(-1)
					test_data[:,n] = subd

				test_coord =np.load(td[l]["Coord"][j]).astype(np.int32)
				test_limits = np.load(td[l]["Limits"][j]).astype(np.float32)
				clust_data = np.load(td[l]["Clust"][j])
				test_chunks = [3]

				print("HERE", td[l]["Output"][j], td[l]["Coord"][j], td[l]["Limits"][j])
				sys.stdout.flush()

		
				#tsne_data = tsne.fit_transform(test_data)
				aff500 = openTSNE.affinity.PerplexityBasedNN(test_data,perplexity=2200, n_jobs=50, random_state=20)
				#aff50 = openTSNE.affinity.PerplexityBasedNN(test_dataperplexity=50, n_jobs=50, random_state=20)
				tsne_data = openTSNE.TSNE(n_jobs=50, verbose=True, metric="euclidean", exaggeration = 4,
					random_state=42).fit(affinities=aff500)
				final = plot_tsne(tsne_data, clust_data, test_coord)
				if final is None:
					print("BAD TSNE, MOVING ON")
					continue

				fname = td[l]["Clust"][j] + ".TSNE.P" + str(p) + ".LR500.npy"
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

	#if tsne_data[:,0].max() < 1 or tsne_data[:,1].max() < 1:
	#	return None

	#data = np.zeros(((int)(fnl-strt)+1, (int)(fnl2-strt2)+1)) - 1

	#print(tsne_data.min(), tsne_data.max(), tsne_data.mean(), tsne_data.std())

	clust_flat = np.zeros((coord.shape[0], 1))
	for pt in range(coord.shape[0]):
		clust_flat[pt] = clust_data[coord[pt,1], coord[pt,2]]
	#	print(data.shape, tsne_data.shape, clust_data.shape)
	#	print(tsne_data[pt,0], tsne_data[pt,1])
	#	print(coord[pt,1], coord[pt,2])	
	#	data[tsne_data[pt,0], tsne_data[pt,1]] = clust_data[coord[pt,1], coord[pt,2]]
	
	#clust_flat = np.array(clust_flat)
	#cmap = ListedColormap(CMAP_COLORS[0:500])
	#plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=clust_flat.astype(np.int32), alpha=0.2)	
	 

	return clust_flat



def main():
	global scalers
	parser = argparse.ArgumentParser(description = "Run a complete lrn2 work flow")

	test_generic()
	




if __name__ == '__main__':
    main()


	




