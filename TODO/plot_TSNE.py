import scipy
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import markers
from pprint import pprint
from matplotlib.colors import ListedColormap
from CMAP import CMAP, CMAP_COLORS


DATA = [
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_OUTPUT17M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P100.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_OUTPUT17M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P2500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_OUTPUT17M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_OUTPUT2M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P100.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_OUTPUT2M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_OUTPUT3M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P100.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_OUTPUT3M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_OUTPUT5M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P100.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_OUTPUT5M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P2500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_OUTPUT5M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT31M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P100.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT31M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P1500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT31M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P2500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT31M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT41M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P100.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT41M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P1500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT41M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P2500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT41M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT42M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P100.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT42M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P1500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT42M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P2500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT42M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT43M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P100.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT43M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P1500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT43M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P2500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT43M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT45M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P100.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT45M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P1500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT45M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P2500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT45M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy.TSNE.P500.npy"
]




for i in range(len(DATA)):
	x = np.load(DATA[i])
	c = np.load(COORD[i])
	
	final 


                                final = plot_tsne(tsne_data, clust_data, test_coord)
                                if final is None:
                                        print("BAD TSNE, MOVING ON")
                                        continue

                                fname = td[l]["Clust"][j] + ".TSNE.P" + str(p) + ".npy"
                                np.save(fname, final)
                                cmap = ListedColormap(CMAP_COLORS[0:500])
                                plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=final.astype(np.int32), alpha=0.2)
                                plt.savefig(fname + ".png", dpi=400)




def plot_tsne(tsne_data, clust_flat, coord):

        clust_tsne = np.zeros((coord[:,1,:].max(), coord[:,2,:].max()))
        for pt in range(coord.shape[0]):
            clust_tsne[coord[pt,1], coord[pt,2]] = clust_flat[pt]
	           clust_flat[pt] = clust_data[coord[pt,1], coord[pt,2]]
        #       print(data.shape, tsne_data.shape, clust_data.shape)
        #       print(tsne_data[pt,0], tsne_data[pt,1])
        #       print(coord[pt,1], coord[pt,2]) 
        #       data[tsne_data[pt,0], tsne_data[pt,1]] = clust_data[coord[pt,1], coord[pt,2]]

        #clust_flat = np.array(clust_flat)
        #cmap = ListedColormap(CMAP_COLORS[0:500])
        #plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=clust_flat.astype(np.int32), alpha=0.2)        


        return clust_flat


