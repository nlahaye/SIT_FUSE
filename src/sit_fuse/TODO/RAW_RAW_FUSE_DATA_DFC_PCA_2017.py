"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""

NUMBER_RAW_FUSE_CHANNELS = 18
CHUNK_SIZE = 3


data_fn = [
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_24.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_4.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_31.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_33.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_19.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_23.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_30.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_3.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_22.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_17.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_27.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_0.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_10.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_8.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_25.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_15.npy']

data_fn_test = [
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_5.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_7.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_20.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_29.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_11.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_16.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_12.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_13.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_32.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_1.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_28.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_2.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_21.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_6.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_9.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_18.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_14.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_26.npy']


TRAIN_DATA_OUTPUT = {
"Output":
[#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_24.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_4.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_31.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_33.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_19.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_23.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_30.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_3.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_22.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_17.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_27.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_0.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_10.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_8.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_25.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_15.npy'],
"Clust":[
#"/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_4M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_THRESH_1e-05_BRANCH_5_CLUST500.npy",
"/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_0M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB_PCA4_THRESH_1e-05_BRANCH_5_CLUST500.npy",
"/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_10M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB_PCA4_THRESH_1e-05_BRANCH_5_CLUST500.npy",
"/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_8M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB_PCA4_THRESH_1e-05_BRANCH_5_CLUST500.npy",
"/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_15M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB_PCA4_THRESH_1e-05_BRANCH_5_CLUST500.npy"]}

TEST_DATA_OUTPUT = {
"Output":
[
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_7.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_20.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_29.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_11.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_16.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_12.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_13.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_32.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_1.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_28.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_2.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_21.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_6.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_9.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_18.npy',
#'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_14.npy',
'/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_26.npy'],
"Clust":[
"/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_20M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_THRESH_1e-05_BRANCH_5_CLUST500.npy",
"/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_29M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_THRESH_1e-05_BRANCH_5_CLUST500.npy",
"/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_16M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_THRESH_1e-05_BRANCH_5_CLUST500.npy",
"/data/nlahaye/remoteSensing/RawFuse/Data_FUSE_RAW_26M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_THRESH_1e-05_BRANCH_5_CLUST500.npy"]
}


"""
TEST_DATA_OUTPUT = {
"Output": 
[
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_OUTPUT17.npy',
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_OUTPUT5.npy',
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_OUTPUT2.npy',   
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_OUTPUT3.npy'
],
"Coord":
[
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_COORD17.npy',
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_COORD5.npy',
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_COORD2.npy',
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_COORD3.npy'
],
"Limits": 
[
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_LIMITS17.npy',
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_LIMITS5.npy',
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_LIMITS2.npy',
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_LIMITS3.npy'
],
"Clust":
[
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_OUTPUT17M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_OUTPUT5M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_OUTPUT2M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TEST_OUTPUT3M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy"
]}


TRAIN_DATA_OUTPUT = {"Output" : 
[
#'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT31.npy',
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT41.npy'],
#'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT42.npy',
#'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT43.npy',
#'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT45.npy'],
"Coord":
[
#'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_COORD31.npy',
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_COORD41.npy',
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_COORD42.npy',
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_COORD43.npy',
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_COORD45.npy'
],
"Limits":
[
#'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_LIMITS31.npy',
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_LIMITS41.npy',
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_LIMITS42.npy',
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_LIMITS43.npy',
'/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_LIMITS45.npy'
],
"Clust":
[
#"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT31M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT41M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT42M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT43M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy",
"/data/nlahaye/dev/remotesensing/remotesensing/src/model/output/output/FUSE_DFC_2017_RBM_OVERLAP/RAW_FUSE_DFC_TRAIN_OUTPUT45M1B2_SVM_COMP_MULTI_SCENE_DBN1_REV3_SUB4_SCALED_THRESH_1e-05_BRANCH_5_CLUST500.npy"
]
}
"""
