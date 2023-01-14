"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
TEST_DATA_OUTPUT = {
"Output" : [
"/data/nlahaye/remoteSensing/MisrModisFuse/Fuse2_9Cam.npy",
"/data/nlahaye/remoteSensing/MisrModisFuse/Fuse5_9Cam.npy",
"/data/nlahaye/remoteSensing/MisrModisFuse/Fuse9_9Cam.npy",
"/data/nlahaye/remoteSensing/MisrModisFuse/FuseArctic_9Cam.npy",
"/data/nlahaye/remoteSensing/MisrModisFuse/FuseAus_9Cam.npy"],
#"/data/nlahaye/remoteSensing/MisrModisFuse/Fuse4_9Cam.npy",
#"/data/nlahaye/remoteSensing/MisrModisFuse/Fuse1_9Cam.npy",
#"/data/nlahaye/remoteSensing/MisrModisFuse/Fuse3_9Cam.npy"],
"Clust" : [
"/data/nlahaye/remoteSensing/MisrModisFuse/Fuse2_9CamM1B2_SVM_COMP_MULTI_SCENE_SILH_DBN1_REV3_SUB_PCA2_THRESH_1e-05_BRANCH_5_CLUST500.npy",
"/data/nlahaye/remoteSensing/MisrModisFuse/Fuse5_9CamM1B2_SVM_COMP_MULTI_SCENE_SILH_DBN1_REV3_SUB_PCA2_THRESH_1e-05_BRANCH_5_CLUST500.npy",
"/data/nlahaye/remoteSensing/MisrModisFuse/Fuse9_9CamM1B2_SVM_COMP_MULTI_SCENE_SILH_DBN1_REV3_SUB_PCA2_THRESH_1e-05_BRANCH_5_CLUST500.npy",
"/data/nlahaye/remoteSensing/MisrModisFuse/FuseArctic_9CamM1B2_SVM_COMP_MULTI_SCENE_SILH_DBN1_REV3_SUB_PCA2_THRESH_1e-05_BRANCH_5_CLUST500.npy",
"/data/nlahaye/remoteSensing/MisrModisFuse/FuseAus_9CamM1B2_SVM_COMP_MULTI_SCENE_SILH_DBN1_REV3_SUB_PCA2_THRESH_1e-05_BRANCH_5_CLUST500.npy"],
#"/data/nlahaye/remoteSensing/MisrModisFuse/Fuse4_9CamM1B2_SVM_COMP_MULTI_SCENE_SILH_DBN1_REV3_SUB1_THRESH_1e-05_BRANCH_5_CLUST500.npy",
#"/data/nlahaye/remoteSensing/MisrModisFuse/Fuse1_9CamM1B2_SVM_COMP_MULTI_SCENE_SILH_DBN1_REV3_SUB1_THRESH_1e-05_BRANCH_5_CLUST500.npy",
#"/data/nlahaye/remoteSensing/MisrModisFuse/Fuse3_9CamM1B2_SVM_COMP_MULTI_SCENE_SILH_DBN1_REV3_SUB1_THRESH_1e-05_BRANCH_5_CLUST500.npy"]
}

TRAIN_DATA_OUTPUT = {
"Output" : [
"/data/nlahaye/remoteSensing/MisrModisFuse/Fuse1_9Cam.npy",
"/data/nlahaye/remoteSensing/MisrModisFuse/Fuse3_9Cam.npy",
"/data/nlahaye/remoteSensing/MisrModisFuse/Fuse4_9Cam.npy"
]
}
#DROP_CHANS = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 51, 53, 54, 56, 66]
#DROP_CHANS = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 53, 66]
DROP_CHANS = [66]

