

from sit_fuse.utils import concat_numpy_files


files = [
"/data/nlahaye/output/Learnergy/IJEPA_TEST_FULL/train_data_0.indices.npy",
"/data/nlahaye/output/Learnergy/IJEPA_TEST_FULL/train_data_1.indices.npy",
"/data/nlahaye/output/Learnergy/IJEPA_TEST_FULL/train_data_2.indices.npy",
"/data/nlahaye/output/Learnergy/IJEPA_TEST_FULL/train_data_3.indices.npy",
"/data/nlahaye/output/Learnergy/IJEPA_TEST_FULL/train_data_4.indices.npy",
"/data/nlahaye/output/Learnergy/IJEPA_TEST_FULL/train_data_5.indices.npy"
]



final_file = "/data/nlahaye/output/Learnergy/IJEPA_TEST_FULL/train_data.indices.npy"


concat_numpy_files(files, final_file)


