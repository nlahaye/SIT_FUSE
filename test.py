import glob
import os
import numpy as np
from torch import load
from utils import read_uavsar, read_yaml

def main():
    # dir = "/work/09562/nleet/ls6/output"
    # # insearch = glob.glob(os.path.join(dir, "caldor_26200_21048_013_210825_L090VVVV*.input"))
    # dsearch = glob.glob(os.path.join(dir, "caldor_26200_21048_013_210825_L090VVVV*.data"))
    # isearch = glob.glob(os.path.join(dir, "caldor_26200_21048_013_210825_L090VVVV*.indices"))
    # d = load(dsearch[0]).numpy()
    # i = load(isearch[1])
    # print(d.shape)
    # print(i.shape)

    # f1 = "/work/09562/nleet/ls6/data/uavsar/fire-scenes/L-band/caldor_08200_21049_026_210831_L090_CX_01/caldor_08200_21049_026_210831_L090HHHH_CX_01.grd"
    # a1 = "/work/09562/nleet/ls6/data/uavsar/fire-scenes/L-band/caldor_08200_21049_026_210831_L090_CX_01/uavsar.asf.alaska.edu_UA_caldor_08200_21049_026_210831_L090_CX_01_caldor_08200_21049_026_210831_L090_CX_01.ann"
    # f2 = "/work/09562/nleet/ls6/data/uavsar/fire-scenes/L-band/caldor_26200_21048_013_210825_L090_CX_01/caldor_26200_21048_013_210825_L090HHHH_CX_01.grd"
    # a2 = "/work/09562/nleet/ls6/data/uavsar/fire-scenes/L-band/caldor_26200_21048_013_210825_L090_CX_01/uavsar.asf.alaska.edu_UA_caldor_26200_21048_013_210825_L090_CX_01_caldor_26200_21048_013_210825_L090_CX_01.ann"
    # d1 = read_uavsar(f1, ann_fps=a1)
    # d2 = read_uavsar(f2, ann_fps=a2)
    # print(d1.shape)
    # print(d2.shape)
    
    yml = "/work/09562/nleet/ls6/SIT_FUSE/config/dbn/uavsar_dbn.yaml"
    yml_conf = read_yaml(yml)
    files_test = yml_conf['data']['files_test']
    reader_kwargs = yml_conf['data']['reader_kwargs']
    
    data = None
    for fp in files_test:
        if isinstance(fp, list):
            fname = os.path.basename(fp[0])
        else:
            fname = os.path.basename(fp)
        data = read_uavsar(fp, **reader_kwargs)
    
    data = np.array(data)
    data = np.moveaxis(data, 0, 2)
    data = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))
    data = scaler.
    print(data.shape)
    print(data.shape)
    
if __name__ == '__main__':
    main()
 


