
import zarr
import argparse
import numpy as np

from utils import read_trop_mod_xr, read_trop_mod_xr_geo, read_yaml




def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)
    data_fname = yml_conf["data"]["filename"]
    data_reader_kwargs = yml_conf["data"]["reader_kwargs"]
 

    data = read_trop_mod_xr(data_fname, **data_reader_kwargs)
    data2 = read_trop_mod_xr_geo(data_fname, **data_reader_kwargs)

    print(data2.shape)
    for i in range(data.shape[1]):
        #dat = np.squeeze(data[:,i,:,:])
        #zarr.save(data_fname + "day_" + str(i), dat)

        zarr.save(data_fname + "day_" + str(i) + "_geo", data2)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)



