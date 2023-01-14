"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
import numpy as np
import zarr
from utils import numpy_to_torch, read_yaml, get_read_func
from osgeo import gdal, osr
import argparse
import os

from skimage.util import view_as_windows
 
def tile_data(yml_conf):

    data_reader =  yml_conf["data"]["reader_type"]
    data_reader_kwargs = yml_conf["data"]["reader_kwargs"]
    filenames = yml_conf["data"]["filenames"]
    channel_dim = yml_conf["data"]["chan_dim"]

    read_func = get_read_func(data_reader)

    window_size = yml_conf["data"]["window_size"]
    step_size = yml_conf["data"]["step_size"]
 
    output_dir = yml_conf["output_dir"]
 
    for i in range(len(filenames)):

        if len(filenames) > 0:
            dat = read_func(filenames[i], **data_reader_kwargs).astype(np.float64)
            if len(dat.shape) < 3:
                dat = np.expand_dims(dat, channel_dim)
                print(dat.shape)

            dat = np.moveaxis(dat, channel_dim, 2)

            print(dat.shape)
            windw_dat = np.squeeze(view_as_windows(dat, window_size, step=step_size))
            print(windw_dat.shape)
            print(os.path.splitext(os.path.basename(filenames[i][0]))[0])
            out_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(filenames[i][0]))[0])
            os.makedirs(out_dir, exist_ok=True)
            for j in range(0,windw_dat.shape[0]):
                for k in range(0,windw_dat.shape[1]):
                    print("TILE SHAPE", np.squeeze(windw_dat[j,k,:,:,:]).shape, out_dir, output_dir)
                    zarr.save(os.path.join(out_dir, "Tile_" + str(j) + "_" + str(k) + ".zarr"), np.squeeze(windw_dat[j,k,:,:,:]))

def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    tile_data(yml_conf)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    main(args.yaml)



