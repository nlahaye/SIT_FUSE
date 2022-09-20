import os
import argparse
from preprocessing.misc_utils import combine_modis_gtiffs_laads
from utils import read_trop_mod_xr, read_yaml


def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)

    fname_heads = yml_conf["fname_heads"]
    bands = yml_conf["band_counts"]
    fname_type1 = yml_conf["fname_type1"]
    fname_type2 = yml_conf["fname_type2"]

    fnames_full = []
    for i in range(len(fname_heads)):
        fnames = []
        process = True
        for j in range(len(bands)):
            for k in range(bands[j]):
                fname = fname_heads[i] + fname_type1[j] + str(k+1) + fname_type2[j]
                fnames.append(fname)
                if not os.path.exists(fname):
                    process = False
        if process:
            fnames_full.append(fnames)
     
    combine_modis_gtiffs_laads(fnames_full)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)





