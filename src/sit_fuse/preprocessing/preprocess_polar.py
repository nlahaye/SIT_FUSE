"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
import os
import argparse
from sit_fuse.preprocessing.misc_utils import combine_modis_gtiffs, combine_viirs_gtiffs, gen_polar_2_grid_cmds, run_cmd
from sit_fuse.utils import read_trop_mod_xr, read_yaml

def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)

    exe_location = yml_conf["exe_location"]
    data_files = yml_conf["data_fnames"]
    loc_files = yml_conf["loc_fnames"]
    instruments = yml_conf["instruments"]
    out_dirs = yml_conf["out_dirs"]
     
    cmds = gen_polar_2_grid_cmds(exe_location, data_files, loc_files, instruments, out_dirs)
  
    for i in range(len(cmds)):
        run_cmd(cmds[i])

    #combine_modis_gtiffs([out_dirs[0]])
    #combine_viirs_gtiffs([out_dirs[0]])



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)





