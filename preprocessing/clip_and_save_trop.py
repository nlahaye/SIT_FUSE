"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
from utils import read_yaml, clip_and_save_trop
import argparse

def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)
    fnames = yml_conf["data"]["files_test"]
    kwargs = yml_conf["data"]["reader_kwargs"]
    fnames.extend( yml_conf["data"]["files_train"])

 
    clip_and_save_trop(fnames, **kwargs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)


