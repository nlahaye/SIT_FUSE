

# Cap native thread pools (OpenMP/BLAS/MKL/joblib) before numpy/torch load
# when run as a script; run_* functions re-apply via apply_compute_limits.
if __name__ == "__main__":
    from sit_fuse.compute_resources import limit_env_from_argv
    limit_env_from_argv()

import os
import argparse

from sit_fuse.utils import read_yaml
from sit_fuse.compute_resources import apply_compute_limits, add_num_threads_arg, apply_num_threads_arg

from sit_fuse.pipelines.habs.hab_training_utils import run_data_preprocessing, run_model_training, gen_geo_zarr_oc_daily


def run_hab_training_pipeline(yml_conf):

    # Applies to geo-zarr generation, preprocessing/scaling and all training
    # stages; the compute: section is also written into each model YAML.
    apply_compute_limits(yml_conf)
 
    zarr_fname = os.path.join(yml_conf["geo_zarr"]["out_dir"], yml_conf["geo_zarr"]["out_uid"] + ".zarr")
    if not os.path.exists(zarr_fname):
        gen_geo_zarr_oc_daily(yml_conf)
 
    configs = run_data_preprocessing(yml_conf)

    run_model_training(yml_conf, configs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    add_num_threads_arg(parser)
    args = parser.parse_args()
    apply_num_threads_arg(args)

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

    run_hab_training_pipeline(yml_conf)


