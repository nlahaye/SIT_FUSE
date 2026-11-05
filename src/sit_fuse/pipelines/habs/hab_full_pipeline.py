
# Cap native thread pools (OpenMP/BLAS/MKL/joblib) before numpy/torch load
# when run as a script; run_* functions re-apply via apply_compute_limits.
if __name__ == "__main__":
    from sit_fuse.compute_resources import limit_env_from_argv
    limit_env_from_argv()

import argparse

from sit_fuse.utils import read_yaml
from sit_fuse.compute_resources import apply_compute_limits, add_num_threads_arg, apply_num_threads_arg

from sit_fuse.pipelines.habs.hab_training_pipeline import run_hab_training_pipeline
from sit_fuse.pipelines.habs.hab_inference_pipeline import run_hab_inference_pipeline
from sit_fuse.pipelines.habs.hab_post_inference_pipeline import run_hab_post_inference_pipeline


def run_full_pipeline(yml_conf):
    """training -> inference (clusters [+ head outputs / MC / OOD]) -> post-inference [+ UQ]."""

    # One process-wide budget for every stage (no-op without a compute: section).
    apply_compute_limits(yml_conf)

    if yml_conf.get("run_training", True):
        run_hab_training_pipeline(yml_conf)
    if yml_conf.get("run_inference", True):
        run_hab_inference_pipeline(yml_conf)
    run_hab_post_inference_pipeline(yml_conf)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    add_num_threads_arg(parser)
    args = parser.parse_args()
    apply_num_threads_arg(args)

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

    run_full_pipeline(yml_conf)
