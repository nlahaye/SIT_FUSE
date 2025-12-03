
import argparse
from sit_fuse.utils import read_yaml
from sit_fuse.pipelines.context_assign.context_assign_utils import run_context_assign_experiment



def run_context_assign_pipeline(yml_conf):

    run_context_assign_experiment(yml_conf)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    yml_conf = read_yaml(args.yaml)

    run_context_assign_pipeline(yml_conf)




