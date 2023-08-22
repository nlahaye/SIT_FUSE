# General imports
import os

# Module imports
import dbn_learnergy
import discretize_clusters
import dbn_learnergy_heirarchical
import postprocessing.generate_cluster_geotiffs
import postprocessing.contour_and_fill

# Data
from utils import read_yaml

# Input parsing
import yaml
import argparse
from datetime import timedelta

def main(**kwargs):
    """
        kwargs:
            dbn_yaml: {YAML file for DBN and output config.}
            dc_yaml: {YAML file for cluster discretization.}
            
    """
    
    dbnyml_conf = read_yaml(kwargs["dbn_yaml"]) 
    # Translate config to dictionary 
    out_dir = dbnyml_conf["output"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    # Run Initial Feature Extraction & Top Level Clustering
    dbn_learnergy.run_dbn(dbnyml_conf)
    
    if "dc_yaml" in kwargs:
        dcyml_config = read_yaml(kwargs["dc_yaml"])
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--dbn_yaml", help="YAML file for DBN and output config.")
    parser.add_argument("-d", "--dc_yaml", nargs='?', help="YAML file for cluster discretization.")
    args = parser.parse_args()
    from timeit import default_timer as timer
    start = timer()
    main(args)
    end = timer()
    print(end - start) # Time in seconds, e.g. 5.38091952400282