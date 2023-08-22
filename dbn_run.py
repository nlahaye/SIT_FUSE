# General imports
import os
from glob import glob
import numpy as np

# Module imports
import torch
# import dbn_learnergy
import discretize_clusters
# import dbn_learnergy_heirarchical
# from postprocessing import generate_cluster_geotiffs
# from postprocessing import contour_and_fill

# Data
from utils import read_yaml
from preprocessing.misc_utils import uavsar_to_geotiff
import pickle

# Input parsing
import yaml
import argparse
from datetime import timedelta

def main(dbn_yaml=None, **kwargs):
    """
        dbn_yaml: {YAML file for DBN and output config.}
        kwargs:
            dc_yaml: {YAML file for cluster discretization.}
            
    """
    dbn_dict = read_yaml(dbn_yaml)
    out_dir = dbn_dict["output"]["out_dir"]
    
    # # Translate config to dictionary for dbn  
    # out_subdir = os.path.join(out_dir, 'run_dbn')
    # os.makedirs(out_dir, exist_ok=True)
    # os.makedirs(out_subdir, exist_ok=True)
    
    # # Run Initial Feature Extraction & Top Level Clustering
    # dbn_learnergy.run_dbn(dbn_yaml)
    # data_files = glob(os.path.join(out_dir, "*.data"))
    
    # # Create yaml for clustering
    # clust_dict = {'data': {'filenames': data_files}}
    # clust_yaml = os.path.join(out_subdir, 'discretize_uavsar_cluster_run_dbn.yml')
    # with open(clust_yaml, 'w') as outfile:
    #     yaml.dump(clust_dict, outfile, default_flow_style=True)
        
    # # Discretize Model Output - Assign Top Level Labels/Clusters
    # discretize_clusters.main(clust_yaml)
    
    # # # Create Heirarchical Tree/Dendrogram of Labels - Assign More Precise Labels
    # # dbn_learnergy.run_dbn(dbn_yaml)
    # # data_files = glob(os.path.join(out_dir, "*.data"))
    
    # # # Create geotiffs from testing files
    # # test_fps = dbn_dict["data"]["files_test"]
    # # reader_kwargs = dbn_dict["data"]["reader_kwargs"]
    # # gtiff_data = uavsar_to_geotiff(test_fps, out_subdir, reader_kwargs)
    
    # # # Create yaml for intermediate product generation
    # # clust_reader_type = "uavsar"
    # # cluster_fnames = glob(os.path.join(out_dir, '*.zarr'))
    # # subset_inds = []
    # # for i in enumerate(cluster_fnames):
    # #     subset_inds.append([])
    # # create_separate = False
    # # apply_context = False
    # # compare_truth = False
    # # generate_union = False
    # # gtiff_dict = {'gen_from_geotiffs': False, 
    # #               'data': 
    # #                   {'clust_reader_type': clust_reader_type, 
    # #                    'reader_kwargs': reader_kwargs}, 
    # #               'subset_inds': subset_inds, 
    # #               'create_separate': create_separate,
    # #               'gtiff_data': gtiff_data,
    # #               'cluster_fnames': cluster_fnames,
    # #               'context': 
    # #                   {'apply_context': apply_context,
    # #                    'clusters': [],
    # #                    'name': "",
    # #                    'compare_truth': compare_truth,
    # #                    'generate_union': generate_union}
    # #               } 
    # # gtiff_yaml = os.path.join(out_subdir, 'generate_cluster_geotiffs_run_dbn.yml')
    # # with open(gtiff_yaml, 'w') as outfile:
    # #     yaml.dump(gtiff_dict, outfile, default_flow_style=True)
    
    # # # GeoTiff/Intermediate Product Generation
    # # generate_cluster_geotiffs.main(gtiff_yaml)
    
    # fp = '/work/09562/nleet/ls6/output/caldor_26200_21048_013_210825_L090HHHH_CX_01.grd.clustoutput_test.data'
    # file = open(fp, 'rb')
    # data = pickle.load(file)
    # print(type(data))
    # file.close()
    
    dat = glob(os.path.join(out_dir, "*.heir_clustoutput*.data"))
    print(dat)
    for i in range(len(dat)):
        if not os.path.exists(dat[i]):
            continue

        print(dat[i])
        data = torch.load(dat[i]).numpy()
        indices = torch.load(dat[i] + ".indices")
        print(data.shape)
        print(data.shape[1])
        print(data[1])

        max_cluster = data.shape[1]
        min_cluster = 0
        if data.shape[1] > 1:
            max_cluster = data.shape[1]
            disc_data = np.argmax(data, axis = 1)
            del data
        else:
            disc_data = data.astype(np.int32)
            max_cluster = disc_data.max() #TODO this better!!!
            del data    

        print(np.unique(disc_data).shape, "UNIQUE LABELS")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--dbn-yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    from timeit import default_timer as timer
    start = timer()
    main(args.dbn_yaml)
    end = timer()
    print(end - start) # Time in seconds, e.g. 5.38091952400282