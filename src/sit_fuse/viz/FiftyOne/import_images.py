from SimpleDatasetImporter import import_dataset, comp_viz
from sit_fuse.utils import numpy_to_torch, read_yaml, insitu_hab_to_tif
import argparse
import fiftyone as fo

def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)


    # Import the dataset
    dataset_dir = yml_conf["dataset_dir"] #"/home/nlahaye/eMAS_DBN_Embeddings/"
    print("Importing dataset from '%s'" % dataset_dir)
    dataset = import_dataset(yml_conf["label_extension"], yml_conf["images_patt"], yml_conf["final_label_dict"], \
            yml_conf["dataset_name"], yml_conf["no_heir"], yml_conf["spatial_cluster_fname"])

    #results = comp_viz(dataset)
    results = comp_viz(dataset, yml_conf["dataset_name"], yml_conf["label_extension"])

 
    # Print a sample
    print(dataset.first())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    main(args.yaml)



