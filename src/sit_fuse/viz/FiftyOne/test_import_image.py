from SimpleDatasetImporter import import_dataset, comp_viz

import fiftyone as fo

# Import the dataset
dataset_dir = "/data/nlahaye/SIT_FUSE_Geo/eMAS_DBN_Embeddings/"
 

print("Importing dataset from '%s'" % dataset_dir)
dataset = import_dataset(dataset_dir)
results = comp_viz(dataset)
 
# Print summary information about the dataset
print(dataset)

# Print a sample
print(dataset.first())


