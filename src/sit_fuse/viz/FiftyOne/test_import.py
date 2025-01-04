from DatasetImporter import CSVImageClassificationDatasetImporter

import fiftyone as fo

# Import the dataset

dataset_dir = "/data/nlahaye/output/Learnergy/GENETICS_POC_COMBINED/"

print("Importing dataset from '%s'" % dataset_dir)
importer = CSVImageClassificationDatasetImporter(dataset_dir)
dataset = fo.Dataset.from_importer(importer)

# Print summary information about the dataset
print(dataset)

# Print a sample
print(dataset.first())


