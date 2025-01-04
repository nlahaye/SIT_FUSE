

dataset_dir = "/tmp/fiftyone/custom-dataset-importer"
num_samples = 1000

#
# Load `num_samples` from CIFAR-10
#
# This command will download the test split of CIFAR-10 from the web the first
# time it is executed, if necessary
#
cifar10_test = foz.load_zoo_dataset("cifar10", split="test")
samples = cifar10_test.limit(num_samples)

# This dataset format requires samples to have their `metadata` fields populated
print("Computing metadata for samples")
samples.compute_metadata()

# Write labels to disk in CSV format
csv_path = os.path.join(dataset_dir, "labels.csv")
print("Writing labels for %d samples to '%s'" % (num_samples, csv_path))
write_csv_labels(samples, csv_path)



