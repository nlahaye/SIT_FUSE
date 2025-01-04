def write_csv_labels(samples, csv_path, label_field="ground_truth"):
    """Writes a labels CSV format for the given samples in the format expected
    by :class:`CSVImageClassificationDatasetImporter`.

    Args:
        samples: an iterable of :class:`fiftyone.core.sample.Sample` instances
        csv_path: the path to write the CSV file
        label_field ("ground_truth"): the label field of the samples to write
    """
    # Ensure base directory exists
    basedir = os.path.dirname(csv_path)
    if basedir and not os.path.isdir(basedir):
        os.makedirs(basedir)

    # Write the labels
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filepath",
            #"size_bytes",
            #"mime_type",
            #"width",
            #"height",
            #"num_channels",
            "label_seg_1",
            "label_seg_2",
            "label_hr_post",
            "label_type",
            "label_dose"])
        ])
        for sample in samples:
            filepath = sample.filepath
            metadata = sample.metadata
            if metadata is None:
                metadata = fo.ImageMetadata.build_for(filepath)

            label = sample[label_field].label
            writer.writerow([
                filepath,
                #metadata.size_bytes,
                #metadata.mime_type,
                #metadata.width,
                #metadata.height,
                metadata.num_channels,
                label,
            ])

