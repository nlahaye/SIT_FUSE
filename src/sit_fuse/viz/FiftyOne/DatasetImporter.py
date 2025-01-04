import csv
import os

import fiftyone as fo
import fiftyone.utils.data as foud


class CSVImageClassificationDatasetImporter(foud.LabeledImageDatasetImporter):
    """Importer for image classification datasets whose filepaths and labels
    are stored on disk in a CSV file.

    Datasets of this type should contain a ``labels.csv`` file in their
    dataset directories in the following format::

        filepath,size_bytes,mime_type,width,height,num_channels,label
        <filepath>,<size_bytes>,<mime_type>,<width>,<height>,<num_channels>,<label>
        <filepath>,<size_bytes>,<mime_type>,<width>,<height>,<num_channels>,<label>
        ...

    Args:
        dataset_dir: the dataset directory
        shuffle (False): whether to randomly shuffle the order in which the
            samples are imported
        seed (None): a random seed to use when shuffling
        max_samples (None): a maximum number of samples to import. By default,
            all samples are imported
    """

    def __init__(
        self,
        dataset_dir,
        shuffle=False,
        seed=None,
        max_samples=None,
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples
        )
        self._labels_file = None
        self._labels = None
        self._iter_labels = None

    def __iter__(self):
        self._iter_labels = iter(self._labels)
        return self

    def __next__(self):
        """Returns information about the next sample in the dataset.

        Returns:
            an  ``(image_path, image_metadata, label)`` tuple, where

            -   ``image_path``: the path to the image on disk
            -   ``image_metadata``: an
                :class:`fiftyone.core.metadata.ImageMetadata` instances for the
                image, or ``None`` if :meth:`has_image_metadata` is ``False``
            -   ``label``: an instance of :meth:`label_cls`, or a dictionary
                mapping field names to :class:`fiftyone.core.labels.Label`
                instances, or ``None`` if the sample is unlabeled

        Raises:
            StopIteration: if there are no more samples to import
        """
        (
            filepath,
            #size_bytes,
            #mime_type,
            #width,
            #height,
            #num_channels,
            label_seg_1,
            label_seg_2,
            label_tracks,
            label_hr_post,
            label_type,
            label_dose,
            #label_sim
        ) = next(self._iter_labels)

        image_metadata = fo.ImageMetadata(
            #size_bytes=size_bytes,
            #mime_type=mime_type,
            #width=width,
            #height=height,
            #num_channels=num_channels,
        )

        label = fo.Classification(label=label_seg_1)
        return filepath, image_metadata, [label_seg_1, label_seg_2, label_tracks, label_hr_post, label_type, label_dose] #, label_sim

    def __len__(self):
        """The total number of samples that will be imported.

        Raises:
            TypeError: if the total number is not known
        """
        return len(self._labels)

    @property
    def has_dataset_info(self):
        """Whether this importer produces a dataset info dictionary."""
        return False

    @property
    def has_image_metadata(self):
        """Whether this importer produces
        :class:`fiftyone.core.metadata.ImageMetadata` instances for each image.
        """
        return True

    @property
    def label_cls(self):
        """The :class:`fiftyone.core.labels.Label` class(es) returned by this
        importer.

        This can be any of the following:

        -   a :class:`fiftyone.core.labels.Label` class. In this case, the
            importer is guaranteed to return labels of this type
        -   a list or tuple of :class:`fiftyone.core.labels.Label` classes. In
            this case, the importer can produce a single label field of any of
            these types
        -   a dict mapping keys to :class:`fiftyone.core.labels.Label` classes.
            In this case, the importer will return label dictionaries with keys
            and value-types specified by this dictionary. Not all keys need be
            present in the imported labels
        -   ``None``. In this case, the importer makes no guarantees about the
            labels that it may return
        """
        return list #fo.Classification

    def setup(self):
        """Performs any necessary setup before importing the first sample in
        the dataset.

        This method is called when the importer's context manager interface is
        entered, :func:`DatasetImporter.__enter__`.
        """
        labels_path = os.path.join(self.dataset_dir, "labels.csv")

        labels = []
        with open(labels_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append((
                    row["filepath"],
                    #row["size_bytes"],
                    #row["mime_type"],
                    #row["width"],
                    #row["height"],
                    #row["num_channels"],
                    row["label_seg_1"],
                    row["label_seg_2"],
                    row["label_tracks"],
                    row["label_hr_post"],
                    row["label_type"],
                    row["label_dose"],
                    #row["label_sim"],
                ))

        # The `_preprocess_list()` function is provided by the base class
        # and handles shuffling/max sample limits
        self._labels = self._preprocess_list(labels)

    def close(self, *args):
        """Performs any necessary actions after the last sample has been
        imported.

        This method is called when the importer's context manager interface is
        exited, :func:`DatasetImporter.__exit__`.

        Args:
            *args: the arguments to :func:`DatasetImporter.__exit__`
        """
        pass

