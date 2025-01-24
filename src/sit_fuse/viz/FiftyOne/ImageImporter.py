import csv
import os

import fiftyone as fo
import fiftyone.utils.data as foud
import numpy as np

class ImagePatchDatasetImporter(foud.FiftyOneImageDetectionDatasetImporter):
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

        super().__init__(
            dataset_dir=dataset_dir,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples,
        )

        self._classes = None
        self._sample_parser = None
        self._image_paths_map = None
        self._labels_map = None
        self._uuids = ["/data/nlahaye/SIT_FUSE_Geo/eMAS_DBN_Embeddings/eMASL1B_19910_20_20190806_2052_2106_V03.tif"]
        self._iter_uuids = iter(self._uuids)
        self._num_samples = 1

    def __iter__(self):
        self._iter_uuids = iter(self._uuids)
        return self
 
    def __len__(self):
        return self._num_samples

    def __next__(self):
        uuid = next(self._iter_uuids)

        if os.path.isabs(uuid):
            image_path = uuid
        else:
            image_path = self._image_paths_map[uuid]

        target = None

        image_metadata = None

        if target is not None:
            self._sample_parser.with_sample((image_path, target))
            label = self._sample_parser.get_label()
        else:
            label = None

        return image_path, image_metadata, label



    @property
    def has_dataset_info(self):
        return self._classes is not None

    @property
    def has_image_metadata(self):
        return False

    @property
    def label_cls(self):
        return fol.Detections


    def setup(self):
        """Performs any necessary setup before importing the first sample in
        the dataset.

        This method is called when the importer's context manager interface is
        entered, :func:`DatasetImporter.__enter__`.
        """

        labels_path = "/data/nlahaye/SIT_FUSE_Geo/eMAS_DBN_Embeddings/eMASL1B_19910_20_20190806_2052_2106_V03.dbn_2_layer_2000.viz_dict.npy"
        my_dict = np.load(labels_path, allow_pickle=True).item()

        detections = []
        #TODO
        w = 1379
        h = 5217
        for i in range(len(my_dict["bb_x"])):
            x = (my_dict["bb_x"][i]-1) / w #Make TL index (from center) and normalize in [0,1]
            y = (my_dict["bb_y"][i]-1) / h 
            w_bb = my_dict["bb_width"][i] / w
            h_bb = my_dict["bb_height"][i] / h
            rel_box = [x, y, w_bb, h_bb]
            detections.append(fo.Detection(fire = my_dict["final_label"][i],
                bounding_box=rel_box,
                no_heir=my_dict["no_heir_label"][i],
                heir=my_dict["heir_label"][i]))

            # The `_preprocess_list()` function is provided by the base class
        # and handles shuffling/max sample limits
        self._labels = self._preprocess_list(detections)

    def close(self, *args):
        """Performs any necessary actions after the last sample has been
        imported.

        This method is called when the importer's context manager interface is
        exited, :func:`DatasetImporter.__exit__`.

        Args:
            *args: the arguments to :func:`DatasetImporter.__exit__`
        """
        pass

