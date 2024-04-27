# Data Config Options

{% code fullWidth="true" %}
```
tile: Boolean noting whether or not data will be tiled (True) or vectorized (False).

val_percent: Percent of training data to be used for validation.

num_loader_workers: Number of worker threads tobe used for DataLoaders.

scaler_data: Boolean noting whether or not to scale data.

pixel_padding: For vectorized datasets, number incicating how layers of neighbors to include in sample.

number_channels: Number of channels in input data (after any channel deletions).

fill_value: Fill value used in data.

valid_min: Valid minimum value in data.

valid_max: Valid maximum value in data.

reader_type: Key specified to select the reader type for the data. See sit_fuse.utils for full list. 
    Will add to documentation in the future.
    
reader_kwargs: Keyword arguments for readers. Will add to documentation in the future.

chan_dim: Dimension representing channels in data once read in.

delete_chans: List of channels to delete before using data. Can be empty list.

transform_default: To be documented later. Not usually used.

files_train: Files to be used to extract training samples from.

files_test: Files to be used for testing/inference.

```
{% endcode %}
