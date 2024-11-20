# Encoder Config Options

{% code fullWidth="true" %}
```
tiled: A boolean value where True indicates the encoder takes in image tiles and 
    False indicates it takes in vectorized image tiles

tune_scaler: A boolean value indicating whether or not to tune a scaler (if available)
    with the provided training data
    
subset_training: A max value for number of training samples. If the data provided has a
    larger number of samples than what is indicated in this field, then subsetting will either be done
    via a) random selection b) unsupervised stratification via k-means. A value of -1 indicates no subsetting
    is desired.
    
overwrite_model: A boolean indicating whether or not to overwrite any pre-existing model state dictionaries that exist.

training: A subsection of the encoder section of the config. Options are discussed in the page linked below.

```
{% endcode %}



{% content-ref url="encoder-training-config-options/" %}
[encoder-training-config-options](encoder-training-config-options/)
{% endcontent-ref %}
