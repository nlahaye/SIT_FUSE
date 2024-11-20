# Data Config Options

{% code fullWidth="true" %}
```
clust_reader_type (str): Key specified to select the reader type for the cluster.
    See "Reader Options".

reader_kwargs (str): Keyword arguments for readers. See "Reader Options".

subset_inds (list): List of lists representing subset indices.

create_separate (bool): Boolean indicatng whether to create seperate data strutures.

gtiff_data (list): GeoTiff files to be used.

cluster_frames (list): Files to be used for clustering.
```
{% endcode %}

{% content-ref url="../../code-configuration/config-options/data-config-options/reader-options.md" %}
[reader-options.md](../../code-configuration/config-options/data-config-options/reader-options.md)
{% endcontent-ref %}
