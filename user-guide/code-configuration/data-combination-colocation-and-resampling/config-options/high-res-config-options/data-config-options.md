# Data Config Options

{% code fullWidth="true" %}
```
valid_min (float): Valid minimum value in data.

valid_max (float): Valid maximum value in data.

reader_type (str): Key specified to select the reader type for the data. See sit_fuse.utils for full list. 
    See "Reader Options".
    
reader_kwargs (str): Keyword arguments for readers. See "Reader Options".

geo_reader_type (str): Key specified to select the reader type for the geo data.

geo_reader_kwargs (str): Keyword arguments for geographic readers.

filenames (list): List of filenames for data.

geo_filenames (list): List of filenames for geospatial data.

chan_dim (int): Dimension representing channels in data once read in.

geo_coord_dim (int): Dimension index for geographic coordinates.

geo_lat_index (int): Index of latitude in the geographic coordinates.

geo_lon_index (int): Index of longitude in the geographic coordinates.
```
{% endcode %}

{% content-ref url="../../../code-configuration/config-options/data-config-options/reader-options.md" %}
[reader-options.md](../../../code-configuration/config-options/data-config-options/reader-options.md)
{% endcontent-ref %}
