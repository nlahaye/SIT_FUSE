---
description: >-
  Configuration for the YAML file that will help us generate our zonal
  histograms.
icon: bird
---

# Code configuration

{% code fullWidth="true" %}
```
input_files: List of lists of paths to SIT_FUSE output files.

shp_files: List of lists of paths to shape files, whose index corresponds to the SIT FUSE output file they represent.

out_file (str): Name of the JSON file to be passed into QGIS.
```
{% endcode %}

**Important note:** For the input files, make sure to use both single AND double quotes around each path, otherwise the zonal histogram tool won't function properly.

**Important note 2:** Make sure each sublist (index) in the `input_files` corresponds to a sublist (index) in the `shp_files`  (AKA there should be the same number of sublists in the `input_files` as in the `shp_files`.

**If every GeoTiff has a unique label, then there should be the same number of sublists/indices in both lists. (e.g. 1 demonstrates this formatting)**

**E.g. 1 configuration:**

{% code title="generate_zonal_histogram_emas_firex.yaml" fullWidth="true" %}
```yaml
input_files:
  [
    [ "'/Users/easley/Desktop/SIT_FUSE_Geo/eMAS_v2/eMASL1B_19910_06_20190806_1815_1824_V03.hdf.clust.data_78197clusters.full_geo.tif'" ],
    [ "'/Users/easley/Desktop/SIT_FUSE_Geo/eMAS_v2/eMASL1B_19910_08_20190806_1834_1846_V03.hdf.clust.data_78197clusters.full_geo.tif'" ],
    [ "'/Users/easley/Desktop/SIT_FUSE_Geo/eMAS_v2/eMASL1B_19910_10_20190806_1858_1910_V03.hdf.clust.data_78197clusters.full_geo.tif'" ]
  ]

shp_files:
  [
    [ '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/background/burnscar_background/eMASL1B_19910_06_20190806_1815_1824_V03_burnscar_background.shp',
      '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/background/fire_background/eMASL1B_19910_06_20190806_1815_1824_V03_fire_background.shp',
      '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/background/smoke_background/eMASL1B_19910_06_20190806_1815_1824_V03_smoke_background.shp',
      '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/labels/smoke/eMASL1B_19910_06_20190806_1815_1824_V03_smoke.shp',
      '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/labels/background/eMASL1B_19910_06_20190806_1815_1824_V03_background.shp',
      '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/labels/fire/eMASL1B_19910_06_20190806_1815_1824_V03_fire.shp',
      '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/labels/burnscar/eMASL1B_19910_06_20190806_1815_1824_V03_burnscar.shp' ],

    [ '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/background/burnscar_background/eMASL1B_19910_08_20190806_1834_1846_V03_burnscar_background.shp',
      '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/background/fire_background/eMASL1B_19910_08_20190806_1834_1846_V03_fire_background.shp',
      '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/background/smoke_background/eMASL1B_19910_08_20190806_1834_1846_V03_smoke_background.shp',
      '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/labels/smoke/eMASL1B_19910_08_20190806_1834_1846_V03_smoke.shp',
      '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/labels/background/eMASL1B_19910_08_20190806_1834_1846_V03_background.shp',
      '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/labels/fire/eMASL1B_19910_08_20190806_1834_1846_V03_fire.shp',
      '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/labels/burnscar/eMASL1B_19910_08_20190806_1834_1846_V03_burnscar.shp' ],

    [ '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/background/burnscar_background/eMASL1B_19910_10_20190806_1858_1910_V03_burnscar_background.shp',
      '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/background/fire_background/eMASL1B_19910_10_20190806_1858_1910_V03_fire_background.shp',
      '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/background/smoke_background/eMASL1B_19910_10_20190806_1858_1910_V03_smoke_background.shp',
      '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/labels/smoke/eMASL1B_19910_10_20190806_1858_1910_V03_smoke.shp',
      '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/labels/background/eMASL1B_19910_10_20190806_1858_1910_V03_background.shp',
      '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/labels/fire/eMASL1B_19910_10_20190806_1858_1910_V03_fire.shp',
      '/Users/easley/Desktop/shapefiles/eMAS/20190806/shapes/labels/burnscar/eMASL1B_19910_10_20190806_1858_1910_V03_burnscar.shp' ]
  ]


out_file: "eMAS_FIREX_Batch_Full.json"

```
{% endcode %}

**However, if you have one label for all of your GeoTiffs, there should only be one sublist/index containing all the files in both (e.g. 2).**

**E.g. 2 configuration**

{% code title="generate_zonal_histogram_goes_firex.yaml" fullWidth="true" %}
```yaml
input_files: 
[
  [ 
    "'/Users/easley/Desktop/SIT_FUSE_Geo/GOES_v2/OR_ABI-L1b-RadC-M6C01_G17_s20192201821196_e20192201823569_c20192201824006.tif.clust.data_79399clusters.zarr.full_geo.tif'",
    "'/Users/easley/Desktop/SIT_FUSE_Geo/GOES_v2/OR_ABI-L1b-RadC-M6C01_G17_s20192201826196_e20192201828569_c20192201829005.tif.clust.data_79399clusters.zarr.full_geo.tif'",
    "'/Users/easley/Desktop/SIT_FUSE_Geo/GOES_v2/OR_ABI-L1b-RadC-M6C01_G17_s20192201831196_e20192201833569_c20192201834012.tif.clust.data_79399clusters.zarr.full_geo.tif'",
    "'/Users/easley/Desktop/SIT_FUSE_Geo/GOES_v2/OR_ABI-L1b-RadC-M6C01_G17_s20192201836196_e20192201838569_c20192201839012.tif.clust.data_79399clusters.zarr.full_geo.tif'",
    "'/Users/easley/Desktop/SIT_FUSE_Geo/GOES_v2/OR_ABI-L1b-RadC-M6C01_G17_s20192201841196_e20192201843569_c20192201844012.tif.clust.data_79399clusters.zarr.full_geo.tif'",
    "'/Users/easley/Desktop/SIT_FUSE_Geo/GOES_v2/OR_ABI-L1b-RadC-M6C01_G17_s20192201846196_e20192201848569_c20192201849023.tif.clust.data_79399clusters.zarr.full_geo.tif'",
    "'/Users/easley/Desktop/SIT_FUSE_Geo/GOES_v2/OR_ABI-L1b-RadC-M6C01_G17_s20192201851196_e20192201853569_c20192201854012.tif.clust.data_79399clusters.zarr.full_geo.tif'",
    "'/Users/easley/Desktop/SIT_FUSE_Geo/GOES_v2/OR_ABI-L1b-RadC-M6C01_G17_s20192201856196_e20192201858570_c20192201859007.tif.clust.data_79399clusters.zarr.full_geo.tif'"
  ]
]

shp_files: 
[
  [ "/Users/easley/Desktop/shapefiles/GOES/shapes/labels/fire/OR_ABI-L1b-RadC-M6C01_G17_s20192201821196_e20192201823569_c20192201824006_fire.shp",
    "/Users/easley/Desktop/shapefiles/GOES/shapes/labels/smoke/OR_ABI-L1b-RadC-M6C01_G17_s20192201821196_e20192201823569_c20192201824006_smoke.shp",
    "/Users/easley/Desktop/shapefiles/GOES/shapes/labels/background/OR_ABI-L1b-RadC-M6C01_G17_s20192201821196_e20192201823569_c20192201824006_background.shp",
    "/Users/easley/Desktop/shapefiles/GOES/shapes/background/fire_background/OR_ABI-L1b-RadC-M6C01_G17_s20192201821196_e20192201823569_c20192201824006_fire_background.shp",
    "/Users/easley/Desktop/shapefiles/GOES/shapes/background/smoke_background/OR_ABI-L1b-RadC-M6C01_G17_s20192201821196_e20192201823569_c20192201824006_smoke_background.shp" 
  ]
]


out_file: "GOES_FIREX_Batch_Full.json"

```
{% endcode %}
