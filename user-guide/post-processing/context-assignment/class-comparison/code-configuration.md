---
description: >-
  Configuration for the YAML file that will help us generate our class
  comparison.
icon: bird
---

# Code configuration

{% code fullWidth="true" %}
```
dbf (bool): Generates dbf

dbf_percentage_thresh: Between 0 and 1. 

dbf_list: List of lists, where each list corresponds to a specifc labeled feature (e.g. fire, smoke, etc.)
```
{% endcode %}

**E.g. configuration:**

{% code title="generate_zonal_histogram_emas_firex.yaml" fullWidth="true" %}
```yaml
dbf: True
dbf_percentage_thresh: 0.51
dbf_list: 
  [
    [
      "/home/stasya/eMAS_dbf/eMASL1B_19910_06_20190806_1815_1824_V03_background_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_08_20190806_1834_1846_V03_background_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_10_20190806_1858_1910_V03_background_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_13_20190806_1923_1925_V03_background_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_14_20190806_1926_1929_V03_background_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_15_20190806_1931_1938_V03_background_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_16_20190806_1941_1956_V03_background_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_17_20190806_2000_2010_V03_background_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_18_20190806_2018_2031_V03_background_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_19_20190806_2035_2048_V03_background_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_20_20190806_2052_2106_V03_background_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_21_20190806_2111_2125_V03_background_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_05_20190807_1817_1833_V03_background_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_08_20190807_1928_1942_V03_background_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_10_20190807_2004_2016_V03_background_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_11_20190807_2021_2035_V03_background_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_12_20190807_2039_2051_V03_background_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19912_07_20190808_1806_1821_V03_background_OUTPUT_0.dbf"],

    [ "/home/stasya/eMAS_dbf/eMASL1B_19910_06_20190806_1815_1824_V03_burnscar_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_08_20190806_1834_1846_V03_burnscar_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_10_20190806_1858_1910_V03_burnscar_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_14_20190806_1926_1929_V03_burnscar_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_16_20190806_1941_1956_V03_burnscar_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_17_20190806_2000_2010_V03_burnscar_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_18_20190806_2018_2031_V03_burnscar_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_19_20190806_2035_2048_V03_burnscar_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_20_20190806_2052_2106_V03_burnscar_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_21_20190806_2111_2125_V03_burnscar_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_05_20190807_1817_1833_V03_burnscar_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_08_20190807_1928_1942_V03_burnscar_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_10_20190807_2004_2016_V03_burnscar_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_11_20190807_2021_2035_V03_burnscar_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_12_20190807_2039_2051_V03_burnscar_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19912_07_20190808_1806_1821_V03_burnscar_OUTPUT_0.dbf"],

    [ "/home/stasya/eMAS_dbf/eMASL1B_19910_06_20190806_1815_1824_V03_fire_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_08_20190806_1834_1846_V03_fire_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_10_20190806_1858_1910_V03_fire_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_14_20190806_1926_1929_V03_fire_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_16_20190806_1941_1956_V03_fire_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_17_20190806_2000_2010_V03_fire_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_18_20190806_2018_2031_V03_fire_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_19_20190806_2035_2048_V03_fire_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_20_20190806_2052_2106_V03_fire_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_21_20190806_2111_2125_V03_fire_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_05_20190807_1817_1833_V03_fire_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_08_20190807_1928_1942_V03_fire_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_09_20190807_1947_2002_V03_fire_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_10_20190807_2004_2016_V03_fire_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_11_20190807_2021_2035_V03_fire_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_12_20190807_2039_2051_V03_fire_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19912_07_20190808_1806_1821_V03_fire_OUTPUT_0.dbf"],

    [ "/home/stasya/eMAS_dbf/eMASL1B_19910_06_20190806_1815_1824_V03_smoke_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_08_20190806_1834_1846_V03_smoke_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_10_20190806_1858_1910_V03_smoke_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_13_20190806_1923_1925_V03_smoke_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_14_20190806_1926_1929_V03_smoke_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_16_20190806_1941_1956_V03_smoke_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_17_20190806_2000_2010_V03_smoke_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_18_20190806_2018_2031_V03_smoke_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_19_20190806_2035_2048_V03_smoke_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_20_20190806_2052_2106_V03_smoke_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19910_21_20190806_2111_2125_V03_smoke_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_05_20190807_1817_1833_V03_smoke_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_08_20190807_1928_1942_V03_smoke_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_09_20190807_1947_2002_V03_smoke_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_10_20190807_2004_2016_V03_smoke_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_11_20190807_2021_2035_V03_smoke_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19911_12_20190807_2039_2051_V03_smoke_OUTPUT_0.dbf",
      "/home/stasya/eMAS_dbf/eMASL1B_19912_07_20190808_1806_1821_V03_smoke_OUTPUT_0.dbf"]
  ]

```
{% endcode %}
