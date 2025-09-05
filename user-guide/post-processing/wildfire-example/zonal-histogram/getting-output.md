---
description: Steps for generating output from the zonal histogram tool in QGIS.
icon: folder-arrow-down
---

# Getting output

After the YAML configuration is set up, run:

```
cd SIT_FUSE/src/sit_fuse/postprocessing && python3 generate_zonal_histogram_batch.py -y ../config/postprocess/generate_zonal_hitogram_batch_planet_firex_aq.yaml
```

This will output a JSON that we will pass into QGIS' zonal histogram tool in the next step.

***

1. Open QGIS.
2. From the menu bar at the top, make sure `View -> Panels -> Processing Toolbox` is checked.
3. From the search bar in the Processing Toolbox, type `zonal histogram` and double click `Zonal histogram`.&#x20;
4. Then from the pop up window, select `Run as Batch Process...` from the bottom.&#x20;
5. Select the yellow folder icon that says `Open` when hovered over. Then, navigate to and select the JSON file that was generated in the last step.
6. Then click the `Run` button in the bottom right.
7. The output histogram files will be in the same location as the **initial shp files**. They can be identified via the “OUTPUT” keyword at the end of their filenames.

There will be many OUTPUT files, but for our purposes we will use the `OUTPUT.dbf` files for the context assignment.
