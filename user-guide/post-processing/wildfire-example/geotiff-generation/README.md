# GeoTiff generation

1. Open the `.txt`  file that was generated in the last step of  [class-comparison](../class-comparison/ "mention").
2. Use  `Cmd f` (or `Ctrl f`)  to search for the keyword `ASSIGN` within the file.

Under this keyword there will be a list of lists, where each sublist is a set of SIT\_FUSE label values that is associated with a specific class involved within the context assignment process _**except the last sublist**_.

* **Important note**: The order of these sublists is dependent upon the order in which the labels were input to the **dbf\_list** entry in the yaml file for the class\_compare.py (the previous step)
  * For example, looking at the example YAML in [code-configuration.md](../class-comparison/code-configuration.md "mention"), we see that the order of the label lists is _backgroud, burnscar, fire, smoke_.
* Note: The last sublist is the set of labels that did not meet the minimum threshold set for any of the desired classes.

3. Copy this list (excluding the last sublist).
4. Paste the copied list into the `clusters` entry of the new YAML (see [code-configuration.md](code-configuration.md "mention"))
   1. Make sure the order of these lists is consistent with the order of the labels in the `name` entry (see **Important note** above).
   2. Note 2: Make sure the `context/apply_context` entry is set to `True` .
5. Run:

```
cd SIT_FUSE/src/sit_fuse/postprocessing 
```

```
python3 generate_cluster_geotiffs.py -y <path_to_yaml>
```
