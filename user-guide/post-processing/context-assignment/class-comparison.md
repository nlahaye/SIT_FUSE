---
description: WIP
icon: landmark
---

# Class comparison

CHANGE THIS FOR CLASS COMPARE BUT SAME STRUCTURE (`*<label_name>_OUTPUT.dbf`)

3. Once you are at the desired directory, locate all .shp files and their respective paths by typing `find $PWD -name "*<label_name>.shp"` in the terminal.&#x20;

Replace `<label_name>` with whatever label you are searching for (e.g. `smoke`), **repeating with each label**.  What _exactly_ the label is called depends on what you named each file in [Creating and modifying shapefiles](../labeling-tutorial/creating-and-modifying-shapefiles.md), which is why it is important to be consistent when naming files.

For example, every smoke label will be found with `find $PWD -name "*smoke.shp"`

* `$PWD` shows us the full path before the name of each file (otherwise we'd only see the file name).&#x20;
* `-name "*smoke.shp"` finds every filename ending `smoke.shp` (AKA every smoke label).
* The `*` is a wildcard, meaning we are telling the computer that we don't care what comes before `smoke.shp`, as long as the file has `smoke.shp` at the end of the name.&#x20;

Repeat with every label. The reason we do `"*<label_name>.shp"` instead of `"*.shp"` is because the latter will result in an output organization that is not optimal for our YAML file and would require us to rearrange all of the files. This is extra steps but saves lots of time since every label is now&#x20;
