---
icon: landmark
---

# Class comparison

1. Gather all the `OUTPUT.dbf` files from the zonal histogram step and place them in the new YAML file (see [code-configuration.md](code-configuration.md "mention")).&#x20;

_These files should be in the same location as the initial shape files. Use terminal commands to gather all of the appropriate files (see_ [gathering-files](../../zonal-histogram/gathering-files/ "mention"))_._

2. Once the YAML file is ready, in your terminal run:&#x20;

<pre><code><strong>cd SIT_FUSE/src/sit_fuse/postprocessing &#x26;&#x26; python3 class_compare.py -y &#x3C;path_to_yaml> >&#x26; CLASS_COMPARE_OUTPUT.txt
</strong></code></pre>

The name of the `.txt` file can be anything, but make sure to know which txt file corresponds to which YAML.

