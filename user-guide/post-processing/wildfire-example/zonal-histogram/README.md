---
description: Tutorials on using the built in Zonal Histogram tool in QGIS.
---

# Zonal Histogram

#### TODO UPDATE TO JUST USE THE SCRIPT

#### This tool is important for our process because it allows us to map the pixel values from SIT FUSE's output to our labels using QGIS. To learn more about this feature, see the [QGIS Documentation](https://docs.qgis.org/3.34/en/docs/user_manual/processing_algs/qgis/rasteranalysis.html#zonal-histogram).

In order to run our zonal histogram, we first need to create and set up a few files with the appropriate configurations and data.

We begin by creating a [YAML file](creating-the-yaml-file.md) with the appropriate [configurations](gathering-files/code-configuration.md). We will then pass this through a python script built into SIT FUSE that converts our input into a JSON file, that will then be used in the zonal histogram. Here's the workflow for this step:

<figure><img src="../../../../.gitbook/assets/flow.drawio.png" alt=""><figcaption></figcaption></figure>

{% content-ref url="creating-the-yaml-file.md" %}
[creating-the-yaml-file.md](creating-the-yaml-file.md)
{% endcontent-ref %}

{% content-ref url="gathering-files/" %}
[gathering-files](gathering-files/)
{% endcontent-ref %}

{% content-ref url="gathering-files/code-configuration.md" %}
[code-configuration.md](gathering-files/code-configuration.md)
{% endcontent-ref %}

{% content-ref url="getting-output.md" %}
[getting-output.md](getting-output.md)
{% endcontent-ref %}
