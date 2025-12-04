---
description: SIT-FUSE's built in functionality to download in-situ data from ERDDAP
---

# Sourcing in-situ data

In-situ data is comprised of measurements taken at a location of interest using instruments physically present at the site. In our HAB case, for example, the in-situ data would be concentrations of different photoplankton species, information about the surrounding body of water (temperature, depth, salinity, etc.), locations of the sites, dates and times, and various chemical concentrations.

In-situ data is critical for the labeling process as it serves as our labels for the context assignment. The in-situ points function as truth values and are used to validate SIT-FUSE's clustering capabilities/output.

The data from the website below was used as a source for HAB data off of the coast of California.

{% embed url="https://erddap.sccoos.org/erddap/index.html" %}

The data format is a CSV, which was then stripped to remove the exact UTC time and any values associated with a depth above a certain threshold. This data can of course be modified differently depending on the goal.



{% hint style="info" %}
Click [here](https://erddap.sccoos.org/erddap/index.html) to access ERDDAP's website and learn more.
{% endhint %}

