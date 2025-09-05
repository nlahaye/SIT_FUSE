---
description: >-
  SIT-FUSE's built in functionality to download satellite data from NASA's
  Earthdata platform.
---

# Sourcing satellite data

Many data types from many instruments/satellites are compatible with SIT-FUSE. NASA's Earthdata is a convenient platform to source data for the detection of HABs.&#x20;

***

Before beginning, it is necessary to create an Earthdata account, if you don't already have one. Click [here](https://urs.earthdata.nasa.gov/users/new) to create an account.

Next, in terminal:

```
cd SIT_FUSE/src/sit_fuse/automation/
```

```
/bin/bash earth_data.sh
```

This will prompt a username and password for Earthdata. Type them in and click enter.

Next, a list of satellites and respective instruments will appear. To select an option, type in the corresponding number and click enter.

Then you will be prompted to type in a range (start and end date) for the data you want. This will download all files between the two dates, with each file corresponding to a band, if applicable.

{% columns %}
{% column width="50%" %}
For example, SNPP VIIRS has 5 bands in the range we want, namely:

```
410, 443, 486, 551, 671
```

{% hint style="info" %}
These bands can be modified for different purposes by altering the band numbers `earth_data.sh` &#x20;

The ones above are just ideal for HAB detection and used as an example.
{% endhint %}

```
# Five bands for one day
SNPP_VIIRS.20250623.L3m.DAY.RRS.Rrs_410.4km.NRT.nc
SNPP_VIIRS.20250623.L3m.DAY.RRS.Rrs_443.4km.NRT.nc
SNPP_VIIRS.20250623.L3m.DAY.RRS.Rrs_486.4km.NRT.nc
SNPP_VIIRS.20250623.L3m.DAY.RRS.Rrs_551.4km.NRT.nc
SNPP_VIIRS.20250623.L3m.DAY.RRS.Rrs_671.4km.NRT.nc
```
{% endcolumn %}

{% column %}
While PACE OCI has all bands in one file, which will be extracted later:

```
# One band for one day
PACE_OCI.20250611.L3m.DAY.RRS.V3_0.Rrs.4km.NRT.nc
```

{% hint style="info" %}
Note the difference in the file naming convention of the two instruments. This will be important later.
{% endhint %}
{% endcolumn %}
{% endcolumns %}

***

{% tabs %}
{% tab title="NOAA-20" %}
<figure><img src="../../../../.gitbook/assets/Screenshot 2025-07-21 at 2.48.35 PM.png" alt=""><figcaption><p>SIT-FUSE clustered output from NOAA-20 VIIRS</p></figcaption></figure>
{% endtab %}

{% tab title="NOAA-21" %}
<figure><img src="../../../../.gitbook/assets/Screenshot 2025-07-21 at 2.48.27 PM.png" alt=""><figcaption><p>SIT-FUSE clustered output from NOAA-21 (JPSS2) VIIRS</p></figcaption></figure>
{% endtab %}

{% tab title="SNPP" %}
<figure><img src="../../../../.gitbook/assets/Screenshot 2025-07-21 at 2.47.38 PM.png" alt=""><figcaption><p>SIT-FUSE clustered output from SNPP VIIRS</p></figcaption></figure>
{% endtab %}

{% tab title="S3A" %}
<figure><img src="../../../../.gitbook/assets/Screenshot 2025-07-21 at 2.48.09 PM.png" alt=""><figcaption><p>SIT-FUSE clustered output from S3A OLCI</p></figcaption></figure>
{% endtab %}

{% tab title="AQUA" %}
<figure><img src="../../../../.gitbook/assets/Screenshot 2025-07-21 at 2.47.55 PM.png" alt=""><figcaption><p>SIT-FUSE clustered output from AQUA MODIS</p></figcaption></figure>
{% endtab %}
{% endtabs %}

The images above demonstrate the applicability of SIT-FUSE on many different instruments.

{% hint style="info" %}
Click [here](https://www.earthdata.nasa.gov/) to access Earthdata's website.
{% endhint %}
