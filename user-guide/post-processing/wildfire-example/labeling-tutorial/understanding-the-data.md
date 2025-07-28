---
description: >-
  There are many data types that can be labeled and used with SIT_FUSE. This
  example will show scenes of one wildfire taken by different satellite
  instruments and compare the images.
---

# Understanding the data

The William's Flat's Fire was well documented through the 2019 FIRE-X AQ field campaign by multiple different airborne and satellite instruments, including [AirMSPI](https://airbornescience.jpl.nasa.gov/instruments/airmspi), [eMAS](https://asapdata.arc.nasa.gov/emas/), [GOES](https://www.star.nesdis.noaa.gov/goes/index.php), etc.

Through each instrument, we are able to see different aspects of the fire due to the abundance of [spectral bands](https://www.earthdatascience.org/courses/earth-analytics/multispectral-remote-sensing-data/introduction-multispectral-imagery-r/). Different bands allow us to see key elements of the fire such as the smoke, the active fire, and the remaining burn scar.&#x20;

For example, [eMAS data](https://asapdata.arc.nasa.gov/emas/data/flt_html/19910.html), originally comprised of HDF files but since [converted to GeoTiff files](../../../pre-processing/collocation-and-resampling.md), has 38 spectral bands. We can look at these spectral band in QGIS. We will use an eMAS image for our demonstration (Figure 1).

1. Open QGIS.
2. Open your converted GeoTiff file (.tif).

<figure><img src="../../../../.gitbook/assets/Screenshot 2024-10-18 at 11.53.19 AM.png" alt="" width="375"><figcaption><p>Figure 1. eMAS data of William's Flat's Fire (Multiband color)</p></figcaption></figure>

3. In the [Layer Styling panel](qgis-configuration.md), select `Multiband color` and then from the drop down that appears, select `Singleband gray`. This will cause the data to be in grayscale (Figure 2).

<div><figure><img src="../../../../.gitbook/assets/Screenshot 2024-10-18 at 11.47.44 AM.png" alt="" width="443"><figcaption><p>Layer Styling Panel</p></figcaption></figure> <figure><img src="../../../../.gitbook/assets/Screenshot 2024-10-18 at 11.53.28 AM.png" alt=""><figcaption><p>Figure 2. Grayscale (Singleband gray)</p></figcaption></figure></div>

4. Now, right under the `Singleband` gray button, we see `Gray band` and `Band 01 (Gray)`. Select the latter and you should see a drop down of all of the different bands. Of course the number of bands will vary based on what instrument the data was sourced from, but for eMAS we see 38.

<figure><img src="../../../../.gitbook/assets/Screenshot 2024-10-18 at 11.51.50 AM.png" alt="" width="119"><figcaption><p>eMAS bands</p></figcaption></figure>

5. For example, using the eMAS image, let's select band 25, which appears much darker except in areas where there is an active fire (Figure 3). Now, we can use QGIS to label this fire.

<figure><img src="../../../../.gitbook/assets/Screenshot 2024-10-18 at 11.53.55 AM.png" alt="" width="563"><figcaption><p>Figure 3. Active fire (band 25) seen in the small white pixels.</p></figcaption></figure>

**Labeler's Note:** For best practices, look through all of the bands when using data from different instruments and make note of which bands from which instruments you can see the desired feature the best. In addition, you should check documentation of the bands from the instrument website itself. Here's an [example](https://www.goes-r.gov/mission/ABI-bands-quick-info.html).

Every instrument will have different bands where certain features appear most prominently.&#x20;

* For example, band 25 on eMAS shows us the active fire while band 30 on MASTER also shows us the active fire. However, if we look at band 30 on the eMAS data, it doesn't show us the active fire at all.&#x20;
* In our eMAS figures, we see that band 1 is good for seeing the smoke plume, band 9 (vegetation band) is good for seeing the burnscar, and bands in the 20's (depending on the instrument) are good for seeing the active fire.&#x20;

Suitable bands for different purposed vary heavily, so make sure to familiarize yourself with the instrument and features that you are looking for.
