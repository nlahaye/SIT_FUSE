---
description: >-
  Segmentation, Instance Tracking, and data Fusion Using multi-SEnsor imagery 
  (SIT-FUSE)
---

# 🛰️ SIT-FUSE Docs

SIT-FUSE utilizes self-supervised machine learning (ML)  that allows users to segment instances of objects in single and multi-sensor scenes, with minimal human intervention, even in low- and no-label environments. Can be used with image like and non image-like data.&#x20;

Currently, this technology is being used with remotely sensed earth data to identify objects including:

* Wildfires and smoke plumes
* Harmful algal blooms and their severity
* Palm oil farms
* Dust and volcanic ash plumes
* Inland water bodies

Figure 1 depicts the full flow of SIT-FUSE and figures 2 and 3 show segmentation maps and the information extracted for instance tracking across scenes. SIT-FUSE’s innovative multi-sensor fire and smoke segmentation precisely detects anomalous observations from instruments with varying spatial and spectral resolutions. This capability creates a sensor web by incorporating observations from multiple satellite-based and suborbital missions. The ML framework’s output also facilitates smoke plume and fire front tracking, a task currently under development by the SIT-FUSE team.

<figure><img src=".gitbook/assets/Screenshot 2024-04-23 at 9.07.28 PM.png" alt=""><figcaption><p>Figure 1. The flow diagram for SIT-FUSE</p></figcaption></figure>

***

<figure><img src="https://lh7-us.googleusercontent.com/YOsc_q62-vEjxvqJVMIF_5Unt5GU4UndqNIQf_q7WCXYLYk7S8-Eax2t8LfL850GFdeTW0t48FllBtA1V8CTfn_2GJm7F61hfPzqgvZ-WM9x8dwDstDgjPyoRhwOz3J2OulNSZ8aaf435xXgvseVCoE" alt=""><figcaption><p>Figure 2. The first row contains scenes from different instruments/instrument sets used as input. The second row shows SIT-FUSE’s output segmentation maps for the input scene, and the third row shows retrieved objects of interest, in this case, fire and smoke.</p></figcaption></figure>

***

<figure><img src="https://lh7-us.googleusercontent.com/F26NyNgjlpnI3QEaLNRgdr3H6H4E9xyhCq-q8Ucr9tJL525esTtTulDBdXv2VlJlwqi3YeVZxDEdlCWGNKrm4oPa8NnlH6FacAPaIKXAF_bWApKCbF7Lsc4VZqrmj3E5NLVJyALxk2gMeQk6dTyRTnQ" alt=""><figcaption><p>Figure 3. Each 4-image set is generated from a separate GOES-17 scene over an observed fire in 2019. The top row of each set depicts radiances and their associated clustering output from software system S. The second row shows the radiances with an overlay of the subset of clusters assigned to the contexts of smoke and fire. The bottom row shows the input radiances with shape approximations for smoke and fire generated via the openCV contour functionality. The green arrows depict the products that can be used for instance tracking. For cross-instrument instance tracking we will use contrastive learning to map the instance signatures across the different domains.</p></figcaption></figure>

***

Recent Talks:

{% embed url="https://vimeo.com/771105424/c1379bc387" %}
2022 ECMWF–ESA Workshop on Machine Learning for Earth Observation and Prediction
{% endembed %}

{% embed url="https://www.google.com/url?opi=89978449&rct=j&sa=t&source=web&url=https://www.youtube.com/watch?v=-cYSpBQVQi4&usg=AOvVaw37WlIcIwp3564Kb6AKPdLP&ved=2ahUKEwiGqOXKs9uFAxUeJEQIHahiBVIQtwJ6BAgWEAI" %}
2022 TIES Annual Meeting
{% endembed %}

References:

* Lahaye, N., Garay, M. J., Bue, B., El-Askary, H., Linstead, E. “A Quantitative Validation of  Multi-Modal Image Fusion and Segmentation for Object Detection and Tracking”. Remote Sens. 2021, 13, 2364. [https://doi.org/10.3390/rs13122364](https://doi.org/10.3390/rs13122364)&#x20;
* Lahaye, N., Ott, J., Garay, M. J., El-Askary, H., and Linstead, E., “Multi-modal object tracking and image fusion with unsupervised deep learning,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 12, no. 8, pp. 3056-3066, Aug. 2019, doi: [https://doi.org/10.1109/JSTARS.2019.2920234](https://doi.org/10.1109/JSTARS.2019.2920234)
