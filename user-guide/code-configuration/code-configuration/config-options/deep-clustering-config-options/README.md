# Deep Clustering Config Options

{% code fullWidth="true" %}
```
training: A subsection of the cluster section of the config. Options are discussed in the page linked below.

heir: A subsection of the cluster section of the config detailing heirarchichal deep clustering options. 
    Options are discussed in the page linked below
    
gauss_noise_stdev: A list of floats (usually of size 1) that represent the std. dev. of the distribution
    (mean is 0) of gaussian noise added to samples to perturb them during training. 
    If more than one is provided, they are rotated across epochs.
    
lambda: Weight hyperparameter for IIC loss.

num_classes: Number of classes to use for deep clustering.
```
{% endcode %}



{% content-ref url="deep-clustering-training-config-options.md" %}
[deep-clustering-training-config-options.md](deep-clustering-training-config-options.md)
{% endcontent-ref %}

{% content-ref url="hierarchical-deep-clustering-config-options/" %}
[hierarchical-deep-clustering-config-options](hierarchical-deep-clustering-config-options/)
{% endcontent-ref %}
