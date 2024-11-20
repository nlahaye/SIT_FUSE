# Hierarchical Deep Clustering Config Options

{% code fullWidth="true" %}
```
tiers: Number of heirarchichal tiers. 

training: A subsection of the cluster section of the config. Options are discussed in the page linked below.
    
gauss_noise_stdev: A list of floats (usually of size 1) that represent the std. dev. of the distribution
    (mean is 0) of gaussian noise added to samples to perturb them during training. 
    If more than one is provided, they are rotated across epochs.
    
lambda: Weight hyperparameter for IIC loss.

num_classes: Number of classes to use for deep clustering.

```
{% endcode %}

{% content-ref url="hierarchichal-deep-clustering-training-config-options.md" %}
[hierarchichal-deep-clustering-training-config-options.md](hierarchichal-deep-clustering-training-config-options.md)
{% endcontent-ref %}
