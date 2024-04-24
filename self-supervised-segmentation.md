---
description: >-
  Once an embedding / representation capabilities are learned, self-supervised
  deep clustering is used to generate a mapping between the embedding space and
  a user-specified set of classes.
---

# Self-Supervised Segmentation

This is done by attaching one or more Multi-Layer Perceptron (MLP) heads to the encoder and either just training the head(s) or also allowing the encoder to be fine-tuned alongside the MLP head(s). This is done using Information Invariant Clustering (IIC) loss, where the optimizer aims to find parameters that allow the model to make label assignments in a way that maximizes the mutual information between samples x and a perturbed version of those samples x'.

* As a novel addition to this paradigm, we have developed the capability to attach downstream heads that only see subsets of the training data (all subsets are assigned the same class by the initial layer of MLP head(s)). By doing this we allow for heirarchichal segmentations/representations to be learned.

<figure><img src=".gitbook/assets/Screenshot 2024-04-23 at 8.18.57 PM.png" alt=""><figcaption><p>A simple exmaple of a 2-layer heirarchichal deep clustering. All of the layer-2 clustering heads train on subsets of the training data that were assigned separate labels from the layer-1 head (e.g. 1,2,3). In doing this, we can extract hierarchical segmentation maps</p></figcaption></figure>

References:

* Xu Ji, João F. Henriques, Andrea Vedaldi: “Invariant Information Clustering for Unsupervised Image Classification and Segmentation”, 2018; [arXiv:1807.06653](http://arxiv.org/abs/1807.06653).

