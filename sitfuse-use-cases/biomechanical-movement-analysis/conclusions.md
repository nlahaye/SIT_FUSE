# Conclusions

### **Key Takeaways**

#### Summary of Findings

Trained without labels on 0.5-second windows of human joint-angle and IMU data, an RTRBM learned a 64-dimensional representation that:

1. identified the locomotion mode (treadmill, level ground, ramp, stair) of unseen subjects better than same-size PCA and autoencoder representations;
2. needed relatively few labels, reaching a balanced accuracy of 0.652 with 5% of the training labels on joint angles alone, compared with 0.483 to 0.603 for the PCA and autoencoder baselines;
3. captured graded conditions (speed, incline, stair height) well once IMU channels were added; and
4. flagged ramps and stairs as unfamiliar after training on level ground and treadmill only (AUROC 0.897 on joint angles, 0.805 with IMU), better than autoregressive, Mahalanobis, and PCA-based scores.

#### **Limitations and Errors**

There are a few sources of possible errors in our study:

1. The study uses nine subjects (AB06 to AB14) from a dataset of 22. The same subjects were used for several batches of tests, so later batches were corrected as a family with a stricter threshold. A replication on the remaining subjects has not yet been run, and no firm claim should be made before it.
2. A simple summary-statistics baseline (mean, standard deviation, minimum, and maximum of each channel) was not beaten on locomotion mode with the random forest probe. The RTRBM's best result, 0.796 balanced accuracy, came from combining the two.
3. Standardizing each subject's embeddings with their own statistics removed most of the RTRBM's lead over the autoencoders. The remaining lead was over PCA.
4. The recurrent connection, which is what distinguishes an RTRBM from an RBM, did not help in any of four pre-registered tests. The standard model's recurrent weights did not learn time order, and the variant that did learn time order was no better at identifying terrain.
5. Several other pre-registered tests did not meet their acceptance rules: graded conditions on joint angles alone, longer windows, held-out terrain levels, detection of subtle differences between terrain levels, filling in missing channels, clustering agreement with terrain labels, ascent versus descent, and level-ground speed classes. We report them here so the results are not overstated.
6. The RTRBM's representation changed more than PCA's or the autoencoders' when the training subjects changed (CKA of 0.956 compared with 0.98 to 0.99).
7. Labels for ramp incline and stair height are ordinal indices that we did not convert to degrees or centimeters.

#### Study Conclusions

For movement data, the self-supervised RTRBM extension of SIT-FUSE produces representations that are useful without labels for identifying terrain, for learning from few labels, and for noticing unfamiliar movement. The evidence points to the learned hidden-layer representation, rather than the recurrent connection, as the source of the benefit.

#### **Final Thoughts**

SIT-FUSE was built for satellite imagery, and this use case shows that the same self-supervised and layer-wise training approach can be applied to a very different type of data. These results are preliminary. The next step is a replication on additional subjects from the dataset, and results should be reported both with and without per-subject standardization. To train an RTDBN on your own data, see the [Training Command](../../user-guide/model-training/training-command-examples.md) and [RTDBN-Specific Config Options](../../user-guide/code-configuration/code-configuration/config-options/rtdbn-specific-config-options.md) pages.

This work builds on the preprint by Sharif and LaHaye (2026), which describes the RTRBM and RTDBN method and is listed in the [Bibliography](bibliography.md).
