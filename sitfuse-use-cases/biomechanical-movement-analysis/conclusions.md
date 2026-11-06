# Conclusions

### **Key Takeaways**

#### Summary of Findings

1. A Gaussian RTRBM with a per-feature learned variance, a validation-selected learning rate, and a two-phase warm-up schedule reconstructs held-out reaching windows far better than Bernoulli or fixed-variance alternatives (median R² of 0.61 versus 0.09 and 0.57, over ten seeds).
2. The RTDBN collapses the temporal hidden states into a 64-dimensional embedding per window. Clustering these embeddings without any labels recovers movement structure that can be read in terms of distance to the target and closing rate, and the clusters transfer to held-out episodes.
3. On two real gait cohorts, the same embeddings classified walking speed in unseen subjects better than PCA and autoencoder embeddings under a random forest probe.
4. The verified implementation runs end to end inside SIT-FUSE, from pretraining and clustering to per-window outputs and TimeSHAP attributions.

#### **Limitations and Errors**

1. The reaching data is simulated, uses a random action policy, and has a single fixed target, so its windows are kinematically diverse but not behaviorally diverse. Real sensor data, variable targets, or task-directed policies would likely give richer cluster structure.
2. On gait data, simple per-channel summary statistics classified walking speed at least as well as the embeddings.
3. PCA and autoencoders reconstruct the reaching windows better than the RTRBM does.

#### **Final Thoughts**

SIT-FUSE was built for satellite imagery, and this use case shows that its self-supervised and layer-wise training approach extends to sequential biomechanical data. In future work, larger and more varied cohorts and joint training with the IIC clustering head are natural next evaluations. To train an RTDBN on your own data, see the [RTDBN Training Command](../../user-guide/model-training/rtdbn-training-command.md) and [RTDBN-Specific Config Options](../../user-guide/code-configuration/code-configuration/config-options/rtdbn-specific-config-options.md) pages.

A more extensive description of this work can be found in the preprint by Sharif and LaHaye (2026), which describes the RTRBM and RTDBN method and is listed in the [Bibliography](bibliography.md).