# SIT-FUSE Integration

### **SIT-FUSE Integration**

#### **Library Implementation**

1. The temporal models are implemented as classes in Learnergy (Roder & de Rosa, 2020), the library SIT-FUSE builds its RBMs and DBNs on: `RTRBM` (Bernoulli), `RTGaussianRBM` (fixed variance), `RTVarianceGaussianRBM` (learned per-feature variance), and `RTDBN`, which stacks RTRBM layers greedily. The implementation follows Learnergy's conventions.
2. SIT-FUSE's data pipeline was extended to handle multi-trial biomechanical recordings. A causal sliding window is applied inside each episode, so windows never span two recordings.

#### **Verification**

We checked the training gradients against exact maximum likelihood on small models whose states can be enumerated. With the exact negative-phase expectation, the contrastive-divergence surrogate used by the code equals the automatic-differentiation gradient of the exact sequence log-likelihood, taken through the recurrent chain of the learned-variance model. The conditional distributions implied by the joint energy equal those the classes sample from.

#### **Running it in SIT-FUSE**

Setting `encoder_type: rtdbn` in a SIT-FUSE configuration file selects the temporal encoder. The stages are:

1. Encoder pretraining and a clustering stage (see [RTDBN Training Command](../../user-guide/model-training/rtdbn-training-command.md)).
2. Inference, which writes one cluster label per window for the held-out splits (see [RTDBN Inference Command](../../user-guide/model-inference/rtdbn-inference-command.md)).
3. FLOP counting.
4. Per-cluster TimeSHAP attributions over time steps and channels, so that the resulting clusters can be inspected by channel and time step.

The configuration options are described in [RTDBN-Specific Config Options](../../user-guide/code-configuration/code-configuration/config-options/rtdbn-specific-config-options.md), with a full example in [Example RTDBN Config](../../user-guide/code-configuration/code-configuration/example-configs/example-rtdbn-config.md).

The results in this use case cluster the embeddings with k-means, as a deliberately separate clustering step from encoder training. SIT-FUSE's pipeline also provides an Invariant Information Clustering (IIC) head (Ji et al., 2019), and training it jointly with the encoder is a natural next evaluation.
