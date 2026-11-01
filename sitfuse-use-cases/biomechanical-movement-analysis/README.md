---
description: Wafi Sharif
---

# Biomechanical Movement Analysis

### **Project Overview**

SIT-FUSE's self-supervised representation learning is not limited to satellite imagery. This use case applies the Recurrent Temporal RBM and DBN (RTRBM/RTDBN) extension of SIT-FUSE to biomechanical movement data: recordings of human joint angles and body-worn inertial sensors (IMUs) collected while people walk on level ground, ramps, stairs, and a treadmill. The question is the same one SIT-FUSE asks of remote sensing data: without any labels during training, can a model learn a representation that organizes the data in a way that is useful afterward? The model is trained only on unlabeled movement windows, and labels such as terrain type, speed, and incline are used afterward, only to test what the learned representation captured. The method is described in Sharif and LaHaye (2026), and the analysis here is a preliminary evaluation on real human movement data.

**Preliminary results.** This study uses nine subjects. A replication on additional subjects has not been run, so the findings below should be read as preliminary.

#### **Why Biomechanical Movement?**

<figure><img src="../../.gitbook/assets/rtrbm-rtdbn-diagram.svg" alt=""><figcaption><p>A depiction of the recurrent connection that turns an RBM into an RTRBM</p></figcaption></figure>

Human movement is rich in information: the way a person's joints move reveals the terrain they are crossing, how fast they are going, and whether something unfamiliar is happening. Wearable sensors, prosthetics and exoskeleton control, and rehabilitation research all depend on understanding this kind of data, and labeling it is slow and requires expert annotation. As with satellite data, a self-supervised approach that does not need labels for training is attractive. Movement data is also temporal: what matters is how the joints change over a short stretch of time, not a single snapshot. Standard RBMs treat every timestep independently, which is why SIT-FUSE includes recurrent temporal RBMs (RTRBMs; Sutskever et al., 2008) that carry a hidden state from one timestep to the next. See [RBMs and DBNs](../../readme/self-supervised-representation-learning/rbms-and-dbns.md) for background on the model.

#### **Our Data Flow**

1. We used the open-source dataset of lower limb biomechanics from Camargo et al. (2021), which records 22 able-bodied adults walking on a treadmill at 28 speeds, on level ground at 3 speeds, on ramps at 6 inclines (ascending and descending), and on stairs at 4 heights (ascending and descending). This study uses nine subjects (AB06 to AB14).
2. Two input sets were prepared and both are reported. **Input set P** contains 14 joint-angle channels from inverse kinematics. **Input set S** contains the same joint angles plus 24 IMU channels (38 channels in total). Signals were resampled to 60 Hz and cut into non-overlapping windows of 30 frames (0.5 seconds) inside a single trial, so no window crosses a trial boundary. Windows with joint-angle glitches (implausible values or jumps from the motion-capture processing) and windows where the person was standing still were excluded with label-free rules.
3. A single-layer RTRBM with 64 hidden units (the single-layer case of an RTDBN) was trained on windows from the training subjects only, with no labels. A scaler was fit on the training windows only. Each window is then summarized as a 64-dimensional embedding (the time-averaged hidden activations).
4. The embeddings were evaluated on subjects the model had never seen, using probes that were trained on the training subjects' embeddings and labels (see [Terrain and Condition Analysis](terrain-and-condition-analysis.md)). They were compared against baselines that produce embeddings of the same size (see [Baseline Comparison](baseline-comparison.md)). The model's free energy was also used to flag unfamiliar terrain (see [Novelty Detection](novelty-detection.md)).

#### **Statistical Methods**

1. **Subject-level splits.** All evaluations hold out whole subjects. In each of 10 random splits, 5 subjects were used for training, 2 for validation, and 2 for testing, so no person appears in more than one partition.
2. **Probes.** Embeddings are standardized and passed to a logistic regression (linear) probe and a random forest probe. Four-way terrain mode (treadmill, level ground, ramp, stair) is scored with **balanced accuracy**, which averages recall across classes so that common classes do not dominate. Graded conditions are scored with **R²** (treadmill speed) and **Spearman rank correlation** (ramp incline and stair height, computed within each direction of travel).
3. **Novelty detection.** The area under the ROC curve (**AUROC**) measures how well a score separates familiar from unfamiliar windows, where 0.5 is chance and 1.0 is perfect separation.
4. **Uncertainty and multiple comparisons.** Differences between methods are paired across the 10 splits and reported with bootstrap 95% confidence intervals. P-values are corrected for multiple comparisons within each hypothesis with the Holm procedure. When several batches of tests were run on the same subjects, later batches were corrected as a family with a stricter threshold.
5. **Pre-registration.** The hypotheses, splits, baselines, and acceptance rules were written down before the results were computed. Any hypothesis that did not meet its rule is reported as not supported, including the ones in [Conclusions](conclusions.md).

{% content-ref url="terrain-and-condition-analysis.md" %}
[terrain-and-condition-analysis.md](terrain-and-condition-analysis.md)
{% endcontent-ref %}

{% content-ref url="baseline-comparison.md" %}
[baseline-comparison.md](baseline-comparison.md)
{% endcontent-ref %}

{% content-ref url="novelty-detection.md" %}
[novelty-detection.md](novelty-detection.md)
{% endcontent-ref %}

{% content-ref url="conclusions.md" %}
[conclusions.md](conclusions.md)
{% endcontent-ref %}

{% content-ref url="bibliography.md" %}
[bibliography.md](bibliography.md)
{% endcontent-ref %}
