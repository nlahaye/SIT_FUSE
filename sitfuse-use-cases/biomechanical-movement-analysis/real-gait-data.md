# Real Gait Data

### **Real Gait Data**

#### **Data Acquisition**

1. To test the embeddings outside simulation, we used two independent public human gait datasets, each with the task of classifying walking speed (three classes) from windows of joint angles.
2. **Cohort 1** contains 57 subjects walking at three speeds (0.5, 0.75, and 1.0 m/s), with 66 joint-angle channels derived from inertial sensors at 60 Hz (Katmah et al., 2026; Pollard et al., 2026).
3. **Cohort 2** is the dataset of Bailey et al. (2021), released through SimTK: 14 healthy adults walking on a treadmill at 70%, 100%, and 130% of their preferred speed, with joint angles from IMU-driven inverse kinematics. We kept the 18 rotational channels that vary by more than one degree and dropped the first five seconds of each trial.

#### **Data Processing**

1. Windows are non-overlapping and 30 frames long, standardized with training-subject statistics.
2. Splits are at the subject level (70/15/15 for cohort 1 and 10/2/2 subjects for cohort 2), with ten splits per cohort, so the probes are always scored on subjects that were not used for training.
3. Each model produces a 64-dimensional embedding. For the RTRBM, this is the temporal mean of the hidden states. Models were trained for 30 epochs, with the learning rate chosen only from validation reconstruction error.

#### **Data Analysis**

1. We compared the RTRBM embedding with four learned baselines of the same size: frame-wise PCA, window PCA, an MLP autoencoder, and a GRU (gated recurrent unit) autoencoder. We also included a training-free baseline whose embedding is the per-channel mean, standard deviation, minimum, and maximum over the window.
2. A logistic regression probe and a random forest probe (40 trees) were fit on the frozen embeddings and scored by accuracy on unseen subjects. Differences were paired over the ten splits, with 95% bootstrap intervals and Holm correction.

#### **Data Results**

Walking-speed accuracy on unseen subjects (mean over ten subject-level splits, chance = 0.333):

| Cohort | Probe | RTRBM | PCA frame | PCA window | MLP autoencoder | GRU autoencoder | Summary statistics |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (57 subjects) | Linear | 0.514 | 0.504 | 0.465 | 0.503 | 0.497 | 0.730 |
| 1 (57 subjects) | Random forest | **0.577** | 0.456 | 0.515 | 0.463 | 0.466 | 0.649 |
| 2 (14 subjects) | Linear | 0.529 | 0.437 | 0.455 | 0.446 | 0.500 | 0.567 |
| 2 (14 subjects) | Random forest | **0.610** | 0.413 | 0.559 | 0.498 | 0.479 | 0.649 |

1. **The RTRBM embedding gave the highest random forest accuracy of all the learned encoders in both cohorts.** It scored 0.577 against 0.456 to 0.515 for the four learned baselines in cohort 1, and 0.610 against 0.413 to 0.559 in cohort 2. All four contrasts held after Holm correction in both cohorts (cohort 1: +0.062 to +0.121, Holm p ≤ 5.5 × 10⁻⁴; cohort 2: +0.051 to +0.197, Holm p ≤ 0.018).
2. **With the linear probe, the advantage was smaller.** The RTRBM embedding was better than frame-wise PCA, window PCA, and the MLP autoencoder in cohort 2, and better than window PCA in cohort 1. The remaining linear contrasts did not separate the encoders, so no advantage is claimed there.
3. **Simple summary statistics are a strong baseline.** The training-free summary statistics scored higher than the RTRBM in the table above. The difference was clear in cohort 1 and within noise in cohort 2. The summary statistics use four times as many dimensions as the 64-dimensional learned embeddings, and this comparison was run after the other results were seen.

#### **Study Implementation**

The gait results are limited to these two cohorts and the walking-speed task. Both cohorts consist of healthy adults walking on a treadmill, and subjects are identified only by code.
