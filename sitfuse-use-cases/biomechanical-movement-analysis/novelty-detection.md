# Novelty Detection

### **Novelty Detection**

#### **Data Processing**

1. The RTRBM was trained only on level-ground and treadmill windows. Ramp and stair windows were never seen during training, so they play the role of unfamiliar terrain.
2. Every held-out window was given a **free energy** score by the trained RTRBM. Free energy is the model's measure of how poorly a window fits what it learned: familiar windows should score low and unfamiliar windows high.

#### **Novelty Detection Analysis**

1. We measured how well the free energy separates held-out level-ground windows (familiar) from ramp and stair windows (unfamiliar), using AUROC.
2. Free energy was compared with three non-RBM scores: the prediction error of a 5th-order autoregressive model (AR(5)), a difference-Mahalanobis distance, and the reconstruction error of window PCA. On input set S, free energy was also compared with the reconstruction errors of the MLP and GRU autoencoders (exploratory comparison, not part of the original test).

#### **Data Results**

| Score | Input set P | Input set S |
| --- | --- | --- |
| RTRBM free energy | 0.897 | 0.805 |
| AR(5) error | 0.638 | 0.544 |
| Difference-Mahalanobis | 0.606 | 0.545 |
| Window PCA error | 0.727 | 0.683 |

1. Free energy detected unfamiliar terrain better than all three comparison scores on both input sets, and this hypothesis was supported on both.
2. On input set S, free energy also exceeded the autoencoder reconstruction errors by 0.048 (MLP) and 0.055 (GRU). Because this comparison was added after the main test, it should be read as exploratory.
3. A harder version of the problem, detecting subtle differences between neighboring terrain levels rather than a different terrain type altogether, did not meet its acceptance rule.

#### **Study Implementation**

Novelty detection (H3) was pre-registered. It was expected to be difficult based on earlier studies of free-energy anomaly detection and is the clearest positive result of this study. As with the other results, it is based on nine subjects and has not been replicated on additional subjects.
