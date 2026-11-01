# Baseline Comparison

### **Baseline Comparison**

#### **Baselines**

1. **Summary statistics.** The mean, standard deviation, minimum, and maximum of each channel over the window (56 features on input set P).
2. **PCA.** Principal components of single frames (frame PCA) and of whole flattened windows (window PCA), with 64 components so the embedding matches the RTRBM's size.
3. **Autoencoders.** A multilayer-perceptron autoencoder and a GRU (gated recurrent unit) autoencoder, each with a 64-dimensional bottleneck.

All baselines used the same subject-level splits, scaler, windows, and probes as the RTRBM.

#### **Data Results**

Mean balanced accuracy for four-way locomotion mode with the random forest probe, on held-out subjects:

| Method | Input set P (joints) | Input set S (joints + IMU) |
| --- | --- | --- |
| RTRBM (64 hidden units) | 0.698 | 0.704 |
| Summary statistics | 0.742 | 0.710 |
| Frame PCA | 0.639 | 0.611 |
| Window PCA | 0.594 | not reported here |
| MLP autoencoder | 0.657 | 0.607 |
| GRU autoencoder | 0.669 | 0.608 |

1. The RTRBM embedding beat the PCA and autoencoder baselines, and the gap to the autoencoders was larger on input set S.
2. The summary-statistics baseline was not beaten. It was ahead on input set P and statistically tied on input set S.
3. With a linear probe on input set P, the RTRBM (0.758) scored higher than summary statistics (0.701). Concatenating the RTRBM embedding with the summary statistics gave the best mode accuracy of any feature set we tested (0.796 with the linear probe), so the two feature sets carry partly complementary information.

#### **Data Analysis: Robustness Checks**

1. **Per-subject standardization.** Standardizing each subject's embedding columns with that subject's own mean and standard deviation (unlabeled, but it uses the test subject's windows) improved nearly every method. With it, the RTRBM's lead over the autoencoders mostly disappeared (RTRBM minus MLP autoencoder: -0.001, minus GRU autoencoder: +0.006), while a lead over PCA remained. Much of the RTRBM's advantage over the autoencoders in the unstandardized comparison is therefore subject-level offset that this simple step also removes. We recommend reporting results both with and without it.
2. **Reproducibility.** Embeddings from models trained on different sets of subjects were compared with linear CKA, where 1.0 means identical representations. The RTRBM scored 0.956, lower than PCA and the autoencoders (0.98 to 0.99), meaning its representation changes more when the training subjects change.
3. **Does the recurrent connection help?** Not in these experiments. Across four pre-registered tests, the RTRBM with its recurrent weights removed (W' = 0) did no worse than the model with recurrent weights, and the W' = 0 variant was the best RTRBM variant for speed and stair height. A test of whether the model had learned the direction of time found that the standard model's recurrent weights were inert (AUROC 0.50, chance level). A different variant (canon_wp1) did learn time order (AUROC 0.70 on input set P) but without any benefit for identifying terrain. In this study, the RTRBM's advantage appears to come from its learned hidden-layer representation of a window, not from its recurrence.

#### **Study Implementation**

These comparisons were pre-registered, and later batches of tests on the same subjects were corrected as a family with a stricter threshold. Other pre-registered tests that did not meet their acceptance rules (clustering agreement with terrain labels, ascent versus descent, level-ground speed classes, held-out terrain levels, and filling in missing channels) are listed on the [Conclusions](conclusions.md) page.
