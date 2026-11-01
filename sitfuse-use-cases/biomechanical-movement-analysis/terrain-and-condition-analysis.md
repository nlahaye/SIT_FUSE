# Terrain and Condition Analysis

### **Terrain and Condition Analysis**

#### **Data Acquisition**

1. We obtained the dataset from Camargo et al. (2021), an open-source collection of lower limb biomechanics released under a CC BY 4.0 license. We used nine of its subjects (AB06 to AB14).
2. Each trial provides inverse-kinematics joint angles, IMU signals, and per-sample labels for the locomotion mode (treadmill, level ground, ramp, stair), direction of travel (ascent or descent), and treadmill speed. Ramp incline and stair height are taken from the trial's condition index. The index is ordinal, and we did not convert it to degrees or centimeters.

#### **Data Processing**

1. Joint angles and IMU signals were resampled to 60 Hz and cut into non-overlapping 30-frame windows (0.5 seconds) inside a single trial. Fourteen joint-angle channels form input set P, and 24 IMU channels added to them form input set S.
2. Windows were excluded everywhere (training, validation, and testing) if they were standing still, contained a joint-angle glitch from the inverse-kinematics processing (an absolute value above 120 degrees or a jump of more than 40 degrees between consecutive frames), or crossed a label change. These rules do not use labels. About 1% of the non-static windows were removed by the glitch filter.
3. The RTRBM was trained without labels on windows from the training subjects only. The learning rate was chosen once, using reconstruction error on validation subjects from a separate pilot split, before any test result was computed.

#### **Data Analysis**

1. **Terrain mode.** A probe predicted the locomotion mode (treadmill, level ground, ramp, stair) from each window's embedding. Scores are balanced accuracy on held-out subjects, averaged over 10 subject-level splits.
2. **Graded conditions.** A ridge regression predicted treadmill speed (R²). Probes predicted ramp incline and stair height, scored with Spearman correlation inside each direction of travel, so the score reflects grading within ascent or descent rather than telling ascent from descent.
3. **Label efficiency.** The mode probe was retrained using only 5% of the training labels (5 random draws per split) to test how much labeled data the embedding needs.

#### **Data Results**

1. **Terrain mode (supported).** Using the random forest probe, the RTRBM embedding identified the locomotion mode of held-out subjects better than same-size PCA and autoencoder embeddings on both input sets. Mean balanced accuracy was 0.698 for the RTRBM on input set P and 0.704 on input set S. The comparison with the summary-statistics baseline is discussed on the [Baseline Comparison](baseline-comparison.md) page.
2. **Graded conditions (supported on input set S only).** On joint angles alone (P), the RTRBM did not beat the best baseline on speed, incline, or stair height. With IMU channels added (S), it did. Treadmill speed R² was 0.846 for the best RTRBM variant versus 0.536 for window PCA, incline correlation was 0.883 versus 0.845 for the GRU autoencoder, and stair-height correlation was 0.753 versus 0.623 for the GRU autoencoder.
3. **Label efficiency (supported on input set P).** With 5% of the training labels, the random forest probe on RTRBM embeddings reached a balanced accuracy of 0.652, compared with 0.571 for frame PCA, 0.593 for the MLP autoencoder, 0.603 for the GRU autoencoder, and 0.483 for window PCA. On input set S the same test narrowly missed its stricter pre-registered threshold, so it is not claimed there.

#### **Study Implementation**

The pre-registered hypotheses for this section were terrain mode (H1), graded condition (H2), and label efficiency (H9). All were evaluated with subject-level splits and Holm-corrected paired bootstrap tests as described in the [overview](README.md). The results are preliminary and have not yet been replicated on a second set of subjects.
