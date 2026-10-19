# Dense Semantic Segmentation of ECG Signals

### A 1D Spatiotemporal U-Net Approach

![Project](https://img.shields.io/badge/Project-Foundations%20of%20Deep%20Learning-1f4e79)
![Domain](https://img.shields.io/badge/Domain-Biomedical%20Signal%20Processing-247a6b)
![Task](https://img.shields.io/badge/Task-1D%20Semantic%20Segmentation-6554a4)

**University of Milano-Bicocca · Master's Degree in Data Science**  
**Foundations of Deep Learning — Final Project · Academic Year 2025/2026**

**Authors:** Simone Fabio Taurisano and Biagio Chiarandà

## Overview

This project explores **dense semantic segmentation of electrocardiogram (ECG) signals** using a **one-dimensional (1D) U-Net**. Rather than assigning a single label to an entire recording, the model predicts a cardiac-wave category **at every time sample**, preserving the temporal location of physiological events.

The approach combines ECG data engineering, patient-aware evaluation, and a spatiotemporal encoder–decoder architecture to identify **P waves, QRS complexes, T waves, and background regions** in ECG recordings sampled at **250 Hz**.

The reported experiment achieved **93% overall sample-level test accuracy** on a patient-separated test set, with strong performance across all four classes. The project also investigates class imbalance, ambiguous wave boundaries, and difficult noisy recordings.

> **Scope:** This repository documents an academic deep learning project. The results are research findings, not evidence of clinical validation, and the model is **not intended for medical diagnosis or clinical decision-making**.

## Project Presentation

- **[View the project presentation (PDF)](./FDL_Project%283%29.pdf)**

The presentation contains the problem formulation, preprocessing pipeline, architecture, quantitative evaluation, qualitative examples, ablation studies, and failure analysis. If you rename the PDF after uploading it, update the link above.

## Research Problem

An ECG contains multiple physiological components whose precise **onset and offset** are important for waveform analysis. A traditional recording-level classifier cannot directly identify the temporal boundaries of these components.

We therefore formulated the task as **sequence-to-sequence, per-sample semantic segmentation**:

| Label | Class | Physiological interpretation |
|---|---|---|
| `0` | Background / N/A | Isoelectric line or unlabeled background |
| `1` | P wave | Atrial depolarization |
| `2` | QRS complex | Ventricular depolarization |
| `3` | T wave | Ventricular repolarization |

**Input:** an ECG window of **1,000 samples (4 seconds at 250 Hz)**.  
**Output:** a temporal sequence of four-class predictions aligned to the input samples.

## Key Challenges

- **Severe class imbalance:** background represents over half of all samples, while P waves account for less than 11%.
- **Precise localization:** the model must identify wave boundaries, not simply detect whether a wave occurs.
- **Noisy recordings:** baseline wander, high-frequency noise, and amplitude variation complicate segmentation.
- **Patient-level generalization:** adjacent windows from one patient can be highly correlated, making subject-aware data splitting essential.

## Data Preparation

The source presentation describes an annotated ECG dataset originating from **PhysioNet** and handled initially through a MATLAB file named `QTData.mat`.

### MATLAB preprocessing

1. Handle nested MATLAB data structures and varying recording lengths using cell arrays.
2. Convert categorical temporal annotations to numerical `uint8` labels.
3. Export a cleaned MATLAB dataset in `-v7` format for subsequent Python processing.

### Python preparation

1. Partition continuous signals into **1,000-sample windows**.
2. **One-hot encode** targets for categorical cross-entropy training.
3. Split data **by patient**, keeping each patient's windows exclusively in either training or testing to reduce leakage.

The presentation reports **18,858 training windows** and **4,715 testing windows**, generated from recordings associated with **105 patients**. These are window counts, not patient counts.

## Model Architecture

The proposed network adapts the **U-Net encoder–decoder design** to one-dimensional ECG time series.

| Component | Role |
|---|---|
| **Conv1D encoder** | Learns hierarchical waveform representations |
| **MaxPooling1D** | Compresses temporal representations and increases receptive field |
| **UpSampling1D decoder** | Restores the original temporal resolution |
| **Skip connections** | Reintroduce fine-grained features to preserve waveform boundaries |
| **Softmax output** | Predicts probabilities for four cardiac-wave classes at every sample |

The encoder captures larger-scale waveform context, while the decoder and skip connections help localize the onsets and offsets of P, QRS, and T regions.

## Training Strategy

To counter background dominance, the project uses **class-weighted categorical cross-entropy**. Sample weights are inversely related to class frequency, with reported example weights of approximately **2.35 for P waves** and **0.49 for background**.

Training also incorporates Keras callbacks:

- **`ReduceLROnPlateau`** to decrease learning rate when the monitored loss stagnates.
- **`EarlyStopping`** to limit overfitting (validation-loss patience: **10 epochs**).
- **`ModelCheckpoint`** to retain the best-performing trained weights.

## Results

### Patient-aware test-set evaluation

| Class | Precision | Recall | Intersection over Union (IoU) | Dice coefficient |
|---|---:|---:|---:|---:|
| Background | 0.96 | 0.91 | 0.88 | 0.94 |
| P wave | 0.87 | 0.95 | 0.83 | 0.91 |
| QRS complex | 0.95 | 0.98 | 0.94 | 0.97 |
| T wave | 0.90 | 0.94 | 0.85 | 0.92 |

**Overall sample-level accuracy: 0.93 (93%).**

These values are reported in the project presentation. They reflect performance under the study's evaluation conditions and should not be treated as independently verified clinical performance.

### Main observations

- **QRS segmentation** achieved the strongest overlap score (**IoU 0.94; Dice 0.97**).
- **P-wave recall** reached **0.95**, despite substantial underrepresentation in the training data.
- Qualitative plots showed close alignment between many predicted regions and the annotated wave boundaries.
- The confusion-matrix discussion identifies the **T-wave/background boundary** as a notable source of errors, where the end of repolarization can be difficult to define precisely.

## Ablation Studies and Alternative Approaches

The project explores two important alternatives:

**Removing U-Net skip connections.** A simpler convolutional configuration produced temporal displacement of predicted QRS boundaries, illustrating the value of retaining higher-resolution features during decoding.

**Dice loss versus weighted categorical cross-entropy.** Experiments with Dice loss showed unstable optimization during early training, particularly for sparse P-wave samples. Weighted categorical cross-entropy was reported as more stable and better converging in this setup.

## Failure Analysis and Limitations

An analysis of the most difficult test segment found an accuracy of **42.10%**. Severe artifacts and baseline wander disrupted temporal localization and caused fragmented predictions.

Additional limitations include:

- Evaluation on the available annotated ECG data does not establish generalization to every recording device, clinical environment, or patient population.
- Heavy noise and shifting baselines can degrade morphology-based boundary detection.
- The dataset's class imbalance makes per-class precision, recall, IoU, and Dice essential alongside overall accuracy.
- The presentation summarizes the experiments but does not, by itself, specify a complete reproducible software environment or public model checkpoint.

## Possible Future Directions

- Improve robustness to motion artifacts and baseline wander through more targeted signal augmentation or filtering.
- Investigate boundary-aware objectives and loss functions for difficult wave transitions.
- Evaluate generalization on additional independently sourced patient cohorts.
- Analyze segmentation uncertainty and develop calibrated confidence estimates.
- Assess inference efficiency for future real-time or wearable ECG applications.

## Tools and Concepts

The project presentation discusses **MATLAB**, **Python**, the **Keras API**, and the following deep learning techniques:

`Conv1D` · `MaxPooling1D` · `UpSampling1D` · `U-Net` · `Softmax` · `One-Hot Encoding` · `Weighted Categorical Cross-Entropy` · `EarlyStopping` · `ReduceLROnPlateau` · `ModelCheckpoint`

## References

- **PhysioNet:** Goldberger, A. L., et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals.* [PhysioNet](https://physionet.org/)
- **U-Net:** Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

## Authors and Academic Context

**Simone Fabio Taurisano** and **Biagio Chiarandà**  
Master's Degree in Data Science  
**University of Milano-Bicocca**  
*Foundations of Deep Learning — Final Project, 2025/2026*

---

*Academic project for educational and research purposes. Not a certified medical device.*
