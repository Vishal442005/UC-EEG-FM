# SOTA Models, Deep Learning Architectures, and Benchmarks in Psychiatric Disorders

## 1. Machine Learning Pipelines for Psychiatric Neurophysiology

Psychiatric electrophysiology requires models that account for both high-dimensional temporal dynamics and whole-brain network reconfiguration:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PSYCHIATRIC ML / DL TAXONOMY                             │
├─────────────────────────┬─────────────────────────┬─────────────────────────┤
│ 1. Microstate Modeling  │ 2. Spatial Graph Models │ 3. Deep Temporal Models │
├─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ • Canonical A, B, C, D  │ • Dynamic wPLI Graph    │ • 1D-CNN + BiLSTM       │
│ • Duration & Occurrence │ • Spectral GCN / GAT    │ • Multi-scale Conformer │
│ • Transition Prob. Matrix│ • Hub Centrality       │ • Pretrained BIOT / SSL │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

---

## 2. Model Architectures & Feature Frameworks

### 2.1 EEG Microstate Sequence Analysis + Markov Chains
1. Continuous multi-channel EEG is band-pass filtered ($2 - 20\text{ Hz}$).
2. Instantaneous global field power (GFP) peaks are extracted:
   $$\text{GFP}(t) = \sqrt{\frac{1}{C}\sum_{i=1}^C (V_i(t) - \bar{V}(t))^2}$$
3. Spatial topographies at GFP peaks are clustered into canonical classes (A, B, C, D) using modified K-Means.
4. Extracted features:
   * **Duration:** Average duration (ms) a microstate class remains stable.
   * **Occurrence:** Frequency of appearance per second.
   * **Coverage:** Percentage of total time spent in each class.
   * **Transition Probability Matrix:** A $4 \times 4$ first-order Markov matrix describing transition dynamics.

### 2.2 Hierarchical Spatial-Temporal Graph Attention Networks (ST-GAT)
* Captures fronto-limbic disconnectivity in depression and thalamocortical desynchronization in schizophrenia.
* Spatial Attention assigns adaptive weights to functional edges ($wPLI$), followed by dilated temporal convolutions extracting oscillatory band frequencies.

### 2.3 Self-Supervised Pretraining (BIOT & BENDR)
* Large models pre-trained via masked autoencoding on clinical corpora (e.g. TUH) learn general representations of physiological noise and brain rhythms.
* Fine-tuning on MODMA or Warsaw datasets converges rapidly and mitigates the small sample size problem ($N < 60$).

---

## 3. SOTA Benchmark Comparisons

### 3.1 MODMA Depression Benchmark: 128-Channel High-Density EEG ($N=53$)

Evaluated under strict **Subject-Independent Cross-Validation** (Group 5-Fold / LOSO):

| Model Architecture | Input Features | Subject-Level Accuracy (%) | Sensitivity (%) | Specificity (%) | AUC-ROC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM (RBF Kernel)** | Band Power PSD + FAA | $73.6\% \pm 3.4\%$ | $70.8\%$ | $75.9\%$ | $0.79$ |
| **Random Forest** | Differential Entropy (DE) | $78.2\% \pm 2.8\%$ | $75.0\%$ | $81.0\%$ | $0.83$ |
| **EEGNet-8,2** | Raw 128-ch Scalp EEG | $80.5\% \pm 2.5\%$ | $79.2\%$ | $82.8\%$ | $0.86$ |
| **Microstate Markov + XGBoost** | Microstate A-D Stats | $82.4\% \pm 2.1\%$ | $83.3\%$ | $81.5\%$ | $0.88$ |
| **Spatial-Temporal GCN (ST-GCN)**| DE + wPLI Graph | $86.7\% \pm 1.8\%$ | $87.5\%$ | $86.2\%$ | $0.92$ |
| **BIOT (Pretrained Transformer)**| Raw Scalp Patches | **$89.5\% \pm 1.4\%$** | **$91.7\%$** | **$87.9\%$** | **$0.95$** |
| *Flawed Random Epoch Split* | *Raw Spectrograms* | *$98.6\%$ (Invalid)* | *$98.2\%$* | *$99.0\%$* | *$0.99$* |

---

### 3.2 MODMA Wearable 3-Channel Portable Depression Screening ($N=55$, `Fp1, Fp2, Fz`)

Translational point-of-care screening using ultra-sparse frontal montages:

| Model Architecture | Input Channels | Accuracy (%) | Sensitivity (%) | Specificity (%) | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | FAA (`Fp2 - Fp1`) | $65.4\%$ | $61.5\%$ | $68.9\%$ | $0.64$ |
| **LightGBM** | DE + Spectral Ratios | $74.5\%$ | $73.1\%$ | $75.8\%$ | $0.74$ |
| **1D-CNN + GRU** | Raw 3-Channel Time-Series | $78.2\%$ | $76.9\%$ | $79.3\%$ | $0.77$ |
| **Conformer-Lite** | Continuous CWT Scalograms | **$81.8\%$** | **$80.8\%$** | **$82.7\%$** | **$0.81$** |

---

### 3.3 Warsaw Schizophrenia Benchmark (19-Channel Resting EEG, $N=28$)

Binary classification (Paranoid Schizophrenia vs. Matched Healthy Controls):

| Model Architecture | Input Modality | Validation Strategy | Accuracy (%) | F1-Score | AUC-ROC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **k-NN / Random Forest** | Wavelet Entropy + PSD | Leave-One-Subject-Out | $78.6\%$ | $0.79$ | $0.82$ |
| **ChronoNet** | Raw 19-Channel Signals | Leave-One-Subject-Out | $82.1\%$ | $0.82$ | $0.87$ |
| **Microstate Dynamic Net** | Class C/D Transition Rates | Leave-One-Subject-Out | $85.7\%$ | $0.86$ | $0.91$ |
| **Graph Attention Network (GAT)**| Dynamic Phase Coherence | Leave-One-Subject-Out | $89.3\%$ | $0.89$ | $0.93$ |
| **LaBraM (Fine-Tuned)** | Neural Spectrum Tokens | Leave-One-Subject-Out | **$92.8\%$** | **$0.93$** | **$0.96$** |

---

## 4. Antidepressant Treatment Response Prediction (The EMBARC Benchmark)

* **Objective:** Predict whether an MDD patient will achieve clinical remission ($\text{HAMD} \le 7$) at Week 8 following treatment with Sertraline vs. Placebo.
* **Clinical Significance:** Current clinical practice relies on empirical trial-and-error, with each medication trial requiring 6–12 weeks to determine efficacy (first-line remission rate is only ~30%).
* **SOTA Electrophysiological Predictor:**
  * **Latent Space Modeling on Pretreatment Resting EEG + Theta Cordance:** Achieves **$73.8\%$ out-of-sample accuracy** and an **AUC-ROC of $0.81$** in predicting differential Sertraline response over Placebo.
  * Patients identified as Sertraline-responsive show significantly greater baseline rostral anterior cingulate (rACC) theta power and specific alpha-band connectivity profiles.
