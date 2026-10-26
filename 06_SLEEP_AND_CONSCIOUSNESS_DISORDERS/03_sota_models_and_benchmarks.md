# SOTA Models, Deep Learning Architectures, and Benchmarks in Sleep & Consciousness

## 1. Algorithmic Formulations for Sleep Staging & Coma

Automatic sleep scoring maps continuous polysomnographic time-series into categorical discrete stages ($y_t \in \{\text{W}, \text{N1}, \text{N2}, \text{N3}, \text{REM}\}$).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TWO-TIER SLEEP STAGING ARCHITECTURE                      │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
   [ Intra-Epoch Feature Extractor ]             [ Inter-Epoch Sequence Model ]
    • Dual-Scale 1D-CNN (Low / High Freq)         • Bidirectional LSTM / GRU
    • Temporal Depthwise Separable Conv           • Multi-Head Self-Attention
    • Captures Spindles, K-Complexes, Delta       • Enforces Sleep Cycle Grammar
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      ▼
                        [ 5-Stage Softmax Classifier ]
                           (W, N1, N2, N3, REM)
```

---

## 2. Benchmark Deep Learning Architectures

### 2.1 DeepSleepNet & TinySleepNet
* **DeepSleepNet (Supratak et al.):**
  * *Intra-Epoch Branch:* Uses two parallel CNN sub-networks: a small filter branch ($f_s / 2$) to extract fine temporal patterns (spindles, bursts), and a large filter branch ($f_s \times 4$) to capture slow background waves (delta, theta).
  * *Inter-Epoch Branch:* Two-layer Bidirectional LSTM capturing historical context (e.g., probability of transitioning from N2 to N3 vs. Wake).
* **TinySleepNet:**
  * Compresses DeepSleepNet into a lightweight architecture ($< 1.3\text{ MB}$ parameter footprint) suitable for on-chip inference in wearable sleep headbands and bedside monitors.

### 2.2 U-Sleep (Fully Convolutional Architecture)
* **Architecture:** Developed by Perslev et al. (NPJ Digital Medicine). Formulated as a 1D U-Net encoder-decoder with skip connections.
* **Advantage:** Eliminates fixed 30-second epoch constraints and handles arbitrary input lengths, variable sampling rates ($50 - 500\text{ Hz}$), and differing electrode channels without retraining.

### 2.3 Post-Cardiac Arrest Coma Prognostication Frameworks
* Evaluates continuous 19-channel ICU EEG in the George B. Moody PhysioNet Challenge 2023.
* **Architecture:** Combines spectral edge frequency, burst suppression ratio (BSR), and self-supervised transformers (BIOT) to predict 72-hour **Cerebral Performance Category (CPC)** scores.
* **Clinical Safety Constraint:** Algorithms must achieve **$\text{False Positive Rate} \le 0.05$** when predicting poor outcome (CPC 3–5) to prevent catastrophic premature withdrawal of life support.

---

## 3. SOTA Benchmark Comparisons

### 3.1 Sleep-EDF Expanded Benchmark (20 Subjects / 78 Subjects - SC Cohort)

Evaluated under strict 10-fold or 20-fold **Subject-Independent Cross-Validation**:

| Architecture | Input Montage | Overall Accuracy (%) | Macro F1-Score | Stage-wise F1: W | N1 | N2 | N3 | REM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AASM Human Inter-Rater**| Manual Expert Scoring | $86.5\%$ | $0.80$ | $0.90$ | $0.52$ | $0.88$ | $0.86$ | $0.87$ |
| **Random Forest + Multi-band**| Single EEG (`Fpz-Cz`) | $74.2\%$ | $0.68$ | $0.81$ | $0.28$ | $0.78$ | $0.74$ | $0.72$ |
| **DeepSleepNet** | Single EEG (`Fpz-Cz`) | $82.0\%$ | $0.76$ | $0.87$ | $0.44$ | $0.85$ | $0.84$ | $0.82$ |
| **TinySleepNet** | Single EEG (`Fpz-Cz`) | $85.4\%$ | $0.80$ | $0.90$ | $0.47$ | $0.88$ | $0.85$ | $0.85$ |
| **SeqSleepNet** | Single EEG (`Fpz-Cz`) | $86.5\%$ | $0.81$ | $0.91$ | $0.49$ | $0.89$ | $0.86$ | $0.86$ |
| **U-Sleep (End-to-End)** | Single EEG (`Fpz-Cz`) | $86.8\%$ | $0.82$ | $0.91$ | $0.51$ | $0.89$ | **$0.88$**| **$0.88$**|
| **AttnSleep + BIOT** | Multimodal (`EEG+EOG`)| **$88.4\%$** | **$0.84$** | **$0.93$** | **$0.55$** | **$0.90$**| $0.87$ | **$0.89$**|

*(Note: N1 accuracy is universally lowest ($45\% - 55\%$) across both human experts and artificial intelligence models due to the subtle, gradual neurophysiological transition between wakefulness and light sleep).*

---

### 3.2 PhysioNet Challenge 2023: Post-Cardiac Arrest Coma Outcome Benchmark

Evaluated on hidden ICU test cohorts predicting 72-hour Cerebral Performance Category (CPC):

| Model Architecture | Input Features | TPR at FPR $\le 0.05$ | AUC-ROC | Challenge Rank |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline Heuristic** | Burst Suppression Ratio (BSR) | $0.34$ | $0.72$ | Baseline |
| **Ensemble Gradient Boosted Trees**| Quantitative EEG (qEEG) Metrics| $0.58$ | $0.86$ | Top 10 |
| **1D-ResNet + Attention** | Multi-channel Continuous ICU EEG| $0.66$ | $0.91$ | Top 3 |
| **BIOT + Clinical Metadata Fusion**| Pretrained Transformer + Age/ROSC| **$0.72$** | **$0.94$** | **Winning Paradigm** |

---

## 4. Clinical Deployment & Edge Computing Considerations

1. **Lightweight Edge Hardware:**
   * TinySleepNet and Mobile-Sleep architectures can run directly on ARM Cortex-M4 or ultra-low-power microcontrollers inside wireless dry-electrode sleep headbands, consuming $< 25\text{ mW}$ of power.
2. **Automated Micro-Arousal & Apnea Detection:**
   * Coupling sleep stage predictions with instantaneous high-frequency power spikes enables automated computation of the **Apnea-Hypopnea Index (AHI)** and **Respiratory Disturbance Index (RDI)** with $>90\%$ clinical agreement with in-lab polysomnography.
