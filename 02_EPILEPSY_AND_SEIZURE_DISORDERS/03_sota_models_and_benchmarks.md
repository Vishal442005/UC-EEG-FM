# SOTA Models, Deep Learning Architectures, and Benchmarks in Epilepsy

## 1. Machine Learning Formulation & Evaluation Frameworks

Epilepsy computational electrophysiology spans three primary clinical tasks, each with distinct problem formulations, loss functions, and evaluation metrics:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             CLINICAL TASKS                                  │
├─────────────────────────┬─────────────────────────┬─────────────────────────┤
│ 1. Seizure Detection    │ 2. Seizure Prediction   │ 3. Epileptogenic Zone   │
│    (Ictal vs. Non-Ictal)│    (Preictal vs. Inter) │    (Focal Localization) │
├─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ • Goal: Real-time alarm │ • Goal: Early warning   │ • Goal: Surgical        │
│   in EMU / ICU          │   window (10 - 60 min)  │   resection planning    │
│ • Metrics: Sensitivity, │ • Metrics: SPH, SOP,    │ • Metrics: Channel-level│
│   FAR/hr, TAES F1       │   AUC-ROC, FAR/24hr     │   Precision / Recall    │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

### 1.1 Clinical Evaluation Metrics (Avoiding the Imbalance Illusion)
In long-term continuous EEG monitoring, seizures account for $< 0.1\%$ of total recording duration. Reporting standard accuracy is clinically meaningless ($99.9\%$ accuracy is achieved by a dummy classifier predicting "no seizure").

Standard clinical metrics required by neurologists and clinical trials:
* **Sensitivity (True Positive Rate / Recall):** Proportion of true clinical seizures detected:
  $$\text{Sensitivity} = \frac{TP}{TP + FN}$$
* **False Alarm Rate per Hour ($\text{FAR/h}$):** Number of false positive seizure alarms triggered per hour of continuous recording:
  $$\text{FAR/h} = \frac{\text{Number of False Alarms}}{\text{Total Monitoring Duration in Hours}}$$
  *(Clinical target in EMU: $\text{FAR/h} < 0.2$; Target in ambulatory implants: $\text{FAR/h} < 0.05$)*.
* **Seizure Prediction Horizon (SPH) & Seizure Occurrence Period (SOP):**
  * $\text{SPH}$: The intervention window between the alarm trigger and the beginning of the seizure (e.g., 5–10 minutes for fast-acting AED administration).
  * $\text{SOP}$: The time interval during which the seizure is predicted to occur (e.g., 30–60 minutes).
* **Time-Aligned Event Scoring (TAES):** Developed by the Temple University NEDC consortium to reward early onset detection while penalizing fragmented or delayed detections.

---

## 2. Advanced Model Architectures

### 2.1 Spatio-Temporal Graph Convolutional Networks (ST-GCN)
Because scalp electrodes are fixed on a 3D Riemannian manifold, Graph Neural Networks excel over standard Euclidean 2D convolutions:
* **Nodes:** Scalp electrodes $v_i \in \mathcal{V}$.
* **Adjacency Matrix $\mathbf{A}$:** Dynamic phase synchrony matrix computed via $wPLI$ or mutual information over sliding windows:
  $$\mathbf{A}_{i,j}(t) = \text{wPLI}_{i,j}(t)$$
* **Graph Convolution:** ChebNet or GCN filters applied to aggregate spatial neighborhood features, followed by 1D dilated temporal convolutions (TCN) capturing temporal rhythmicity.

### 2.2 BIOT (Bio-Transformer) Foundation Model
* **Innovations:** BIOT introduces a channel-agnostic patch tokenizer. Each EEG channel is divided into short temporal patches (e.g., 1-second chunks), embedded via 1D convolutions, and augmented with spatial positional embeddings corresponding to 3D Cartesian electrode coordinates $(x, y, z)$.
* **Montage Invariance:** Enables pre-training on disparate corpora (CHB-MIT, TUSZ, private clinical archives) without requiring fixed channel counts or montages.

### 2.3 ChronoNet & Inception-Time
* Employs parallel 1D convolutional filters with multiple kernel sizes ($32, 64, 128$ samples), capturing both brief sharp-wave transients ($\sim 50\text{ ms}$) and sustained rhythmic delta slow-wave activity ($\sim 500\text{ ms}$) simultaneously, followed by recurrent GRU layers.

---

## 3. SOTA Benchmark Comparisons

### 3.1 CHB-MIT Seizure Detection Benchmark (Pediatric Scalp EEG)

Evaluated under strict cross-subject or patient-specific continuous protocols:

| Architecture | Paradigm | Sensitivity (%) | FAR / Hour | Latency (s) | Key Reference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM + Wavelet Energy / Entropy** | Patient-Specific | $91.2\%$ | $0.28\text{ / h}$ | $4.2\text{ s}$ | Shoeb et al. (MIT) |
| **EEGNet-Fusion** | Cross-Subject (LOSO) | $88.5\%$ | $0.34\text{ / h}$ | $3.1\text{ s}$ | Lawhern et al. |
| **ChronoNet (CNN + GRU)** | Cross-Subject (LOSO) | $93.4\%$ | $0.18\text{ / h}$ | $2.4\text{ s}$ | Roy et al. |
| **Spatio-Temporal GNN (ST-GNN)** | Cross-Subject (LOSO) | $95.8\%$ | $0.12\text{ / h}$ | $1.8\text{ s}$ | Wang et al. (IEEE TBME) |
| **BIOT (Pretrained Bio-Transformer)** | Zero-Shot / Fine-Tuned | **$97.2\%$** | **$0.08\text{ / h}$** | **$1.2\text{ s}$** | Yang et al. (NeurIPS) |
| **Diffusion-Augmented Conformer** | Patient-Specific | **$98.6\%$** | **$0.05\text{ / h}$** | **$1.1\text{ s}$** | 2024 SOTA Benchmark |

---

### 3.2 Temple University Hospital Seizure Corpus (TUSZ v2.0+) Benchmark

TUSZ is notoriously challenging due to real-world clinical artifacts, multi-etiology pathology, and diverse hospital noise:

| Model Architecture | Task | Overall Sensitivity (%) | Specificity (%) | TAES F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **NEDC Baseline (HMM + MLP)** | Binary Seizure Detect | $56.3\%$ | $74.2\%$ | $0.32$ |
| **ResNet-1D + LSTM** | Binary Seizure Detect | $68.4\%$ | $82.5\%$ | $0.46$ |
| **DenseNet-BiLSTM** | Binary Seizure Detect | $74.1\%$ | $88.3\%$ | $0.53$ |
| **Graph Attention Net (GAT-TCN)** | 6-Class Seizure Typing | $77.6\%$ | $91.4\%$ | $0.59$ |
| **BIOT (Self-Supervised Pretraining)** | Binary Seizure Detect | **$84.2\%$** | **$94.1\%$** | **$0.68$** |
| **LaBraM (Large Brain Model)** | Binary Seizure Detect | **$86.5\%$** | **$95.2\%$** | **$0.71$** |

---

### 3.3 American Epilepsy Society / Melbourne Seizure Prediction Benchmark

Forecasting pre-ictal transitions prior to onset:

| Architecture | Prediction Horizon (SPH / SOP) | Patient Sensitivity (%) | AUC-ROC | False Positive Rate / 24h |
| :--- | :--- | :--- | :--- | :--- |
| **Spectral Power + Random Forest** | SPH: 5 min, SOP: 30 min | $74.2\%$ | $0.78$ | $0.32\text{ / 24h}$ |
| **Bi-LSTM on Phase-Locking Value** | SPH: 10 min, SOP: 40 min | $81.5\%$ | $0.84$ | $0.21\text{ / 24h}$ |
| **3D-CNN on Spectrograms** | SPH: 5 min, SOP: 30 min | $87.3\%$ | $0.89$ | $0.15\text{ / 24h}$ |
| **Transformer + Wavelet Scatter** | SPH: 15 min, SOP: 60 min | **$92.1\%$** | **$0.94$** | **$0.09\text{ / 24h}$** |

---

## 4. Engineering & Deployment Guidelines for Real-Time Systems

1. **Sliding Window Inference:**
   * Analysis window: $2.0 - 4.0\text{ seconds}$ with $75\% - 87.5\%$ overlap ($0.5\text{ s}$ stride) ensures low latency detection ($\le 2.0\text{ s}$ delay).
2. **Post-Processing Hysteresis Thresholding:**
   * Raw model probabilities fluctuate rapidly. Apply dual-threshold hysteresis or moving average smoothing:
     * *Trigger Condition:* Output probability $P(t) \ge \tau_{\text{high}}$ (e.g., $0.85$) for at least 3 consecutive frames.
     * *Clear Condition:* $P(t) \le \tau_{\text{low}}$ (e.g., $0.30$) for 5 consecutive frames.
3. **Artifact Disqualification:**
   * High-amplitude muscle artifact (EMG) or electrode pop must trigger an "Artifact / Uninterpretable" flag rather than an erroneous seizure alarm.
