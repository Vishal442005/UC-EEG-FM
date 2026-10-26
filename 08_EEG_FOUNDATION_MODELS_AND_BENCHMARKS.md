# Universal EEG Foundation Models, Self-Supervised Learning, and Cross-Disorder Benchmarks

## 1. The Paradigm Shift: From Narrow DL to Foundation Models

Historically, deep learning for electrophysiology relied on training compact, task-specific networks (e.g., EEGNet, ShallowFBCSPNet) from scratch on small cohorts ($N = 20 - 100$). This paradigm suffered from:
1. **Severe Overfitting & Brittle Generalization:** Models memorized subject-specific skull conductivity or room noise rather than universal neurophysiology.
2. **Channel & Montage Incompatibility:** A model trained on a 64-channel 10-20 cap could not run on a 19-channel clinical montage or a 3-channel wearable headband without re-engineering.
3. **Data Scarcity:** Individual clinical cohorts rarely exceed a few dozen patients.

**The Foundation Model Solution:**
Self-Supervised Learning (SSL) on tens of thousands of hours of unlabeled clinical electrophysiology (e.g., Temple University Hospital Corpus, Sleep Heart Health Study) to learn universal spatio-temporal dynamics, followed by lightweight parameter-efficient fine-tuning (PEFT / LoRA) or linear probing on downstream disorder tasks.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EEG FOUNDATION MODEL LIFECYCLE                           │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
   [ Pre-Training Phase (Unsupervised) ]            [ Downstream Fine-Tuning ]
    • 10,000+ hours of continuous EEG                • Task 1: Seizure (TUSZ)
    • Masked Autoencoding (MAE)                      • Task 2: Alcoholism (UCI/ds005907)
    • Vector Quantized Neural Spectrum (VQ-NSP)      • Task 3: Dementia (ds004504)
    • Contrastive Learning (InfoNCE)                 • Task 4: Depression (MODMA)
    • Universal Spatio-Temporal Tokenizer            • Task 5: Sleep Staging (Sleep-EDF)
```

---

## 2. Leading EEG Foundation Model Architectures

### 2.1 BIOT: Bio-Transformer (Yang et al., NeurIPS)
* **Core Breakthrough:** **Channel-Agnostic Patch Tokenization**.
* **Tokenization Mechanism:**
  * Rather than treating the entire multi-channel montage as a single 2D matrix, BIOT segments each individual channel $c_i$ into non-overlapping temporal patches of duration $\tau$ (e.g., 1.0 second).
  * A 1D convolutional patch encoder projects each temporal patch into a $D$-dimensional latent vector $z_i(t) \in \mathbb{R}^D$.
  * **Spatial Positional Embedding:** Each patch is augmented with a learned embedding derived from the 3D Cartesian coordinates $(x, y, z)$ of electrode $c_i$ in standard MNI space.
* **Pre-Training Objective:** Masked Autoencoding (MAE). $50\% - 75\%$ of spatial-temporal patches are randomly masked out, and the transformer encoder-decoder is tasked with reconstructing the raw time-series.
* **Montage Invariance:** Can ingest 1 channel, 19 channels, or 128 channels interchangeably without changing model weights.

---

### 2.2 LaBraM: Large Brain Model (Jiang et al., ICLR)
* **Core Breakthrough:** **Vector-Quantized Neural Spectrum Prediction (VQ-NSP)**.
* **Architecture:**
  1. **Neural Tokenizer (VQ-VAE):** Converts continuous multi-channel Fourier / Wavelet spectrogram patches into discrete token indices from a learned codebook $\mathcal{C} = \{e_k\}_{k=1}^K$ ($K = 8192$).
  2. **Transformer Backbone:** A massive bidirectional Vision Transformer (ViT-Base / ViT-Large) that processes masked neural tokens.
* **Scale:** Pre-trained on $>25,000$ hours of continuous electrophysiological data pooled across TUH EEG, CHB-MIT, and sleep databases.
* **Performance:** Demonstrates state-of-the-art zero-shot and few-shot transfer across clinical pathology detection, seizure detection, and cognitive state classification.

---

### 2.3 BENDR: BERTing EEG Pathologies (Kostas et al., Frontiers in Neuroscience)
* **Inspiration:** Adapted from wav2vec 2.0 (speech foundation model).
* **Architecture:**
  * Multi-layer 1D Temporal CNN encoder that maps raw multi-channel samples down to latent representations at a lower temporal resolution.
  * Transformer contextualizer with self-attention across the sequence.
* **Pre-Training Loss:**
  * **Contrastive Loss (InfoNCE):** The model must identify the true latent representation of a masked time interval among a set of distractors sampled from other temporal positions.

---

### 2.4 NeuroSync-MoE: Hierarchical Mixture-of-Experts Architecture
* **Specialized Inductive Bias for Electrophysiology:**
  * Recognizing that low-frequency oscillations (Delta/Theta: slow background, ERPs) and high-frequency oscillations (Beta/Gamma: local synchrony, bursts) are governed by fundamentally different biophysical generators:
  * Employs a **Dynamic Routing Gating Network** that routes input EEG tokens to specialized expert sub-networks:
    * *Expert 1 (Slow-Wave & ERP Specialist):* Tuned to long-duration receptive fields ($0.5 - 7\text{ Hz}$).
    * *Expert 2 (Oscillatory & Burst Specialist):* Tuned to fast band-pass filters ($13 - 45\text{ Hz}$).
    * *Expert 3 (Phase-Synchrony Specialist):* Tuned to cross-channel phase difference matrices.
* Outperforms monolithic transformers on heterogeneous multi-disorder transfer tasks.

---

## 3. Comprehensive Cross-Disorder Benchmark Comparison

The following table summarizes the comparative performance of Foundation Models versus traditional supervised baselines across downstream clinical and cognitive datasets under strict **Subject-Independent Cross-Validation**:

| Dataset & Task | Target Metric | Classical ML (SVM / RF) | Supervised DL (EEGNet / ChronoNet) | BENDR (Pretrained) | BIOT (Pretrained) | LaBraM (Pretrained) | NeuroSync-MoE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TUH TUSZ** (Seizure Detection) | TAES F1-Score | $0.32$ | $0.53$ | $0.61$ | $0.68$ | **$0.71$** | $0.69$ |
| **CHB-MIT** (Pediatric Seizure) | Sensitivity / FAR | $91.2\%$ / $0.28\text{h}$ | $93.4\%$ / $0.18\text{h}$ | $95.1\%$ / $0.14\text{h}$ | $97.2\%$ / $0.08\text{h}$ | **$97.8\%$ / $0.07\text{h}$** | $97.4\%$ / $0.07\text{h}$ |
| **Sleep-EDFX** (5-Stage Sleep) | Macro F1-Score | $0.68$ | $0.80$ | $0.81$ | $0.84$ | $0.84$ | **$0.85$** |
| **OpenNeuro ds004504** (AD Dementia) | Balanced Accuracy | $81.5\%$ | $82.8\%$ | $86.4\%$ | $89.4\%$ | **$91.8\%$** | $91.2\%$ |
| **UCI Alcoholism** (AUD Classification)| Subject Accuracy | $72.4\%$ | $78.6\%$ | $79.8\%$ | $82.1\%$ | $83.7\%$ | **$84.5\%$** |
| **MODMA** (128-ch Depression) | Balanced Accuracy | $78.2\%$ | $80.5\%$ | $83.6\%$ | $89.5\%$ | $90.2\%$ | **$91.4\%$** |
| **IEEE Dataport** (ADHD Screening) | Subject Accuracy | $82.4\%$ | $87.6\%$ | $88.9\%$ | $93.7\%$ | **$94.2\%$** | $93.9\%$ |
| **SEED** (Emotion Recognition) | 3-Class Accuracy | $78.4\%$ | $86.2\%$ | $89.1\%$ | $91.5\%$ | **$94.8\%$** | $93.6\%$ |

---

## 4. Key Takeaways & Best Practices for Fine-Tuning

1. **Parameter-Efficient Fine-Tuning (PEFT / LoRA):**
   * Freezing the transformer backbone and injecting Low-Rank Adaptation (LoRA) matrices into the query and value projection layers ($r = 8$, $\alpha = 16$) achieves $98\%$ of full fine-tuning performance while reducing trainable parameters by $95\%$.
2. **Channel Harmonization:**
   * When deploying models across varying montages, utilize 3D spatial coordinate projection rather than zero-padding missing channels.
3. **Data Augmentation Strategies:**
   * Time-domain jittering, random channel masking (Spatial Dropout $p = 0.15$), and random phase perturbation improve cross-subject generalization significantly.
