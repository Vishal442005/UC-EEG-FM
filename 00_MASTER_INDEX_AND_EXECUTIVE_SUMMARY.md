# Master Research Index and Executive Summary: Comprehensive EEG Disorders & Datasets Compendium

## 1. Executive Summary & Purpose

This compendium represents an exhaustive, state-of-the-art research survey and data registry across all major clinical, cognitive, and behavioral disorders investigated via **Electroencephalography (EEG)**. 

Rooted in neuroscience, clinical electrophysiology, and advanced machine learning / foundation models, this repository organizes the complete landscape of EEG biomarker discoveries, open-access clinical datasets, algorithmic architectures, and validation standards.

---

## 2. Compendium Architecture & Navigation Directory

The repository is structured into modular research domains. Click below to navigate directly to specialized files:

```
All EEG/
├── 00_MASTER_INDEX_AND_EXECUTIVE_SUMMARY.md                (This Master Document)
│
├── 01_ALCOHOLISM_AND_SUBSTANCE_ADDICTION/
│   ├── 01_neurobiology_and_biomarkers.md                  (AUD, Opioids, Cocaine, Cannabis, IGD)
│   ├── 02_comprehensive_datasets_catalogue.md             (UCI Begleiter, ds005907, ds004515, ds008851)
│   └── 03_sota_models_and_benchmarks.md                   (EEGNet, ST-GNN, MoE, Subject-KFold)
│
├── 02_EPILEPSY_AND_SEIZURE_DISORDERS/
│   ├── 01_neurobiology_and_biomarkers.md                  (IEDs, HFOs, PAC, Ictal/Preictal/Postictal)
│   ├── 02_comprehensive_datasets_catalogue.md             (CHB-MIT, TUH TUSZ, Bonn, Bern-Barcelona, Siena)
│   └── 03_sota_models_and_benchmarks.md                   (Detection vs. Prediction, BIOT, LaBraM, TAES)
│
├── 03_NEURODEGENERATIVE_AND_DEMENTIA/
│   ├── 01_neurobiology_and_biomarkers.md                  (Alzheimer's, MCI, FTD, Parkinson's Beta Bursts)
│   ├── 02_comprehensive_datasets_catalogue.md             (OpenNeuro ds004504, Zenodo Dementia, ds002778)
│   └── 03_sota_models_and_benchmarks.md                   (Early AD, Differential FTD, CWT ResNet, FC-GAT)
│
├── 04_PSYCHIATRIC_AND_MOOD_DISORDERS/
│   ├── 01_neurobiology_and_biomarkers.md                  (MDD FAA, rACC Theta, P50 Gating, MMN, ASSR)
│   ├── 02_comprehensive_datasets_catalogue.md             (MODMA 128/3-ch, EMBARC/PRED+CT, Warsaw SZ)
│   └── 03_sota_models_and_benchmarks.md                   (Microstates, Treatment Response, Wearable Screening)
│
├── 05_NEURODEVELOPMENTAL_DISORDERS/
│   ├── 01_neurobiology_and_biomarkers.md                  (ADHD TBR, N200/P300, ASD Local/Long Connectivity)
│   ├── 02_comprehensive_datasets_catalogue.md             (IEEE Dataport ADHD, Child Mind Inst HBN, KAU ASD)
│   └── 03_sota_models_and_benchmarks.md                   (Wavelet Energy, DWT, Spatial-Temporal GAT)
│
├── 06_SLEEP_AND_CONSCIOUSNESS_DISORDERS/
│   ├── 01_neurobiology_and_biomarkers.md                  (AASM Staging, Spindles, OSA, RBD, Coma PCI)
│   ├── 02_comprehensive_datasets_catalogue.md             (Sleep-EDFX, NSRR SHHS, MASS, PhysioNet 2023)
│   └── 03_sota_models_and_benchmarks.md                   (DeepSleepNet, TinySleepNet, U-Sleep, CPC Outcome)
│
├── 07_MASTER_DATASETS_DIRECTORY.md                        (Unified 50+ Dataset Global Registry Table)
├── 08_EEG_FOUNDATION_MODELS_AND_BENCHMARKS.md             (BIOT, LaBraM, BENDR, NeuroSync-MoE, SSL)
└── 09_STANDARDIZED_SIGNAL_PROCESSING_AND_VALIDATION.md    (BIDS, ASR, ICA, Leakage Prevention, Pipelines)
```

---

## 3. High-Level Synthesis Across Disorders

### 3.1 Cross-Cutting Electrophysiological Principles

| Neurophysiological Feature | Normal Function | Clinical Pathology & Associated Disorders |
| :--- | :--- | :--- |
| **P300 (P3b) Amplitude** | Context updating, attention allocation, working memory refresh | **Blunted:** AUD genetic predisposition, Schizophrenia, ADHD, Binge Drinking, Major Depression |
| **Spectral Slowing ($\uparrow \delta/\theta, \downarrow \alpha/\beta$)** | Deep non-REM sleep, infantile maturation | **Pathological Slowing:** Alzheimer's Disease, Opioid Use Disorder, Post-ictal state, Anoxic Coma, Encephalopathy |
| **Frontal Asymmetry (FAA)** | Approach (Left) vs. Withdrawal (Right) motivation balance | **Rightward Frontal Hyperactivity:** Major Depressive Disorder, Chronic dysphoria; **Leftward bias during craving:** AUD, Cocaine cue exposure |
| **Sensory Gating (P50)** | Inhibitory filtering of irrelevant background sensory noise | **Failure of Gating ($S_2/S_1 \uparrow$):** Schizophrenia, Acute Psychosis, Bipolar Mania |
| **Auditory Mismatch Negativity (MMN)** | Pre-attentive detection of acoustic deviations (NMDA mediated) | **Suppressed / Blunted:** Schizophrenia (NMDA hypofunction), Severe Cognitive Impairment, Chronic Opioids |
| **High-Frequency Synchronization (Gamma / ASSR)** | Fast-spiking parvalbumin ($PV^+$) GABAergic interneuron pacing | **Impaired 40-Hz ASSR:** Schizophrenia, Autism Spectrum Disorder, Cannabis Use Disorder |
| **Beta Bursting ($13 - 30\text{ Hz}$)** | Sensorimotor state maintenance ("status quo" motor signaling) | **Pathological Beta Bursts:** Parkinson's Disease (OFF levodopa); **Elevated resting beta:** Alcoholism hyperarousal (*GABRA2*) |
| **Sleep Spindles ($11 - 16\text{ Hz}$)** | Thalamocortical gating, memory consolidation in NREM stage 2 | **Reduced Spindle Density:** Schizophrenia, Alzheimer's Disease, Normal Aging; **Absent:** Severe Coma |

---

## 4. Key Takeaways for Machine Learning Researchers

1. **The Data Leakage Epidemic:**
   * Never perform random epoch-level train/test splits. Always enforce **Subject-Independent Cross-Validation** (Group K-Fold or Leave-One-Subject-Out). Reported accuracies dropping from $99\%$ down to $75\% - 85\%$ reflect true clinical generalization rather than subject memorization.
2. **Channel-Agnostic Foundation Models:**
   * The field is rapidly transitioning from specialized small models (EEGNet, ShallowFBCSPNet) to **Self-Supervised Pretrained Foundation Models (BIOT, LaBraM)** that ingest variable montages using 3D spatial positional embeddings and achieve superior transfer with minimal fine-tuning data.
3. **Multimodal and Explainable Integration:**
   * Future clinical translation mandates explainable AI (Grad-CAM, SHAP, Integrated Gradients) confirming that deep representations align with known neuropathological topographies (e.g., parietal P300 attenuation in addiction, temporal slowing in dementia, frontocentral TBR in ADHD).
