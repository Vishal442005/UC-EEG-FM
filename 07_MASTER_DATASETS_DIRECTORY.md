# Master EEG Datasets Directory: Universal Cross-Disorder Repository

This directory contains an exhaustive, cross-indexed registry of over 50 public, open-access, and benchmark electroencephalography (EEG) datasets spanning all major neurological, psychiatric, cognitive, sleep, and behavioral domains.

---

## 1. Master Cross-Index Table

| # | Dataset Name | Primary Disorder / Domain | Subjects ($N$) | Channels | Sampling Rate ($f_s$) | Paradigms / Tasks | Format | Access License / Terms | Direct URL / Repository Link |
| :- | :--- | :--- | :- | :- | :- | :--- | :--- | :--- | :--- |
| **1** | **UCI Alcoholism Database** | Alcohol Use Disorder | 122 | 64 | 256 Hz | Visual S1-S2 Object Matching ERP | Raw Text / CSV | Open (Public Domain) | [UCI ML Repo](https://archive.ics.uci.edu/dataset/121/eeg+database) |
| **2** | **OpenNeuro ds005907** | Alcoholism / Hazardous Drinking | 53 | 64 | 500 Hz | 3-Armed Bandit RL with Alcohol Cues | BIDS (BrainVision) | Open (CC0) | [OpenNeuro ds005907](https://openneuro.org/datasets/ds005907) |
| **3** | **OpenNeuro ds004515** | Alcohol Binge Drinking | 40+ | 32 / 64 | 500 Hz | Alcohol Imagery + Reinforcement Learning | BIDS (BrainVision) | Open (CC0) | [OpenNeuro ds004515](https://openneuro.org/datasets/ds004515) |
| **4** | **OpenNeuro ds008851** | Alcohol Use Disorder Inpatients | 45 | 64 | 500 Hz | Resting EO/EC + Visual Cue-Reactivity | BIDS | Open (CC0) | [OpenNeuro ds008851](https://openneuro.org/datasets/ds008851) |
| **5** | **Zenodo Opioid Cohort** | Opioid Use Disorder (Methadone) | 60 | 19 | 250 Hz | Continuous Resting State Eyes Closed | EDF / MAT | Open (CC-BY 4.0) | [Zenodo Opioid](https://zenodo.org/) |
| **6** | **Mendeley Internet Addiction**| Internet Gaming Disorder (IGD) | 48 | 32 | 500 Hz | Resting-State + Go/No-Go Inhibition | EEGLAB (.set) | Open (CC-BY) | [Mendeley Data](https://data.mendeley.com/) |
| **7** | **CHB-MIT Scalp Database** | Epilepsy (Pediatric Seizures) | 23 | 24–26 | 256 Hz | Continuous ICU Telemetry (198 Seizures) | EDF | Open (PhysioNet) | [PhysioNet CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) |
| **8** | **TUH Seizure Corpus (TUSZ)**| Epilepsy (Clinical Multi-class) | 500+ | 19–22 | 250–1000 Hz | Clinical Hospital Records (>3,000 Seizures) | EDF | Open Registration (NEDC) | [TUH NEDC Portal](https://isip.piconepress.com/projects/tuh_eeg/) |
| **9** | **TUH Abnormal EEG (TUAB)** | Neurological Abnormality / Pathology| 2,993 | 19–22 | 250–1000 Hz | Clinical Neurologist Normal vs Abnormal | EDF | Open Registration (NEDC) | [TUH NEDC TUAB](https://isip.piconepress.com/projects/tuh_eeg/) |
| **10**| **TUH Artifact Corpus (TUAR)**| EEG Artifacts (EMG, EOG, Chewing)| 300+ | 19–22 | 250–1000 Hz | Annotated Physiological Artifacts | EDF | Open Registration (NEDC) | [TUH NEDC TUAR](https://isip.piconepress.com/projects/tuh_eeg/) |
| **11**| **Bonn University Dataset** | Epilepsy / Seizure Benchmark | 15 | 1 (Intra/Scalp)| 173.6 Hz | Sets A–E (Healthy, Interictal, Ictal) | ASCII Text | Open Public Domain | [Bonn Repository](https://www.upf.edu/web/ntrg/eeg-data) |
| **12**| **Bern-Barcelona Database** | Focal vs. Non-Focal Epilepsy | 5 | 2 (iEEG) | 512 Hz | 7,500 Pairwise iEEG Recordings | ASCII Text | Open Public Domain | [Bern-Barcelona](https://www.upf.edu/web/ntrg/focal-non-focal-eeg-data) |
| **13**| **Siena Scalp Database** | Adult Epilepsy / Seizure Detect | 14 | 29 | 512 Hz | Long-term Video-EEG (47 Seizures) | EDF | Open (PhysioNet) | [PhysioNet Siena](https://physionet.org/content/siena-scalp-eeg/1.0.0/) |
| **14**| **Kaggle AES Challenge** | Seizure Prediction (Forecasting) | 7 | 16 (iEEG) | 400 Hz | Preictal (10m) vs Interictal Telemetry | MATLAB (.mat) | Open (Kaggle) | [Kaggle AES Challenge](https://www.kaggle.com/c/seizure-prediction) |
| **15**| **OpenNeuro ds004504** | Alzheimer’s (AD) & FTD Dementia | 88 | 19 | 500 Hz | Resting-State Eyes Closed (~13.5 min) | BIDS (EEGLAB) | Open (CC0) | [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504) |
| **16**| **Zenodo Dementia 7566271** | Alzheimer’s, FTD & Controls | 88 | 19 | 500 Hz | Cleaned 12s Epochs + MMSE Scores | EDF / MAT | Open (CC-BY 4.0) | [Zenodo 7566271](https://zenodo.org/records/7566271) |
| **17**| **OpenNeuro ds002778** | Parkinson’s Disease (ON/OFF L-DOPA)| 31 | 32 | 512 Hz | Resting-State Eyes Open / Closed | BIDS | Open (CC0) | [OpenNeuro ds002778](https://openneuro.org/datasets/ds002778) |
| **18**| **OpenNeuro ds003404** | Parkinson’s Disease High-Density | 42 | 64 | 512 Hz | High-Density Resting + Auditory Oddball | BIDS | Open (CC0) | [OpenNeuro ds003404](https://openneuro.org/datasets/ds003404) |
| **19**| **OpenNeuro ds004746** | Parkinson’s Cognitive Control | 50+ | 64 | 500 Hz | Flanker & Simon Conflict Paradigms | BIDS | Open (CC0) | [OpenNeuro ds004746](https://openneuro.org/datasets/ds004746) |
| **20**| **MODMA 128-Channel** | Major Depressive Disorder (MDD) | 53 | 128 | 250 Hz | Resting-State + Dot-Probe Emotion ERP | MAT / Raw | Open Academic | [MODMA Portal](http://modma.lzu.edu.cn/) |
| **21**| **MODMA 3-Channel Portable** | Wearable Depression Screening | 55 | 3 (`Fp1,Fp2,Fz`) | 250 Hz | Wearable Frontal Resting-State | CSV / MAT | Open Academic | [MODMA Portal](http://modma.lzu.edu.cn/) |
| **22**| **EMBARC / PRED+CT** | Antidepressant Treatment Response | 300+ | 72 | 500 Hz | Randomized Sertraline Clinical Trial | EDF / BIDS | Controlled (NIMH NDA) | [PRED+CT](http://predct.org/) |
| **23**| **Warsaw Schizophrenia** | Paranoid Schizophrenia | 28 | 19 | 250 Hz | 15 min Resting-State Eyes Closed | ASCII / EDF | Open (Kaggle) | [Kaggle Warsaw SZ](https://www.kaggle.com/datasets/broach/button-tone-sz) |
| **24**| **OpenNeuro ds003507** | Schizophrenia Auditory Oddball | 49 | 64 | 512 Hz | Two-Tone Oddball (P300 / N100) | BIDS (BDF) | Open (CC0) | [OpenNeuro ds003507](https://openneuro.org/datasets/ds003507) |
| **25**| **OpenNeuro ds003478** | Depression & Beck Inventory | 60+ | 64 | 500 Hz | Resting-State with BDI-II Scores | BIDS | Open (CC0) | [OpenNeuro ds003478](https://openneuro.org/datasets/ds003478) |
| **26**| **OpenNeuro ds004513** | Psychosis / Mismatch Negativity | 40+ | 64 | 500 Hz | Multi-feature MMN (NMDA Hypofunction) | BIDS | Open (CC0) | [OpenNeuro ds004513](https://openneuro.org/datasets/ds004513) |
| **27**| **IEEE Dataport ADHD** | ADHD (Inattentive / Combined) | 121 | 19 | 128 Hz | Visual Attention Character Counting | MATLAB (.mat) | Open (IEEE DataPort) | [IEEE Dataport ADHD](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children) |
| **28**| **Child Mind Inst. HBN** | ADHD, ASD, Anxiety, Learning | 2,500+ | 128 | 500 Hz | Resting + Movie Watching + Oddball | BIDS / EGI | Open Registration | [CMI HBN Portal](https://healthybrainnetwork.org/) |
| **29**| **KAU Autism Dataset** | Autism Spectrum Disorder (ASD) | 60 | 16 | 256 Hz | Relaxed + Social Facial Exposure | CSV / MAT | Open (Mendeley) | [Mendeley KAU ASD](https://data.mendeley.com/) |
| **30**| **OpenNeuro ds003826** | Pediatric Executive Function | 80+ | 64 | 500 Hz | Cognitive Control Battery | BIDS | Open (CC0) | [OpenNeuro ds003826](https://openneuro.org/datasets/ds003826) |
| **31**| **PhysioNet Sleep-EDFX** | Normal Sleep & Insomnia | 197 nights | 2 EEG + PSG | 100 Hz | Whole-night In-Home / Hospital Sleep | EDF | Open (PhysioNet) | [PhysioNet Sleep-EDFX](https://physionet.org/content/sleep-edfx/1.0.0/) |
| **32**| **NSRR SHHS Database** | Sleep Apnea & Cardiovascular | 5,800+ | 2 EEG + PSG | 125 Hz | Longitudinal In-Home Polysomnography | EDF | Open Registration | [NSRR SHHS](https://sleepdata.org/datasets/shhs) |
| **33**| **MASS Sleep Archive** | Sleep Spindles & K-Complexes | 200 | 20 EEG + PSG | 256 Hz | High-Density PSG + Double-Scoring | EDF | Open Research | [MASS Archive](http://massdb.herokuapp.com/en/) |
| **34**| **ISRUC-Sleep Database** | Sleep Apnea, Insomnia, RLS | 118 | 6 EEG + PSG | 200 Hz | Clinical Diagnostic Staging | EDF | Open Access | [ISRUC-Sleep](https://sleeptight.isr.uc.pt/) |
| **35**| **PhysioNet Challenge 2023**| Coma Outcome Prognostication | 1,000+ | 19 ICU EEG | 250–500 Hz | Post-Cardiac Arrest ICU Telemetry | WFDB / EDF | Open Credentialed | [PhysioNet 2023](https://physionet.org/content/challenge-2023/1.0.0/) |
| **36**| **PhysioNet Challenge 2018**| Sleep Arousals Detection | 1,983 | Multi-channel | 200 Hz | Detection of Non-Apneic Sleep Arousals | WFDB | Open (PhysioNet) | [PhysioNet 2018](https://physionet.org/content/challenge-2018/1.0.0/) |
| **37**| **BCI Competition IV 2a** | Motor Imagery (Left, Right, Foot, Tongue)| 9 | 22 | 250 Hz | 4-Class Motor Imagery BCI | GDF / MAT | Open Benchmark | [BCI Comp IV 2a](https://www.bbci.de/competition/iv/) |
| **38**| **BCI Competition IV 2b** | Motor Imagery (Left vs Right Hand)| 9 | 3 (`C3,Cz,C4`)| 250 Hz | Continuous BCI with Visual Feedback | GDF / MAT | Open Benchmark | [BCI Comp IV 2b](https://www.bbci.de/competition/iv/) |
| **39**| **High-Gamma Dataset** | Fine Motor Movement & BCI | 14 | 128 | 500 Hz | Real & Imagined Complex Movements | EDF / EEGLAB | Open (Schirrmeister)| [GNode High-Gamma](https://gin.g-node.org/) |
| **40**| **SEED (SJTU Emotion)** | Affective Computing / Emotion | 15 | 62 | 200 Hz | Positive, Neutral, Negative Movie Clips | MATLAB (.mat) | Open Academic Request | [SEED Dataset](https://bcmi.sjtu.edu.cn/home/seed/) |
| **41**| **SEED-IV** | 4-Emotion Affective Dynamics | 15 | 62 | 200 Hz | Happy, Sad, Fear, Neutral Clips | MATLAB (.mat) | Open Academic Request | [SEED-IV Dataset](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html) |
| **42**| **DEAP Dataset** | Valence-Arousal Emotion Modeling | 32 | 32 EEG + Peripheral| 512 Hz down 128| 40 One-Minute Music Video Clips | CSV / BDF / MAT | Open Academic Request | [DEAP Dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) |
| **43**| **DREAMER Dataset** | Affective State (Valence/Arousal)| 23 | 14 (Emotiv) | 128 Hz | Audio-Visual Film Clip Induction | MATLAB (.mat) | Open (Zenodo) | [DREAMER Dataset](https://zenodo.org/) |
| **44**| **STEW (Workload Dataset)**| Cognitive Mental Workload | 48 | 14 (Emotiv) | 128 Hz | Rest vs. Multi-Attribute Task Battery | CSV | Open (IEEE DataPort)| [IEEE DataPort STEW](https://ieee-dataport.org/) |
| **45**| **Cam-CAN Lifespan Corpus** | Healthy Aging & Lifespan Cognition| 700+ | 70 (EEG/MEG) | 1000 Hz | Rest + Sensorimotor + Cognitive Tasks | BIDS / FIF | Open Registration | [Cam-CAN Portal](https://www.cam-can.org/) |
| **46**| **OpenNeuro ds003645** | Stroke / Motor Rehabilitation | 35 | 64 | 500 Hz | Upper Limb Movement Attempt in Stroke | BIDS | Open (CC0) | [OpenNeuro ds003645](https://openneuro.org/datasets/ds003645) |
| **47**| **OpenNeuro ds002680** | Auditory MMN & Sensory Gating | 50 | 64 | 512 Hz | Multi-feature Sensory Battery | BIDS | Open (CC0) | [OpenNeuro ds002680](https://openneuro.org/datasets/ds002680) |
| **48**| **OpenNeuro ds002718** | Visual Evoked Potentials (VEP) | 60 | 64 | 1000 Hz | Checkerboard Pattern Reversal VEP | BIDS | Open (CC0) | [OpenNeuro ds002718](https://openneuro.org/datasets/ds002718) |
| **49**| **OpenNeuro ds003061** | Pain & Somatosensory Processing | 40 | 64 | 1000 Hz | Laser-Evoked Potentials (LEP) in Pain | BIDS | Open (CC0) | [OpenNeuro ds003061](https://openneuro.org/datasets/ds003061) |
| **50**| **OpenNeuro ds004184** | Substance Cue-Reactivity (Cannabis)| 38 | 64 | 500 Hz | Drug Paraphernalia Visual Exposure | BIDS | Open (CC0) | [OpenNeuro ds004184](https://openneuro.org/datasets/ds004184) |

---

## 2. Standardized Repositories Overview

```
                      GLOBAL EEG DATA REPOSITORIES
                                   │
      ┌────────────────────────────┼────────────────────────────┐
      ▼                            ▼                            ▼
 [ OpenNeuro ]                [ PhysioNet ]                 [ Zenodo ]
  • BIDS Mandatory             • Clinical Time-Series        • Catch-All Research
  • Cognitive / Psychiatric    • Seizures & Sleep ICU        • Boutique ML Datasets
  • Versioned via DataLad      • High-Duration Telemetry     • Direct DOIs
```

### 2.1 Automated BIDS Dataset Download Recipe (DataLad / AWS S3)
To download any OpenNeuro dataset directly via terminal:
```bash
# Using DataLad
datalad install https://github.com/OpenNeuroDatasets/ds004504.git
cd ds004504
datalad get sub-001/

# Using AWS CLI (No authentication required for open buckets)
aws s3 sync --no-sign-request s3://openneuro.org/ds004504 ./ds004504_download
```
