# Comprehensive EEG Datasets Catalogue: Psychiatric and Mood Disorders

This catalogue provides an exhaustive registry of electroencephalography (EEG) datasets dedicated to Major Depressive Disorder (MDD), Schizophrenia (SZ), Bipolar Disorder (BD), and Anxiety/PTSD.

---

## 1. MODMA: Multi-Modal Open Dataset for Mental-Disorder Analysis (Lanzhou University)

* **Repository:** [MODMA Dataset Portal (Lanzhou University)](http://modma.lzu.edu.cn/data/index/) / Mirrored on Science Data Bank & PhysioNet
* **Institution:** Ubiquitous Awareness and Intelligent Solutions Lab, Lanzhou University, China.
* **Clinical Groups:**
  * Patients diagnosed with Major Depressive Disorder (MDD) based on DSM-IV / MINI criteria alongside age- and education-matched healthy controls (HC).
  * Evaluated with Hamilton Depression Rating Scale (HAMD-17) and Self-Rating Depression Scale (SDS).
* **Two Sub-Cohorts & Recording Configurations:**
  1. **High-Density 128-Channel Scalp EEG:**
     * $N = 53$ participants (24 clinical MDD outpatients, 29 healthy controls).
     * **System:** EGI HydroCel GSN 128-channel sensor net ($f_s = 250\text{ Hz}$).
     * **Paradigms:** Continuous resting-state (Eyes Closed / Eyes Open) + Auditory event-related potential (dot-probe emotion task).
  2. **Wearable Portable 3-Channel Frontal EEG:**
     * $N = 55$ participants (26 MDD patients, 29 healthy controls).
     * **Electrodes:** Frontal channels `Fp1, Fp2, Fz` ($f_s = 250\text{ Hz}$).
     * Designed specifically for translational, wearable point-of-care depression screening.
* **Access Status:** Open for non-commercial research (data access request form on portal).

---

## 2. PRED+CT / EMBARC Multi-Site Clinical Depression EEG Database

* **Repository:** [PRED+CT Data Archive](http://predct.org/) / NIMH Data Archive (NDA Study 2199)
* **Full Title:** Establishing Moderators and Biosignatures of Antidepressant Response in Clinical Care (EMBARC).
* **Study Design:** Landmark multi-site, randomized, double-blind, placebo-controlled trial evaluating Sertraline (SSRI) vs. Placebo in unmedicated patients with Major Depressive Disorder.
* **Subjects:** Over 300 MDD patients and healthy controls recorded across 4 major academic medical centers:
  * Columbia University, UT Southwestern Medical Center, Massachusetts General Hospital, and University of Michigan.
* **Recording Specifications:**
  * **System:** High-density 72-channel or 64-channel EEG caps ($f_s = 500\text{ Hz}$).
  * **Protocols Recorded at Baseline (Week 0) and Early Treatment (Week 1):**
    1. Resting-state (Eyes Open / Eyes Closed).
    2. Flanker Task (Error-Related Negativity - ERN).
    3. Reward Processing / Probabilistic Reward Task (Reward Positivity - RewP).
* **Significance:** The international gold standard for training machine learning algorithms to predict **differential antidepressant treatment response**.
* **Access Status:** Controlled Access via NIMH Data Archive (NDA) / Open preview subsets via PRED+CT.

---

## 3. Warsaw Institute of Psychiatry and Neurology: Schizophrenia EEG Dataset

* **Repository:** [Kaggle - Schizophrenia EEG Dataset](https://www.kaggle.com/datasets/broach/button-tone-sz) / Zenodo Mirror
* **Primary Reference:** Olejarczyk & Jernajczyk (2017), *Frontiers in Human Neuroscience*.
* **Subjects:**
  * $N = 28$ participants:
    * 14 patients diagnosed with paranoid schizophrenia (7 males, 7 females, mean age 27.9).
    * 14 matched healthy control subjects (7 males, 7 females, mean age 26.8).
* **Recording Specifications:**
  * **Montage:** 19 scalp channels standard International 10-20 layout:
    `Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2` (referenced to linked mastoids).
  * **Sampling Frequency:** $f_s = 250\text{ Hz}$.
  * **Condition:** 15 minutes of continuous resting-state EEG with eyes closed.
* **Research Value:** Highly utilized worldwide for evaluating functional connectivity (directed transfer function, mutual information, wPLI) and microstate dynamics in schizophrenia.
* **Access Status:** Public domain / Open.

---

## 4. OpenNeuro Accession `ds003507`: Auditory Oddball Task in Schizophrenia

* **Repository:** [OpenNeuro Accession ds003507](https://openneuro.org/datasets/ds003507)
* **Title:** Auditory oddball event-related potentials in schizophrenia patients and matched controls.
* **Subjects:**
  * $N = 49$ participants (24 schizophrenia patients, 25 healthy controls).
* **Recording Specifications:**
  * **Montage:** 64-channel ActiveTwo Biosemi system ($f_s = 1024\text{ Hz}$ downsampled to $512\text{ Hz}$).
  * **Experimental Paradigm:** Two-tone auditory oddball task presenting standard tones ($1000\text{ Hz}$, $80\%$ probability) and infrequent target tones ($1500\text{ Hz}$, $20\%$ probability requiring button press).
* **Biomarkers Captured:** P50 sensory gating, N100, N200, and P300 (P3b) amplitude and latency attenuation.
* **Access Status:** Open (CC0).

---

## 5. OpenNeuro Accession `ds003478`: Resting-State EEG with Depressive Phenotyping

* **Repository:** [OpenNeuro Accession ds003478](https://openneuro.org/datasets/ds003478)
* **Title:** Resting-state EEG recordings with Beck Depression Inventory (BDI-II) phenotyping.
* **Cohort Profile:** Healthy controls and clinical individuals evaluated across depressive symptoms, state-trait anxiety (STAI), and anhedonia scores.
* **Montage:** 64-channel BrainProducts EEG ($f_s = 500\text{ Hz}$) with standardized 10-20 sensor coordinates.
* **Access Status:** Open (CC0).

---

## 6. OpenNeuro Accession `ds004513`: Auditory MMN in Schizophrenia & Psychosis

* **Repository:** [OpenNeuro Accession ds004513](https://openneuro.org/datasets/ds004513)
* **Title:** Mismatch Negativity (MMN) and gamma-band oscillations in psychotic disorders.
* **Tasks:** Multi-feature auditory mismatch negativity paradigm evaluating pitch, duration, and frequency deviants to quantify cortical NMDA receptor hypofunction.
* **Access Status:** Open (CC0).

---

## Summary Matrix of Psychiatric EEG Datasets

| Dataset | Target Disorder | Subjects ($N$) | Channels | Sampling Rate | Format | Key Paradigm | Access Link |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MODMA 128-ch** | Major Depression (MDD) | 53 | 128 | 250 Hz | MAT / Raw | Resting-State EO/EC + Dot-Probe | [MODMA Portal](http://modma.lzu.edu.cn/) |
| **MODMA 3-ch** | MDD (Wearable) | 55 | 3 (`Fp1,Fp2,Fz`) | 250 Hz | CSV / MAT | Wearable Frontal Resting State | [MODMA Portal](http://modma.lzu.edu.cn/) |
| **EMBARC / PRED+CT** | MDD Treatment Response | 300+ | 72 | 500 Hz | EDF / BIDS | Sertraline Trial (Resting + Flanker) | [PRED+CT](http://predct.org/) |
| **Warsaw SZ** | Schizophrenia | 28 | 19 | 250 Hz | ASCII / EDF | 15 min Resting Eyes Closed | [Kaggle SZ](https://www.kaggle.com/datasets/broach/button-tone-sz) |
| **OpenNeuro ds003507** | Schizophrenia | 49 | 64 | 512 Hz | BIDS (BDF) | Auditory Oddball Task (P300/N100) | [OpenNeuro ds003507](https://openneuro.org/datasets/ds003507) |
| **OpenNeuro ds003478** | Depression / BDI-II | 60+ | 64 | 500 Hz | BIDS | Resting-State with BDI-II scores | [OpenNeuro ds003478](https://openneuro.org/datasets/ds003478) |
| **OpenNeuro ds004513** | Psychosis / NMDA | 40+ | 64 | 500 Hz | BIDS | Multi-feature Mismatch Negativity | [OpenNeuro ds004513](https://openneuro.org/datasets/ds004513) |
