# Comprehensive EEG Datasets Catalogue: Neurodegenerative Disorders & Dementia

This catalogue lists publicly available open-access datasets dedicated to dementia, Alzheimer’s Disease, Mild Cognitive Impairment, Frontotemporal Dementia, and Parkinson’s Disease.

---

## 1. OpenNeuro Accession `ds004504`: Resting-State EEG in AD, FTD, and Healthy Controls

* **Repository:** [OpenNeuro Accession ds004504](https://openneuro.org/datasets/ds004504)
* **Title:** A dataset of resting-state EEG recordings in patients with Alzheimer's disease, Frontotemporal dementia, and healthy controls.
* **Standards Compliance:** Fully BIDS (Brain Imaging Data Structure) compliant.
* **Subjects & Clinical Cohort:**
  * Total $N = 88$ participants:
    * **Alzheimer’s Disease (AD):** $N = 36$ patients (mean MMSE $= 17.75 \pm 4.5$, meeting NINCDS-ADRDA criteria).
    * **Frontotemporal Dementia (FTD):** $N = 23$ patients (mean MMSE $= 22.17 \pm 8.2$, meeting Lund-Manchester criteria).
    * **Healthy Controls (CN):** $N = 29$ cognitively normal age-matched older adults (mean MMSE $= 30.0$).
* **Recording Specifications:**
  * **System:** Nihon Kohden EEG-2100 clinical electroencephalograph.
  * **Montage:** 19 scalp channels positioned according to the International 10-20 system:
    `Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2` (with earlobe reference `A1+A2`).
  * **Sampling Rate:** $f_s = 500\text{ Hz}$.
  * **Condition:** Continuous resting-state under relaxed eyes-closed conditions (average recording duration ~13.5 minutes per subject).
* **Data Organization:**
  * Formatted as standard BIDS: EEG files in `.set` / `.fdt` (EEGLAB format) with associated `participants.tsv`, `scans.tsv`, and channel coordinate JSON files.
* **Access Status:** Completely Open (CC0 Public Domain Dedication).

---

## 2. Zenodo Dementia EEG Dataset (Miltiadous et al. 2023)

* **Repository:** [Zenodo Record 7566271](https://zenodo.org/records/7566271)
* **Primary Reference:** Miltiadous et al. (2023), *Scientific Data*, "A dataset of scalp EEG recordings of Alzheimer's disease, Frontotemporal dementia and Healthy subjects".
* **Data Characteristics:**
  * Contains both raw continuous recordings and cleaned, artifact-free 12-second epochs with ocular ICA components removed.
  * Includes complete clinical profiles: age, sex, formal MMSE cognitive scores, disease duration, and medication profiles.
* **Format:** European Data Format (`.edf`) and MATLAB matrices (`.mat`).
* **Access Status:** Open Access (Creative Commons Attribution 4.0 International).

---

## 3. OpenNeuro Accession `ds002778`: Resting-State EEG in Parkinson's Disease (ON vs. OFF Levodopa)

* **Repository:** [OpenNeuro Accession ds002778](https://openneuro.org/datasets/ds002778)
* **Title:** Resting state EEG in Parkinson's disease with and without levodopa.
* **Subjects:**
  * $N = 15$ Parkinson’s Disease patients recorded in two distinct pharmacological sessions:
    1. **OFF-Medication:** After $\ge 12$-hour withdrawal of all dopaminergic medication.
    2. **ON-Medication:** Re-tested after administration of standard therapeutic dose of L-DOPA.
  * $N = 16$ age-matched healthy control subjects.
* **Recording Specifications:**
  * **Montage:** 32-channel BrainProducts actiCAP (Standard 10-20).
  * **Sampling Rate:** $f_s = 512\text{ Hz}$.
  * **Protocols:** 3-minute Eyes-Closed resting state and 3-minute Eyes-Open resting state per session.
* **Clinical Variables:** Unified Parkinson's Disease Rating Scale (UPDRS-III motor score) recorded in both ON and OFF conditions.
* **Access Status:** Open (CC0).

---

## 4. OpenNeuro Accession `ds003404`: High-Density EEG in Parkinson's Disease

* **Repository:** [OpenNeuro Accession ds003404](https://openneuro.org/datasets/ds003404)
* **Title:** High-density EEG resting-state recordings in Parkinson's disease and healthy elderly.
* **Montage:** 64-channel BioSemi ActiveTwo system ($f_s = 2048\text{ Hz}$ downsampled to $512\text{ Hz}$).
* **Tasks:** Resting state (Eyes Open, Eyes Closed) + Auditory oddball paradigm.
* **Utility:** Ideal for cortical source localization (sLORETA, eLORETA) of subthalamic and sensorimotor beta bursting.

---

## 5. OpenNeuro Accession `ds004746`: UNM Parkinson's Cognitive Control EEG

* **Repository:** [OpenNeuro Accession ds004746](https://openneuro.org/datasets/ds004746)
* **Institution:** University of New Mexico (UNM).
* **Focus:** Electrophysiological assessment of executive dysfunction and cognitive slowing in PD patients performing Flanker and Simon tasks.
* **Montage:** 64-channel BrainProducts system ($f_s = 500\text{ Hz}$).
* **Access Status:** Open (CC0).

---

## Summary Matrix of Neurodegenerative Datasets

| Dataset | Primary Diagnosis | Subjects ($N$) | Channels | Sampling Rate | Format | Key Paradigm | Access Link |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenNeuro ds004504** | AD / FTD / Controls | 88 | 19 | 500 Hz | BIDS / EEGLAB | 13 min Resting Eyes-Closed | [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504) |
| **Zenodo Dementia** | AD / FTD / Controls | 88 | 19 | 500 Hz | EDF / MAT | Preprocessed 12s Epochs | [Zenodo 7566271](https://zenodo.org/records/7566271) |
| **OpenNeuro ds002778** | PD (ON/OFF L-DOPA) | 31 | 32 | 512 Hz | BIDS | Resting Eyes Open / Closed | [OpenNeuro ds002778](https://openneuro.org/datasets/ds002778) |
| **OpenNeuro ds003404** | PD vs. Controls | 42 | 64 | 512 Hz | BIDS | High-Density Resting + Oddball | [OpenNeuro ds003404](https://openneuro.org/datasets/ds003404) |
| **OpenNeuro ds004746** | PD Cognitive Control | 50+ | 64 | 500 Hz | BIDS | Executive Tasks (Flanker) | [OpenNeuro ds004746](https://openneuro.org/datasets/ds004746) |
