# Comprehensive EEG Datasets Catalogue: Epilepsy and Seizure Disorders

This catalogue provides detailed technical specifications and access instructions for all major open and clinical benchmark electroencephalography (EEG) datasets dedicated to epilepsy detection, prediction, and localization.

---

## 1. CHB-MIT Scalp EEG Database (PhysioNet)

* **Repository:** [PhysioNet CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/)
* **Institution:** Children’s Hospital Boston (CHB) & Massachusetts Institute of Technology (MIT).
* **Subjects & Demographics:**
  * $N = 23$ pediatric patients ($N = 24$ cases, as patient `chb01` and `chb21` are the same individual recorded at different intervals).
  * 5 males (ages 3–22), 17 females (ages 1.5–19).
* **Dataset Scale:**
  * **Total Recording Hours:** Over 982 hours of continuous multi-channel scalp EEG.
  * **Seizure Events:** Exactly 198 clinically and electrographically confirmed seizure episodes annotated with microsecond start and end timestamps.
* **Recording Specifications:**
  * **Montage:** 23 to 26 scalp electrodes arranged in the International 10-20 System (bipolar montage: e.g., `FP1-F7`, `F7-T7`, `T7-P7`, `P7-O1`, `FP1-F3`, `F3-C3`, `C3-P3`, `P3-O1`, etc.).
  * **Sampling Rate:** $f_s = 256\text{ Hz}$ at 16-bit analog-to-digital resolution.
  * **Format:** European Data Format (`.edf`) accompanied by ASCII `.txt` annotation summaries.
* **Access Status:** Completely Open (PhysioNet Open Access, ODC Public Domain Dedication and License).
* **Download Command:**
```bash
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/
```

---

## 2. Temple University Hospital EEG Seizure Corpus (TUSZ)

* **Repository:** [TUH EEG Corpus - Neural Engineering Data Consortium (NEDC)](https://isip.piconepress.com/projects/tuh_eeg/)
* **Institution:** Temple University Hospital, Philadelphia, USA.
* **Significance:** The world's largest open clinical corpus of archived electroencephalograms, representing real-world hospital telemetry, emergency department, and Intensive Care Unit (ICU) recordings.
* **Subsets & Scale:**
  * **TUSZ (v2.0.0+):** Over 5,000+ EEG sessions from 500+ patients containing >3,000 verified seizures.
  * **TUAB (TUH Abnormal):** 2,993 sessions categorized as Normal vs. Pathological/Abnormal by certified neurologists.
  * **TUEP (TUH Epilepsy):** Subset tracking subjects with confirmed chronic epilepsy vs. non-epileptic control cohorts.
  * **TUAR (TUH Artifact):** Annotated with specific muscle (EMG), ocular (EOG), chewing, electrode pop, and movement artifacts.
* **Electrode Montages:** Standard 10-20 clinical TCP (Temporal Central Parasagittal) montage:
  * 19–22 channels (`FP1-F7`, `F7-T3`, `T3-T5`, `T5-O1`, `FP2-F8`, `F8-T4`, `T4-T6`, `T6-O2`, `FP1-F3`, `F3-C3`, `C3-P3`, `P3-O1`, etc.).
* **Sampling Rate:** $250\text{ Hz}$ to $1000\text{ Hz}$ (standardized across clinical sub-studies).
* **Seizure Type Annotations:** Granular multi-class labels for seizure morphologies:
  * Focal Non-Specific Seizures (`FNSZ`)
  * Generalized Non-Specific Seizures (`GNSZ`)
  * Complex Partial Seizures (`CPSZ`)
  * Tonic-Clonic Seizures (`TCSZ`)
  * Absence Seizures (`ABSZ`)
  * Myoclonic Seizures (`MYSZ`)
* **Access Status:** Free open registration required (academic/research agreement via NEDC portal).

---

## 3. Bonn University EEG Database (Andrzejak Benchmark)

* **Repository:** [University of Bonn Department of Epileptology](https://www.upf.edu/web/ntrg/eeg-data) / Widely mirrored on Zenodo & Kaggle.
* **Primary Reference:** Andrzejak et al. (2001), *Physical Review E*.
* **Structure (Five Sets: A, B, C, D, E):**
  * **Set A (Z):** Surface EEG from 5 healthy volunteers, relaxed, eyes open.
  * **Set B (O):** Surface EEG from 5 healthy volunteers, relaxed, eyes closed.
  * **Set C (N):** Intracranial EEG (iEEG) from hippocampal formation of epilepsy patients during *interictal* periods (opposite to epileptogenic zone).
  * **Set D (F):** Intracranial EEG (iEEG) from within the *epileptogenic zone* during *interictal* periods.
  * **Set E (S):** Intracranial EEG recorded during active *ictal* (seizure) activity.
* **Specifications:**
  * Each set contains 100 single-channel EEG text segments of 23.6 seconds duration ($N = 4097$ samples per file).
  * Sampling rate: $f_s = 173.61\text{ Hz}$, 12-bit ADC.
* **Standard Experimental Benchmarks:**
  * Binary: Set A vs. Set E (Healthy vs. Seizure).
  * 3-Class: Sets A/B (Healthy) vs. Sets C/D (Interictal) vs. Set E (Ictal).
  * 5-Class: Complete multi-class classification across all sets.
* **Access Status:** Open public domain.

---

## 4. Bern-Barcelona Intracranial EEG Database

* **Repository:** [Bern-Barcelona Neurophysics EEG Database](https://www.upf.edu/web/ntrg/focal-non-focal-eeg-data)
* **Primary Reference:** Andrzejak et al. (2012), *EPL (Europhysics Letters)*.
* **Design & Purpose:** Designed specifically for **focal vs. non-focal** intracranial signal discrimination.
* **Data Scale:**
  * 3,750 pairs of simultaneous bivariate iEEG signals (7,500 single-channel signals total).
  * Recorded from 5 drug-resistant temporal lobe epilepsy patients undergoing presurgical evaluation.
* **Specifications:**
  * Signal length: 20 seconds ($N = 10,240$ samples per signal).
  * Sampling rate: $f_s = 512\text{ Hz}$ or $1024\text{ Hz}$ downsampled.
  * Label: Focal (recorded from channels where the first ictal changes were detected) vs. Non-focal (recorded from uninvolved brain regions).
* **Access Status:** Completely Open.

---

## 5. Siena Scalp EEG Database (PhysioNet)

* **Repository:** [PhysioNet Siena Scalp EEG Database](https://physionet.org/content/siena-scalp-eeg/1.0.0/)
* **Institution:** Unit of Neurology and Neurophysiology, University of Siena, Italy.
* **Subjects & Seizures:**
  * $N = 14$ patients (9 males aged 25–71, 5 females aged 20–58).
  * 47 total seizure events recorded across 128 hours of continuous monitoring.
* **Specifications:**
  * **Montage:** High-density 29 scalp channels (10-20 system + additional subtemporal channels: `F9, F10, T9, T10, P9, P10`).
  * **Sampling Rate:** $f_s = 512\text{ Hz}$.
  * **Format:** European Data Format (`.edf`).
* **Access Status:** Open access (ODC-BY).

---

## 6. American Epilepsy Society Seizure Prediction Challenge (Kaggle / NIH)

* **Repository:** [Kaggle Seizure Prediction Challenge](https://www.kaggle.com/c/seizure-prediction)
* **Partners:** American Epilepsy Society, NIH, University of Melbourne, NeuroVista.
* **Subjects:**
  * Continuous multi-month intracranial EEG recordings from canines with naturally occurring epilepsy ($N = 5$) and human patients ($N = 2$) equipped with NeuroVista ambulatory telemetry implants.
* **Task Formulation (Prediction):**
  * Binary classification: **Preictal** (10-minute segments occurring within the 65–5 minute window before clinical seizure onset) vs. **Interictal** (10-minute segments occurring $\ge 4$ hours away from any seizure).
* **Sampling Rate:** $f_s = 400\text{ Hz}$ to $5000\text{ Hz}$ across 16 continuous intracranial channels.
* **Significance:** Standard international benchmark for clinical seizure forecasting.

---

## Summary Matrix of Epilepsy Datasets

| Dataset | Modality | Patients ($N$) | Seizures ($N$) | Duration (Hours) | Sampling Rate | Key Clinical Purpose | Access Link |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CHB-MIT** | Scalp EEG | 23 | 198 | 982 hrs | 256 Hz | Pediatric Seizure Detection | [PhysioNet CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) |
| **TUH TUSZ** | Scalp EEG | 500+ | >3,000 | >4,000 hrs | 250–1000 Hz | Multi-class Seizure Classification | [TUH NEDC Portal](https://isip.piconepress.com/projects/tuh_eeg/) |
| **Bonn University** | Scalp + iEEG | 15 | 100 ictal | ~1.5 hrs | 173.6 Hz | Classic Benchmark / Proof of Concept | [Bonn Repository](https://www.upf.edu/web/ntrg/eeg-data) |
| **Bern-Barcelona** | Intracranial (iEEG) | 5 | Interictal | ~41 hrs | 512 Hz | Focal vs. Non-focal Localization | [Bern-Barcelona](https://www.upf.edu/web/ntrg/focal-non-focal-eeg-data) |
| **Siena Scalp** | Scalp EEG | 14 | 47 | 128 hrs | 512 Hz | Adult Scalp Seizure Detection | [PhysioNet Siena](https://physionet.org/content/siena-scalp-eeg/1.0.0/) |
| **AES Seizure Pred.** | Intracranial Telemetry | 7 | >100 | >1,000 hrs | 400 Hz | Seizure Prediction / Forecasting | [Kaggle AES Challenge](https://www.kaggle.com/c/seizure-prediction) |
