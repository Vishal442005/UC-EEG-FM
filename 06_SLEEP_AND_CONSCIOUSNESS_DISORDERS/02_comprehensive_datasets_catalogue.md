# Comprehensive EEG Datasets Catalogue: Sleep and Consciousness Disorders

This catalogue details the world's primary open benchmarks and clinical registries for sleep staging, sleep apnea, insomnia, coma prognostication, and disorders of consciousness.

---

## 1. PhysioNet: Sleep-EDF & Sleep-EDF Expanded (Sleep-EDFX)

* **Repository:** [PhysioNet Sleep-EDF Database Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)
* **Primary Reference:** Kemp et al. (2000), *IEEE Transactions on Biomedical Engineering*.
* **Scale & Sub-Cohorts:**
  * **Sleep Cassette Study (SC):** 153 whole-night polysomnograms from 78 healthy subjects (aged 25–101 years, recorded in their homes over two consecutive nights).
  * **Sleep Telemetry Study (ST):** 44 whole-night recordings from 22 subjects with mild difficulty falling asleep, recorded in hospital environments.
* **Recording Specifications:**
  * **EEG Channels:** Bipolar montage: `Fpz-Cz` and `Pz-Oz` ($f_s = 100\text{ Hz}$).
  * **Auxiliary Channels:** Horizontal EOG ($100\text{ Hz}$), Submental Chin EMG ($100\text{ Hz}$), Oronasal respiration ($1\text{ Hz}$), Rectal temperature.
  * **Format:** European Data Format (`.edf`).
* **Annotations:**
  * Contiguous 30-second epoch hypnograms manually scored by professional sleep technologists according to the Rechtschaffen & Kales (R&K) rules (mapped directly to AASM: `W, N1, N2, N3, N4, REM, MOVEMENT`).
* **Access Status:** Public Domain Open Access.

---

## 2. National Sleep Research Resource (NSRR): Sleep Heart Health Study (SHHS)

* **Repository:** [NSRR Sleep Heart Health Study](https://sleepdata.org/datasets/shhs)
* **Scale & Significance:**
  * Landmark multi-center cohort sponsored by NHLBI/NIH.
  * Over **$5,800+$ participants** ($>10,000+$ full-night in-home polysomnograms recorded at Baseline and Follow-up).
  * Contains exhaustive cardiovascular, respiratory, and cognitive phenotyping: Apnea-Hypopnea Index (AHI), Oxygen Desaturation Index (ODI), daytime sleepiness (Epworth Sleepiness Scale - ESS).
* **Recording Specifications:**
  * **EEG Channels:** `C3-A2` and `C4-A1` ($f_s = 125\text{ Hz}$).
  * **Auxiliary:** Bilateral EOG, Chin EMG, Chest/Abdominal Inductance Plethysmography, Finger Pulse Oximetry ($\text{SpO}_2$), Airflow, ECG.
* **Format:** Compumedics EDF files with XML hypnograms and event annotations.
* **Access Status:** Open Access with simple academic registration on sleepdata.org.

---

## 3. Montreal Archive of Sleep Studies (MASS)

* **Repository:** [MASS Sleep Archive](http://massdb.herokuapp.com/en/)
* **Institution:** Université de Montréal & McGill University, Canada.
* **Scale:** 200 high-density whole-night laboratory PSG recordings from healthy participants aged 18–76 years.
* **Subsets:**
  * Divided into 5 distinct cohorts (`SS1` to `SS5`) tailored for specific algorithmic targets.
  * **SS2:** Contains expert micro-architectural scoring: thousands of manually validated **sleep spindles** and **K-complexes** with onset/offset sample indices.
  * **SS3:** 62 recordings with independent double-scoring by multiple accredited somnologists for inter-rater reliability benchmarks.
* **Montage:** 20 scalp EEG channels (Standard 10-20 layout), 4 EOG, 3 EMG, 1 ECG ($f_s = 256\text{ Hz}$).
* **Access Status:** Free open academic research agreement.

---

## 4. ISRUC-Sleep Dataset

* **Repository:** [ISRUC-Sleep Database](https://sleeptight.isr.uc.pt/)
* **Institution:** Sleep Medicine Centre of the Hospital of Coimbra University, Portugal.
* **Cohorts ($N = 118$ total):**
  * **Group 1:** 100 adult subjects diagnosed with sleep disorders (Obstructive Sleep Apnea, Insomnia, Restless Legs Syndrome), each with full-night PSG scored independently by two sleep experts.
  * **Group 2:** 8 subjects recorded for two consecutive nights (evaluating first-night effect).
  * **Group 3:** 10 healthy control subjects.
* **Montage:** 6 EEG channels (`F3-A2, C3-A2, O1-A2, F4-A1, C4-A1, O2-A1`), 2 EOG, 3 EMG, 1 ECG ($f_s = 200\text{ Hz}$).
* **Access Status:** Completely Open.

---

## 5. George B. Moody PhysioNet Challenge 2023: Predicting Neurological Outcome in Comatose Patients

* **Repository:** [PhysioNet Challenge 2023 Portal](https://physionet.org/content/challenge-2023/1.0.0/)
* **Clinical Target:** Prognosticating awakening and neurological recovery in comatose patients resuscitated from cardiac arrest.
* **Dataset Scale:**
  * Continuous multi-channel clinical ICU EEG telemetry recorded from over **1,000 comatose patients** admitted to Intensive Care Units across 7 academic medical centers in the US and Europe.
* **Montage & Format:** Standard 18–19 channel clinical 10-20 montage recorded continuously for up to 72 hours post-cardiac arrest.
* **Target Clinical Outcome:**
  * 72-hour and 6-month **Cerebral Performance Category (CPC)** scores:
    * *Favorable Outcome (CPC 1–2):* Good cerebral performance or moderate disability (conscious, independent).
    * *Unfavorable Outcome (CPC 3–5):* Severe cerebral disability, unresponsive wakefulness syndrome (UWS), or brain death.
* **Access Status:** Open Access (PhysioNet Credentialed).

---

## Summary Matrix of Sleep & Consciousness Datasets

| Dataset | Clinical Domain | Subjects ($N$) | Channels | Sampling Rate | Format | Annotations | Access Link |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sleep-EDFX** | Normal Sleep & Insomnia | 197 nights | 2 EEG + EOG/EMG | 100 Hz | EDF | 30s AASM Hypnograms | [PhysioNet Sleep-EDFX](https://physionet.org/content/sleep-edfx/1.0.0/) |
| **SHHS (NSRR)** | Sleep Apnea & Cardiac | 5,800+ | 2 EEG + Full PSG | 125 Hz | EDF | AASM Staging + AHI Apnea Events | [NSRR SHHS](https://sleepdata.org/datasets/shhs) |
| **MASS** | Spindles & K-Complexes | 200 | 20 EEG + PSG | 256 Hz | EDF | Micro-architecture + Staging | [MASS Archive](http://massdb.herokuapp.com/en/) |
| **ISRUC-Sleep** | Sleep Disorders & Apnea | 118 | 6 EEG + PSG | 200 Hz | EDF | Double-Expert AASM Staging | [ISRUC-Sleep](https://sleeptight.isr.uc.pt/) |
| **PhysioNet 2023**| Coma Prognostication | 1,000+ | 19 ICU EEG | 250–500 Hz | WFDB/EDF | 72h / 6-Month CPC Scores | [PhysioNet 2023](https://physionet.org/content/challenge-2023/1.0.0/) |
