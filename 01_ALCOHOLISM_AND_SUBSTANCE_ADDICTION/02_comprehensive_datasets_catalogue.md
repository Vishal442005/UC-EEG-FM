# Comprehensive EEG Datasets Catalogue: Alcoholism and Substance Addiction

This catalogue provides an exhaustive registry of publicly available electroencephalography (EEG) datasets dedicated to alcoholism, substance use disorders (SUD), and behavioral addictions.

---

## 1. UCI Machine Learning Repository: EEG Alcoholism Database (Begleiter Cohort)

* **Repository:** [UCI Machine Learning Repository - EEG Database](https://archive.ics.uci.edu/dataset/121/eeg+database) / Kaggle Mirror ([Kaggle EEG-Alcoholism](https://www.kaggle.com/datasets/nnair25/Alcoholics))
* **Primary Investigator:** Dr. Henri Begleiter, Neurodynamics Laboratory, State University of New York (SUNY) Health Science Center at Brooklyn.
* **Subjects:**
  * Total $N = 122$ subjects (77 diagnosed alcoholic subjects fulfilling DSM criteria, 45 age- and sex-matched healthy control subjects).
  * Cohort comprises male subjects aged 20–40 years.
* **Recording Specifications:**
  * **Electrodes:** 64 scalp electrodes arranged according to the Standard 10-20 / Extended 10-20 system.
  * **Sampling Frequency:** $f_s = 256\text{ Hz}$.
  * **Epoch Duration:** Exactly 1.0 second per trial (256 samples per channel).
* **Experimental Paradigm & Stimuli:**
  * Visual Event-Related Potential (ERP) paradigm using pictorial stimuli from the standardized Snodgrass and Vanderwart object picture set.
  * **Conditions:**
    1. **Single Stimulus ($S_1$):** A single picture is presented for 300 ms.
    2. **Two Stimuli Matched ($S_2\text{ match}$):** $S_1$ is presented, followed by a 1.6-second inter-stimulus interval (ISI), then $S_2$ which is identical to $S_1$.
    3. **Two Stimuli Non-Matched ($S_2\text{ non-match}$):** $S_2$ is conceptually and physically different from $S_1$.
* **Data Organization:**
  * Distributed in three packages: *Small* (examplary subset), *Large* (selected subjects, ~600 MB), and *Full* (complete trial database, ~1.2 GB compressed text files).
  * Data format: ASCII text files containing trial headers and 64 columns of microvolt measurements.
* **Python Quick-Load Recipe:**
```python
import pandas as pd
import numpy as np

def load_uci_trial(file_path):
    # UCI text files contain lines with "#" comment headers
    # Data lines: trial_number, condition, channel, sample_idx, microvolts
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if len(parts) >= 4:
                data.append([parts[0], int(parts[1]), float(parts[2]), parts[3]])
    df = pd.DataFrame(data, columns=['channel', 'sample_idx', 'uV', 'subject_id'])
    return df
```

---

## 2. OpenNeuro Accession `ds005907`: Alcohol Reinforcement Learning (3-Armed Bandit)

* **Repository:** [OpenNeuro Accession ds005907](https://openneuro.org/datasets/ds005907)
* **Title:** EEG and Behavioral Data in a 3-Armed Bandit Decision-Making Task with Alcohol Cues in Hazardous Drinkers and Healthy Controls.
* **Standards Compliance:** Fully BIDS (Brain Imaging Data Structure) compliant.
* **Subjects:**
  * Cohort: $N = 53$ participants (hazardous/harmful alcohol drinkers determined via AUDIT $\ge 8$ vs. healthy low-risk drinkers AUDIT $< 8$).
  * Rich demographic and psychological questionnaires included: AUDIT scores, Obsessive-Compulsive Drinking Scale (OCDS), Barratt Impulsiveness Scale (BIS-11).
* **Recording Specifications:**
  * **System:** 64-channel BrainProducts actiCHamp active electrode system.
  * **Sampling Frequency:** $f_s = 500\text{ Hz}$ (or $1000\text{ Hz}$ raw).
  * **Electrode Layout:** Standard 10-20 layout with ground and reference at `FPz` / `FCz`.
* **Experimental Paradigm:**
  * Probabilistic 3-Armed Bandit reinforcement learning task.
  * Payoff schedules include winning or losing points associated with images of alcoholic beverages (beer, wine, spirits) versus non-alcoholic beverages (water, juice, soda).
  * Evaluates electrophysiological correlates of reward prediction error: Reward Positivity (RewP), Feedback-Related Negativity (FRN), and cue-locked P300.
* **File Formats:** BrainVision format (`.vhdr`, `.vmrk`, `.eeg`) and BIDS JSON sidecars (`events.tsv`, `channels.tsv`).

---

## 3. OpenNeuro Accession `ds004515`: Alcohol Imagery Reinforcement Learning Task

* **Repository:** [OpenNeuro Accession ds004515](https://openneuro.org/datasets/ds004515)
* **Title:** EEG: Alcohol imagery reinforcement learning task with light and heavy drinker participants.
* **Standards Compliance:** BIDS format.
* **Subjects:**
  * Cohort: Young adults stratified into Light Drinkers vs. Heavy Binge Drinkers.
* **Recording Specifications:**
  * **Electrodes:** 32/64 scalp EEG channels.
  * **Sampling Frequency:** $f_s = 500\text{ Hz}$.
  * **Electrode Impedances:** Maintained $< 10\text{ k}\Omega$.
* **Experimental Paradigm:**
  * Combines guided mental imagery of alcohol vs. neutral scenes with a subsequent reinforcement learning / feedback task.
  * Maps how mental representation of alcohol alters anterior cingulate cortex (ACC) monitoring and striatal dopamine signaling via event-related potentials.
* **File Formats:** BIDS BrainVision format (`.vhdr`, `.vmrk`, `.eeg`).

---

## 4. OpenNeuro Accession `ds008851`: Alcohol Cue-Reactivity and Craving Dynamics

* **Repository:** [OpenNeuro Accession ds008851](https://openneuro.org/datasets/ds008851)
* **Title:** Resting-State and Cue-Reactivity EEG in Alcohol Use Disorder.
* **Subjects:**
  * Inpatients undergoing alcohol detoxification treatment compared with social drinker control subjects.
* **Recording Specifications:**
  * **Montage:** 64-channel ActiCap.
  * **Protocols:** 5-minute Eyes-Open (EO) resting state, 5-minute Eyes-Closed (EC) resting state, followed by a passive visual cue-reactivity task (standardized Geneva Alcohol Pictures - GAP).
  * **Biomarkers Measured:** Late Positive Potential (LPP), Frontal Alpha Asymmetry (FAA), resting-state functional connectivity (wPLI).

---

## 5. Zenodo / Clinical Repositories: Opioid Use Disorder (OUD) EEG Datasets

* **Repository:** Zenodo & Clinical Electrophysiology Open Archives (e.g., [Zenodo Opioid Electrophysiology Collection](https://zenodo.org/))
* **Cohort Profile:**
  * Individuals with severe Opioid Use Disorder undergoing Methadone Maintenance Treatment (MMT) or Buprenorphine pharmacotherapy ($N = 30$ to $N = 65$ per cohort), compared with matched non-opioid using controls.
* **Recording Specifications:**
  * **System:** 19-channel standard clinical 10-20 montage (`Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2`).
  * **Sampling Frequency:** $f_s = 250\text{ Hz}$ or $500\text{ Hz}$.
  * **Conditions:** Continuous resting state (Eyes Closed).
* **Research Value:** Widely used in testing nonlinear dynamic features (Higuchi Fractal Dimension, sample entropy, Lempel-Ziv complexity) and delta/theta power slowing induced by chronic opioid receptor downregulation.

---

## 6. Mendeley Data / Open Repositories: Internet Gaming Disorder (IGD) and Behavioral Addiction

* **Repository:** [Mendeley Data - EEG in Internet Gaming Disorder](https://data.mendeley.com/)
* **Subject Demographics:**
  * Young adult and adolescent subjects diagnosed with Internet Gaming Disorder (IGD) via DSM-5 criteria ($N = 35–50$) vs. recreational video game players and non-gamer controls.
* **Recording Specifications:**
  * **Montage:** 16-channel to 32-channel EEG systems.
  * **Paradigms:**
    1. Resting-state (Eyes Open / Eyes Closed).
    2. Go/No-Go inhibitory control task with gaming-related distractor cues.
    3. Cue-exposure ERP task (displaying battle scenes, loot boxes, and character selection screens).
* **Primary Markers:** Frontal Theta-to-Beta Ratio (TBR), N200 conflict monitoring amplitude, and cue-elicited LPP.

---

## Summary Matrix of Alcoholism & Addiction Datasets

| Dataset Identifier | Domain | Subjects ($N$) | Channels | Sampling Rate ($f_s$) | Format | Tasks / Paradigms | Access Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **UCI Begleiter EEG** | Alcohol Use Disorder | 122 | 64 | 256 Hz | Raw ASCII / CSV | Visual S1-S2 Object Matching ERP | Public (Open) |
| **OpenNeuro ds005907** | AUD / Hazardous Drinking | 53 | 64 | 500 Hz | BIDS (BrainVision) | 3-Armed Bandit RL with Alcohol Cues | Open (CC0) |
| **OpenNeuro ds004515** | Binge Drinking | ~40 | 32/64 | 500 Hz | BIDS (BrainVision) | Alcohol Imagery + Reinforcement Learning | Open (CC0) |
| **OpenNeuro ds008851** | AUD Inpatients / Detox | 45 | 64 | 500 Hz | BIDS | Resting EO/EC + Visual Cue-Reactivity | Open (CC0) |
| **Zenodo OUD Cohort** | Opioid Use Disorder | 60 | 19 | 250 Hz | EDF / MAT | Resting-State Eyes Closed | Open (CC-BY) |
| **Mendeley IGD Dataset** | Internet Gaming Disorder | 48 | 32 | 500 Hz | EEGLAB (.set) | Resting-State + Go/No-Go Task | Open (CC-BY 4.0) |
