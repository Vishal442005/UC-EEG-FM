# Standardized EEG Signal Processing, Feature Engineering, and Validation Standards

## 1. The BIDS Standardization Protocol

The **Brain Imaging Data Structure (BIDS)** is the universally adopted standard for organizing, formatting, and sharing human neuroimaging and electrophysiological data.

### 1.1 Directory Hierarchy & Sidecars
A standard BIDS-EEG dataset adheres strictly to the following layout:

```
project_root/
├── dataset_description.json
├── participants.tsv
├── participants.json
├── README.md
├── sub-001/
│   └── eeg/
│       ├── sub-001_task-rest_eeg.edf (or .vhdr / .set / .bdf)
│       ├── sub-001_task-rest_eeg.json
│       ├── sub-001_task-rest_events.tsv
│       ├── sub-001_task-rest_channels.tsv
│       ├── sub-001_electrodes.tsv
│       └── sub-001_coordsystem.json
```

* **`_events.tsv`:** Tab-separated file logging trial events: `onset`, `duration`, `trial_type`, `response_time`, `stimulus_id`.
* **`_channels.tsv`:** Tab-separated metadata: `name`, `type` (`EEG`, `EOG`, `EMG`, `MISC`), `units` (`uV`), `low_cutoff`, `high_cutoff`, `status` (`good`, `bad`).
* **`_electrodes.tsv`:** 3D Cartesian coordinates $(x, y, z)$ in standard MNI space for spatial modeling and source localization.

---

## 2. Standard Preprocessing Pipeline (MNE-Python & EEGLAB)

A clinically validated, reproducible preprocessing pipeline consists of six sequential stages:

```
[ Raw Continuous EEG ]
         │
         ▼
[ 1. Filtering ] ───> Zero-phase FIR: High-pass (0.5 Hz) + Low-pass (45 Hz) + Notch (50/60 Hz)
         │
         ▼
[ 2. Re-Referencing ] ───> Common Average Reference (CAR) or REST
         │
         ▼
[ 3. Bad Channel Repair ] ───> Flatline/Noisy channel detection + Spherical Spline Interpolation
         │
         ▼
[ 4. ASR Cleaning ] ───> Artifact Subspace Reconstruction (ASR, k = 15 - 20)
         │
         ▼
[ 5. ICA & ICLabel ] ───> Extended Infomax ICA -> Automatic component tagging (Eye/Muscle removal)
         │
         ▼
[ Cleaned Analysis-Ready EEG ]
```

### 2.1 Complete MNE-Python Preprocessing Pipeline
```python
import mne
from mne.preprocessing import ICA

def preprocess_eeg_raw(file_path, l_freq=0.5, h_freq=45.0, notch_freq=50.0):
    # 1. Load Raw Recording
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    
    # 2. Standardize Channel Names & Montage
    raw.rename_channels(lambda s: s.strip('.').upper())
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')
    
    # 3. Filtering (Zero-phase Hamming windowed FIR)
    raw.notch_filter(freqs=notch_freq, fir_design='firwin', verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
    
    # 4. Re-referencing (Common Average Reference)
    raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False)
    
    # 5. Independent Component Analysis (ICA) for Artifact Rejection
    # Fit ICA on high-pass 1.0 Hz filtered copy for stability
    raw_ica_copy = raw.copy().filter(l_freq=1.0, h_freq=40.0, verbose=False)
    ica = ICA(n_components=min(20, len(raw.ch_names) - 1), random_state=42, max_iter=800)
    ica.fit(raw_ica_copy)
    
    # Automatic detection of ocular blink components using frontal channels
    eog_indices, _ = ica.find_bads_eog(raw, ch_name='FP1' if 'FP1' in raw.ch_names else None)
    ica.exclude = eog_indices
    raw_cleaned = ica.apply(raw.copy())
    
    return raw_cleaned
```

---

## 3. Mathematical Formulations of Key Biomarkers

### 3.1 Differential Entropy (DE)
For a continuous EEG signal $X$ segment following a Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$:
$$h(X) = \int_{-\infty}^{\infty} -f(x)\ln(f(x))\,dx = \frac{1}{2}\ln\left(2\pi e \sigma^2\right)$$
When computed across fixed frequency bands (Delta, Theta, Alpha, Beta, Gamma), DE provides higher inter-session stability and lower variance than raw spectral power.

### 3.2 Weighted Phase Lag Index (wPLI)
To quantify true functional connectivity while completely discarding false zero-lag volume conduction artifacts:
$$\text{wPLI}_{xy} = \frac{\left| \sum_{t=1}^N |\text{Im}(S_{xy}(t))| \cdot \text{sgn}(\text{Im}(S_{xy}(t))) \right|}{\sum_{t=1}^N |\text{Im}(S_{xy}(t))|}$$
where $S_{xy}(t)$ is the cross-spectral density between channel $x$ and channel $y$ at time $t$.

### 3.3 Higuchi Fractal Dimension (HFD)
Measures the self-similarity and complexity of a time series $X(1), X(2), \dots, X(N)$ across multiple time scales $k \in \{1, \dots, k_{\max}\}$:
$$L_m(k) = \frac{\sum_{i=1}^{\lfloor \frac{N-m}{k} \rfloor} |X(m + ik) - X(m + (i-1)k)| \cdot \frac{N-1}{\lfloor \frac{N-m}{k} \rfloor \cdot k}}{k}$$
The average length $\langle L(k) \rangle \propto k^{-D_{\text{HFD}}}$. The slope of $\ln\langle L(k) \rangle$ vs. $\ln(1/k)$ yields the fractal dimension $D_{\text{HFD}}$.

---

## 4. The Cardinal Validation Rule: Preventing the Data Leakage Disaster

### 4.1 The Epidemic of Flawed Epoch-Level Splitting
In peer-reviewed EEG literature, hundreds of papers report classification accuracies $>98\%$ on small cohorts ($N = 20 - 50$ subjects). In over $80\%$ of these cases, the high performance is an illusion caused by **Random Epoch Splitting**:

```
WRONG / FLAWED (Epoch Leakage):
Subject 1 (200 epochs) ──┐
Subject 2 (200 epochs) ──┼──> Pooled (24,000 epochs) ──> Random 80/20 Split
Subject 3 (200 epochs) ──┘                                      │
                                                 ┌──────────────┴──────────────┐
                                                 ▼                             ▼
                                        Train Set (80%)                Test Set (20%)
                                   (Contains Subj 1, 2, 3)        (Contains Subj 1, 2, 3!)
                                   ==> Model memorizes subject identity, NOT disorder!

CORRECT / CLINICAL (Subject-Independent):
Subject 1, 2, ..., 40   ──────> Train Set (Exclusively Subjects 1-40)
Subject 41, 42, ..., 50 ──────> Test Set (Exclusively Subjects 41-50: Unseen Patients)
                                ==> Evaluates true clinical generalization!
```

### 4.2 Standard Validation Protocols
1. **Leave-One-Subject-Out (LOSO) Cross-Validation:**
   * In a cohort of $N$ subjects, train on $N-1$ subjects and evaluate strictly on the remaining held-out subject. Repeat $N$ times and report mean and standard deviation.
2. **Subject-Stratified Group K-Fold:**
   * Group epochs by `subject_id`. Partition subjects into $K$ folds ensuring that no subject's epochs appear simultaneously in training and validation folds.
3. **Cross-Dataset External Validation:**
   * Train on Dataset A (e.g., UCI Alcoholism, recorded in New York) and test directly without retraining on Dataset B (e.g., OpenNeuro `ds005907`, recorded in Europe). This provides the definitive test of algorithmic robustness.
