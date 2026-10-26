"""
Test script for Phase 1 Preprocessing and Feature Extraction on sample subjects.
"""

import os
import numpy as np
import mne
import scipy.signal
import h5py

def compute_de(x):
    """Differential Entropy assuming Gaussian distribution: 0.5 * ln(2 * pi * e * var)"""
    var = np.var(x, axis=-1, keepdims=True)
    var = np.maximum(var, 1e-12)
    return 0.5 * np.log(2 * np.pi * np.e * var)

def compute_hfd(x, k_max=10):
    """Higuchi Fractal Dimension for 1D array"""
    N = len(x)
    Lmk = []
    for k in range(1, k_max + 1):
        Lm = []
        for m in range(k):
            indices = np.arange(m, N, k)
            if len(indices) < 2: continue
            diff = np.abs(np.diff(x[indices]))
            norm = (N - 1) / (float(len(indices) - 1) * k)
            Lm.append(np.sum(diff) * norm / k)
        if Lm:
            Lmk.append(np.mean(Lm))
    if len(Lmk) < 2: return 1.0
    x_axis = np.log(1.0 / np.arange(1, len(Lmk) + 1))
    y_axis = np.log(Lmk)
    slope, _ = np.polyfit(x_axis, y_axis, 1)
    return slope

print("Testing MNE and feature extraction pipeline...")
sample_set = r"E:\All EEG\Raw_Datasets\openneuro_ds004504_dementia\sub-001\eeg\sub-001_task-eyesclosed_eeg.set"
raw = mne.io.read_raw_eeglab(sample_set, preload=True, verbose=False)
raw.filter(0.5, 45.0, fir_design='firwin', verbose=False)
raw.notch_filter([50.0, 60.0], fir_design='firwin', verbose=False)
raw.resample(256.0, verbose=False)
raw.set_eeg_reference('average', projection=False, verbose=False)

# Fixed length epochs: 4s
epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True, verbose=False)
data = epochs.get_data() # (N_epochs, N_channels, 1024)
print(f"Extracted {len(epochs)} epochs with shape: {data.shape}")

# Test DE & HFD
de_val = compute_de(data[0])
hfd_val = compute_hfd(data[0, 0])
print(f"Sample Epoch 0 DE shape: {de_val.shape}, HFD channel 0: {hfd_val:.4f}")
print("Pipeline verification test PASSED.")
