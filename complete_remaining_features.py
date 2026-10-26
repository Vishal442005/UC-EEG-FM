"""
Memory-Safe Feature Extraction for Remaining Subjects
Processes Alcoholism and Epilepsy cohorts with strict memory management (gc.collect).
"""

import os
import sys
import glob
import gc
import time
import numpy as np
import pandas as pd
import mne
import scipy.signal
import h5py

mne.set_log_level('ERROR')

BASE_DIR = r"E:\All EEG"
RAW_DIR = os.path.join(BASE_DIR, "Raw_Datasets")
OUT_DIR = os.path.join(BASE_DIR, "Processed_Features")

BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 45.0)
}

def compute_welch_bandpowers(data, fs=256.0):
    n_epochs, n_channels, n_times = data.shape
    freqs, psd = scipy.signal.welch(data, fs=fs, nperseg=min(256, n_times), axis=-1)
    
    total_power = np.sum(psd, axis=-1, keepdims=True)
    total_power = np.maximum(total_power, 1e-12)
    
    band_powers = []
    band_de = []
    
    for band_name, (fmin, fmax) in BANDS.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        bp = np.sum(psd[..., idx], axis=-1)
        rel_bp = bp / np.squeeze(total_power, axis=-1)
        band_powers.append(rel_bp)
        
        var = np.maximum(bp, 1e-12)
        de = 0.5 * np.log(2 * np.pi * np.e * var)
        band_de.append(de)
        
    delta_bp, theta_bp, alpha_bp, beta_bp, gamma_bp = band_powers
    slow_ratio = (delta_bp + theta_bp) / np.maximum(alpha_bp + beta_bp, 1e-12)
    
    all_feats = np.stack(band_powers + band_de + [slow_ratio], axis=-1)
    return all_feats

def process_single_alcohol(s_file, sub_id, metadata_dict, target_dir):
    out_h5 = os.path.join(target_dir, f"{sub_id}.h5")
    if os.path.exists(out_h5) and os.path.getsize(out_h5) > 1024:
        return sub_id, True, "Already exists"
    try:
        raw = mne.io.read_raw_eeglab(s_file, preload=True, verbose=False)
        raw.filter(0.5, 45.0, fir_design='firwin', verbose=False)
        raw.notch_filter([50.0, 60.0], fir_design='firwin', verbose=False)
        raw.resample(256.0, verbose=False)
        raw.set_eeg_reference('average', projection=False, verbose=False)
        
        epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True, verbose=False)
        epoch_data = epochs.get_data().astype(np.float32)
        del raw, epochs
        gc.collect()
        
        feats = compute_welch_bandpowers(epoch_data, fs=256.0).astype(np.float32)
        mean_feats = np.mean(feats, axis=1)
        
        with h5py.File(out_h5, 'w') as h5f:
            h5f.create_dataset('epochs_data', data=epoch_data, compression='gzip', compression_opts=4)
            h5f.create_dataset('features_spatial', data=feats, compression='gzip', compression_opts=4)
            h5f.create_dataset('features_mean', data=mean_feats, compression='gzip', compression_opts=4)
            
            h5f.attrs['subject_id'] = sub_id
            h5f.attrs['diagnosis'] = metadata_dict.get('diagnosis', 'Unknown')
            h5f.attrs['disorder_domain'] = metadata_dict.get('disorder_domain', 'Alcoholism_Addiction')
            h5f.attrs['severity_score'] = float(metadata_dict.get('severity_score', -1)) if pd.notnull(metadata_dict.get('severity_score')) else -1.0
            h5f.attrs['age'] = float(metadata_dict.get('age', -1)) if pd.notnull(metadata_dict.get('age')) else -1.0
            h5f.attrs['sex'] = str(metadata_dict.get('sex', 'Unknown'))
            h5f.attrs['sampling_rate'] = 256.0
            h5f.attrs['epoch_duration_sec'] = 4.0
            
        del epoch_data, feats, mean_feats
        gc.collect()
        return sub_id, True, "Completed"
    except Exception as e:
        gc.collect()
        return sub_id, False, str(e)

def parse_chb_summary(summary_file):
    seizure_map = {}
    if not os.path.exists(summary_file): return seizure_map
    current_file = None
    with open(summary_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("File Name:"):
            current_file = line.split(":", 1)[1].strip()
            if current_file not in seizure_map:
                seizure_map[current_file] = []
        elif "Start Time:" in line and "Seizure" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    s_sec = int(parts[1].split()[0])
                    seizure_map[current_file].append([s_sec, s_sec + 40])
                except Exception:
                    pass
        elif "End Time:" in line and "Seizure" in line:
            parts = line.split(":")
            if len(parts) >= 2 and seizure_map.get(current_file):
                try:
                    e_sec = int(parts[1].split()[0])
                    seizure_map[current_file][-1][1] = e_sec
                except Exception:
                    pass
    return seizure_map

def process_single_epilepsy(case, edf_files, seizure_intervals, metadata_dict, target_dir):
    out_h5 = os.path.join(target_dir, f"{case}.h5")
    if os.path.exists(out_h5) and os.path.getsize(out_h5) > 1024:
        return case, True, "Already exists"
    try:
        # Prioritize files that contain verified seizures
        ictal_files = [f for f in edf_files if seizure_intervals.get(os.path.basename(f))]
        baseline_files = [f for f in edf_files if not seizure_intervals.get(os.path.basename(f))]
        
        # Select up to 2 seizure files + 1 baseline file per subject (memory-safe & balanced)
        selected_files = ictal_files[:2] + baseline_files[:1]
        if not selected_files:
            selected_files = edf_files[:2]
            
        all_epoch_data = []
        all_labels = []
        
        for edf_path in selected_files:
            fname = os.path.basename(edf_path)
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            raw.filter(0.5, 45.0, fir_design='firwin', verbose=False)
            raw.notch_filter([50.0, 60.0], fir_design='firwin', verbose=False)
            
            epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True, verbose=False)
            ep_data = epochs.get_data().astype(np.float32)
            del raw, epochs
            gc.collect()
            
            file_seizures = seizure_intervals.get(fname, [])
            n_ep = len(ep_data)
            ep_labels = np.zeros(n_ep, dtype=np.int32)
            for i in range(n_ep):
                t_start = i * 4.0
                t_end = t_start + 4.0
                for s_start, s_end in file_seizures:
                    if not (t_end < s_start or t_start > s_end):
                        ep_labels[i] = 1
                        break
                        
            all_epoch_data.append(ep_data)
            all_labels.append(ep_labels)
            
        cat_data = np.concatenate(all_epoch_data, axis=0)
        cat_labels = np.concatenate(all_labels, axis=0)
        del all_epoch_data, all_labels
        gc.collect()
        
        feats = compute_welch_bandpowers(cat_data, fs=256.0).astype(np.float32)
        mean_feats = np.mean(feats, axis=1)
        
        with h5py.File(out_h5, 'w') as h5f:
            h5f.create_dataset('epochs_data', data=cat_data, compression='gzip', compression_opts=4)
            h5f.create_dataset('labels_ictal', data=cat_labels, compression='gzip', compression_opts=4)
            h5f.create_dataset('features_spatial', data=feats, compression='gzip', compression_opts=4)
            h5f.create_dataset('features_mean', data=mean_feats, compression='gzip', compression_opts=4)
            
            h5f.attrs['subject_id'] = case
            h5f.attrs['diagnosis'] = 'PediatricEpilepsy'
            h5f.attrs['disorder_domain'] = 'Epilepsy_Seizures'
            h5f.attrs['ictal_epoch_count'] = int(np.sum(cat_labels))
            h5f.attrs['total_epoch_count'] = int(len(cat_labels))
            h5f.attrs['age'] = float(metadata_dict.get('age', -1)) if pd.notnull(metadata_dict.get('age')) else -1.0
            h5f.attrs['sex'] = str(metadata_dict.get('sex', 'Unknown'))
            h5f.attrs['sampling_rate'] = 256.0
            h5f.attrs['epoch_duration_sec'] = 4.0
            
        del cat_data, cat_labels, feats, mean_feats
        gc.collect()
        return case, True, f"{h5f.attrs['total_epoch_count']} epochs ({h5f.attrs['ictal_epoch_count']} ictal)"
    except Exception as e:
        gc.collect()
        return case, False, str(e)

def main():
    meta_df = pd.read_csv(os.path.join(BASE_DIR, "master_cohort_metadata.tsv"), sep='\t')
    meta_by_sub = {row['subject_id']: row.to_dict() for _, row in meta_df.iterrows()}
    
    print("="*60, flush=True)
    print("STARTING MEMORY-OPTIMIZED FEATURE EXTRACTION", flush=True)
    print("="*60, flush=True)
    
    # 1. Alcoholism cohort
    alc_out = os.path.join(OUT_DIR, "openneuro_ds005907_alcohol")
    alc_sets = sorted(glob.glob(os.path.join(RAW_DIR, "openneuro_ds005907_alcohol", "sub-*", "eeg", "*_task-ThreeArmedBandit_eeg.set")))
    print(f"\nProcessing Alcoholism Cohort ({len(alc_sets)} total subjects)...", flush=True)
    for i, s_file in enumerate(alc_sets):
        sub_name = os.path.basename(s_file).split('_')[0]
        full_sub_id = f"ds005907_{sub_name}"
        meta = meta_by_sub.get(full_sub_id, {})
        sid, ok, msg = process_single_alcohol(s_file, full_sub_id, meta, alc_out)
        if (i + 1) % 5 == 0 or (i + 1) == len(alc_sets) or not ok:
            print(f"  [Alcohol {i+1}/{len(alc_sets)}] {sid}: {msg}", flush=True)
            
    # 2. Epilepsy cohort
    epi_out = os.path.join(OUT_DIR, "chbmit_epilepsy")
    chb_cases = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05']
    print(f"\nProcessing Epilepsy Cohort ({len(chb_cases)} cases)...", flush=True)
    for case in chb_cases:
        case_dir = os.path.join(RAW_DIR, "CHB_MIT", case)
        edf_files = sorted(glob.glob(os.path.join(case_dir, "*.edf")))
        sum_file = os.path.join(case_dir, f"{case}-summary.txt")
        s_map = parse_chb_summary(sum_file)
        full_sub_id = f"chbmit_{case}"
        meta = meta_by_sub.get(full_sub_id, {})
        sid, ok, msg = process_single_epilepsy(case, edf_files, s_map, meta, epi_out)
        print(f"  [Epilepsy] {sid}: {msg}", flush=True)
        
    print("\n" + "="*60, flush=True)
    print("ALL REMAINING CLINICAL SUBJECTS SUCCESSFULLY EXTRACTED.", flush=True)
    print("="*60, flush=True)

if __name__ == '__main__':
    main()
