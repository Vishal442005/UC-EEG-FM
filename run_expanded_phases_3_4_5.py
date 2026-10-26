"""
Deepened Clinical Benchmarking (Phase 3), Bidirectional Transfer (Phase 4),
and Scalp Topographical Brain Mapping (Phase 5) for UC-EEG-FM.
"""

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import h5py
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score,
    confusion_matrix, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne

BASE_DIR = r"E:\All EEG"
sys.path.insert(0, BASE_DIR)

FEAT_DIR = os.path.join(BASE_DIR, "Processed_Features")
MODEL_DIR = os.path.join(BASE_DIR, "Models")
FIG_DIR = os.path.join(BASE_DIR, "Figures")
os.makedirs(FIG_DIR, exist_ok=True)
ARTIFACT_DIR = r"C:\Users\Visha\.gemini\antigravity\brain\a487a487-156f-4781-ba99-0b2ec1ba776a"

print("=" * 80)
print("DEEPENED CLINICAL VALIDATION: PHASES 3, 4, AND 5")
print("=" * 80)

# 1. Load Embeddings and Data
embed_data = np.load(os.path.join(MODEL_DIR, "latent_manifold_embeddings.npz"), allow_pickle=True)
Z_fm = embed_data['embeddings'] # (49472, 64)
labels_diag = embed_data['labels'] # 0: Control, 1: AD, 2: AUD, 3: FTD, 4: Epilepsy
groups_all = embed_data['groups']
domains_all = embed_data['domains']

raw_feats_list = []
chb_ictal_labels = []

for folder, domain in [('openneuro_ds004504_dementia', 'Neurodegeneration_Dementia'),
                       ('openneuro_ds005907_alcohol', 'Alcoholism_Addiction'),
                       ('chbmit_epilepsy', 'Epilepsy_Seizures')]:
    h5_files = sorted(os.listdir(os.path.join(FEAT_DIR, folder)))
    for f in h5_files:
        if not f.endswith('.h5'): continue
        with h5py.File(os.path.join(FEAT_DIR, folder, f), 'r') as h5f:
            feats = np.array(h5f['features_mean'])
            raw_feats_list.append(feats)
            if 'labels_ictal' in h5f:
                chb_ictal_labels.append(np.array(h5f['labels_ictal']))
            else:
                chb_ictal_labels.append(np.zeros(len(feats), dtype=np.int32))

X_raw = np.concatenate(raw_feats_list, axis=0).astype(np.float32)
y_ictal = np.concatenate(chb_ictal_labels, axis=0)

# =========================================================================
# 1. PHASE 3: COMPREHENSIVE CLINICAL METRICS & MULTI-MODEL BENCHMARK
# =========================================================================
print("\n--- PHASE 3 DEEPENING: MULTI-MODEL COMPARISON & CLINICAL METRICS ---", flush=True)

def evaluate_multi_model(X_raw_sub, Z_fm_sub, y_sub, groups_sub, task_name="Task"):
    gkf = GroupKFold(n_splits=5)
    
    models = {
        'Logistic Regression (Raw)': LogisticRegression(C=1.0, max_iter=500, random_state=42),
        'Support Vector Machine (Raw RBF)': SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
        'Gradient Boosting (Raw GBDT)': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        'Random Forest (Raw RF)': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
        'UC-EEG-FM (Foundation Probe)': LogisticRegression(C=1.0, max_iter=500, random_state=42)
    }
    
    fold_aurocs = {m: [] for m in models}
    oof_probs = {m: np.zeros(len(y_sub)) for m in models}
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_raw_sub, y_sub, groups=groups_sub)):
        for name, clf in models.items():
            if 'UC-EEG-FM' in name:
                clf.fit(Z_fm_sub[train_idx], y_sub[train_idx])
                p = clf.predict_proba(Z_fm_sub[test_idx])[:, 1]
            else:
                clf.fit(X_raw_sub[train_idx], y_sub[train_idx])
                p = clf.predict_proba(X_raw_sub[test_idx])[:, 1]
            oof_probs[name][test_idx] = p
            auc = roc_auc_score(y_sub[test_idx], p)
            fold_aurocs[name].append(auc)
            
    summary = {}
    for name in models:
        y_prob = oof_probs[name]
        fpr, tpr, thresholds = roc_curve(y_sub, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]
        y_pred = (y_prob >= best_thresh).astype(int)
        
        cm = confusion_matrix(y_sub, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        overall_auc = roc_auc_score(y_sub, y_prob)
        f1 = f1_score(y_sub, y_pred, average='macro')
        
        summary[name] = {
            'overall_auroc': float(overall_auc),
            'fold_aurocs': [float(x) for x in fold_aurocs[name]],
            'mean_fold_auroc': float(np.mean(fold_aurocs[name])),
            'std_fold_auroc': float(np.std(fold_aurocs[name])),
            'macro_f1': float(f1),
            'sensitivity': float(sens),
            'specificity': float(spec),
            'ppv': float(ppv),
            'npv': float(npv),
            'optimal_threshold': float(best_thresh),
            'youden_j': float(j_scores[best_idx])
        }
        
    stat, p_val_rf = wilcoxon(fold_aurocs['UC-EEG-FM (Foundation Probe)'], fold_aurocs['Random Forest (Raw RF)'])
    summary['wilcoxon_fm_vs_rf_pval'] = float(p_val_rf)
    
    print(f"\nTask: {task_name}")
    print(f"{'Model':<32} | {'AUROC':<7} | {'Sens':<7} | {'Spec':<7} | {'PPV':<7} | {'NPV':<7} | {'F1':<7}")
    print("-" * 85)
    for name, m in summary.items():
        if name == 'wilcoxon_fm_vs_rf_pval': continue
        print(f"{name:<32} | {m['overall_auroc']:.4f}  | {m['sensitivity']*100:5.1f}% | {m['specificity']*100:5.1f}% | {m['ppv']*100:5.1f}% | {m['npv']*100:5.1f}% | {m['macro_f1']:.4f}")
    print(f"Wilcoxon signed-rank test (UC-EEG-FM vs Random Forest across folds): p = {p_val_rf:.4f}")
    
    return summary

# 1. Alcoholism Task
mask_alc = (domains_all == 'Alcoholism_Addiction')
X_alc = X_raw[mask_alc]
Z_alc = Z_fm[mask_alc]
y_alc = (labels_diag[mask_alc] == 2).astype(int)
groups_alc = groups_all[mask_alc]
phase3_alc = evaluate_multi_model(X_alc, Z_alc, y_alc, groups_alc, "Alcoholism (AUD vs Control)")

# 2. Dementia AD vs Control Task
mask_ad_cn = (domains_all == 'Neurodegeneration_Dementia') & np.isin(labels_diag, [1, 0])
X_ad = X_raw[mask_ad_cn]
Z_ad = Z_fm[mask_ad_cn]
y_ad = (labels_diag[mask_ad_cn] == 1).astype(int)
groups_ad = groups_all[mask_ad_cn]
phase3_ad = evaluate_multi_model(X_ad, Z_ad, y_ad, groups_ad, "Alzheimer's Disease (AD vs Control)")

# =========================================================================
# 2. PHASE 4: BIDIRECTIONAL CROSS-DISORDER TRANSFER
# =========================================================================
print("\n--- PHASE 4 DEEPENING: BIDIRECTIONAL ZERO-SHOT TRANSFER ---", flush=True)

# Dementia -> Alcoholism
clf_dem_to_alc = LogisticRegression(C=1.0, max_iter=500, random_state=42)
clf_dem_to_alc.fit(Z_ad, y_ad)
prob_dem_to_alc = clf_dem_to_alc.predict_proba(Z_alc)[:, 1]
auc_dem_to_alc = roc_auc_score(y_alc, prob_dem_to_alc)

# Alcoholism -> Dementia
clf_alc_to_dem = LogisticRegression(C=1.0, max_iter=500, random_state=42)
clf_alc_to_dem.fit(Z_alc, y_alc)
prob_alc_to_dem = clf_alc_to_dem.predict_proba(Z_ad)[:, 1]
auc_alc_to_dem = roc_auc_score(y_ad, prob_alc_to_dem)

print(f"Zero-Shot Dementia -> Alcoholism AUROC: {auc_dem_to_alc:.4f}")
print(f"Zero-Shot Alcoholism -> Dementia AUROC: {auc_alc_to_dem:.4f}")

# =========================================================================
# 3. PHASE 5: SCALP TOPOGRAPHICAL BRAIN MAPS (TOPOMAPS)
# =========================================================================
print("\n--- PHASE 5 DEEPENING: SCALP TOPOGRAPHICAL ACTIVATION MAPS ---", flush=True)

montage = mne.channels.make_standard_montage('standard_1020')

# Standard 10-20 channels for Dementia
ch_names_dem = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
info_dem = mne.create_info(ch_names=ch_names_dem, sfreq=256.0, ch_types='eeg')
info_dem.set_montage(montage)

# Load spatial features from Dementia subjects
dem_folder = os.path.join(FEAT_DIR, "openneuro_ds004504_dementia")
ad_spatial_list = []
cn_spatial_list = []

for f in sorted(os.listdir(dem_folder)):
    if not f.endswith('.h5'): continue
    with h5py.File(os.path.join(dem_folder, f), 'r') as h5f:
        diag = str(h5f.attrs['diagnosis'])
        sp_feats = np.array(h5f['features_spatial']) # (N_epochs, 19, 11)
        mean_sp = np.mean(sp_feats, axis=0) # (19, 11)
        if diag == 'AlzheimersDisease':
            ad_spatial_list.append(mean_sp)
        elif diag == 'HealthyControl':
            cn_spatial_list.append(mean_sp)

print(f"Loaded Dementia Spatial: AD={len(ad_spatial_list)} subjects, CN={len(cn_spatial_list)} subjects", flush=True)
ad_spatial_mean = np.mean(ad_spatial_list, axis=0) # (19, 11)
cn_spatial_mean = np.mean(cn_spatial_list, axis=0) # (19, 11)

delta_slowing_ad = ad_spatial_mean[:, 10] - cn_spatial_mean[:, 10]
delta_alpha_ad = (ad_spatial_mean[:, 2] - cn_spatial_mean[:, 2]) * 100.0 # Percentage points

# Alcoholism: 18 mapped channels in 56-channel montage
ch_names_alc = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
info_alc = mne.create_info(ch_names=ch_names_alc, sfreq=256.0, ch_types='eeg')
info_alc.set_montage(montage)

# Channel indices corresponding to ch_names_alc in ds005907
alc_all_names = ['Fp1', 'Fz', 'F3', 'F7', 'FC1', 'C3', 'T7', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2', 'Cz', 'C4', 'FC6', 'FC2', 'F4', 'F8', 'Fp2']
alc_indices = [0, 24, 2, 22, 5, 19, 9, 14, 11, 13, 3, 23, 6, 10, 15, 1, 18, 8]

alc_folder = os.path.join(FEAT_DIR, "openneuro_ds005907_alcohol")
aud_spatial_list = []
cn_alc_spatial_list = []

for f in sorted(os.listdir(alc_folder)):
    if not f.endswith('.h5'): continue
    with h5py.File(os.path.join(alc_folder, f), 'r') as h5f:
        diag = str(h5f.attrs['diagnosis'])
        sp_feats = np.array(h5f['features_spatial']) # (N_epochs, 56, 11)
        mean_sp = np.mean(sp_feats[:, alc_indices, :], axis=0) # (18, 11)
        if diag == 'AlcoholUseDisorder':
            aud_spatial_list.append(mean_sp)
        elif diag == 'HealthyControl':
            cn_alc_spatial_list.append(mean_sp)

print(f"Loaded Alcohol Spatial: AUD={len(aud_spatial_list)} subjects, CN={len(cn_alc_spatial_list)} subjects", flush=True)
aud_spatial_mean = np.mean(aud_spatial_list, axis=0)
cn_alc_spatial_mean = np.mean(cn_alc_spatial_list, axis=0)

delta_beta_aud = (aud_spatial_mean[:, 3] - cn_alc_spatial_mean[:, 3]) * 100.0
delta_gamma_de_aud = aud_spatial_mean[:, 9] - cn_alc_spatial_mean[:, 9]

# -------------------------------------------------------------
# PLOT FIGURE 7: SCALP TOPOGRAPHICAL ACTIVATION MAPS (300 DPI)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

# Topo 1: Alzheimer's Slowing Ratio
im0, _ = mne.viz.plot_topomap(delta_slowing_ad, info_dem, axes=axes[0], show=False, cmap='RdBu_r')
axes[0].set_title("A. Alzheimer's Disease (AD vs CN)\nCortical Slowing Ratio $\\Delta (\\delta+\\theta)/(\\alpha+\\beta)$", fontsize=11, fontweight='bold')
cbar0 = plt.colorbar(im0, ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.08)
cbar0.set_label("$\\Delta$ Slowing Ratio", fontsize=9)

# Topo 2: Alzheimer's Alpha Power Decay
im1, _ = mne.viz.plot_topomap(delta_alpha_ad, info_dem, axes=axes[1], show=False, cmap='RdBu_r')
axes[1].set_title("B. Alzheimer's Disease (AD vs CN)\nPosterior $\\alpha$ Power Desynchronization", fontsize=11, fontweight='bold')
cbar1 = plt.colorbar(im1, ax=axes[1], orientation='horizontal', fraction=0.046, pad=0.08)
cbar1.set_label("$\\Delta$ Relative $\\alpha$ Power (%)", fontsize=9)

# Topo 3: Alcoholism Beta Power Elevation
im2, _ = mne.viz.plot_topomap(delta_beta_aud, info_alc, axes=axes[2], show=False, cmap='RdBu_r')
axes[2].set_title("C. Alcohol Use Disorder (AUD vs CN)\nFrontal $\\beta$ Power Hyperarousal", fontsize=11, fontweight='bold')
cbar2 = plt.colorbar(im2, ax=axes[2], orientation='horizontal', fraction=0.046, pad=0.08)
cbar2.set_label("$\\Delta$ Relative $\\beta$ Power (%)", fontsize=9)

# Topo 4: Alcoholism Gamma Differential Entropy
im3, _ = mne.viz.plot_topomap(delta_gamma_de_aud, info_alc, axes=axes[3], show=False, cmap='RdBu_r')
axes[3].set_title("D. Alcohol Use Disorder (AUD vs CN)\n$\\gamma$ Differential Entropy Elevation", fontsize=11, fontweight='bold')
cbar3 = plt.colorbar(im3, ax=axes[3], orientation='horizontal', fraction=0.046, pad=0.08)
cbar3.set_label("$\\Delta$ Differential Entropy (nats)", fontsize=9)

plt.suptitle("Figure 7: Scalp Topographical Distribution of Electrophysiological Biomarkers Across Clinical Cohorts", fontsize=15, fontweight='bold', y=1.04)
plt.tight_layout()
fig7_path = os.path.join(FIG_DIR, "Figure_7_Topographical_Scalp_Activation_Maps.png")
plt.savefig(fig7_path, dpi=300, bbox_inches='tight')
plt.close()
shutil.copy(fig7_path, os.path.join(ARTIFACT_DIR, "Figure_7_Topographical_Scalp_Activation_Maps.png"))
print("Figure 7 generated and saved successfully.", flush=True)

# Save combined summary
expanded_results = {
    'phase3_multimodel_alcoholism': phase3_alc,
    'phase3_multimodel_dementia': phase3_ad,
    'phase4_bidirectional_transfer': {
        'dementia_to_alcoholism_auroc': float(auc_dem_to_alc),
        'alcoholism_to_dementia_auroc': float(auc_alc_to_dem)
    },
    'phase5_topographical_stats': {
        'ad_channels': ch_names_dem,
        'alc_channels': ch_names_alc,
        'ad_delta_slowing_mean': float(np.mean(delta_slowing_ad)),
        'ad_delta_alpha_mean_pct': float(np.mean(delta_alpha_ad)),
        'aud_delta_beta_mean_pct': float(np.mean(delta_beta_aud)),
        'aud_delta_gamma_de_mean': float(np.mean(delta_gamma_de_aud))
    }
}

json_path = os.path.join(MODEL_DIR, "expanded_phases_3_4_5_benchmarks.json")
with open(json_path, 'w') as f:
    json.dump(expanded_results, f, indent=2)

print(f"Expanded Phase 3, 4, 5 results saved to {json_path}.", flush=True)
