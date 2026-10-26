"""
Phase 3 Multi-Task Clinical Benchmarking Engine (Fixed)
Evaluates downstream diagnostic classification and severity score regression
using strict Subject-Stratified Group 5-Fold Cross-Validation (Zero Data Leakage).
Compares Classical ML Baselines against the UC-EEG-FM Foundation Model.
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score,
    confusion_matrix, mean_absolute_error, r2_score, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = r"E:\All EEG"
FEAT_DIR = os.path.join(BASE_DIR, "Processed_Features")
MODEL_DIR = os.path.join(BASE_DIR, "Models")
FIG_DIR = os.path.join(BASE_DIR, "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Load Pre-trained Foundation Embeddings & Raw Features
print("Loading foundation embeddings and clinical metadata...")
embed_data = np.load(os.path.join(MODEL_DIR, "latent_manifold_embeddings.npz"), allow_pickle=True)
Z_fm = embed_data['embeddings'] # (49472, 64)
labels_diag = embed_data['labels']
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

mask_dementia = (domains_all == 'Neurodegeneration_Dementia')
mask_alcohol = (domains_all == 'Alcoholism_Addiction')
mask_epilepsy = (domains_all == 'Epilepsy_Seizures')

def bootstrap_ci(y_true, y_pred_prob, metric_fn, n_bootstraps=300, is_multiclass=False):
    scores = []
    rng = np.random.RandomState(42)
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < len(np.unique(y_true)):
            continue
        try:
            if is_multiclass:
                score = metric_fn(y_true[idx], y_pred_prob[idx], multi_class='ovr', average='macro')
            else:
                score = metric_fn(y_true[idx], y_pred_prob[idx])
            scores.append(score)
        except Exception:
            pass
    if not scores:
        return 0.0, 0.0
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

def evaluate_classification_groupkfold(X, y, groups, is_multiclass=False, task_name="Task", balance=False):
    gkf = GroupKFold(n_splits=5)
    y_oof_prob = np.zeros((len(y), len(np.unique(y)))) if is_multiclass else np.zeros(len(y))
    y_oof_pred = np.zeros(len(y))
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        cw = 'balanced' if balance else None
        clf = LogisticRegression(C=1.0, class_weight=cw, max_iter=1000, random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        
        if is_multiclass:
            probs = clf.predict_proba(X[test_idx])
            y_oof_prob[test_idx] = probs
            y_oof_pred[test_idx] = np.argmax(probs, axis=1)
        else:
            probs = clf.predict_proba(X[test_idx])[:, 1]
            y_oof_prob[test_idx] = probs
            y_oof_pred[test_idx] = (probs >= 0.5).astype(int)
            
    if is_multiclass:
        auroc = roc_auc_score(y, y_oof_prob, multi_class='ovr', average='macro')
        ci_low, ci_high = bootstrap_ci(y, y_oof_prob, roc_auc_score, is_multiclass=True)
    else:
        auroc = roc_auc_score(y, y_oof_prob)
        ci_low, ci_high = bootstrap_ci(y, y_oof_prob, roc_auc_score, is_multiclass=False)
        
    f1 = f1_score(y, y_oof_pred, average='macro')
    acc = accuracy_score(y, y_oof_pred)
    bal_acc = balanced_accuracy_score(y, y_oof_pred)
    cm = confusion_matrix(y, y_oof_pred)
    
    return {
        'auroc': float(auroc),
        'auroc_ci': [ci_low, ci_high],
        'macro_f1': float(f1),
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'confusion_matrix': cm.tolist(),
        'y_true': y,
        'y_prob': y_oof_prob,
        'y_pred': y_oof_pred
    }

def evaluate_baseline_classical(X_raw_sub, y_sub, groups_sub, is_multiclass=False, balance=False):
    gkf = GroupKFold(n_splits=5)
    y_oof_prob = np.zeros((len(y_sub), len(np.unique(y_sub)))) if is_multiclass else np.zeros(len(y_sub))
    y_oof_pred = np.zeros(len(y_sub))
    
    for train_idx, test_idx in gkf.split(X_raw_sub, y_sub, groups=groups_sub):
        cw = 'balanced' if balance else None
        clf = RandomForestClassifier(n_estimators=100, class_weight=cw, random_state=42, n_jobs=-1)
        clf.fit(X_raw_sub[train_idx], y_sub[train_idx])
        if is_multiclass:
            probs = clf.predict_proba(X_raw_sub[test_idx])
            y_oof_prob[test_idx] = probs
            y_oof_pred[test_idx] = np.argmax(probs, axis=1)
        else:
            probs = clf.predict_proba(X_raw_sub[test_idx])[:, 1]
            y_oof_prob[test_idx] = probs
            y_oof_pred[test_idx] = (probs >= 0.5).astype(int)
            
    if is_multiclass:
        auroc = roc_auc_score(y_sub, y_oof_prob, multi_class='ovr', average='macro')
        ci_low, ci_high = bootstrap_ci(y_sub, y_oof_prob, roc_auc_score, is_multiclass=True)
    else:
        auroc = roc_auc_score(y_sub, y_oof_prob)
        ci_low, ci_high = bootstrap_ci(y_sub, y_oof_prob, roc_auc_score, is_multiclass=False)
        
    f1 = f1_score(y_sub, y_oof_pred, average='macro')
    acc = accuracy_score(y_sub, y_oof_pred)
    bal_acc = balanced_accuracy_score(y_sub, y_oof_pred)
    
    return {
        'auroc': float(auroc),
        'auroc_ci': [ci_low, ci_high],
        'macro_f1': float(f1),
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'y_true': y_sub,
        'y_prob': y_oof_prob
    }

print("\n" + "="*70)
print("RUNNING SUBJECT-INDEPENDENT CLINICAL BENCHMARKS (GROUP 5-FOLD CV)")
print("="*70)

# Task 1: Alcoholism Detection (AUD vs Control)
print("\n--- Benchmark 1: Alcohol Use Disorder (AUD vs. Healthy Control) ---")
X_alc_raw = X_raw[mask_alcohol]
Z_alc_fm = Z_fm[mask_alcohol]
y_alc = (labels_diag[mask_alcohol] == 2).astype(int)
groups_alc = groups_all[mask_alcohol]

res_alc_base = evaluate_baseline_classical(X_alc_raw, y_alc, groups_alc, is_multiclass=False)
res_alc_fm = evaluate_classification_groupkfold(Z_alc_fm, y_alc, groups_alc, is_multiclass=False, task_name="Alcoholism")
print(f"  [Baseline Random Forest]     AUROC: {res_alc_base['auroc']:.4f} (95% CI: {res_alc_base['auroc_ci'][0]:.3f} - {res_alc_base['auroc_ci'][1]:.3f}) | F1: {res_alc_base['macro_f1']:.4f}")
print(f"  [UC-EEG-FM Foundation Model] AUROC: {res_alc_fm['auroc']:.4f} (95% CI: {res_alc_fm['auroc_ci'][0]:.3f} - {res_alc_fm['auroc_ci'][1]:.3f}) | F1: {res_alc_fm['macro_f1']:.4f}")

# Task 2: Dementia Differential Diagnosis (AD vs FTD vs Controls)
print("\n--- Benchmark 2: Dementia Differential Diagnosis (AD vs. FTD vs. CN) ---")
X_dem_raw = X_raw[mask_dementia]
Z_dem_fm = Z_fm[mask_dementia]
y_dem = labels_diag[mask_dementia]
y_dem_3class = np.zeros_like(y_dem)
y_dem_3class[y_dem == 1] = 1 # AD
y_dem_3class[y_dem == 3] = 2 # FTD
groups_dem = groups_all[mask_dementia]

res_dem_base = evaluate_baseline_classical(X_dem_raw, y_dem_3class, groups_dem, is_multiclass=True)
res_dem_fm = evaluate_classification_groupkfold(Z_dem_fm, y_dem_3class, groups_dem, is_multiclass=True, task_name="Dementia")
print(f"  [Baseline Random Forest]     AUROC: {res_dem_base['auroc']:.4f} (95% CI: {res_dem_base['auroc_ci'][0]:.3f} - {res_dem_base['auroc_ci'][1]:.3f}) | F1: {res_dem_base['macro_f1']:.4f}")
print(f"  [UC-EEG-FM Foundation Model] AUROC: {res_dem_fm['auroc']:.4f} (95% CI: {res_dem_fm['auroc_ci'][0]:.3f} - {res_dem_fm['auroc_ci'][1]:.3f}) | F1: {res_dem_fm['macro_f1']:.4f}")

# Task 3: Pediatric Epilepsy (Ictal Seizure vs. Interictal)
print("\n--- Benchmark 3: Pediatric Epilepsy (Ictal Seizure vs. Interictal) ---")
X_epi_raw = X_raw[mask_epilepsy]
Z_epi_fm = Z_fm[mask_epilepsy]
y_epi = y_ictal[mask_epilepsy]
groups_epi = groups_all[mask_epilepsy]

res_epi_base = evaluate_baseline_classical(X_epi_raw, y_epi, groups_epi, is_multiclass=False, balance=True)
res_epi_fm = evaluate_classification_groupkfold(Z_epi_fm, y_epi, groups_epi, is_multiclass=False, task_name="Epilepsy", balance=True)
print(f"  [Baseline Random Forest]     AUROC: {res_epi_base['auroc']:.4f} (95% CI: {res_epi_base['auroc_ci'][0]:.3f} - {res_epi_base['auroc_ci'][1]:.3f}) | F1: {res_epi_base['macro_f1']:.4f}")
print(f"  [UC-EEG-FM Foundation Model] AUROC: {res_epi_fm['auroc']:.4f} (95% CI: {res_epi_fm['auroc_ci'][0]:.3f} - {res_epi_fm['auroc_ci'][1]:.3f}) | F1: {res_epi_fm['macro_f1']:.4f}")

# Benchmark 4: Severity Score Regression
print("\n--- Benchmark 4: Severity Score Regression (MMSE & AUDIT) ---")
meta_df = pd.read_csv(os.path.join(BASE_DIR, "master_cohort_metadata.tsv"), sep='\t')
sub_to_mmse = {row['subject_id']: float(row['severity_score']) for _, row in meta_df.iterrows() if row['disorder_domain'] == 'Neurodegeneration_Dementia' and pd.notnull(row['severity_score'])}
sub_to_audit = {row['subject_id']: float(row['severity_score']) for _, row in meta_df.iterrows() if row['disorder_domain'] == 'Alcoholism_Addiction' and pd.notnull(row['severity_score'])}

y_mmse = np.array([sub_to_mmse[g] for g in groups_dem])
y_audit = np.array([sub_to_audit[g] for g in groups_alc])

gkf = GroupKFold(n_splits=5)
y_oof_mmse = np.zeros(len(y_mmse))
for train_idx, test_idx in gkf.split(Z_dem_fm, y_mmse, groups=groups_dem):
    reg = Ridge(alpha=10.0)
    reg.fit(Z_dem_fm[train_idx], y_mmse[train_idx])
    y_oof_mmse[test_idx] = reg.predict(Z_dem_fm[test_idx])

mmse_mae = mean_absolute_error(y_mmse, y_oof_mmse)
mmse_r2 = r2_score(y_mmse, y_oof_mmse)
print(f"  [MMSE Score Regression (0-30 scale)]  MAE: {mmse_mae:.2f} points | R2: {mmse_r2:.3f}")

y_oof_audit = np.zeros(len(y_audit))
for train_idx, test_idx in gkf.split(Z_alc_fm, y_audit, groups=groups_alc):
    reg = Ridge(alpha=10.0)
    reg.fit(Z_alc_fm[train_idx], y_audit[train_idx])
    y_oof_audit[test_idx] = reg.predict(Z_alc_fm[test_idx])

audit_mae = mean_absolute_error(y_audit, y_oof_audit)
audit_r2 = r2_score(y_audit, y_oof_audit)
print(f"  [AUDIT Score Regression (0-40 scale)] MAE: {audit_mae:.2f} points | R2: {audit_r2:.3f}")

# Plots
print("\nGenerating Figure 3: Clinical ROC Curves...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

# Alcohol ROC
ax1 = axes[0]
fpr_b, tpr_b, _ = roc_curve(y_alc, res_alc_base['y_prob'])
fpr_fm, tpr_fm, _ = roc_curve(y_alc, res_alc_fm['y_prob'])
ax1.plot(fpr_fm, tpr_fm, color='#e67e22', lw=2.5, label=f"UC-EEG-FM (AUROC = {res_alc_fm['auroc']:.3f})")
ax1.plot(fpr_b, tpr_b, color='#7f8c8d', lw=2, linestyle='--', label=f"Baseline RF (AUROC = {res_alc_base['auroc']:.3f})")
ax1.plot([0, 1], [0, 1], 'k:', lw=1)
ax1.set_title("Alcohol Use Disorder (AUD vs. Control)\nSubject-Independent 5-Fold CV", fontsize=12, fontweight='bold')
ax1.set_xlabel("False Positive Rate", fontsize=11)
ax1.set_ylabel("True Positive Rate", fontsize=11)
ax1.legend(loc='lower right', frameon=True, fontsize=10)

# Dementia ROC
ax2 = axes[1]
for c_idx, c_name in [(0, 'Healthy Control'), (1, "Alzheimer's"), (2, 'Frontotemporal Dem.')]:
    fpr_c, tpr_c, _ = roc_curve((y_dem_3class == c_idx).astype(int), res_dem_fm['y_prob'][:, c_idx])
    auc_c = roc_auc_score((y_dem_3class == c_idx).astype(int), res_dem_fm['y_prob'][:, c_idx])
    ax2.plot(fpr_c, tpr_c, lw=2, label=f"{c_name} (AUC = {auc_c:.3f})")
ax2.plot([0, 1], [0, 1], 'k:', lw=1)
ax2.set_title(f"Dementia Staging (3-Class OvR)\nMacro AUROC = {res_dem_fm['auroc']:.3f}", fontsize=12, fontweight='bold')
ax2.set_xlabel("False Positive Rate", fontsize=11)
ax2.set_ylabel("True Positive Rate", fontsize=11)
ax2.legend(loc='lower right', frameon=True, fontsize=10)

# Epilepsy ROC
ax3 = axes[2]
fpr_eb, tpr_eb, _ = roc_curve(y_epi, res_epi_base['y_prob'])
fpr_efm, tpr_efm, _ = roc_curve(y_epi, res_epi_fm['y_prob'])
ax3.plot(fpr_efm, tpr_efm, color='#2980b9', lw=2.5, label=f"UC-EEG-FM (AUROC = {res_epi_fm['auroc']:.3f})")
ax3.plot(fpr_eb, tpr_eb, color='#7f8c8d', lw=2, linestyle='--', label=f"Baseline RF (AUROC = {res_epi_base['auroc']:.3f})")
ax3.plot([0, 1], [0, 1], 'k:', lw=1)
ax3.set_title("Epileptic Seizure Detection (Ictal vs. Interictal)\nTelemetry Monitoring 5-Fold CV", fontsize=12, fontweight='bold')
ax3.set_xlabel("False Positive Rate", fontsize=11)
ax3.set_ylabel("True Positive Rate", fontsize=11)
ax3.legend(loc='lower right', frameon=True, fontsize=10)

plt.tight_layout()
fig3_path = os.path.join(FIG_DIR, "Figure_3_Clinical_ROC_Curves.png")
plt.savefig(fig3_path, dpi=300)
plt.close()

# Figure 4: Confusion Matrices
print("Generating Figure 4: Clinical Confusion Matrices...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

ax1 = axes[0]
cm_alc = res_alc_fm['confusion_matrix']
sns.heatmap(cm_alc, annot=True, fmt='d', cmap='Oranges', ax=ax1, cbar=False,
            xticklabels=['Control', 'AUD'], yticklabels=['Control', 'AUD'], annot_kws={"size": 14, "weight": "bold"})
ax1.set_title(f"Alcoholism Detection (AUDIT >= 8)\nBalanced Acc: {res_alc_fm['balanced_accuracy']*100:.1f}%", fontsize=12, fontweight='bold')
ax1.set_xlabel("Predicted Clinical Label", fontsize=11)
ax1.set_ylabel("Ground-Truth Clinical Label", fontsize=11)

ax2 = axes[1]
cm_dem = res_dem_fm['confusion_matrix']
sns.heatmap(cm_dem, annot=True, fmt='d', cmap='Purples', ax=ax2, cbar=False,
            xticklabels=['Control', "Alzheimer's", 'FTD'], yticklabels=['Control', "Alzheimer's", 'FTD'], annot_kws={"size": 14, "weight": "bold"})
ax2.set_title(f"Dementia Differential Diagnosis\nMacro F1: {res_dem_fm['macro_f1']*100:.1f}%", fontsize=12, fontweight='bold')
ax2.set_xlabel("Predicted Clinical Label", fontsize=11)
ax2.set_ylabel("Ground-Truth Clinical Label", fontsize=11)

plt.tight_layout()
fig4_path = os.path.join(FIG_DIR, "Figure_4_Confusion_Matrices.png")
plt.savefig(fig4_path, dpi=300)
plt.close()

brain_dir = r"C:\Users\Visha\.gemini\antigravity\brain\a487a487-156f-4781-ba99-0b2ec1ba776a"
shutil.copy(fig3_path, os.path.join(brain_dir, "Figure_3_Clinical_ROC_Curves.png"))
shutil.copy(fig4_path, os.path.join(brain_dir, "Figure_4_Confusion_Matrices.png"))

master_results = {
    'task_alcoholism': {
        'baseline_random_forest': res_alc_base,
        'uc_eeg_fm': res_alc_fm,
        'audit_score_regression': {'mae': float(audit_mae), 'r2': float(audit_r2)}
    },
    'task_dementia': {
        'baseline_random_forest': res_dem_base,
        'uc_eeg_fm': res_dem_fm,
        'mmse_score_regression': {'mae': float(mmse_mae), 'r2': float(mmse_r2)}
    },
    'task_epilepsy': {
        'baseline_random_forest': res_epi_base,
        'uc_eeg_fm': res_epi_fm
    }
}

# Clean arrays for JSON export
for task_k in ['task_alcoholism', 'task_dementia', 'task_epilepsy']:
    for m_k in ['baseline_random_forest', 'uc_eeg_fm']:
        if m_k in master_results[task_k]:
            for arr_k in ['y_true', 'y_prob', 'y_pred']:
                master_results[task_k][m_k].pop(arr_k, None)

with open(os.path.join(MODEL_DIR, "phase3_clinical_benchmarks.json"), 'w') as jf:
    json.dump(master_results, jf, indent=2)

print("\n" + "="*70)
print("PHASE 3 CLINICAL BENCHMARKING COMPLETED SUCCESSFULLY.")
print(f"Results saved to: {os.path.join(MODEL_DIR, 'phase3_clinical_benchmarks.json')}")
print("="*70)
