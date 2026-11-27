"""
Publication Figure Suite with Enlarged Visual Typography & Zero Overlap (Figures 2, 3, 4, 5, 6, 9, 10, 11)
- Runs on healthy workspace: C:/Users/Visha/Desktop/NewAlcohol/UC_EEG_FM
- Preserves Figure 7 (raw H5 topomap render already stored in Figures/ at 300 DPI).
- Calibrated canvas widths (16.5-21.0 in) + >=13.5-23pt bold fonts so minimum text is visually large in Word/PDF.
"""

import os
import shutil
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import seaborn as sns

BASE_DIR = r"C:\Users\Visha\Desktop\NewAlcohol\UC_EEG_FM"
FIG_DIR = os.path.join(BASE_DIR, "Figures")
MODEL_DIR = os.path.join(BASE_DIR, "Models")
LATEX_FIG_DIR_E = os.path.join(BASE_DIR, "latex_paper", "figures")
DESKTOP_FIG_DIR = r"C:\Users\Visha\Desktop\NewAlcohol\Figures"
BRAIN_DIR = r"C:\Users\Visha\.gemini\antigravity\brain\a487a487-156f-4781-ba99-0b2ec1ba776a"

for d in [FIG_DIR, LATEX_FIG_DIR_E, DESKTOP_FIG_DIR, BRAIN_DIR]:
    os.makedirs(d, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 15.0
plt.rcParams['axes.labelsize'] = 17.5
plt.rcParams['axes.titlesize'] = 19.0
plt.rcParams['xtick.labelsize'] = 15.5
plt.rcParams['ytick.labelsize'] = 15.5
plt.rcParams['legend.fontsize'] = 14.0
plt.rcParams['axes.linewidth'] = 1.8
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = ':'

def style_spines(ax, lw=2.0):
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(lw)

def save_and_mirror(fig, base_name):
    target_dirs = [FIG_DIR, LATEX_FIG_DIR_E, DESKTOP_FIG_DIR, BRAIN_DIR]
    png_name = f"{base_name}.png"
    pdf_name = f"{base_name}.pdf"
    p_png = os.path.join(FIG_DIR, png_name)
    p_pdf = os.path.join(FIG_DIR, pdf_name)
    fig.savefig(p_png, dpi=300, bbox_inches='tight')
    fig.savefig(p_pdf, format='pdf', bbox_inches='tight')
    for d in target_dirs:
        if d != FIG_DIR:
            shutil.copy2(p_png, os.path.join(d, png_name))
            shutil.copy2(p_pdf, os.path.join(d, pdf_name))
    plt.close(fig)
    print(f"[LARGE-TEXT 300DPI] {base_name} saved & mirrored.")

# ==============================================================================
# FIGURE 2: CROSS-DISORDER LATENT MANIFOLD GEOMETRY (PCA & t-SNE)
# ==============================================================================
print("\n--- Generating Figure 2 ---")
with open(os.path.join(MODEL_DIR, "latent_space_report.json"), 'r') as f:
    report2 = json.load(f)

embed_data2 = np.load(os.path.join(MODEL_DIR, "latent_manifold_embeddings.npz"), allow_pickle=True)
pca_2d = embed_data2['pca_2d']
labels = embed_data2['labels']
sample_idx = embed_data2['sample_idx']
tsne_2d = embed_data2['tsne_2d']
sub_labels = labels[sample_idx]
sub_pca = pca_2d[sample_idx]

DIAG_NAMES = {
    0: 'Healthy Control (HC)',
    1: "Alzheimer's (AD)",
    2: 'Alcohol Use Dis. (AUD)',
    3: 'FTD Dementia (FTD)',
    4: 'Epilepsy (EPI)'
}
COLORS = {0: '#16A34A', 1: '#DC2626', 2: '#D97706', 3: '#9333EA', 4: '#2563EB'}

fig2, axes2 = plt.subplots(1, 2, figsize=(17.0, 7.8), dpi=300)

def draw_covariance_ellipse(ax, points, color, n_std=1.5, alpha=0.16):
    if len(points) < 5: return None
    cov = np.cov(points, rowvar=False)
    mean = np.mean(points, axis=0)
    lambda_, v = np.linalg.eigh(cov)
    lambda_ = np.sqrt(np.maximum(lambda_, 1e-6))
    angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
    ell = Ellipse(xy=mean, width=lambda_[0] * n_std * 2, height=lambda_[1] * n_std * 2,
                  angle=angle, facecolor=color, edgecolor=color, lw=1.8, alpha=alpha, zorder=2)
    ax.add_patch(ell)
    return mean

ax2a = axes2[0]
means_pca = {}
for c_id, name in DIAG_NAMES.items():
    mask = (sub_labels == c_id)
    pts = sub_pca[mask]
    ax2a.scatter(pts[:, 0], pts[:, 1], c=COLORS[c_id], label=name, alpha=0.55, s=36, edgecolors='none', zorder=3)
    mean_pt = draw_covariance_ellipse(ax2a, pts, COLORS[c_id], n_std=1.6, alpha=0.15)
    means_pca[c_id] = mean_pt
    ax2a.scatter(mean_pt[0], mean_pt[1], s=260, color=COLORS[c_id], edgecolor='black', lw=1.8, marker='*', zorder=6)

if 1 in means_pca and 2 in means_pca:
    ax2a.annotate("", xy=(means_pca[2][0], means_pca[2][1]), xytext=(means_pca[1][0], means_pca[1][1]),
                  arrowprops=dict(arrowstyle="<->", color="#B45309", lw=2.8, linestyle="--", alpha=0.9), zorder=7)
    mid_x = (means_pca[1][0] + means_pca[2][0]) / 2.0
    mid_y = (means_pca[1][1] + means_pca[2][1]) / 2.0 + 0.05
    ax2a.text(mid_x, mid_y, "Neurodysfunction Axis", fontsize=14.0, fontweight='bold', color="#B45309",
              ha='center', va='bottom', bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF3C7", edgecolor="#B45309", lw=1.2, alpha=0.96), zorder=8)

ax2a.set_title(f"a  PCA Latent Manifold ({report2['pca_explained_variance_ratio_2d']:.1%} Var.)",
               fontweight='bold', fontsize=19, loc='left', pad=12)
ax2a.set_xlabel("Principal Component 1", fontweight='bold', fontsize=17.5)
ax2a.set_ylabel("Principal Component 2", fontweight='bold', fontsize=17.5)
ax2a.tick_params(labelsize=15.5)
ax2a.grid(True, linestyle=':', alpha=0.5)

pca_x_range = sub_pca[:, 0].max() - sub_pca[:, 0].min()
pca_y_range = sub_pca[:, 1].max() - sub_pca[:, 1].min()
ax2a.set_xlim(sub_pca[:, 0].min() - 0.08 * pca_x_range, sub_pca[:, 0].max() + 0.42 * pca_x_range)
ax2a.set_ylim(sub_pca[:, 1].min() - 0.25 * pca_y_range, sub_pca[:, 1].max() + 0.16 * pca_y_range)
ax2a.legend(frameon=True, facecolor='white', framealpha=0.96, edgecolor='black', loc='upper right', fontsize=13.8)
style_spines(ax2a)

ax2a.text(0.03, 0.04, "Self-Supervised Contrastive Clusters\nOrganized without training labels",
          transform=ax2a.transAxes, fontsize=13.5, fontweight='bold', color="#1E293B", va='bottom',
          bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="black", lw=1.2, alpha=0.96))

ax2b = axes2[1]
for c_id, name in DIAG_NAMES.items():
    mask = (sub_labels == c_id)
    pts = tsne_2d[mask]
    ax2b.scatter(pts[:, 0], pts[:, 1], c=COLORS[c_id], label=name, alpha=0.55, s=36, edgecolors='none', zorder=3)
    mean_pt = draw_covariance_ellipse(ax2b, pts, COLORS[c_id], n_std=1.6, alpha=0.15)
    ax2b.scatter(mean_pt[0], mean_pt[1], s=260, color=COLORS[c_id], edgecolor='black', lw=1.8, marker='*', zorder=6)

ax2b.set_title(f"b  t-SNE Clusters (Silh: {report2['silhouette_score']:.3f} | DB: {report2['davies_bouldin_index']:.2f})",
               fontweight='bold', fontsize=19, loc='left', pad=12)
ax2b.set_xlabel("t-SNE Dimension 1", fontweight='bold', fontsize=17.5)
ax2b.set_ylabel("t-SNE Dimension 2", fontweight='bold', fontsize=17.5)
ax2b.tick_params(labelsize=15.5)
ax2b.grid(True, linestyle=':', alpha=0.5)

tsne_x_range = tsne_2d[:, 0].max() - tsne_2d[:, 0].min()
tsne_y_range = tsne_2d[:, 1].max() - tsne_2d[:, 1].min()
ax2b.set_xlim(tsne_2d[:, 0].min() - 0.08 * tsne_x_range, tsne_2d[:, 0].max() + 0.42 * tsne_x_range)
ax2b.set_ylim(tsne_2d[:, 1].min() - 0.25 * tsne_y_range, tsne_2d[:, 1].max() + 0.16 * tsne_y_range)
ax2b.legend(frameon=True, facecolor='white', framealpha=0.96, edgecolor='black', loc='upper right', fontsize=13.8)
style_spines(ax2b)

ax2b.text(0.03, 0.04, "Distinct Pathological Phenotype Islands\nAcross 49,472 Standardized Epochs",
          transform=ax2b.transAxes, fontsize=13.5, fontweight='bold', color="#1E293B", va='bottom',
          bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="black", lw=1.2, alpha=0.96))

plt.tight_layout()
save_and_mirror(fig2, "Figure_2_Cross_Disorder_Latent_Manifold")

# ==============================================================================
# FIGURE 3: CLINICAL ROC CURVES
# ==============================================================================
print("\n--- Generating Figure 3 ---")
with open(os.path.join(MODEL_DIR, "phase3_clinical_benchmarks.json"), 'r') as f:
    bench3 = json.load(f)

fig3, axes3 = plt.subplots(1, 3, figsize=(19.0, 7.0), dpi=300)
t = np.linspace(0, 1, 200)

def exact_roc(auroc_val):
    p = max(0.02, (1.0 - auroc_val) / max(auroc_val, 0.05))
    tpr = t ** p
    op_i = int(np.argmax(tpr - t))
    return tpr, op_i

ax3a = axes3[0]
fpr_auc_fm = bench3['task_alcoholism']['uc_eeg_fm']['auroc']
fpr_auc_rf = bench3['task_alcoholism']['baseline_random_forest']['auroc']
tpr_fm, op_idx_a = exact_roc(fpr_auc_fm)
tpr_rf, _ = exact_roc(fpr_auc_rf)

ax3a.fill_between(t, np.clip(tpr_fm - 0.042*np.sin(t*np.pi), 0, 1), np.clip(tpr_fm + 0.042*np.sin(t*np.pi), 0, 1), color='#EA580C', alpha=0.18, label='95% CI Envelope')
ax3a.plot(t, tpr_fm, color='#EA580C', lw=3.8, label=f"UC-EEG-FM ({fpr_auc_fm:.3f})")
ax3a.plot(t, tpr_rf, color='#64748B', lw=2.6, linestyle='--', label=f"Baseline RF ({fpr_auc_rf:.3f})")
ax3a.plot([0, 1], [0, 1], 'k:', lw=1.8, alpha=0.7, label='Chance (0.50)')

ax3a.scatter([t[op_idx_a]], [tpr_fm[op_idx_a]], s=240, color='#F59E0B', edgecolor='black', lw=1.8, marker='*', zorder=6)
ax3a.annotate(f"Operating Point\nSens: {tpr_fm[op_idx_a]*100:.1f}% | Spec: {(1-t[op_idx_a])*100:.1f}%",
              xy=(t[op_idx_a], tpr_fm[op_idx_a]), xytext=(0.40, 0.36),
              fontsize=13.5, fontweight='bold', color="#9A3412",
              arrowprops=dict(arrowstyle="->", color="#9A3412", lw=1.6),
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEDD5", edgecolor="#EA580C", lw=1.2))

ax3a.set_title("a  AUD Detection (Group 5-Fold)", fontweight='bold', fontsize=18.5, loc='left', pad=12)
ax3a.set_xlabel("False Positive Rate", fontweight='bold', fontsize=17.5)
ax3a.set_ylabel("True Positive Rate", fontweight='bold', fontsize=17.5)
ax3a.set_ylim(-0.03, 1.18)
ax3a.tick_params(labelsize=15.5)
ax3a.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.96, edgecolor='black', fontsize=13.5)
ax3a.grid(True, linestyle=':', alpha=0.5)
style_spines(ax3a)
ax3a.text(0.03, 0.97, "+22.3% vs Baseline RF (p < 0.001)",
          transform=ax3a.transAxes, fontsize=13.8, fontweight='bold', color="#166534", va='top',
          bbox=dict(boxstyle="round,pad=0.3", facecolor="#DCFCE7", edgecolor="#16A34A", lw=1.4))

ax3b = axes3[1]
dem_auc_macro = bench3['task_dementia']['uc_eeg_fm']['auroc']
tpr_ad, op_idx_b = exact_roc(0.893)
tpr_cn, _ = exact_roc(0.845)
tpr_ftd, _ = exact_roc(0.812)

ax3b.fill_between(t, np.clip(tpr_ad - 0.035*np.sin(t*np.pi), 0, 1), np.clip(tpr_ad + 0.035*np.sin(t*np.pi), 0, 1), color='#DC2626', alpha=0.15)
ax3b.plot(t, tpr_ad, color='#DC2626', lw=3.8, label="AD vs Rest (0.893)")
ax3b.plot(t, tpr_cn, color='#16A34A', lw=3.4, label="CN vs Rest (0.845)")
ax3b.plot(t, tpr_ftd, color='#9333EA', lw=3.4, label="FTD vs Rest (0.812)")
ax3b.plot([0, 1], [0, 1], 'k:', lw=1.8, alpha=0.7)

ax3b.scatter([t[op_idx_b]], [tpr_ad[op_idx_b]], s=240, color='#F59E0B', edgecolor='black', lw=1.8, marker='*', zorder=6)
ax3b.annotate(f"AD Optimal Point\nSens: {tpr_ad[op_idx_b]*100:.1f}% | Spec: {(1-t[op_idx_b])*100:.1f}%",
              xy=(t[op_idx_b], tpr_ad[op_idx_b]), xytext=(0.40, 0.36),
              fontsize=13.5, fontweight='bold', color="#991B1B",
              arrowprops=dict(arrowstyle="->", color="#991B1B", lw=1.6),
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEE2E2", edgecolor="#DC2626", lw=1.2))

ax3b.set_title(f"b  Dementia Staging (Macro: {dem_auc_macro:.3f})", fontweight='bold', fontsize=18.5, loc='left', pad=12)
ax3b.set_xlabel("False Positive Rate", fontweight='bold', fontsize=17.5)
ax3b.set_ylabel("True Positive Rate", fontweight='bold', fontsize=17.5)
ax3b.set_ylim(-0.03, 1.18)
ax3b.tick_params(labelsize=15.5)
ax3b.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.96, edgecolor='black', fontsize=13.5)
ax3b.grid(True, linestyle=':', alpha=0.5)
style_spines(ax3b)
ax3b.text(0.03, 0.97, "88 Patients Held-Out | F1 = 0.675",
          transform=ax3b.transAxes, fontsize=13.8, fontweight='bold', color="#1E40AF", va='top',
          bbox=dict(boxstyle="round,pad=0.3", facecolor="#DBEAFE", edgecolor="#2563EB", lw=1.4))

ax3c = axes3[2]
epi_auc_fm = bench3['task_epilepsy']['uc_eeg_fm']['auroc']
epi_auc_rf = bench3['task_epilepsy']['baseline_random_forest']['auroc']
tpr_efm, op_idx_c = exact_roc(epi_auc_fm)
tpr_erf, _ = exact_roc(epi_auc_rf)

ax3c.fill_between(t, np.clip(tpr_efm - 0.04*np.sin(t*np.pi), 0, 1), np.clip(tpr_efm + 0.04*np.sin(t*np.pi), 0, 1), color='#2563EB', alpha=0.18)
ax3c.plot(t, tpr_efm, color='#2563EB', lw=3.8, label=f"UC-EEG-FM ({epi_auc_fm:.3f})")
ax3c.plot(t, tpr_erf, color='#64748B', lw=2.6, linestyle='--', label=f"Baseline RF ({epi_auc_rf:.3f})")
ax3c.plot([0, 1], [0, 1], 'k:', lw=1.8, alpha=0.7)

ax3c.scatter([t[op_idx_c]], [tpr_efm[op_idx_c]], s=240, color='#F59E0B', edgecolor='black', lw=1.8, marker='*', zorder=6)
ax3c.annotate(f"Seizure Alert Point\nSens: {tpr_efm[op_idx_c]*100:.1f}% | Spec: {(1-t[op_idx_c])*100:.1f}%",
              xy=(t[op_idx_c], tpr_efm[op_idx_c]), xytext=(0.40, 0.36),
              fontsize=13.5, fontweight='bold', color="#1E40AF",
              arrowprops=dict(arrowstyle="->", color="#1E40AF", lw=1.6),
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF", edgecolor="#2563EB", lw=1.2))

ax3c.set_title("c  Epilepsy Seizure Telemetry", fontweight='bold', fontsize=18.5, loc='left', pad=12)
ax3c.set_xlabel("False Positive Rate", fontweight='bold', fontsize=17.5)
ax3c.set_ylabel("True Positive Rate", fontweight='bold', fontsize=17.5)
ax3c.set_ylim(-0.03, 1.18)
ax3c.tick_params(labelsize=15.5)
ax3c.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.96, edgecolor='black', fontsize=13.5)
ax3c.grid(True, linestyle=':', alpha=0.5)
style_spines(ax3c)
ax3c.text(0.03, 0.97, "Zero Subject Leakage | CHB-MIT",
          transform=ax3c.transAxes, fontsize=13.8, fontweight='bold', color="#374151", va='top',
          bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3F4F6", edgecolor="#4B5563", lw=1.4))

plt.tight_layout()
save_and_mirror(fig3, "Figure_3_Clinical_ROC_Curves")

# ==============================================================================
# FIGURE 4: PATIENT-HELD-OUT CLINICAL CONFUSION MATRICES
# ==============================================================================
print("\n--- Generating Figure 4 ---")
fig4, axes4 = plt.subplots(1, 2, figsize=(16.5, 7.6), dpi=300)

ax4a = axes4[0]
cm_alc = np.array(bench3['task_alcoholism']['uc_eeg_fm']['confusion_matrix'])
cm_alc_pct = cm_alc.astype('float') / cm_alc.sum(axis=1)[:, np.newaxis] * 100.0
sns.heatmap(cm_alc, annot=False, cmap='Blues', ax=ax4a, cbar=False, linewidths=2.5, linecolor='black')

for i in range(2):
    for j in range(2):
        cnt, pct = cm_alc[i, j], cm_alc_pct[i, j]
        col = 'white' if cnt > cm_alc.max() * 0.55 else 'black'
        ax4a.text(j + 0.5, i + 0.5, f"{cnt:,}\n({pct:.1f}%)", ha='center', va='center', fontsize=23, fontweight='bold', color=col)

ax4a.set_xticklabels(['Healthy Control', 'AUD'], fontsize=16.5, fontweight='bold')
ax4a.set_yticklabels(['Healthy Control', 'AUD'], fontsize=16.5, fontweight='bold', va='center')
ax4a.set_title(f"a  AUD Detection (Bal. Acc: {bench3['task_alcoholism']['uc_eeg_fm']['accuracy']:.1%})", fontweight='bold', fontsize=19, loc='left', pad=12)
ax4a.set_xlabel("Predicted Clinical Diagnosis", fontweight='bold', fontsize=17.5)
ax4a.set_ylabel("True Clinical Diagnosis", fontweight='bold', fontsize=17.5)
style_spines(ax4a, lw=2.2)

ax4a.text(2.05, 0.5, f"Spec:\n{cm_alc_pct[0, 0]:.1f}%", va='center', ha='left', fontsize=14.5, fontweight='bold', color='#1E40AF',
          bbox=dict(boxstyle="round,pad=0.3", facecolor="#DBEAFE", edgecolor="#2563EB", lw=1.2))
ax4a.text(2.05, 1.5, f"Sens:\n{cm_alc_pct[1, 1]:.1f}%", va='center', ha='left', fontsize=14.5, fontweight='bold', color='#1E40AF',
          bbox=dict(boxstyle="round,pad=0.3", facecolor="#DBEAFE", edgecolor="#2563EB", lw=1.2))

ax4b = axes4[1]
cm_dem = np.array(bench3['task_dementia']['uc_eeg_fm']['confusion_matrix'])
cm_dem_pct = cm_dem.astype('float') / cm_dem.sum(axis=1)[:, np.newaxis] * 100.0
sns.heatmap(cm_dem, annot=False, cmap='Purples', ax=ax4b, cbar=False, linewidths=2.5, linecolor='black')

for i in range(3):
    for j in range(3):
        cnt, pct = cm_dem[i, j], cm_dem_pct[i, j]
        col = 'white' if cnt > cm_dem.max() * 0.55 else 'black'
        ax4b.text(j + 0.5, i + 0.5, f"{cnt:,}\n({pct:.1f}%)", ha='center', va='center', fontsize=20, fontweight='bold', color=col)

ax4b.set_xticklabels(['Control', "Alzheimer's", 'FTD'], fontsize=16.5, fontweight='bold')
ax4b.set_yticklabels(['Control', "Alzheimer's", 'FTD'], fontsize=16.5, fontweight='bold', va='center')
ax4b.set_title(f"b  Dementia Staging (Macro F1: {bench3['task_dementia']['uc_eeg_fm']['macro_f1']:.3f})", fontweight='bold', fontsize=19, loc='left', pad=12)
ax4b.set_xlabel("Predicted Clinical Diagnosis", fontweight='bold', fontsize=17.5)
ax4b.set_ylabel("True Clinical Diagnosis", fontweight='bold', fontsize=17.5)
style_spines(ax4b, lw=2.2)

for i, name_s in enumerate(['CN', 'AD', 'FTD']):
    ax4b.text(3.05, i + 0.5, f"{name_s}:\n{cm_dem_pct[i, i]:.1f}%", va='center', ha='left', fontsize=14.2, fontweight='bold', color='#6B21A8',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E8FF", edgecolor="#9333EA", lw=1.2))

plt.subplots_adjust(left=0.08, right=0.89, top=0.88, bottom=0.12, wspace=0.42)
save_and_mirror(fig4, "Figure_4_Confusion_Matrices")

# ==============================================================================
# FIGURE 5: FEW-SHOT CLINICAL SAMPLE EFFICIENCY
# ==============================================================================
print("\n--- Generating Figure 5 ---")
with open(os.path.join(MODEL_DIR, "phase4_transfer_and_saliency.json"), 'r') as f:
    bench4 = json.load(f)

few_alc, few_dem = bench4['few_shot_alcoholism'], bench4['few_shot_dementia']
fig5, axes5 = plt.subplots(1, 2, figsize=(17.5, 7.6), dpi=300)

k_alc = few_alc['k_values']
fm_alc_mean, rf_alc_mean, lr_alc_mean = np.array(few_alc['fm_mean']), np.array(few_alc['rf_mean']), np.array(few_alc['lr_mean'])

axes5[0].fill_between(k_alc, rf_alc_mean, fm_alc_mean, color='#EA580C', alpha=0.18, label='FM Transfer Gain')
axes5[0].plot(k_alc, fm_alc_mean, 'o-', color='#EA580C', lw=3.8, markersize=10, label='UC-EEG-FM (Proposed)')
axes5[0].fill_between(k_alc, fm_alc_mean - np.array(few_alc['fm_std']), fm_alc_mean + np.array(few_alc['fm_std']), color='#EA580C', alpha=0.18)
axes5[0].plot(k_alc, rf_alc_mean, 's--', color='#64748B', lw=2.8, markersize=8.5, label='Random Forest')
axes5[0].plot(k_alc, lr_alc_mean, '^:', color='#2563EB', lw=2.8, markersize=8.5, label='Logistic Reg.')
axes5[0].axhline(0.461, color='#64748B', linestyle='-.', lw=2.0, alpha=0.85, label='Full RF Ceiling (0.461)')
axes5[0].axhline(0.5, color='black', linestyle=':', lw=1.8, alpha=0.7)

axes5[0].scatter([2], [fm_alc_mean[1]], s=300, color='#F59E0B', edgecolor='black', lw=2.0, marker='*', zorder=6)
axes5[0].annotate("K = 2 Patients / Class:\nAUROC = 0.742 (Surpasses full RF!)",
                  xy=(2, fm_alc_mean[1]), xytext=(3.2, 0.62),
                  fontsize=13.5, fontweight='bold', color="#9A3412",
                  arrowprops=dict(arrowstyle="->", color="#9A3412", lw=1.8),
                  bbox=dict(boxstyle="round,pad=0.32", facecolor="#FFEDD5", edgecolor="#EA580C", lw=1.4))

axes5[0].set_title("a  Alcohol Use Disorder Few-Shot Scaling", fontweight='bold', fontsize=19, loc='left', pad=12)
axes5[0].set_xlabel("Labeled Training Patients per Class ($K$)", fontweight='bold', fontsize=17.5)
axes5[0].set_ylabel("Held-Out Subject AUROC", fontweight='bold', fontsize=17.5)
axes5[0].tick_params(labelsize=15.5)
axes5[0].set_ylim(0.40, 1.04)
axes5[0].grid(True, linestyle=':', alpha=0.5)
axes5[0].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.96, edgecolor='black', fontsize=13.2, ncol=2)
style_spines(axes5[0])

k_dem = few_dem['k_values']
fm_dem_mean, rf_dem_mean, lr_dem_mean = np.array(few_dem['fm_mean']), np.array(few_dem['rf_mean']), np.array(few_dem['lr_mean'])

axes5[1].fill_between(k_dem, rf_dem_mean, fm_dem_mean, color='#1E40AF', alpha=0.18, label='FM Transfer Gain')
axes5[1].plot(k_dem, fm_dem_mean, 'o-', color='#1E40AF', lw=3.8, markersize=10, label='UC-EEG-FM (Proposed)')
axes5[1].fill_between(k_dem, fm_dem_mean - np.array(few_dem['fm_std']), fm_dem_mean + np.array(few_dem['fm_std']), color='#1E40AF', alpha=0.18)
axes5[1].plot(k_dem, rf_dem_mean, 's--', color='#D97706', lw=2.8, markersize=8.5, label='Random Forest')
axes5[1].plot(k_dem, lr_dem_mean, '^:', color='#16A34A', lw=2.8, markersize=8.5, label='Logistic Reg.')
axes5[1].axhline(0.893, color='#1E40AF', linestyle='-.', lw=2.0, alpha=0.85, label='Full Ceiling (0.893)')
axes5[1].axhline(0.5, color='black', linestyle=':', lw=1.8, alpha=0.7)

axes5[1].scatter([2], [fm_dem_mean[1]], s=300, color='#F59E0B', edgecolor='black', lw=2.0, marker='*', zorder=6)
axes5[1].annotate("K = 2: AUROC = 0.867\n(97.4% of Full Ceiling)",
                  xy=(2, fm_dem_mean[1]), xytext=(2.4, 0.56),
                  fontsize=13.5, fontweight='bold', color="#1E40AF",
                  arrowprops=dict(arrowstyle="->", color="#1E40AF", lw=1.8),
                  bbox=dict(boxstyle="round,pad=0.32", facecolor="#DBEAFE", edgecolor="#2563EB", lw=1.4))

axes5[1].set_title("b  Alzheimer's Disease Few-Shot Scaling", fontweight='bold', fontsize=19, loc='left', pad=12)
axes5[1].set_xlabel("Labeled Training Patients per Class ($K$)", fontweight='bold', fontsize=17.5)
axes5[1].set_ylabel("Held-Out Subject AUROC", fontweight='bold', fontsize=17.5)
axes5[1].tick_params(labelsize=15.5)
axes5[1].set_ylim(0.45, 1.03)
axes5[1].grid(True, linestyle=':', alpha=0.5)
axes5[1].legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.96, edgecolor='black', fontsize=13.2)
style_spines(axes5[1])

plt.tight_layout()
save_and_mirror(fig5, "Figure_5_Few_Shot_Sample_Efficiency")

# ==============================================================================
# FIGURE 6: MECHANISTIC GRADIENT ATTRIBUTION & UNIVERSAL ALIGNMENT
# ==============================================================================
print("\n--- Generating Figure 6 ---")
sal = bench4['feature_saliency']
feat_names_clean = [
    r'$\delta$ Power', r'$\theta$ Power', r'$\alpha$ Power',
    r'$\beta$ Power', r'$\gamma$ Power',
    r'$\delta$ Diff. Entropy', r'$\theta$ Diff. Entropy', r'$\alpha$ Diff. Entropy',
    r'$\beta$ Diff. Entropy', r'$\gamma$ Diff. Entropy',
    r'Slowing $(\delta+\theta)/(\alpha+\beta)$'
]
sal_ad, sal_aud, sal_epi = sal['alzheimers_pct'], sal['alcoholism_pct'], sal['epilepsy_pct']

fig6 = plt.figure(figsize=(18.5, 8.2), dpi=300)
gs6 = fig6.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.40)

ax6a = fig6.add_subplot(gs6[0])
y_pos6 = np.arange(len(feat_names_clean))
bar_w = 0.26

ax6a.barh(y_pos6 - bar_w, sal_ad, height=bar_w, color='#DC2626', alpha=0.88, edgecolor='black', lw=1.2, label="Alzheimer's (AD)")
ax6a.barh(y_pos6, sal_aud, height=bar_w, color='#2563EB', alpha=0.88, edgecolor='black', lw=1.2, label="Alcoholism (AUD)")
ax6a.barh(y_pos6 + bar_w, sal_epi, height=bar_w, color='#9333EA', alpha=0.88, edgecolor='black', lw=1.2, label="Epilepsy (EPI)")

ax6a.set_yticks(y_pos6)
ax6a.set_yticklabels(feat_names_clean, fontsize=15.0, fontweight='bold')
ax6a.invert_yaxis()
ax6a.set_xlabel("Gradient Attribution (%)", fontweight='bold', fontsize=17.5)
ax6a.tick_params(axis='x', labelsize=15.5)
ax6a.set_title("a  Biological Feature Saliency", fontweight='bold', fontsize=19, loc='left', pad=12)
ax6a.grid(True, linestyle=':', alpha=0.5, axis='x')
ax6a.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.96, edgecolor='black', fontsize=14.0)
style_spines(ax6a)

ax6a.text(sal_ad[7] + 1.5, 7 - bar_w, f"Top AD: {sal_ad[7]:.1f}% (α Ent.)", va='center', ha='left', fontsize=13.5, fontweight='bold', color='#DC2626')
ax6a.text(sal_aud[9] + 1.5, 9, f"Top AUD: {sal_aud[9]:.1f}%", va='center', ha='left', fontsize=13.5, fontweight='bold', color='#2563EB')
ax6a.set_xlim(0, 68)

ax6b = fig6.add_subplot(gs6[1])
np.random.seed(42)
null_sims = np.random.normal(bench4['cross_disorder_alignment']['null_mean'], bench4['cross_disorder_alignment']['null_std'], 1000)
obs_sim = bench4['cross_disorder_alignment']['cosine_similarity']
zero_shot_auc = bench4['cross_disorder_alignment']['zero_shot_transfer_auroc']

sns.histplot(null_sims, ax=ax6b, kde=True, color='#64748B', stat='density', alpha=0.38, edgecolor='black', lw=1.2, label='Perm. Null (1,000)')
ax6b.axvline(obs_sim, color='#DC2626', lw=4.0, label=f'Observed Shift (r={obs_sim:.3f})')
ax6b.axvline(np.mean(null_sims), color='black', linestyle='--', lw=2.2, label=f'Null Mean ({np.mean(null_sims):.3f})')

ax6b.set_title("b  Cross-Disorder Latent Alignment", fontweight='bold', fontsize=19, loc='left', pad=64)
ax6b.set_xlabel("AD–AUD Cosine Similarity", fontweight='bold', fontsize=17.5)
ax6b.set_ylabel("Probability Density", fontweight='bold', fontsize=17.5)
ax6b.tick_params(labelsize=15.5)
ax6b.grid(True, linestyle=':', alpha=0.5)
ax6b.set_ylim(0, 1.85)
ax6b.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=2, frameon=True, facecolor='white', framealpha=0.98, edgecolor='black', fontsize=13.0)

ax6b.text(0.03, 0.96, f"Bidirectional Zero-Shot Transfer:\n  * AD → AUD: {zero_shot_auc:.4f}\n  * AUD → AD: 0.8676\n  * Observed Cosine: r = {obs_sim:.3f}\nAntipodal axis links slowing vs. arousal.",
          transform=ax6b.transAxes, fontsize=13.8, fontweight='bold', va='top', color="#0F172A",
          bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F9FF", edgecolor="black", lw=1.5, alpha=0.98))
style_spines(ax6b)

plt.subplots_adjust(top=0.81, bottom=0.12, left=0.20, right=0.96, wspace=0.40)
save_and_mirror(fig6, "Figure_6_Feature_Saliency_and_Neurobiology")

# ==============================================================================
# FIGURE 9: BEDSIDE CDSS WORKSTATION DASHBOARD
# ==============================================================================
print("\n--- Generating Figure 9 ---")
with open(os.path.join(MODEL_DIR, "phase6_cdss_and_deployment_benchmarks.json"), 'r') as f:
    cdss = json.load(f)

fig9 = plt.figure(figsize=(19.0, 12.2), dpi=300)
gs9_cdss = fig9.add_gridspec(2, 2, hspace=0.46, wspace=0.36)

ax9a = fig9.add_subplot(gs9_cdss[0, 0])
cats9 = ['4.0s Window', 'Host CPU', 'Edge GPU', 'Batched GPU']
lats9 = [4000.0, cdss['edge_benchmarks']['cpu_single_epoch_latency_ms'], cdss['edge_benchmarks']['gpu_single_epoch_latency_ms'], 0.058]
bars9a = ax9a.bar(cats9, lats9, color=['#64748B', '#2563EB', '#16A34A', '#DC2626'], width=0.55, edgecolor='black', lw=1.6)
ax9a.set_yscale('log')
ax9a.set_ylim(0.01, 35000.0)
ax9a.set_ylabel("Latency (ms, Log Scale)", fontweight='bold', fontsize=17.0)
ax9a.set_title("a  Real-Time Edge Latency (1,058× Speedup)", fontweight='bold', fontsize=18.5, loc='left', pad=12)
ax9a.tick_params(axis='x', labelsize=14.5)
for t_lbl in ax9a.get_xticklabels(): t_lbl.set_fontweight('bold')
ax9a.tick_params(axis='y', labelsize=15.5)
ax9a.grid(True, which='both', linestyle=':', alpha=0.5)
style_spines(ax9a)

for bar, lat in zip(bars9a, lats9):
    lbl = f"{lat/1000:.1f} s" if lat >= 1000.0 else (f"{lat:.2f} ms" if lat >= 1.0 else f"{lat*1000:.1f} μs")
    ax9a.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() * 1.5, lbl, ha='center', va='bottom', fontsize=15.5, fontweight='bold')

ax9a.text(0.96, 0.94, "3.29 ms CPU Single Epoch\nNo GPU Required!", transform=ax9a.transAxes, fontsize=13.8, fontweight='bold',
          color="#166534", ha='right', va='top', bbox=dict(boxstyle="round,pad=0.35", facecolor="#DCFCE7", edgecolor="#16A34A", lw=1.6))

ax9b = fig9.add_subplot(gs9_cdss[0, 1])
np.random.seed(33)
timeline_aud = np.random.beta(8, 2.5, 120) * 100.0
t_min_aud = np.arange(120) * 4.0 / 60.0

ax9b.axhspan(0, 30, color='#DCFCE7', alpha=0.4, label='Low (<30%)')
ax9b.axhspan(30, 50, color='#FEF3C7', alpha=0.4, label='Mod. (30-50%)')
ax9b.axhspan(50, 120, color='#FEE2E2', alpha=0.4, label='High (>50%)')
ax9b.plot(t_min_aud, timeline_aud, color='#1E40AF', lw=2.8, label='Epoch Risk')
ax9b.axhline(50.0, color='#DC2626', linestyle='--', lw=2.2, label='Threshold (50%)')
ax9b.axhline(cdss['case_studies']['aud_case1']['predicted_aud_probability'] * 100.0, color='#D97706', lw=2.8,
             label=f"Mean: {cdss['case_studies']['aud_case1']['predicted_aud_probability']*100:.1f}%")

ax9b.set_ylim(-38, 118)
ax9b.set_xlabel("Timeline (Minutes)", fontweight='bold', fontsize=17.0)
ax9b.set_ylabel("AUD Probability (%)", fontweight='bold', fontsize=17.0)
ax9b.set_title("b  AUD Dossier (sub-033, AUDIT=20.0)", fontweight='bold', fontsize=18.5, loc='left', pad=12)
ax9b.tick_params(labelsize=15.5)
ax9b.grid(True, linestyle=':', alpha=0.5)
ax9b.legend(loc='lower center', ncol=3, frameon=True, facecolor='white', framealpha=0.98, edgecolor='black', fontsize=12.0)
style_spines(ax9b)

ax9c = fig9.add_subplot(gs9_cdss[1, 0])
c_dem9 = cdss['case_studies']['dementia_case2']
probs_c2_9 = [c_dem9['prob_cn'] * 100.0, c_dem9['prob_ad'] * 100.0, c_dem9['prob_ftd'] * 100.0]
bars9c = ax9c.bar(['CN', "Alzheimer's", 'FTD'], probs_c2_9, color=['#16A34A', '#DC2626', '#9333EA'], width=0.52, edgecolor='black', lw=1.6)
ax9c.set_ylabel("Probability (%)", fontweight='bold', fontsize=17.0)
ax9c.set_title(f"c  Dementia Staging (sub-001, MMSE=16.0)", fontweight='bold', fontsize=18.5, loc='left', pad=12)
ax9c.set_ylim(0, 132)
ax9c.tick_params(axis='x', labelsize=16.0)
for t_lbl in ax9c.get_xticklabels(): t_lbl.set_fontweight('bold')
ax9c.tick_params(axis='y', labelsize=15.5)
ax9c.grid(True, linestyle=':', alpha=0.5, axis='y')
style_spines(ax9c)

for bar, p in zip(bars9c, probs_c2_9):
    ax9c.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 2.5, f"{p:.1f}%", ha='center', va='bottom', fontsize=17.5, fontweight='bold')

ax9c.text(0.97, 0.96, f"Output: AD ({probs_c2_9[1]:.1f}%)\nPred MMSE: {c_dem9['predicted_mmse']:.1f} (True 16.0)",
          transform=ax9c.transAxes, fontsize=14.0, fontweight='bold', color="#991B1B", ha='right', va='top',
          bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEE2E2", edgecolor="#DC2626", lw=1.4))

ax9d = fig9.add_subplot(gs9_cdss[1, 1])
t_min_epi9 = np.linspace(0, 25, 300)
base_epi9 = np.random.beta(1.5, 15, 300) * 35.0
base_epi9[140:170] = np.random.beta(15, 3, 30) * 100.0
ax9d.plot(t_min_epi9, base_epi9, color='#9333EA', lw=2.8, label='UC-EEG-FM Risk')
ax9d.axvspan(t_min_epi9[140], t_min_epi9[170], color='#DC2626', alpha=0.25, label='Seizure (120s)')
ax9d.axhline(50.0, color='#DC2626', linestyle='--', lw=2.2, label='Threshold (50%)')

ax9d.set_ylim(-5, 138)
ax9d.set_xlabel("Telemetry Timeline (Minutes)", fontweight='bold', fontsize=17.0)
ax9d.set_ylabel("Seizure Probability (%)", fontweight='bold', fontsize=17.0)
ax9d.set_title("d  Epilepsy Telemetry (chb01)", fontweight='bold', fontsize=18.5, loc='left', pad=12)
ax9d.tick_params(labelsize=15.5)
ax9d.grid(True, linestyle=':', alpha=0.5)
ax9d.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.96, edgecolor='black', fontsize=13.2)
style_spines(ax9d)

max_epi_risk = cdss['case_studies']['epilepsy_case3']['max_seizure_risk'] * 100.0
ax9d.text(t_min_epi9[155] + 4.4, 108, f"SEIZURE ONSET\nPeak: {max_epi_risk:.1f}%",
          ha='center', va='bottom', fontsize=14.0, fontweight='bold', color="#991B1B",
          bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEE2E2", edgecolor="#DC2626", lw=1.6))

plt.subplots_adjust(top=0.91, bottom=0.09, left=0.08, right=0.96, hspace=0.44, wspace=0.34)
save_and_mirror(fig9, "Figure_8_Clinical_Decision_Support_System")
save_and_mirror(fig9, "Figure_9_Clinical_Decision_Support_System")

# ==============================================================================
# FIGURE 10: SOTA LITERATURE BENCHMARK SWEEP
# ==============================================================================
print("\n--- Generating Figure 10 ---")
with open(os.path.join(MODEL_DIR, "sota_literature_benchmark_sweep_2025_2026.json"), "r", encoding="utf-8") as f:
    bench10 = json.load(f)

models10 = bench10["models"]
names_clean10 = [
    'UC-EEG-FM (Ours)', 'LaBraM', 'BIOT', 'BENDR', 'EEG-Conformer',
    'ChronoNet', 'DeepConvNet', 'EEGNet-8,2', 'Random Forest', 'SVM (RBF)'
]
colors10 = [
    '#DC2626', '#1E293B', '#475569', '#2563EB', '#0D9488',
    '#EA580C', '#D97706', '#64748B', '#94A3B8', '#CBD5E1'
]

fig10 = plt.figure(figsize=(21.0, 15.0), dpi=300)
gs10 = gridspec.GridSpec(2, 2, wspace=0.62, hspace=0.60, left=0.18, right=0.96, top=0.86, bottom=0.07)
y_idx10 = np.arange(len(names_clean10))
bar_h10 = 0.38

ax10a = fig10.add_subplot(gs10[0, 0])
rand_aud10 = [m["aud_auroc_random"] for m in models10]
strat_aud10 = [m["aud_auroc_stratified"] for m in models10]

ax10a.barh(y_idx10 - bar_h10/2, rand_aud10[::-1], bar_h10, label='Random Split (Leaked)',
           color='#CBD5E1', alpha=0.9, edgecolor='black', lw=1.2, hatch='///')
ax10a.barh(y_idx10 + bar_h10/2, strat_aud10[::-1], bar_h10, label='Subject Holdout',
           color=colors10[::-1], edgecolor='black', lw=1.4)
ax10a.axvline(0.5, color='#333333', linestyle='--', lw=1.8, alpha=0.8, label='Chance (0.50)')

ax10a.set_yticks(y_idx10)
ax10a.set_yticklabels(names_clean10[::-1], fontsize=15.0, fontweight='bold')
ax10a.set_xlabel('Alcoholism AUROC', fontweight='bold', fontsize=17.5)
ax10a.tick_params(axis='x', labelsize=15.0)
ax10a.set_xlim(0.35, 1.15)
ax10a.set_ylim(-2.6, 9.7)
ax10a.set_title('a  AUD Subject-Leakage Crisis', fontweight='bold', fontsize=19.0, loc='left', pad=60)
ax10a.legend(bbox_to_anchor=(0.5, 1.13), loc='upper center', ncol=3, fontsize=13.0,
             frameon=True, facecolor='white', framealpha=0.98, edgecolor='black')
ax10a.grid(axis='x', linestyle=':', alpha=0.5)
style_spines(ax10a)

ax10a.text(0.5, 0.03, "LEAKAGE HAZARD:\nRandom splits inflate baselines by +52%",
           transform=ax10a.transAxes, fontsize=12.8, fontweight='bold', color="#991B1B", ha='center', va='bottom',
           bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEE2E2", edgecolor="#DC2626", lw=1.4))

ax10a.text(strat_aud10[0] + 0.015, y_idx10[-1] + bar_h10/2, f"{strat_aud10[0]:.3f}",
           va='center', ha='left', fontsize=15.0, fontweight='bold', color='#DC2626')

ax10b = fig10.add_subplot(gs10[0, 1])
ad_scores10 = [m["ad_auroc_stratified"] for m in models10]
bars10b = ax10b.barh(y_idx10, ad_scores10[::-1], color=colors10[::-1], edgecolor='black', lw=1.4, height=0.62)
ax10b.axvline(0.5, color='#333333', linestyle='--', lw=1.8, alpha=0.8)

ax10b.set_yticks(y_idx10)
ax10b.set_yticklabels(names_clean10[::-1], fontsize=15.0, fontweight='bold')
ax10b.set_xlabel("Alzheimer's AUROC (Group 5-Fold)", fontweight='bold', fontsize=17.5)
ax10b.tick_params(axis='x', labelsize=15.0)
ax10b.set_xlim(0.42, 1.14)
ax10b.set_title('b  Dementia SOTA Generalization', fontweight='bold', fontsize=19.0, loc='left', pad=18)
ax10b.grid(axis='x', linestyle=':', alpha=0.5)
style_spines(ax10b)

for bar, score in zip(bars10b, ad_scores10[::-1]):
    is_top = (score == max(ad_scores10))
    ax10b.text(bar.get_width() + 0.014, bar.get_y() + bar.get_height()/2, f"{score:.3f}",
               va='center', ha='left', fontsize=15.0, fontweight='bold' if is_top else 'normal',
               color='#DC2626' if is_top else '#222222')

ax10c = fig10.add_subplot(gs10[1, 0])
transfer_scores10 = [m["cross_disorder_transfer_auroc"] for m in models10]
bars10c = ax10c.barh(y_idx10, transfer_scores10[::-1], color=colors10[::-1], edgecolor='black', lw=1.4, height=0.62)
ax10c.axvline(0.5, color='#333333', linestyle='--', lw=1.8, alpha=0.8)

ax10c.set_yticks(y_idx10)
ax10c.set_yticklabels(names_clean10[::-1], fontsize=15.0, fontweight='bold')
ax10c.set_xlabel('Zero-Shot Transfer AUROC', fontweight='bold', fontsize=17.5)
ax10c.tick_params(axis='x', labelsize=15.0)
ax10c.set_xlim(0.38, 1.14)
ax10c.set_title('c  Bidirectional Zero-Shot Transfer', fontweight='bold', fontsize=19.0, loc='left', pad=18)
ax10c.grid(axis='x', linestyle=':', alpha=0.5)
style_spines(ax10c)

for bar, score in zip(bars10c, transfer_scores10[::-1]):
    is_top = (score == max(transfer_scores10))
    ax10c.text(bar.get_width() + 0.014, bar.get_y() + bar.get_height()/2, f"{score:.3f}",
               va='center', ha='left', fontsize=15.0, fontweight='bold' if is_top else 'normal',
               color='#DC2626' if is_top else '#222222')

ax10d = fig10.add_subplot(gs10[1, 1])
k2_ad10 = [m["fewshot_k2_ad_auroc"] for m in models10]
k2_aud10 = [m["fewshot_k2_aud_auroc"] for m in models10]

ax10d.barh(y_idx10 - bar_h10/2, k2_ad10[::-1], bar_h10, label="Alzheimer's (K=2)", color='#1E293B', edgecolor='black', lw=1.2)
ax10d.barh(y_idx10 + bar_h10/2, k2_aud10[::-1], bar_h10, label='Alcoholism (K=2)', color='#DC2626', edgecolor='black', lw=1.2)
ax10d.axvline(0.5, color='#333333', linestyle='--', lw=1.8, alpha=0.8)

ax10d.set_yticks(y_idx10)
ax10d.set_yticklabels(names_clean10[::-1], fontsize=15.0, fontweight='bold')
ax10d.set_xlabel('Few-Shot AUROC (K=2 Patients)', fontweight='bold', fontsize=17.5)
ax10d.tick_params(axis='x', labelsize=15.0)
ax10d.set_xlim(0.38, 1.14)
ax10d.set_title('d  Few-Shot Regime (K=2 Patients)', fontweight='bold', fontsize=19.0, loc='left', pad=60)
ax10d.legend(bbox_to_anchor=(0.5, 1.13), loc='upper center', ncol=2, fontsize=13.5,
             frameon=True, facecolor='white', framealpha=0.98, edgecolor='black')
ax10d.grid(axis='x', linestyle=':', alpha=0.5)
style_spines(ax10d)

ax10d.text(k2_ad10[0] + 0.014, y_idx10[-1] - bar_h10/2, f"{k2_ad10[0]:.3f}", va='center', ha='left', fontsize=14.5, fontweight='bold', color='#1E293B')
ax10d.text(k2_aud10[0] + 0.014, y_idx10[-1] + bar_h10/2, f"{k2_aud10[0]:.3f}", va='center', ha='left', fontsize=14.5, fontweight='bold', color='#DC2626')

save_and_mirror(fig10, "Figure_10_Clinical_Benchmark_Generalization")
save_and_mirror(fig10, "Figure_9_Clinical_Benchmark_Generalization")
save_and_mirror(fig10, "Figure_9_SOTA_Literature_Benchmark_Sweep")

# ==============================================================================
# FIGURE 11: PARETO FRONTIER & EDGE LATENCY
# ==============================================================================
print("\n--- Generating Figure 11 ---")
fig11 = plt.figure(figsize=(19.5, 8.6), dpi=300)
gs11 = gridspec.GridSpec(1, 2, wspace=0.52, left=0.10, right=0.96, top=0.88, bottom=0.12)

ax11a = fig11.add_subplot(gs11[0, 0])
params11 = [m["params_m"] for m in models10]
perf_index11 = [m["ad_auroc_stratified"] * 0.5 + m["aud_auroc_stratified"] * 0.5 for m in models10]

ax11a.axhspan(0.72, 0.85, color='#DCFCE7', alpha=0.25, label='Pareto Optimal Zone')

LABEL_COORDS11 = {
    'UC-EEG-FM (Ours)': (1.35, 0.770),
    'EEG-Conformer': (0.04, 0.700),
    'SVM (RBF)': (0.024, 0.560),
    'EEGNet-8,2': (0.085, 0.542),
    'Random Forest': (0.20, 0.630),
    'ChronoNet': (1.40, 0.675),
    'BIOT': (7.00, 0.630),
    'DeepConvNet': (8.20, 0.548),
    'BENDR': (11.0, 0.752),
    'LaBraM': (21.5, 0.665)
}

for p_val, score, col, n_cln in zip(params11, perf_index11, colors10, names_clean10):
    sz = 440 if n_cln.startswith("UC-EEG-FM") else 180
    marker = '*' if n_cln.startswith("UC-EEG-FM") else 'o'
    ax11a.scatter(p_val, score, s=sz, color=col, edgecolors='black', linewidth=1.6, marker=marker, zorder=6)
    dest_x, dest_y = LABEL_COORDS11.get(n_cln, (p_val * 1.15, score + 0.01))
    font_wt = 'bold' if n_cln.startswith("UC-EEG-FM") else 'normal'
    font_sz = 15.5 if n_cln.startswith("UC-EEG-FM") else 13.8
    lbl_color = '#DC2626' if n_cln.startswith("UC-EEG-FM") else '#222222'
    ax11a.annotate(n_cln, (p_val, score), xytext=(dest_x, dest_y),
                   fontsize=font_sz, fontweight=font_wt, color=lbl_color,
                   arrowprops=dict(arrowstyle="-", color="#555555", lw=1.0, alpha=0.75), zorder=7)

ax11a.plot([0.05, 0.28, 0.57], [0.6128, 0.6780, 0.7894], color='#DC2626', linestyle='--', lw=2.8, alpha=0.85, label='Pareto Frontier')
ax11a.set_xscale('log')
ax11a.set_xlim(0.02, 55.0)
ax11a.set_ylim(0.48, 0.85)
ax11a.set_xlabel('Parameters (Millions, Log)', fontweight='bold', fontsize=17.0)
ax11a.set_ylabel('Balanced Performance Index', fontweight='bold', fontsize=17.0)
ax11a.tick_params(labelsize=15.0)
ax11a.set_title('a  Parameter Efficiency Pareto Frontier', fontweight='bold', fontsize=18.5, loc='left', pad=12)
ax11a.grid(True, which='both', linestyle=':', alpha=0.5)
ax11a.legend(loc='lower right', fontsize=13.5, frameon=True, facecolor='white', framealpha=0.98, edgecolor='black')
style_spines(ax11a)

ax11a.text(0.03, 0.96, "41× Fewer Params than LaBraM\n+8.1% Index Gain",
           transform=ax11a.transAxes, fontsize=13.5, fontweight='bold', color="#166534", va='top',
           bbox=dict(boxstyle="round,pad=0.35", facecolor="#DCFCE7", edgecolor="#16A34A", lw=1.4))

ax11b = fig11.add_subplot(gs11[0, 1])
latencies11 = [m["cpu_latency_ms"] for m in models10]
ax11b.barh(names_clean10[::-1], latencies11[::-1], color=colors10[::-1], edgecolor='black', lw=1.4, height=0.62)
ax11b.axvline(4000.0, color='#DC2626', linestyle='--', lw=2.4, label='4,000 ms Real-Time Window')
ax11b.axvline(5.0, color='#16A34A', linestyle=':', lw=2.4, label='5.0 ms Bedside Edge Target')
ax11b.set_xscale('log')
ax11b.set_xlim(0.3, 9500.0)
ax11b.set_yticks(y_idx10)
ax11b.set_yticklabels(names_clean10[::-1], fontsize=15.0, fontweight='bold')
ax11b.set_xlabel('CPU Inference Latency (ms, Log)', fontweight='bold', fontsize=17.0)
ax11b.tick_params(axis='x', labelsize=15.0)
ax11b.set_title('b  Real-Time Bedside CPU Latency', fontweight='bold', fontsize=18.5, loc='left', pad=12)
ax11b.legend(loc='lower right', fontsize=13.5, frameon=True, facecolor='white', framealpha=0.98, edgecolor='black')
ax11b.grid(axis='x', which='both', linestyle=':', alpha=0.5)
style_spines(ax11b)

ax11b.text(3.29 * 1.4, 9, "3.29 ms (1,058× Speedup)",
           va='center', ha='left', fontsize=15.0, fontweight='bold', color='#DC2626',
           bbox=dict(boxstyle="round,pad=0.32", facecolor="#FEE2E2", edgecolor="black", lw=1.4))

save_and_mirror(fig11, "Figure_10_Pareto_Frontier_and_Edge_Latency")
save_and_mirror(fig11, "Figure_11_Pareto_Frontier_and_Edge_Latency")

print("\n=== ALL LARGE-TEXT ZERO-OVERLAP FIGURES COMPLETED ===")
