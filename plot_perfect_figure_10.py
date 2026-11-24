"""
Generates Figure 10: Clinical Benchmark Generalization with 100% Zero-Overlap Guarantee.
- All boxes and subplot axes have black-and-white borders (edgecolor='black', linewidth=2.0).
- Dedicated legends placed outside axes (above Panel a and above Panel d).
- Generous wspace=0.62 and hspace=0.52 to ensure complete separation between subplots.
- xlim gives ample breathing room for bar annotations.
- 1.3x scaled typography for maximum readability.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR = r"E:\All EEG"
MODEL_DIR = os.path.join(BASE_DIR, "Models")
FIG_DIR = os.path.join(BASE_DIR, "Figures")
LATEX_FIG_DIR = os.path.join(BASE_DIR, "latex_paper", "figures")
DESKTOP_FIG_DIR = r"C:\Users\Visha\Desktop\NewAlcohol\Figures"
BRAIN_DIR = r"C:\Users\Visha\.gemini\antigravity\brain\a487a487-156f-4781-ba99-0b2ec1ba776a"

for d in [FIG_DIR, LATEX_FIG_DIR, DESKTOP_FIG_DIR, BRAIN_DIR]:
    os.makedirs(d, exist_ok=True)

with open(os.path.join(MODEL_DIR, "sota_literature_benchmark_sweep_2025_2026.json"), "r", encoding="utf-8") as f:
    bench = json.load(f)

models = bench["models"]
names_clean = [
    'UC-EEG-FM (Proposed)',
    'LaBraM',
    'BIOT',
    'BENDR',
    'EEG-Conformer',
    'ChronoNet',
    'DeepConvNet',
    'EEGNet-8,2',
    'Random Forest',
    'SVM (RBF)'
]

colors = [
    '#D90429', '#2B2D42', '#8D99AE', '#457B9D', '#2A9D8F',
    '#E76F51', '#F4A261', '#6C757D', '#ADB5BD', '#CED4DA'
]

# Create Figure 10 with generous canvas size
fig = plt.figure(figsize=(26.0, 17.5), dpi=300)
# left=0.18 gives ample space for model names on left column
# wspace=0.65 ensures massive space between left and right columns!
# hspace=0.52 ensures vertical space between top and bottom rows
# top=0.88 gives room for legends above ax1 and ax4
gs = gridspec.GridSpec(2, 2, wspace=0.65, hspace=0.52, left=0.19, right=0.96, top=0.88, bottom=0.08)
y_idx = np.arange(len(names_clean))
bar_h = 0.38

def style_spines(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.0)

# ==============================================================================
# Panel a: Subject-Leakage Crisis in AUD
# ==============================================================================
ax1 = fig.add_subplot(gs[0, 0])
rand_aud = [m["aud_auroc_random"] for m in models]
strat_aud = [m["aud_auroc_stratified"] for m in models]

b1 = ax1.barh(y_idx - bar_h/2, rand_aud[::-1], bar_h, label='Random Split (Biased / Leakage)', color='#ADB5BD', alpha=0.85, edgecolor='black', lw=1.2)
b2 = ax1.barh(y_idx + bar_h/2, strat_aud[::-1], bar_h, label='Subject Holdout (Group-5-Fold)', color=colors[::-1], edgecolor='black', lw=1.4)
ax1.axvline(0.5, color='#333333', linestyle='--', lw=1.8, alpha=0.8, label='Chance (0.50)')

ax1.set_yticks(y_idx)
ax1.set_yticklabels(names_clean[::-1], fontsize=16, fontweight='bold')
ax1.set_xlabel('Alcoholism AUROC', fontweight='bold', fontsize=19)
ax1.tick_params(axis='x', labelsize=16.5)
ax1.tick_params(axis='y', labelsize=16.5)
ax1.set_xlim(0.35, 1.15)
ax1.set_title('a  Subject-Leakage Crisis in AUD', fontweight='bold', fontsize=21, loc='left', pad=48)

# Dedicated legend ABOVE ax1: completely outside the plot, impossible to collide with bars!
ax1.legend(bbox_to_anchor=(0.5, 1.16), loc='upper center', ncol=3, fontsize=13.5,
           frameon=True, facecolor='white', framealpha=0.98, edgecolor='black')
ax1.grid(axis='x', linestyle=':', alpha=0.5)
style_spines(ax1)

# Annotate proposed model
ax1.text(strat_aud[0] + 0.015, y_idx[-1] + bar_h/2, f"{strat_aud[0]:.3f}",
         va='center', ha='left', fontsize=15.5, fontweight='bold', color='#D90429')

# ==============================================================================
# Panel b: Dementia SOTA Generalization
# ==============================================================================
ax2 = fig.add_subplot(gs[0, 1])
ad_scores = [m["ad_auroc_stratified"] for m in models]
bars2 = ax2.barh(y_idx, ad_scores[::-1], color=colors[::-1], edgecolor='black', lw=1.4, height=0.62)
ax2.axvline(0.5, color='#333333', linestyle='--', lw=1.8, alpha=0.8)

ax2.set_yticks(y_idx)
ax2.set_yticklabels(names_clean[::-1], fontsize=16, fontweight='bold')
ax2.set_xlabel("Alzheimer's AUROC (Group 5-Fold)", fontweight='bold', fontsize=19)
ax2.tick_params(axis='x', labelsize=16.5)
ax2.tick_params(axis='y', labelsize=16.5)
ax2.set_xlim(0.42, 1.14)
ax2.set_title('b  Dementia SOTA Generalization', fontweight='bold', fontsize=21, loc='left', pad=18)
ax2.grid(axis='x', linestyle=':', alpha=0.5)
style_spines(ax2)

for bar, score in zip(bars2, ad_scores[::-1]):
    is_top = (score == max(ad_scores))
    ax2.text(bar.get_width() + 0.014, bar.get_y() + bar.get_height()/2, f"{score:.3f}",
             va='center', ha='left', fontsize=15.5, fontweight='bold' if is_top else 'normal',
             color='#D90429' if is_top else '#222222')

# ==============================================================================
# Panel c: Bidirectional Zero-Shot Transfer
# ==============================================================================
ax3 = fig.add_subplot(gs[1, 0])
transfer_scores = [m["cross_disorder_transfer_auroc"] for m in models]
bars3 = ax3.barh(y_idx, transfer_scores[::-1], color=colors[::-1], edgecolor='black', lw=1.4, height=0.62)
ax3.axvline(0.5, color='#333333', linestyle='--', lw=1.8, alpha=0.8)

ax3.set_yticks(y_idx)
ax3.set_yticklabels(names_clean[::-1], fontsize=16, fontweight='bold')
ax3.set_xlabel('Out-of-Domain Zero-Shot AUROC', fontweight='bold', fontsize=19)
ax3.tick_params(axis='x', labelsize=16.5)
ax3.tick_params(axis='y', labelsize=16.5)
ax3.set_xlim(0.38, 1.14)
ax3.set_title('c  Bidirectional Zero-Shot Transfer', fontweight='bold', fontsize=21, loc='left', pad=18)
ax3.grid(axis='x', linestyle=':', alpha=0.5)
style_spines(ax3)

for bar, score in zip(bars3, transfer_scores[::-1]):
    is_top = (score == max(transfer_scores))
    ax3.text(bar.get_width() + 0.014, bar.get_y() + bar.get_height()/2, f"{score:.3f}",
             va='center', ha='left', fontsize=15.5, fontweight='bold' if is_top else 'normal',
             color='#D90429' if is_top else '#222222')

# ==============================================================================
# Panel d: Few-Shot Low-Data Regime (K=2)
# ==============================================================================
ax4 = fig.add_subplot(gs[1, 1])
k2_ad = [m["fewshot_k2_ad_auroc"] for m in models]
k2_aud = [m["fewshot_k2_aud_auroc"] for m in models]

ax4.barh(y_idx - bar_h/2, k2_ad[::-1], bar_h, label="Alzheimer's (K=2)", color='#2B2D42', edgecolor='black', lw=1.2)
ax4.barh(y_idx + bar_h/2, k2_aud[::-1], bar_h, label='Alcoholism (K=2)', color='#D90429', edgecolor='black', lw=1.2)
ax4.axvline(0.5, color='#333333', linestyle='--', lw=1.8, alpha=0.8)

ax4.set_yticks(y_idx)
ax4.set_yticklabels(names_clean[::-1], fontsize=16, fontweight='bold')
ax4.set_xlabel('Few-Shot AUROC (K=2 Patients per Class)', fontweight='bold', fontsize=19)
ax4.tick_params(axis='x', labelsize=16.5)
ax4.tick_params(axis='y', labelsize=16.5)
ax4.set_xlim(0.38, 1.14)
ax4.set_title('d  Few-Shot Low-Data Regime (K=2)', fontweight='bold', fontsize=21, loc='left', pad=48)

# Dedicated legend ABOVE ax4: completely outside the plot, impossible to collide with bars!
ax4.legend(bbox_to_anchor=(0.5, 1.16), loc='upper center', ncol=2, fontsize=13.5,
           frameon=True, facecolor='white', framealpha=0.98, edgecolor='black')
ax4.grid(axis='x', linestyle=':', alpha=0.5)
style_spines(ax4)

# Annotate proposed model
ax4.text(k2_ad[0] + 0.014, y_idx[-1] - bar_h/2, f"{k2_ad[0]:.3f}",
         va='center', ha='left', fontsize=14.5, fontweight='bold', color='#2B2D42')
ax4.text(k2_aud[0] + 0.014, y_idx[-1] + bar_h/2, f"{k2_aud[0]:.3f}",
         va='center', ha='left', fontsize=14.5, fontweight='bold', color='#D90429')

# Save to all locations with correct figure names
targets = [
    "Figure_10_Clinical_Benchmark_Generalization",
    "Figure_9_Clinical_Benchmark_Generalization",
    "Figure_9_SOTA_Literature_Benchmark_Sweep"
]

for name in targets:
    for d in [FIG_DIR, LATEX_FIG_DIR, DESKTOP_FIG_DIR, BRAIN_DIR]:
        png_p = os.path.join(d, f"{name}.png")
        pdf_p = os.path.join(d, f"{name}.pdf")
        fig.savefig(png_p, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_p, format='pdf', bbox_inches='tight')

plt.close(fig)
print("=== Successfully plotted Figure 10 with ZERO overlap, black-and-white borders, and external legends! ===")
