"""
Publication-Grade Master System Architecture & Clinical Translation Pipeline (Figure 1)
Designed for Computers in Biology and Medicine (Elsevier) & Nature Biomedical Engineering.
Strict formatting:
- Black and white box outlines (edgecolor='black', linewidth=1.8 to 2.2).
- Inside content is rich in color, visual diagrams, schematics, and waveforms.
- Visually understandable with concise labels and minimal dense text.
- 1.3x scaled typography with 100% zero text overlap or overflow.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch, Circle
import numpy as np

BASE_DIR = r"E:\All EEG"
FIG_DIR = os.path.join(BASE_DIR, "Figures")
LATEX_FIG_DIR = os.path.join(BASE_DIR, "latex_paper", "figures")
DESKTOP_FIG_DIR = r"C:\Users\Visha\Desktop\NewAlcohol\Figures"

for d in [FIG_DIR, LATEX_FIG_DIR, DESKTOP_FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# 1.3x typography scaling
plt.rcParams['font.family'] = 'DejaVu Sans'
SCALE = 1.3

fig = plt.figure(figsize=(24.0, 13.0), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Color palette for interior components
BG_STAGE_A = "#F0F4F8"  # Soft blue-grey
BG_STAGE_B = "#FEF9E7"  # Soft warm cream
BG_STAGE_C = "#F0FDF4"  # Soft mint green
BG_CARD = "#FFFFFF"     # Crisp white for sub-cards

# Helper to draw a box with black/white border and optional colored interior
def draw_box(x, y, w, h, bg_color=BG_CARD, edgecolor="black", lw=2.0, r=1.0, zorder=2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.3,rounding_size={r}",
                         facecolor=bg_color, edgecolor=edgecolor,
                         linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box

# Helper to draw clean arrows
def draw_arrow(x1, y1, x2, y2, color="#1E293B", lw=2.5, style="-|>", rad=0.0):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=f"{style},head_length=8,head_width=5",
                            color=color, linewidth=lw,
                            connectionstyle=f"arc3,rad={rad}", zorder=10)
    ax.add_patch(arrow)

# -----------------------------------------------------------------------------
# 1. TOP HEADER / TITLE BANNER
# -----------------------------------------------------------------------------
draw_box(2, 92, 96, 6.5, bg_color="#FFFFFF", edgecolor="black", lw=2.2, r=0.8)
ax.text(50, 96.2, "Unified Cross-Disorder EEG Foundation Model (UC-EEG-FM)",
        ha='center', va='center', fontsize=22 * SCALE, fontweight='bold', color='#0F172A', zorder=5)
ax.text(50, 93.5, "End-to-End Clinical Pre-Training, Dual Self-Supervised Objectives, and Bedside Decision Support Pipeline",
        ha='center', va='center', fontsize=13 * SCALE, fontweight='normal', color='#475569', zorder=5)

# -----------------------------------------------------------------------------
# 2. THREE PRIMARY HORIZONTAL STAGES (PANELS A, B, C)
# -----------------------------------------------------------------------------

# Panel A: Clinical Ingestion & Tokenization (x: 2 to 31)
draw_box(2, 3, 29.5, 87, bg_color=BG_STAGE_A, edgecolor="black", lw=2.0, r=1.2)
ax.text(3.5, 87.5, "a  Clinical Ingestion & Tokenization", fontsize=15 * SCALE, fontweight='bold', color='#0F172A', zorder=5)

# Panel B: Spatial-Temporal Transformer & Dual SSL (x: 33.5 to 66.5)
draw_box(33.5, 3, 33.0, 87, bg_color=BG_STAGE_B, edgecolor="black", lw=2.0, r=1.2)
ax.text(35.0, 87.5, "b  UC-EEG-FM Dual Self-Supervised Backbone", fontsize=15 * SCALE, fontweight='bold', color='#0F172A', zorder=5)

# Panel C: Downstream Clinical Adaptation & Telemetry (x: 68.5 to 98)
draw_box(68.5, 3, 29.5, 87, bg_color=BG_STAGE_C, edgecolor="black", lw=2.0, r=1.2)
ax.text(70.0, 87.5, "c  Downstream Generalization & Bedside CDSS", fontsize=15 * SCALE, fontweight='bold', color='#0F172A', zorder=5)

# Connect Stage A to B, and B to C with large bold arrows
draw_arrow(31.5, 46.5, 33.5, 46.5, color="#2563EB", lw=3.5)
draw_arrow(66.5, 46.5, 68.5, 46.5, color="#16A34A", lw=3.5)

# =============================================================================
# DETAILS: PANEL A (Ingestion & Harmonization)
# =============================================================================

# Card A1: Multi-Cohort Clinical Datasets
draw_box(3.5, 64.5, 26.5, 21.0, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.8)
ax.text(4.5, 83.2, "Heterogeneous Clinical Cohorts (146 Subs)", fontsize=11.5 * SCALE, fontweight='bold', color='#1E293B', zorder=5)

# Sub-cohort items with visual icons/badges
cohorts = [
    ("AUD (OpenNeuro ds005907)", "64/56 Channels | 500 Hz | N=53", "#E0F2FE", "#0284C7"),
    ("AD/FTD (OpenNeuro ds004504)", "19 Channels (10-20) | 500 Hz | N=88", "#FEE2E2", "#DC2626"),
    ("Pediatric Epilepsy (CHB-MIT)", "23 Channels Bipolar | 256 Hz | N=5", "#F3E8FF", "#9333EA")
]
for i, (c_title, c_sub, badge_bg, badge_border) in enumerate(cohorts):
    cy = 77.0 - i * 5.5
    draw_box(4.5, cy, 24.5, 4.6, bg_color=badge_bg, edgecolor="black", lw=1.0, r=0.5)
    ax.text(5.5, cy + 2.8, c_title, fontsize=10 * SCALE, fontweight='bold', color='#0F172A', zorder=6)
    ax.text(5.5, cy + 1.1, c_sub, fontsize=8.5 * SCALE, color='#334155', zorder=6)

# Flow arrow A1 -> A2
draw_arrow(16.75, 64.5, 16.75, 60.5, color="#0284C7", lw=2.2)

# Card A2: Signal Preprocessing & Standardization
draw_box(3.5, 43.5, 26.5, 17.0, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.8)
ax.text(4.5, 58.0, "Harmonization & Standardization", fontsize=11.5 * SCALE, fontweight='bold', color='#1E293B', zorder=5)

prep_steps = [
    ("CAR Re-referencing:", r"$x_c^{\mathrm{CAR}}(t) = x_c(t) - \frac{1}{C}\sum x_j(t)$", "#F8FAFC"),
    ("Bandpass Filtering:", "0.5 – 45.0 Hz FIR (Zero-Phase)", "#F8FAFC"),
    ("Uniform Resampling:", "256.0 Hz Uniform Grid", "#F8FAFC"),
    ("Window Segmentation:", "4.0 s Non-overlapping (1,024 Samples)", "#F8FAFC")
]
for i, (p_lbl, p_val, bg) in enumerate(prep_steps):
    py = 53.5 - i * 3.4
    ax.text(4.8, py, p_lbl, fontsize=9.2 * SCALE, fontweight='bold', color='#334155', zorder=6)
    ax.text(14.5, py, p_val, fontsize=9.0 * SCALE, color='#0284C7' if "$" in p_val else '#475569', zorder=6)

# Flow arrow A2 -> A3
draw_arrow(16.75, 43.5, 16.75, 39.5, color="#0284C7", lw=2.2)

# Card A3: Channel-Agnostic Feature Tokenization (L_tok = 11)
draw_box(3.5, 5.0, 26.5, 34.5, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.8)
ax.text(4.5, 37.0, "Channel-Agnostic Biosignal Tokenizer", fontsize=11.5 * SCALE, fontweight='bold', color='#1E293B', zorder=5)
ax.text(4.5, 34.8, r"Invariant Token Sequence: $L_{\mathrm{tok}} = 11$ per channel", fontsize=9.5 * SCALE, color='#64748B', zorder=5)

# Visual Token Matrix Graphic (11 colorful token blocks)
token_names = [
    ("δ (0.5-4)", "#EF4444"), ("θ (4-8)", "#F97316"), ("α (8-13)", "#F59E0B"),
    ("β (13-30)", "#10B981"), ("γ (30-45)", "#06B6D4"), ("TAR (θ/α)", "#3B82F6"),
    ("TBR (θ/β)", "#6366F1"), ("ABR (α/β)", "#8B5CF6"), ("H_spec", "#EC4899"),
    ("H_diff", "#D946EF"), ("Hjorth", "#64748B")
]
for idx, (t_name, t_col) in enumerate(token_names):
    ty = 30.5 - (idx // 3) * 6.0
    tx = 4.8 + (idx % 3) * 8.0
    draw_box(tx, ty, 7.2, 5.0, bg_color=t_col, edgecolor="black", lw=1.2, r=0.4)
    ax.text(tx + 3.6, ty + 2.5, t_name, ha='center', va='center', fontsize=9.2 * SCALE, fontweight='bold', color='#FFFFFF', zorder=6)

# Bottom summary in Panel A
draw_box(4.5, 6.2, 24.5, 5.2, bg_color="#EFF6FF", edgecolor="black", lw=1.0, r=0.5)
ax.text(16.75, 9.8, "49,472 Standardized Epochs (54.97 Hours)", ha='center', fontsize=9.5 * SCALE, fontweight='bold', color='#1E40AF', zorder=6)
ax.text(16.75, 7.6, "Strict Subject-Stratified Group Holdout", ha='center', fontsize=8.8 * SCALE, color='#3B82F6', zorder=6)


# =============================================================================
# DETAILS: PANEL B (UC-EEG-FM Dual SSL Backbone)
# =============================================================================

# Card B1: Input Linear Token Projection & Positional Encoding
draw_box(35.0, 68.0, 30.0, 17.5, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.8)
ax.text(36.0, 83.0, "Input Token Embedding & Positional Encoding", fontsize=11.5 * SCALE, fontweight='bold', color='#1E293B', zorder=5)

# Visual formula & projection diagram
draw_box(36.0, 75.0, 28.0, 6.2, bg_color="#FEF3C7", edgecolor="black", lw=1.0, r=0.4)
ax.text(50.0, 78.1, r"$H_0 = \mathrm{LayerNorm}(\mathrm{GELU}(X_i W_{\mathrm{in}} + b_{\mathrm{in}})) + \mathrm{PE} \in \mathbb{R}^{L_{\mathrm{tok}} \times 128}$",
        ha='center', va='center', fontsize=10.5 * SCALE, fontweight='bold', color='#78350F', zorder=6)

ax.text(36.5, 71.5, "• Linear Projection:", fontsize=9.5 * SCALE, fontweight='bold', color='#334155', zorder=6)
ax.text(47.5, 71.5, r"$W_{\mathrm{in}} \in \mathbb{R}^{11 \times 128}$, $b_{\mathrm{in}} \in \mathbb{R}^{128}$", fontsize=9.5 * SCALE, color='#0284C7', zorder=6)
ax.text(36.5, 69.2, "• Positional Encoding:", fontsize=9.5 * SCALE, fontweight='bold', color='#334155', zorder=6)
ax.text(48.5, 69.2, "Learnable 1D Spectral Sequence PE", fontsize=9.2 * SCALE, color='#475569', zorder=6)

# Flow arrow B1 -> B2
draw_arrow(50.0, 68.0, 50.0, 64.0, color="#D97706", lw=2.2)

# Card B2: 4-Layer Spatial-Temporal Transformer Encoder
draw_box(35.0, 34.0, 30.0, 30.0, bg_color="#FFFFFF", edgecolor="black", lw=1.8, r=0.8)
ax.text(36.0, 61.5, "4-Layer Transformer Encoder (0.57M Params)", fontsize=11.5 * SCALE, fontweight='bold', color='#1E293B', zorder=5)

# Draw 4 stacked layers schematically
layer_colors = ["#FFFBEB", "#FEF3C7", "#FDE68A", "#FCD34D"]
for l_idx in range(4):
    ly = 35.5 + l_idx * 5.8
    draw_box(36.5, ly, 27.0, 4.8, bg_color=layer_colors[l_idx], edgecolor="black", lw=1.2, r=0.4)
    ax.text(37.5, ly + 2.4, f"Transformer Layer {l_idx+1}", fontsize=9.8 * SCALE, fontweight='bold', color='#92400E', zorder=6)
    ax.text(53.0, ly + 2.4, "MHA (8 heads, d=128) + MLP (512)", fontsize=8.8 * SCALE, color='#78350F', zorder=6)

ax.text(36.5, 58.8, "• Latent Dimension: d = 128  |  FFN Dimension: 512", fontsize=9.0 * SCALE, color='#64748B', zorder=6)

# Flow arrows from Transformer to Dual SSL Heads (Split Branch)
draw_arrow(43.0, 34.0, 41.0, 30.5, color="#DC2626", lw=2.2)
draw_arrow(57.0, 34.0, 59.0, 30.5, color="#2563EB", lw=2.2)

# Card B3: Dual Self-Supervised Learning Heads
# Branch 1: Masked Token Reconstruction
draw_box(35.0, 5.0, 14.5, 25.5, bg_color="#FFF5F5", edgecolor="black", lw=1.5, r=0.6)
ax.text(36.0, 28.0, "Head 1: Masked Recon", fontsize=10.5 * SCALE, fontweight='bold', color='#991B1B', zorder=5)
ax.text(36.0, 25.5, "Mask Ratio p = 20%", fontsize=9.0 * SCALE, color='#B91C1C', zorder=5)

draw_box(36.0, 16.0, 12.5, 7.8, bg_color="#FFFFFF", edgecolor="black", lw=1.0, r=0.4)
ax.text(42.25, 21.0, "Linear Decoder", ha='center', fontsize=9.0 * SCALE, fontweight='bold', color='#1E293B', zorder=6)
ax.text(42.25, 18.0, r"$\hat{x} \in \mathbb{R}^{L_{\mathrm{tok}}}$", ha='center', fontsize=10.0 * SCALE, color='#DC2626', zorder=6)

draw_box(36.0, 6.2, 12.5, 7.8, bg_color="#FEE2E2", edgecolor="black", lw=1.0, r=0.4)
ax.text(42.25, 11.5, "Spectral MSE Loss", ha='center', fontsize=8.8 * SCALE, fontweight='bold', color='#991B1B', zorder=6)
ax.text(42.25, 8.2, r"$\mathcal{L}_{\mathrm{recon}} = \frac{1}{|\mathcal{M}|}\sum (x - \hat{x})^2$",
        ha='center', fontsize=8.2 * SCALE, fontweight='bold', color='#991B1B', zorder=6)

# Branch 2: Contrastive InfoNCE Hypersphere Projection
draw_box(50.5, 5.0, 14.5, 25.5, bg_color="#EFF6FF", edgecolor="black", lw=1.5, r=0.6)
ax.text(51.5, 28.0, "Head 2: InfoNCE Sphere", fontsize=10.5 * SCALE, fontweight='bold', color='#1E40AF', zorder=5)
ax.text(51.5, 25.5, r"Hypersphere $\mathbb{S}^{63}$, $\tau=0.07$", fontsize=9.0 * SCALE, color='#1D4ED8', zorder=5)

draw_box(51.5, 16.0, 12.5, 7.8, bg_color="#FFFFFF", edgecolor="black", lw=1.0, r=0.4)
ax.text(57.75, 21.0, "MLP Projection Head", ha='center', fontsize=9.0 * SCALE, fontweight='bold', color='#1E293B', zorder=6)
ax.text(57.75, 18.0, r"$z_i = g(h_L)/\|g(h_L)\|_2$", ha='center', fontsize=9.5 * SCALE, color='#2563EB', zorder=6)

draw_box(51.5, 6.2, 12.5, 7.8, bg_color="#DBEAFE", edgecolor="black", lw=1.0, r=0.4)
ax.text(57.75, 11.5, "Cross-Subject Contrast", ha='center', fontsize=8.8 * SCALE, fontweight='bold', color='#1E40AF', zorder=6)
ax.text(57.75, 8.2, r"$\mathcal{L}_{\mathrm{contrast}}(\tau=0.07)$",
        ha='center', fontsize=8.8 * SCALE, fontweight='bold', color='#1E40AF', zorder=6)


# =============================================================================
# DETAILS: PANEL C (Downstream Adaptation & Bedside CDSS)
# =============================================================================

# Card C1: Zero-Shot Cross-Disorder Transfer
draw_box(70.0, 64.5, 26.5, 21.0, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.8)
ax.text(71.0, 83.2, "1. Zero-Shot Cross-Disorder Transfer", fontsize=11.5 * SCALE, fontweight='bold', color='#1E293B', zorder=5)

draw_box(71.0, 73.0, 24.5, 8.0, bg_color="#F0FDF4", edgecolor="black", lw=1.0, r=0.5)
ax.text(83.25, 78.5, "Dementia -> Alcoholism Transfer", ha='center', fontsize=9.5 * SCALE, fontweight='bold', color='#15803D', zorder=6)
ax.text(83.25, 75.2, "AUROC = 0.6744 (+17.4% over chance)", ha='center', fontsize=9.2 * SCALE, color='#166534', zorder=6)

draw_box(71.0, 66.0, 24.5, 6.0, bg_color="#DCFCE7", edgecolor="black", lw=1.0, r=0.5)
ax.text(83.25, 70.0, "Alcoholism -> Dementia Transfer", ha='center', fontsize=9.5 * SCALE, fontweight='bold', color='#15803D', zorder=6)
ax.text(83.25, 67.5, "AUROC = 0.8676 (Universal Manifold)", ha='center', fontsize=9.2 * SCALE, color='#166534', zorder=6)

# Flow arrow C1 -> C2
draw_arrow(83.25, 64.5, 83.25, 60.5, color="#16A34A", lw=2.2)

# Card C2: Few-Shot Linear Probing (Low-Data Clinical Regime)
draw_box(70.0, 39.5, 26.5, 21.0, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.8)
ax.text(71.0, 58.0, "2. Extreme Few-Shot Clinical Probing", fontsize=11.5 * SCALE, fontweight='bold', color='#1E293B', zorder=5)

# Visual few-shot bar badges
draw_box(71.0, 50.0, 24.5, 6.2, bg_color="#EFF6FF", edgecolor="black", lw=1.0, r=0.4)
ax.text(72.0, 54.0, "K = 2 Patients per Class (AD)", fontsize=9.2 * SCALE, fontweight='bold', color='#1E40AF', zorder=6)
ax.text(72.0, 51.5, "AUROC = 0.8674 ± 0.073 (Frozen Backbone)", fontsize=8.8 * SCALE, color='#2563EB', zorder=6)

draw_box(71.0, 41.5, 24.5, 6.2, bg_color="#FEF2F2", edgecolor="black", lw=1.0, r=0.4)
ax.text(72.0, 45.5, "K = 2 Patients per Class (AUD)", fontsize=9.2 * SCALE, fontweight='bold', color='#991B1B', zorder=6)
ax.text(72.0, 43.0, "AUROC = 0.7420 (Matches 5x Training Cohort)", fontsize=8.8 * SCALE, color='#DC2626', zorder=6)

# Flow arrow C2 -> C3
draw_arrow(83.25, 39.5, 83.25, 35.5, color="#16A34A", lw=2.2)

# Card C3: Real-Time Bedside CDSS Telemetry & Deployment
draw_box(70.0, 5.0, 26.5, 30.5, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.8)
ax.text(71.0, 33.0, "3. Bedside Decision Support & Telemetry", fontsize=11.5 * SCALE, fontweight='bold', color='#1E293B', zorder=5)

telemetry_items = [
    ("Single-Epoch CPU Latency:", "3.29 ms (1,058x Real-Time)", "#16A34A"),
    ("Batched GPU Throughput:", "58.0 μs / epoch (68.7k/s)", "#0284C7"),
    ("Model Parameters:", "0.57 Million (2.23 MB Binary)", "#D97706"),
    ("Clinical Telemetry Support:", "AUD, AD Staging, Seizure Alerts", "#4F46E5")
]
for i, (t_lbl, t_val, col) in enumerate(telemetry_items):
    ty = 28.5 - i * 5.6
    draw_box(71.0, ty, 24.5, 4.8, bg_color="#F8FAFC", edgecolor="black", lw=1.0, r=0.4)
    ax.text(72.0, ty + 2.9, t_lbl, fontsize=8.8 * SCALE, fontweight='bold', color='#334155', zorder=6)
    ax.text(72.0, ty + 1.2, t_val, fontsize=9.0 * SCALE, fontweight='bold', color=col, zorder=6)

# Save high-resolution outputs
for d in [FIG_DIR, LATEX_FIG_DIR, DESKTOP_FIG_DIR]:
    png_p = os.path.join(d, "Figure_1_Master_Architecture.png")
    pdf_p = os.path.join(d, "Figure_1_Master_Architecture.pdf")
    fig.savefig(png_p, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_p, format='pdf', bbox_inches='tight')
    print(f"Saved: {png_p} & {pdf_p}")

plt.close(fig)
print("=== FIGURE 1 MASTER ARCHITECTURE COMPLETED SUCCESSFULLY ===")
