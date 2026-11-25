"""
World-Class Master System Architecture & Clinical Translation Framework (Figure 1 v2)
Hyper-visual, highly intuitive, minimal text, publication-grade aesthetics.
Strict formatting:
- Black and white box outlines (edgecolor='black', linewidth=2.0).
- Inside content is rich in color, graphical diagrams, head topographies, waveforms, attention matrices, and 3D spheres.
- Visually understandable at a glance.
- 1.3x scaled typography with 100% zero text overlap or overflow.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Wedge, Rectangle
import numpy as np

BASE_DIR = r"E:\All EEG"
FIG_DIR = os.path.join(BASE_DIR, "Figures")
LATEX_FIG_DIR = os.path.join(BASE_DIR, "latex_paper", "figures")
DESKTOP_FIG_DIR = r"C:\Users\Visha\Desktop\NewAlcohol\Figures"

for d in [FIG_DIR, LATEX_FIG_DIR, DESKTOP_FIG_DIR]:
    os.makedirs(d, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
SCALE = 1.3

fig = plt.figure(figsize=(25.0, 14.5), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Helpers
def draw_box(x, y, w, h, bg_color="#FFFFFF", edgecolor="black", lw=2.0, r=0.8, zorder=2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.25,rounding_size={r}",
                         facecolor=bg_color, edgecolor=edgecolor,
                         linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box

def draw_arrow(x1, y1, x2, y2, color="#1E293B", lw=2.5, style="-|>", rad=0.0):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=f"{style},head_length=8,head_width=5",
                            color=color, linewidth=lw,
                            connectionstyle=f"arc3,rad={rad}", zorder=12)
    ax.add_patch(arrow)

def draw_head_outline(cx, cy, r=2.2, color="#0F172A", zorder=5):
    # Head circle
    c = Circle((cx, cy), r, facecolor="#F8FAFC", edgecolor=color, lw=1.5, zorder=zorder)
    ax.add_patch(c)
    # Nose triangle
    nose = patches.Polygon([[cx - 0.4, cy + r - 0.1], [cx, cy + r + 0.6], [cx + 0.4, cy + r - 0.1]],
                           closed=True, facecolor="#F8FAFC", edgecolor=color, lw=1.4, zorder=zorder)
    ax.add_patch(nose)
    # Ears
    ear_l = Ellipse((cx - r - 0.2, cy), 0.4, 0.9, facecolor="#F8FAFC", edgecolor=color, lw=1.4, zorder=zorder)
    ear_r = Ellipse((cx + r + 0.2, cy), 0.4, 0.9, facecolor="#F8FAFC", edgecolor=color, lw=1.4, zorder=zorder)
    ax.add_patch(ear_l)
    ax.add_patch(ear_r)

def draw_electrode_dots(cx, cy, r=2.2, num_dots=19, color="#2563EB", zorder=6):
    np.random.seed(42)
    # Generate polar points inside head
    angles = np.linspace(0, 2*np.pi, num_dots, endpoint=False)
    radii = np.random.uniform(0.3, r - 0.4, num_dots)
    xs = cx + radii * np.cos(angles)
    ys = cy + radii * np.sin(angles)
    ax.scatter(xs, ys, s=24, color=color, edgecolor="black", lw=0.6, zorder=zorder)

def draw_clean_waveform(x, y, w, h, freq=4.0, col="#2563EB", lw=2.0, zorder=6):
    t = np.linspace(0, w, 250)
    # Multi-component realistic EEG signal
    sig = np.sin(2 * np.pi * freq * (t / w)) + 0.4 * np.sin(2 * np.pi * (freq * 2.5) * (t / w)) + 0.2 * np.random.normal(0, 0.3, len(t))
    sig = (sig / np.max(np.abs(sig))) * (h / 2.2)
    ax.plot(x + t, y + h/2.0 + sig, color=col, lw=lw, zorder=zorder)

# -----------------------------------------------------------------------------
# 1. TOP HEADER BANNER
# -----------------------------------------------------------------------------
draw_box(1.5, 93.2, 97.0, 5.6, bg_color="#FFFFFF", edgecolor="black", lw=2.2, r=0.8)
ax.text(50.0, 96.6, "Unified Cross-Disorder EEG Foundation Model (UC-EEG-FM)",
        ha='center', va='center', fontsize=21 * SCALE, fontweight='bold', color='#0F172A', zorder=5)
ax.text(50.0, 94.4, "End-to-End System Architecture, Tokenized Sequence Attention, and Clinical Translation Framework",
        ha='center', va='center', fontsize=12.5 * SCALE, color='#475569', zorder=5)

# -----------------------------------------------------------------------------
# 2. THREE PRIMARY COLUMNS (STAGES A, B, C)
# -----------------------------------------------------------------------------
# Stage A: Ingestion & Tokenization (x: 1.5 to 31.0, y: 2.0 to 91.5)
draw_box(1.5, 2.0, 29.5, 89.5, bg_color="#F8FAFC", edgecolor="black", lw=2.0, r=1.0)
ax.text(3.0, 88.5, "a  Multi-Cohort Ingestion & Harmonization", fontsize=14.5 * SCALE, fontweight='bold', color='#0F172A', zorder=5)

# Stage B: Transformer Dual SSL Backbone (x: 33.0 to 67.0, y: 2.0 to 91.5)
draw_box(33.0, 2.0, 34.0, 89.5, bg_color="#FFFDF5", edgecolor="black", lw=2.0, r=1.0)
ax.text(34.5, 88.5, "b  UC-EEG-FM Dual Self-Supervised Backbone", fontsize=14.5 * SCALE, fontweight='bold', color='#0F172A', zorder=5)

# Stage C: Downstream Clinical Adaptation & CDSS (x: 69.0 to 98.5, y: 2.0 to 91.5)
draw_box(69.0, 2.0, 29.5, 89.5, bg_color="#F0FDF4", edgecolor="black", lw=2.0, r=1.0)
ax.text(70.5, 88.5, "c  Downstream Generalization & Bedside CDSS", fontsize=14.5 * SCALE, fontweight='bold', color='#0F172A', zorder=5)

# Primary pipeline flow arrows between stages
draw_arrow(31.0, 48.0, 33.0, 48.0, color="#2563EB", lw=3.8)
draw_arrow(67.0, 48.0, 69.0, 48.0, color="#16A34A", lw=3.8)

# =============================================================================
# STAGE A CONTENT: VISUAL INGESTION, RAW SIGNALS, TOKENIZER
# =============================================================================

# Card A1: 3 Heterogeneous Clinical Cohorts with Visual Head Schematics
draw_box(2.8, 67.5, 26.9, 19.5, bg_color="#FFFFFF", edgecolor="black", lw=1.6, r=0.6)
ax.text(3.8, 84.5, "1. Heterogeneous Clinical Cohorts (N=146)", fontsize=11.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

# 3 Mini Cohort Panels with drawn heads
cohort_configs = [
    ("AUD (ds005907)", "64-ch | 500 Hz | N=53", 6.8, "#0284C7", 32),
    ("AD/FTD (ds004504)", "19-ch | 500 Hz | N=88", 16.2, "#DC2626", 19),
    ("Epilepsy (CHB-MIT)", "23-ch | 256 Hz | N=5", 25.6, "#9333EA", 23)
]
for title, meta, cx, col, dots in cohort_configs:
    draw_box(cx - 4.1, 68.6, 8.2, 14.5, bg_color="#F8FAFC", edgecolor="black", lw=1.2, r=0.5)
    draw_head_outline(cx, 78.5, r=2.2, color=col)
    draw_electrode_dots(cx, 78.5, r=2.2, num_dots=dots, color=col)
    ax.text(cx, 73.8, title, ha='center', fontsize=9.2 * SCALE, fontweight='bold', color='#0F172A', zorder=7)
    ax.text(cx, 71.0, meta, ha='center', fontsize=7.8 * SCALE, color='#475569', zorder=7)

# Flow arrow A1 -> A2
draw_arrow(16.25, 67.5, 16.25, 63.8, color="#0284C7", lw=2.2)

# Card A2: Multi-Channel Continuous Raw EEG Waveforms & CAR Preprocessing
draw_box(2.8, 41.5, 26.9, 21.5, bg_color="#FFFFFF", edgecolor="black", lw=1.6, r=0.6)
ax.text(3.8, 60.5, "2. Multi-Channel EEG & CAR Standardization", fontsize=11.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

# Draw 4 multi-channel EEG waveform traces (shifted up to leave clean clearance above CAR box)
trace_colors = ["#2563EB", "#DC2626", "#16A34A", "#9333EA"]
trace_labels = ["Fp1 (Frontal)", "C3 (Central)", "Pz (Parietal)", "O1 (Occipital)"]
for i in range(4):
    wy = 56.5 - i * 2.65
    ax.text(3.8, wy + 0.35, trace_labels[i], fontsize=8.4 * SCALE, fontweight='bold', color=trace_colors[i], zorder=7)
    draw_clean_waveform(12.5, wy, 16.2, 1.9, freq=3.0 + i*1.5, col=trace_colors[i], lw=1.8)

# Processing badges
draw_box(3.8, 42.3, 24.9, 3.6, bg_color="#F1F5F9", edgecolor="black", lw=1.1, r=0.4)
ax.text(16.25, 44.1, r"CAR: $x_c^{\mathrm{CAR}}(t) = x_c(t) - \frac{1}{C}\sum x_j(t)$  |  0.5–45 Hz  |  4.0s Epoch",
        ha='center', va='center', fontsize=8.5 * SCALE, fontweight='bold', color='#0F172A', zorder=7)

# Flow arrow A2 -> A3
draw_arrow(16.25, 41.5, 16.25, 37.8, color="#0284C7", lw=2.2)

# Card A3: 11-Dimensional Biosignal Tokenizer with Miniature PSD Spectrum
draw_box(2.8, 4.0, 26.9, 33.0, bg_color="#FFFFFF", edgecolor="black", lw=1.6, r=0.6)
ax.text(3.8, 34.5, "3. Channel-Agnostic Biosignal Tokenizer", fontsize=11.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)
ax.text(3.8, 32.2, r"Invariant Token Sequence: $L_{\mathrm{tok}} = 11$ Biomarkers per Epoch", fontsize=9.2 * SCALE, color='#64748B', zorder=6)

# Visual PSD Spectrum Curve Graphic
draw_box(3.8, 18.5, 24.9, 12.8, bg_color="#FAFAFA", edgecolor="black", lw=1.2, r=0.5)
ax.text(4.8, 29.2, "Power Spectral Density (PSD, Welch 0.5–45 Hz)", fontsize=9.0 * SCALE, fontweight='bold', color='#1E293B', zorder=7)

# Draw simulated PSD curve with 5 shaded frequency bands
freqs = np.linspace(0.5, 45, 100)
psd = 1.0 / (freqs**1.2) + 0.3 * np.exp(-((freqs - 10)**2)/6) # 1/f with alpha bump
psd_norm = (psd - np.min(psd)) / (np.max(psd) - np.min(psd)) * 6.5

bands = [
    (0.5, 4.0, "δ (0.5-4)", "#EF4444"),
    (4.0, 8.0, "θ (4-8)", "#F97316"),
    (8.0, 13.0, "α (8-13)", "#F59E0B"),
    (13.0, 30.0, "β (13-30)", "#10B981"),
    (30.0, 45.0, "γ (30-45)", "#06B6D4")
]
for f_min, f_max, b_name, b_col in bands:
    mask = (freqs >= f_min) & (freqs <= f_max)
    x_sub = 5.0 + (freqs[mask] - 0.5) / 44.5 * 22.5
    y_sub = 20.0 + psd_norm[mask]
    ax.fill_between(x_sub, 20.0, y_sub, color=b_col, alpha=0.35, zorder=6)
    ax.plot(x_sub, y_sub, color=b_col, lw=2.0, zorder=7)
    # Band label
    f_mid = (f_min + f_max) / 2.0
    ax.text(5.0 + (f_mid - 0.5) / 44.5 * 22.5, 20.6, b_name[:1], ha='center', fontsize=8.5 * SCALE, fontweight='bold', color=b_col, zorder=8)

# 11 Extracted Tokens Block
token_items = [
    ("δ", "#EF4444"), ("θ", "#F97316"), ("α", "#F59E0B"), ("β", "#10B981"), ("γ", "#06B6D4"),
    ("TAR", "#3B82F6"), ("TBR", "#6366F1"), ("ABR", "#8B5CF6"),
    ("H_spec", "#EC4899"), ("H_diff", "#D946EF"), ("Hjorth", "#475569")
]
for idx, (t_sym, t_c) in enumerate(token_items):
    tx = 3.8 + idx * 2.26
    draw_box(tx, 5.5, 2.1, 10.5, bg_color=t_c, edgecolor="black", lw=1.0, r=0.3)
    ax.text(tx + 1.05, 10.75, t_sym, ha='center', va='center', rotation=90,
            fontsize=8.5 * SCALE, fontweight='bold', color="#FFFFFF", zorder=7)


# =============================================================================
# STAGE B CONTENT: TRANSFORMER ENCODER, MHSA MATRIX, DUAL SSL HEADS
# =============================================================================

# Card B1: Input Linear Projection & Sinusoidal Positional Encoding
draw_box(34.2, 70.0, 31.6, 17.0, bg_color="#FFFFFF", edgecolor="black", lw=1.6, r=0.6)
ax.text(35.2, 84.5, "1. Linear Token Projection & Positional Encoding", fontsize=11.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

# Formula badge
draw_box(35.5, 77.0, 29.0, 5.8, bg_color="#FEF3C7", edgecolor="black", lw=1.2, r=0.4)
ax.text(50.0, 79.9, r"$H_0 = \mathrm{LayerNorm}(\mathrm{GELU}(X_i W_{\mathrm{in}} + b_{\mathrm{in}})) + \mathrm{PE} \in \mathbb{R}^{L_{\mathrm{tok}} \times 128}$",
        ha='center', va='center', fontsize=10.0 * SCALE, fontweight='bold', color='#92400E', zorder=7)

# Small schematic: Linear layer + Sinusoidal PE wave
draw_box(35.5, 71.2, 13.8, 4.6, bg_color="#F8FAFC", edgecolor="black", lw=1.0, r=0.4)
ax.text(42.4, 73.5, r"Linear $W_{\mathrm{in}} \in \mathbb{R}^{11 \times 128}$", ha='center', va='center', fontsize=8.8 * SCALE, fontweight='bold', color='#0284C7', zorder=7)

draw_box(50.7, 71.2, 13.8, 4.6, bg_color="#F8FAFC", edgecolor="black", lw=1.0, r=0.4)
ax.text(57.6, 73.5, "Sinusoidal Sequence PE", ha='center', va='center', fontsize=8.8 * SCALE, fontweight='bold', color='#7C3AED', zorder=7)

# Flow arrow B1 -> B2
draw_arrow(50.0, 70.0, 50.0, 66.5, color="#D97706", lw=2.2)

# Card B2: 4-Layer Transformer Encoder with Explicit MHSA Heatmap
draw_box(34.2, 33.5, 31.6, 32.0, bg_color="#FFFFFF", edgecolor="black", lw=1.8, r=0.6)
ax.text(35.2, 62.8, "2. Spatial-Temporal Transformer Encoder (0.57M Params)", fontsize=11.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

# Internal MHSA Block (Left side)
draw_box(35.2, 35.2, 14.5, 25.3, bg_color="#FFFBEB", edgecolor="black", lw=1.2, r=0.5)
ax.text(42.45, 58.3, "Multi-Head Attention", ha='center', fontsize=9.5 * SCALE, fontweight='bold', color='#B45309', zorder=7)
ax.text(42.45, 56.2, "8 Heads (d_k = 16)", ha='center', fontsize=8.2 * SCALE, color='#92400E', zorder=7)

# Draw simulated attention weight heatmap (4x4 colored cells cleanly separated above formula)
np.random.seed(10)
att_map = np.random.uniform(0.1, 1.0, (4, 4))
for r in range(4):
    for c in range(4):
        val = att_map[r, c]
        cell_col = plt.cm.Blues(val * 0.8 + 0.2)
        ax.add_patch(Rectangle((37.6 + c * 2.4, 51.0 - r * 2.4), 2.15, 2.15,
                               facecolor=cell_col, edgecolor="black", lw=0.7, zorder=8))
ax.text(42.45, 39.2, "Q  x  K^T -> Softmax -> V", ha='center', fontsize=7.6 * SCALE, color='#475569', zorder=9)
ax.text(42.45, 36.8, r"$A = \mathrm{softmax}(QK^T/\sqrt{d_k})V$", ha='center', fontsize=8.2 * SCALE, fontweight='bold', color='#1E3A8A', zorder=9)

# Internal FFN Block (Right side)
draw_box(50.8, 35.0, 14.0, 25.5, bg_color="#FEF3C7", edgecolor="black", lw=1.2, r=0.5)
ax.text(57.8, 58.2, "Position-Wise FFN", ha='center', fontsize=9.5 * SCALE, fontweight='bold', color='#B45309', zorder=7)
ax.text(57.8, 56.0, "Dim: 128 -> 512 -> 128", ha='center', fontsize=8.2 * SCALE, color='#92400E', zorder=7)

# Draw FFN layers schematic
draw_box(52.0, 48.0, 11.6, 4.5, bg_color="#FFFFFF", edgecolor="black", lw=1.0, r=0.3)
ax.text(57.8, 50.2, "Linear + GELU (512)", ha='center', va='center', fontsize=8.2 * SCALE, fontweight='bold', color='#0284C7', zorder=8)
draw_arrow(57.8, 48.0, 57.8, 44.5, color="#B45309", lw=1.6)
draw_box(52.0, 40.0, 11.6, 4.5, bg_color="#FFFFFF", edgecolor="black", lw=1.0, r=0.3)
ax.text(57.8, 42.2, "Linear (128) + Norm", ha='center', va='center', fontsize=8.2 * SCALE, fontweight='bold', color='#0284C7', zorder=8)

ax.text(57.8, 36.8, "Residual Connections", ha='center', fontsize=8.0 * SCALE, color='#64748B', zorder=8)

# Stack indicator
draw_box(35.2, 33.8, 29.6, 1.2, bg_color="#F59E0B", edgecolor="none", zorder=7)
ax.text(50.0, 31.8, "x4 Identical Stacked Layers  |  Global Mean Pooling -> h_L in R^128",
        ha='center', fontsize=8.5 * SCALE, fontweight='bold', color='#78350F', zorder=7)

# Flow arrows from Transformer to Dual SSL Heads (Split Branch)
draw_arrow(42.5, 30.5, 40.0, 26.5, color="#DC2626", lw=2.2)
draw_arrow(57.5, 30.5, 60.0, 26.5, color="#2563EB", lw=2.2)

# Card B3: Dual Self-Supervised Objectives
# Branch 1: Masked Token Reconstruction
draw_box(34.2, 4.0, 15.2, 22.0, bg_color="#FFF5F5", edgecolor="black", lw=1.6, r=0.6)
ax.text(41.8, 23.8, "Head 1: Masked Recon", ha='center', fontsize=9.8 * SCALE, fontweight='bold', color='#991B1B', zorder=6)

# Visual masked sequence: 5 small tokens with [?]
mask_tokens = [("δ", "#EF4444"), ("?", "#DC2626"), ("α", "#F59E0B"), ("?", "#DC2626"), ("γ", "#06B6D4")]
for m_i, (m_s, m_col) in enumerate(mask_tokens):
    mx = 35.4 + m_i * 2.5
    is_mask = (m_s == "?")
    draw_box(mx, 17.5, 2.2, 4.2, bg_color="#FEE2E2" if is_mask else m_col,
             edgecolor="red" if is_mask else "black", lw=1.5 if is_mask else 0.8, r=0.3)
    ax.text(mx + 1.1, 19.6, m_s, ha='center', va='center', fontsize=8.5 * SCALE,
            fontweight='bold', color="red" if is_mask else "white", zorder=8)

ax.text(41.8, 15.6, "Mask Ratio p = 20%", ha='center', fontsize=8.0 * SCALE, color='#B91C1C', zorder=7)

draw_box(35.2, 5.2, 13.2, 8.5, bg_color="#FFFFFF", edgecolor="black", lw=1.0, r=0.4)
ax.text(41.8, 11.2, "MSE Spectral Loss:", ha='center', fontsize=8.5 * SCALE, fontweight='bold', color='#991B1B', zorder=8)
ax.text(41.8, 8.2, r"$\mathcal{L}_{\mathrm{recon}} = \frac{1}{|\mathcal{M}|}\sum (x - \hat{x})^2$",
        ha='center', fontsize=8.5 * SCALE, fontweight='bold', color='#B91C1C', zorder=8)

# Branch 2: Contrastive InfoNCE on 3D Hypersphere
draw_box(50.6, 4.0, 15.2, 22.0, bg_color="#EFF6FF", edgecolor="black", lw=1.6, r=0.6)
ax.text(58.2, 23.8, "Head 2: InfoNCE Sphere", ha='center', fontsize=9.8 * SCALE, fontweight='bold', color='#1E40AF', zorder=6)

# Draw 3D Shaded Sphere Schematic
s_cx, s_cy, s_r = 58.2, 17.0, 4.2
sphere = Circle((s_cx, s_cy), s_r, facecolor='#DBEAFE', edgecolor='black', lw=1.4, zorder=7)
ax.add_patch(sphere)
# Equator
ax.add_patch(Ellipse((s_cx, s_cy), s_r*2, s_r*0.8, facecolor='none', edgecolor='#60A5FA', linestyle='--', lw=1.0, zorder=8))

# Unit vectors: Anchor (zi), Positive (zp), Negative (za)
ax.scatter([s_cx - 1.8], [s_cy + 1.5], s=80, color="#16A34A", edgecolor="black", lw=1.0, zorder=10) # Anchor
ax.text(s_cx - 2.8, s_cy + 1.8, "z_i", fontsize=8.0 * SCALE, fontweight='bold', color='#15803D', zorder=11)

ax.scatter([s_cx - 0.5], [s_cy + 2.2], s=80, color="#2563EB", edgecolor="black", lw=1.0, zorder=10) # Positive
ax.text(s_cx + 0.2, s_cy + 2.5, "z_p", fontsize=8.0 * SCALE, fontweight='bold', color='#1D4ED8', zorder=11)

# Green attraction arrow
draw_arrow(s_cx - 1.5, s_cy + 1.6, s_cx - 0.7, s_cy + 2.1, color="#16A34A", lw=1.8)

ax.scatter([s_cx + 2.2], [s_cy - 1.8], s=80, color="#DC2626", edgecolor="black", lw=1.0, zorder=10) # Negative
ax.text(s_cx + 2.6, s_cy - 2.2, "z_a", fontsize=8.0 * SCALE, fontweight='bold', color='#DC2626', zorder=11)

# Red repulsion dashed arrow
draw_arrow(s_cx - 1.5, s_cy + 1.2, s_cx + 1.8, s_cy - 1.5, color="#DC2626", lw=1.4, style="->")

draw_box(51.6, 5.2, 13.2, 6.0, bg_color="#FFFFFF", edgecolor="black", lw=1.0, r=0.4)
ax.text(58.2, 9.2, r"$\mathcal{L}_{\mathrm{contrast}}(\tau=0.07)$", ha='center', fontsize=8.8 * SCALE, fontweight='bold', color='#1E40AF', zorder=8)
ax.text(58.2, 6.8, r"$z_i = g(h_L)/\|g(h_L)\|_2 \in \mathbb{S}^{63}$", ha='center', fontsize=8.0 * SCALE, color='#2563EB', zorder=8)


# =============================================================================
# STAGE C CONTENT: CLINICAL DOWNSTREAM, ZERO-SHOT, FEW-SHOT, CDSS
# =============================================================================

# Card C1: Bidirectional Zero-Shot Transfer
draw_box(70.2, 66.5, 27.1, 20.5, bg_color="#FFFFFF", edgecolor="black", lw=1.6, r=0.6)
ax.text(71.2, 84.5, "1. Bidirectional Zero-Shot Transfer", fontsize=11.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

# Dual Disorder Nodes with Bidirectional Arrows
draw_box(71.5, 71.0, 10.5, 10.0, bg_color="#EFF6FF", edgecolor="black", lw=1.2, r=0.5)
ax.text(76.75, 78.0, "Alcoholism", ha='center', fontsize=9.5 * SCALE, fontweight='bold', color='#1E40AF', zorder=7)
ax.text(76.75, 75.2, "(AUD Cohort)", ha='center', fontsize=8.0 * SCALE, color='#3B82F6', zorder=7)
ax.text(76.75, 72.8, "N=53 Subjects", ha='center', fontsize=8.0 * SCALE, color='#64748B', zorder=7)

draw_box(85.5, 71.0, 10.5, 10.0, bg_color="#FEF2F2", edgecolor="black", lw=1.2, r=0.5)
ax.text(90.75, 78.0, "Dementia", ha='center', fontsize=9.5 * SCALE, fontweight='bold', color='#991B1B', zorder=7)
ax.text(90.75, 75.2, "(AD/FTD)", ha='center', fontsize=8.0 * SCALE, color='#DC2626', zorder=7)
ax.text(90.75, 72.8, "N=88 Subjects", ha='center', fontsize=8.0 * SCALE, color='#64748B', zorder=7)

# Transfer arrows
draw_arrow(82.2, 78.0, 85.2, 78.0, color="#16A34A", lw=2.2, rad=0.2)
ax.text(83.8, 80.8, "0.8676", ha='center', fontsize=8.2 * SCALE, fontweight='bold', color='#15803D', zorder=8)

draw_arrow(85.2, 73.5, 82.2, 73.5, color="#2563EB", lw=2.2, rad=0.2)
ax.text(83.8, 71.0, "0.6744", ha='center', fontsize=8.2 * SCALE, fontweight='bold', color='#1D4ED8', zorder=8)

draw_box(71.5, 67.5, 24.5, 2.8, bg_color="#F0FDF4", edgecolor="black", lw=0.8, r=0.3)
ax.text(83.75, 68.9, "Reveals Continuous Shared Neurodysfunction Axis",
        ha='center', va='center', fontsize=8.2 * SCALE, fontweight='bold', color='#15803D', zorder=7)

# Flow arrow C1 -> C2
draw_arrow(83.75, 66.5, 83.75, 63.0, color="#16A34A", lw=2.2)

# Card C2: Few-Shot Low-Data Regime (K=2 Patients per Class)
draw_box(70.2, 42.5, 27.1, 20.0, bg_color="#FFFFFF", edgecolor="black", lw=1.6, r=0.6)
ax.text(71.2, 59.8, "2. Extreme Few-Shot Clinical Adaptation", fontsize=11.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

# Visual bars for K=2 performance
draw_box(71.5, 51.5, 24.5, 6.2, bg_color="#EFF6FF", edgecolor="black", lw=1.0, r=0.4)
ax.text(72.5, 55.5, "Alzheimer's Disease (K = 2 Patients):", fontsize=8.8 * SCALE, fontweight='bold', color='#1E40AF', zorder=7)
ax.text(72.5, 53.0, "AUROC = 0.8674 ± 0.073  (Matches Full Baseline N=88)", fontsize=8.5 * SCALE, fontweight='bold', color='#2563EB', zorder=7)

draw_box(71.5, 43.8, 24.5, 6.2, bg_color="#FEF2F2", edgecolor="black", lw=1.0, r=0.4)
ax.text(72.5, 47.8, "Alcohol Use Disorder (K = 2 Patients):", fontsize=8.8 * SCALE, fontweight='bold', color='#991B1B', zorder=7)
ax.text(72.5, 45.3, "AUROC = 0.7420 ± 0.065  (Zero Leakage Collapse)", fontsize=8.5 * SCALE, fontweight='bold', color='#DC2626', zorder=7)

# Flow arrow C2 -> C3
draw_arrow(83.75, 42.5, 83.75, 39.0, color="#16A34A", lw=2.2)

# Card C3: Bedside Decision Support System & Edge Telemetry Monitor
draw_box(70.2, 4.0, 27.1, 34.5, bg_color="#FFFFFF", edgecolor="black", lw=1.8, r=0.6)
ax.text(71.2, 36.0, "3. Bedside Clinical Decision Support (CDSS)", fontsize=11.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

# Bedside Monitor Graphic Container
draw_box(71.5, 14.5, 24.5, 19.5, bg_color="#0F172A", edgecolor="black", lw=1.5, r=0.5)
ax.text(72.5, 32.0, "BEDSIDE TELEMETRY MONITOR", fontsize=8.5 * SCALE, fontweight='bold', color='#38BDF8', zorder=7)

# Mini EEG strip on monitor
draw_clean_waveform(72.5, 27.0, 22.5, 3.5, freq=6.0, col="#4ADE80", lw=1.6)

# Gauges and patient probabilities on monitor (with intuitive visual bar & risk arc)
draw_box(72.5, 15.8, 10.5, 9.8, bg_color="#1E293B", edgecolor="#475569", lw=1.0, r=0.4)
ax.text(77.75, 23.8, "AUD Risk Gauge", ha='center', fontsize=8.0 * SCALE, fontweight='bold', color='#94A3B8', zorder=8)
# Visual risk bar meter inside dark box
draw_box(73.6, 21.0, 8.3, 1.3, bg_color="#0F172A", edgecolor="#475569", lw=0.8, r=0.2)
draw_box(73.6, 21.0, 8.3 * 0.771, 1.3, bg_color="#EF4444", edgecolor="none", lw=0.0, r=0.2)
ax.text(77.75, 18.7, "77.1%", ha='center', fontsize=13.5 * SCALE, fontweight='bold', color='#F87171', zorder=8)
ax.text(77.75, 16.6, "HIGH RISK (sub-033)", ha='center', fontsize=7.0 * SCALE, fontweight='bold', color='#FCA5A5', zorder=8)

draw_box(84.0, 15.8, 11.0, 9.8, bg_color="#1E293B", edgecolor="#475569", lw=1.0, r=0.4)
ax.text(89.5, 23.8, "AD Differential", ha='center', fontsize=8.0 * SCALE, fontweight='bold', color='#94A3B8', zorder=8)
# Mini horizontal probability bars
for idx_m, (lbl_m, val_m, col_m, y_m) in enumerate([("AD", 0.853, "#F87171", 21.3), ("FTD", 0.078, "#C084FC", 19.0), ("CN", 0.069, "#4ADE80", 16.7)]):
    ax.text(84.6, y_m + 0.3, f"{lbl_m}", fontsize=7.2 * SCALE, fontweight='bold', color=col_m, zorder=9)
    draw_box(87.0, y_m, 4.4, 1.1, bg_color="#0F172A", edgecolor="#334155", lw=0.6, r=0.15)
    draw_box(87.0, y_m, max(0.35, 4.4 * val_m), 1.1, bg_color=col_m, edgecolor="none", lw=0.0, r=0.15)
    ax.text(94.6, y_m + 0.3, f"{val_m*100:.1f}%", ha='right', fontsize=7.0 * SCALE, fontweight='bold', color="#E2E8F0", zorder=9)

# Hardware Telemetry Chip Badge
draw_box(71.5, 5.5, 24.5, 7.8, bg_color="#F8FAFC", edgecolor="black", lw=1.2, r=0.4)
ax.text(72.5, 10.8, "Ultra-Low Edge Hardware Latency:", fontsize=8.5 * SCALE, fontweight='bold', color='#0F172A', zorder=7)
ax.text(72.5, 8.8, "• Single-Epoch CPU: 3.29 ms (1,058x Real-Time)", fontsize=8.2 * SCALE, fontweight='bold', color='#16A34A', zorder=7)
ax.text(72.5, 6.8, "• Batched GPU: 58.0 μs / epoch (68,736 epochs/s)", fontsize=8.2 * SCALE, fontweight='bold', color='#0284C7', zorder=7)

# Save high-resolution outputs
for d in [FIG_DIR, LATEX_FIG_DIR, DESKTOP_FIG_DIR]:
    png_p = os.path.join(d, "Figure_1_Master_Architecture.png")
    pdf_p = os.path.join(d, "Figure_1_Master_Architecture.pdf")
    fig.savefig(png_p, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_p, format='pdf', bbox_inches='tight')
    print(f"Saved: {png_p} & {pdf_p}")

plt.close(fig)
print("=== FIGURE 1 MASTER ARCHITECTURE V2 COMPLETED SUCCESSFULLY ===")
