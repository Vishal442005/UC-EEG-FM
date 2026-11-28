"""
Publication-Grade Master Architecture & Pipeline Flowchart (Figure 1)
Inspired by BIOT (Yang et al., 2023), LaBraM (Jiang et al., 2024), and DeepTokenEEG (Li et al., 2026).
Design Principles:
1. Pure WHITE (#FFFFFF) main containers and structural boxes (zero pastel shade fills).
2. Crisp simple BLACK (#000000) bold typography, calibrated to (18.5, 11.2) @ 300 DPI
   so minimum printed font size is large and effortlessly legible everywhere.
3. Rich visual shapes and diagrams inside: 10-20 head montages, real multi-channel EEG traces,
   Welch PSD spectrum curve, colored square token blocks, projection grid + PE wave + (+) node,
   explicit MHSA (Q/K/V/Softmax heatmap/FFN MLP graph/residual arcs), Group-K-Fold patient icons,
   masked decoder + reconstructed wave, 3D S^63 sphere, brain icons, and pinned CPU chip.
4. Strict ZERO text overlap or overflow.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Wedge, Rectangle, Polygon

REPO_DIR = r"C:\Users\Visha\Desktop\NewAlcohol\UC_EEG_FM"
FIG_DIR = os.path.join(REPO_DIR, "Figures")
LATEX_FIG_DIR = os.path.join(REPO_DIR, "latex_paper", "figures")
DESKTOP_FIG_DIR = r"C:\Users\Visha\Desktop\NewAlcohol\Figures"
BRAIN_DIR = r"C:\Users\Visha\.gemini\antigravity\brain\a487a487-156f-4781-ba99-0b2ec1ba776a"

for d in [FIG_DIR, LATEX_FIG_DIR, DESKTOP_FIG_DIR, BRAIN_DIR]:
    os.makedirs(d, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'

fig = plt.figure(figsize=(18.5, 11.2), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Clean white background explicitly set
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

# -----------------------------------------------------------------------------
# CLEAN GEOMETRIC DRAWING PRIMITIVES
# -----------------------------------------------------------------------------
def draw_panel_box(x, y, w, h, lw=2.0, ls='-', r=0.8, zorder=2):
    """Pure white structural container with crisp black solid/dashed border."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.2,rounding_size={r}",
                         facecolor="#FFFFFF", edgecolor="#000000",
                         linewidth=lw, linestyle=ls, zorder=zorder)
    ax.add_patch(box)
    return box

def draw_block(x, y, w, h, fc="#FFFFFF", ec="#000000", lw=1.6, ls='-', r=0.45, zorder=4):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.15,rounding_size={r}",
                         facecolor=fc, edgecolor=ec,
                         linewidth=lw, linestyle=ls, zorder=zorder)
    ax.add_patch(box)
    return box

def draw_arrow(x1, y1, x2, y2, color="#000000", lw=2.2, style="-|>", rad=0.0, ls="-", zorder=12):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=f"{style},head_length=7,head_width=4.5",
                            color=color, linewidth=lw, linestyle=ls,
                            connectionstyle=f"arc3,rad={rad}", zorder=zorder)
    ax.add_patch(arrow)

def draw_head_map(cx, cy, r, num_dots=19, dot_col="#2563EB", zorder=6):
    ax.add_patch(Circle((cx, cy), r, facecolor="#FFFFFF", edgecolor="#000000", lw=1.5, zorder=zorder))
    nose = Polygon([[cx - r*0.22, cy + r*0.95], [cx, cy + r*1.25], [cx + r*0.22, cy + r*0.95]],
                   closed=True, facecolor="#FFFFFF", edgecolor="#000000", lw=1.3, zorder=zorder)
    ax.add_patch(nose)
    ax.add_patch(Ellipse((cx - r*1.03, cy), r*0.18, r*0.42, facecolor="#FFFFFF", edgecolor="#000000", lw=1.2, zorder=zorder))
    ax.add_patch(Ellipse((cx + r*1.03, cy), r*0.18, r*0.42, facecolor="#FFFFFF", edgecolor="#000000", lw=1.2, zorder=zorder))
    rng = np.random.default_rng(100 + num_dots)
    ang = rng.uniform(0, 2*np.pi, num_dots)
    rad = r * 0.78 * np.sqrt(rng.uniform(0.05, 1.0, num_dots))
    ax.scatter(cx + rad*np.cos(ang), cy + rad*np.sin(ang), s=22, color=dot_col, edgecolor="#000000", lw=0.6, zorder=zorder+1)

def draw_eeg_trace(x, y, w, h, freq=4.0, col="#000000", lw=1.8, seed=1, zorder=6):
    t = np.linspace(0, w, 120)
    sig = 0.55*np.sin(2*np.pi*freq*(t/w)) + 0.30*np.cos(2*np.pi*freq*2.4*(t/w) + seed)
    sig = sig / (np.max(np.abs(sig)) + 1e-6) * (h * 0.44)
    ax.plot(x + t, y + h/2.0 + sig, color=col, lw=lw, zorder=zorder)

def draw_human(cx, cy, s=1.0, col="#16A34A", zorder=8):
    ax.add_patch(Circle((cx, cy + 1.05*s), 0.62*s, facecolor=col, edgecolor="#000000", lw=1.2, zorder=zorder))
    ax.add_patch(Wedge((cx, cy - 0.6*s), 1.18*s, 0, 180, facecolor=col, edgecolor="#000000", lw=1.2, zorder=zorder))

def draw_brain_icon(cx, cy, s=1.0, hot_col="#2563EB", hot_side="left", zorder=7):
    pts = np.array([
        [-2.2, -0.3], [-2.1, 1.1], [-1.3, 2.0], [0.0, 2.3], [1.4, 2.0],
        [2.2, 1.0], [2.3, -0.3], [1.6, -1.1], [0.4, -1.1], [-0.4, -0.7], [-1.5, -0.9]
    ]) * s
    pts[:, 0] += cx
    pts[:, 1] += cy
    ax.add_patch(Polygon(pts, closed=True, facecolor="#FFFFFF", edgecolor="#000000", lw=1.6, zorder=zorder))
    hx = cx - 0.9*s if hot_side == "left" else cx + 0.8*s
    ax.add_patch(Circle((hx, cy + 0.7*s), 0.78*s, facecolor=hot_col, alpha=0.65, edgecolor="#000000", lw=1.1, zorder=zorder+1))

# =============================================================================
# UPPER TIER: (a) END-TO-END UC-EEG-FM ARCHITECTURE (LEFT-TO-RIGHT FLOW)
# y: 47.5 to 98.0 (pure white panel, crisp black outline)
# =============================================================================
draw_panel_box(1.2, 47.5, 97.6, 50.8, lw=2.2, ls='-', r=0.8)
ax.text(2.6, 95.3, "a  UC-EEG-FM Architecture: Hardware-Agnostic Tokenization & Spatial-Temporal Transformer Encoder",
        fontsize=18.5, fontweight='bold', color='#000000', zorder=5)

# --- MODULE 1: Multi-Cohort Inputs & CAR Standardization (x: 2.6 to 22.8) ---
draw_panel_box(2.6, 49.4, 20.0, 43.2, lw=1.6, ls='--', r=0.6, zorder=3)
ax.text(12.6, 90.2, "1. Heterogeneous Inputs", ha='center', fontsize=15.5, fontweight='bold', color='#000000', zorder=5)

# 3 Mini Scalp Topographies across top
for lbl, ch, cx, col, nd in [("AUD", "64ch", 6.0, "#2563EB", 28), ("AD/FTD", "19ch", 12.6, "#DC2626", 19), ("Epilepsy", "23ch", 19.2, "#9333EA", 23)]:
    draw_head_map(cx, 84.4, r=2.1, num_dots=nd, dot_col=col)
    ax.text(cx, 80.5, lbl, ha='center', fontsize=13.0, fontweight='bold', color='#000000', zorder=7)
    ax.text(cx, 78.2, ch, ha='center', fontsize=12.0, fontweight='bold', color='#000000', zorder=7)

draw_arrow(12.6, 77.2, 12.6, 74.6, lw=2.2)

# Stacked Multi-Channel Raw Waveforms inside clean grid box
draw_block(3.6, 58.6, 18.0, 15.6, fc="#FFFFFF", ec="#000000", lw=1.4)
for idx_t, (ch_n, c_tr) in enumerate([("Fp1", "#2563EB"), ("C3", "#DC2626"), ("Pz", "#16A34A"), ("O1", "#9333EA")]):
    wy = 70.6 - idx_t * 3.3
    ax.text(4.3, wy + 0.6, ch_n, fontsize=12.5, fontweight='bold', color='#000000', zorder=7)
    draw_eeg_trace(8.0, wy, 12.6, 2.2, freq=3.2 + idx_t*1.5, col=c_tr, lw=1.8, seed=idx_t)

# CAR & 4.0s Filtering Block at bottom
draw_arrow(12.6, 58.6, 12.6, 56.4, lw=2.2)
draw_block(3.6, 50.8, 18.0, 5.2, fc="#FFFFFF", ec="#000000", lw=1.5)
ax.text(12.6, 53.4, "CAR + 0.5–45 Hz (4.0s)", ha='center', va='center', fontsize=13.8, fontweight='bold', color='#000000', zorder=7)

# Main Flow Arrow 1 -> 2
draw_arrow(22.8, 70.5, 25.0, 70.5, lw=2.8)

# --- MODULE 2: Channel-Agnostic Spectral Tokenizer & Embedding (x: 25.2 to 48.0) ---
draw_panel_box(25.2, 49.4, 22.8, 43.2, lw=1.6, ls='--', r=0.6, zorder=3)
ax.text(36.6, 90.2, "2. Biosignal Tokenizer", ha='center', fontsize=15.5, fontweight='bold', color='#000000', zorder=5)

# Welch PSD Graphic
draw_block(26.4, 74.6, 20.4, 13.0, fc="#FFFFFF", ec="#000000", lw=1.4)
ax.text(36.6, 85.4, "Welch PSD Spectrum", ha='center', fontsize=13.5, fontweight='bold', color='#000000', zorder=7)
freqs = np.linspace(0.5, 45, 100)
psd = 1.0 / (freqs**1.15) + 0.35 * np.exp(-((freqs - 10)**2)/5.5)
psd_n = (psd - np.min(psd)) / (np.max(psd) - np.min(psd)) * 7.0
for f0, f1, bn, bc in [(0.5,4,"δ","#EF4444"),(4,8,"θ","#F97316"),(8,13,"α","#F59E0B"),(13,30,"β","#10B981"),(30,45,"γ","#06B6D4")]:
    m = (freqs >= f0) & (freqs <= f1)
    xs = 27.6 + (freqs[m]-0.5)/44.5 * 18.0
    ys = 76.2 + psd_n[m]
    ax.fill_between(xs, 76.2, ys, color=bc, alpha=0.45, zorder=6)
    ax.plot(xs, ys, color=bc, lw=2.2, zorder=7)
    ax.text(27.6 + ((f0+f1)/2 - 0.5)/44.5 * 18.0, 76.8, bn, ha='center', fontsize=12.5, fontweight='bold', color='#000000', zorder=8)

draw_arrow(36.6, 74.6, 36.6, 71.8, lw=2.2)

# 11 Square Colored Biomarker Token Blocks (inspired by LaBraM/BIOT token squares)
tok_cols = ["#EF4444","#F97316","#F59E0B","#10B981","#06B6D4","#3B82F6","#6366F1","#8B5CF6","#EC4899","#D946EF","#475569"]
tok_names = ["δ","θ","α","β","γ","TAR","TBR","ABR","Hsp","Hdf","Hjr"]
for ti, (tn, tc) in enumerate(zip(tok_names, tok_cols)):
    row = 0 if ti < 6 else 1
    col = ti if ti < 6 else ti - 6
    tx = 26.8 + col * 3.35 + (1.6 if row == 1 else 0.0)
    ty = 67.2 - row * 3.8
    ax.add_patch(Rectangle((tx, ty), 2.9, 3.1, facecolor=tc, edgecolor="#000000", lw=1.2, zorder=7))
    ax.text(tx + 1.45, ty + 1.55, tn, ha='center', va='center', fontsize=11.8, fontweight='bold', color='#FFFFFF', zorder=8)

draw_arrow(36.6, 62.6, 36.6, 60.2, lw=2.2)

# Linear W_in + (+) PE Wave -> H0 (Pure shapes!)
draw_block(26.4, 51.0, 6.2, 8.6, fc="#FFFFFF", ec="#000000", lw=1.4)
for ri in range(3):
    for ci in range(3):
        ax.add_patch(Rectangle((27.2+ci*1.6, 56.8-ri*1.6), 1.3, 1.3, facecolor="#3B82F6", alpha=0.35+0.2*ci, edgecolor="#000000", lw=0.7, zorder=7))
ax.text(29.5, 52.3, r"$W_{\mathrm{in}}$", ha='center', fontsize=13.5, fontweight='bold', color='#000000', zorder=8)

draw_arrow(32.6, 55.3, 34.4, 55.3, lw=2.0)
ax.add_patch(Circle((35.8, 55.3), 1.35, facecolor="#FFFFFF", edgecolor="#000000", lw=1.6, zorder=8))
ax.text(35.8, 55.3, "+", ha='center', va='center', fontsize=18, fontweight='bold', color='#000000', zorder=9)

draw_arrow(39.4, 55.3, 37.2, 55.3, lw=2.0)
draw_block(39.6, 51.0, 7.2, 8.6, fc="#FFFFFF", ec="#000000", lw=1.4)
t_sin = np.linspace(0, 5.4, 80)
ax.plot(40.5 + t_sin, 56.5 + 1.6*np.sin(2*np.pi*t_sin/2.7), color="#9333EA", lw=2.2, zorder=7)
ax.text(43.2, 52.3, "Sin. PE", ha='center', fontsize=13.0, fontweight='bold', color='#000000', zorder=8)

# Main Flow Arrow 2 -> 3
draw_arrow(48.0, 55.3, 50.6, 55.3, lw=2.8)
ax.text(49.3, 57.5, r"$H_0$", ha='center', fontsize=14.0, fontweight='bold', color='#000000', zorder=8)

# --- MODULE 3: Transformer Encoder Stack (BIOT/LaBraM Style, x: 50.8 to 97.4) ---
draw_panel_box(50.8, 49.4, 46.6, 43.2, lw=1.6, ls='--', r=0.6, zorder=3)
ax.text(74.1, 90.2, "3. Spatial-Temporal Transformer Encoder (0.57M Params, ×4 Blocks)", ha='center', fontsize=15.5, fontweight='bold', color='#000000', zorder=5)

# Left-to-Right Visual Architecture Inside Transformer Block (white boxes + colored inner tensors/heatmaps/MLP)
# Stage A: LayerNorm 1
draw_block(52.4, 62.0, 4.0, 17.5, fc="#FFFFFF", ec="#000000", lw=1.5)
ax.text(54.4, 70.75, "LayerNorm", ha='center', va='center', rotation=90, fontsize=14.0, fontweight='bold', color='#000000', zorder=7)

draw_arrow(56.4, 70.75, 58.6, 70.75, lw=2.2)

# Stage B: Multi-Head Self-Attention (Q, K, V strips + Attention Heatmap)
draw_block(58.8, 52.0, 15.6, 35.0, fc="#FFFFFF", ec="#000000", lw=1.6)
ax.text(66.6, 84.4, "MHSA (4 Heads)", ha='center', fontsize=14.5, fontweight='bold', color='#000000', zorder=7)

for qi, (qlbl, qcol) in enumerate([("Q", "#EF4444"), ("K", "#F59E0B"), ("V", "#2563EB")]):
    qx = 60.4 + qi * 4.4
    ax.add_patch(Rectangle((qx, 77.6), 3.4, 4.2, facecolor=qcol, edgecolor="#000000", lw=1.2, zorder=7))
    ax.text(qx + 1.7, 79.7, qlbl, ha='center', va='center', fontsize=14.5, fontweight='bold', color='#FFFFFF', zorder=8)
    draw_arrow(qx + 1.7, 77.6, qx + 1.7, 74.8, lw=1.8)

np.random.seed(10)
att_m = np.random.uniform(0.15, 1.0, (4, 4))
for r_i in range(4):
    for c_i in range(4):
        ax.add_patch(Rectangle((61.0 + c_i*2.8, 70.6 - r_i*2.8), 2.5, 2.5,
                               facecolor=plt.cm.Blues(att_m[r_i, c_i]*0.8 + 0.2), edgecolor="#000000", lw=0.8, zorder=8))
ax.text(66.6, 56.2, r"$\mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V$", ha='center', fontsize=14.5, fontweight='bold', color='#000000', zorder=8)

# Residual Arc 1 (around bottom of MHSA to (+) Add 1)
ax.add_patch(Circle((76.6, 70.75), 1.4, facecolor="#FFFFFF", edgecolor="#000000", lw=1.6, zorder=9))
ax.text(76.6, 70.75, "+", ha='center', va='center', fontsize=18, fontweight='bold', color='#000000', zorder=10)
draw_arrow(74.4, 70.75, 75.2, 70.75, lw=2.0)
# Skip bypass line over top
ax.plot([51.6, 51.6, 76.6, 76.6], [70.75, 88.2, 88.2, 72.3], color="#000000", lw=1.5, ls="--", zorder=6)

draw_arrow(78.0, 70.75, 79.8, 70.75, lw=2.0)

# Stage C: LayerNorm 2 + FFN MLP Node-Link Graph
draw_block(80.0, 52.0, 12.2, 35.0, fc="#FFFFFF", ec="#000000", lw=1.6)
ax.text(86.1, 84.4, "FFN + LN", ha='center', fontsize=14.5, fontweight='bold', color='#000000', zorder=7)
ax.text(86.1, 81.2, "128→256→128", ha='center', fontsize=12.5, fontweight='bold', color='#000000', zorder=7)

lx_arr = [82.0, 86.1, 90.2]
ly_arr = [np.linspace(60.0, 76.0, 4), np.linspace(57.5, 78.0, 6), np.linspace(60.0, 76.0, 4)]
for y1 in ly_arr[0]:
    for y2 in ly_arr[1]:
        ax.plot([lx_arr[0], lx_arr[1]], [y1, y2], color="#64748B", lw=0.9, alpha=0.6, zorder=6)
for y2 in ly_arr[1]:
    for y3 in ly_arr[2]:
        ax.plot([lx_arr[1], lx_arr[2]], [y2, y3], color="#0284C7", lw=0.9, alpha=0.6, zorder=6)
for li, (lx, ys) in enumerate(zip(lx_arr, ly_arr)):
    nc = "#2563EB" if li != 1 else "#EA580C"
    ax.scatter([lx]*len(ys), ys, s=110, color=nc, edgecolor="#000000", lw=1.2, zorder=8)
ax.text(86.1, 54.0, "GELU", ha='center', fontsize=13.5, fontweight='bold', color='#000000', zorder=8)

# Output Vector Strip h_L
draw_arrow(92.2, 70.75, 93.8, 70.75, lw=2.2)
draw_block(94.0, 56.0, 2.4, 29.0, fc="#FFFFFF", ec="#000000", lw=1.5)
for k_h in range(8):
    ax.add_patch(Rectangle((94.45, 58.0 + k_h*3.2), 1.5, 2.5, facecolor=plt.cm.viridis(k_h/8.0), edgecolor="#000000", lw=0.7, zorder=7))
ax.text(95.2, 52.0, r"$h_L$", ha='center', fontsize=15.0, fontweight='bold', color='#000000', zorder=8)

# =============================================================================
# LOWER TIER LEFT: (b) GROUP-K-FOLD & DUAL SELF-SUPERVISED PRE-TRAINING
# x: 1.2 to 51.0, y: 2.0 to 45.2
# =============================================================================
draw_panel_box(1.2, 2.0, 49.8, 43.2, lw=2.2, ls='-', r=0.8)
ax.text(2.6, 42.2, "b  Leakage-Free Pre-Training & Dual SSL Objectives",
        fontsize=17.5, fontweight='bold', color='#000000', zorder=5)

# Top Strip inside b: DeepTokenEEG-style Subject-Disjoint Group K-Fold Visual
draw_block(2.6, 28.4, 47.0, 11.8, fc="#FFFFFF", ec="#000000", lw=1.5)
ax.text(4.0, 37.8, "Strict Subject-Disjoint Group K-Fold:", fontsize=14.0, fontweight='bold', color='#000000', zorder=7)
ax.text(35.0, 37.8, r"$\{S_{\mathrm{train}}\} \cap \{S_{\mathrm{test}}\} = \emptyset$", fontsize=14.5, fontweight='bold', color='#000000', zorder=7)

# Draw Train Humans vs Test Humans separated by dashed vertical line
for hi in range(5):
    draw_human(6.5 + hi*4.6, 32.0, s=1.05, col="#16A34A" if hi % 2 == 0 else "#DC2626")
ax.text(15.7, 29.3, "Train Cohort Subjects", ha='center', fontsize=13.0, fontweight='bold', color='#000000', zorder=8)

ax.plot([29.5, 29.5], [29.0, 36.6], color="#000000", lw=2.0, ls="--", zorder=8)

for hi in range(3):
    draw_human(34.0 + hi*5.0, 32.0, s=1.05, col="#16A34A" if hi == 1 else "#DC2626")
ax.text(39.0, 29.3, "Held-Out Test Subjects", ha='center', fontsize=13.0, fontweight='bold', color='#000000', zorder=8)

# Branch 1 (Bottom-Left): Masked Reconstruction
draw_block(2.6, 3.6, 22.8, 23.2, fc="#FFFFFF", ec="#000000", lw=1.5)
ax.text(14.0, 24.4, "1. Masked Recon (15%)", ha='center', fontsize=14.5, fontweight='bold', color='#000000', zorder=7)

for mi, (ms, mc) in enumerate([("δ","#EF4444"),("?","#DC2626"),("α","#F59E0B"),("?","#DC2626"),("γ","#06B6D4")]):
    mx = 4.2 + mi * 4.0
    ism = (ms == "?")
    ax.add_patch(Rectangle((mx, 17.6), 3.3, 4.2, facecolor="#FFFFFF" if ism else mc,
                           edgecolor="#DC2626" if ism else "#000000", lw=2.0 if ism else 1.2, ls="--" if ism else "-", zorder=7))
    ax.text(mx + 1.65, 19.7, ms, ha='center', va='center', fontsize=14.5, fontweight='bold', color='#DC2626' if ism else '#FFFFFF', zorder=8)

draw_arrow(14.0, 17.2, 14.0, 14.6, lw=2.0)
draw_eeg_trace(4.6, 9.2, 18.8, 4.6, freq=3.6, col="#DC2626", lw=2.2, seed=4)
ax.text(14.0, 5.6, r"$\mathcal{L}_{\mathrm{recon}}\ (\mathrm{MSE})$", ha='center', fontsize=14.5, fontweight='bold', color='#000000', zorder=8)

# Branch 2 (Bottom-Right): Contrastive InfoNCE on S^63 Sphere
draw_block(26.8, 3.6, 22.8, 23.2, fc="#FFFFFF", ec="#000000", lw=1.5)
ax.text(38.2, 24.4, r"2. Contrastive $\mathbb{S}^{63}$", ha='center', fontsize=14.5, fontweight='bold', color='#000000', zorder=7)

scx, scy, sr = 38.2, 15.2, 5.8
ax.add_patch(Circle((scx, scy), sr, facecolor="#FFFFFF", edgecolor="#000000", lw=1.8, zorder=6))
ax.add_patch(Ellipse((scx, scy), sr*2, sr*0.68, facecolor="none", edgecolor="#94A3B8", ls="--", lw=1.2, zorder=7))
ax.add_patch(Ellipse((scx, scy), sr*0.68, sr*2, facecolor="none", edgecolor="#94A3B8", ls="--", lw=1.2, zorder=7))

ax.scatter([scx - 2.4], [scy + 1.8], s=140, color="#16A34A", edgecolor="#000000", lw=1.2, zorder=9)
ax.text(scx - 4.2, scy + 2.5, r"$z_i$", fontsize=14.0, fontweight='bold', color='#000000', zorder=10)
ax.scatter([scx - 0.3], [scy + 3.0], s=140, color="#2563EB", edgecolor="#000000", lw=1.2, zorder=9)
ax.text(scx + 0.8, scy + 3.2, r"$z_p$", fontsize=14.0, fontweight='bold', color='#000000', zorder=10)
draw_arrow(scx - 2.0, scy + 2.0, scx - 0.7, scy + 2.8, color="#16A34A", lw=2.2)

ax.scatter([scx + 2.8], [scy - 2.2], s=140, color="#DC2626", edgecolor="#000000", lw=1.2, zorder=9)
ax.text(scx + 3.6, scy - 3.0, r"$z_a$", fontsize=14.0, fontweight='bold', color='#000000', zorder=10)
draw_arrow(scx - 1.9, scy + 1.4, scx + 2.2, scy - 1.7, color="#DC2626", lw=1.8, ls="--")

ax.text(38.2, 5.6, r"$\mathcal{L}_{\mathrm{contrast}}\ (\tau = 0.07)$", ha='center', fontsize=14.5, fontweight='bold', color='#000000', zorder=8)

# =============================================================================
# LOWER TIER RIGHT: (c) DOWNSTREAM TRANSFER & BEDSIDE TELEMETRY
# x: 52.8 to 98.8, y: 2.0 to 45.2
# =============================================================================
draw_panel_box(52.8, 2.0, 46.0, 43.2, lw=2.2, ls='-', r=0.8)
ax.text(54.2, 42.2, "c  Downstream Clinical Transfer & Bedside Telemetry",
        fontsize=17.5, fontweight='bold', color='#000000', zorder=5)

# Block C1: Cross-Disorder Zero-Shot Bridge (Top-Left of c)
draw_block(54.2, 23.8, 21.0, 16.4, fc="#FFFFFF", ec="#000000", lw=1.5)
ax.text(64.7, 38.0, "Zero-Shot Transfer", ha='center', fontsize=14.5, fontweight='bold', color='#000000', zorder=7)
draw_brain_icon(58.6, 31.8, s=1.05, hot_col="#2563EB", hot_side="left")
ax.text(58.6, 26.2, "AUD (53)", ha='center', fontsize=13.5, fontweight='bold', color='#000000', zorder=8)
draw_brain_icon(70.8, 31.8, s=1.05, hot_col="#DC2626", hot_side="right")
ax.text(70.8, 26.2, "AD (88)", ha='center', fontsize=13.5, fontweight='bold', color='#000000', zorder=8)

draw_arrow(61.5, 33.4, 67.8, 33.4, color="#16A34A", lw=2.4, rad=0.22)
ax.text(64.7, 35.2, "0.868", ha='center', fontsize=13.5, fontweight='bold', color='#000000', zorder=9)
draw_arrow(67.8, 30.0, 61.5, 30.0, color="#2563EB", lw=2.4, rad=0.22)
ax.text(64.7, 27.6, "0.674", ha='center', fontsize=13.5, fontweight='bold', color='#000000', zorder=9)

# Block C2: Few-Shot Adaptation K=2 (Top-Right of c)
draw_block(76.6, 23.8, 20.8, 16.4, fc="#FFFFFF", ec="#000000", lw=1.5)
ax.text(87.0, 38.0, "Few-Shot (K=2)", ha='center', fontsize=14.5, fontweight='bold', color='#000000', zorder=7)
draw_human(79.4, 33.0, s=0.95, col="#2563EB")
draw_human(82.4, 33.0, s=0.95, col="#DC2626")
t_fs = np.linspace(0, 6.2, 30)
ax.plot(77.8 + t_fs, 26.0 + 3.8*(1 - np.exp(-0.7*t_fs)), color="#16A34A", lw=2.6, zorder=8)
ax.scatter([78.5], [27.6], s=90, color="#F59E0B", edgecolor="#000000", lw=1.0, marker="*", zorder=9)

# Clean AD & AUD performance bars with black text
ax.text(86.0, 33.8, "AD: 0.867", fontsize=13.5, fontweight='bold', color='#000000', zorder=8)
ax.add_patch(Rectangle((86.0, 31.2), 9.6, 1.8, facecolor="#FFFFFF", edgecolor="#000000", lw=1.0, zorder=7))
ax.add_patch(Rectangle((86.0, 31.2), 9.6*0.867, 1.8, facecolor="#2563EB", edgecolor="none", zorder=8))

ax.text(86.0, 28.4, "AUD: 0.742", fontsize=13.5, fontweight='bold', color='#000000', zorder=8)
ax.add_patch(Rectangle((86.0, 25.8), 9.6, 1.8, facecolor="#FFFFFF", edgecolor="#000000", lw=1.0, zorder=7))
ax.add_patch(Rectangle((86.0, 25.8), 9.6*0.742, 1.8, facecolor="#DC2626", edgecolor="none", zorder=8))

# Block C3: Bedside CDSS Telemetry & CPU Edge Chip (Bottom of c)
draw_block(54.2, 3.6, 43.2, 18.8, fc="#FFFFFF", ec="#000000", lw=1.5)
ax.text(55.4, 20.0, "Bedside CDSS Monitor & Edge Deployment", fontsize=14.5, fontweight='bold', color='#000000', zorder=7)

# Left visual: Clinical Waveform + Risk Speedometer Arc
draw_block(55.4, 5.0, 21.8, 13.6, fc="#FFFFFF", ec="#000000", lw=1.3)
draw_eeg_trace(56.4, 13.8, 19.8, 3.8, freq=6.0, col="#16A34A", lw=2.0, seed=7)
ax.add_patch(Wedge((61.0, 6.8), 4.0, 0, 180, width=1.2, facecolor="#E2E8F0", edgecolor="#000000", lw=0.8, zorder=7))
ax.add_patch(Wedge((61.0, 6.8), 4.0, 180*(1-0.771), 180, width=1.2, facecolor="#DC2626", edgecolor="none", zorder=8))
ax.text(61.0, 7.6, "77.1%", ha='center', fontsize=13.5, fontweight='bold', color='#000000', zorder=9)
ax.text(61.0, 5.4, "AUD Risk", ha='center', fontsize=12.0, fontweight='bold', color='#000000', zorder=9)

for lm, vm, cm, ym in [("AD",0.853,"#DC2626",11.0),("FTD",0.078,"#9333EA",8.5),("CN",0.069,"#16A34A",6.0)]:
    ax.text(66.5, ym+0.3, lm, fontsize=11.8, fontweight='bold', color='#000000', zorder=8)
    ax.add_patch(Rectangle((70.2, ym), 5.8, 1.5, facecolor="#FFFFFF", edgecolor="#000000", lw=0.8, zorder=7))
    ax.add_patch(Rectangle((70.2, ym), max(0.5, 5.8*vm), 1.5, facecolor=cm, edgecolor="none", zorder=8))

# Right visual: Silicon CPU Chip + Latency Badges
draw_block(78.4, 5.0, 17.8, 13.6, fc="#FFFFFF", ec="#000000", lw=1.3)
cx_c, cy_c = 82.2, 11.8
for off in [-1.2, 0.0, 1.2]:
    ax.plot([cx_c+off, cx_c+off], [cy_c-2.6, cy_c+2.6], color="#000000", lw=2.0, zorder=6)
    ax.plot([cx_c-2.6, cx_c+2.6], [cy_c+off, cy_c+off], color="#000000", lw=2.0, zorder=6)
draw_block(cx_c - 1.8, cy_c - 1.8, 3.6, 3.6, fc="#1E293B", ec="#000000", lw=1.4, r=0.2, zorder=8)
ax.text(cx_c, cy_c, "CPU", ha='center', va='center', fontsize=12.0, fontweight='bold', color='#FFFFFF', zorder=9)

ax.text(86.2, 13.8, "3.29 ms", fontsize=16.5, fontweight='bold', color='#000000', zorder=8)
ax.text(86.2, 10.6, "1,058× RT", fontsize=14.0, fontweight='bold', color='#000000', zorder=8)
ax.text(86.2, 6.8, "GPU: 58 μs", fontsize=13.5, fontweight='bold', color='#000000', zorder=8)

for d in [FIG_DIR, LATEX_FIG_DIR, DESKTOP_FIG_DIR, BRAIN_DIR]:
    png_p = os.path.join(d, "Figure_1_Master_Architecture.png")
    pdf_p = os.path.join(d, "Figure_1_Master_Architecture.pdf")
    fig.savefig(png_p, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    fig.savefig(pdf_p, format='pdf', bbox_inches='tight', facecolor='#FFFFFF')

plt.close(fig)
print("=== REDESIGNED PURE-WHITE BLACK-TEXT HYPER-VISUAL FIGURE 1 COMPLETED ===")
