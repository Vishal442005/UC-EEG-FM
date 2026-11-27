"""
Hyper-Visual Master System Architecture & Clinical Translation Framework (Figure 1)
- Runs cleanly on healthy workspace: C:/Users/Visha/Desktop/NewAlcohol/UC_EEG_FM
- Canvas calibrated to (20.0, 12.2) at 300 DPI (6000x3660 px) with large bold fonts (min 12.0-24pt).
- Completely visual block diagrams: matrix blocks, sinusoidal waves, CAR DSP node, Q/K/V split + attention heatmap,
  MLP synaptic node-link network, masked reconstruction wave, 3D-shaded S^63 hypersphere, brain silhouettes,
  few-shot mini curve, semi-circular risk speedometer, and silicon CPU chip with pins.
- Strict black-and-white structural outlines (edgecolor='black', lw=2.0), rich interior color, ZERO overlap.
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

fig = plt.figure(figsize=(20.0, 12.2), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

def draw_box(x, y, w, h, bg_color="#FFFFFF", edgecolor="black", lw=1.8, r=0.6, zorder=2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.2,rounding_size={r}",
                         facecolor=bg_color, edgecolor=edgecolor,
                         linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box

def draw_arrow(x1, y1, x2, y2, color="#1E293B", lw=2.4, style="-|>", rad=0.0, ls="-", zorder=12):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=f"{style},head_length=7,head_width=4.5",
                            color=color, linewidth=lw, linestyle=ls,
                            connectionstyle=f"arc3,rad={rad}", zorder=zorder)
    ax.add_patch(arrow)

def draw_head_topomap(cx, cy, r, num_dots=19, color="#0284C7", zorder=6):
    c = Circle((cx, cy), r, facecolor="#F8FAFC", edgecolor="black", lw=1.5, zorder=zorder)
    ax.add_patch(c)
    nose = Polygon([[cx - r*0.22, cy + r*0.95], [cx, cy + r*1.25], [cx + r*0.22, cy + r*0.95]],
                   closed=True, facecolor="#F8FAFC", edgecolor="black", lw=1.3, zorder=zorder)
    ax.add_patch(nose)
    ax.add_patch(Ellipse((cx - r*1.03, cy), r*0.18, r*0.42, facecolor="#E2E8F0", edgecolor="black", lw=1.2, zorder=zorder))
    ax.add_patch(Ellipse((cx + r*1.03, cy), r*0.18, r*0.42, facecolor="#E2E8F0", edgecolor="black", lw=1.2, zorder=zorder))
    rng = np.random.default_rng(42 + num_dots)
    angles = rng.uniform(0, 2*np.pi, num_dots)
    radii = r * 0.78 * np.sqrt(rng.uniform(0.05, 1.0, num_dots))
    ax.scatter(cx + radii * np.cos(angles), cy + radii * np.sin(angles),
               s=22, color=color, edgecolor="black", lw=0.6, zorder=zorder + 1)

def draw_waveform(x, y, w, h, freq=4.0, col="#1E40AF", lw=2.0, seed=1, zorder=6):
    t = np.linspace(0, w, 140)
    sig = 0.52 * np.sin(2 * np.pi * freq * (t / w)) + 0.28 * np.cos(2 * np.pi * freq * 2.3 * (t / w) + seed)
    sig_norm = sig / (np.max(np.abs(sig)) + 1e-6) * (h * 0.44)
    ax.plot(x + t, y + h / 2.0 + sig_norm, color=col, lw=lw, zorder=zorder)

def draw_patient_icon(cx, cy, s=1.0, col="#2563EB", zorder=8):
    head = Circle((cx, cy + 0.9*s), 0.55*s, facecolor=col, edgecolor="black", lw=1.2, zorder=zorder)
    body = Wedge((cx, cy - 0.5*s), 1.05*s, 0, 180, facecolor=col, edgecolor="black", lw=1.2, zorder=zorder)
    ax.add_patch(head)
    ax.add_patch(body)

def draw_brain_silhouette(cx, cy, s=1.0, fill="#DBEAFE", hot_col="#2563EB", hot_side="left", zorder=7):
    pts = np.array([
        [-2.2, -0.3], [-2.1, 1.1], [-1.3, 2.0], [0.0, 2.3], [1.4, 2.0],
        [2.2, 1.0], [2.3, -0.3], [1.6, -1.1], [0.4, -1.1], [-0.4, -0.7], [-1.5, -0.9]
    ]) * s
    pts[:, 0] += cx
    pts[:, 1] += cy
    ax.add_patch(Polygon(pts, closed=True, facecolor=fill, edgecolor="black", lw=1.5, zorder=zorder))
    hx = cx - 0.9*s if hot_side == "left" else cx + 0.8*s
    hy = cy + 0.7*s
    ax.add_patch(Circle((hx, hy), 0.75*s, facecolor=hot_col, alpha=0.55, edgecolor=hot_col, lw=1.2, zorder=zorder+1))

# -----------------------------------------------------------------------------
# 1. TOP TITLE BANNER
# -----------------------------------------------------------------------------
draw_box(1.2, 93.4, 97.6, 5.4, bg_color="#FFFFFF", edgecolor="black", lw=2.2, r=0.8)
ax.text(50.0, 96.7, "Unified Cross-Disorder EEG Foundation Model (UC-EEG-FM)",
        ha='center', va='center', fontsize=24, fontweight='bold', color='#0F172A', zorder=5)
ax.text(50.0, 94.4, "Visual Architecture: Hardware-Agnostic Tokenization, Dual Self-Supervised Backbone, and Bedside Translation",
        ha='center', va='center', fontsize=15.5, fontweight='bold', color='#475569', zorder=5)

# -----------------------------------------------------------------------------
# 2. THREE PRIMARY COLUMNS (STAGES a, b, c)
# -----------------------------------------------------------------------------
draw_box(1.2, 1.8, 30.2, 90.0, bg_color="#F8FAFC", edgecolor="black", lw=2.2, r=1.0)
ax.text(2.4, 89.2, "a  Ingestion & Harmonization", fontsize=19, fontweight='bold', color='#0F172A', zorder=5)

draw_box(33.2, 1.8, 33.6, 90.0, bg_color="#FFFDF5", edgecolor="black", lw=2.2, r=1.0)
ax.text(34.4, 89.2, "b  Dual SSL Foundation Backbone", fontsize=19, fontweight='bold', color='#0F172A', zorder=5)

draw_box(68.6, 1.8, 30.2, 90.0, bg_color="#F0FDF4", edgecolor="black", lw=2.2, r=1.0)
ax.text(69.8, 89.2, "c  Transfer & Bedside CDSS", fontsize=19, fontweight='bold', color='#0F172A', zorder=5)

draw_arrow(31.4, 48.0, 33.2, 48.0, color="#2563EB", lw=3.8)
draw_arrow(66.8, 48.0, 68.6, 48.0, color="#16A34A", lw=3.8)

# =============================================================================
# STAGE A: INGESTION, CAR DSP DIAGRAM, PSD & 11 COLORED TOKENS
# =============================================================================
draw_box(2.3, 68.2, 28.0, 19.3, bg_color="#FFFFFF", edgecolor="black", lw=1.8, r=0.6)
ax.text(3.3, 85.2, "1. Multi-Cohort Inputs (N=146)", fontsize=16, fontweight='bold', color='#1E293B', zorder=6)

cohorts = [
    ("AUD", "64ch", "N=53", 6.9, "#0284C7", 30),
    ("AD/FTD", "19ch", "N=88", 16.3, "#DC2626", 19),
    ("Epilepsy", "23ch", "N=5", 25.7, "#9333EA", 23)
]
for title, ch_s, n_s, cx, col, dots in cohorts:
    draw_box(cx - 4.2, 69.3, 8.4, 14.5, bg_color="#F8FAFC", edgecolor="black", lw=1.4, r=0.5)
    draw_head_topomap(cx, 79.4, r=2.25, num_dots=dots, color=col)
    ax.text(cx, 74.2, title, ha='center', fontsize=14.5, fontweight='bold', color='#0F172A', zorder=7)
    ax.text(cx, 70.6, f"{ch_s} | {n_s}", ha='center', fontsize=12.8, fontweight='bold', color=col, zorder=7)

draw_arrow(16.3, 68.2, 16.3, 65.2, color="#0284C7", lw=2.6)

draw_box(2.3, 41.8, 28.0, 23.0, bg_color="#FFFFFF", edgecolor="black", lw=1.8, r=0.6)
ax.text(3.3, 62.5, "2. CAR & 4.0s Epoch Framing", fontsize=16, fontweight='bold', color='#1E293B', zorder=6)

trace_cols = ["#2563EB", "#DC2626", "#16A34A", "#9333EA"]
trace_names = ["Fp1", "C3", "Pz", "O1"]
for i in range(4):
    wy = 58.2 - i * 3.05
    ax.text(3.2, wy + 0.45, trace_names[i], fontsize=13.5, fontweight='bold', color=trace_cols[i], zorder=7)
    draw_waveform(7.2, wy, 13.8, 2.2, freq=3.2 + i*1.4, col=trace_cols[i], lw=2.0, seed=i+1)

draw_box(21.6, 48.8, 7.8, 12.2, bg_color="#EFF6FF", edgecolor="black", lw=1.4, r=0.5)
ax.add_patch(Circle((25.5, 57.6), 2.1, facecolor="#DBEAFE", edgecolor="black", lw=1.5, zorder=7))
ax.text(25.5, 57.6, "CAR", ha='center', va='center', fontsize=13.5, fontweight='bold', color='#1E40AF', zorder=8)
ax.text(25.5, 53.5, "0.5–45 Hz", ha='center', fontsize=12.5, fontweight='bold', color='#0F172A', zorder=8)
ax.text(25.5, 50.4, "4.0 s", ha='center', fontsize=13.0, fontweight='bold', color='#16A34A', zorder=8)

draw_box(3.3, 42.9, 26.0, 4.2, bg_color="#F1F5F9", edgecolor="black", lw=1.2, r=0.4)
ax.text(16.3, 45.0, r"$x_c^{\mathrm{CAR}}(t) = x_c(t) - \frac{1}{C}\sum x_j(t)$",
        ha='center', va='center', fontsize=14.5, fontweight='bold', color='#0F172A', zorder=7)

draw_arrow(16.3, 41.8, 16.3, 38.8, color="#0284C7", lw=2.6)

draw_box(2.3, 3.3, 28.0, 35.1, bg_color="#FFFFFF", edgecolor="black", lw=1.8, r=0.6)
ax.text(3.3, 36.1, "3. 11-Token Spectral Tokenizer", fontsize=16, fontweight='bold', color='#1E293B', zorder=6)

draw_box(3.3, 21.4, 26.0, 12.8, bg_color="#FAFAFA", edgecolor="black", lw=1.4, r=0.5)
ax.text(4.3, 32.0, "Welch PSD (0.5–45 Hz)", fontsize=13.5, fontweight='bold', color='#1E293B', zorder=7)

freqs = np.linspace(0.5, 45, 120)
psd = 1.0 / (freqs**1.15) + 0.35 * np.exp(-((freqs - 10)**2)/5.5)
psd_norm = (psd - np.min(psd)) / (np.max(psd) - np.min(psd)) * 7.2

bands = [
    (0.5, 4.0, "δ", "#EF4444"),
    (4.0, 8.0, "θ", "#F97316"),
    (8.0, 13.0, "α", "#F59E0B"),
    (13.0, 30.0, "β", "#10B981"),
    (30.0, 45.0, "γ", "#06B6D4")
]
for f_min, f_max, b_name, b_col in bands:
    mask = (freqs >= f_min) & (freqs <= f_max)
    x_sub = 4.6 + (freqs[mask] - 0.5) / 44.5 * 23.4
    y_sub = 22.8 + psd_norm[mask]
    ax.fill_between(x_sub, 22.8, y_sub, color=b_col, alpha=0.38, zorder=6)
    ax.plot(x_sub, y_sub, color=b_col, lw=2.4, zorder=7)
    f_mid = (f_min + f_max) / 2.0
    ax.text(4.6 + (f_mid - 0.5) / 44.5 * 23.4, 23.5, b_name, ha='center', fontsize=14, fontweight='bold', color='black', zorder=8)

draw_arrow(16.3, 21.4, 16.3, 18.8, color="#475569", lw=2.0)

token_items = [
    ("δ", "#EF4444"), ("θ", "#F97316"), ("α", "#F59E0B"), ("β", "#10B981"), ("γ", "#06B6D4"),
    ("TAR", "#3B82F6"), ("TBR", "#6366F1"), ("ABR", "#8B5CF6"),
    ("Hsp", "#EC4899"), ("Hdf", "#D946EF"), ("Hjr", "#475569")
]
for idx, (t_sym, t_c) in enumerate(token_items):
    tx = 3.3 + idx * 2.38
    draw_box(tx, 4.6, 2.05, 13.4, bg_color=t_c, edgecolor="black", lw=1.2, r=0.3)
    ax.text(tx + 1.02, 11.3, t_sym, ha='center', va='center', rotation=90,
            fontsize=13.5, fontweight='bold', color="#FFFFFF", zorder=7)

# =============================================================================
# STAGE B: HYPER-VISUAL TRANSFORMER & DUAL SSL
# =============================================================================
draw_box(34.3, 69.2, 31.4, 18.3, bg_color="#FFFFFF", edgecolor="black", lw=1.8, r=0.6)
ax.text(35.3, 85.2, "1. Token Projection & Positional Wave", fontsize=16, fontweight='bold', color='#1E293B', zorder=6)

draw_box(35.5, 72.8, 8.2, 10.2, bg_color="#EFF6FF", edgecolor="black", lw=1.4, r=0.4)
for r_i in range(4):
    for c_i in range(3):
        ax.add_patch(Rectangle((36.3 + c_i*2.2, 79.8 - r_i*1.8), 1.9, 1.5, facecolor="#3B82F6", alpha=0.4+0.15*c_i, edgecolor="black", lw=0.7, zorder=7))
ax.text(39.6, 70.8, "Linear 11→128", ha='center', fontsize=13, fontweight='bold', color='#1E40AF', zorder=8)

ax.add_patch(Circle((46.4, 78.0), 1.8, facecolor="#FEF3C7", edgecolor="black", lw=1.6, zorder=8))
ax.text(46.4, 78.0, "+", ha='center', va='center', fontsize=20, fontweight='bold', color='#B45309', zorder=9)
draw_arrow(43.7, 78.0, 44.6, 78.0, color="#1E40AF", lw=2.2)

draw_box(49.2, 72.8, 8.2, 10.2, bg_color="#FAF5FF", edgecolor="black", lw=1.4, r=0.4)
t_pe = np.linspace(0, 6.6, 100)
for ph, pcol in [(0, "#9333EA"), (1.5, "#EC4899")]:
    ax.plot(50.0 + t_pe, 77.9 + 2.2*np.sin(2*np.pi*t_pe/3.3 + ph), color=pcol, lw=2.2, zorder=7)
ax.text(53.3, 70.8, "Sinusoidal PE", ha='center', fontsize=13, fontweight='bold', color='#7E22CE', zorder=8)
draw_arrow(49.2, 78.0, 48.2, 78.0, color="#7E22CE", lw=2.2)

draw_arrow(57.4, 78.0, 59.5, 78.0, color="#D97706", lw=2.4)
draw_box(59.8, 72.8, 4.8, 10.2, bg_color="#FEF3C7", edgecolor="black", lw=1.4, r=0.4)
for k_c in range(5):
    ax.add_patch(Rectangle((60.4 + k_c*0.76, 74.0), 0.55, 7.6, facecolor="#F59E0B", edgecolor="black", lw=0.6, zorder=7))
ax.text(62.2, 70.8, r"$H_0 \in \mathbb{R}^{11\times128}$", ha='center', fontsize=13, fontweight='bold', color='#92400E', zorder=8)

draw_arrow(50.0, 69.2, 50.0, 66.2, color="#D97706", lw=2.6)

draw_box(34.3, 31.8, 31.4, 34.0, bg_color="#FFFFFF", edgecolor="black", lw=1.8, r=0.6)
ax.text(35.3, 63.4, "2. Transformer Encoder (0.57M Params)", fontsize=16, fontweight='bold', color='#1E293B', zorder=6)

draw_box(35.3, 35.6, 14.3, 25.8, bg_color="#FFFBEB", edgecolor="black", lw=1.4, r=0.5)
ax.text(42.45, 59.2, "MHSA (8 Heads)", ha='center', fontsize=14.5, fontweight='bold', color='#B45309', zorder=7)

for qi, (qlbl, qcol) in enumerate([("Q", "#EF4444"), ("K", "#F59E0B"), ("V", "#2563EB")]):
    qx = 37.3 + qi * 3.8
    draw_box(qx, 54.0, 2.8, 3.4, bg_color=qcol, edgecolor="black", lw=1.0, r=0.3)
    ax.text(qx + 1.4, 55.7, qlbl, ha='center', va='center', fontsize=13.5, fontweight='bold', color='white', zorder=8)
    draw_arrow(qx + 1.4, 54.0, qx + 1.4, 51.8, color="black", lw=1.6)

np.random.seed(10)
att_map = np.random.uniform(0.15, 1.0, (4, 4))
for r in range(4):
    for c in range(4):
        cell_col = plt.cm.Blues(att_map[r, c] * 0.8 + 0.2)
        ax.add_patch(Rectangle((37.8 + c * 2.35, 48.9 - r * 2.25), 2.1, 2.0,
                               facecolor=cell_col, edgecolor="black", lw=0.7, zorder=8))
ax.text(42.45, 37.4, r"$\mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V$", ha='center', fontsize=13.5, fontweight='bold', color='#1E3A8A', zorder=9)

draw_box(50.6, 35.6, 14.1, 25.8, bg_color="#FEF3C7", edgecolor="black", lw=1.4, r=0.5)
ax.text(57.65, 59.2, "FFN (128→512→128)", ha='center', fontsize=14.2, fontweight='bold', color='#B45309', zorder=7)

layer_x = [53.0, 57.65, 62.3]
layer_ys = [np.linspace(43.5, 55.0, 4), np.linspace(41.5, 56.5, 6), np.linspace(43.5, 55.0, 4)]
for y1 in layer_ys[0]:
    for y2 in layer_ys[1]:
        ax.plot([layer_x[0], layer_x[1]], [y1, y2], color="#64748B", lw=0.8, alpha=0.55, zorder=7)
for y2 in layer_ys[1]:
    for y3 in layer_ys[2]:
        ax.plot([layer_x[1], layer_x[2]], [y2, y3], color="#0284C7", lw=0.8, alpha=0.55, zorder=7)
for li, (lx, ys_l) in enumerate(zip(layer_x, layer_ys)):
    ncol = "#0284C7" if li != 1 else "#D97706"
    ax.scatter([lx]*len(ys_l), ys_l, s=110, color=ncol, edgecolor="black", lw=1.1, zorder=9)
ax.text(57.65, 37.4, "GELU + Add & Norm", ha='center', fontsize=13.2, fontweight='bold', color='#0F172A', zorder=8)

draw_box(35.3, 32.6, 29.4, 2.2, bg_color="#F59E0B", edgecolor="black", lw=1.2, r=0.3)
ax.text(50.0, 33.7, r"$\times 4$ Encoder Blocks  →  Global Mean Pool  $h_L \in \mathbb{R}^{128}$",
        ha='center', va='center', fontsize=13.5, fontweight='bold', color='#FFFFFF', zorder=8)

draw_arrow(42.5, 31.8, 41.8, 28.4, color="#DC2626", lw=2.6)
draw_arrow(57.5, 31.8, 58.2, 28.4, color="#2563EB", lw=2.6)

draw_box(34.3, 3.3, 15.1, 24.6, bg_color="#FFF5F5", edgecolor="black", lw=1.8, r=0.6)
ax.text(41.85, 25.6, "Head 1: Masked Recon", ha='center', fontsize=14.8, fontweight='bold', color='#991B1B', zorder=6)

mask_tokens = [("δ", "#EF4444"), ("?", "#DC2626"), ("α", "#F59E0B"), ("?", "#DC2626"), ("γ", "#06B6D4")]
for m_i, (m_s, m_col) in enumerate(mask_tokens):
    mx = 35.3 + m_i * 2.65
    is_m = (m_s == "?")
    draw_box(mx, 19.6, 2.2, 4.2, bg_color="#FEE2E2" if is_m else m_col,
             edgecolor="red" if is_m else "black", lw=1.6 if is_m else 1.0, r=0.3)
    ax.text(mx + 1.1, 21.7, m_s, ha='center', va='center', fontsize=14,
            fontweight='bold', color="red" if is_m else "white", zorder=8)

draw_box(35.3, 10.6, 13.1, 7.6, bg_color="#FFFFFF", edgecolor="black", lw=1.2, r=0.4)
draw_waveform(36.0, 12.2, 11.7, 4.4, freq=3.5, col="#DC2626", lw=2.2, seed=5)
ax.text(41.85, 11.4, "Reconstructed Spectrum", ha='center', fontsize=11.5, fontweight='bold', color='#991B1B', zorder=8)

draw_box(35.3, 4.5, 13.1, 4.8, bg_color="#FFFFFF", edgecolor="black", lw=1.2, r=0.4)
ax.text(41.85, 6.9, r"$\mathcal{L}_{\mathrm{recon}}\ (\mathrm{MSE},\ 20\%)$",
        ha='center', va='center', fontsize=13.5, fontweight='bold', color='#B91C1C', zorder=8)

draw_box(50.6, 3.3, 15.1, 24.6, bg_color="#EFF6FF", edgecolor="black", lw=1.8, r=0.6)
ax.text(58.15, 25.6, "Head 2: InfoNCE Sphere", ha='center', fontsize=14.8, fontweight='bold', color='#1E40AF', zorder=6)

s_cx, s_cy, s_r = 58.15, 16.8, 5.2
ax.add_patch(Circle((s_cx, s_cy), s_r, facecolor='#DBEAFE', edgecolor='black', lw=1.6, zorder=7))
ax.add_patch(Ellipse((s_cx, s_cy), s_r*2, s_r*0.72, facecolor='none', edgecolor='#60A5FA', ls='--', lw=1.2, zorder=8))
ax.add_patch(Ellipse((s_cx, s_cy), s_r*0.72, s_r*2, facecolor='none', edgecolor='#60A5FA', ls='--', lw=1.2, zorder=8))

ax.scatter([s_cx - 2.2], [s_cy + 1.8], s=140, color="#16A34A", edgecolor="black", lw=1.2, zorder=10)
ax.text(s_cx - 3.6, s_cy + 2.5, r"$z_i$", fontsize=13.5, fontweight='bold', color='#15803D', zorder=11)
ax.scatter([s_cx - 0.3], [s_cy + 2.8], s=140, color="#2563EB", edgecolor="black", lw=1.2, zorder=10)
ax.text(s_cx + 0.5, s_cy + 3.0, r"$z_p$", fontsize=13.5, fontweight='bold', color='#1D4ED8', zorder=11)
draw_arrow(s_cx - 1.9, s_cy + 1.9, s_cx - 0.6, s_cy + 2.6, color="#16A34A", lw=2.2)

ax.scatter([s_cx + 2.6], [s_cy - 2.2], s=140, color="#DC2626", edgecolor="black", lw=1.2, zorder=10)
ax.text(s_cx + 3.1, s_cy - 3.1, r"$z_a$", fontsize=13.5, fontweight='bold', color='#DC2626', zorder=11)
draw_arrow(s_cx - 1.8, s_cy + 1.4, s_cx + 2.1, s_cy - 1.7, color="#DC2626", lw=1.8, ls="--")

draw_box(51.6, 4.5, 13.1, 4.8, bg_color="#FFFFFF", edgecolor="black", lw=1.2, r=0.4)
ax.text(58.15, 6.9, r"$\mathcal{L}_{\mathrm{contrast}}\ (\mathbb{S}^{63},\ \tau=0.07)$",
        ha='center', va='center', fontsize=13.2, fontweight='bold', color='#1E40AF', zorder=8)

# =============================================================================
# STAGE C: VISUAL BRAIN TRANSFER, K=2 CURVE, TELEMETRY MONITOR & CPU CHIP
# =============================================================================
draw_box(69.7, 67.2, 28.0, 20.3, bg_color="#FFFFFF", edgecolor="black", lw=1.8, r=0.6)
ax.text(70.7, 85.2, "1. Zero-Shot Cross-Disorder Transfer", fontsize=15.5, fontweight='bold', color='#1E293B', zorder=6)

draw_box(70.8, 72.2, 10.6, 11.4, bg_color="#EFF6FF", edgecolor="black", lw=1.4, r=0.5)
draw_brain_silhouette(76.1, 79.4, s=1.05, fill="#DBEAFE", hot_col="#2563EB", hot_side="left")
ax.text(76.1, 74.0, "AUD (N=53)", ha='center', fontsize=13.5, fontweight='bold', color='#1E40AF', zorder=8)

draw_box(86.0, 72.2, 10.6, 11.4, bg_color="#FEF2F2", edgecolor="black", lw=1.4, r=0.5)
draw_brain_silhouette(91.3, 79.4, s=1.05, fill="#FEE2E2", hot_col="#DC2626", hot_side="right")
ax.text(91.3, 74.0, "AD (N=88)", ha='center', fontsize=13.5, fontweight='bold', color='#991B1B', zorder=8)

draw_arrow(81.5, 80.2, 85.8, 80.2, color="#16A34A", lw=2.6, rad=0.22)
ax.text(83.7, 82.6, "0.868", ha='center', fontsize=13.8, fontweight='bold', color='#15803D', zorder=9)
draw_arrow(85.8, 75.5, 81.5, 75.5, color="#2563EB", lw=2.6, rad=0.22)
ax.text(83.7, 72.8, "0.674", ha='center', fontsize=13.8, fontweight='bold', color='#1D4ED8', zorder=9)

draw_box(70.8, 68.1, 25.8, 3.2, bg_color="#F0FDF4", edgecolor="black", lw=1.0, r=0.3)
ax.text(83.7, 69.7, "Shared Neurodysfunction Axis", ha='center', va='center', fontsize=13.2, fontweight='bold', color='#15803D', zorder=7)

draw_arrow(83.7, 67.2, 83.7, 64.2, color="#16A34A", lw=2.6)

draw_box(69.7, 41.8, 28.0, 22.0, bg_color="#FFFFFF", edgecolor="black", lw=1.8, r=0.6)
ax.text(70.7, 61.5, "2. Extreme Few-Shot (K = 2 Patients)", fontsize=15.5, fontweight='bold', color='#1E293B', zorder=6)

draw_box(70.8, 43.2, 8.2, 16.4, bg_color="#F8FAFC", edgecolor="black", lw=1.3, r=0.4)
draw_patient_icon(73.3, 55.0, s=1.05, col="#2563EB")
draw_patient_icon(76.5, 55.0, s=1.05, col="#DC2626")
ax.text(74.9, 51.0, "K = 2", ha='center', fontsize=14.5, fontweight='bold', color='#0F172A', zorder=9)
t_fs = np.linspace(0, 6.2, 30)
y_fs = 44.8 + 4.4 * (1 - np.exp(-0.7 * t_fs))
ax.plot(71.8 + t_fs, y_fs, color="#16A34A", lw=2.6, zorder=9)
ax.scatter([72.4], [46.5], s=80, color="#F59E0B", edgecolor="black", lw=1.0, marker="*", zorder=10)

draw_box(79.8, 51.8, 16.8, 7.8, bg_color="#EFF6FF", edgecolor="black", lw=1.2, r=0.4)
ax.text(80.8, 57.0, "Alzheimer's (K=2):", fontsize=13.5, fontweight='bold', color='#1E40AF', zorder=8)
draw_box(80.8, 53.0, 10.5, 2.4, bg_color="#DBEAFE", edgecolor="black", lw=0.8, r=0.2)
draw_box(80.8, 53.0, 10.5*0.867, 2.4, bg_color="#2563EB", edgecolor="none", lw=0, r=0.2)
ax.text(92.0, 54.2, "0.867", va='center', fontsize=14, fontweight='bold', color='#1E40AF', zorder=8)

draw_box(79.8, 43.2, 16.8, 7.8, bg_color="#FEF2F2", edgecolor="black", lw=1.2, r=0.4)
ax.text(80.8, 48.4, "Alcoholism (K=2):", fontsize=13.5, fontweight='bold', color='#991B1B', zorder=8)
draw_box(80.8, 44.4, 10.5, 2.4, bg_color="#FEE2E2", edgecolor="black", lw=0.8, r=0.2)
draw_box(80.8, 44.4, 10.5*0.742, 2.4, bg_color="#DC2626", edgecolor="none", lw=0, r=0.2)
ax.text(92.0, 45.6, "0.742", va='center', fontsize=14, fontweight='bold', color='#991B1B', zorder=8)

draw_arrow(83.7, 41.8, 83.7, 38.8, color="#16A34A", lw=2.6)

draw_box(69.7, 3.3, 28.0, 35.1, bg_color="#FFFFFF", edgecolor="black", lw=1.8, r=0.6)
ax.text(70.7, 36.1, "3. Bedside CDSS & Edge Telemetry", fontsize=15.5, fontweight='bold', color='#1E293B', zorder=6)

draw_box(70.8, 14.0, 25.8, 20.4, bg_color="#0F172A", edgecolor="black", lw=1.6, r=0.5)
ax.text(71.8, 32.2, "LIVE CDSS MONITOR", fontsize=13.2, fontweight='bold', color='#38BDF8', zorder=7)
draw_waveform(71.8, 27.2, 23.8, 3.6, freq=6.0, col="#4ADE80", lw=1.8, seed=9)

draw_box(71.8, 15.2, 11.4, 11.0, bg_color="#1E293B", edgecolor="#475569", lw=1.0, r=0.4)
ax.text(77.5, 24.4, "AUD Risk", ha='center', fontsize=13.0, fontweight='bold', color='#94A3B8', zorder=8)
ax.add_patch(Wedge((77.5, 19.0), 3.6, 0, 180, width=1.1, facecolor="#334155", edgecolor="none", zorder=8))
ax.add_patch(Wedge((77.5, 19.0), 3.6, 180*(1-0.771), 180, width=1.1, facecolor="#EF4444", edgecolor="none", zorder=9))
ax.text(77.5, 19.2, "77.1%", ha='center', va='center', fontsize=14.5, fontweight='bold', color='#F87171', zorder=10)
ax.text(77.5, 16.3, "HIGH RISK", ha='center', fontsize=11.8, fontweight='bold', color='#FCA5A5', zorder=10)

draw_box(84.2, 15.2, 11.4, 11.0, bg_color="#1E293B", edgecolor="#475569", lw=1.0, r=0.4)
ax.text(89.9, 24.4, "Dementia", ha='center', fontsize=13.0, fontweight='bold', color='#94A3B8', zorder=8)
for lbl_m, val_m, col_m, y_m in [("AD", 0.853, "#F87171", 21.3), ("FTD", 0.078, "#C084FC", 18.7), ("CN", 0.069, "#4ADE80", 16.1)]:
    ax.text(84.8, y_m + 0.5, lbl_m, fontsize=11.5, fontweight='bold', color=col_m, zorder=9)
    draw_box(87.9, y_m + 0.2, 4.2, 1.3, bg_color="#0F172A", edgecolor="#334155", lw=0.6, r=0.15)
    draw_box(87.9, y_m + 0.2, max(0.45, 4.2*val_m), 1.3, bg_color=col_m, edgecolor="none", lw=0, r=0.15)
    ax.text(95.2, y_m + 0.5, f"{val_m*100:.0f}%", ha='right', fontsize=11.5, fontweight='bold', color="#E2E8F0", zorder=9)

draw_box(70.8, 4.5, 25.8, 8.4, bg_color="#F8FAFC", edgecolor="black", lw=1.4, r=0.4)
cx_c, cy_c = 74.2, 8.7
for off in [-1.1, 0.0, 1.1]:
    ax.plot([cx_c + off, cx_c + off], [cy_c - 2.3, cy_c + 2.3], color="black", lw=2.0, zorder=7)
    ax.plot([cx_c - 2.3, cx_c + 2.3], [cy_c + off, cy_c + off], color="black", lw=2.0, zorder=7)
draw_box(cx_c - 1.6, cy_c - 1.6, 3.2, 3.2, bg_color="#0F172A", edgecolor="black", lw=1.4, r=0.2, zorder=8)
ax.text(cx_c, cy_c, "CPU", ha='center', va='center', fontsize=11.2, fontweight='bold', color='#4ADE80', zorder=9)

ax.text(78.2, 9.8, "CPU: 3.29 ms (1,058× Real-Time)", fontsize=13.5, fontweight='bold', color='#16A34A', zorder=8)
ax.text(78.2, 6.4, "GPU: 58.0 μs / epoch (Batched)", fontsize=13.5, fontweight='bold', color='#0284C7', zorder=8)

for d in [FIG_DIR, LATEX_FIG_DIR, DESKTOP_FIG_DIR, BRAIN_DIR]:
    png_p = os.path.join(d, "Figure_1_Master_Architecture.png")
    pdf_p = os.path.join(d, "Figure_1_Master_Architecture.pdf")
    fig.savefig(png_p, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_p, format='pdf', bbox_inches='tight')

plt.close(fig)
print("=== HYPER-VISUAL FIGURE 1 COMPLETED (LARGE TEXT & ZERO OVERLAP) ===")
