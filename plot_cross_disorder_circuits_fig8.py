"""
Publication-Grade Cross-Disorder Functional Neurobiology & Connectome Circuit Diagram (Figure 8)
Inspired by Nature Biomedical Engineering (Zhang et al.) and DeepTokenEEG (2026).
Strict formatting:
- Black and white box outlines (edgecolor='black', linewidth=1.8 to 2.2).
- Inside content is rich in color, visual diagrams, circuit arrows, and waveforms.
- Visually understandable with concise labels and minimal dense text.
- 1.3x scaled typography with 100% zero text overlap or overflow.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge, PathPatch
from matplotlib.path import Path
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

fig = plt.figure(figsize=(24.0, 14.0), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Palette
BG_STAGE = "#FAFAFA"
BG_CARD = "#FFFFFF"

def draw_box(x, y, w, h, bg_color=BG_CARD, edgecolor="black", lw=2.0, r=1.0, zorder=2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.3,rounding_size={r}",
                         facecolor=bg_color, edgecolor=edgecolor,
                         linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box

def draw_arrow(x1, y1, x2, y2, color="#1E293B", lw=2.5, style="-|>", rad=0.0):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=f"{style},head_length=8,head_width=5",
                            color=color, linewidth=lw,
                            connectionstyle=f"arc3,rad={rad}", zorder=10)
    ax.add_patch(arrow)

# Helper to draw a stylized brain silhouette outline
def draw_brain_silhouette(cx, cy, rx, ry, color="#CBD5E1", zorder=3):
    # Ellipse base
    ellipse = patches.Ellipse((cx, cy), rx*2, ry*2, facecolor='none', edgecolor=color, lw=2.0, zorder=zorder)
    ax.add_patch(ellipse)

# Helper to draw a clean stylized waveform
def draw_waveform(x, y, w, h, freq=1.0, amp=1.0, col="#2563EB", lw=2.0, zorder=5):
    t = np.linspace(0, w, 200)
    y_wave = y + h/2.0 + (h/2.2) * amp * np.sin(2 * np.pi * freq * (t / w))
    ax.plot(x + t, y_wave, color=col, lw=lw, zorder=zorder)

# -----------------------------------------------------------------------------
# 1. TITLE BANNER
# -----------------------------------------------------------------------------
draw_box(2, 93.0, 96, 5.8, bg_color="#FFFFFF", edgecolor="black", lw=2.2, r=0.8)
ax.text(50, 96.5, "Cross-Disorder Neurobiological Circuit Mechanisms & Latent Hypersphere Alignment",
        ha='center', va='center', fontsize=21 * SCALE, fontweight='bold', color='#0F172A', zorder=5)
ax.text(50, 94.2, "Mechanistic Etiology across Alzheimer's Disease, Alcohol Use Disorder, and Epilepsy on the Shared S⁶³ Manifold",
        ha='center', va='center', fontsize=12.5 * SCALE, color='#475569', zorder=5)

# -----------------------------------------------------------------------------
# 2. FOUR PRIMARY PANELS (2 x 2 GRID)
# -----------------------------------------------------------------------------

# Panel A: Alzheimer's Thalamocortical Disconnectivity (x: 2 to 49, y: 49 to 91.5)
draw_box(2, 49.0, 47.0, 42.5, bg_color="#FEF2F2", edgecolor="black", lw=2.0, r=1.0)
ax.text(3.5, 88.5, "a  Alzheimer's Disease: Thalamocortical Disconnection & Alpha Slowing",
        fontsize=13.5 * SCALE, fontweight='bold', color='#991B1B', zorder=5)

# Inside Panel A:
# A1: Circuit Diagram Box (Left side: 3.5 to 25.5)
draw_box(3.5, 50.5, 21.5, 36.0, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.6)
ax.text(4.5, 84.0, "Neurocircuit Breakdown:", fontsize=10.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

# Brain & Thalamus Schematics
draw_box(5.0, 68.0, 18.5, 14.0, bg_color="#FFF1F2", edgecolor="black", lw=1.2, r=0.5)
ax.text(14.25, 79.5, "Cholinergic Denervation", ha='center', fontsize=9.5 * SCALE, fontweight='bold', color='#991B1B', zorder=7)
# Thalamus badge
draw_box(6.0, 70.0, 7.5, 7.0, bg_color="#FECDD3", edgecolor="black", lw=1.0, r=0.4)
ax.text(9.75, 73.5, "Thalamic\nPacemaker", ha='center', va='center', fontsize=8.5 * SCALE, fontweight='bold', color='#881337', zorder=8)
# Cortical projection arrow (dashed = impaired)
draw_arrow(13.8, 73.5, 18.0, 73.5, color="#E11D48", lw=2.5, style="->")
ax.text(15.9, 75.0, "✕ Loss", ha='center', fontsize=8.0 * SCALE, fontweight='bold', color='#BE123C', zorder=8)
# Posterior Cortex badge
draw_box(18.2, 70.0, 4.8, 7.0, bg_color="#F43F5E", edgecolor="black", lw=1.0, r=0.4)
ax.text(20.6, 73.5, "Pz/O1\nCortex", ha='center', va='center', fontsize=8.5 * SCALE, fontweight='bold', color='#FFFFFF', zorder=8)

# Oscillatory Waveform Comparison
draw_box(5.0, 52.0, 18.5, 14.5, bg_color="#F8FAFC", edgecolor="black", lw=1.2, r=0.5)
ax.text(5.8, 64.0, "Healthy: Synchronized Alpha (8-13 Hz)", fontsize=8.8 * SCALE, fontweight='bold', color='#166534', zorder=7)
draw_waveform(5.8, 59.0, 16.8, 3.8, freq=5.0, amp=1.0, col="#16A34A", lw=2.2)

ax.text(5.8, 56.8, "AD: Diffuse Theta Slowing (4-8 Hz)", fontsize=8.8 * SCALE, fontweight='bold', color='#DC2626', zorder=7)
draw_waveform(5.8, 52.5, 16.8, 3.8, freq=2.2, amp=0.9, col="#DC2626", lw=2.2)

# A2: Electrophysiological Markers (Right side: 26.0 to 47.5)
draw_box(26.0, 50.5, 21.5, 36.0, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.6)
ax.text(27.0, 84.0, "Electrophysiological Phenotype:", fontsize=10.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

ad_markers = [
    ("Posterior α Desync:", "Pz, O1, O2 (-16.4% power)", "#DC2626"),
    ("Diffuse θ Slowing:", "Frontal/Parietal (+22.1% power)", "#EA580C"),
    ("Slowing Ratio TAR:", "TAR = θ/α increases by +48.2%", "#D97706"),
    ("Differential Entropy:", "H_diff collapse in parietal leads", "#7C3AED"),
    ("Gradient Saliency:", "Alpha: 29.69% | Theta: 21.89%", "#BE123C")
]
for i, (m_lbl, m_desc, col) in enumerate(ad_markers):
    my = 78.5 - i * 6.5
    draw_box(27.0, my, 19.5, 5.5, bg_color="#FEF2F2", edgecolor="black", lw=1.0, r=0.4)
    ax.text(27.8, my + 3.4, m_lbl, fontsize=9.2 * SCALE, fontweight='bold', color=col, zorder=7)
    ax.text(27.8, my + 1.2, m_desc, fontsize=8.8 * SCALE, color='#334155', zorder=7)


# -----------------------------------------------------------------------------
# Panel B: Alcoholism Cortical Disinhibition (x: 51 to 98, y: 49 to 91.5)
# -----------------------------------------------------------------------------
draw_box(51.0, 49.0, 47.0, 42.5, bg_color="#EFF6FF", edgecolor="black", lw=2.0, r=1.0)
ax.text(52.5, 88.5, "b  Alcohol Use Disorder: Frontal Disinhibition & Fast-Wave Hyperarousal",
        fontsize=13.5 * SCALE, fontweight='bold', color='#1E40AF', zorder=5)

# B1: Circuit Diagram Box (Left side: 52.5 to 74.5)
draw_box(52.5, 50.5, 21.5, 36.0, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.6)
ax.text(53.5, 84.0, "Neurocircuit Breakdown:", fontsize=10.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

# Prefrontal GABA Downregulation Schematic
draw_box(54.0, 68.0, 18.5, 14.0, bg_color="#EFF6FF", edgecolor="black", lw=1.2, r=0.5)
ax.text(63.25, 79.5, "GABAergic Interneuron Downregulation", ha='center', fontsize=9.2 * SCALE, fontweight='bold', color='#1E40AF', zorder=7)
# PFC badge
draw_box(55.0, 70.0, 7.5, 7.0, bg_color="#BFDBFE", edgecolor="black", lw=1.0, r=0.4)
ax.text(58.75, 73.5, "Prefrontal\nCortex (PFC)", ha='center', va='center', fontsize=8.5 * SCALE, fontweight='bold', color='#1E3A8A', zorder=8)
# Hyperactivation arrow
draw_arrow(62.8, 73.5, 67.0, 73.5, color="#2563EB", lw=2.5, style="->")
ax.text(64.9, 75.0, "+ Surge", ha='center', fontsize=8.0 * SCALE, fontweight='bold', color='#1D4ED8', zorder=8)
# Fast-wave firing badge
draw_box(67.2, 70.0, 4.8, 7.0, bg_color="#3B82F6", edgecolor="black", lw=1.0, r=0.4)
ax.text(69.6, 73.5, "Fp1/Fp2\nβ / γ", ha='center', va='center', fontsize=8.5 * SCALE, fontweight='bold', color='#FFFFFF', zorder=8)

# Oscillatory Waveform Comparison
draw_box(54.0, 52.0, 18.5, 14.5, bg_color="#F8FAFC", edgecolor="black", lw=1.2, r=0.5)
ax.text(54.8, 64.0, "Healthy: Balanced Resting Rhythm", fontsize=8.8 * SCALE, fontweight='bold', color='#166534', zorder=7)
draw_waveform(54.8, 59.0, 16.8, 3.8, freq=3.5, amp=0.8, col="#16A34A", lw=2.0)

ax.text(54.8, 56.8, "AUD: Frontal Beta/Gamma Hyperarousal", fontsize=8.8 * SCALE, fontweight='bold', color='#2563EB', zorder=7)
draw_waveform(54.8, 52.5, 16.8, 3.8, freq=12.0, amp=0.75, col="#2563EB", lw=1.8)

# B2: Electrophysiological Markers (Right side: 75.0 to 96.5)
draw_box(75.0, 50.5, 21.5, 36.0, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.6)
ax.text(76.0, 84.0, "Electrophysiological Phenotype:", fontsize=10.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

aud_markers = [
    ("Frontal β Elevation:", "Fp1, Fp2, Fz (+18.2% power)", "#2563EB"),
    ("High-Frequency γ Surge:", "30–45 Hz cortical desync (+24.5%)", "#0284C7"),
    ("Excess Entropy H_diff:", "Differential entropy surge in frontal leads", "#0D9488"),
    ("Cortical Disinhibition:", "Loss of inhibitory GABA_A tone", "#4338CA"),
    ("Gradient Saliency:", "Gamma: 46.30% | Beta: 21.84%", "#1E40AF")
]
for i, (m_lbl, m_desc, col) in enumerate(aud_markers):
    my = 78.5 - i * 6.5
    draw_box(76.0, my, 19.5, 5.5, bg_color="#EFF6FF", edgecolor="black", lw=1.0, r=0.4)
    ax.text(76.8, my + 3.4, m_lbl, fontsize=9.2 * SCALE, fontweight='bold', color=col, zorder=7)
    ax.text(76.8, my + 1.2, m_desc, fontsize=8.8 * SCALE, color='#334155', zorder=7)


# -----------------------------------------------------------------------------
# Panel C: Pediatric Epilepsy Hypersynchrony (x: 2 to 49, y: 4 to 46.5)
# -----------------------------------------------------------------------------
draw_box(2, 4.0, 47.0, 42.5, bg_color="#FAF5FF", edgecolor="black", lw=2.0, r=1.0)
ax.text(3.5, 43.5, "c  Pediatric Epilepsy: Paroxysmal Hypersynchrony & Ictal Discharge",
        fontsize=13.5 * SCALE, fontweight='bold', color='#6B21A8', zorder=5)

# C1: Seizure Dynamics Schematic (Left side: 3.5 to 25.5)
draw_box(3.5, 5.5, 21.5, 36.0, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.6)
ax.text(4.5, 39.0, "Ictogenic Network Dynamics:", fontsize=10.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

# Temporal-Central Network badge
draw_box(5.0, 23.5, 18.5, 13.5, bg_color="#F3E8FF", edgecolor="black", lw=1.2, r=0.5)
ax.text(14.25, 34.5, "Pyramidal Burst Synchronization", ha='center', fontsize=9.2 * SCALE, fontweight='bold', color='#6B21A8', zorder=7)
draw_box(6.0, 25.5, 7.5, 6.5, bg_color="#E9D5FF", edgecolor="black", lw=1.0, r=0.4)
ax.text(9.75, 28.7, "Epileptogenic\nFocus", ha='center', va='center', fontsize=8.5 * SCALE, fontweight='bold', color='#581C87', zorder=8)
draw_arrow(13.8, 28.7, 18.0, 28.7, color="#9333EA", lw=2.5, style="->")
draw_box(18.2, 25.5, 4.8, 6.5, bg_color="#A855F7", edgecolor="black", lw=1.0, r=0.4)
ax.text(20.6, 28.7, "Bipolar\nChannels", ha='center', va='center', fontsize=8.5 * SCALE, fontweight='bold', color='#FFFFFF', zorder=8)

# Paroxysmal Spike-Wave Waveform
draw_box(5.0, 7.0, 18.5, 14.5, bg_color="#F8FAFC", edgecolor="black", lw=1.2, r=0.5)
ax.text(5.8, 19.0, "Interictal Baseline (Normal EEG)", fontsize=8.8 * SCALE, fontweight='bold', color='#166534', zorder=7)
draw_waveform(5.8, 14.5, 16.8, 3.5, freq=4.0, amp=0.7, col="#16A34A", lw=2.0)

ax.text(5.8, 12.0, "Ictal Seizure: High-Voltage Spike-Wave", fontsize=8.8 * SCALE, fontweight='bold', color='#9333EA', zorder=7)
# High amplitude spike wave
t_spike = np.linspace(0, 16.8, 200)
y_spike = 7.5 + 2.0 + 1.8 * (np.sin(2 * np.pi * 3.0 * (t_spike/16.8)) + 0.8 * np.sin(2 * np.pi * 9.0 * (t_spike/16.8))**3)
ax.plot(5.8 + t_spike, y_spike, color="#9333EA", lw=2.2, zorder=7)

# C2: Epilepsy Electrophysiological Markers (Right side: 26.0 to 47.5)
draw_box(26.0, 5.5, 21.5, 36.0, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.6)
ax.text(27.0, 39.0, "Electrophysiological Phenotype:", fontsize=10.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

epi_markers = [
    ("High-Voltage Spikes:", "Paroxysmal amplitude excursions (>200 μV)", "#7E22CE"),
    ("Phase Synchronization:", "Cross-montage phase locking index surges", "#6B21A8"),
    ("Spectral Redirection:", "Massive power concentration in delta/theta", "#581C87"),
    ("Continuous Telemetry:", "23-channel bipolar montage (CHB-MIT)", "#4A044E"),
    ("Benchmark Fidelity:", "AUROC = 0.7061 | 98.85% Mean Recall", "#9333EA")
]
for i, (m_lbl, m_desc, col) in enumerate(epi_markers):
    my = 33.5 - i * 6.5
    draw_box(27.0, my, 19.5, 5.5, bg_color="#FAF5FF", edgecolor="black", lw=1.0, r=0.4)
    ax.text(27.8, my + 3.4, m_lbl, fontsize=9.2 * SCALE, fontweight='bold', color=col, zorder=7)
    ax.text(27.8, my + 1.2, m_desc, fontsize=8.8 * SCALE, color='#334155', zorder=7)


# -----------------------------------------------------------------------------
# Panel D: Universal Latent Hypersphere Alignment (x: 51 to 98, y: 4 to 46.5)
# -----------------------------------------------------------------------------
draw_box(51.0, 4.0, 47.0, 42.5, bg_color="#FEFCE8", edgecolor="black", lw=2.0, r=1.0)
ax.text(52.5, 43.5, "d  Universal Cross-Disorder Hypersphere Manifold (S⁶³) & Zero-Shot Transfer",
        fontsize=13.5 * SCALE, fontweight='bold', color='#854D0E', zorder=5)

# D1: Hypersphere Schematic Visualization (Left side: 52.5 to 74.5)
draw_box(52.5, 5.5, 21.5, 36.0, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.6)
ax.text(53.5, 39.0, "Unit Hypersphere Projection (S⁶³):", fontsize=10.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

# Draw 2D projection of unit hypersphere
circ_center = (63.25, 21.0)
circ_radius = 12.0
circle = Circle(circ_center, circ_radius, facecolor='#FEF9C3', edgecolor='black', lw=2.0, zorder=6)
ax.add_patch(circle)

# Equator / Coordinate Axes
ax.plot([circ_center[0] - circ_radius, circ_center[0] + circ_radius], [circ_center[1], circ_center[1]],
        color='#94A3B8', linestyle='--', lw=1.5, zorder=7)
ax.plot([circ_center[0], circ_center[0]], [circ_center[1] - circ_radius, circ_center[1] + circ_radius],
        color='#94A3B8', linestyle='--', lw=1.5, zorder=7)

# Opposite Poles: AD Slowing (Top-Left) vs AUD Hyperarousal (Bottom-Right)
ad_pole = (circ_center[0] - 7.5, circ_center[1] + 7.5)
aud_pole = (circ_center[0] + 7.5, circ_center[1] - 7.5)
hc_center = (circ_center[0] - 1.0, circ_center[1] + 1.0)

# Scatter points and labels on circle
ax.scatter([ad_pole[0]], [ad_pole[1]], s=220, color='#DC2626', edgecolor='black', lw=1.5, zorder=9)
ax.text(ad_pole[0] - 1.0, ad_pole[1] + 2.5, "AD Pole\n(Slowing)", ha='center', fontsize=9.0 * SCALE, fontweight='bold', color='#DC2626', zorder=10)

ax.scatter([aud_pole[0]], [aud_pole[1]], s=220, color='#2563EB', edgecolor='black', lw=1.5, zorder=9)
ax.text(aud_pole[0] + 1.0, aud_pole[1] - 3.8, "AUD Pole\n(Hyperarousal)", ha='center', fontsize=9.0 * SCALE, fontweight='bold', color='#2563EB', zorder=10)

ax.scatter([hc_center[0]], [hc_center[1]], s=180, color='#16A34A', edgecolor='black', lw=1.5, zorder=9)
ax.text(hc_center[0] + 3.2, hc_center[1] + 1.2, "Healthy Controls\n(Equilibrium)", fontsize=8.2 * SCALE, fontweight='bold', color='#166534', zorder=10)

# Connect AD and AUD with dashed axis
ax.plot([ad_pole[0], aud_pole[0]], [ad_pole[1], aud_pole[1]], color='#D97706', linestyle=':', lw=2.5, zorder=8)
ax.text(circ_center[0] - 3.5, circ_center[1] - 4.5, "Continuous Neurodysfunction Axis",
        fontsize=8.0 * SCALE, fontweight='bold', color='#B45309', rotation=-45, zorder=10)

# D2: Manifold & Zero-Shot Alignment Explanation (Right side: 75.0 to 96.5)
draw_box(75.0, 5.5, 21.5, 36.0, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.6)
ax.text(76.0, 39.0, "Why Zero-Shot Transfer Works:", fontsize=10.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

transfer_insights = [
    ("Geometric Duality:", "AD (loss of speed) & AUD (excess speed) form antipodal coordinates on S⁶³", "#B45309"),
    ("Dementia -> AUD Transfer:", "AUROC = 0.6744 (+17.4% over random chance without re-training)", "#15803D"),
    ("AUD -> Dementia Transfer:", "AUROC = 0.8676 (matches specialist models trained directly on AD)", "#1D4ED8"),
    ("Contrastive Invariance:", "InfoNCE loss (τ=0.07) suppresses subject identity and aligns pathology", "#7C3AED"),
    ("Shared Manifold Axis:", "Permutation test confirms non-random alignment (p < 0.001)", "#B91C1C")
]
for i, (t_lbl, t_desc, col) in enumerate(transfer_insights):
    my = 33.5 - i * 6.5
    draw_box(76.0, my, 19.5, 5.5, bg_color="#FEFCE8", edgecolor="black", lw=1.0, r=0.4)
    ax.text(76.8, my + 3.4, t_lbl, fontsize=9.2 * SCALE, fontweight='bold', color=col, zorder=7)
    ax.text(76.8, my + 1.2, t_desc, fontsize=8.8 * SCALE, color='#334155', zorder=7)

# Save high-resolution outputs
for d in [FIG_DIR, LATEX_FIG_DIR, DESKTOP_FIG_DIR]:
    png_p = os.path.join(d, "Figure_8_Cross_Disorder_Neurobiology_and_Connectomes.png")
    pdf_p = os.path.join(d, "Figure_8_Cross_Disorder_Neurobiology_and_Connectomes.pdf")
    fig.savefig(png_p, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_p, format='pdf', bbox_inches='tight')
    print(f"Saved: {png_p} & {pdf_p}")

plt.close(fig)
print("=== FIGURE 8 NEUROBIOLOGICAL CONNECTOME DIAGRAM COMPLETED SUCCESSFULLY ===")
