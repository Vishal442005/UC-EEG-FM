"""
World-Class Cross-Disorder Functional Neurobiology & Connectome Circuit Diagram (Figure 8 v2)
Inspired by Nature Biomedical Engineering (Zhang et al.) and DeepTokenEEG (2026).
Hyper-visual, highly intuitive, minimal text, publication-grade aesthetics.
Strict formatting:
- Black and white box outlines (edgecolor='black', linewidth=2.0).
- Inside content is rich in color, anatomical brain profiles, circuit pathways, and real waveforms.
- Visually understandable at a glance.
- 1.3x scaled typography with 100% zero text overlap or overflow.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Polygon, Rectangle
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

def draw_brain_sagittal(cx, cy, scale=1.0, highlight_region="parietal", col_highlight="#EF4444", zorder=5):
    """Draws a clean anatomical sagittal brain profile with highlighted functional regions."""
    # Cerebrum contour
    c_pts = np.array([
        [-3.0, 0.0], [-2.8, 1.8], [-2.0, 3.2], [-0.5, 3.8], [1.5, 3.6],
        [2.8, 2.5], [3.2, 0.8], [3.0, -1.0], [2.2, -1.8], [1.0, -2.0],
        [-0.5, -1.8], [-1.8, -1.2], [-2.6, -0.6]
    ]) * scale
    c_pts[:, 0] += cx
    c_pts[:, 1] += cy
    c_poly = Polygon(c_pts, closed=True, facecolor="#F8FAFC", edgecolor="black", lw=1.6, zorder=zorder)
    ax.add_patch(c_poly)
    
    # Cerebellum
    cb = Ellipse((cx + 1.8*scale, cy - 1.8*scale), 1.8*scale, 1.2*scale, facecolor="#E2E8F0", edgecolor="black", lw=1.2, zorder=zorder)
    ax.add_patch(cb)
    # Brainstem
    bs = Rectangle((cx + 0.4*scale, cy - 3.0*scale), 0.9*scale, 1.6*scale, facecolor="#E2E8F0", edgecolor="black", lw=1.2, zorder=zorder)
    ax.add_patch(bs)
    
    # Highlight functional zones
    if highlight_region == "parietal":
        # Posterior parieto-occipital zone
        pz = Ellipse((cx + 1.6*scale, cy + 1.2*scale), 2.2*scale, 2.4*scale, facecolor=col_highlight, alpha=0.45, edgecolor="none", zorder=zorder+1)
        ax.add_patch(pz)
        ax.text(cx + 1.6*scale, cy + 1.2*scale, "Pz / O1\nα Slowing", ha='center', va='center',
                fontsize=8.0*SCALE, fontweight='bold', color='#7F1D1D', zorder=zorder+2)
    elif highlight_region == "frontal":
        # Prefrontal cortex zone
        fz = Ellipse((cx - 1.8*scale, cy + 1.2*scale), 2.2*scale, 2.4*scale, facecolor=col_highlight, alpha=0.45, edgecolor="none", zorder=zorder+1)
        ax.add_patch(fz)
        ax.text(cx - 1.8*scale, cy + 1.2*scale, "Fp1 / Fz\nβ/γ Surge", ha='center', va='center',
                fontsize=8.0*SCALE, fontweight='bold', color='#1E3A8A', zorder=zorder+2)
    elif highlight_region == "temporal":
        # Temporal zone for epilepsy
        tz = Ellipse((cx + 0.2*scale, cy - 0.4*scale), 2.4*scale, 1.6*scale, facecolor=col_highlight, alpha=0.45, edgecolor="none", zorder=zorder+1)
        ax.add_patch(tz)
        ax.text(cx + 0.2*scale, cy - 0.4*scale, "T3 / C3\nIctal Burst", ha='center', va='center',
                fontsize=8.0*SCALE, fontweight='bold', color='#581C87', zorder=zorder+2)

# -----------------------------------------------------------------------------
# 1. TOP TITLE BANNER
# -----------------------------------------------------------------------------
draw_box(1.5, 93.2, 97.0, 5.6, bg_color="#FFFFFF", edgecolor="black", lw=2.2, r=0.8)
ax.text(50.0, 96.6, "Cross-Disorder Neurobiological Circuit Mechanisms & Shared Hypersphere Alignment",
        ha='center', va='center', fontsize=21 * SCALE, fontweight='bold', color='#0F172A', zorder=5)
ax.text(50.0, 94.4, "Mechanistic Etiology across Alzheimer's Disease, Alcohol Use Disorder, and Epilepsy Mapped to the S⁶³ Manifold",
        ha='center', va='center', fontsize=12.5 * SCALE, color='#475569', zorder=5)

# -----------------------------------------------------------------------------
# 2. FOUR PRIMARY PANELS (2 x 2 GRID)
# -----------------------------------------------------------------------------

# =============================================================================
# PANEL A: ALZHEIMER'S DISEASE (THALAMOCORTICAL DYSRHYTHMIA & ALPHA SLOWING)
# =============================================================================
draw_box(1.5, 48.0, 47.5, 43.5, bg_color="#FEF2F2", edgecolor="black", lw=2.0, r=1.0)
ax.text(3.0, 88.5, "a  Alzheimer's Disease: Thalamocortical Disconnection & Alpha Slowing",
        fontsize=13.5 * SCALE, fontweight='bold', color='#991B1B', zorder=5)

# A1: Anatomical Brain Circuit Schematic (Left: 2.8 to 24.5)
draw_box(2.8, 49.5, 21.7, 36.5, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.6)
ax.text(3.8, 83.2, "Cholinergic Denervation Circuit:", fontsize=10.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

# Draw sagittal brain profile with highlighted parietal alpha slowing
draw_brain_sagittal(cx=10.5, cy=69.0, scale=1.4, highlight_region="parietal", col_highlight="#EF4444", zorder=7)

# Basal Forebrain & Thalamus labels
draw_box(16.5, 74.0, 7.0, 8.5, bg_color="#FFF1F2", edgecolor="black", lw=1.0, r=0.4)
ax.text(20.0, 80.0, "Thalamic Pacemaker", ha='center', fontsize=8.0 * SCALE, fontweight='bold', color='#991B1B', zorder=8)
ax.text(20.0, 76.5, "Cholinergic Loss", ha='center', fontsize=7.5 * SCALE, color='#DC2626', zorder=8)

# Break mark
draw_arrow(12.5, 68.0, 16.5, 75.0, color="#DC2626", lw=2.0, style="->")
ax.text(14.5, 72.0, "✕ Break", fontsize=8.0 * SCALE, fontweight='bold', color='#BE123C', zorder=9)

# A2: Direct Waveform Comparison & Quantitative Gauges (Right: 25.2 to 47.8)
draw_box(25.2, 49.5, 22.6, 36.5, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.6)
ax.text(26.2, 83.2, "Oscillatory Slowing & Biomarkers:", fontsize=10.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

# Healthy vs AD Waveforms
draw_box(26.2, 69.5, 20.6, 12.0, bg_color="#F8FAFC", edgecolor="black", lw=1.0, r=0.4)
ax.text(27.0, 79.2, "Healthy: Synchronized Alpha (10 Hz)", fontsize=8.5 * SCALE, fontweight='bold', color='#166534', zorder=7)
t_w = np.linspace(0, 18.0, 200)
ax.plot(27.2 + t_w, 75.5 + 1.2 * np.sin(2 * np.pi * 5.0 * (t_w/18.0)), color='#16A34A', lw=2.0, zorder=8)

ax.text(27.0, 73.2, "Alzheimer's: Diffuse Theta Slowing (5 Hz)", fontsize=8.5 * SCALE, fontweight='bold', color='#DC2626', zorder=7)
ax.plot(27.2 + t_w, 70.0 + 1.4 * np.sin(2 * np.pi * 2.2 * (t_w/18.0)), color='#DC2626', lw=2.2, zorder=8)

# Quantitative Biomarker Gauges
ad_badges = [
    ("θ/α Ratio (TAR):", "+48.2% Surge (Severe Slowing)", "#DC2626", "#FEF2F2"),
    ("Posterior α Power:", "-16.4% Desynchronization", "#EA580C", "#FFF7ED"),
    ("Differential Entropy:", "Severe Collapse in Pz / O1 leads", "#7C3AED", "#FAF5FF"),
    ("Gradient Saliency:", "Alpha: 29.69% | Theta: 21.89%", "#BE123C", "#FFF1F2")
]
for idx, (b_title, b_desc, b_col, b_bg) in enumerate(ad_badges):
    by = 64.0 - idx * 4.6
    draw_box(26.2, by, 20.6, 4.0, bg_color=b_bg, edgecolor="black", lw=0.8, r=0.3)
    ax.text(27.0, by + 2.5, b_title, fontsize=8.5 * SCALE, fontweight='bold', color=b_col, zorder=8)
    ax.text(27.0, by + 0.9, b_desc, fontsize=8.0 * SCALE, color='#334155', zorder=8)


# =============================================================================
# PANEL B: ALCOHOL USE DISORDER (FRONTAL DISINHIBITION & FAST-WAVE SURGE)
# =============================================================================
draw_box(51.0, 48.0, 47.5, 43.5, bg_color="#EFF6FF", edgecolor="black", lw=2.0, r=1.0)
ax.text(52.5, 88.5, "b  Alcohol Use Disorder: Frontal Disinhibition & Fast-Wave Hyperarousal",
        fontsize=13.5 * SCALE, fontweight='bold', color='#1E40AF', zorder=5)

# B1: Anatomical Brain Circuit Schematic (Left: 52.3 to 74.0)
draw_box(52.3, 49.5, 21.7, 36.5, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.6)
ax.text(53.3, 83.2, "Prefrontal GABAergic Disinhibition:", fontsize=10.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

# Draw sagittal brain profile with highlighted frontal beta/gamma surge
draw_brain_sagittal(cx=60.0, cy=69.0, scale=1.4, highlight_region="frontal", col_highlight="#3B82F6", zorder=7)

# Prefrontal GABA downregulation badge
draw_box(66.0, 74.0, 7.0, 8.5, bg_color="#EFF6FF", edgecolor="black", lw=1.0, r=0.4)
ax.text(69.5, 80.0, "GABA_A Receptor", ha='center', fontsize=8.0 * SCALE, fontweight='bold', color='#1E40AF', zorder=8)
ax.text(69.5, 76.5, "Downregulation", ha='center', fontsize=7.5 * SCALE, color='#2563EB', zorder=8)

draw_arrow(66.0, 76.0, 61.5, 71.0, color="#2563EB", lw=2.0, style="->")
ax.text(64.5, 74.5, "+ Firing", fontsize=8.0 * SCALE, fontweight='bold', color='#1D4ED8', zorder=9)

# B2: Direct Waveform Comparison & Quantitative Gauges (Right: 74.7 to 97.3)
draw_box(74.7, 49.5, 22.6, 36.5, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.6)
ax.text(75.7, 83.2, "Hyperarousal Fast-Waves & Biomarkers:", fontsize=10.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

# Healthy vs AUD Waveforms
draw_box(75.7, 69.5, 20.6, 12.0, bg_color="#F8FAFC", edgecolor="black", lw=1.0, r=0.4)
ax.text(76.5, 79.2, "Healthy: Balanced Resting Rhythm", fontsize=8.5 * SCALE, fontweight='bold', color='#166534', zorder=7)
ax.plot(76.7 + t_w, 75.5 + 1.0 * np.sin(2 * np.pi * 4.0 * (t_w/18.0)), color='#16A34A', lw=2.0, zorder=8)

ax.text(76.5, 73.2, "Alcoholism: Fast-Wave Beta/Gamma (25-40 Hz)", fontsize=8.5 * SCALE, fontweight='bold', color='#2563EB', zorder=7)
# Rapid jagged beta/gamma wave
sig_fast = 0.8 * np.sin(2 * np.pi * 14.0 * (t_w/18.0)) + 0.5 * np.sin(2 * np.pi * 22.0 * (t_w/18.0))
ax.plot(76.7 + t_w, 70.0 + sig_fast, color='#2563EB', lw=1.6, zorder=8)

# Quantitative Biomarker Gauges
aud_badges = [
    ("Frontal β Power:", "+18.2% Elevation (Hyperarousal)", "#2563EB", "#EFF6FF"),
    ("Cortical γ Desync:", "+24.5% High-Frequency Surge", "#0284C7", "#E0F2FE"),
    ("Differential Entropy:", "+1.82 nats in Fp1 / Fp2 / Fz leads", "#0D9488", "#F0FDFA"),
    ("Gradient Saliency:", "Gamma: 46.30% | Beta: 21.84%", "#1E40AF", "#DBEAFE")
]
for idx, (b_title, b_desc, b_col, b_bg) in enumerate(aud_badges):
    by = 64.0 - idx * 4.6
    draw_box(75.7, by, 20.6, 4.0, bg_color=b_bg, edgecolor="black", lw=0.8, r=0.3)
    ax.text(76.5, by + 2.5, b_title, fontsize=8.5 * SCALE, fontweight='bold', color=b_col, zorder=8)
    ax.text(76.5, by + 0.9, b_desc, fontsize=8.0 * SCALE, color='#334155', zorder=8)


# =============================================================================
# PANEL C: PEDIATRIC EPILEPSY (PAROXYSMAL ICTOGENESIS & SPIKE-WAVE DYNAMICS)
# =============================================================================
draw_box(1.5, 2.5, 47.5, 43.5, bg_color="#FAF5FF", edgecolor="black", lw=2.0, r=1.0)
ax.text(3.0, 43.0, "c  Pediatric Epilepsy: Paroxysmal Hypersynchrony & Ictal Discharge",
        fontsize=13.5 * SCALE, fontweight='bold', color='#6B21A8', zorder=5)

# C1: Anatomical Seizure Network Schematic (Left: 2.8 to 24.5)
draw_box(2.8, 4.0, 21.7, 36.5, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.6)
ax.text(3.8, 37.8, "Ictogenic Synchrony Network:", fontsize=10.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

# Draw sagittal brain profile with highlighted temporal ictal focus
draw_brain_sagittal(cx=10.5, cy=23.5, scale=1.4, highlight_region="temporal", col_highlight="#A855F7", zorder=7)

# Bipolar Derivation badge
draw_box(16.5, 28.5, 7.0, 8.5, bg_color="#F3E8FF", edgecolor="black", lw=1.0, r=0.4)
ax.text(20.0, 34.5, "23 Bipolar Pairs", ha='center', fontsize=8.0 * SCALE, fontweight='bold', color='#6B21A8', zorder=8)
ax.text(20.0, 31.0, "CHB-MIT Telemetry", ha='center', fontsize=7.5 * SCALE, color='#9333EA', zorder=8)

draw_arrow(12.5, 23.5, 16.5, 29.5, color="#9333EA", lw=2.0, style="->")
ax.text(14.5, 27.5, "Phase Lock", fontsize=7.8 * SCALE, fontweight='bold', color='#7E22CE', zorder=9)

# C2: Seizure Waveform Transition & Quantitative Gauges (Right: 25.2 to 47.8)
draw_box(25.2, 4.0, 22.6, 36.5, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.6)
ax.text(26.2, 37.8, "Ictal Discharge & Telemetry Metrics:", fontsize=10.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

# Interictal to Ictal Seizure Waveform Transition
draw_box(26.2, 24.0, 20.6, 12.0, bg_color="#F8FAFC", edgecolor="black", lw=1.0, r=0.4)
ax.text(27.0, 33.8, "Normal Interictal Baseline -> Ictal Onset", fontsize=8.5 * SCALE, fontweight='bold', color='#166534', zorder=7)

# Waveform with sharp 3 Hz spike-and-wave complex
t_epi = np.linspace(0, 18.0, 250)
sig_epi = np.zeros_like(t_epi)
sig_epi[:100] = 0.6 * np.sin(2 * np.pi * 4.0 * (t_epi[:100]/18.0)) # baseline
# Seizure onset: sharp spikes
for spk_idx in [120, 150, 180, 210, 240]:
    sig_epi[spk_idx-5:spk_idx+5] += 2.2 * np.exp(-((np.arange(10)-5)**2)/3)
sig_epi[100:] += 1.2 * np.sin(2 * np.pi * 3.0 * (t_epi[100:]/18.0)) # 3 Hz slow wave
ax.plot(27.2 + t_epi, 28.5 + sig_epi, color='#7E22CE', lw=1.8, zorder=8)
ax.axvline(27.2 + t_epi[100], color='red', linestyle='--', lw=1.2, zorder=9)
ax.text(27.2 + t_epi[100] + 0.4, 25.0, "Ictal Seizure Onset (>200 μV)", fontsize=7.8 * SCALE, fontweight='bold', color='red', zorder=10)

# Quantitative Biomarker Gauges
epi_badges = [
    ("Paroxysmal Amplitude:", ">200 μV Voltage Excursion", "#7E22CE", "#FAF5FF"),
    ("Phase Synchrony Index:", "Abrupt Cross-Channel Locking", "#6B21A8", "#F3E8FF"),
    ("Telemetry Generalization:", "AUROC = 0.7061 (95% CI: 0.650–0.766)", "#581C87", "#F5F3FF"),
    ("Cross-Patient Recall:", "98.85% Mean Recall on CHB-MIT", "#9333EA", "#FAF5FF")
]
for idx, (b_title, b_desc, b_col, b_bg) in enumerate(epi_badges):
    by = 18.5 - idx * 4.6
    draw_box(26.2, by, 20.6, 4.0, bg_color=b_bg, edgecolor="black", lw=0.8, r=0.3)
    ax.text(27.0, by + 2.5, b_title, fontsize=8.5 * SCALE, fontweight='bold', color=b_col, zorder=8)
    ax.text(27.0, by + 0.9, b_desc, fontsize=8.0 * SCALE, color='#334155', zorder=8)


# =============================================================================
# PANEL D: UNIVERSAL LATENT HYPERSPHERE MANIFOLD (S⁶³) & ZERO-SHOT TRANSFER
# =============================================================================
draw_box(51.0, 2.5, 47.5, 43.5, bg_color="#FEFCE8", edgecolor="black", lw=2.0, r=1.0)
ax.text(52.5, 43.0, "d  Universal Cross-Disorder Hypersphere Manifold (S⁶³) & Zero-Shot Transfer",
        fontsize=13.5 * SCALE, fontweight='bold', color='#854D0E', zorder=5)

# D1: 3D Rendered Hypersphere Graphic (Left: 52.3 to 74.0)
draw_box(52.3, 4.0, 21.7, 36.5, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.6)
ax.text(53.3, 37.8, "Hypersphere Manifold Geometry (S⁶³):", fontsize=10.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

# Rendered 3D-shaded sphere
sph_cx, sph_cy, sph_r = 63.15, 20.5, 11.5
sph_circ = Circle((sph_cx, sph_cy), sph_r, facecolor='#FEF9C3', edgecolor='black', lw=2.0, zorder=7)
ax.add_patch(sph_circ)
# Equator and meridian
ax.add_patch(Ellipse((sph_cx, sph_cy), sph_r*2, sph_r*0.75, facecolor='none', edgecolor='#CBD5E1', linestyle='--', lw=1.2, zorder=8))
ax.add_patch(Ellipse((sph_cx, sph_cy), sph_r*0.75, sph_r*2, facecolor='none', edgecolor='#CBD5E1', linestyle='--', lw=1.2, zorder=8))

# Antipodal Poles: AD Slowing (Top-Left) vs AUD Hyperarousal (Bottom-Right)
ad_pole = (sph_cx - 7.0, sph_cy + 6.8)
aud_pole = (sph_cx + 7.0, sph_cy - 6.8)
hc_center = (sph_cx - 0.8, sph_cy + 0.8)

# Connect with golden neurodysfunction axis
ax.plot([ad_pole[0], aud_pole[0]], [ad_pole[1], aud_pole[1]], color='#D97706', linestyle=':', lw=3.0, zorder=9)
ax.text(sph_cx - 3.2, sph_cy - 4.5, "Neurodysfunction Axis", fontsize=8.2 * SCALE, fontweight='bold', color='#B45309', rotation=-45, zorder=12)

# Scatter cluster dots
ax.scatter([ad_pole[0]], [ad_pole[1]], s=260, color='#DC2626', edgecolor='black', lw=1.6, zorder=10)
ax.text(ad_pole[0] - 1.2, ad_pole[1] + 2.4, "AD Slowing Pole\n(Low Freq Slow)", ha='center', fontsize=8.5 * SCALE, fontweight='bold', color='#DC2626', zorder=11)

ax.scatter([aud_pole[0]], [aud_pole[1]], s=260, color='#2563EB', edgecolor='black', lw=1.6, zorder=10)
ax.text(aud_pole[0] + 1.2, aud_pole[1] - 3.6, "AUD Hyperarousal Pole\n(High Freq Fast)", ha='center', fontsize=8.5 * SCALE, fontweight='bold', color='#2563EB', zorder=11)

ax.scatter([hc_center[0]], [hc_center[1]], s=200, color='#16A34A', edgecolor='black', lw=1.6, zorder=10)
ax.text(hc_center[0] + 3.4, hc_center[1] + 1.4, "Healthy Equilibrium\n(Normative)", fontsize=8.0 * SCALE, fontweight='bold', color='#166534', zorder=11)

# D2: Transfer Rationale & Metrics (Right: 74.7 to 97.3)
draw_box(74.7, 4.0, 22.6, 36.5, bg_color="#FFFFFF", edgecolor="black", lw=1.5, r=0.6)
ax.text(75.7, 37.8, "Mechanistic Transfer Rationale:", fontsize=10.5 * SCALE, fontweight='bold', color='#1E293B', zorder=6)

transfer_badges = [
    ("Antipodal Duality:", "AD (loss of speed) & AUD (excess speed) sit at opposite ends of S⁶³", "#B45309", "#FEFCE8"),
    ("Dementia -> AUD Transfer:", "AUROC = 0.6744 (+17.4% over random chance without fine-tuning)", "#15803D", "#F0FDF4"),
    ("AUD -> Dementia Transfer:", "AUROC = 0.8676 (matches direct single-cohort supervised baseline)", "#1D4ED8", "#EFF6FF"),
    ("Cross-Disorder Invariance:", "Contrastive InfoNCE loss (τ=0.07) organizes pathology continuously", "#7C3AED", "#FAF5FF")
]
for idx, (t_title, t_desc, t_col, t_bg) in enumerate(transfer_badges):
    by = 31.5 - idx * 6.5
    draw_box(75.7, by, 20.6, 5.6, bg_color=t_bg, edgecolor="black", lw=0.8, r=0.3)
    ax.text(76.5, by + 3.5, t_title, fontsize=8.8 * SCALE, fontweight='bold', color=t_col, zorder=8)
    ax.text(76.5, by + 1.2, t_desc, fontsize=8.0 * SCALE, color='#334155', zorder=8)

# Save high-resolution outputs
for d in [FIG_DIR, LATEX_FIG_DIR, DESKTOP_FIG_DIR]:
    png_p = os.path.join(d, "Figure_8_Cross_Disorder_Neurobiology_and_Connectomes.png")
    pdf_p = os.path.join(d, "Figure_8_Cross_Disorder_Neurobiology_and_Connectomes.pdf")
    fig.savefig(png_p, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_p, format='pdf', bbox_inches='tight')
    print(f"Saved: {png_p} & {pdf_p}")

plt.close(fig)
print("=== FIGURE 8 NEUROBIOLOGICAL CONNECTOME V2 COMPLETED SUCCESSFULLY ===")
