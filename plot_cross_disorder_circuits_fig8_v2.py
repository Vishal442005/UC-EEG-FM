"""
Cross-Disorder Functional Neurobiology & Connectome Circuit Diagram (Figure 8)
- Runs cleanly on healthy workspace: C:/Users/Visha/Desktop/NewAlcohol/UC_EEG_FM
- Canvas calibrated to (20.0, 12.2) at 300 DPI (6000x3660 px) so every label is huge and legible in print.
- Visual enhancements: anatomical sagittal brains with colored lesion hotspots, quantitative meter bars inside biomarker cards,
  direct healthy vs. pathological EEG waveforms, bounded ictal transition markers, and 3D-shaded S^63 hypersphere.
- Strict black-and-white structural outlines (edgecolor='black', lw=2.0), rich inside colors, ZERO text overlap.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Polygon, Rectangle

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

def draw_arrow(x1, y1, x2, y2, color="#1E293B", lw=2.4, style="-|>", rad=0.0, zorder=12):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=f"{style},head_length=7,head_width=4.5",
                            color=color, linewidth=lw,
                            connectionstyle=f"arc3,rad={rad}", zorder=zorder)
    ax.add_patch(arrow)

def draw_brain_sagittal(cx, cy, scale=1.15, highlight_region="parietal", col_highlight="#EF4444", zorder=5):
    c_pts = np.array([
        [-3.0, 0.0], [-2.8, 1.8], [-2.0, 3.2], [-0.5, 3.8], [1.5, 3.6],
        [2.8, 2.5], [3.2, 0.8], [3.0, -1.0], [2.2, -1.8], [1.0, -2.0],
        [-0.5, -1.8], [-1.8, -1.2], [-2.6, -0.6]
    ]) * scale
    c_pts[:, 0] += cx
    c_pts[:, 1] += cy
    ax.add_patch(Polygon(c_pts, closed=True, facecolor="#F8FAFC", edgecolor="black", lw=1.6, zorder=zorder))

    cb = Ellipse((cx + 1.8*scale, cy - 1.8*scale), 1.8*scale, 1.2*scale, facecolor="#E2E8F0", edgecolor="black", lw=1.2, zorder=zorder)
    ax.add_patch(cb)
    bs = Rectangle((cx + 0.4*scale, cy - 3.0*scale), 0.9*scale, 1.6*scale, facecolor="#E2E8F0", edgecolor="black", lw=1.2, zorder=zorder-1)
    ax.add_patch(bs)

    if highlight_region == "parietal":
        ax.add_patch(Ellipse((cx + 1.4*scale, cy + 1.2*scale), 2.3*scale, 2.3*scale, facecolor=col_highlight, alpha=0.45, edgecolor=col_highlight, lw=1.6, zorder=zorder+1))
        ax.text(cx + 1.2*scale, cy + 1.2*scale, "Pz/O1", ha='center', va='center', fontsize=12.5, fontweight='bold', color='#7F1D1D', zorder=zorder+2)
    elif highlight_region == "frontal":
        ax.add_patch(Ellipse((cx - 1.6*scale, cy + 1.2*scale), 2.3*scale, 2.3*scale, facecolor=col_highlight, alpha=0.45, edgecolor=col_highlight, lw=1.6, zorder=zorder+1))
        ax.text(cx - 1.5*scale, cy + 1.2*scale, "Fp1/Fz", ha='center', va='center', fontsize=12.5, fontweight='bold', color='#1E3A8A', zorder=zorder+2)
    elif highlight_region == "temporal":
        ax.add_patch(Ellipse((cx + 0.1*scale, cy - 0.3*scale), 2.5*scale, 1.7*scale, facecolor=col_highlight, alpha=0.45, edgecolor=col_highlight, lw=1.6, zorder=zorder+1))
        ax.text(cx + 0.1*scale, cy - 0.3*scale, "T3/C3", ha='center', va='center', fontsize=12.5, fontweight='bold', color='#581C87', zorder=zorder+2)

def draw_metric_bar(x, y, w, h, title, val_str, frac, col, bg):
    draw_box(x, y, w, h, bg_color=bg, edgecolor="black", lw=1.0, r=0.3)
    ax.text(x + 0.7, y + h*0.64, title, fontsize=12.5, fontweight='bold', color=col, va='center', zorder=8)
    ax.text(x + w - 0.7, y + h*0.64, val_str, fontsize=12.5, fontweight='bold', color='#0F172A', ha='right', va='center', zorder=8)
    bar_w = w - 1.4
    ax.add_patch(Rectangle((x + 0.7, y + 0.55), bar_w, 0.8, facecolor="#FFFFFF", edgecolor="black", lw=0.6, zorder=8))
    ax.add_patch(Rectangle((x + 0.7, y + 0.55), max(0.6, bar_w * frac), 0.8, facecolor=col, edgecolor="none", zorder=9))

draw_box(1.2, 93.4, 97.6, 5.4, bg_color="#FFFFFF", edgecolor="black", lw=2.2, r=0.8)
ax.text(50.0, 96.7, "Cross-Disorder Neurobiological Circuits & Hypersphere Alignment",
        ha='center', va='center', fontsize=23, fontweight='bold', color='#0F172A', zorder=5)
ax.text(50.0, 94.4, "Pathophysiological Mechanisms Mapped to Shared Unit Hypersphere Geometry (S⁶³)",
        ha='center', va='center', fontsize=15, fontweight='bold', color='#475569', zorder=5)

t_w = np.linspace(0, 19.0, 220)

# PANEL a: ALZHEIMER'S DISEASE
draw_box(1.2, 48.2, 47.9, 43.6, bg_color="#FEF2F2", edgecolor="black", lw=2.2, r=1.0)
ax.text(2.4, 89.2, "a  Alzheimer's Disease: Thalamocortical Slowing", fontsize=17.5, fontweight='bold', color='#991B1B', zorder=5)

draw_box(2.4, 49.6, 21.8, 37.2, bg_color="#FFFFFF", edgecolor="black", lw=1.6, r=0.6)
ax.text(3.3, 84.4, "Cholinergic Denervation:", fontsize=14.2, fontweight='bold', color='#1E293B', zorder=6)
draw_brain_sagittal(cx=10.2, cy=66.5, scale=1.35, highlight_region="parietal", col_highlight="#EF4444", zorder=7)

draw_box(15.2, 74.6, 8.2, 7.8, bg_color="#FFF1F2", edgecolor="black", lw=1.2, r=0.4)
ax.text(19.3, 79.8, "Thalamus", ha='center', fontsize=13.0, fontweight='bold', color='#991B1B', zorder=8)
ax.text(19.3, 76.2, "ACh Loss", ha='center', fontsize=12.5, fontweight='bold', color='#DC2626', zorder=8)
draw_arrow(12.2, 68.5, 16.0, 74.4, color="#DC2626", lw=2.4)
ax.text(13.5, 55.0, "Posterior Alpha Collapse\n+ Diffuse Theta Slowing", ha='center', fontsize=12.5, fontweight='bold', color='#991B1B', zorder=8)

draw_box(25.0, 49.6, 22.9, 37.2, bg_color="#FFFFFF", edgecolor="black", lw=1.6, r=0.6)
ax.text(25.9, 84.4, "Waveforms & Biomarkers:", fontsize=14.2, fontweight='bold', color='#1E293B', zorder=6)
draw_box(25.9, 70.6, 21.1, 12.2, bg_color="#F8FAFC", edgecolor="black", lw=1.2, r=0.4)
ax.text(26.6, 80.5, "Healthy: 10 Hz Alpha", fontsize=12.5, fontweight='bold', color='#166534', zorder=7)
ax.plot(26.8 + t_w, 77.6 + 1.0 * np.sin(2 * np.pi * 5.0 * (t_w/19.0)), color='#16A34A', lw=2.2, zorder=8)
ax.text(26.6, 74.6, "AD: 5 Hz Theta Slowing", fontsize=12.5, fontweight='bold', color='#DC2626', zorder=7)
ax.plot(26.8 + t_w, 72.0 + 1.15 * np.sin(2 * np.pi * 2.2 * (t_w/19.0)), color='#DC2626', lw=2.4, zorder=8)

draw_metric_bar(25.9, 65.0, 21.1, 4.4, "θ/α Ratio (TAR):", "+48.2% Surge", 0.85, "#DC2626", "#FEF2F2")
draw_metric_bar(25.9, 59.9, 21.1, 4.4, "Posterior α Power:", "-16.4% Drop", 0.65, "#EA580C", "#FFF7ED")
draw_metric_bar(25.9, 54.8, 21.1, 4.4, "α Diff. Entropy:", "29.7% Saliency", 0.72, "#7C3AED", "#FAF5FF")
draw_metric_bar(25.9, 49.8, 21.1, 4.4, "θ Diff. Entropy:", "21.9% Saliency", 0.55, "#B91C1C", "#FFF1F2")

# PANEL b: ALCOHOL USE DISORDER
draw_box(50.9, 48.2, 47.9, 43.6, bg_color="#EFF6FF", edgecolor="black", lw=2.2, r=1.0)
ax.text(52.1, 89.2, "b  Alcohol Use Disorder: Fast-Wave Hyperarousal", fontsize=17.5, fontweight='bold', color='#1E40AF', zorder=5)

draw_box(52.1, 49.6, 21.8, 37.2, bg_color="#FFFFFF", edgecolor="black", lw=1.6, r=0.6)
ax.text(53.0, 84.4, "GABAergic Disinhibition:", fontsize=14.2, fontweight='bold', color='#1E293B', zorder=6)
draw_brain_sagittal(cx=59.9, cy=66.5, scale=1.35, highlight_region="frontal", col_highlight="#3B82F6", zorder=7)

draw_box(64.9, 74.6, 8.2, 7.8, bg_color="#EFF6FF", edgecolor="black", lw=1.2, r=0.4)
ax.text(69.0, 79.8, "GABA_A", ha='center', fontsize=13.0, fontweight='bold', color='#1E40AF', zorder=8)
ax.text(69.0, 76.2, "Reduced", ha='center', fontsize=12.5, fontweight='bold', color='#2563EB', zorder=8)
draw_arrow(65.5, 74.4, 61.5, 69.5, color="#2563EB", lw=2.4)
ax.text(63.2, 55.0, "Prefrontal Disinhibition\n+ Fast β/γ Synchrony", ha='center', fontsize=12.5, fontweight='bold', color='#1E40AF', zorder=8)

draw_box(74.7, 49.6, 22.9, 37.2, bg_color="#FFFFFF", edgecolor="black", lw=1.6, r=0.6)
ax.text(75.6, 84.4, "Waveforms & Biomarkers:", fontsize=14.2, fontweight='bold', color='#1E293B', zorder=6)
draw_box(75.6, 70.6, 21.1, 12.2, bg_color="#F8FAFC", edgecolor="black", lw=1.2, r=0.4)
ax.text(76.3, 80.5, "Healthy: Resting Rhythm", fontsize=12.5, fontweight='bold', color='#166534', zorder=7)
ax.plot(76.5 + t_w, 77.6 + 1.0 * np.sin(2 * np.pi * 4.5 * (t_w/19.0)), color='#16A34A', lw=2.2, zorder=8)
ax.text(76.3, 74.6, "AUD: Fast β/γ (25–40 Hz)", fontsize=12.5, fontweight='bold', color='#2563EB', zorder=7)
sig_fast = 0.75 * np.sin(2 * np.pi * 14.0 * (t_w/19.0)) + 0.45 * np.sin(2 * np.pi * 23.0 * (t_w/19.0))
ax.plot(76.5 + t_w, 72.0 + sig_fast, color='#2563EB', lw=1.8, zorder=8)

draw_metric_bar(75.6, 65.0, 21.1, 4.4, "Frontal β Power:", "+18.2% Rise", 0.70, "#2563EB", "#EFF6FF")
draw_metric_bar(75.6, 59.9, 21.1, 4.4, "Cortical γ Desync:", "+24.5% Rise", 0.78, "#0284C7", "#E0F2FE")
draw_metric_bar(75.6, 54.8, 21.1, 4.4, "γ Diff. Entropy:", "46.3% Saliency", 0.92, "#0D9488", "#F0FDFA")
draw_metric_bar(75.6, 49.8, 21.1, 4.4, "β Diff. Entropy:", "21.8% Saliency", 0.56, "#1E40AF", "#DBEAFE")

# PANEL c: PEDIATRIC EPILEPSY
draw_box(1.2, 2.2, 47.9, 43.6, bg_color="#FAF5FF", edgecolor="black", lw=2.2, r=1.0)
ax.text(2.4, 43.2, "c  Pediatric Epilepsy: Paroxysmal Ictal Synchrony", fontsize=17.5, fontweight='bold', color='#6B21A8', zorder=5)

draw_box(2.4, 3.6, 21.8, 37.2, bg_color="#FFFFFF", edgecolor="black", lw=1.6, r=0.6)
ax.text(3.3, 38.4, "Ictogenic Synchrony:", fontsize=14.2, fontweight='bold', color='#1E293B', zorder=6)
draw_brain_sagittal(cx=10.2, cy=20.5, scale=1.35, highlight_region="temporal", col_highlight="#A855F7", zorder=7)

draw_box(15.2, 28.6, 8.2, 7.8, bg_color="#F3E8FF", edgecolor="black", lw=1.2, r=0.4)
ax.text(19.3, 33.8, "23 Pairs", ha='center', fontsize=13.0, fontweight='bold', color='#6B21A8', zorder=8)
ax.text(19.3, 30.2, "CHB-MIT", ha='center', fontsize=12.5, fontweight='bold', color='#9333EA', zorder=8)
draw_arrow(12.2, 22.5, 16.0, 28.4, color="#9333EA", lw=2.4)
ax.text(13.5, 8.5, "Sudden Pyramidal Phase Lock\n+ High-Voltage Spike Waves", ha='center', fontsize=12.5, fontweight='bold', color='#6B21A8', zorder=8)

draw_box(25.0, 3.6, 22.9, 37.2, bg_color="#FFFFFF", edgecolor="black", lw=1.6, r=0.6)
ax.text(25.9, 38.4, "Ictal Transition & Metrics:", fontsize=14.2, fontweight='bold', color='#1E293B', zorder=6)
draw_box(25.9, 24.6, 21.1, 12.2, bg_color="#F8FAFC", edgecolor="black", lw=1.2, r=0.4)
ax.text(26.6, 34.5, "Interictal → Ictal Onset", fontsize=12.5, fontweight='bold', color='#166534', zorder=7)

t_epi = np.linspace(0, 19.0, 250)
sig_epi = np.zeros_like(t_epi)
sig_epi[:100] = 0.5 * np.sin(2 * np.pi * 4.0 * (t_epi[:100]/19.0))
for spk_idx in [120, 150, 180, 210, 240]:
    sig_epi[spk_idx-5:spk_idx+5] += 2.0 * np.exp(-((np.arange(10)-5)**2)/3)
sig_epi[100:] += 1.0 * np.sin(2 * np.pi * 3.0 * (t_epi[100:]/19.0))
ax.plot(26.8 + t_epi, 29.0 + sig_epi, color='#7E22CE', lw=1.9, zorder=8)
onset_x = 26.8 + t_epi[100]
ax.plot([onset_x, onset_x], [25.0, 33.8], color='red', linestyle='--', lw=1.8, zorder=9)
ax.text(onset_x + 0.5, 25.4, "Seizure >200 μV", fontsize=12.0, fontweight='bold', color='red', zorder=10)

draw_metric_bar(25.9, 19.0, 21.1, 4.4, "Peak Voltage:", ">200 μV Spikes", 0.90, "#7E22CE", "#FAF5FF")
draw_metric_bar(25.9, 13.9, 21.1, 4.4, "Telemetry AUROC:", "0.706 (5-Fold)", 0.71, "#6B21A8", "#F3E8FF")
draw_metric_bar(25.9, 8.8, 21.1, 4.4, "θ/α Diff. Ent.:", "28.2% / 27.9%", 0.68, "#581C87", "#F5F3FF")
draw_metric_bar(25.9, 3.8, 21.1, 4.4, "CHB-MIT Recall:", "98.85% Mean", 0.98, "#9333EA", "#FAF5FF")

# PANEL d: UNIVERSAL HYPERSPHERE MANIFOLD S^63
draw_box(50.9, 2.2, 47.9, 43.6, bg_color="#FEFCE8", edgecolor="black", lw=2.2, r=1.0)
ax.text(52.1, 43.2, "d  Universal Hypersphere Manifold (S⁶³) & Zero-Shot Axis", fontsize=17.5, fontweight='bold', color='#854D0E', zorder=5)

draw_box(52.1, 3.6, 21.8, 37.2, bg_color="#FFFFFF", edgecolor="black", lw=1.6, r=0.6)
ax.text(53.0, 38.4, "Antipodal Sphere (S⁶³):", fontsize=14.2, fontweight='bold', color='#1E293B', zorder=6)

sph_cx, sph_cy, sph_r = 63.0, 20.4, 9.6
ax.add_patch(Circle((sph_cx, sph_cy), sph_r, facecolor='#FEF9C3', edgecolor='black', lw=2.0, zorder=7))
ax.add_patch(Ellipse((sph_cx, sph_cy), sph_r*2, sph_r*0.72, facecolor='none', edgecolor='#94A3B8', linestyle='--', lw=1.3, zorder=8))
ax.add_patch(Ellipse((sph_cx, sph_cy), sph_r*0.72, sph_r*2, facecolor='none', edgecolor='#94A3B8', linestyle='--', lw=1.3, zorder=8))

ad_pole = (sph_cx - 5.4, sph_cy + 5.4)
aud_pole = (sph_cx + 5.4, sph_cy - 5.4)
hc_center = (sph_cx - 0.3, sph_cy + 0.3)

ax.plot([ad_pole[0], aud_pole[0]], [ad_pole[1], aud_pole[1]], color='#D97706', linestyle=':', lw=3.2, zorder=9)
ax.text(sph_cx - 3.8, sph_cy - 2.4, "Neurodysfunction\nAxis", fontsize=11.5, fontweight='bold', color='#B45309',
        rotation=-45, ha='center', va='center',
        bbox=dict(boxstyle="round,pad=0.20", facecolor="#FEFCE8", edgecolor="#D97706", lw=1.0), zorder=11)

ax.scatter([ad_pole[0]], [ad_pole[1]], s=260, color='#DC2626', edgecolor='black', lw=1.6, zorder=10)
ax.text(ad_pole[0] + 0.8, ad_pole[1] + 2.6, "AD Slowing Pole", ha='center', fontsize=12.8, fontweight='bold', color='#DC2626',
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#DC2626", lw=1.0), zorder=12)

ax.scatter([aud_pole[0]], [aud_pole[1]], s=260, color='#2563EB', edgecolor='black', lw=1.6, zorder=10)
ax.text(aud_pole[0] + 0.3, aud_pole[1] - 3.3, "AUD Hyperarousal Pole", ha='center', fontsize=12.2, fontweight='bold', color='#2563EB',
        bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFFFFF", edgecolor="#2563EB", lw=1.0), zorder=12)

ax.scatter([hc_center[0]], [hc_center[1]], s=210, color='#16A34A', edgecolor='black', lw=1.6, zorder=10)
ax.text(hc_center[0] + 1.5, hc_center[1] + 2.0, "Healthy\nEquilibrium", ha='left', fontsize=12.2, fontweight='bold', color='#166534',
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#F0FDF4", edgecolor="#16A34A", lw=1.0), zorder=11)

draw_box(74.7, 3.6, 22.9, 37.2, bg_color="#FFFFFF", edgecolor="black", lw=1.6, r=0.6)
ax.text(75.6, 38.4, "Transfer Evidence:", fontsize=14.2, fontweight='bold', color='#1E293B', zorder=6)

draw_metric_bar(75.6, 28.8, 21.1, 6.2, "Antipodal Duality:", "Opposite Poles", 0.88, "#B45309", "#FEFCE8")
draw_metric_bar(75.6, 21.1, 21.1, 6.2, "Dementia → AUD:", "0.674 AUROC", 0.674, "#15803D", "#F0FDF4")
draw_metric_bar(75.6, 13.4, 21.1, 6.2, "AUD → Dementia:", "0.868 AUROC", 0.868, "#1D4ED8", "#EFF6FF")
draw_metric_bar(75.6, 5.7, 21.1, 6.2, "InfoNCE Alignment:", "τ = 0.07 Loss", 0.82, "#7C3AED", "#FAF5FF")

for d in [FIG_DIR, LATEX_FIG_DIR, DESKTOP_FIG_DIR, BRAIN_DIR]:
    png_p = os.path.join(d, "Figure_8_Cross_Disorder_Neurobiology_and_Connectomes.png")
    pdf_p = os.path.join(d, "Figure_8_Cross_Disorder_Neurobiology_and_Connectomes.pdf")
    fig.savefig(png_p, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_p, format='pdf', bbox_inches='tight')

plt.close(fig)
print("=== VISUAL FIGURE 8 COMPLETED (LARGE TEXT & ZERO OVERLAP) ===")
