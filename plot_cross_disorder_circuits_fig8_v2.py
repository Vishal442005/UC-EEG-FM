"""
Cross-Disorder Neurobiological Circuits & Hypersphere Alignment (Figure 8)
Inspired by Nature Biomedical Engineering & DeepTokenEEG PSD/Circuit diagrams.
Design Principles:
1. Pure WHITE (#FFFFFF) main panels and boxes (zero pastel background shading inside boxes).
2. Crisp simple BLACK (#000000) large bold text (min 13-22pt on 18.5x11.2 in canvas @ 300 DPI).
3. Rich visuals inside: sagittal cortical brain shapes + colored functional foci, DeepTokenEEG-style
   comparative FFT PSD spectral curves (Healthy vs AD/AUD), raw spike-wave seizure trace,
   clean magnitude bars, and 3D-shaded S^63 unit hypersphere.
4. Strict ZERO text overlap or overflow.
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

fig = plt.figure(figsize=(18.5, 11.2), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

def draw_panel(x, y, w, h, lw=2.0, ls='-', r=0.8, zorder=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={r}",
                                facecolor="#FFFFFF", edgecolor="#000000", linewidth=lw, linestyle=ls, zorder=zorder))

def draw_box(x, y, w, h, fc="#FFFFFF", ec="#000000", lw=1.4, ls='-', r=0.45, zorder=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.15,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder))

def draw_arrow(x1, y1, x2, y2, color="#000000", lw=2.2, style="-|>", rad=0.0, zorder=12):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=f"{style},head_length=7,head_width=4.5",
                                 color=color, linewidth=lw, connectionstyle=f"arc3,rad={rad}", zorder=zorder))

def draw_brain(cx, cy, s=1.3, region="parietal", hcol="#EF4444", zorder=6):
    pts = np.array([
        [-3.0, 0.0], [-2.8, 1.8], [-2.0, 3.2], [-0.5, 3.8], [1.5, 3.6],
        [2.8, 2.5], [3.2, 0.8], [3.0, -1.0], [2.2, -1.8], [1.0, -2.0],
        [-0.5, -1.8], [-1.8, -1.2], [-2.6, -0.6]
    ]) * s
    pts[:, 0] += cx
    pts[:, 1] += cy
    ax.add_patch(Polygon(pts, closed=True, facecolor="#FFFFFF", edgecolor="#000000", lw=1.6, zorder=zorder))
    ax.add_patch(Ellipse((cx + 1.8*s, cy - 1.8*s), 1.8*s, 1.2*s, facecolor="#F1F5F9", edgecolor="#000000", lw=1.2, zorder=zorder))
    ax.add_patch(Rectangle((cx + 0.4*s, cy - 3.0*s), 0.9*s, 1.6*s, facecolor="#F1F5F9", edgecolor="#000000", lw=1.2, zorder=zorder-1))

    if region == "parietal":
        ax.add_patch(Ellipse((cx + 1.4*s, cy + 1.2*s), 2.3*s, 2.3*s, facecolor=hcol, alpha=0.50, edgecolor="#000000", lw=1.3, zorder=zorder+1))
        ax.text(cx + 1.3*s, cy + 1.2*s, "Pz/O1", ha='center', va='center', fontsize=13.0, fontweight='bold', color='#000000', zorder=zorder+2)
    elif region == "frontal":
        ax.add_patch(Ellipse((cx - 1.6*s, cy + 1.2*s), 2.3*s, 2.3*s, facecolor=hcol, alpha=0.50, edgecolor="#000000", lw=1.3, zorder=zorder+1))
        ax.text(cx - 1.5*s, cy + 1.2*s, "Fp1/Fz", ha='center', va='center', fontsize=13.0, fontweight='bold', color='#000000', zorder=zorder+2)
    elif region == "temporal":
        ax.add_patch(Ellipse((cx + 0.1*s, cy - 0.3*s), 2.5*s, 1.7*s, facecolor=hcol, alpha=0.50, edgecolor="#000000", lw=1.3, zorder=zorder+1))
        ax.text(cx + 0.1*s, cy - 0.3*s, "T3/C3", ha='center', va='center', fontsize=13.0, fontweight='bold', color='#000000', zorder=zorder+2)

def draw_clean_bar(x, y, w, h, lbl, val, frac, col):
    draw_box(x, y, w, h, fc="#FFFFFF", ec="#000000", lw=1.1, r=0.3)
    ax.text(x + 0.6, y + h*0.64, lbl, fontsize=13.0, fontweight='bold', color='#000000', va='center', zorder=8)
    ax.text(x + w - 0.6, y + h*0.64, val, fontsize=13.0, fontweight='bold', color='#000000', ha='right', va='center', zorder=8)
    bw = w - 1.2
    ax.add_patch(Rectangle((x + 0.6, y + 0.5), bw, 0.9, facecolor="#FFFFFF", edgecolor="#000000", lw=0.7, zorder=8))
    ax.add_patch(Rectangle((x + 0.6, y + 0.5), max(0.6, bw*frac), 0.9, facecolor=col, edgecolor="none", zorder=9))

# =============================================================================
# PANEL a: ALZHEIMER'S DISEASE
# =============================================================================
draw_panel(1.2, 51.0, 48.0, 46.8, lw=2.0, ls='-', r=0.8)
ax.text(2.6, 94.8, "a  Alzheimer's Disease: Thalamocortical Slowing", fontsize=17.5, fontweight='bold', color='#000000', zorder=5)

draw_box(2.6, 52.6, 21.8, 39.8, fc="#FFFFFF", ec="#000000", lw=1.4, ls='--')
ax.text(13.5, 89.8, "Cholinergic Denervation", ha='center', fontsize=14.5, fontweight='bold', color='#000000', zorder=6)
draw_brain(10.0, 71.0, s=1.35, region="parietal", hcol="#EF4444")

draw_box(15.0, 79.0, 8.6, 8.2, fc="#FFFFFF", ec="#000000", lw=1.3)
ax.text(19.3, 84.4, "Thalamus", ha='center', fontsize=13.5, fontweight='bold', color='#000000', zorder=8)
ax.text(19.3, 80.8, "ACh Loss", ha='center', fontsize=13.0, fontweight='bold', color='#000000', zorder=8)
draw_arrow(12.2, 72.8, 16.0, 78.8, color="#DC2626", lw=2.6)
ax.text(13.5, 57.5, "Posterior α Loss +\nDiffuse θ Slowing", ha='center', fontsize=13.5, fontweight='bold', color='#000000', zorder=8)

# Right: PSD Spectral Shift Plot (Healthy Alpha Peak vs AD Theta Shift)
draw_box(25.2, 52.6, 22.8, 39.8, fc="#FFFFFF", ec="#000000", lw=1.4, ls='--')
ax.text(36.6, 89.8, "PSD Shift & Biomarkers", ha='center', fontsize=14.5, fontweight='bold', color='#000000', zorder=6)

draw_box(26.2, 73.8, 20.8, 14.0, fc="#FFFFFF", ec="#000000", lw=1.2)
f_x = np.linspace(0, 18.8, 160)
# Healthy 10 Hz peak (emerald), AD 5 Hz peak (crimson)
psd_hc = 1.0 + 4.8 * np.exp(-((f_x - 7.5)**2)/2.2)
psd_ad = 1.4 + 6.4 * np.exp(-((f_x - 3.6)**2)/2.6)
ax.plot(27.2 + f_x, 75.0 + psd_hc, color="#16A34A", lw=2.4, zorder=7)
ax.fill_between(27.2 + f_x, 75.0, 75.0 + psd_hc, color="#16A34A", alpha=0.15, zorder=6)
ax.plot(27.2 + f_x, 75.0 + psd_ad, color="#DC2626", lw=2.6, zorder=8)
ax.fill_between(27.2 + f_x, 75.0, 75.0 + psd_ad, color="#DC2626", alpha=0.22, zorder=7)
ax.text(30.8, 84.8, "AD 5 Hz", ha='center', fontsize=12.5, fontweight='bold', color='#000000', zorder=9)
ax.text(38.0, 82.8, "Healthy 10 Hz", ha='center', fontsize=12.5, fontweight='bold', color='#000000', zorder=9)

draw_clean_bar(26.2, 68.4, 20.8, 4.5, "θ/α (TAR):", "+48.2%", 0.85, "#DC2626")
draw_clean_bar(26.2, 63.3, 20.8, 4.5, "α Power:", "-16.4%", 0.65, "#EA580C")
draw_clean_bar(26.2, 58.2, 20.8, 4.5, "α Entropy:", "29.7%", 0.72, "#9333EA")
draw_clean_bar(26.2, 53.1, 20.8, 4.5, "θ Entropy:", "21.9%", 0.55, "#B91C1C")

# =============================================================================
# PANEL b: ALCOHOL USE DISORDER
# =============================================================================
draw_panel(50.8, 51.0, 48.0, 46.8, lw=2.0, ls='-', r=0.8)
ax.text(52.2, 94.8, "b  Alcohol Use Disorder: Fast-Wave Hyperarousal", fontsize=17.5, fontweight='bold', color='#000000', zorder=5)

draw_box(52.2, 52.6, 21.8, 39.8, fc="#FFFFFF", ec="#000000", lw=1.4, ls='--')
ax.text(63.1, 89.8, "GABAergic Disinhibition", ha='center', fontsize=14.5, fontweight='bold', color='#000000', zorder=6)
draw_brain(59.6, 71.0, s=1.35, region="frontal", hcol="#3B82F6")

draw_box(64.6, 79.0, 8.6, 8.2, fc="#FFFFFF", ec="#000000", lw=1.3)
ax.text(68.9, 84.4, r"$\mathbf{GABA_A}$", ha='center', fontsize=13.5, fontweight='bold', color='#000000', zorder=8)
ax.text(68.9, 80.8, "Reduced", ha='center', fontsize=13.0, fontweight='bold', color='#000000', zorder=8)
draw_arrow(65.4, 78.8, 61.2, 73.6, color="#2563EB", lw=2.6)
ax.text(63.1, 57.5, "Prefrontal Disinhibition\n+ Fast β/γ Synchrony", ha='center', fontsize=13.5, fontweight='bold', color='#000000', zorder=8)

draw_box(74.8, 52.6, 22.8, 39.8, fc="#FFFFFF", ec="#000000", lw=1.4, ls='--')
ax.text(86.2, 89.8, "Fast-Wave & Biomarkers", ha='center', fontsize=14.5, fontweight='bold', color='#000000', zorder=6)

draw_box(75.8, 73.8, 20.8, 14.0, fc="#FFFFFF", ec="#000000", lw=1.2)
psd_aud = 1.0 + 3.2 * np.exp(-((f_x - 7.5)**2)/2.2) + 5.6 * np.exp(-((f_x - 13.8)**2)/4.5)
ax.plot(76.8 + f_x, 75.0 + psd_hc, color="#16A34A", lw=2.4, zorder=7)
ax.plot(76.8 + f_x, 75.0 + psd_aud, color="#2563EB", lw=2.6, zorder=8)
ax.fill_between(76.8 + f_x, 75.0 + psd_hc, 75.0 + np.maximum(psd_hc, psd_aud), color="#2563EB", alpha=0.22, zorder=7)
ax.text(90.5, 84.2, "AUD β/γ Surge\n(25–40 Hz)", ha='center', fontsize=12.0, fontweight='bold', color='#000000', zorder=9)

draw_clean_bar(75.8, 68.4, 20.8, 4.5, "Frontal β:", "+18.2%", 0.70, "#2563EB")
draw_clean_bar(75.8, 63.3, 20.8, 4.5, "γ Desync:", "+24.5%", 0.78, "#0284C7")
draw_clean_bar(75.8, 58.2, 20.8, 4.5, "γ Entropy:", "46.3%", 0.92, "#0D9488")
draw_clean_bar(75.8, 53.1, 20.8, 4.5, "β Entropy:", "21.8%", 0.56, "#1E40AF")

# =============================================================================
# PANEL c: PEDIATRIC EPILEPSY
# =============================================================================
draw_panel(1.2, 2.2, 48.0, 46.8, lw=2.0, ls='-', r=0.8)
ax.text(2.6, 46.0, "c  Pediatric Epilepsy: Paroxysmal Ictal Synchrony", fontsize=17.5, fontweight='bold', color='#000000', zorder=5)

draw_box(2.6, 3.8, 21.8, 39.8, fc="#FFFFFF", ec="#000000", lw=1.4, ls='--')
ax.text(13.5, 41.0, "Ictogenic Synchrony", ha='center', fontsize=14.5, fontweight='bold', color='#000000', zorder=6)
draw_brain(10.0, 22.2, s=1.35, region="temporal", hcol="#A855F7")

draw_box(15.0, 30.2, 8.6, 8.2, fc="#FFFFFF", ec="#000000", lw=1.3)
ax.text(19.3, 35.6, "23 Pairs", ha='center', fontsize=13.5, fontweight='bold', color='#000000', zorder=8)
ax.text(19.3, 32.0, "CHB-MIT", ha='center', fontsize=13.0, fontweight='bold', color='#000000', zorder=8)
draw_arrow(12.2, 24.0, 16.0, 30.0, color="#9333EA", lw=2.6)
ax.text(13.5, 8.8, "Pyramidal Phase Lock +\nSpike-Wave Discharges", ha='center', fontsize=13.5, fontweight='bold', color='#000000', zorder=8)

draw_box(25.2, 3.8, 22.8, 39.8, fc="#FFFFFF", ec="#000000", lw=1.4, ls='--')
ax.text(36.6, 41.0, "Ictal Onset & Telemetry", ha='center', fontsize=14.5, fontweight='bold', color='#000000', zorder=6)

draw_box(26.2, 25.0, 20.8, 14.0, fc="#FFFFFF", ec="#000000", lw=1.2)
ax.text(36.6, 36.8, "Interictal → Ictal Transition", ha='center', fontsize=13.0, fontweight='bold', color='#000000', zorder=7)
t_epi = np.linspace(0, 18.8, 240)
sig_e = np.zeros_like(t_epi)
sig_e[:95] = 0.55*np.sin(2*np.pi*4.0*(t_epi[:95]/18.8))
for si in [115, 145, 175, 205, 230]:
    sig_e[si-5:si+5] += 2.3 * np.exp(-((np.arange(10)-5)**2)/2.8)
sig_e[95:] += 1.0*np.sin(2*np.pi*3.0*(t_epi[95:]/18.8))
ax.plot(27.2 + t_epi, 30.2 + sig_e, color="#7E22CE", lw=2.0, zorder=8)
ox = 27.2 + t_epi[95]
ax.plot([ox, ox], [25.6, 35.4], color="#DC2626", ls="--", lw=1.8, zorder=9)
ax.text(ox + 0.5, 26.2, "Seizure >200 μV", fontsize=12.5, fontweight='bold', color='#000000', zorder=10)

draw_clean_bar(26.2, 19.6, 20.8, 4.5, "Peak Volt:", ">200 μV", 0.90, "#7E22CE")
draw_clean_bar(26.2, 14.5, 20.8, 4.5, "Telemetry:", "0.706 AUROC", 0.71, "#6B21A8")
draw_clean_bar(26.2, 9.4, 20.8, 4.5, "θ/α Ent.:", "28.2% / 27.9%", 0.68, "#581C87")
draw_clean_bar(26.2, 4.3, 20.8, 4.5, "Recall:", "98.85%", 0.98, "#9333EA")

# =============================================================================
# PANEL d: UNIVERSAL CONTRASTIVE HYPERSPHERE S^63
# =============================================================================
draw_panel(50.8, 2.2, 48.0, 46.8, lw=2.0, ls='-', r=0.8)
ax.text(52.2, 46.0, "d  Universal Hypersphere Manifold (S⁶³) & Zero-Shot Axis", fontsize=17.5, fontweight='bold', color='#000000', zorder=5)

draw_box(52.2, 3.8, 21.8, 39.8, fc="#FFFFFF", ec="#000000", lw=1.4, ls='--')
ax.text(63.1, 41.0, "Antipodal Geometry", ha='center', fontsize=14.5, fontweight='bold', color='#000000', zorder=6)

sph_cx, sph_cy, sph_r = 63.1, 22.8, 8.6
ax.add_patch(Circle((sph_cx, sph_cy), sph_r, facecolor="#FFFFFF", edgecolor="#000000", lw=2.0, zorder=7))
ax.add_patch(Ellipse((sph_cx, sph_cy), sph_r*2, sph_r*0.72, facecolor="none", edgecolor="#94A3B8", ls="--", lw=1.3, zorder=8))
ax.add_patch(Ellipse((sph_cx, sph_cy), sph_r*0.72, sph_r*2, facecolor="none", edgecolor="#94A3B8", ls="--", lw=1.3, zorder=8))

ad_p = (sph_cx - 4.6, sph_cy + 4.6)
aud_p = (sph_cx + 4.6, sph_cy - 4.6)
hc_c = (sph_cx - 0.1, sph_cy + 0.1)

ax.plot([ad_p[0], aud_p[0]], [ad_p[1], aud_p[1]], color="#D97706", ls=":", lw=3.2, zorder=9)

ax.scatter([ad_p[0]], [ad_p[1]], s=260, color="#DC2626", edgecolor="#000000", lw=1.6, zorder=10)
ax.text(ad_p[0] + 1.0, 34.4, "AD Slowing Pole", ha='center', fontsize=13.0, fontweight='bold', color="#000000",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#DC2626", lw=1.3), zorder=12)

ax.scatter([aud_p[0]], [aud_p[1]], s=260, color="#2563EB", edgecolor="#000000", lw=1.6, zorder=10)
ax.text(63.1, 11.2, "AUD Hyperarousal Pole", ha='center', fontsize=12.5, fontweight='bold', color="#000000",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#2563EB", lw=1.3), zorder=12)

ax.scatter([hc_c[0]], [hc_c[1]], s=210, color="#16A34A", edgecolor="#000000", lw=1.6, zorder=10)
ax.text(hc_c[0] + 1.4, hc_c[1] + 2.2, "Healthy\nEquilibrium", ha='left', fontsize=12.5, fontweight='bold', color="#000000",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#16A34A", lw=1.3), zorder=12)

draw_box(53.8, 4.8, 18.6, 3.6, fc="#FFFFFF", ec="#D97706", lw=1.3)
ax.plot([54.6, 57.2], [6.6, 6.6], color="#D97706", ls=":", lw=3.2, zorder=12)
ax.text(64.6, 6.6, "Neurodysfunction Axis", ha='center', va='center', fontsize=12.0, fontweight='bold', color="#000000", zorder=12)

draw_box(74.8, 3.8, 22.8, 39.8, fc="#FFFFFF", ec="#000000", lw=1.4, ls='--')
ax.text(86.2, 41.0, "Zero-Shot Evidence", ha='center', fontsize=14.5, fontweight='bold', color='#000000', zorder=6)

draw_clean_bar(75.8, 30.8, 20.8, 6.4, "Antipodal Duality:", "Opposite Ends", 0.88, "#D97706")
draw_clean_bar(75.8, 22.6, 20.8, 6.4, "Dementia → AUD:", "0.674 AUROC", 0.674, "#16A34A")
draw_clean_bar(75.8, 14.4, 20.8, 6.4, "AUD → Dementia:", "0.868 AUROC", 0.868, "#2563EB")
draw_clean_bar(75.8, 6.2, 20.8, 6.4, "InfoNCE Loss:", "τ = 0.07", 0.82, "#9333EA")

for d in [FIG_DIR, LATEX_FIG_DIR, DESKTOP_FIG_DIR, BRAIN_DIR]:
    png_p = os.path.join(d, "Figure_8_Cross_Disorder_Neurobiology_and_Connectomes.png")
    pdf_p = os.path.join(d, "Figure_8_Cross_Disorder_Neurobiology_and_Connectomes.pdf")
    fig.savefig(png_p, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    fig.savefig(pdf_p, format='pdf', bbox_inches='tight', facecolor='#FFFFFF')

plt.close(fig)
print("=== REDESIGNED PURE-WHITE BLACK-TEXT VISUAL FIGURE 8 COMPLETED ===")
