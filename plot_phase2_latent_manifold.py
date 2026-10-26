"""
Generate High-Resolution Publication Figures for Phase 2 Latent Manifold
Plots PCA and t-SNE projections and loss convergence curves.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = r"E:\All EEG"
MODEL_DIR = os.path.join(BASE_DIR, "Models")
FIG_DIR = os.path.join(BASE_DIR, "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Load data
embed_data = np.load(os.path.join(MODEL_DIR, "latent_manifold_embeddings.npz"), allow_pickle=True)
with open(os.path.join(MODEL_DIR, "latent_space_report.json"), 'r') as f:
    report = json.load(f)

pca_2d = embed_data['pca_2d']
labels = embed_data['labels']
sample_idx = embed_data['sample_idx']
tsne_2d = embed_data['tsne_2d']
sample_labels = labels[sample_idx]

DIAG_NAMES = {
    0: 'Healthy Control',
    1: "Alzheimer's Disease",
    2: 'Alcohol Use Disorder',
    3: 'Frontotemporal Dementia',
    4: 'Pediatric Epilepsy'
}

COLORS = {
    0: '#2ecc71', # Green
    1: '#e74c3c', # Red
    2: '#e67e22', # Orange
    3: '#9b59b6', # Purple
    4: '#3498db'  # Blue
}

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# 1. 2D PCA Latent Manifold Projection
ax1 = axes[0]
for class_idx, name in DIAG_NAMES.items():
    mask = (labels[sample_idx] == class_idx)
    ax1.scatter(
        pca_2d[sample_idx][mask, 0],
        pca_2d[sample_idx][mask, 1],
        c=COLORS[class_idx],
        label=name,
        alpha=0.65,
        s=20,
        edgecolors='none'
    )

ax1.set_title(f"Cross-Disorder Latent Manifold (PCA)\nExplained Variance: {report['pca_explained_variance_ratio_2d']*100:.1f}%", fontsize=14, fontweight='bold')
ax1.set_xlabel("Latent Principal Component 1", fontsize=12)
ax1.set_ylabel("Latent Principal Component 2", fontsize=12)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right', fontsize=10)

# 2. 2D t-SNE Manifold Projection
ax2 = axes[1]
for class_idx, name in DIAG_NAMES.items():
    mask = (sample_labels == class_idx)
    ax2.scatter(
        tsne_2d[mask, 0],
        tsne_2d[mask, 1],
        c=COLORS[class_idx],
        label=name,
        alpha=0.65,
        s=20,
        edgecolors='none'
    )

ax2.set_title(f"Cross-Disorder Nonlinear Embedding (t-SNE)\nSilhouette Score: {report['silhouette_score']:.3f} | Davies-Bouldin: {report['davies_bouldin_index']:.2f}", fontsize=14, fontweight='bold')
ax2.set_xlabel("t-SNE Dimension 1", fontsize=12)
ax2.set_ylabel("t-SNE Dimension 2", fontsize=12)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right', fontsize=10)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "Figure_2_Cross_Disorder_Latent_Manifold.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"Publication figure generated: {fig_path}")

# Plot Training Convergence
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
epochs = [h['epoch'] for h in report['history']]
train_loss = [h['train_loss'] for h in report['history']]
val_loss = [h['val_loss'] for h in report['history']]
val_mse = [h['val_recon_mse'] for h in report['history']]

ax.plot(epochs, train_loss, 'o-', color='#2980b9', label='Train Total Loss (Recon + InfoNCE)', linewidth=2)
ax.plot(epochs, val_loss, 's--', color='#e74c3c', label='Val Total Loss (Unseen Subjects)', linewidth=2)
ax.set_title("Self-Supervised Foundation Model Training Convergence", fontsize=13, fontweight='bold')
ax.set_xlabel("Training Epochs", fontsize=11)
ax.set_ylabel("Loss Magnitude", fontsize=11)
ax.legend(frameon=True, loc='upper right', fontsize=10)

plt.tight_layout()
loss_fig_path = os.path.join(FIG_DIR, "Figure_1_Training_Convergence.png")
plt.savefig(loss_fig_path, dpi=300)
plt.close()
print(f"Training convergence figure generated: {loss_fig_path}")
