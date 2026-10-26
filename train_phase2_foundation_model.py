"""
Phase 2 Foundation Model Pre-training & Cross-Disorder Latent Manifold Extraction
Trains the UC-EEG-FM architecture with Masked Reconstruction and Cross-Subject Contrastive Loss.
"""

import os
import glob
import time
import json
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score

from model_architecture import UCEEGFoundationModel, compute_contrastive_loss

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing Phase 2 on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

BASE_DIR = r"E:\All EEG"
FEAT_DIR = os.path.join(BASE_DIR, "Processed_Features")
MODEL_DIR = os.path.join(BASE_DIR, "Models")
os.makedirs(MODEL_DIR, exist_ok=True)

DIAGNOSIS_MAP = {
    'HealthyControl': 0,
    'AlzheimersDisease': 1,
    'AlcoholUseDisorder': 2,
    'FrontotemporalDementia': 3,
    'PediatricEpilepsy': 4
}

def load_all_clinical_data():
    all_features = []
    all_labels = []
    all_subjects = []
    all_domains = []
    subject_metadata = {}
    
    cohort_dirs = [
        ('openneuro_ds004504_dementia', 'Neurodegeneration_Dementia'),
        ('openneuro_ds005907_alcohol', 'Alcoholism_Addiction'),
        ('chbmit_epilepsy', 'Epilepsy_Seizures')
    ]
    
    print("\nLoading standardized feature tensors from all clinical cohorts...")
    for folder, domain in cohort_dirs:
        dir_path = os.path.join(FEAT_DIR, folder)
        h5_files = sorted(glob.glob(os.path.join(dir_path, "*.h5")))
        for f in h5_files:
            try:
                with h5py.File(f, 'r') as h5f:
                    feats = np.array(h5f['features_mean']) # (N_epochs, 11)
                    sub_id = str(h5f.attrs['subject_id'])
                    diag = str(h5f.attrs['diagnosis'])
                    label = DIAGNOSIS_MAP.get(diag, 0)
                    
                    n_ep = len(feats)
                    all_features.append(feats)
                    all_labels.append(np.full(n_ep, label, dtype=np.int64))
                    all_subjects.append(np.full(n_ep, sub_id, dtype=object))
                    all_domains.append(np.full(n_ep, domain, dtype=object))
                    
                    subject_metadata[sub_id] = {
                        'diagnosis': diag,
                        'domain': domain,
                        'age': float(h5f.attrs.get('age', -1)),
                        'sex': str(h5f.attrs.get('sex', 'Unknown')),
                        'epoch_count': n_ep
                    }
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
    X = np.concatenate(all_features, axis=0).astype(np.float32) # (Total_epochs, 11)
    y = np.concatenate(all_labels, axis=0)
    groups = np.concatenate(all_subjects, axis=0)
    domains = np.concatenate(all_domains, axis=0)
    
    print(f"Successfully loaded {len(X)} epochs across {len(subject_metadata)} clinical subjects.")
    return X, y, groups, domains, subject_metadata

class EEGDataset(Dataset):
    def __init__(self, X, y, groups):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.groups = groups
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_foundation_model():
    X, y, groups, domains, subject_metadata = load_all_clinical_data()
    
    # 1. Feature normalization (z-score scaling per feature across dataset)
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-8
    X_norm = (X - mean) / std
    
    # Save normalization statistics
    np.savez(os.path.join(MODEL_DIR, "normalization_stats.npz"), mean=mean, std=std)
    
    # 2. Strict Subject-Independent Group Split (75% Train Subjects, 25% Test Subjects)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)
    train_idx, val_idx = next(gss.split(X_norm, y, groups=groups))
    
    train_subs = np.unique(groups[train_idx])
    val_subs = np.unique(groups[val_idx])
    print(f"\nSubject-Independent Partition:")
    print(f"  Train: {len(train_idx)} epochs ({len(train_subs)} unseen subjects)")
    print(f"  Val:   {len(val_idx)} epochs ({len(val_subs)} unseen subjects)")
    
    train_dataset = EEGDataset(X_norm[train_idx], y[train_idx], groups[train_idx])
    val_dataset = EEGDataset(X_norm[val_idx], y[val_idx], groups[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # 3. Model Initialization
    model = UCEEGFoundationModel(in_features=11, d_model=128, nhead=4, num_layers=4, latent_dim=64, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)
    
    mse_loss_fn = torch.nn.MSELoss()
    
    print("\n" + "="*65)
    print("STARTING SELF-SUPERVISED PRE-TRAINING (15 EPOCHS)")
    print("Loss = Masked Reconstruction (MSE) + 0.5 * Supervised InfoNCE")
    print("="*65)
    
    best_val_loss = float('inf')
    history = []
    
    start_time = time.time()
    
    for epoch in range(1, 16):
        model.train()
        train_recon_loss = 0.0
        train_contra_loss = 0.0
        total_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            out = model(batch_x, mask_prob=0.25) # 25% masked patch reconstruction
            
            # Reconstruction Loss
            recon_loss = mse_loss_fn(out['reconstruction'], batch_x)
            
            # Supervised Contrastive Loss
            contra_loss = compute_contrastive_loss(out['projection'], batch_y, temperature=0.1)
            
            loss = recon_loss + 0.5 * contra_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_recon_loss += recon_loss.item()
            train_contra_loss += contra_loss.item()
            total_batches += 1
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_recon_loss = 0.0
        val_contra_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out = model(batch_x, mask_prob=0.0)
                recon_loss = mse_loss_fn(out['reconstruction'], batch_x)
                contra_loss = compute_contrastive_loss(out['projection'], batch_y, temperature=0.1)
                
                val_recon_loss += recon_loss.item()
                val_contra_loss += contra_loss.item()
                val_batches += 1
                
        avg_train_loss = (train_recon_loss + 0.5 * train_contra_loss) / total_batches
        avg_val_loss = (val_recon_loss + 0.5 * val_contra_loss) / val_batches
        
        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_recon_mse': val_recon_loss / val_batches,
            'val_contra_loss': val_contra_loss / val_batches
        })
        
        print(f"Epoch [{epoch:2d}/15] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} (MSE: {val_recon_loss/val_batches:.4f}, InfoNCE: {val_contra_loss/val_batches:.4f})")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "uc_eeg_fm_backbone.pt"))
            
    print(f"\nPre-training completed in {(time.time() - start_time):.1f} seconds. Best Val Loss: {best_val_loss:.4f}")
    
    # 4. Extract Latent Manifold Embeddings for ALL 146 Subjects
    print("\nExtracting cross-disorder latent space embeddings for all 49,472 epochs...")
    model.eval()
    all_embeddings = []
    
    full_dataset = EEGDataset(X_norm, y, groups)
    full_loader = DataLoader(full_dataset, batch_size=256, shuffle=False)
    
    with torch.no_grad():
        for batch_x, _ in full_loader:
            batch_x = batch_x.to(device)
            out = model(batch_x, mask_prob=0.0)
            all_embeddings.append(out['projection'].cpu().numpy())
            
    Z = np.concatenate(all_embeddings, axis=0) # (Total_epochs, 64)
    
    # 5. Latent Manifold Dimensionality Reduction (PCA & t-SNE)
    print("Computing 2D Latent Manifold Projections (PCA & t-SNE)...")
    pca = PCA(n_components=2, random_state=42)
    Z_pca = pca.fit_transform(Z)
    
    # Subsample 5,000 representative points for t-SNE to compute fast & clean 2D manifold
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(Z), size=min(5000, len(Z)), replace=False)
    tsne = TSNE(n_components=2, perplexity=35, random_state=42, max_iter=1000)
    Z_tsne = tsne.fit_transform(Z[sample_idx])
    
    # Quantitative Manifold Evaluation Metrics
    sil_score = silhouette_score(Z[sample_idx], y[sample_idx])
    db_score = davies_bouldin_score(Z[sample_idx], y[sample_idx])
    
    print("\n" + "="*65)
    print("CROSS-DISORDER LATENT MANIFOLD EVALUATION METRICS")
    print("="*65)
    print(f"Explained Variance Ratio (First 2 PCA dims): {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
    print(f"Silhouette Coefficient:                     {sil_score:.4f} (Higher is better)")
    print(f"Davies-Bouldin Index:                       {db_score:.4f} (Lower is better)")
    print("="*65)
    
    # 6. Save Manifold Embeddings & Verification Report
    embed_file = os.path.join(MODEL_DIR, "latent_manifold_embeddings.npz")
    np.savez_compressed(
        embed_file,
        embeddings=Z,
        pca_2d=Z_pca,
        labels=y,
        groups=groups,
        domains=domains,
        sample_idx=sample_idx,
        tsne_2d=Z_tsne
    )
    
    report = {
        'training_epochs': 15,
        'final_val_loss': float(best_val_loss),
        'pca_explained_variance_ratio_2d': float(np.sum(pca.explained_variance_ratio_)),
        'silhouette_score': float(sil_score),
        'davies_bouldin_index': float(db_score),
        'total_evaluated_epochs': int(len(Z)),
        'total_evaluated_subjects': int(len(subject_metadata)),
        'history': history,
        'diagnoses': {k: int(np.sum(y == v)) for k, v in DIAGNOSIS_MAP.items()}
    }
    
    with open(os.path.join(MODEL_DIR, "latent_space_report.json"), 'w') as jf:
        json.dump(report, jf, indent=2)
        
    print(f"All model weights, manifold embeddings, and reports saved to: {MODEL_DIR}")

if __name__ == '__main__':
    train_foundation_model()
