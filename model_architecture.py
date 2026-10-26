"""
Unified Cross-Disorder EEG Foundation Model (UC-EEG-FM) Architecture
Spatial-Temporal Transformer Encoder with Masked Patch Modeling and Cross-Subject Contrastive Heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class UCEEGFoundationModel(nn.Module):
    """
    Channel-Agnostic Spatial-Temporal Transformer for Cross-Disorder EEG Representation Learning.
    Ingests channel-aggregated or spatial feature tokens (11 biomarkers per epoch)
    and temporal patch embeddings.
    """
    def __init__(self, in_features=11, d_model=128, nhead=4, num_layers=4, latent_dim=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Linear projection for input features to transformer embedding space
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=32)
        
        # Transformer Encoder Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Masked Feature Reconstruction Head (Self-Supervised Pre-training)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, in_features)
        )
        
        # Non-linear Contrastive Projection Head (Maps to Unit Hypersphere)
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim)
        )
        
    def forward(self, x, mask_prob=0.0):
        """
        x shape: (batch_size, in_features) or (batch_size, seq_len, in_features)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1) # (batch_size, 1, in_features)
            
        b, seq_len, feat_dim = x.shape
        
        # Optional random masking for self-supervised reconstruction
        mask = None
        if self.training and mask_prob > 0.0:
            mask = torch.rand((b, seq_len), device=x.device) < mask_prob
            x_masked = x.clone()
            x_masked[mask] = 0.0
            h = self.input_proj(x_masked)
        else:
            h = self.input_proj(x)
            
        h = self.pos_encoder(h)
        encoded = self.transformer_encoder(h)
        
        # Global pooled representation
        latent_rep = encoded.mean(dim=1) # (batch_size, d_model)
        
        # Reconstruction output
        reconstruction = self.reconstruction_head(encoded)
        if reconstruction.size(1) == 1:
            reconstruction = reconstruction.squeeze(1)
            
        # Contrastive embedding normalized to unit norm
        proj = self.projection_head(latent_rep)
        proj_norm = F.normalize(proj, p=2, dim=-1)
        
        return {
            'latent_rep': latent_rep,
            'projection': proj_norm,
            'reconstruction': reconstruction,
            'mask': mask
        }

def compute_contrastive_loss(projections, labels, temperature=0.1):
    """
    Supervised Contrastive Loss (InfoNCE) across different clinical subjects.
    Pulls representations of the same clinical category together while pushing others apart.
    """
    device = projections.device
    batch_size = projections.shape[0]
    
    # Cosine similarity matrix: (batch_size, batch_size)
    sim_matrix = torch.matmul(projections, projections.T) / temperature
    
    # Positive mask: 1 where labels match, 0 otherwise
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    
    # Exclude self-similarity diagonal
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask
    
    # Numerical stability
    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()
    
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
    
    # Mean of log-likelihood over positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
    loss = -mean_log_prob_pos.mean()
    return loss
