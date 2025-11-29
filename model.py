## model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- CONFIGURATION (from config.py) ---
# Assuming these are loaded/imported:
D_EMBED_DIM = 768
N_INPUT_DIM = 1536  # D + A_subsampled + F_subsampled (512*3)
TRIPLET_MARGIN = 0.1 # Configurable margin
ALPHA_COSINE = 1.0   # Weight for Reconstruction Loss
BETA_TRIPLET = 0.5   # Weight for Discrimination Loss


# ------------------------------------------------
# I. F_decon NETWORK ARCHITECTURE (The O(1) Component)
# ------------------------------------------------
class F_decon(nn.Module):
    """
    The ParityCache Deconvolution Network.
    Input: [D, A_subsampled, F_subsampled] (1536 dims)
    Output: Raw Reconstructed Embedding (768 dims)
    """
    def __init__(self):
        super(F_decon, self).__init__()
        
        # Layer 1: Compress and process the input features
        self.fc1 = nn.Linear(N_INPUT_DIM, 1024)
        # Layer 2: Deeper feature interaction
        self.fc2 = nn.Linear(1024, 1024)
        # Layer 3: Project to the final embedding space
        self.fc3 = nn.Linear(1024, D_EMBED_DIM)
        
        # We will use ReLU activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (Batch_Size, 1536)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # Output is the raw embedding, L2 normalization happens in the loss function
        output_raw = self.fc3(x)
        
        # output_raw shape: (Batch_Size, 768)
        return output_raw


# ------------------------------------------------
# II. HYBRID LOSS FUNCTION
# ------------------------------------------------
class HybridLoss(nn.Module):
    """
    Combines Reconstruction Loss (MSE/Cosine) and Discrimination Loss (Triplet).
    L = alpha * L_Cosine + beta * L_Triplet
    """
    def __init__(self, alpha=ALPHA_COSINE, beta=BETA_TRIPLET, margin=TRIPLET_MARGIN):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin

    def forward(self, E_retrieved_raw, E_true, E_neg):
        """
        Calculates the hybrid loss.
        E_retrieved_raw: Network output (unnormalized)
        E_true: Ground truth positive embedding
        E_neg: Negative embedding (Hard Negative for discrimination)
        """
        
        # 1. L2 Normalization (applied in the loss, as requested)
        # This converts the raw network output into the required unit vector space
        E_retrieved = F.normalize(E_retrieved_raw, p=2, dim=1)
        
        # --- A. Reconstruction Loss (L_Cosine) ---
        # MSE is a simple proxy, but Cosine similarity is better for high-dim vectors.
        # Cosine Loss: 1 - cosine_similarity(E_retrieved, E_true)
        cosine_sim = F.cosine_similarity(E_retrieved, E_true, dim=1)
        L_cosine = torch.mean(1.0 - cosine_sim)
        
        # --- B. Discrimination Loss (L_Triplet) ---
        # Anchor (A) = E_retrieved, Positive (P) = E_true, Negative (N) = E_neg
        
        # Calculate Cosine Distances: dist = 1 - sim
        d_ap = 1.0 - F.cosine_similarity(E_retrieved, E_true, dim=1)   # Distance to Positive
        d_an = 1.0 - F.cosine_similarity(E_retrieved, E_neg, dim=1)    # Distance to Negative
        
        # Triplet Loss: max(0, d(A, P) - d(A, N) + margin)
        losses_triplet = d_ap - d_an + self.margin
        L_triplet = torch.mean(torch.relu(losses_triplet))
        
        # --- C. Final Hybrid Loss ---
        L_hybrid = (self.alpha * L_cosine) + (self.beta * L_triplet)
        
        return L_hybrid, L_cosine, L_triplet

# --- EXAMPLE USAGE ---
if __name__ == '__main__':
    # Initialize the model and loss
    model = F_decon()
    criterion = HybridLoss()
    
    # Simulate a batch of data (Batch size = 32)
    B = 32
    input_batch = torch.randn(B, N_INPUT_DIM).float()
    E_true_batch = F.normalize(torch.randn(B, D_EMBED_DIM).float(), dim=1)
    E_neg_batch = F.normalize(torch.randn(B, D_EMBED_DIM).float(), dim=1)

    # Forward pass
    E_retrieved_raw = model(input_batch)

    # Calculate Loss
    total_loss, cosine_loss, triplet_loss = criterion(E_retrieved_raw, E_true_batch, E_neg_batch)

    print(f"\nModel Output Structure Test:")
    print(f"Input Shape: {input_batch.shape}")
    print(f"Output Raw Shape: {E_retrieved_raw.shape}")
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Cosine Loss (Reconstruction): {cosine_loss.item():.4f}")
    print(f"Triplet Loss (Discrimination): {triplet_loss.item():.4f}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} (Target: ~2.3M)")
    
    # Ensure L2 Norm is applied correctly
    E_retrieved_norm = F.normalize(E_retrieved_raw[0].unsqueeze(0), p=2, dim=1)
    print(f"Sample Output Norm: {torch.linalg.norm(E_retrieved_norm).item():.4f} (Must be 1.0)")