## retrieval.py

import numpy as np
from config import CONFIG
from storage import V_S  # We import V_S to compute the difference

# --- CONSTANTS ---
# Fix: Explicitly define N and L here to avoid NameErrors
N = CONFIG['N_STATE_BITS']
L = CONFIG['L_KEY_BITS']

D_OUT = CONFIG['E_RETRIEVED_DIM']

def compute_difference_key(K_Q: np.ndarray) -> np.ndarray:
    """
    O(1) function to calculate the Difference Key D (Similarity Mask).
    K_Q must be an L-bit binary vector.
    """
    # Fix: Ensure K_Q is an array of integers (uint8) before padding
    if K_Q.dtype != np.uint8:
        K_Q = K_Q.astype(np.uint8)

    # Pad K_Q from L bits to N bits
    # pad_width is (0, N - L) -> (0, 2048) typically
    K_Q_PADDED = np.pad(K_Q, (0, N - L), 'constant').astype(np.uint8)
    
    # D is the residual context: D = V_S XOR K_Q_PADDED
    D = V_S ^ K_Q_PADDED
    
    return D

def deconvolve_baseline(D: np.ndarray, A: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    UNTRAINED BASELINE: Simulates the O(1) deconvolution network execution.
    This function is primarily for testing the mechanical pipeline without the trained model.
    """
    # --- Input Subsampling (Simulation of input to F_decon) ---
    SUBSAMPLE_RATE = 8
    D_subsampled = D[::SUBSAMPLE_RATE]
    A_subsampled = A[::SUBSAMPLE_RATE]
    F_subsampled = F[::SUBSAMPLE_RATE]
    
    # 2. Output is E_retrieved (768 dimensions) - Random Baseline
    E_retrieved = np.random.rand(D_OUT).astype(np.float32)
    
    # 3. Normalize output embedding
    norm_E = np.linalg.norm(E_retrieved)
    return E_retrieved / norm_E if norm_E > 1e-6 else E_retrieved