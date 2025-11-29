## encoder.py

import numpy as np
from scipy.sparse import coo_matrix

from config import CONFIG, ENCODER_PARAMS

# --- GLOBAL PROJECTION MATRIX (P) ---
P_sparse = None

def initialize_projection_matrix():
    """Generates the sparse, random (L x D) projection matrix P."""
    global P_sparse
    D = CONFIG['D_EMBED_DIM']   # 768
    L = CONFIG['L_KEY_BITS']    # 2048
    rho = ENCODER_PARAMS['PROJECTION_DENSITY']
    seed = ENCODER_PARAMS['RANDOM_SEED']
    
    np.random.seed(seed)
    
    # Total number of non-zero elements
    num_non_zeros = int(D * L * rho)
    
    # --- FIX: Matrix Orientation for P @ E ---
    # We want output size L, so we need L rows.
    # We take input size D, so we need D columns.
    # Shape must be (L, D) to multiply with vector (D,)
    
    row_indices = np.random.randint(0, L, num_non_zeros) # Range 0 to 2047
    col_indices = np.random.randint(0, D, num_non_zeros) # Range 0 to 767
    
    # Values are +/- 1 for the SimHash projection
    values = np.random.choice([-1.0, 1.0], num_non_zeros)
    
    # Create the sparse matrix (COO format is fast for multiplication)
    P_sparse = coo_matrix((values, (row_indices, col_indices)), shape=(L, D))
    
    print(f"Encoder initialized: P_sparse shape {P_sparse.shape} (L, D), density {rho*100:.2f}%.")

def encode_to_key(E_pooled: np.ndarray) -> np.ndarray:
    """
    Generates the L-bit Context Key (K_C) from a pooled embedding (E).
    Complexity: O(1) via sparse matrix-vector multiplication.
    """
    if P_sparse is None:
        raise RuntimeError("Projection matrix P not initialized. Call initialize_projection_matrix() first.")
    
    # 1. Normalize Embedding (crucial for LSH)
    norm_E = np.linalg.norm(E_pooled)
    E_normalized = E_pooled / norm_E if norm_E > 1e-6 else np.zeros_like(E_pooled)

    # 2. Sparse Projection: 
    # (L, D) @ (D,) -> (L,)
    # projection shape: (2048,)
    projection = P_sparse.dot(E_normalized)

    # 3. SimHash (Sign Function): K_C is a binary vector (0 or 1).
    K_C = (projection >= 0).astype(np.uint8)
    
    return K_C