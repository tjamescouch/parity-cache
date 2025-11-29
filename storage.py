## storage.py

import numpy as np
from config import CONFIG, STORAGE_PARAMS

# --- GLOBAL STATE VECTORS ---
N = CONFIG['N_STATE_BITS']
L = CONFIG['L_KEY_BITS']
PRECISION = STORAGE_PARAMS['A_F_PRECISION']
LAMBDA = STORAGE_PARAMS['LAMBDA_DECAY']

V_S = np.zeros(N, dtype=np.uint8)    # State Vector
A = np.zeros(N, dtype=PRECISION)     # Age Vector
F = np.zeros(N, dtype=PRECISION)     # Frequency Vector

def initialize_state():
    """Resets the ParityCache state vectors."""
    global V_S, A, F
    V_S[:] = 0
    A[:] = 0
    F[:] = 0
    print(f"Storage initialized: V_S size {N} bits.")

def update_state(K_C: np.ndarray) -> np.ndarray:
    """
    O(1) Vectorized Update of V_S with Context-Aware Decay.
    K_C must be an L-bit binary vector (0 or 1).
    """
    global V_S, A, F
    
    # 1. Prepare Key: Pad K_C from L bits to N bits
    K_C_PADDED = np.pad(K_C, (0, N - L), 'constant').astype(np.uint8)

    # 2. Core Accumulation & Flip Identification (O(1))
    FLIPPED_BITS = V_S ^ K_C_PADDED
    V_S_NEW = V_S ^ K_C_PADDED
    
    # 3. Update Age Vector (A) - O(1)
    A = np.clip(A + 1, 0, 255)                  # Increment all
    A[FLIPPED_BITS == 1] = 0                    # Reset where flipped
    
    # 4. Update Frequency Vector (F) - O(1)
    F_update = F.copy()
    F_update[V_S_NEW == 1] = F_update[V_S_NEW == 1] + 1
    F = np.clip(F_update, 0, 255)

    # 5. Generate Context-Aware Decay Mask (M) - TRUE O(1) Vectorization
    A_float = A.astype(np.float32)
    F_float = F.astype(np.float32)
    
    # P(decay) = A / (A + lambda * F) + Epsilon for stability
    P_DECAY_VECTOR = A_float / (A_float + LAMBDA * F_float + 1e-6)
    
    RANDOM_VECTOR = np.random.uniform(0.0, 1.0, N)
    
    # M[i] = 0 (Decay) if P_DECAY > Random, otherwise 1 (Protect)
    M = (RANDOM_VECTOR <= P_DECAY_VECTOR).astype(np.uint8) # Note: <= P_decay for decay

    # 6. Apply the Decay Mask
    V_S = V_S_NEW & (1 - M) # Apply Mask M by setting decayed bits to 0
    
    return V_S