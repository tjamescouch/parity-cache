## config.py

import numpy as np

# --- 1. CORE ARCHITECTURE ---
CONFIG = {
    'N_STATE_BITS': 4096,      # Total size of the State Vector (V_S) in bits
    'L_KEY_BITS': 2048,        # Size of the Context Key (K_C) in bits
    'D_EMBED_DIM': 768,        # Dimensionality of the input embedding (E)
    'E_RETRIEVED_DIM': 768,    # Dimensionality of the reconstructed embedding (E_retrieved)
}

# --- 2. STORAGE (ParityCore) PARAMETERS ---
STORAGE_PARAMS = {
    'A_F_PRECISION': np.uint8,  # Precision for Age (A) and Frequency (F) vectors (8 bits, 0-255)
    'LAMBDA_DECAY': 1.0,        # Weight (lambda) balancing Age vs. Frequency in decay mask
}

# --- 3. ENCODER (SimHash) PARAMETERS ---
ENCODER_PARAMS = {
    'PROJECTION_DENSITY': 0.01, # Density (rho) of the sparse random projection matrix (P)
    'RANDOM_SEED': 42,          # Seed for reproducible random projection matrix
    'POOLING_STRATEGY': 'mean_N_tokens', # Primary strategy: Mean pooling over N tokens
    'N_TOKENS_BLOCK': 128,      # N tokens for the fixed-size pooling block
}