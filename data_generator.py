## data_generator.py

import numpy as np
import random
from typing import List, Tuple, Any
from abc import ABC, abstractmethod

# Import the core ParityCache modules
from config import CONFIG
import encoder
import storage
import retrieval

# --- 1. ABSTRACT INTERFACE ---
class AbstractEmbeddingProvider(ABC):
    """
    Interface for integrating real-world embedding models (Ollama, HuggingFace, etc.).
    All implementations MUST conform to this contract.
    """
    
    @abstractmethod
    def __init__(self, source: Any):
        pass

    @abstractmethod
    def get_pooled_embedding(self, text_block: str) -> np.ndarray:
        """
        Returns the single, normalized, D-dimensional pooled embedding vector (E).
        """
        pass
    
    @abstractmethod
    def get_random_context(self) -> str:
        """Returns a random text block from the corpus."""
        pass

# --- 2. CONCRETE SIMULATION PROVIDER ---
class SimulatedEmbeddingProvider(AbstractEmbeddingProvider):
    """
    Simulated provider for prototype testing. 
    Returns random vectors based on text hash to ensure determinism without a model.
    """
    
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.D_EMBED_DIM = CONFIG['D_EMBED_DIM']
    
    def get_pooled_embedding(self, text_block: str) -> np.ndarray:
        # For simulation, we return a random normalized vector based on the text hash
        # This ensures that encoding the same text always yields the same "embedding"
        random.seed(hash(text_block) % (2**32))
        E_raw = np.random.rand(self.D_EMBED_DIM).astype(np.float32)
        
        # Normalize
        norm_E = np.linalg.norm(E_raw)
        return E_raw / norm_E if norm_E > 1e-6 else E_raw

    def get_random_context(self) -> str:
        return random.choice(self.corpus)

# --- 3. DATA GENERATION LOGIC ---
T_HISTORY_LENGTH = 100       # Number of updates (K_C's) to accumulate per training sample
# NOTE: For quick debugging, reduced N_SAMPLES default. Override in trainer.
N_SAMPLES = 1000            

def generate_training_data(embedding_provider: AbstractEmbeddingProvider, num_samples: int = N_SAMPLES) -> List[dict]:
    """
    Runs the ParityCache simulation to generate the required training dataset
    for the F_decon deconvolution network.
    """
    
    training_data = []
    
    # Initialize the core components once
    encoder.initialize_projection_matrix()
    
    print(f"\n--- Starting ParityCache Data Generation ({num_samples} samples) ---")

    for sample_idx in range(num_samples):
        
        # 1. Reset State for a New Sample (New V_S History)
        storage.initialize_state()
        history = [] # Stores (K_C, E_true, text_block) for ground truth tracking
        
        # 2. Context Accumulation Simulation (T_HISTORY_LENGTH updates)
        for _ in range(T_HISTORY_LENGTH):
            # a. Get Text and True Embedding
            text_block = embedding_provider.get_random_context()
            E_true = embedding_provider.get_pooled_embedding(text_block)
            
            # b. Encode to Key
            K_C = encoder.encode_to_key(E_true)
            
            # c. Update State Vector and track history
            storage.update_state(K_C)
            history.append({'K_C': K_C, 'E_true': E_true, 'text': text_block})

        # 3. Select Ground Truth via Weighted Selection (James's heuristic)
        # We start looking after index 10 to allow V_S to warm up
        candidate_history = history[10:]
        if not candidate_history:
            continue

        # Calculate selection weights based on current Age (A) and Frequency (F)
        selection_weights = []
        for record in candidate_history:
            # Find which bits in K_C are set to 1
            bit_positions = np.where(record['K_C'] == 1)[0]
            
            if len(bit_positions) == 0:
                selection_weights.append(0.0)
                continue

            # Get average Age and Frequency for these bits from the CURRENT storage state
            avg_age = np.mean(storage.A[bit_positions])
            avg_freq = np.mean(storage.F[bit_positions])
            
            # Weight formula: Prefer Low Age (Recent) and High Frequency (Important)
            # Add epsilon to age to avoid division by zero
            weight = (1.0 / (avg_age + 1.0)) * (avg_freq + 1.0)
            selection_weights.append(weight)

        # Normalize weights to probabilities
        selection_weights = np.array(selection_weights)
        total_weight = np.sum(selection_weights)
        
        if total_weight > 0:
            selection_probs = selection_weights / total_weight
            target_idx = np.random.choice(len(candidate_history), p=selection_probs)
            target_record = candidate_history[target_idx]
        else:
            target_record = random.choice(candidate_history)

        K_Q = target_record['K_C']
        E_true = target_record['E_true']

        # a. Compute Difference Key (D) - Retrieval Step
        D = retrieval.compute_difference_key(K_Q)

        # b. Extract Subsampled Inputs (D, A, F)
        SUBSAMPLE_RATE = 8 
        D_subsampled = D[::SUBSAMPLE_RATE]
        A_subsampled = storage.A[::SUBSAMPLE_RATE]
        F_subsampled = storage.F[::SUBSAMPLE_RATE]
        
        # c. Concatenate inputs for the F_decon network
        F_decon_input = np.concatenate([D_subsampled, A_subsampled, F_subsampled])

        # d. Save the training record
        training_data.append({
            'input_vector': F_decon_input,
            'target_embedding': E_true,
            'source_key_text': target_record['text']
        })
        
        if (sample_idx + 1) % 100 == 0:
            print(f"Generated {sample_idx + 1}/{num_samples} samples.")

    print("--- Data Generation Complete ---")
    return training_data

# Simple corpus for testing
sample_corpus = [
    "The ParityCache project aims to achieve O(1) context retrieval using XOR accumulation.",
    "XOR is a non-linear operation that destroys superposition, making deconvolution difficult.",
    "The sparse random matrix P reduces computational cost by two orders of magnitude.",
    "A Context-Aware Decay Mask (M) is required to prevent saturation of the State Vector V_S.",
    "The final deconvolution network F_decon is a small network trained on D, A, and F statistics.",
    "We are currently generating synthetic data to train the F_decon model using a hybrid loss function.",
    "Claude provided architectural critique, and James is steering the research direction.",
    "The State Vector V_S is 4096 bits, while the Context Key K_C is 2048 bits.",
]