## ollama_provider.py

import ollama
import numpy as np
import random
from typing import List, Any
import time

# Import the Abstract Interface
from data_generator import AbstractEmbeddingProvider
from config import CONFIG

class OllamaProvider(AbstractEmbeddingProvider):
    """
    Concrete implementation of EmbeddingProvider using local Ollama instance.
    Recommended Model: 'nomic-embed-text' (768d)
    """
    
    def __init__(self, corpus: List[str], model_name: str = "nomic-embed-text"):
        """
        source: A list of strings (the text corpus).
        model_name: The Ollama model tag to use.
        """
        self.corpus = corpus
        self.model_name = model_name
        self.D_EMBED_DIM = CONFIG['D_EMBED_DIM']
        
        # Test connection and dimension on init
        print(f"--- Initializing OllamaProvider with model: {self.model_name} ---")
        try:
            test_resp = ollama.embeddings(model=self.model_name, prompt="Hello world")
            dim = len(test_resp['embedding'])
            print(f"Connection successful. Embedding Dimension: {dim}")
            
            if dim != self.D_EMBED_DIM:
                raise ValueError(f"Model dimension ({dim}) does not match CONFIG ({self.D_EMBED_DIM})")
                
        except Exception as e:
            print(f"ERROR: Could not connect to Ollama. Is it running? Error: {e}")
            raise e

    def get_pooled_embedding(self, text_block: str) -> np.ndarray:
        """
        Fetches the embedding from Ollama.
        Note: nomic-embed-text handles pooling internally (CLS/Mean).
        """
        # Call Ollama API
        response = ollama.embeddings(model=self.model_name, prompt=text_block)
        
        # Extract vector
        E_raw = np.array(response['embedding'], dtype=np.float32)
        
        # Enforce Normalization (Crucial for SimHash & Cosine Similarity)
        norm_E = np.linalg.norm(E_raw)
        if norm_E > 1e-6:
            E_normalized = E_raw / norm_E
        else:
            E_normalized = E_raw # Should not happen with valid text
            
        return E_normalized

    def get_random_context(self) -> str:
        """Returns a random text block from the loaded corpus."""
        return random.choice(self.corpus)

    def get_batch_embeddings(self, text_blocks: List[str]) -> np.ndarray:
        """
        Optimized batch fetching if needed later.
        Currently iterates, but can be updated for batch API if available.
        """
        embeddings = []
        for text in text_blocks:
            embeddings.append(self.get_pooled_embedding(text))
        return np.array(embeddings)