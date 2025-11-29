## trainer.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple
import numpy as np

# Import the core ParityCache modules
from config import CONFIG, ENCODER_PARAMS
from model import F_decon, HybridLoss, D_EMBED_DIM, N_INPUT_DIM
# Updated Import:
from data_generator import generate_training_data, AbstractEmbeddingProvider, SimulatedEmbeddingProvider, sample_corpus

# --- TRAINING CONFIGURATION ---
TRAINING_CONFIG = {
    'BATCH_SIZE': 32, # Reduced for small prototype runs
    'LEARNING_RATE': 1e-4,
    'NUM_EPOCHS': 5, 
    'SUBSAMPLE_RATE': 8,
    'VALIDATION_SPLIT': 0.1,
}

# ------------------------------------------------
# I. Data Preparation (Uses data_generator)
# ------------------------------------------------
def prepare_data(provider: AbstractEmbeddingProvider, num_samples: int):
    """
    Generates data using the ParityCache simulator and converts it to PyTorch Tensors.
    """
    print(f"\n--- Preparing Data for Training (Target: {num_samples} samples) ---")
    raw_data = generate_training_data(provider, num_samples=num_samples)

    if not raw_data:
        raise ValueError("No training data generated. Check corpus or history length.")

    # 1. Separate components
    input_vectors = [d['input_vector'] for d in raw_data]
    E_true_targets = [d['target_embedding'] for d in raw_data]
    
    # 2. Simulate Hard Negative Sampling (Prototype Only)
    # In prototype, we pick a random embedding from the batch as the negative
    all_embeddings = np.array(E_true_targets)
    # Random indices for negatives
    neg_indices = np.random.choice(len(all_embeddings), size=len(all_embeddings), replace=True)
    E_neg_targets = all_embeddings[neg_indices]

    # 3. Convert to Tensors
    X = torch.tensor(np.array(input_vectors), dtype=torch.float32)
    Y_true = torch.tensor(np.array(E_true_targets), dtype=torch.float32)
    Y_neg = torch.tensor(np.array(E_neg_targets), dtype=torch.float32)

    # 4. Create Dataset and Split
    dataset = TensorDataset(X, Y_true, Y_neg)
    
    val_size = int(TRAINING_CONFIG['VALIDATION_SPLIT'] * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG['BATCH_SIZE'], shuffle=False)
    
    print(f"Dataset prepared: {train_size} training samples, {val_size} validation samples.")
    return train_loader, val_loader


# ------------------------------------------------
# II. Training Loop
# ------------------------------------------------
def train_model(train_loader, val_loader):
    """
    Runs the PyTorch training process for the F_decon network.
    """
    # Initialize Model, Loss Function, and Optimizer
    model = F_decon()
    criterion = HybridLoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG['LEARNING_RATE'])
    
    # Use CUDA if available, else MPS (Mac) or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): # Support for Mac Studio (M1/M2)
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using Device: {device}")
    model.to(device)

    print(f"\n--- Starting Training ({TRAINING_CONFIG['NUM_EPOCHS']} epochs) ---")

    for epoch in range(TRAINING_CONFIG['NUM_EPOCHS']):
        model.train()
        total_train_loss = 0
        
        for batch_idx, (X_batch, Y_true_batch, Y_neg_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            Y_true_batch = Y_true_batch.to(device)
            Y_neg_batch = Y_neg_batch.to(device)

            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            E_retrieved_raw = model(X_batch)
            
            # Calculate Hybrid Loss
            loss, L_cosine, L_triplet = criterion(E_retrieved_raw, Y_true_batch, Y_neg_batch)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        # Validation Step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val, Y_true_val, Y_neg_val in val_loader:
                X_val = X_val.to(device)
                Y_true_val = Y_true_val.to(device)
                Y_neg_val = Y_neg_val.to(device)
                
                E_retrieved_raw_val = model(X_val)
                val_loss, _, _ = criterion(E_retrieved_raw_val, Y_true_val, Y_neg_val)
                total_val_loss += val_loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{TRAINING_CONFIG['NUM_EPOCHS']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    print("\n--- Training Complete ---")
    return model


# In trainer.py (Main Execution block)

if __name__ == '__main__':
    from ollama_provider import OllamaProvider
    
    # 1. Load a Real Corpus (e.g., a text file)
    # For a quick test, you can just use a list of 10-20 long sentences
    print("Loading corpus...")
    # real_corpus = open("my_corpus.txt", "r").readlines() 
    
    # Or use a slightly larger sample list for now:
    real_corpus = [
        "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn'.",
        "The transformer is a deep learning architecture that relies on the parallel multi-head attention mechanism.",
        "High-performance computing relies on parallel processing and specialized hardware like GPUs and FPGAs.",
        "The mitochondria is the powerhouse of the cell, generating most of the cell's supply of adenosine triphosphate.",
        # ... add 50-100 real sentences ...
    ]

    # 2. Instantiate Ollama Provider
    # Make sure you ran: `ollama pull nomic-embed-text`
    provider_instance = OllamaProvider(corpus=real_corpus, model_name="nomic-embed-text")
    
    # 3. Data Generation (Reduced sample count because real embeddings take time!)
    # 500 samples * 100 history = 50,000 calls. 
    # WARNING: This will take ~30-45 minutes on local hardware. 
    # Set history to 10 or 20 for a quick sanity check first!
    NUM_REAL_SAMPLES = 50 
    
    # Update global config T_HISTORY_LENGTH in data_generator.py if possible, 
    # or just accept the wait.
    
    train_loader, val_loader = prepare_data(provider_instance, num_samples=NUM_REAL_SAMPLES)
    trained_model = train_model(train_loader, val_loader)