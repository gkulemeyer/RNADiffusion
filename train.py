import torch as tr
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
import time

# --- PROJECT IMPORTS ---
from src.dataset import SeqDataset, pad_batch
from src.diffusion import DiffusionModel
from src.layers.simpleunet import SimpleUNet
from src.metrics import contact_f1

# --- UTILITIES ---

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_config(config, path):
    with open(os.path.join(path, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

# --- CORE FUNCTIONS ---

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    epoch_loss = []
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for batch in pbar:
        cond = batch["outer"].to(device)
        target = batch["contact_oh"].to(device)
        mask = batch["mask"].to(device)       
        
        optimizer.zero_grad()
        
        # Forward pass (diffusion loss)
        loss = model.forward_all_timesteps(target, cond, mask=mask)
             
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_loss.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return np.mean(epoch_loss)

@tr.no_grad()
def validate(model, loader, device):
    """
    Computes Validation Loss and F1 Score (via sampling).
    """
    model.eval()
    val_loss = []
    val_f1 = []
    
    for batch in tqdm(loader, desc="Validating", leave=False):
        cond = batch["outer"].to(device)
        target = batch["contact_oh"].to(device)
        mask = batch["mask"].to(device)       
        lens = batch["length"]
        
        # 1. Validation Loss (No sampling)
        loss = model.forward_all_timesteps(target, cond, mask=mask)
        val_loss.append(loss.item())
        
        # 2. Validation F1 (Sampling required)
        samples = model._sample(cond)
        f1_score = contact_f1(samples, target, lengths=lens, reduce=True)
        val_f1.append(f1_score)
        
    return np.mean(val_loss), np.mean(val_f1)

# --- EXPERIMENT RUNNER ---

def run_experiment(config):
    """
    Runs a full training session based on the provided configuration dictionary.
    """
    # 1. Directory Setup
    timestamp = get_timestamp()
    exp_name = f"exp_T{config['timesteps']}_E{config['epochs']}_{timestamp}"
    log_dir = os.path.join("logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    save_config(config, log_dir)
    print(f"\n{'='*40}")
    print(f"STARTING EXPERIMENT: {exp_name}")
    print(f"Config: {config}")
    print(f"{'='*40}")

    device = tr.device("cuda" if tr.cuda.is_available() else "cpu")

    # 2. Data Loading
    train_ds = SeqDataset(config["train_path"])
    val_ds = SeqDataset(config["val_path"])
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        collate_fn=pad_batch, 
        num_workers=2
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        collate_fn=pad_batch, 
        num_workers=2
    )

    # 3. Model Initialization
    unet = SimpleUNet(in_channels=18, out_channels=2, base_dim=64)
    model = DiffusionModel(
        num_classes=2, 
        embedding_dim=64, 
        time_steps=config["timesteps"], 
        model=lambda **kwargs: unet
    )
    model.to(device)
    
    optimizer = tr.optim.Adam(model.parameters(), lr=config["lr"])
    
    # 4. Training Loop
    metrics = []
    best_val_f1 = -1.0
    
    for epoch in range(1, config["epochs"] + 1):
        start_time = time.perf_counter()
        print(f"\nEpoch {epoch}/{config['epochs']}")
        
        # Train and valid
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        end_time = time.perf_counter()
        
        val_loss, val_f1 = validate(model, val_loader, device)
        
        # Logging
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        
        # Update History
        metrics.append({
            "epoch": epoch,
            "epoch_time": end_time - start_time,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_f1
        })
        
        # Save CSV (updates every epoch)
        pd.DataFrame(metrics).to_csv(os.path.join(log_dir, "metrics.csv"), index=False)
        
        # Checkpointing
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            tr.save(model.state_dict(), os.path.join(log_dir, "best_model.pt"))
            print("Best Model Saved!")
            
        # Save latest state
        tr.save(model.state_dict(), os.path.join(log_dir, "last_model.pt"))

    print(f"Experiment finished. Logs saved to: {log_dir}")
    return log_dir, best_val_f1

def set_experiment_config(base_config, epochs=None, timesteps=None, note=""):
    """
    Merges base configuration with overrides for specific experiments.
    """
    config = base_config.copy()
    config["epochs"] = epochs if epochs is not None else config.get("epochs", 1)
    config["timesteps"] = timesteps if timesteps is not None else config.get("timesteps", 1)
    config["note"] = note
    return config

# --- ABLATION STUDY SETUP ---
if __name__ == "__main__":
    
    BASE_DATA_DIR = "data/ArchiveII_128_random_split"
    
    base_dict = {
        "train_path": f"{BASE_DATA_DIR}/train.csv",
        "val_path": f"{BASE_DATA_DIR}/val.csv",
        "batch_size": 4,
        "lr": 1e-3
        }

    # timestep_options = [5, 10, 50]
    epoch_options = [50]
    
    # experiment_configs1 = [set_experiment_config(base_dict, epochs=10, timesteps=t, note="ablation") 
    #                       for t in timestep_options]
    experiment_configs1 = []
    experiment_configs2 = [set_experiment_config(base_dict, epochs=e, timesteps=50, note="ablation") 
                          for e in epoch_options]

    print(f"Running {len(experiment_configs1 + experiment_configs2)} experiments.")
    
    for i, conf in enumerate(experiment_configs1 + experiment_configs2):
        try:
            run_experiment(conf)
        except Exception as e:
            print(f"ERROR in experiment {i}: {e}")
            continue