import os
import torch as tr
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- PROJECT IMPORTS ---
###################################################
from src.utils import load_config, load_model
from src.dataset import SeqDataset, pad_batch 
from src.metrics import contact_f1
###################################################

# --- CONFIGURATION ---
###################################################
LOGS_DIR = "logs/"
TEST_DATA_PATH = "data/ArchiveII_128_random_split/test.csv"
DEVICE = tr.device("cuda" if tr.cuda.is_available() else "cpu")
BATCH_SIZE = 8 

@tr.no_grad()
def evaluate_model(model, loader, device):
    """
    Evaluates a single model on the test set.
    Returns detailed lists and mean metrics.
    """
    model.eval()
    
    results = {
        "loss": [],
        "f1": [] 
    }
    
    for batch in tqdm(loader, desc="Testing", leave=False):
        cond = batch["outer"].to(device)
        target = batch["contact_oh"].to(device)
        mask = batch["mask"].to(device)
        lens = batch["length"]
        
        # 1. Calculate Test Loss (Consistency check)
        loss = model.forward_all_timesteps(target, cond, mask=mask)
        results["loss"].append(loss.item())
        
        # 2. Sampling and F1
        samples = model._sample(cond)
        f1_scores = contact_f1(samples, target, lengths=lens, reduce=False)
        
        # Store results
        results["f1"].extend(f1_scores.cpu().tolist())

    return results

def main():
    print(f"--- STARTING EVALUATION ON: {TEST_DATA_PATH} ---")
    
    # 1. Load Test Data
    test_ds = SeqDataset(TEST_DATA_PATH)
    test_loader = DataLoader(
        test_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=pad_batch, 
        num_workers=2
    )
    print(f"Test Dataset Size: {len(test_ds)}")
 
    experiments = [f for f in os.listdir(LOGS_DIR) if os.path.isdir(os.path.join(LOGS_DIR, f))]
    experiments.sort() # Ensure consistent order
    
    summary_table = []

    for exp_name in experiments:
        exp_path = os.path.join(LOGS_DIR, exp_name)
        checkpoint_path = os.path.join(exp_path, "best_model.pt")
        
        # Skip if no checkpoint found  
        if not os.path.exists(checkpoint_path):
            continue
            
        print(f"\nProcessing: {exp_name}")
        
        try:
            config = load_config(exp_path)
            model = load_model(config=config, eval=True, checkpoint_path=checkpoint_path)            
            
            results = evaluate_model(model, test_loader, DEVICE)
            
            mean_f1 = np.mean(results["f1"])
            std_f1 = np.std(results["f1"])
            mean_loss = np.mean(results["loss"])
            
            summary_table.append({
                "experiment_id": exp_name,
                "timesteps": config.get("timesteps"),
                "epochs": config.get("epochs"), 
                "test_loss": mean_loss,
                "test_f1": mean_f1 
            })
            print(f" -> Test F1: {mean_f1:.4f} (+/- {std_f1:.4f})")
            
        except Exception as e:
            print(f"Error evaluating {exp_name}: {e}")

    if summary_table:
        df_summary = pd.DataFrame(summary_table)
        # Sort by F1 Score descending
        df_summary = df_summary.sort_values(by="test_f1", ascending=False)

        out_path = LOGS_DIR + "all_test_summary.csv"
        df_summary.to_csv(out_path, index=False)
        print(f"\n{'='*50}")
        print(f"Evaluation Complete. Summary saved to: {out_path}")
        print(f"{'='*50}")
        print(df_summary.to_string())
    else:
        print("No valid experiments found to evaluate.")

if __name__ == "__main__":
    main()