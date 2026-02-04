import os 
import torch as tr  
from torch.utils.data import DataLoader
from tqdm import tqdm   

from src.dataset import SeqDataset, pad_batch 
from src.utils import load_config, load_model

def generate_raw_ensemble(model, loader, save_dir, num_samples=50, base_seed=42):
    """
    Generates N samples per sequence and saves the RAW contact maps (N) individually with the seed.    
    Args:
        save_dir: folder filled with .pt (1 per seq)
        base_seed: initial seed. Samples use base_seed, base_seed+1, ...
    """
    model.eval()
    
    device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Generating Raw Ensembles (N={num_samples}) en {save_dir}...")

    for batch in tqdm(loader, desc=f"Generating in directory: {save_dir}"):
        B = batch["outer"].shape[0]
        
        for i in range(B):  
            seq_id = batch["id"][i]  
            path = os.path.join(save_dir, f"{seq_id}.pt")
            if os.path.exists(path): continue

            cond = batch["outer"][i:i+1].to(device) 
            sample_list = []
            used_seeds = []
             
            for k in range(num_samples):
                current_seed = base_seed + k
                tr.manual_seed(current_seed)  
                
                with tr.no_grad(): 
                    s = model._sample(cond).squeeze().cpu().to(tr.int8)
                sample_list.append(s)
                used_seeds.append(current_seed)
            
            data = {
                "samples": tr.stack(sample_list), 
                "seeds": used_seeds,              
                "target": batch["contact_oh"][i].cpu().to(tr.int8), 
                "length": batch["length"][i]
            }
            tr.save(data, path)

    print("Generation finished.")
    
###################################################################
#######################      MAIN       ###########################

BATCH_SIZE = 8 
NUM_SAMPLES = 50

def main():
    BASE_LOG = "logs/ArchiveII_simfold_128/"
    BASE_DATA_PATH = "data/simfolds/simfolds_max128/joined/"
    for sim in ["sim60", "sim70", "sim80", "sim90"]:
        LOGS_DIR = os.path.join(BASE_LOG, sim)  
        TEST_DATA_PATH = os.path.join(BASE_DATA_PATH, sim, "test.csv")
    
        print(f"--- STARTING {sim} EVALUATION ON TEST: {TEST_DATA_PATH} |LOG| {LOGS_DIR}---")
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
        experiments.sort()

        for exp_name in experiments:
            exp_path = os.path.join(LOGS_DIR, exp_name)
            checkpoint_path = os.path.join(exp_path, "best_model.pt") 
            save_folder = os.path.join(exp_path, "raw_samples")
            try:
                config = load_config(exp_path)
                model = load_model(config=config, eval=True, checkpoint_path=checkpoint_path)
                generate_raw_ensemble(model, test_loader,  save_folder, num_samples=NUM_SAMPLES, base_seed=42)
                    
            except Exception as e:
                print(f"Error sampling {exp_name}: {e}")
            
if __name__ == "__main__":
    main()