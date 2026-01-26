import os
import sys 
import torch as tr
import numpy as np
import pandas as pd
import random 
from tqdm import tqdm
sys.path.append("../")

import torch as tr
import numpy as np
from sklearn.metrics import f1_score

class SeqEnsemble:
    def __init__(self, data_path):
        """Loads the raw ensemble results of a model."""
        self.data = tr.load(data_path)
        self.samples = self.data["samples"].float() # [N, L, L] 
        self.seeds = self.data["seeds"]
        self.target = self.data["target"]
        self.length = self.data["length"]
        self.num_avail_samples = len(self.samples)
        self.path = data_path 
        if self.target.ndim == 3: self.target = self.target.argmax(dim=0)
        self.target_np = self.target.numpy()

    def evaluate_consensus(self, indices=None):
        """
        Calculates the consensus using ONLY the specified indices.
        If indices is None, uses all.
        """
        if indices is None:
            subset = self.samples
        else:
            subset = self.samples[indices]
        
        prob_map = subset.mean(dim=0) # n_consensus avg
        consensus = (prob_map > 0.5).numpy().astype(int)   # the strategy is 0.5     
        return consensus    

    def get_uncertainty_map(self, indices=None):
        """Return the std dev ."""
        subset = self.samples if indices is None else self.samples[indices]
        return subset.std(dim=0).numpy()
    
    def _compute_f1(self, pred_matrix):
        L = int(self.length)
        rows, cols = np.triu_indices(L, k=1)
        p_flat = pred_matrix[:L, :L][rows, cols]
        r_flat = self.target_np[:L, :L][rows, cols]
        return f1_score(r_flat, p_flat, zero_division=0)

    def evaluate_single_seed(self, seed_idx):
        """Evalúa solo la muestra correspondiente a un índice (0 a N-1)."""
        pred = self.samples[seed_idx].numpy()
        return self._compute_f1(pred)
    
    def evaluate_consensus_f1(self, indices=None): 
        consensus = self.evaluate_consensus(indices)
        return self._compute_f1(consensus)
        

def evaluate_rna_k_consensus(rna, chosen_seeds):
    row = {"seq_id": rna.path.split("/")[-1].replace(".pt","")} 
    for k in N_CONSENSUS:
        f1_list = []
        for idx in chosen_seeds[k]: 
            f1_list.append(rna.evaluate_consensus_f1(indices=idx))
        
        row[f"cons_k{k}_mean"] = np.mean(f1_list)
        row[f"cons_k{k}_std"]  = np.std(f1_list)
    return row

###################################################################
#######################      MAIN       ###########################

LOGS_PATH = "logs/"
TRIALS = 20
N_CONSENSUS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 14, 15, 20, 25]    

def main():
    ALL_EXPS = [
        exp for exp in os.listdir(LOGS_PATH) 
        if not exp.endswith(".csv")
                ]
    for exp_dir in ALL_EXPS:
        
        exp_dir = LOGS_PATH + exp_dir + "/"
        samples = exp_dir + "raw_samples"
        print(f"Executing {exp_dir}")
        
        paths = [os.path.join(samples, f) 
                for f in os.listdir(samples) if f.endswith(".pt")]
        paths.sort()
        
        test_ensemble = [SeqEnsemble(p) for p in tqdm(paths, desc="Loading")] 
        max_samples = min([ens.num_avail_samples for ens in test_ensemble])

        ### Choose the #Trial seeds to use for the each N_consensus
        chosen_seeds = {}
        for k in N_CONSENSUS:
            chosen_seeds[k] = [
                random.sample(range(max_samples), k)
                for _ in range(TRIALS)
            ]
                
        all_stats = []
        for rna_samples in tqdm(test_ensemble, desc="Evaluating"):
            row = evaluate_rna_k_consensus(rna_samples, chosen_seeds)
            all_stats.append(row)

        df_stats = pd.DataFrame(all_stats) 
        df_stats.to_csv(exp_dir + f"enemble_stats_{TRIALS}_trials.csv")     
        

if __name__ == "__main__":
    main()