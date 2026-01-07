import torch as tr
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os

from src.dataset import SeqDataset, pad_batch
from src.diffusion import DiffusionModel
from src.layers.simpleunet import SimpleUNet

# --- CONFIGURACIÓN ---
LOG_PATH = "logs/20260106_213350/"
CHECKPOINT_PATH = LOG_PATH + "best_model.pt" # Cambia esto

DEVICE = tr.device("cuda" if tr.cuda.is_available() else "cpu")

TEST_PATH = "data/ArchiveII_128_random_split/test.csv"
TIMESTEPS = 25
BATCH_SIZE = 2

def load_model(path):
    unet = SimpleUNet(in_channels=18, out_channels=2, base_dim=64)
    model = DiffusionModel(num_classes=2, embedding_dim=64, time_steps=TIMESTEPS, model=lambda **kwargs: unet)
    model.load_state_dict(tr.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

@tr.no_grad()
def run_global_evaluation(model, dataset):
    """Calcula F1, Precision y Recall promediado sobre todo el test set."""
    all_f1 = []
    
    loader = tr.utils.data.DataLoader(dataset, batch_size=1, collate_fn=pad_batch)
    
    print(f"Calculando métricas globales sobre {len(dataset)} secuencias...")
    for batch in tqdm(loader):
        condition = batch["outer"].to(DEVICE)
        target = batch["contact"][0].cpu() # [L, L]
        length = batch["length"][0]
        mask = batch["mask"][0, 0].cpu() # [L, L]
        
        # Generar
        gen_full = model._sample(condition)[0].cpu() # [L, L]
        
        # Recortar padding y aplicar máscara de validez
        y_true = target[:length, :length].numpy().flatten()
        y_pred = gen_full[:length, :length].numpy().flatten()
        
        # Filtrar el padding (-1) para el cálculo de F1
        valid_idx = y_true != -1
        if valid_idx.sum() == 0: continue
        
        f1 = f1_score(y_true[valid_idx], y_pred[valid_idx], zero_division=0)
        all_f1.append(f1)
        
    print(f"\n--- RESULTADOS GLOBALES ---")
    print(f"F1-Score promedio: {np.mean(all_f1):.4f} (+/- {np.std(all_f1):.4f})")
    return all_f1

@tr.no_grad()
def analyze_sequence_stability(model, dataset, idx=None, n_runs=10, LOG_PATH=LOG_PATH):
    """Analiza qué tan consistente es el modelo generando la misma secuencia N veces."""
    if idx is None:
        idx = np.random.randint(len(dataset))
        
    item = dataset[idx]
    batch = pad_batch([item])
    condition = batch["outer"].to(DEVICE)
    target = batch["contact"][0, :batch["length"][0], :batch["length"][0]].to(DEVICE)
    target = tr.where(target == -1, tr.tensor(0, device=DEVICE), target)
    
    generations = []
    print(f"Analizando estabilidad para ID: {batch['id'][0]} ({n_runs} ejecuciones)...")
    
    for _ in range(n_runs):
        gen = model._sample(condition)[0, :batch["length"][0], :batch["length"][0]]
        generations.append(gen)
        
    # Stack y cálculos estadísticos
    stack = tr.stack(generations).float() # [N, L, L]
    mean_map = tr.mean(stack, dim=0)     # Mapa de probabilidad/frecuencia
    std_map = tr.std(stack, dim=0)       # Mapa de incertidumbre
    
    # Visualización
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(target.cpu(), cmap="Greys")
    axes[0].set_title("Ground Truth")
    
    im1 = axes[1].imshow(mean_map.cpu(), cmap="Blues")
    axes[1].set_title(f"Frecuencia (Mean of {n_runs})")
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(std_map.cpu(), cmap="hot")
    axes[2].set_title("(Std Dev)")
    plt.colorbar(im2, ax=axes[2])
    plt.suptitle(f"Análisis de - {batch['id'][0]}")
    plt.savefig(os.path.join(LOG_PATH, f"analysis_{batch['id'][0]}.png"))
    plt.show()

# --- EJECUCIÓN ---
if __name__ == "__main__":
    test_dataset = SeqDataset(TEST_PATH)
    model = load_model(CHECKPOINT_PATH)
    
    # 1. Métricas globales
    f1_list = run_global_evaluation(model, test_dataset)
    
    # 2. Análisis profundo de una secuencia aleatoria
    analyze_sequence_stability(model, test_dataset, n_runs=2)