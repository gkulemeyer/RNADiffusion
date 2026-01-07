import torch as tr
import numpy as np
import datetime
from src.dataset import SeqDataset, pad_batch
from src.diffusion import DiffusionModel
from src.layers.simpleunet import SimpleUNet

def diagnostic_inspector(data_path, checkpoint_path=None):
    # 1. Preparar el archivo de log
    log_file = "debug_log.txt"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, "w") as f:
        f.write(f"=== REPORTE DE DIAGNÓSTICO DE TENSORES ({timestamp}) ===\n\n")

    def log_and_print(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    # 2. Cargar un Batch real
    dataset = SeqDataset(data_path)
    loader = tr.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=pad_batch)
    batch = next(iter(loader))

    log_and_print("--- ESTRUCTURA DEL BATCH ---")
    for k, v in batch.items():
        if isinstance(v, tr.Tensor):
            log_and_print(f"Tensor: {k:12} | Shape: {str(list(v.shape)):20} | Dtype: {v.dtype}")
        else:
            log_and_print(f"Extra:  {k:12} | Valor: {v}")

    # 3. Análisis profundo de Padding
    contact = batch["contact"] # [B, L, L] con -1 en padding
    mask = batch["mask"]       # [B, 1, L, L] con 0.0 en padding
    

    print("\n--- ANÁLISIS DE PADDING EN EL BATCH ---")
    print(f"Contactos (con -1 en padding):")
    log_and_print(str(contact[0, -10:, -10:])) 
    print(f"\nMáscara (0.0 en padding):")
    log_and_print(str(mask[0, 0, -10:, -10:])) 

    
    # 5. Análisis del Forward Pass (si hay modelo)
    if checkpoint_path:
        log_and_print("\n--- ANÁLISIS DE SALIDA DEL MODELO (FORWARD) ---")
        device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
        
        # Cargar modelo
        unet = SimpleUNet(in_channels=18, out_channels=2, base_dim=64)
        model = DiffusionModel(num_classes=2, embedding_dim=64, time_steps=25, model=lambda **kwargs: unet)
        model.load_state_dict(tr.load(checkpoint_path, map_location=device))
        model.to(device).eval()

        with tr.no_grad():
            # Simular un paso de difusión
            condition = batch["outer"].to(device)
            t = tr.zeros((condition.shape[0],), device=device).long()
            x_noisy = tr.randn((condition.shape[0], 2, condition.shape[2], condition.shape[3])).to(device)
            
            logits = model.predict_start(x_noisy, t, condition, return_logits=True)
            
            # Separar logits de zonas válidas vs padding
            mask_bool = (contact != -1).to(device)
            logits_valid = logits.permute(0, 2, 3, 1)[mask_bool]
            logits_padding = logits.permute(0, 2, 3, 1)[~mask_bool]

            log_and_print(f"Logits en zonas VÁLIDAS:  Media={logits_valid.mean():.4f} | Std={logits_valid.std():.4f}")
            log_and_print(f"Logits en zonas PADDING:  Media={logits_padding.mean():.4f} | Std={logits_padding.std():.4f}")
            
            if logits_padding.abs().mean() > 1e-1:
                log_and_print("AVISO: El modelo está generando valores altos en el padding. ¿Estás aplicando la máscara en el forward?")

    log_and_print(f"\nReporte guardado en: {log_file}")

# --- PARA EJECUTAR ---

if __name__ == "__main__":
    diagnostic_inspector("data/ArchiveII_128_random_split/test.csv", "logs/20260106_213350/best_model.pt")