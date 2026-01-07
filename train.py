import datetime
from tqdm import tqdm
import os
import torch as tr
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split   
from src.dataset import SeqDataset, pad_batch
from src.diffusion import DiffusionModel
from src.layers.simpleunet import SimpleUNet

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# --- HYPERPARAMETERS ---
BATCH_SIZE = 2       
LR = 1e-3           
EPOCHS = 5         
TIMESTEPS = 25     # diffusion steps
LAMBDA_VLB = 0.1   # loss VLB coefficent
DEVICE = tr.device("cuda" if tr.cuda.is_available() else "cpu")

# --- LOG AND DATASETS ---
LOG_PATH = "logs/" + timestamp
os.makedirs(LOG_PATH, exist_ok=True) 


DATA_PATH  = "data/ArchiveII_128_random_split/"
TRAIN_PATH = DATA_PATH +  "train.csv"
VAL_PATH   = DATA_PATH + "val.csv"

train_dataset = SeqDataset(TRAIN_PATH)
val_dataset   = SeqDataset(VAL_PATH) 
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")


# --- DATALOADERS ---
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=pad_batch, 
    num_workers=2
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    collate_fn=pad_batch,
    num_workers=2
)

# --- MODEL ---
unet = SimpleUNet(in_channels=18, out_channels=2, base_dim=64)
model = DiffusionModel(
    num_classes=2, 
    embedding_dim=64, 
    time_steps=TIMESTEPS, 
    model=lambda **kwargs: unet
)
model.to(DEVICE)

optimizer = tr.optim.Adam(model.parameters(), lr=LR)
tr.cuda.empty_cache()

# --- HISTORY ---
history = {'train_loss': [], 'val_loss': []}
best_val_loss = float('inf')

print("START TRAINING...")

for epoch in range(EPOCHS):
    # ==========================
    # 1. TRAINING
    # ==========================
    model.train()
    train_loss_accum = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=True)
    for batch in pbar:
        optimizer.zero_grad()
        
        # LOAD DATA
        x0_idx = batch["contact"].to(DEVICE)       # -1 en padding
        x0_oh = batch["contact_oh"].to(DEVICE)     # 0.0 en padding
        condition = batch["outer"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)            # 1.0 valido, 0.0 padding
        
        # SAMPLE t
        t = tr.randint(0, TIMESTEPS, (x0_idx.shape[0],), device=DEVICE).long()
        
        # NOISE - Forward 
        xt_noisy = model.q_sample(x0_oh, t)
        
        # clean noise on padding
        xt_noisy = xt_noisy * mask.squeeze(1).long()
        
        # predict x0
        logits_pred = model.predict_start(xt_noisy, t, condition, return_logits=True)
        
        # Loss Simple + VLB
        loss_simple = F.cross_entropy(logits_pred, x0_idx, ignore_index=-1)
        loss_vlb = model.compute_vlb(x0_oh, xt_noisy, t, condition, mask=mask)
        loss = loss_simple + (LAMBDA_VLB * loss_vlb)
        
        loss.backward()
        optimizer.step()
        
        train_loss_accum += loss.item()

    avg_train_loss = train_loss_accum / len(train_loader)
    
    # ==========================
    # 2. VALIDATION
    # ==========================
    model.eval() 
    val_loss_accum = 0
    
    with tr.no_grad(): 
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val  ]", leave=True)
        for batch in val_pbar:
            x0_idx = batch["contact"].to(DEVICE)
            x0_oh = batch["contact_oh"].to(DEVICE)
            condition = batch["outer"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            
            # Same noise to validate the loss
            t = tr.randint(0, TIMESTEPS, (x0_idx.shape[0],), device=DEVICE).long()
            xt_noisy = model.q_sample(x0_oh, t)
            xt_noisy = xt_noisy * mask.squeeze(1).long()
            
            logits_pred = model.predict_start(xt_noisy, t, condition, return_logits=True)
            
            loss_simple = F.cross_entropy(logits_pred, x0_idx, ignore_index=-1)
            loss_vlb = model.compute_vlb(x0_oh, xt_noisy, t, condition, mask=mask)
            
            val_loss = loss_simple + (LAMBDA_VLB * loss_vlb)
            val_loss_accum += val_loss.item()
            
    avg_val_loss = val_loss_accum / len(val_loader)
    
    # ==========================
    # 3. LOGGING AND CHECKPOINT
    # ==========================
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        tr.save(model.state_dict(), LOG_PATH + "/best_model.pt")
        print(f"   --> SAVING BEST MODEL (Val Loss: {avg_val_loss:.4f}) \t epoch: {epoch}")