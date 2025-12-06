# train_stage2.py
import os, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import MelDataset
from model import build_efficientnet_b0
import config
from pathlib import Path

DEVICE = torch.device("cpu")

# full dataset (all classes from processed folder)
ds = MelDataset(processed_dir=config.PROCESSED_DIR, train=True, augment=True)
num_classes = len(ds.classes)
# stratified split
indices = list(range(len(ds)))
labels = [int(ds.class_to_idx[Path(p).parent.name]) for p in ds.samples] if hasattr(ds, "samples") else [0]*len(ds)
train_idx, val_idx = train_test_split(indices, test_size=config.VAL_SPLIT, random_state=config.RANDOM_SEED, stratify=ds.labels if hasattr(ds,'labels') else None)
from torch.utils.data import Subset
train_ds = Subset(ds, train_idx)
val_ds = Subset(ds, val_idx)
train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

model = build_efficientnet_b0(num_classes=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

best_val=0; patience=0
Path(config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
for epoch in range(1, config.EPOCHS+1):
    model.train()
    tloss=0; tcnt=0; tcorr=0
    for x,y in train_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        # mixup
        if config.MIXUP_ALPHA>0 and np.random.rand()<0.7:
            lam = np.random.beta(config.MIXUP_ALPHA, config.MIXUP_ALPHA)
            idx = torch.randperm(x.size(0))
            x = lam*x + (1-lam)*x[idx]
            y_a, y_b = y, y[idx]
            out = model(x)
            loss = lam*criterion(out,y_a)+(1-lam)*criterion(out,y_b)
        else:
            out = model(x)
            loss = criterion(out,y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        tloss += loss.item()*x.size(0); tcnt += x.size(0)
        preds = out.argmax(1)
        tcorr += (preds==y).sum().item()
    train_loss=tloss/tcnt; train_acc=100*tcorr/tcnt

    # validation
    model.eval()
    vloss=0; vcnt=0; vcorr=0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out,y)
            vloss += loss.item()*x.size(0); vcnt += x.size(0)
            vcorr += (out.argmax(1)==y).sum().item()
    val_loss = vloss/vcnt; val_acc=100*vcorr/vcnt

    print(f"Epoch {epoch}/{config.EPOCHS} | Train loss {train_loss:.4f} acc {train_acc:.2f}% | Val loss {val_loss:.4f} acc {val_acc:.2f}%")
    scheduler.step()
    if val_acc>best_val:
        best_val=val_acc
        torch.save(model.state_dict(), config.MODEL_OUT_STAGE2)
        print("Saved stage2:", config.MODEL_OUT_STAGE2)
        patience=0
    else:
        patience+=1
        if patience>=config.PATIENCE:
            print("Early stop stage2"); break
# save class map
np.save(config.CLASS_MAP, np.array(ds.classes))
print("Stage2 done. Best:", best_val)
