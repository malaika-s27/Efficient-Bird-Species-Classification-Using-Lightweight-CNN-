# train_stage1.py
import os, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from dataset import MelDataset
from model import build_efficientnet_b0
import config
from pathlib import Path

DEVICE = torch.device("cpu")

# Building dataset but only two classes: 'noise' and 'bird' (merging all non-noise into 'bird')
processed = Path(config.PROCESSED_DIR)
classes = sorted([d.name for d in processed.iterdir() if d.is_dir()])
noise_label = config.NOISE_LABEL
bird_classes = [c for c in classes if c != noise_label]
# Create temporary dataset listing: we'll map bird_classes -> "bird"
# Instead of changing files, we create a dataset loader wrapper
class Stage1Dataset(torch.utils.data.Dataset):
    def __init__(self, processed_dir, bird_classes, noise_class, train=True):
        self.root = Path(processed_dir)
        self.samples = []
        self.labels = []
        for c in bird_classes:
            for f in (self.root/c).glob("*.npy"):
                self.samples.append(str(f))
                self.labels.append(1)  # bird
        for f in (self.root/noise_class).glob("*.npy"):
            self.samples.append(str(f))
            self.labels.append(0)  # noise
        # shuffle
        perm = np.arange(len(self.samples))
        np.random.seed(config.RANDOM_SEED)
        np.random.shuffle(perm)
        self.samples = [self.samples[i] for i in perm]
        self.labels = [self.labels[i] for i in perm]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        mel = np.load(self.samples[idx])
        # simple augmentations
        if np.random.rand() < 0.4:
            mel = mel + 0.01 * np.random.randn(*mel.shape)
        if np.random.rand() < 0.4:
            # time shift
            shift = np.random.randint(-20,20)
            mel = np.roll(mel, shift, axis=1)
        x = torch.tensor(mel, dtype=torch.float32).permute(2,0,1)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# dataset
full_ds = Stage1Dataset(config.PROCESSED_DIR, bird_classes, noise_label)
n = len(full_ds)
train_idx, val_idx = train_test_split(list(range(n)), test_size=config.VAL_SPLIT, random_state=config.RANDOM_SEED,
                                     stratify=full_ds.labels)
train_ds = Subset(full_ds, train_idx)
val_ds = Subset(full_ds, val_idx)
train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

# model: binary
model = build_efficientnet_b0(num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

best_val = 0; patience=0
Path(config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
for epoch in range(1, config.EPOCHS+1):
    model.train()
    tloss=0; tcnt=0; tcorr=0
    for x,y in train_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward(); optimizer.step()
        tloss += loss.item()*x.size(0); tcnt += x.size(0)
        tcorr += (out.argmax(1)==y).sum().item()
    train_loss = tloss/tcnt; train_acc=100*tcorr/tcnt

    # val
    model.eval()
    vloss=0; vcnt=0; vcorr=0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out,y)
            vloss += loss.item()*x.size(0); vcnt += x.size(0)
            vcorr += (out.argmax(1)==y).sum().item()
    val_loss=vloss/vcnt; val_acc=100*vcorr/vcnt

    print(f"Epoch {epoch}/{config.EPOCHS} | Train loss {train_loss:.4f} acc {train_acc:.2f}% | Val loss {val_loss:.4f} acc {val_acc:.2f}%")
    scheduler.step()
    if val_acc > best_val:
        best_val=val_acc
        torch.save(model.state_dict(), config.MODEL_OUT_STAGE1)
        print("Saved stage1:", config.MODEL_OUT_STAGE1)
        patience=0
    else:
        patience+=1
        if patience>=config.PATIENCE:
            print("Early stop stage1")
            break
print("Stage1 done. Best val:", best_val)
