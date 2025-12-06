import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from model import build_efficientnet_b0
import config
from dataset import MelDataset
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Full dataset (all classes from processed folder)
ds = MelDataset(processed_dir=config.PROCESSED_DIR, train=True, augment=True)
num_classes = len(ds.classes)

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

# Initialize variables for tracking the overall performance
fold_accuracies = []
fold_train_losses = []
fold_val_losses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(ds)))):

    print(f"Fold {fold+1}/{kf.get_n_splits()}")

    # Create training and validation datasets
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # Model setup
    model = build_efficientnet_b0(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    best_val_acc = 0
    patience = 0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        tloss = 0
        tcnt = 0
        tcorr = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Mixup augmentation
            if config.MIXUP_ALPHA > 0 and np.random.rand() < 0.7:
                lam = np.random.beta(config.MIXUP_ALPHA, config.MIXUP_ALPHA)
                idx = torch.randperm(x.size(0))
                x = lam * x + (1 - lam) * x[idx]
                y_a, y_b = y, y[idx]
                out = model(x)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            else:
                out = model(x)
                loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tloss += loss.item() * x.size(0)
            tcnt += x.size(0)
            preds = out.argmax(1)
            tcorr += (preds == y).sum().item()

        train_loss = tloss / tcnt
        train_acc = 100 * tcorr / tcnt

        # Validation phase
        model.eval()
        vloss = 0
        vcnt = 0
        vcorr = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                vloss += loss.item() * x.size(0)
                vcnt += x.size(0)
                vcorr += (out.argmax(1) == y).sum().item()

        val_loss = vloss / vcnt
        val_acc = 100 * vcorr / vcnt

        print(f"Epoch {epoch}/{config.EPOCHS} | Train loss {train_loss:.4f} acc {train_acc:.2f}% | "
              f"Val loss {val_loss:.4f} acc {val_acc:.2f}%")

        scheduler.step()

        # Save the best model for each fold
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{config.MODEL_DIR}/model_fold{fold+1}.pt")
            print(f"Saved best model for fold {fold+1} at epoch {epoch}")
            patience = 0  # Reset patience

        # Early stopping
        patience += 1
        if patience >= config.PATIENCE:
            print(f"Early stop for fold {fold+1}")
            break

    fold_accuracies.append(best_val_acc)
    fold_train_losses.append(train_loss)
    fold_val_losses.append(val_loss)
    print(f"Best Validation Accuracy for fold {fold+1}: {best_val_acc:.2f}%\n")

# Average accuracy over all folds
avg_accuracy = np.mean(fold_accuracies)
avg_train_loss = np.mean(fold_train_losses)
avg_val_loss = np.mean(fold_val_losses)
print(f"Average Validation Accuracy: {avg_accuracy:.2f}%")
print(f"Average Train Loss: {avg_train_loss:.4f}")
print(f"Average Validation Loss: {avg_val_loss:.4f}")
