# dataset.py
import os, random, numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import config
from augmentations import freq_mask_np, time_mask_np, add_noise_np, time_shift_np

class MelDataset(Dataset):
    def __init__(self, processed_dir=config.PROCESSED_DIR, classes=None, train=True, augment=True, include_noise=True):
        self.root = Path(processed_dir)
        self.train = train
        self.augment = augment and train
        self.samples = []
        self.labels = []
        # build classes from folders
        folders = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        if classes is None:
            self.classes = folders
        else:
            self.classes = classes
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        # collect samples only for chosen classes
        for c in self.classes:
            folder = self.root / c
            if not folder.exists(): continue
            for f in folder.glob("*.npy"):
                self.samples.append(str(f))
                self.labels.append(self.class_to_idx[c])
        # reproducible shuffle
        combined = list(zip(self.samples, self.labels))
        random.Random(config.RANDOM_SEED).shuffle(combined)
        if combined:
            self.samples, self.labels = zip(*combined)
        else:
            self.samples, self.labels = [], []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mel = np.load(self.samples[idx])  # (n_mels, time, 2)
        if self.augment:
            if random.random() < 0.4:
                mel = add_noise_np(mel, std=0.01)
            if random.random() < 0.4:
                mel = time_shift_np(mel, max_shift=30)
            if random.random() < 0.5:
                mel = freq_mask_np(mel, F=24, num_masks=2)
            if random.random() < 0.5:
                mel = time_mask_np(mel, T=50, num_masks=2)
        mel = torch.tensor(mel, dtype=torch.float32).permute(2,0,1)  # (2, n_mels, time)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel, label
