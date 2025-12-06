# augmentations.py
import numpy as np
import random
import torch

def freq_mask_np(spec, F=24, num_masks=1):
    spec = spec.copy()
    n_mels = spec.shape[0]
    for _ in range(num_masks):
        f = random.randint(0, F)
        if f == 0: continue
        f0 = random.randint(0, max(0, n_mels - f))
        spec[f0:f0+f, :, :] = 0
    return spec

def time_mask_np(spec, T=50, num_masks=1):
    spec = spec.copy()
    t = spec.shape[1]
    for _ in range(num_masks):
        tt = random.randint(0, min(T, t-1))
        if tt == 0: continue
        t0 = random.randint(0, t - tt)
        spec[:, t0:t0+tt, :] = 0
    return spec

def add_noise_np(spec, std=0.01):
    return spec + np.random.normal(0, std, spec.shape)

def time_shift_np(spec, max_shift=20):
    shift = random.randint(-max_shift, max_shift)
    return np.roll(spec, shift, axis=1)
