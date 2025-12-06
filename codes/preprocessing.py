# preprocessing.py
import os, shutil
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
import config

SR = config.SAMPLE_RATE
TARGET_SAMPLES = config.TARGET_SAMPLES
N_MELS = config.N_MELS
N_FFT = config.N_FFT
HOP_LENGTH = config.HOP_LENGTH
FRAME_WIDTH = config.FRAME_WIDTH

DATASET_ROOT = Path(config.DATASET_ROOT)
OUT_DIR = Path(config.PROCESSED_DIR)
BAD_DIR = OUT_DIR.parent / "bad_files_detected"

OUT_DIR.mkdir(parents=True, exist_ok=True)
BAD_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}
MIN_DURATION = SR // 2  # 0.5s

def safe_load(p):
    try:
        y, sr = sf.read(p, dtype="float32")
        if y is None:
            raise RuntimeError("soundfile returned None")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR:
            y = librosa.resample(y, sr, SR)
        return y.astype(np.float32)
    except Exception:
        try:
            y, _ = librosa.load(p, sr=SR, mono=True)
            return y.astype(np.float32)
        except Exception:
            return None

def is_valid(y):
    if y is None: return False
    if len(y) < MIN_DURATION: return False
    if np.isnan(y).any(): return False
    if np.allclose(y, 0.0): return False
    return True

def move_bad(p):
    tgt = BAD_DIR / p.relative_to(DATASET_ROOT)
    tgt.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(p), str(tgt))
    except Exception:
        pass

def pad_or_trim(y):
    if len(y) >= TARGET_SAMPLES:
        return y[:TARGET_SAMPLES]
    return np.pad(y, (0, TARGET_SAMPLES - len(y)))

def rms_normalize(y, target=0.1):
    rms = np.sqrt(np.mean(y**2) + 1e-9)
    if rms < 1e-9: return y
    return y * (target / rms)

def extract_mel(y):
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    # HPSS
    try:
        y_h, y_p = librosa.decompose.hpss(y)
    except Exception:
        y_h, y_p = y, y
    S_h = librosa.feature.melspectrogram(y=y_h, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    S_p = librosa.feature.melspectrogram(y=y_p, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    S_h_db = librosa.power_to_db(S_h, ref=np.max)
    S_p_db = librosa.power_to_db(S_p, ref=np.max)
    mel = np.stack([S_h_db, S_p_db], axis=-1)
    mel = np.nan_to_num(mel)
    # enforce time width
    if mel.shape[1] < FRAME_WIDTH:
        mel = np.pad(mel, ((0,0),(0, FRAME_WIDTH - mel.shape[1]),(0,0)))
    else:
        mel = mel[:, :FRAME_WIDTH, :]
    return mel.astype(np.float32)

def process_file(p: Path):
    y = safe_load(str(p))
    if not is_valid(y):
        move_bad(p)
        return False
    y = rms_normalize(y)
    y = pad_or_trim(y)
    try:
        mel = extract_mel(y)
    except Exception:
        move_bad(p)
        return False
    out_dir = OUT_DIR / p.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / (p.stem + ".npy"), mel)
    return True

def main():
    files = [p for p in DATASET_ROOT.rglob("*") if p.suffix.lower() in SUPPORTED]
    print("Total files found:", len(files))
    ok=0; bad=0
    for p in tqdm(files):
        if process_file(p):
            ok+=1
        else:
            bad+=1
    print("Saved mels:", ok, "Bad moved:", bad, "Bad dir:", BAD_DIR)

if __name__=="__main__":
    main()
