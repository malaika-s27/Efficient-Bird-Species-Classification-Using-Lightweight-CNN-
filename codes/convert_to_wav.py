# convert_to_wav.py
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment
import config
from tqdm import tqdm

SRC_ROOT = Path(config.DATASET_ROOT)  # original folder with .ogg etc
DST_ROOT = SRC_ROOT.parent / (SRC_ROOT.name + "_wav")  # e.g., MergedDataset_wav
DST_EXT = ".wav"
SR = config.SAMPLE_RATE

SUPPORTED_EXTS = (".ogg", ".mp3", ".flac", ".wav", ".WAV", ".OGG", ".MP3", ".FLAC")

def ensure_dst(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def convert_with_soundfile(in_path: Path, out_path: Path):
    # Try reading with soundfile and write wav
    try:
        data, sr = sf.read(str(in_path), dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SR:
            data = librosa.resample(data, orig_sr=sr, target_sr=SR)
            sr = SR
        sf.write(str(out_path), data, sr, subtype='PCM_16')
        return True
    except Exception:
        return False

def convert_with_librosa(in_path: Path, out_path: Path):
    # librosa uses audioread backend which is robust for ogg/mp3 without ffmpeg installed
    try:
        data, sr = librosa.load(str(in_path), sr=SR, mono=True)
        sf.write(str(out_path), data, SR, subtype='PCM_16')
        return True
    except Exception:
        return False

def convert_with_pydub(in_path: Path, out_path: Path):
    try:
        audio = AudioSegment.from_file(str(in_path))
        # pydub's export will use ffmpeg/avlib if installed; but pydub can still decode some formats using ffmpeg-less backends rarely
        audio = audio.set_frame_rate(SR).set_channels(1)
        ensure_dst(out_path)
        audio.export(str(out_path), format="wav")
        return True
    except Exception:
        return False

def convert_file(in_path: Path, out_path: Path):
    # skip if already exists
    if out_path.exists():
        return True

    ensure_dst(out_path)
    # Try soundfile
    if convert_with_soundfile(in_path, out_path):
        return True
    # Try librosa (audioread backend)
    if convert_with_librosa(in_path, out_path):
        return True
    # Last resort: pydub
    if convert_with_pydub(in_path, out_path):
        return True

    return False

def main():
    print("Source root:", SRC_ROOT)
    print("Destination root:", DST_ROOT)
    failures = []
    total = 0
    for root, dirs, files in os.walk(SRC_ROOT):
        for f in files:
            ext = Path(f).suffix.lower()
            if ext not in SUPPORTED_EXTS:
                continue
            total += 1
            rel = Path(root).joinpath(f).relative_to(SRC_ROOT)
            src_path = SRC_ROOT.joinpath(rel)
            dst_path = DST_ROOT.joinpath(rel).with_suffix(DST_EXT)
            success = convert_file(src_path, dst_path)
            if not success:
                failures.append(str(src_path))

    print(f"Scanned {total} audio files.")
    if failures:
        print("Failed to convert these files (inspect manually):")
        for p in failures:
            print(" -", p)
    else:
        print("All files converted (or were already wav).")
    print("Converted dataset available at:", DST_ROOT)

if __name__ == "__main__":
    main()
