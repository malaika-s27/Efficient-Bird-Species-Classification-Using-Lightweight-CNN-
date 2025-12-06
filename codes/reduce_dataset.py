import os
import shutil
from pathlib import Path
from collections import Counter

SOURCE = r"E:\MLEndSemesterProject\MergedDataset_reduced"
TARGET = r"E:\MLEndSemesterProject\MergedDataset_reduced"
KEEP_NOISE = "environmental_noise"
TOP_N = 30  # number of bird classes to keep


def count_files():
    class_counts = {}

    for cls in os.listdir(SOURCE):
        cls_path = os.path.join(SOURCE, cls)
        if not os.path.isdir(cls_path):
            continue

        # count audio files
        files = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".wav", ".ogg", ".mp3", ".flac"))
        ]

        class_counts[cls] = len(files)

    return class_counts


def reduce_dataset():
    print("Scanning dataset...")

    counts = count_files()

    # Always keep environmental_noise
    if KEEP_NOISE not in counts:
        print(f"❌ Folder '{KEEP_NOISE}' not found!")
        return
    
    noise_count = counts.pop(KEEP_NOISE)

    # Select top N bird classes (excluding noise)
    top_classes = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
    top_class_names = [cls for cls, _ in top_classes]

    print("\n📌 Keeping classes:")
    for cls, cnt in top_classes:
        print(f"{cls} → {cnt} samples")

    print(f"\n📌 Keeping noise class: {KEEP_NOISE} → {noise_count} samples")

    # Make output directory
    os.makedirs(TARGET, exist_ok=True)

    # Copy selected folders
    for cls in top_class_names + [KEEP_NOISE]:
        src = os.path.join(SOURCE, cls)
        dst = os.path.join(TARGET, cls)

        print(f"Copying {cls}...")
        shutil.copytree(src, dst, dirs_exist_ok=True)

    print("\n✅ Reduced dataset created at:")
    print(TARGET)


if __name__ == "__main__":
    reduce_dataset()
