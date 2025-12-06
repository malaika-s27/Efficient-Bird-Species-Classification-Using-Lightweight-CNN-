# config.py
from pathlib import Path

# Paths (change if needed)
DATASET_ROOT = r"E:\MLEndSemesterProject\MergedDataset_reduced"   # raw audio (if used)
PROCESSED_DIR = r"E:\MLEndSemesterProject\processed_mels"        # .npy mel features (output of preprocessing)
MODEL_DIR = str(Path(__file__).resolve().parent.joinpath("models"))

# Audio
SAMPLE_RATE = 32000
DURATION = 5
TARGET_SAMPLES = SAMPLE_RATE * DURATION

# Mel
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FRAME_WIDTH = 300   # time frames forced in preprocessing; adjust if needed

# Training
BATCH_SIZE = 16
EPOCHS = 30
LR = 2e-4
WEIGHT_DECAY = 1e-5
MIXUP_ALPHA = 0.2
VAL_SPLIT = 0.2
RANDOM_SEED = 42
PATIENCE = 6   # early stopping

# Other
MODEL_OUT_STAGE1 = str(Path(MODEL_DIR).joinpath("stage1_birdnoise.pth"))
MODEL_OUT_STAGE2 = str(Path(MODEL_DIR).joinpath("stage2_species.pth"))
CLASS_MAP = str(Path(MODEL_DIR).joinpath("class_map.npy"))
NOISE_LABEL = "environmental noise"  # name of your noise folder in processed_mels
PROPOSAL_PATH = "/mnt/data/malaika_marbia(proposal).pdf"  # user uploaded proposal
