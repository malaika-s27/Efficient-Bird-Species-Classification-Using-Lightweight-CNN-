# predict.py (FIXED VERSION)
import numpy as np, torch, librosa
from pathlib import Path
import config
from model import build_efficientnet_b0
import argparse

DEVICE = torch.device("cpu")
FRAME_WIDTH = config.FRAME_WIDTH

# load models
stage1 = build_efficientnet_b0(num_classes=2)
stage1.load_state_dict(torch.load(config.MODEL_OUT_STAGE1, map_location='cpu'))
stage1.eval()

stage2 = None
if Path(config.MODEL_OUT_STAGE2).exists():
    class_map = np.load(config.CLASS_MAP, allow_pickle=True)
    num_cls = len(class_map)
    stage2 = build_efficientnet_b0(num_classes=num_cls)
    stage2.load_state_dict(torch.load(config.MODEL_OUT_STAGE2, map_location='cpu'))
    stage2.eval()
else:
    class_map = []


def load_mel_from_audio(path):
    # ---- FIX 1: always use librosa.load() ----
    y, sr = librosa.load(path, sr=None, mono=True)

    # resample
    if sr != config.SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=config.SAMPLE_RATE)

    # pad/cut
    if len(y) < config.TARGET_SAMPLES:
        y = np.pad(y, (0, config.TARGET_SAMPLES-len(y)))
    else:
        y = y[:config.TARGET_SAMPLES]

    # HPSS
    try:
        y_h, y_p = librosa.decompose.hpss(y)
    except:
        y_h, y_p = y, y

    # ---- FIX 2: use keyword arguments for librosa 0.10+ ----
    mel_h = librosa.feature.melspectrogram(
        y=y_h, sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, n_mels=config.N_MELS
    )
    mel_p = librosa.feature.melspectrogram(
        y=y_p, sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, n_mels=config.N_MELS
    )

    mel_h = librosa.power_to_db(mel_h, ref=np.max)
    mel_p = librosa.power_to_db(mel_p, ref=np.max)

    mel = np.stack([mel_h, mel_p], axis=-1)

    # pad or crop width
    if mel.shape[1] < FRAME_WIDTH:
        mel = np.pad(mel, ((0,0),(0,FRAME_WIDTH-mel.shape[1]),(0,0)))
    else:
        mel = mel[:, :FRAME_WIDTH, :]

    return mel.astype(np.float32)


def predict_file(path, thresh_stage1=0.5, species_conf_thresh=0.65, tta=3):
    mel = load_mel_from_audio(path)

    probs_stage1 = []
    probs_stage2 = []

    for t in range(tta):
        if t > 0:
            shift = np.random.randint(-20, 20)
            m = np.roll(mel, shift, axis=1)
        else:
            m = mel

        x = torch.tensor(m, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            p1 = torch.softmax(stage1(x), dim=1).cpu().numpy()[0]
            probs_stage1.append(p1)

            if stage2 is not None:
                p2 = torch.softmax(stage2(x), dim=1).cpu().numpy()[0]
                probs_stage2.append(p2)

    p1 = np.mean(probs_stage1, axis=0)
    bird_prob = float(p1[1])

    # Noise detector
    if bird_prob < thresh_stage1:
        return {"label": "Noise/Non-bird", "confidence": float(1 - bird_prob), "details": p1.tolist()}

    # Bird but no species model exists
    if stage2 is None:
        return {"label": "Bird (species model missing)", "confidence": bird_prob}

    p2 = np.mean(probs_stage2, axis=0)
    top_idx = int(p2.argmax())
    top_conf = float(p2[top_idx])

    # Unknown species threshold
    if top_conf < species_conf_thresh:
        return {"label": "Unknown bird species", "confidence": top_conf, "top_probs": p2.tolist()}

    return {"label": str(class_map[top_idx]), "confidence": top_conf, "top_probs": p2.tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--tta", type=int, default=3)
    args = parser.parse_args()
    print(predict_file(args.audio, tta=args.tta))
