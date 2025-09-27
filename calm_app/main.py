\
"""
Windows + Python MVP:
 - Streams mic audio
 - Uses TF Hub YAMNet to detect 'argument-like' events (scream/shout/yell/argument)
 - Plays calming music from ./calm_tracks on trigger

Prepare:
 1) python -m venv .venv && .venv\Scripts\activate (Windows PowerShell)
 2) pip install -r requirements.txt
 3) Add a few .wav/.mp3 files into ./calm_tracks
 4) python -m calm_app.main
"""

import os, time, csv, datetime, sys, threading
import numpy as np
import yaml

# Audio + ML
import tensorflow as tf
import tensorflow_hub as hub

from .audio_utils import AudioStream
from .player import play_playlist, night_volume_scale

# --------- Config ---------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(ROOT, "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

SR = int(CFG.get("sample_rate", 16000))
WIN_S = float(CFG.get("window_seconds", 0.96))
HOP_S = float(CFG.get("hop_seconds", 0.48))
THRESH = float(CFG.get("prob_threshold", 0.45))
K = int(CFG.get("consecutive_windows", 3))
COOLDOWN_S = int(CFG.get("cooldown_seconds", 120))

PLAY_MIN = int(CFG.get("play_minutes", 12))
FADE_IN = int(CFG.get("fade_in_ms", 3000))
FADE_OUT = int(CFG.get("fade_out_ms", 2000))
CALM_DIR = os.path.join(ROOT, CFG.get("calm_tracks_dir","./calm_tracks"))

NIGHT_RANGE = tuple(CFG.get("night_quiet_hours",[22,7]))
NIGHT_SCALE = float(CFG.get("night_volume_scale", 0.4))

# --------- Model ---------
print("[INFO] Loading YAMNet from TF Hub (first run may download model weights)...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
class_map_path = yamnet.class_map_path().numpy().decode("utf-8")
class_names = []
with tf.io.gfile.GFile(class_map_path) as f:
    reader = csv.DictReader(f)
    for r in reader:
        class_names.append(r["display_name"])

# Build a set of class indices that indicate 'argument-like' signals
KEYWORDS = ["scream", "screaming", "shout", "yell", "yelling", "argument", "angry", "cry", "siren"]
argument_class_ids = [i for i, name in enumerate(class_names) if any(kw in name.lower() for kw in KEYWORDS)]
if not argument_class_ids:
    # Fallback: track a few common ones if map changes
    print("[WARN] No matching classes found with keywords; using top-1 score as proxy.")
print(f"[INFO] Monitoring {len(argument_class_ids)} classes: {[class_names[i] for i in argument_class_ids[:8]]} ...")

def score_argument_like(waveform: np.ndarray) -> float:
    """
    waveform: 1-D float32 numpy array at 16kHz
    Returns probability (0..1) aggregated over 'argument-like' labels.
    """
    # yamnet expects [samples] float32 in range [-1,1] at 16k
    wf = tf.convert_to_tensor(waveform, dtype=tf.float32)
    # Model outputs:
    # scores: [frames, 521], embeddings: [frames, 1024], spectrogram: [frames, 64]
    scores, embeddings, spectrogram = yamnet(wf)
    s = scores.numpy()  # (frames, 521)
    if argument_class_ids:
        prob = float(np.mean(s[:, argument_class_ids].sum(axis=1)))
    else:
        prob = float(np.mean(np.max(s, axis=1)))
    return prob

def detector_loop():
    block_samples = int(HOP_S * SR)   # hop size
    window_samples = int(WIN_S * SR)
    ring = np.zeros(window_samples, dtype=np.float32)

    last_trigger_t = 0.0
    consec = 0

    with AudioStream(samplerate=SR, block_seconds=HOP_S) as stream:
        print("[INFO] Listening... Press Ctrl+C to stop.")
        while True:
            block = stream.read_block()
            # roll and append
            ring = np.concatenate([ring[len(block):], block]) if len(block) < len(ring) else block[-len(ring):]

            # Score current 0.96s window
            prob = score_argument_like(ring)

            # UI print (throttle)
            now = time.time()
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] prob={prob:.3f} (thr={THRESH})   ", end="\r")

            if prob >= THRESH:
                consec += 1
            else:
                consec = 0

            cooled = (now - last_trigger_t) > COOLDOWN_S
            if consec >= K and cooled:
                # Trigger calming playlist
                last_trigger_t = now
                consec = 0
                print("\n[TRIGGER] Elevated agitation detected → starting calming playlist.")
                hour_now = datetime.datetime.now().hour
                vol_scale = night_volume_scale(hour_now, NIGHT_RANGE, NIGHT_SCALE)

                t = threading.Thread(
                    target=play_playlist,
                    kwargs=dict(folder=CALM_DIR, minutes=PLAY_MIN, fade_in_ms=FADE_IN, fade_out_ms=FADE_OUT, night_scale=vol_scale),
                    daemon=True
                )
                t.start()
                # Continue listening during playback

if __name__ == "__main__":
    try:
        detector_loop()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
