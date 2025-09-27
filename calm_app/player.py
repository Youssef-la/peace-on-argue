\
import os, time, random, math, datetime
from typing import List
import pygame

def init_audio():
    if not pygame.mixer.get_init():
        pygame.mixer.init()

def list_audio_files(folder: str) -> List[str]:
    exts = (".wav", ".mp3", ".ogg", ".flac")
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def play_playlist(folder: str, minutes: int = 10, fade_in_ms: int = 2000, fade_out_ms: int = 1000, night_scale: float = 1.0):
    files = list_audio_files(folder)
    if not files:
        print(f"[WARN] No audio files found in {folder}. Please add calm music files (wav/mp3).")
        return

    init_audio()
    end_time = time.time() + minutes * 60
    idx = 0
    random.shuffle(files)
    base_volume = 0.75 * max(0.0, min(1.0, night_scale))

    while time.time() < end_time:
        path = files[idx % len(files)]
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.set_volume(base_volume)
            pygame.mixer.music.play(fade_ms=fade_in_ms)
            # Try to approximate remaining time
            # We can't easily know track length without extra libs; play ~3-4 mins each then continue.
            seg = min(240, max(120, int(end_time - time.time())))
            for _ in range(seg * 10):
                if time.time() >= end_time: break
                time.sleep(0.1)
            pygame.mixer.music.fadeout(fade_out_ms)
        except Exception as e:
            print(f"[ERR] playback {path}: {e}")
        idx += 1

def night_volume_scale(hour_now: int, night_range=(22, 7), night_scale=0.4) -> float:
    start, end = night_range
    if start <= hour_now or hour_now < end:
        return night_scale
    return 1.0
