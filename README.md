# Calm Argument MVP (Windows + Python)

This is a minimal, **offline inference** prototype that:
- Listens to the microphone
- Uses **YAMNet** (TF Hub) locally to estimate probability of 'argument-like' sounds (scream/shout/yell/argument)
- **Auto-plays** calming music from `./calm_tracks` for N minutes when the probability crosses a threshold for K consecutive windows
- Applies a **cooldown** to avoid repeated triggers

> ⚠️ Privacy: Audio stays on-device; the script does not record or upload audio. It just performs streaming inference.

## Quickstart (Windows, PowerShell)

```powershell
cd calm_argument_mvp
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Add a few .wav/.mp3 tracks into the folder:
# calm_argument_mvp\calm_tracks\

# Run
python -m calm_app.main
```

If you get an error about **FFTW/NumPy/Sounddevice**:
- Update pip: `python -m pip install --upgrade pip`
- If your default audio device is not found, open **Sound Settings → Input** and ensure a working microphone is selected and accessible by desktop apps.

## Tuning

Edit `config.yaml`:

- `prob_threshold`: lower it to be more sensitive (e.g., 0.35) or raise to reduce false positives.
- `consecutive_windows`: increase if you get false triggers from TV/game clips.
- `cooldown_seconds`: e.g., 180–300 to suppress re-triggers.
- `night_quiet_hours` & `night_volume_scale`: lower volume at night.
- `play_minutes`: default 12 minutes.
- Add tracks into `./calm_tracks` (instrumental, 60–80 BPM recommended).
  - You can later swap this for Spotify/YouTube Music APIs.

## Notes

- First run will **download YAMNet weights** from TF Hub to your local cache.
- The detector aggregates probabilities of labels containing keywords: `scream, shout, yell, argument, angry, cry, siren`.
- This is an MVP; for production you should:
  - Add a system tray UI / start-minimized window
  - Persist basic stats (counts, time-of-day histograms)
  - Implement a TV/whitelist or AGC to reduce false triggers
  - Optionally fine-tune a small classifier on your environment
