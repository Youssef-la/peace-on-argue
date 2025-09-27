\
import numpy as np
import sounddevice as sd
import librosa

class AudioStream:
    def __init__(self, samplerate=16000, block_seconds=0.48):
        self.samplerate = samplerate
        self.blocksize = int(block_seconds * samplerate)
        self.stream = None

    def __enter__(self):
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            dtype="float32",
            latency="low"
        )
        self.stream.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass

    def read_block(self):
        """Return mono float32 audio block length blocksize"""
        data, _ = self.stream.read(self.blocksize)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        return np.asarray(data, dtype=np.float32)

def resample_to_16k(audio, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    y = librosa.resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr, res_type="kaiser_best")
    return y.astype(np.float32)
