# src/audio_loader.py

from scipy.io import wavfile
import numpy as np

def load_audio(path):
    sr, x = wavfile.read(path)

    # Convertir a mono si es estéreo
    if len(x.shape) == 2:
        x = np.mean(x, axis=1)

    # Normalizar
    x = x.astype(np.float32)
    x /= np.max(np.abs(x))

    return x, sr