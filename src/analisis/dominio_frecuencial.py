import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from audio_loader import load_audio

# -------------------------------
# Audios seleccionados (2 por clase)
# -------------------------------
AUDIO_FILES = {
    "exterior": [
        "2-117615-B-48.wav",
        "2-70938-A-42.wav"
    ],
    "persona": [
        "3-103599-B-25.wav",
        "3-107123-A-26.wav"
    ],
    "animal": [
        "5-9032-A-0.wav",
        "1-15689-B-4.wav"
    ],
    "interior": [
        "2-141584-A-38.wav",
        "1-137-A-32.wav"
    ],
    "natural": [
        "5-219379-A-11.wav",
        "4-187504-B-17.wav"
    ]
}

BASE_PATH = "data"

# Funciones de análisis frecuencial
def compute_spectrum(x, sr):
    """
    Calcula el espectro de magnitud usando FFT.
    Devuelve frecuencias positivas y magnitud asociada.
    """
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x), d=1/sr)

    idx = freqs >= 0
    return freqs[idx], np.abs(X[idx])

fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
fig.suptitle("Espectro de frecuencia de las señales analizadas", fontsize=16)

row = 0
for clase, files in AUDIO_FILES.items():
    for col, file in enumerate(files):
        audio_path = os.path.join(BASE_PATH, file)
        x, sr = load_audio(audio_path)

        freqs, magnitude = compute_spectrum(x, sr)

        ax = axes[row, col]
        ax.plot(freqs, magnitude)
        ax.set_xlim(0, 5000)
        ax.set_title(f"{clase} | {file}", fontsize=9)
        ax.set_ylabel("Magnitud")
        ax.grid(True)

    row += 1

for ax in axes[-1, :]:
    ax.set_xlabel("Frecuencia (Hz)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()