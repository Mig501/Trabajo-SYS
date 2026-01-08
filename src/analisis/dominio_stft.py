import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Permitir importar audio_loader desde src/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from audio_loader import load_audio

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

fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True, sharey=True)
fig.suptitle("Espectrogramas (Spectrogram) de las señales acústicas analizadas", fontsize=16)

row = 0
for clase, files in AUDIO_FILES.items():
    for col, file in enumerate(files):
        audio_path = os.path.join(BASE_PATH, file)
        x, sr = load_audio(audio_path)

        # Spectrograma
        f, t, Sxx = spectrogram(
            x,
            fs=sr,
            window="hann",
            nperseg=1024,
            noverlap=512,
            scaling="density",
            mode="magnitude"
        )

        # Escala logarítmica (dB)
        Sxx_db = 20 * np.log10(Sxx + 1e-10)

        ax = axes[row, col]
        ax.pcolormesh(t, f, Sxx_db, shading="gouraud")
        ax.set_title(f"{clase} | {file}", fontsize=9)
        ax.set_ylabel("Frecuencia (Hz)")
        ax.set_ylim(0, 5000)

    row += 1

for ax in axes[-1, :]:
    ax.set_xlabel("Tiempo (s)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
