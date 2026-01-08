import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import mfcc, logfbank

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

# -------------------------------
# FIGURA 1: ESPECTROGRAMAS MEL
# -------------------------------
fig_mel, axes_mel = plt.subplots(5, 2, figsize=(14, 12), sharex=True, sharey=True)
fig_mel.suptitle("Espectrogramas en escala Mel", fontsize=16)

row = 0
for clase, files in AUDIO_FILES.items():
    for col, file in enumerate(files):
        x, sr = load_audio(os.path.join(BASE_PATH, file))

        mel_spec = logfbank(
            x,
            samplerate=sr,
            winlen=0.025,
            winstep=0.01,
            nfilt=40
        )

        ax = axes_mel[row, col]
        ax.imshow(
            mel_spec.T,
            aspect="auto",
            origin="lower"
        )
        ax.set_title(f"{clase} | {file}", fontsize=9)
        ax.set_ylabel("Banda Mel")

    row += 1

for ax in axes_mel[-1, :]:
    ax.set_xlabel("Tiempo (frames)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()