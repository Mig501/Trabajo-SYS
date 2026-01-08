import os
import sys
import numpy as np

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

P0 = 1.0  # Potencia de referencia

print("Clase | Archivo | Energía | Potencia | Nivel Potencia (dB)")
print("-" * 75)

for clase, files in AUDIO_FILES.items():
    for file in files:
        x, sr = load_audio(os.path.join(BASE_PATH, file))

        energia = np.sum(x**2)
        potencia = energia / len(x)
        nivel_potencia = 10 * np.log10(potencia / P0 + 1e-12)

        print(
            f"{clase:8s} | {file:20s} | "
            f"{energia:10.2f} | {potencia:10.6f} | {nivel_potencia:8.2f} dB"
        )