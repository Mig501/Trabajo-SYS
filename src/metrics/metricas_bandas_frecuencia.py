import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from audio_loader import load_audio

AUDIO_FILES = {
    "exterior": ["1-18527-B-44.wav", "3-111102-B-46.wav"],
    "persona": ["3-103599-B-25.wav", "3-107123-A-26.wav"],
    "animal": ["5-9032-A-0.wav", "2-109231-C-9.wav"],
    "interior": ["1-12654-B-15.wav", "1-137-A-32.wav"],
    "natural": ["5-219379-A-11.wav", "1-23222-A-19.wav"]
}

BASE_PATH = "data"

# Definición de bandas
BANDS = {
    "Bajas (0-500 Hz)": (0, 500),
    "Medias (500-2000 Hz)": (500, 2000),
    "Altas (2000-5000 Hz)": (2000, 5000)
}

print("\nANÁLISIS ESPECTRAL POR BANDAS DE FRECUENCIA")
print("-" * 65)

for clase, files in AUDIO_FILES.items():
    print(f"\nClase: {clase}")
    for file in files:
        x, sr = load_audio(os.path.join(BASE_PATH, file))

        # FFT
        X = np.fft.fft(x)
        freqs = np.fft.fftfreq(len(x), d=1/sr)

        power_spectrum = np.abs(X) ** 2

        print(f"  Archivo: {file}")
        for band_name, (fmin, fmax) in BANDS.items():
            idx = np.logical_and(freqs >= fmin, freqs < fmax)
            band_energy = np.sum(power_spectrum[idx])
            print(f"    {band_name}: {band_energy:.2e}")
