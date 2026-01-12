import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from audio_loader import load_audio

# Audios de ejemplo (uno por tipo)
'''
AUDIO_FILES = {
    "exterior": "2-117615-B-48.wav",
    "persona": "3-103599-B-25.wav",
    "animal": "5-9032-A-0.wav",
    "interior": "2-141584-A-38.wav",
    "natural": "5-219379-A-11.wav"
}
'''
AUDIO_FILES = {
    "exterior": "2-70938-A-42.wav",
    "persona": "3-107123-A-26.wav",
    "animal": "1-15689-B-4.wav",
    "interior": "1-137-A-32.wav",
    "natural": "4-187504-B-17.wav"
}

BASE_PATH = "data"

def butter_filter(signal, sr, cutoff, btype, order=4):
    nyq = 0.5 * sr
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype=btype)
    return filtfilt(b, a, signal)

fig, axes = plt.subplots(5, 2, figsize=(14, 12))
fig.suptitle("Filtrado en frecuencia: señal original vs filtrada", fontsize=16)

row = 0
for clase, file in AUDIO_FILES.items():
    x, sr = load_audio(os.path.join(BASE_PATH, file))

    # Filtrado
    x_low = butter_filter(x, sr, cutoff=1000, btype="low")
    x_high = butter_filter(x, sr, cutoff=500, btype="high")

    t = np.arange(len(x)) / sr

    # Señal original
    axes[row, 0].plot(t, x)
    axes[row, 0].set_title(f"{clase} | Original")
    axes[row, 0].set_ylabel("Amplitud")
    axes[row, 0].grid(True)

    # Señal filtrada
    axes[row, 1].plot(t, x_low, color='green', label="Pasa-bajo (1 kHz)")
    axes[row, 1].plot(t, x_high, color='orange', label="Pasa-alto (500 Hz)", alpha=0.7)
    axes[row, 1].set_title(f"{clase} | Filtrada")
    axes[row, 1].legend()
    axes[row, 1].grid(True)

    row += 1

for ax in axes[-1, :]:
    ax.set_xlabel("Tiempo (s)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()