import numpy as np
import matplotlib.pyplot as plt

sampling_rate = 100
duration = 20

t = np.arange(0, duration, 1 / sampling_rate)

# First 10 seconds: 2 Hz
# Next 10 seconds: 8 Hz

signal = np.zeros_like(t)

signal[:1000] = np.sin(2 * np.pi * 2 * t[:1000])
signal[1000:] = np.sin(2 * np.pi * 8 * t[1000:])

plt.specgram(
    signal,
    Fs=sampling_rate,
    NFFT=256,
    noverlap=128
)

plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram")

plt.colorbar(label="Power")

plt.show()