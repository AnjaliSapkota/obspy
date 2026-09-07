import numpy as np
import matplotlib.pyplot as plt

sampling_rate = 100
duration = 10

t = np.arange(0, duration, 1/sampling_rate)

signal = np.sin(2* np.pi * 1 * t)

fft_result = np.fft.rfft(signal)

frequencies = np.fft.rfftfreq(
    len(signal),
    d = 1/sampling_rate
)

amplitude = np.abs(fft_result)

plt.plot(frequencies, amplitude)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("FFT of 1 Hz Signal")

plt.xlim(0, 10)

plt.show()
