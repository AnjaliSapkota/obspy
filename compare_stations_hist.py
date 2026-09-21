import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import (
    hilbert, windows, correlate, correlation_lags, 
    welch, spectrogram, coherence
)

# 1. Fetch Waveforms
client = Client("https://seiscomp.alertnepal.online")

start = UTCDateTime("2026-08-26T02:50:00")
end   = UTCDateTime("2026-08-26T02:55:00")

st_a = client.get_waveforms(network="NK", station="KKN", location="*", channel="BHZ", starttime=start, endtime=end)
st_b = client.get_waveforms(network="IO", station="EVN", location="*", channel="BHZ", starttime=start, endtime=end)

tr_a = st_a[0].copy()
tr_b = st_b[0].copy()

# 2. Basic Preprocessing & Filtering (1 - 8 Hz)
freqmin, freqmax = 1.0, 8.0
tr_a.detrend("linear").detrend("demean").taper(max_percentage=0.05, type="cosine")
tr_b.detrend("linear").detrend("demean").taper(max_percentage=0.05, type="cosine")

tr_a.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=True)
tr_b.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=True)

# 3. Interpolation & Time Alignment
target_fs = 20.0
tr_a.interpolate(sampling_rate=target_fs, method="weighted_average_slopes")
tr_b.interpolate(sampling_rate=target_fs, method="weighted_average_slopes")

common_start = max(tr_a.stats.starttime, tr_b.stats.starttime)
common_end   = min(tr_a.stats.endtime, tr_b.stats.endtime)

tr_a = tr_a.trim(common_start, common_end)
tr_b = tr_b.trim(common_start, common_end)

raw_a = tr_a.data.astype(float)
raw_b = tr_b.data.astype(float)

# Ensure matching lengths
min_len = min(len(raw_a), len(raw_b))
raw_a = raw_a[:min_len]
raw_b = raw_b[:min_len]

# 4. Hilbert Envelope & Edge Artifact Removal
env_a = np.abs(hilbert(raw_a))
env_b = np.abs(hilbert(raw_b))

crop = int(0.05 * len(env_a))
env_a = env_a[crop:-crop]
env_b = env_b[crop:-crop]
raw_a = raw_a[crop:-crop]
raw_b = raw_b[crop:-crop]

# 5. Envelope Cross-Correlation
env_a_zero = env_a - np.mean(env_a)
env_b_zero = env_b - np.mean(env_b)

n = len(env_a_zero)
taper = windows.tukey(n, alpha=0.05)
a = env_a_zero * taper
b = env_b_zero * taper

fs = tr_a.stats.sampling_rate

corr_full = correlate(a, b, mode="full")
lags_full = correlation_lags(len(a), len(b), mode="full")
norm_factor = np.sqrt(np.sum(a**2) * np.sum(b**2))
normalized_cc_full = corr_full / norm_factor

max_shift_seconds = 30
max_shift_samples = int(round(max_shift_seconds * fs))

mask = (lags_full >= -max_shift_samples) & (lags_full <= max_shift_samples)
lags_samples = lags_full[mask]
normalized_cc = normalized_cc_full[mask]
lags_seconds = lags_samples / fs

peak_index = np.argmax(normalized_cc)
best_lag_samples = lags_samples[peak_index]
best_lag_seconds = lags_seconds[peak_index]
max_corr = normalized_cc[peak_index]

print(f"Max search shift: {max_shift_seconds} s")
print(f"Best lag in samples: {best_lag_samples}")
print(f"Best lag in seconds: {best_lag_seconds:.3f} s")
print(f"Normalized Correlation Coefficient: {max_corr:.4f}")

# 6. Spectral Analysis
freq_a, psd_a = welch(raw_a, fs=fs, window="hann", nperseg=int(20 * fs), noverlap=int(10 * fs))
freq_b, psd_b = welch(raw_b, fs=fs, window="hann", nperseg=int(20 * fs), noverlap=int(10 * fs))

f_a, t_a, Sxx_a = spectrogram(raw_a, fs=fs, window="hann", nperseg=int(10 * fs), noverlap=int(8 * fs), scaling="density")
f_b, t_b, Sxx_b = spectrogram(raw_b, fs=fs, window="hann", nperseg=int(10 * fs), noverlap=int(8 * fs), scaling="density")

f_coh, Cxy = coherence(raw_a, raw_b, fs=fs, window="hann", nperseg=int(20 * fs), noverlap=int(10 * fs))

# 7. Visualization
fig, axes = plt.subplots(5, 1, figsize=(12, 16))

# Time Domain Envelopes
t = np.arange(len(env_a)) / fs
axes[0].plot(t, env_a, label="KKN Envelope", color="tab:blue")
axes[0].plot(t, env_b, label="EVN Envelope", color="tab:orange")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Time Domain — Hilbert Envelopes")
axes[0].legend()
axes[0].grid(True)

# Cross-Correlation Function
axes[1].plot(lags_seconds, normalized_cc, color="black")
axes[1].axvline(best_lag_seconds, color="red", linestyle="--", label=f"Peak Lag: {best_lag_seconds:.3f} s (CC: {max_corr:.2f})")
axes[1].set_xlabel("Time Lag (s)")
axes[1].set_ylabel("Normalized CC")
axes[1].set_title("Envelope Cross-Correlation")
axes[1].legend()
axes[1].grid(True)

# Power Spectral Density
axes[2].semilogy(freq_a, psd_a, label="KKN")
axes[2].semilogy(freq_b, psd_b, label="EVN")
axes[2].set_xlim(0, 10)
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("PSD")
axes[2].set_title("Frequency Domain — Power Spectral Density")
axes[2].legend()
axes[2].grid(True)

# Spectrograms
mesh1 = axes[3].pcolormesh(t_a, f_a, 10 * np.log10(Sxx_a + 1e-20), shading="auto", cmap="viridis")
axes[3].set_ylim(0, 10)
axes[3].set_ylabel("Frequency (Hz)")
axes[3].set_title("KKN Spectrogram (dB)")
fig.colorbar(mesh1, ax=axes[3], label="dB")

mesh2 = axes[4].pcolormesh(t_b, f_b, 10 * np.log10(Sxx_b + 1e-20), shading="auto", cmap="viridis")
axes[4].set_ylim(0, 10)
axes[4].set_xlabel("Time (s)")
axes[4].set_ylabel("Frequency (Hz)")
axes[4].set_title("EVN Spectrogram (dB)")
fig.colorbar(mesh2, ax=axes[4], label="dB")

plt.tight_layout()
plt.show()

# Coherence Plot
plt.figure(figsize=(12, 4))
plt.plot(f_coh, Cxy, color="purple")
plt.xlim(0, 10)
plt.ylim(0, 1)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Coherence")
plt.title("Frequency-Domain Coherence — KKN vs EVN")
plt.grid(True)
plt.show()