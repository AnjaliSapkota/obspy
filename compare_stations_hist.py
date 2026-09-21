import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import hilbert, windows, correlate, correlation_lags

client = Client("https://seiscomp.alertnepal.online")

start = UTCDateTime("2026-08-26T02:50:00")
end   = UTCDateTime("2026-08-26T02:55:00")

st_a = client.get_waveforms(network="NK", station="KKN", location="*", channel="BHZ", starttime=start, endtime=end)
st_b = client.get_waveforms(network="IO", station="EVN", location="*", channel="BHZ", starttime=start, endtime=end)

tr_a = st_a[0].copy()
tr_b = st_b[0].copy()

# Basic Preprocessing
tr_a.detrend("linear").detrend("demean").taper(max_percentage=0.05, type="cosine")
tr_b.detrend("linear").detrend("demean").taper(max_percentage=0.05, type="cosine")

# Bandpass filtering (1 - 8 Hz)
freqmin, freqmax = 1.0, 8.0
tr_a.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=True)
tr_b.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=True)

# Interpolate to common sampling rate
target_fs = 20.0
tr_a.interpolate(sampling_rate=target_fs, method="weighted_average_slopes")
tr_b.interpolate(sampling_rate=target_fs, method="weighted_average_slopes")

# Synchronize time boundaries
common_start = max(tr_a.stats.starttime, tr_b.stats.starttime)
common_end   = min(tr_a.stats.endtime, tr_b.stats.endtime)

tr_a = tr_a.slice(common_start, common_end)
tr_b = tr_b.slice(common_start, common_end)

raw_a = tr_a.data.astype(float)
raw_b = tr_b.data.astype(float)

# Compute Hilbert Envelopes
env_a = np.abs(hilbert(raw_a))
env_b = np.abs(hilbert(raw_b))

# Crop 5% to remove Hilbert boundary artifacts
crop = int(0.05 * len(env_a))
env_a = env_a[crop:-crop]
env_b = env_b[crop:-crop]
raw_a = raw_a[crop:-crop]
raw_b = raw_b[crop:-crop]

# Zero-mean the envelopes (without dividing by std before tapering)
env_a_zero = env_a - np.mean(env_a)
env_b_zero = env_b - np.mean(env_b)

# Taper
n = len(env_a_zero)
taper = windows.tukey(n, alpha=0.05)
a = env_a_zero * taper
b = env_b_zero * taper

fs = tr_a.stats.sampling_rate

# Compute Cross-Correlation directly via SciPy
corr_full = correlate(a, b, mode="full")
lags_full = correlation_lags(len(a), len(b), mode="full")

# Normalize by energy of signals
norm_factor = np.sqrt(np.sum(a**2) * np.sum(b**2))
normalized_cc_full = corr_full / norm_factor

# Constrain lag search to +/- 30 seconds
max_shift_seconds = 30
max_shift_samples = int(round(max_shift_seconds * fs))

mask = (lags_full >= -max_shift_samples) & (lags_full <= max_shift_samples)
lags_samples = lags_full[mask]
normalized_cc = normalized_cc_full[mask]
lags_seconds = lags_samples / fs

# Find peak
peak_index = np.argmax(normalized_cc)
best_lag_samples = lags_samples[peak_index]
best_lag_seconds = lags_seconds[peak_index]
max_corr = normalized_cc[peak_index]

print(f"Max search shift: {max_shift_seconds} s")
print(f"Best lag in samples: {best_lag_samples}")
print(f"Best lag in seconds: {best_lag_seconds:.3f} s")
print(f"Normalized Correlation Coefficient: {max_corr:.4f}")

# Plotting
t_vec = np.arange(n) / fs

plt.figure(figsize=(12, 8))

# Subplot 1: Raw Signals & Envelopes
plt.subplot(2, 1, 1)
plt.plot(t_vec, env_a, color="crimson", linewidth=1.2, label="KKN Envelope")
plt.plot(t_vec, env_b, color="navy", linewidth=1.2, label="EVN Envelope")
plt.ylabel("Envelope Amplitude")
plt.title("Bandpass Filtered Envelopes (1-8 Hz)")
plt.legend(loc="upper right")
plt.grid(True)

# Subplot 2: Normalized Cross-Correlation
plt.subplot(2, 1, 2)
plt.plot(lags_seconds, normalized_cc, color="black", label="Envelope Cross-Correlation")
plt.axvline(best_lag_seconds, color="red", linestyle="--", 
            label=f"Peak at {best_lag_seconds:.2f} s (CC = {max_corr:.3f})")
plt.axhline(0, color="gray", linestyle=":")
plt.xlabel("Lag / Time Delay (seconds)")
plt.ylabel("Normalized CC")
plt.title("Envelope Cross-Correlation (KKN vs EVN)")
plt.legend(loc="upper right")
plt.grid(True)

plt.tight_layout()
plt.show()