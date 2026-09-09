import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

client = Client("EARTHSCOPE")

t_start = UTCDateTime("2026-08-26T01:30:00")
t_end   = UTCDateTime("2026-08-26T04:00:00")

st = client.get_waveforms(network="IO", station="EVN", location="*", channel="HHZ, BHZ", starttime=t_start, endtime=t_end)

# Fill gaps and select high-rate vertical channel (HHZ)
st.merge(fill_value="latest")
tr = st.select(channel="HHZ")[0]

tr1 = tr.copy()
tr1.detrend("demean")
tr1.detrend("linear")
tr1.filter("bandpass", freqmin=0.5, freqmax=10.0, corners=4, zerophase=True)

# Print Trace Metadata
fs = tr1.stats.sampling_rate
print(f"Channel: {tr1.id}")
print(f"Sampling Rate: {fs} Hz")
print(f"Number of Samples: {tr1.stats.npts}")
print(f"Max Amplitude: {tr1.data.max():.2f} counts")

# Compute Spectrogram via SciPy
# Using 2-second window (2 * fs) with 85% overlap for sharp HHZ temporal resolution
f, t, Sxx = spectrogram(
    tr1.data,
    fs=fs,
    nperseg=int(2 * fs),
    noverlap=int(1.7 * fs)
)

# Combined Plot: Time Series Waveform & Spectrogram
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [1, 2]}
)

# Top Subplot: Bandpass-Filtered Time-Series
ax1.plot(tr1.times(), tr1.data, color="black", lw=0.6)
ax1.set_ylabel("Counts")
ax1.set_title(f"{tr1.id} — Waveform & Spectrogram Analysis ({tr1.stats.starttime.strftime('%Y-%m-%d %H:%M:%S UTC')})")
ax1.grid(True, linestyle="--", alpha=0.5)

# Bottom Subplot: Spectrogram in Decibels (dB)

pcm = ax2.pcolormesh(
    t, f, 
    10 * np.log10(Sxx + 1e-10),  # Convert power spectrum to dB
    shading="gouraud", 
    cmap="magma"
)
ax2.set_ylabel("Frequency (Hz)")
ax2.set_xlabel("Time (seconds from start)")
# ax2.set_ylim(0, fs / 2)  # Up to Nyquist frequency
ax2.set_ylim(0, 20)
# Add Colorbar for Power Spectral Density
fig.colorbar(pcm, ax=ax2, label="Power (dB)")

plt.tight_layout()
plt.show()
