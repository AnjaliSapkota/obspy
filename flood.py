import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

client = Client("EARTHSCOPE")

start = UTCDateTime("2026-08-26T02:00:00")
end   = UTCDateTime("2026-08-26T04:10:00")

st = client.get_waveforms(network="IO", station="EVN", location="*", channel="HHZ", starttime=start, endtime=end)

st.merge(fill_value="latest")
tr = st[0].copy()

tr.detrend("demean")
tr.detrend("linear")
tr.filter("bandpass", freqmin=0.5, freqmax=10.0, corners=4, zerophase=True)

# Print Trace Metadata
print(f"Channel: {tr.id}")
print(f"Sampling Rate: {tr.stats.sampling_rate} Hz")
print(f"Number of Samples: {tr.stats.npts}")

sampling_rate = tr.stats.sampling_rate

time = np.arange(tr.stats.npts) / sampling_rate

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(14, 9),gridspec_kw={"height_ratios": [1, 2]})

# waveformm

ax1.plot(time, tr.data)

ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Amplitude")

ax1.set_title("Waveform")

ax1.grid(True)

# plt.tight_layout()
# plt.show()

# spectogram

frequencies, times, Sxx = spectrogram(
    tr.data,
    fs=sampling_rate,
    nperseg=1024,
    noverlap=512
)


# pcm = ax2.pcolormesh(
#     times,
#     frequencies,
#     10 * np.log10(Sxx + 1e-10),
#     shading="gouraud", 
#     cmap="magma"
# )

# ax2.set_xlabel("Time (seconds)")
# ax2.set_ylabel("Frequency (Hz)")

# ax2.set_title("Spectrogram")

# # fig.colorbar(label="Power (dB)")

# ax2.set_ylim(0, 10)

# fig.colorbar(
#     pcm,
#     ax=ax2,
#     label="Power (dB)"
# )


# plt.tight_layout()
# plt.show()


# # Combined Plot: Time Series Waveform & Spectrogram
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [1, 2]}
# )

# # Top Subplot: Bandpass-Filtered Time-Series
# ax1.plot(tr1.times(), tr1.data, color="black", lw=0.6)
# ax1.set_ylabel("Counts")
# ax1.set_title(f"{tr1.id} — Waveform & Spectrogram Analysis ({tr1.stats.starttime.strftime('%Y-%m-%d %H:%M:%S UTC')})")
# ax1.grid(True, linestyle="--", alpha=0.5)

# # Bottom Subplot: Spectrogram in Decibels (dB)

pcm = ax2.pcolormesh(
    times, frequencies, 
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
