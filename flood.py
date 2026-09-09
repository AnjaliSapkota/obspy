import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

client = Client("EARTHSCOPE")

t_start = UTCDateTime("2026-08-26T01:30:00")
t_end = UTCDateTime("2026-08-26T04:00:00")

st = client.get_waveforms(
    network="IO",
    station="EVN",
    location="*",
    channel="HHZ, BHZ",
    starttime=t_start,
    endtime=t_end,
)

st.merge(fill_value="latest")

tr = st.select(channel="HHZ")[0]

# Create a copy so you don't overwrite raw data
tr_proc = tr.copy()

# Remove offset and trend
tr_proc.detrend("demean")
tr_proc.detrend("linear")

# Apply a bandpass filter (e.g., 0.5 Hz to 10.0 Hz)
tr_proc.filter("bandpass", freqmin=0.5, freqmax=10.0, corners=4, zerophase=True)

print(f"Sampling Rate: {tr_proc.stats.sampling_rate} Hz")
print(f"Number of Samples: {tr_proc.stats.npts}")
print(f"Max Amplitude: {tr_proc.data.max()}")

# Convert sample times to relative seconds from start
times = tr_proc.times()  # Array of time values in seconds
data = tr_proc.data      # Raw amplitude array

plt.figure(figsize=(10, 4))
plt.plot(times, data, color='black', linewidth=0.8)
plt.title(f"{tr_proc.id} — {tr_proc.stats.starttime.strftime('%Y-%m-%d %H:%M:%S UTC')}")
plt.xlabel("Time (seconds from start)")
plt.ylabel("Counts")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Create a copy and apply basic processing
tr_spec = tr.copy()
tr_spec.detrend("demean")

# Plot spectrogram (wlen = window length in seconds, log=True for log-frequency scale)
tr_spec.spectrogram(
    wlen=10.0,          # Window size in seconds (adjust for time/freq resolution)
    per_lap=0.9,         # 90% overlap between windows
    dbscale=True,        # Logarithmic power scale (decibels)
    log=False,           # Set to True if you want logarithmic frequency axis
    cmap="viridis"       # Color map ('viridis', 'magma', or 'plasma' work best)
)
# Preprocess signal
# tr.detrend("linear")
# tr.taper(max_percentage=0.05)

# Method 1: ObsPy direct plot (requires plt.show() in scripts)
# fig = tr.spectrogram(
#     log=True,
#     wlen=10.0,
#     per_lap=0.85,
#     dbscale=True,
#     cmap="viridis",
#     title=f"Spectrogram: {tr.id} ({t_start.date})",
#     show=False,  # Return figure object instead of auto-showing
# )

# # Render plot window explicitly
# plt.show()

