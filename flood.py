import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

client = Client("https://seiscomp.alertnepal.online")

start = UTCDateTime("2026-08-26T02:00:00")
end   = UTCDateTime("2026-08-26T04:00:00")

baseline_start = UTCDateTime("2026-08-26T02:00:00")
baseline_end =UTCDateTime("2026-08-26T02:30:00")

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

fs = tr.stats.sampling_rate

# 10 sec window to calculate rms values
window_seconds = 10

samples_per_window = int(window_seconds * fs)

data = tr.data

num_windows = len(data) // samples_per_window

rms_values = []
frequency_ratios = []
window_times = []

for i in range(num_windows):
    start_index = i* samples_per_window
    end_index = start_index + samples_per_window
    window = data[start_index:end_index]
    rms = np.sqrt(np.mean(window**2))
    rms_values.append(rms)

frequencies, power = welch(window, fs=fs, npserg=min(1024, len(window)))

# 1 - 10 hz energy

band_mask =((frequencies>=1.0) & (frequencies<= 1.0))

total_mask = ((frequencies >= 0.5) & (frequencies <= 20.0))

band_energy = np.trapezoid(power[band_mask], frequencies[band_mask])

total_energy = np.trapezoid(power[total_mask], frequencies[total_mask])

# energy ratio
if total_energy > 0:
    ratio = band_energy / total_energy
else:
    ratio = 0

frequency_ratios.append(ratio)

#time
# time = np.arange(tr.stats.npts) / fs
current_time = (tr.stats.starttime + i* window)

window_times.append(current_time.datetime)

#convert to numpy arrays
rms_values = np.array(rms_values)
frequency_ratios = np.array(frequency_ratios)

# find baseline windows

window_times_utc = [
    UTCDateTime(t)
    for t in window_times
]

baseline_mask = np.array([baseline_start <= t < baseline_end for t in window_times_utc])
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
    fs=fs,
    nperseg=1024,
    noverlap=512
)

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
