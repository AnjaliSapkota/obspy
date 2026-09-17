import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import matplotlib.dates as mdates

client = Client("https://seiscomp.alertnepal.online")

start = UTCDateTime("2026-08-26T02:50:00")
end   = UTCDateTime("2026-08-26T04:00:00")

# st = client.get_waveforms(network="IO", station="EVN", location="*", channel="HHZ", starttime=start, endtime=end)

st = client.get_waveforms(network="NK", station="KKN", location="*", channel="BHZ", starttime=start, endtime=end)

st.merge(fill_value="latest")
tr = st[0].copy()

tr.detrend("demean")
tr.detrend("linear")
# tr.filter("bandpass", freqmin=0.5, freqmax=10.0, corners=4, zerophase=True)

# # Print Trace Metadata
# print(f"Channel: {tr.id}")
# print(f"Sampling Rate: {tr.stats.sampling_rate} Hz")
# print(f"Number of Samples: {tr.stats.npts}")

fs = tr.stats.sampling_rate
data = tr.data.astype(float)
t0 = tr.stats.starttime 

win_sec = 10
win_len = int(win_sec * fs)
num_windows = len(data) // win_len
lta_windows = 5

rms_values = []
band_power = []

# sta_lta_values = []
# frequency_sta_lta_values = []

for i in range(num_windows):
    start_index = i* win_len
    end_index = start_index + win_len
    window = data[start_index:end_index]
    rms = np.sqrt(np.mean(window**2))
    rms_values.append(rms)

    # spectral power
    freqs, Pxx = welch(window, fs=fs, nperseg=min(1024, len(window)))

    # 1 - 10 hz energy

    band_mask =((freqs>=1.0) & (freqs<= 10.0))
    band_energy = np.trapezoid(Pxx[band_mask], freqs[band_mask]) # integrated PSD = band power
    band_power.append(band_energy) 

# Compute STA/LTA Ratios
sta_lta_time = np.zeros(num_windows)
sta_lta_freq = np.zeros(num_windows)

trigger_threshold = 5.0
detrigger_threshold = 2.0

event_active_time = False
frozen_lta_time = None

event_active_freq = False
frozen_lta_freq = None

for i in range(lta_windows, num_windows):
    # Time Domain 
    if not event_active_time:
        # Dynamic LTA: Average of the prior background windows
        lta_t = np.mean(rms_values[i - lta_windows : i])
    else:
        # Locked LTA: Freeze background level during event
        lta_t = frozen_lta_time

    sta_t = rms_values[i]
    sta_lta_time[i] = sta_t / lta_t if lta_t > 0 else 0

    if not event_active_time and sta_lta_time[i] > trigger_threshold:
        event_active_time = True
        frozen_lta_time = np.mean(rms_values[i - lta_windows : i])  # Capture pre-event noise
        event_time = t0 + i * win_sec
        print(f"[TIME] TRIGGER at {event_time} | STA/LTA = {sta_lta_time[i]:.2f}")

    elif event_active_time and sta_lta_time[i] < detrigger_threshold:
        event_active_time = False
        event_time = t0 + i * win_sec
        print(f"[TIME] DETRIGGER at {event_time} | STA/LTA = {sta_lta_time[i]:.2f}")

    # Frequency-Domain
    if not event_active_freq:
        lta_f = np.mean(band_power[i - lta_windows : i])
    else:
        lta_f = frozen_lta_freq

    sta_f = band_power[i]
    sta_lta_freq[i] = sta_f / lta_f if lta_f > 0 else 0

    if not event_active_freq and sta_lta_freq[i] > trigger_threshold:
        event_active_freq = True
        frozen_lta_freq = np.mean(band_power[i - lta_windows : i])
        event_time = t0 + i * win_sec
        print(f"[FREQ] TRIGGER at {event_time} | STA/LTA = {sta_lta_freq[i]:.2f}")

    elif event_active_freq and sta_lta_freq[i] < detrigger_threshold:
        event_active_freq = False
        event_time = t0 + i * win_sec
        print(f"[FREQ] DETRIGGER at {event_time} | STA/LTA = {sta_lta_freq[i]:.2f}")

# fig, (ax1, ax2) = plt.subplots(2,1)

# time_minutes = np.arange(len(sta_lta_values)) * win_sec / 60

# # plt.figure(figsize=(12, 5))

# ax1.plot(time_minutes, sta_lta_values)

# ax1.axhline(5,linestyle="--",label="STA/LTA threshold = 5")

# ax1.set_xlabel("Minutes since 02:00 UTC")
# ax1.set_ylabel("STA/LTA")
# ax1.set_title("STA/LTA over time")

# ax1.legend()
# ax1.grid()


# # frequency staa/lta

# frequency_time_minutes = np.arange(len(frequency_sta_lta_values)) * win_sec / 60
# # plt.figure(figsize=(12, 5))

# ax2.plot(frequency_time_minutes, frequency_sta_lta_values)

# ax2.axhline(
#     5,
#     linestyle="--",
#     label="STA/LTA threshold = 5"
# )

# ax2.set_xlabel("Minutes since 02:00 UTC")
# ax2.set_ylabel("1-10 hz power STA/LTA")
# ax2.set_title("1-10 hz frequency band STA/LTA over time")

# ax2.legend()
# ax2.grid()


# plt.show()



#time
time = np.arange(tr.stats.npts) / fs


# #convert to numpy arrays
rms_values = np.array(rms_values)
# frequency_ratios = np.array(frequency_ratios)
# band_power = np.array(band_power)

time_minutes = np.arange(len(rms_values)) * win_sec / 60


fig, (ax1, ax2) = plt.subplots(2,1,figsize=(14, 9),gridspec_kw={"height_ratios": [1, 2]})

# waveformm

ax1.plot(time, tr.data)

ax1.set_xlabel("Time")
ax1.set_ylabel("Amplitude")

ax1.set_title("Waveform")

ax1.grid(True)

# plt.tight_layout()
# plt.show()

# spectogram

frequencies, times, Sxx = spectrogram(
    tr.data,
    fs=fs,
    nperseg=500 ,
    noverlap=128
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

# Visualization Setup
# times_datetime = [(t0 + t).datetime for t in np.arange(tr.stats.npts) / fs]
# window_times_datetime = [(t0 + i * win_sec).datetime for i in range(num_windows)]

# fig, (ax1, ax2) = plt.subplots(
#     1, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.5]}
# )

# # Subplot 1: Raw Waveform
# ax1.plot(times_datetime, tr.data, color="black", lw=0.5)
# ax1.set_ylabel("Amplitude")
# ax1.set_title(f"Waveform & STA/LTA Detection - Station: {tr.id}")
# ax1.grid(True, linestyle=":", alpha=0.6)

# # Subplot 2: STA/LTA Curves
# ax2.plot(window_times_datetime, sta_lta_time, label="Time Domain STA/LTA", color="tab:blue")
# ax2.plot(window_times_datetime, sta_lta_freq, label="1-10 Hz Band STA/LTA", color="tab:orange", linestyle="--")
# ax2.axhline(trigger_threshold, color="red", linestyle=":", label=f"Trigger ({trigger_threshold})")
# ax2.axhline(detrigger_threshold, color="green", linestyle=":", label=f"Detrigger ({detrigger_threshold})")
# ax2.set_ylabel("STA / LTA Ratio")
# ax2.legend(loc="upper right")
# ax2.grid(True, linestyle=":", alpha=0.6)

# # Subplot 3: Spectrogram
# nperseg = 1024
# noverlap = 256
# frequencies, times_spec, Sxx = spectrogram(tr.data, fs=fs, nperseg=nperseg, noverlap=noverlap)
# spec_times_datetime = [(t0 + t).datetime for t in times_spec]

# Sxx_db = 10 * np.log10(Sxx / np.max(Sxx) + 1e-12)
# pcm = ax3.pcolormesh(
#     spec_times_datetime, frequencies, Sxx_db, shading="gouraud", cmap="magma", vmin=-60, vmax=0
# )
# ax3.set_ylabel("Frequency (Hz)")
# ax3.set_xlabel("UTC Time")
# ax3.set_ylim(0, 20)

# # Format UTC X-Axis
# ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
# fig.autofmt_xdate()

# fig.colorbar(pcm, ax=ax3, label="Normalized Power (dB)")
# plt.tight_layout()
# plt.show()
