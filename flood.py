import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import matplotlib.dates as mdates

client = Client("https://seiscomp.alertnepal.online")

start = UTCDateTime("2026-08-26T02:00:00")
end   = UTCDateTime("2026-08-26T04:00:00")

# baseline_start = UTCDateTime("2026-08-26T02:00:00")
# baseline_end =  UTCDateTime("2026-08-26T04:30:00")

st = client.get_waveforms(network="IO", station="EVN", location="*", channel="HHZ", starttime=start, endtime=end)

st.merge(fill_value="latest")
tr = st[0].copy()

# tr.detrend("demean")
# tr.detrend("linear")'
# tr.filter("bandpass", freqmin=0.5, freqmax=10.0, corners=4, zerophase=True)

# Print Trace Metadata
print(f"Channel: {tr.id}")
print(f"Sampling Rate: {tr.stats.sampling_rate} Hz")
print(f"Number of Samples: {tr.stats.npts}")

fs = tr.stats.sampling_rate
data = tr.data.astype(float)
t0 = tr.stats.starttime 

win_sec = 60


win_len = int(win_sec * fs)

# data = window.data

num_windows = len(data) // win_len

rms_values = []
# frequency_ratios = []
# window_times = []
band_power = []

sta_lta_values = []
frequency_sta_lta_values = []
lta_windows = 5

for i in range(num_windows):
    start_index = i* win_len
    end_index = start_index + win_len
    window = data[start_index:end_index]
    rms = np.sqrt(np.mean(window**2))
    rms_values.append(rms)

    # STA/LTA calculation
    if len(rms_values) > lta_windows:

        # Previous 5 RMS values = long-term/background level
        lta = np.mean(rms_values[-lta_windows-1:-1])

        # Current RMS = short-term/current level
        sta = rms

        if lta > 0:
            sta_lta = sta / lta
        else:   
            sta_lta = 0

        sta_lta_values.append(sta_lta)

        if sta_lta > 5:
            print(f"Possible event at "f"{t0 + start_index / fs}"f" | STA/LTA = {sta_lta:.2f}"
            )

    else:
        sta_lta_values.append(0)

    freqs, Pxx = welch(window, fs=fs, nperseg=min(1024, len(window)))

# 1 - 10 hz energy

    band_mask =((freqs>=1.0) & (freqs<= 10.0))

    # total_mask = ((freqs >= 0.5) & (freqs <= 20.0))

    band_energy = np.trapezoid(Pxx[band_mask], freqs[band_mask]) # integrated PSD = band power

    # total_energy = np.trapezoid(Pxx[total_mask], freqs[total_mask])

    band_power.append(band_energy) 


for i in range(len(band_power)):

    if i >= lta_windows:

        
        # Previous 5 windows = background 1–10 Hz power
        lta = np.mean(band_power[i-lta_windows:i])

        # Current 1–10 Hz power
        sta = band_power[i]

        if lta > 0:
            freq_sta_lta = sta / lta
        else:
            freq_sta_lta = 0

        frequency_sta_lta_values.append(freq_sta_lta)

        if freq_sta_lta > 5:
            event_time = t0 + i * win_sec
            print(f"Possible event at "f"{event_time}"f" | Frequency STA/LTA = {freq_sta_lta:.2f}" )
    else:
        frequency_sta_lta_values.append(0)


fig, (ax1, ax2) = plt.subplots(2,1)

time_minutes = np.arange(len(sta_lta_values))

# plt.figure(figsize=(12, 5))

ax1.plot(time_minutes, sta_lta_values)

ax1.axhline(5,linestyle="--",label="STA/LTA threshold = 5")

ax1.set_xlabel("Minutes since 02:00 UTC")
ax1.set_ylabel("STA/LTA")
ax1.set_title("STA/LTA over time")

ax1.legend()
ax1.grid()


# frequency staa/lta

frequency_time_minutes = np.arange(len(frequency_sta_lta_values))
# plt.figure(figsize=(12, 5))

ax2.plot(frequency_time_minutes, frequency_sta_lta_values)

ax2.axhline(
    5,
    linestyle="--",
    label="STA/LTA threshold = 5"
)

ax2.set_xlabel("Minutes since 02:00 UTC")
ax2.set_ylabel("1-10 hz power STA/LTA")
ax2.set_title("1-10 hz frequency band STA/LTA over time")

ax2.legend()
ax2.grid()


plt.show()

    # # energy ratio
    # if total_energy > 0:
    #     ratio = band_energy / total_energy
    # else:
    #     ratio = 0

    # frequency_ratios.append(ratio)

#     #time
#     current_time = (tr.stats.starttime + i* window)

#     window_times.append(current_time.datetime)

# #convert to numpy arrays
# rms_values = np.array(rms_values)
# frequency_ratios = np.array(frequency_ratios)
# band_power = np.array(band_power)


# fig, (ax1, ax2) = plt.subplots(2,1,figsize=(14, 9),gridspec_kw={"height_ratios": [1, 2]})

# # waveformm

# ax1.plot(time, tr.data)

# ax1.set_xlabel("Time (seconds)")
# ax1.set_ylabel("Amplitude")

# ax1.set_title("Waveform")

# ax1.grid(True)

# # plt.tight_layout()
# # plt.show()

# # spectogram

# frequencies, times, Sxx = spectrogram(
#     tr.data,
#     fs=fs,
#     nperseg=1024,
#     noverlap=512
# )

# pcm = ax2.pcolormesh(
#     times, frequencies, 
#     10 * np.log10(Sxx + 1e-10),  # Convert power spectrum to dB
#     shading="gouraud", 
#     cmap="magma"
# )
# ax2.set_ylabel("Frequency (Hz)")
# ax2.set_xlabel("Time (seconds from start)")
# # ax2.set_ylim(0, fs / 2)  # Up to Nyquist frequency
# ax2.set_ylim(0, 20)
# # Add Colorbar for Power Spectral Density
# fig.colorbar(pcm, ax=ax2, label="Power (dB)")

# plt.tight_layout()
# plt.show()
time_minutes = np.arange(len(sta_lta_values))

plt.figure(figsize=(12, 5))

plt.plot(time_minutes, sta_lta_values)

plt.axhline(
    5,
    linestyle="--",
    label="STA/LTA threshold = 5"
)

plt.xlabel("Minutes since 02:00 UTC")
plt.ylabel("STA/LTA")
plt.title("STA/LTA over time")

plt.legend()
plt.grid()

plt.show()