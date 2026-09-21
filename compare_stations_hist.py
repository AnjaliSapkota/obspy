import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.cross_correlation import correlate, xcorr_max
import matplotlib.dates as mdates

client = Client("https://seiscomp.alertnepal.online")

start = UTCDateTime("2026-08-26T02:52:30")
end   = UTCDateTime("2026-08-26T02:54:00")


st_a = client.get_waveforms(network="NK", station="KKN", location="*", channel="BHZ", starttime=start, endtime=end)
st_b = client.get_waveforms(network="IO", station="EVN", location="*", channel="BHZ", starttime=start, endtime=end)

# st_a.merge(fill_value="latest")
# st_b.merge(fill_value="latest")

# Get the first trace
tr_a = st_a[0]
tr_b = st_b[0]

# print("\nStation A:")
# print(tr_a)

# print("\nStation B:")
# print(tr_b)

fs_a = tr_a.stats.sampling_rate
fs_b = tr_b.stats.sampling_rate

# shift_seconds = 13.74
# tr_b.stats.starttime -= shift_seconds

if fs_a != fs_b:
    tr_a.resample(fs_b)

# print("\nStation A:")
# print(tr_a)

# print("\nStation B:")
# print(tr_b)

common_start = max(tr_a.stats.starttime,tr_b.stats.starttime)

common_end = min(tr_a.stats.endtime,tr_b.stats.endtime)

tr_a = tr_a.slice(common_start, common_end)
tr_b = tr_b.slice(common_start, common_end)

# # Remove mean
# tr_a.detrend("demean")
# tr_b.detrend("demean")

# # Remove linear trend
# tr_a.detrend("linear")
# tr_b.detrend("linear")

# Bandpass filter
tr_a.filter("bandpass",freqmin=0.5,freqmax=8.0)

tr_b.filter("bandpass",freqmin=0.5,freqmax=8.0)

fs = tr_a.stats.sampling_rate

# print("\nFinal sampling rate:", fs, "Hz")


time_a = tr_a.times("matplotlib")
time_b = tr_b.times("matplotlib")

# plt.figure(figsize=(14, 7))

# plt.subplot(2, 1, 1)

# plt.plot(
#     time_a,
#     tr_a.data,
# )

# plt.ylabel("Amplitude")
# plt.title("Station A: NK.KKN.*.BHZ")
# # plt.legend()
# plt.grid(True)

# plt.gca().xaxis.set_major_formatter(
#     mdates.DateFormatter("%H:%M:%S")
# )

# plt.subplot(2, 1, 2)

# plt.plot(
#     time_b,
#     tr_b.data,
# )

# plt.xlabel("Time (seconds)")
# plt.ylabel("Amplitude")
# plt.title("Station B: IO.EVN.*.HHZ")
# # plt.legend()
# plt.grid(True)

# plt.gca().xaxis.set_major_formatter(
#     mdates.DateFormatter("%H:%M:%S")
# )

# plt.tight_layout()
# plt.show()



max_shift_seconds = 20

max_shift_samples = int(max_shift_seconds * fs)

print("\nMaximum correlation shift:",
      max_shift_seconds,
      "seconds")


corr = correlate(
    tr_a.data,
    tr_b.data,
    shift=max_shift_samples
)


lag_samples, coefficient = xcorr_max(corr)


# Convert samples → seconds

lag_seconds = lag_samples / fs


print("Lag in samples:", lag_samples)

print("Lag in seconds:", lag_seconds)

print("Correlation coefficient:", coefficient)


lags_seconds = np.arange(
    -max_shift_samples,
    max_shift_samples + 1
) / fs


# # Correlation Curve Visualization
# plt.figure(figsize=(12, 5))
# plt.plot(lags_seconds, corr, color='navy', label="Cross-correlation Spectrum")

# plt.axvline(lag_seconds, color='red', linestyle='--', 
#             label=f"Max Correlation at {lag_seconds:.2f} s (Coeff: {coefficient:.2f})")
# plt.axhline(0, color='gray', linestyle=':')

# plt.xlabel("Lag / Shift Range (Seconds)")
# plt.ylabel("Normalized Correlation Coefficient")
# plt.title("Waveform Cross-Correlation Spectrum (KKN vs EVN)")
# plt.grid(True)
# plt.legend(loc="upper right")
# plt.tight_layout()
# plt.show()