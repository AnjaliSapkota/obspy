import numpy as np
import matplotlib.pyplot as plt

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import correlate, correlation_lags

# FDSN client
client = Client("EARTHSCOPE")

start = UTCDateTime("2026-08-26T02:30:00")
end   = UTCDateTime("2026-08-26T03:00:00")
# Processing settings
target_fs = 20.0
freqmin = 1.0
freqmax = 8.0

# Stations

st_kkn = client.get_waveforms(network="NK", station="KKN", location="*", channel="BHZ", starttime=start, endtime=end)
st_evn = client.get_waveforms(network="IO", station="EVN", location="*", channel="BHZ", starttime=start, endtime=end)
st_knset = client.get_waveforms(network="NQ", station="KNSET", location="01", channel="HNZ", starttime=start, endtime=end)


# Merge gaps or segments
# st_kkn.merge(method=1, fill_value="interpolate")
# st_evn.merge(method=1, fill_value="interpolate")
# st_knset.merge(method=1, fill_value="interpolate")

def preprocess(stream):
    stream = stream.copy()
    stream.merge( method=1,fill_value="interpolate")
    stream.detrend("linear")
    stream.detrend("demean")
    stream.taper(max_percentage=0.05,type="cosine")
    stream.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=True)

    # Resample all stations to the same sampling rate
    stream.interpolate(
        sampling_rate=target_fs,
        method="lanczos",
        a=12
    )

    return stream

st_kkn = preprocess(st_kkn)
st_evn = preprocess(st_evn)
st_knset = preprocess(st_knset)

def get_trace(stream):
    if len(stream) == 0:
        raise RuntimeError("No waveform found.")

    return stream[0]


# Get the first trace from each stream
tr_kkn = st_kkn[0]
tr_evn = st_evn[0]
tr_knset = st_knset[0]


# # Print basic information
# print("\nKKN")
# print(tr_kkn)

# print("\nEVN")
# print(tr_evn)

# print("\nKNSET")
# print(tr_knset)


# Find common time window
common_start = max(tr_kkn.stats.starttime,tr_evn.stats.starttime,tr_knset.stats.starttime)

common_end = min(tr_kkn.stats.endtime,tr_evn.stats.endtime,tr_knset.stats.endtime)

# print("\nCommon time window:")
# print("Start:", common_start)
# print("End  :", common_end)


# Trim all three to exactly the same time window
tr_kkn.trim(starttime=common_start,endtime=common_end)
tr_evn.trim(starttime=common_start,endtime=common_end)
tr_knset.trim(starttime=common_start,endtime=common_end)

# # Print final information
# print("\nAfter preprocessing and trimming:")

# print( "KKN:",
#     len(tr_kkn.data),
#     "samples,",
#     tr_kkn.stats.sampling_rate,
#     "Hz"
# )

# print(
#     "EVN:",
#     len(tr_evn.data),
#     "samples,",
#     tr_evn.stats.sampling_rate,
#     "Hz"
# )

# print(
#     "KNSET:",
#     len(tr_knset.data),
#     "samples,",
#     tr_knset.stats.sampling_rate,
#     "Hz"
# )

# # Cross-correlation between two stations
# signal_kkn = tr_kkn.data.astype(float)
# signal_evn = tr_evn.data.astype(float)

# # Remove mean
# signal_kkn = signal_kkn - np.mean(signal_kkn)
# signal_evn = signal_evn - np.mean(signal_evn)

# # Normalize
# signal_kkn = signal_kkn / np.linalg.norm(signal_kkn)
# signal_evn = signal_evn / np.linalg.norm(signal_evn)

# # Calculate cross-correlation
# correlation = correlate(
#     signal_evn,
#     signal_kkn,
#     mode="full"
# )

# # Calculate corresponding lags
# lags = correlation_lags(
#     len(signal_evn),
#     len(signal_kkn),
#     mode="full"
# )

# # Find maximum correlation
# max_index = np.argmax(correlation)

# best_lag_samples = lags[max_index]

# # Convert samples to seconds
# best_lag_seconds = best_lag_samples / target_fs

# print("\nCross-correlation result:")
# print("Maximum correlation:", correlation[max_index])
# print("Lag:", best_lag_samples, "samples")
# print("Time difference:", best_lag_seconds, "seconds")

def calculate_lag(trace_a, trace_b, sampling_rate):

    signal_a = trace_a.data.astype(float)
    signal_b = trace_b.data.astype(float)

    # Normalize
    signal_a = signal_a / np.linalg.norm(signal_a)
    signal_b = signal_b / np.linalg.norm(signal_b)

    # Cross-correlation
    correlation = correlate(
        signal_b,
        signal_a,
        mode="full"
    )

    # Corresponding lags
    lags = correlation_lags(
        len(signal_b),
        len(signal_a),
        mode="full"
    )

    # Find maximum correlation
    max_index = np.argmax(correlation)

    lag_samples = lags[max_index]
    lag_seconds = lag_samples / sampling_rate

    max_correlation = correlation[max_index]

    return lag_seconds, max_correlation


# Calculate pairwise time differences

lag_kkn_evn, cc_kkn_evn = calculate_lag(
    tr_kkn,
    tr_evn,
    target_fs
)

lag_kkn_knset, cc_kkn_knset = calculate_lag(
    tr_kkn,
    tr_knset,
    target_fs
)

lag_evn_knset, cc_evn_knset = calculate_lag(
    tr_evn,
    tr_knset,
    target_fs
)


# Print results

print("\nPairwise cross-correlation results")

print(f"KKN - EVN: " f"lag = {lag_kkn_evn:.3f} s, " f"CC = {cc_kkn_evn:.3f}")

print(f"KKN → KNSET: "f"lag = {lag_kkn_knset:.3f} s, "f"CC = {cc_kkn_knset:.3f}")

print(f"EVN → KNSET: "f"lag = {lag_evn_knset:.3f} s, "f"CC = {cc_evn_knset:.3f}")
# # Plot the three preprocessed signals
# plt.figure(figsize=(12, 8))

# plt.subplot(3, 1, 1)
# plt.plot(tr_kkn.times(), tr_kkn.data)
# plt.ylabel("KKN")
# plt.grid()

# plt.subplot(3, 1, 2)
# plt.plot(tr_evn.times(), tr_evn.data)
# plt.ylabel("EVN")
# plt.grid()

# plt.subplot(3, 1, 3)
# plt.plot(tr_knset.times(), tr_knset.data)
# plt.ylabel("KNSET")
# plt.xlabel("Time (seconds)")
# plt.grid()

# plt.suptitle("Preprocessed Seismic Waveforms")

# plt.tight_layout()
# plt.show()
