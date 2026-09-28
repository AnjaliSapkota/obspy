import numpy as np
import matplotlib.pyplot as plt

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import correlate, correlation_lags


# FDSN client
client = Client("EARTHSCOPE")


# Time window
start = UTCDateTime("2026-08-26T02:52:00")
end = UTCDateTime("2026-08-26T02:55:00")


# Processing settings
target_fs = 20.0
freqmin = 1.0
freqmax = 6.0

pre_filt = (0.1, 0.2, 8.0, 10.0)


# Fetch waveforms
print("Fetching waveforms...")

st_kkn = client.get_waveforms(
    network="NK",
    station="KKN",
    location="*",
    channel="BHZ",
    starttime=start,
    endtime=end
)

st_evn = client.get_waveforms(
    network="IO",
    station="EVN",
    location="*",
    channel="BHZ",
    starttime=start,
    endtime=end
)

st_knset = client.get_waveforms(
    network="NQ",
    station="KNSET",
    location="01",
    channel="HNZ",
    starttime=start,
    endtime=end
)


# Get response metadata
print("Fetching instrument responses...")

inv_kkn = client.get_stations(
    network="NK",
    station="KKN",
    location="*",
    channel="BHZ",
    starttime=start,
    endtime=end,
    level="response"
)

inv_evn = client.get_stations(
    network="IO",
    station="EVN",
    location="*",
    channel="BHZ",
    starttime=start,
    endtime=end,
    level="response"
)

inv_knset = client.get_stations(
    network="NQ",
    station="KNSET",
    location="01",
    channel="HNZ",
    starttime=start,
    endtime=end,
    level="response"
)


# Preprocess KKN
def preprocess_velocity(stream, inventory):
    stream = stream.copy()

    stream.merge(
        method=1,
        fill_value="interpolate"
    )

    stream.detrend("linear")
    stream.detrend("demean")

    stream.taper(
        max_percentage=0.05,
        type="cosine"
    )

    print("Removing instrument response → VELOCITY")

    stream.remove_response(
        inventory=inventory,
        output="VEL",
        pre_filt=pre_filt,
        zero_mean=True,
        taper=True
    )

    stream.filter(
        "bandpass",
        freqmin=freqmin,
        freqmax=freqmax,
        zerophase=True
    )

    stream.interpolate(
        sampling_rate=target_fs,
        method="weighted_average_slopes"
    )

    return stream


# Preprocess KNSET
def preprocess_acceleration_to_velocity(stream, inventory):
    stream = stream.copy()

    stream.merge(
        method=1,
        fill_value="interpolate"
    )

    stream.detrend("linear")
    stream.detrend("demean")

    stream.taper(
        max_percentage=0.05,
        type="cosine"
    )

    print("Removing instrument response → ACCELERATION")

    stream.remove_response(
        inventory=inventory,
        output="ACC",
        pre_filt=pre_filt,
        zero_mean=True,
        taper=True
    )

    # Convert acceleration to velocity
    for tr in stream:
        print("Integrating KNSET acceleration → velocity")
        tr.integrate()

    stream.detrend("linear")
    stream.detrend("demean")

    stream.filter(
        "bandpass",
        freqmin=freqmin,
        freqmax=freqmax,
        zerophase=True
    )

    stream.interpolate(
        sampling_rate=target_fs,
        method="weighted_average_slopes"
    )

    return stream


# Process all three stations
print("\nProcessing KKN...")
st_kkn = preprocess_velocity(
    st_kkn,
    inv_kkn
)

print("\nProcessing EVN...")
st_evn = preprocess_velocity(
    st_evn,
    inv_evn
)

print("\nProcessing KNSET...")
st_knset = preprocess_acceleration_to_velocity(
    st_knset,
    inv_knset
)


# Check that traces exist
if len(st_kkn) == 0:
    raise RuntimeError("No processed KKN waveform found.")

if len(st_evn) == 0:
    raise RuntimeError("No processed EVN waveform found.")

if len(st_knset) == 0:
    raise RuntimeError("No processed KNSET waveform found.")


# Get first trace
tr_kkn = st_kkn[0]
tr_evn = st_evn[0]
tr_knset = st_knset[0]


# Print waveform information
print("\nProcessed traces:")

print("\nKKN")
print(tr_kkn)

print("\nEVN")
print(tr_evn)

print("\nKNSET")
print(tr_knset)


# Find common time window
common_start = max(
    tr_kkn.stats.starttime,
    tr_evn.stats.starttime,
    tr_knset.stats.starttime
)

common_end = min(
    tr_kkn.stats.endtime,
    tr_evn.stats.endtime,
    tr_knset.stats.endtime
)


print("\nCommon time window:")
print("Start:", common_start)
print("End  :", common_end)


# Trim all three to exactly the same time window
tr_kkn.trim(
    starttime=common_start,
    endtime=common_end
)

tr_evn.trim(
    starttime=common_start,
    endtime=common_end
)

tr_knset.trim(
    starttime=common_start,
    endtime=common_end
)


# Check sampling rates
print("\nSampling rates:")
print("KKN   :", tr_kkn.stats.sampling_rate)
print("EVN   :", tr_evn.stats.sampling_rate)
print("KNSET :", tr_knset.stats.sampling_rate)


# Cross-correlation function
def calculate_lag(trace_a, trace_b, sampling_rate):

    signal_a = trace_a.data.astype(float)
    signal_b = trace_b.data.astype(float)

    # Remove remaining mean
    signal_a = signal_a - np.mean(signal_a)
    signal_b = signal_b - np.mean(signal_b)

    # Normalize
    norm_a = np.linalg.norm(signal_a)
    norm_b = np.linalg.norm(signal_b)

    if norm_a == 0:
        raise RuntimeError("Trace A has zero norm.")

    if norm_b == 0:
        raise RuntimeError("Trace B has zero norm.")

    signal_a = signal_a / norm_a
    signal_b = signal_b / norm_b

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

    # Maximum correlation
    max_index = np.argmax(correlation)

    lag_samples = lags[max_index]

    lag_seconds = lag_samples / sampling_rate

    max_correlation = correlation[max_index]

    return (
        lag_seconds,
        max_correlation,
        correlation,
        lags
    )


# Calculate pairwise TDOA
lag_kkn_evn, cc_kkn_evn, corr_kkn_evn, lags_kkn_evn = calculate_lag(
    tr_kkn,
    tr_evn,
    target_fs
)

lag_kkn_knset, cc_kkn_knset, corr_kkn_knset, lags_kkn_knset = calculate_lag(
    tr_kkn,
    tr_knset,
    target_fs
)

lag_evn_knset, cc_evn_knset, corr_evn_knset, lags_evn_knset = calculate_lag(
    tr_evn,
    tr_knset,
    target_fs
)


# Convert lag arrays to seconds
lags_kkn_evn = lags_kkn_evn / target_fs
lags_kkn_knset = lags_kkn_knset / target_fs
lags_evn_knset = lags_evn_knset / target_fs


# TDOA closure
closure_error = (
    lag_kkn_knset
    - (lag_kkn_evn + lag_evn_knset)
)


# Print results
print("\nTDOA results:")

print(
    f"KKN → EVN:   "
    f"lag = {lag_kkn_evn:+.3f} s, "
    f"CC = {cc_kkn_evn:.3f}"
)

print(
    f"KKN → KNSET: "
    f"lag = {lag_kkn_knset:+.3f} s, "
    f"CC = {cc_kkn_knset:.3f}"
)

print(
    f"EVN → KNSET: "
    f"lag = {lag_evn_knset:+.3f} s, "
    f"CC = {cc_evn_knset:.3f}"
)

print(
    f"\nTDOA closure error: "
    f"{closure_error:+.3f} s"
)


# Plot processed waveforms
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(
    tr_kkn.times(),
    tr_kkn.data
)
plt.ylabel("KKN")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(
    tr_evn.times(),
    tr_evn.data
)
plt.ylabel("EVN")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(
    tr_knset.times(),
    tr_knset.data
)
plt.ylabel("KNSET")
plt.xlabel("Time (seconds)")
plt.grid()

plt.suptitle(
    "Response-Corrected Velocity Waveforms"
)

plt.tight_layout()
plt.show()


# Plot cross-correlations
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(
    lags_kkn_evn,
    corr_kkn_evn
)
plt.axvline(0, linestyle="--")
plt.xlabel("Lag (s)")
plt.ylabel("Correlation")
plt.title(
    f"KKN → EVN | Lag = {lag_kkn_evn:+.3f} s | CC = {cc_kkn_evn:.3f}"
)
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(
    lags_kkn_knset,
    corr_kkn_knset
)
plt.axvline(0, linestyle="--")
plt.xlabel("Lag (s)")
plt.ylabel("Correlation")
plt.title(
    f"KKN → KNSET | Lag = {lag_kkn_knset:+.3f} s | CC = {cc_kkn_knset:.3f}"
)
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(
    lags_evn_knset,
    corr_evn_knset
)
plt.axvline(0, linestyle="--")
plt.xlabel("Lag (s)")
plt.ylabel("Correlation")
plt.title(
    f"EVN → KNSET | Lag = {lag_evn_knset:+.3f} s | CC = {cc_evn_knset:.3f}"
)
plt.grid()

plt.suptitle("Pairwise Cross-Correlation")

plt.tight_layout()
plt.show()