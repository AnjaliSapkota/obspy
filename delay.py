import numpy as np
import matplotlib.pyplot as plt

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import butter, correlate, correlation_lags, filtfilt, hilbert, windows

# FDSN client
client = Client("https://seiscomp.alertnepal.online")

# Time window
start = UTCDateTime("2026-08-26T02:52:00")
end = UTCDateTime("2026-08-26T02:55:00")

# Processing settings
target_fs = 20.0
freqmin = 1.0
freqmax = 8.0
max_shift_seconds = 30.0

# Fetch waveforms
print("Fetching waveforms")

st_kkn = client.get_waveforms(network="NK",station="KKN",location="*",channel="BHZ",starttime=start,endtime=end)

st_evn = client.get_waveforms(network="IO",station="EVN",location="*",channel="BHZ",starttime=start,endtime=end)

st_eqm10 = client.get_waveforms(network="NP",station="EQM10",location="*",channel="EHZ",starttime=start,endtime=end)

# Preprocess function
def preprocess(stream, is_acceleration=False):
    stream = stream.copy()

    stream.merge(method=1,fill_value="interpolate")
    stream.detrend("linear")
    stream.detrend("demean")
    stream.taper(max_percentage=0.05,type="hann")
    stream.interpolate(sampling_rate=target_fs,method="linear")
    stream.filter("bandpass",freqmin=freqmin,freqmax=freqmax,corners=4,zerophase=True)
    return stream

st_kkn = preprocess(st_kkn)
st_evn = preprocess(st_evn)
st_eqm10 = preprocess(st_eqm10)

# Check traces
if len(st_kkn) == 0:
    raise RuntimeError("No processed KKN waveform found.")

if len(st_evn) == 0:
    raise RuntimeError("No processed EVN waveform found.")

if len(st_eqm10) == 0:
    raise RuntimeError("No processed eqm10 waveform found.")

# Get first trace
tr_kkn = st_kkn[0]
tr_evn = st_evn[0]
tr_eqm10 = st_eqm10[0]

# # Print information
# print("\nProcessed traces:")
# print("\nKKN\n", tr_kkn)
# print("\nEVN\n", tr_evn)
# print("\neqm10\n", tr_eqm10)

# Find common time window
common_start = max(
    tr_kkn.stats.starttime,
    tr_evn.stats.starttime,
    tr_eqm10.stats.starttime
)

common_end = min(
    tr_kkn.stats.endtime,
    tr_evn.stats.endtime,
    tr_eqm10.stats.endtime
)

# print("\nCommon time window:")
# print("Start:", common_start)
# print("End  :", common_end)

# Trim to common window
tr_kkn.trim(starttime=common_start, endtime=common_end)
tr_evn.trim(starttime=common_start, endtime=common_end)
tr_eqm10.trim(starttime=common_start, endtime=common_end)

# # Check sampling rates
# print("\nSampling rates:")
# print("KKN   :", tr_kkn.stats.sampling_rate)
# print("EVN   :", tr_evn.stats.sampling_rate)
# print("eqm10 :", tr_eqm10.stats.sampling_rate)


# Hilbert envelope computation with optional smoothing
def get_hilbert_envelope(trace, smooth_cutoff=0.5):
    signal = trace.data.astype(float)
    signal = signal - np.mean(signal)
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)

    # Low-pass filter envelope to remove high-frequency noise spikes
    if smooth_cutoff is not None:
        fs = trace.stats.sampling_rate
        b, a = butter(N=2, Wn=smooth_cutoff / (0.5 * fs), btype="low")
        envelope = filtfilt(b, a, envelope)
        envelope = np.maximum(envelope, 0)

    return envelope


# Calculate envelopes
print("\nCalculating Hilbert envelopes")
env_kkn = get_hilbert_envelope(tr_kkn, smooth_cutoff=0.5)
env_evn = get_hilbert_envelope(tr_evn, smooth_cutoff=0.5)
env_eqm10 = get_hilbert_envelope(tr_eqm10, smooth_cutoff=0.5)

# Calculate envelope lag
def calculate_envelope_lag(
    envelope_a,
    envelope_b,
    sampling_rate,
    max_shift_seconds
):
    # Make same length
    min_len = min(len(envelope_a), len(envelope_b))
    envelope_a = envelope_a[:min_len]
    envelope_b = envelope_b[:min_len]

    # Remove 5% from both edges
    crop = int(0.05 * len(envelope_a))
    envelope_a = envelope_a[crop:-crop]
    envelope_b = envelope_b[crop:-crop]

    # Remove envelope mean
    envelope_a = envelope_a - np.mean(envelope_a)
    envelope_b = envelope_b - np.mean(envelope_b)

    # Tukey taper
    taper = windows.tukey(len(envelope_a), alpha=0.05)
    a = envelope_a * taper
    b = envelope_b * taper

    # Cross-correlation
    correlation = correlate(a, b, mode="full")

    # Corresponding lags
    lags = correlation_lags(len(a), len(b), mode="full")

    # Normalize
    norm_factor = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if norm_factor == 0:
        raise RuntimeError("Envelope has zero energy.")

    correlation = correlation / norm_factor

    # Convert lags to seconds
    lag_seconds = lags / sampling_rate

    # Limit lag search
    max_shift_samples = int(round(max_shift_seconds * sampling_rate))

    valid = (lags >= -max_shift_samples) & (lags <= max_shift_samples)
    valid_corr = correlation[valid]
    valid_lags = lags[valid]

    # Find maximum correlation
    peak_index = np.argmax(valid_corr)
    lag_samples = valid_lags[peak_index]
    lag_seconds_value = lag_samples / sampling_rate
    max_correlation = valid_corr[peak_index]

    return (
        lag_seconds_value,
        max_correlation,
        correlation,
        lag_seconds
    )


# Pairwise TDOA
lag_kkn_evn, cc_kkn_evn, corr_kkn_evn, lags_kkn_evn = calculate_envelope_lag(
    env_kkn, env_evn, target_fs, max_shift_seconds
)

lag_kkn_eqm10, cc_kkn_eqm10, corr_kkn_eqm10, lags_kkn_eqm10 = calculate_envelope_lag(
    env_kkn, env_eqm10, target_fs, max_shift_seconds
)

lag_evn_eqm10, cc_evn_eqm10, corr_evn_eqm10, lags_evn_eqm10 = calculate_envelope_lag(
    env_evn, env_eqm10, target_fs, max_shift_seconds
)

# TDOA closure
closure_error = lag_kkn_eqm10 - (lag_kkn_evn + lag_evn_eqm10)

# Print results
print("\nHilbert Envelope TDOA results:")
print(f"KKN - EVN:   lag = {lag_kkn_evn:+.3f} s, CC = {cc_kkn_evn:.3f}")
print(f"KKN - eqm10: lag = {lag_kkn_eqm10:+.3f} s, CC = {cc_kkn_eqm10:.3f}")
print(f"EVN - eqm10: lag = {lag_evn_eqm10:+.3f} s, CC = {cc_evn_eqm10:.3f}")
print(f"\nTDOA closure error: {closure_error:+.3f} s")
