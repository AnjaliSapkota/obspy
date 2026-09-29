import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import butter, correlate, correlation_lags, filtfilt, hilbert, windows
from obspy.geodetics import gps2dist_azimuth
from scipy.optimize import least_squares

# FDSN client
client = Client("EARTHSCOPE")

# Time window
start = UTCDateTime("2026-08-26T02:52:00")
end = UTCDateTime("2026-08-26T02:56:00")

# Processing settings
target_fs = 20.0
freqmin = 1.0
freqmax = 8.0
max_shift_seconds = 30.0
velocity = 2.5

stations = {
    "KKN": {"lat": 27.8000, "lon": 85.2790, "net": "NK", "chan": "BHZ", "loc": "*", "is_acc": False},
    "EVN": {"lat": 27.95865, "lon": 86.811653, "net": "IO", "chan": "BHZ", "loc": "*", "is_acc": False},
    "KNSET": {"lat": 27.65337, "lon": 85.302528, "net": "NQ", "chan": "HNZ", "loc": "01", "is_acc": True}
}

# Fetch waveforms
print("Fetching waveforms")
st_kkn = client.get_waveforms(
    network=stations["KKN"]["net"], station="KKN", location=stations["KKN"]["loc"], 
    channel=stations["KKN"]["chan"], starttime=start, endtime=end
)

st_evn = client.get_waveforms(
    network=stations["EVN"]["net"], station="EVN", location=stations["EVN"]["loc"], 
    channel=stations["EVN"]["chan"], starttime=start, endtime=end
)

st_knset = client.get_waveforms(
    network=stations["KNSET"]["net"], station="KNSET", location=stations["KNSET"]["loc"], 
    channel=stations["KNSET"]["chan"], starttime=start, endtime=end
)

# Preprocess function
def preprocess(stream, is_acceleration=False):
    stream = stream.copy()
    stream.merge(method=1, fill_value="interpolate")
    stream.detrend("linear")
    stream.detrend("demean")
    stream.taper(max_percentage=0.05, type="hann")
    # # Convert acceleration (m/s^2) to velocity (m/s) if needed
    # if is_acceleration:
    #     stream.integrate(method="cumtrapz")
    #     # High-pass filter immediately after to suppress integration drift
    #     stream.filter("highpass", freq=0.1)
    #     stream.detrend("linear")

    stream.interpolate(sampling_rate=target_fs, method="linear")
    stream.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)
    
    return stream


# Process stations
# print("\nProcessing KKN")
# st_kkn = preprocess(st_kkn, is_acceleration=False)

# print("Processing EVN...")
# st_evn = preprocess(st_evn, is_acceleration=False)

# print("Processing KNSET...")
# st_knset = preprocess(st_knset, is_acceleration=True)

print("\nProcessing KKN")
st_kkn = preprocess(st_kkn)

print("Processing EVN")
st_evn = preprocess(st_evn)

print("Processing KNSET")
st_knset = preprocess(st_knset)

# Check traces
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

# # Print information
# print("\nProcessed traces:")
# print("\nKKN\n", tr_kkn)
# print("\nEVN\n", tr_evn)
# print("\nKNSET\n", tr_knset)

# Trim to common window
common_start = max(tr_kkn.stats.starttime, tr_evn.stats.starttime, tr_knset.stats.starttime)
common_end = min(tr_kkn.stats.endtime, tr_evn.stats.endtime, tr_knset.stats.endtime)

# print("\nCommon time window:")
# print("Start:", common_start)
# print("End  :", common_end)

# Trim to common window
tr_kkn.trim(starttime=common_start, endtime=common_end)
tr_evn.trim(starttime=common_start, endtime=common_end)
tr_knset.trim(starttime=common_start, endtime=common_end)

# Check sampling rates
print("\nSampling rates:")
print("KKN   :", tr_kkn.stats.sampling_rate)
print("EVN   :", tr_evn.stats.sampling_rate)
print("KNSET :", tr_knset.stats.sampling_rate)


# Hilbert envelope computation with optional smoothing
def get_hilbert_envelope(trace, smooth_cutoff=0.5):
    signal = trace.data.astype(float)
    signal = signal - np.mean(signal)

    # Hilbert transform
    analytic_signal = hilbert(signal)

    # Envelope
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
env_knset = get_hilbert_envelope(tr_knset, smooth_cutoff=0.5)


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

lag_kkn_knset, cc_kkn_knset, corr_kkn_knset, lags_kkn_knset = calculate_envelope_lag(
    env_kkn, env_knset, target_fs, max_shift_seconds
)

lag_evn_knset, cc_evn_knset, corr_evn_knset, lags_evn_knset = calculate_envelope_lag(
    env_evn, env_knset, target_fs, max_shift_seconds
)

# TDOA closure
closure_error = lag_kkn_knset - (lag_kkn_evn + lag_evn_knset)

# Print results
print("\nHilbert Envelope TDOA results:")
print(f"KKN -> EVN:   lag = {lag_kkn_evn:+.3f} s, CC = {cc_kkn_evn:.3f}")
print(f"KKN -> KNSET: lag = {lag_kkn_knset:+.3f} s, CC = {cc_kkn_knset:.3f}")
print(f"EVN -> KNSET: lag = {lag_evn_knset:+.3f} s, CC = {cc_evn_knset:.3f}")
print(f"\nTDOA closure error: {closure_error:+.3f} s")

# Use KKN as the local coordinate origin
origin_lat = stations["KKN"]["lat"]
origin_lon = stations["KKN"]["lon"]

# Convert latitude/longitude to local Cartesian coordinates
def latlon_to_xy(lat, lon):
    distance, azimuth, _ = gps2dist_azimuth(origin_lat, origin_lon, lat, lon)
    azimuth_rad = np.radians(azimuth)
    x = (distance / 1000.0) * np.sin(azimuth_rad)
    y = (distance / 1000.0) * np.cos(azimuth_rad)
    return x, y


station_xy = {name: latlon_to_xy(info["lat"], info["lon"]) for name, info in stations.items()}

delta_kkn_evn = velocity * (-lag_kkn_evn)
delta_kkn_knset = velocity * (-lag_kkn_knset)
delta_evn_knset = velocity * (-lag_evn_knset)

# Source Inversion (Least Squares)

def tdoa_residuals(xy):
    x, y = xy
    x_kkn, y_kkn = station_xy["KKN"]
    x_evn, y_evn = station_xy["EVN"]
    x_knset, y_knset = station_xy["KNSET"]

    d_kkn = np.sqrt((x - x_kkn) ** 2 + (y - y_kkn) ** 2)
    d_evn = np.sqrt((x - x_evn) ** 2 + (y - y_evn) ** 2)
    d_knset = np.sqrt((x - x_knset) ** 2 + (y - y_knset) ** 2)

    res_1 = (d_evn - d_kkn) - delta_kkn_evn
    res_2 = (d_knset - d_kkn) - delta_kkn_knset
    res_3 = (d_knset - d_evn) - delta_evn_knset

    return [res_1, res_2, res_3]

initial_guess = [0.0, 0.0]
opt_res = least_squares(tdoa_residuals, initial_guess)
x_src, y_src = opt_res.x

print(f"Inverted Source Location:")
print(f"X (East) : {x_src:+.3f} km from KKN")
print(f"Y (North): {y_src:+.3f} km from KKN")
print(f"Residual norm: {opt_res.cost:.4f}")

# Hyperbola intersection plot

# Get station coordinates
x_kkn, y_kkn = station_xy["KKN"]
x_evn, y_evn = station_xy["EVN"]
x_knset, y_knset = station_xy["KNSET"]


# Create plotting region
margin = 150.0
resolution = 0.5

xmin = min(x_kkn, x_evn, x_knset) - margin
xmax = max(x_kkn, x_evn, x_knset) + margin

ymin = min(y_kkn, y_evn, y_knset) - margin
ymax = max(y_kkn, y_evn, y_knset) + margin

x = np.arange(xmin, xmax, resolution)
y = np.arange(ymin, ymax, resolution)

X, Y = np.meshgrid(x, y)


# Distance from every grid point to each station
D_KKN = np.sqrt(
    (X - x_kkn) ** 2 +
    (Y - y_kkn) ** 2
)

D_EVN = np.sqrt(
    (X - x_evn) ** 2 +
    (Y - y_evn) ** 2
)

D_KNSET = np.sqrt(
    (X - x_knset) ** 2 +
    (Y - y_knset) ** 2
)


# Hyperbola equations
#
# H = 0 represents the hyperbola.
#
# KKN -> EVN
# EVN arrives 23.950 s after KKN:
#
# d(EVN) - d(KKN) = velocity * 23.950
#
H_KKN_EVN = (
    D_EVN
    - D_KKN
    - delta_kkn_evn
)


# KKN -> KNSET
#
# KNSET arrives 6.750 s after KKN:
#
# d(KNSET) - d(KKN) = velocity * 6.750
#
H_KKN_KNSET = (
    D_KNSET
    - D_KKN
    - delta_kkn_knset
)


# EVN -> KNSET
#
# KNSET arrives 13.200 s before EVN:
#
# d(KNSET) - d(EVN) = velocity * (-13.200)
#
H_EVN_KNSET = (
    D_KNSET
    - D_EVN
    - delta_evn_knset
)


# Plot hyperbolas
plt.figure(figsize=(12, 10))

plt.contour(
    X,
    Y,
    H_KKN_EVN,
    levels=[0],
    colors="blue",
    linewidths=2.5
)

plt.contour(
    X,
    Y,
    H_KKN_KNSET,
    levels=[0],
    colors="red",
    linewidths=2.5
)

plt.contour(
    X,
    Y,
    H_EVN_KNSET,
    levels=[0],
    colors="green",
    linewidths=2.5
)


# Plot stations
plt.scatter(
    x_kkn,
    y_kkn,
    marker="^",
    s=160,
    zorder=5
)

plt.scatter(
    x_evn,
    y_evn,
    marker="^",
    s=160,
    zorder=5
)

plt.scatter(
    x_knset,
    y_knset,
    marker="^",
    s=160,
    zorder=5
)


# Station labels
plt.text(
    x_kkn + 5,
    y_kkn + 5,
    "KKN",
    fontsize=12,
    fontweight="bold"
)

plt.text(
    x_evn + 5,
    y_evn + 5,
    "EVN",
    fontsize=12,
    fontweight="bold"
)

plt.text(
    x_knset + 5,
    y_knset + 5,
    "KNSET",
    fontsize=12,
    fontweight="bold"
)


# Plot least-squares source
plt.scatter(
    x_src,
    y_src,
    marker="*",
    s=300,
    zorder=10,
    label="TDOA inverted source"
)


# Legend
legend_lines = [
    Line2D(
        [0],
        [0],
        color="blue",
        linewidth=2.5,
        label="KKN–EVN hyperbola"
    ),
    Line2D(
        [0],
        [0],
        color="red",
        linewidth=2.5,
        label="KKN–KNSET hyperbola"
    ),
    Line2D(
        [0],
        [0],
        color="green",
        linewidth=2.5,
        label="EVN–KNSET hyperbola"
    ),
    Line2D(
        [0],
        [0],
        marker="^",
        color="black",
        linestyle="None",
        markersize=10,
        label="Station"
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        color="black",
        linestyle="None",
        markersize=15,
        label="Least-squares source"
    )
]

plt.legend(
    handles=legend_lines,
    loc="best"
)


plt.xlabel(
    "East-West distance from KKN (km)"
)

plt.ylabel(
    "North-South distance from KKN (km)"
)

plt.title(
    "Three-Station TDOA Hyperbola Intersection\n"
    f"Velocity = {velocity:.2f} km/s"
)

plt.grid(True)

plt.axis("equal")

plt.tight_layout()

plt.show()

# # Plot Hilbert envelopes
# plt.figure(figsize=(12, 8))

# plt.subplot(3, 1, 1)
# plt.plot(tr_kkn.times(), env_kkn)
# plt.ylabel("KKN")
# plt.title("KKN Hilbert Envelope")
# plt.grid()

# plt.subplot(3, 1, 2)
# plt.plot(tr_evn.times(), env_evn)
# plt.ylabel("EVN")
# plt.title("EVN Hilbert Envelope")
# plt.grid()

# plt.subplot(3, 1, 3)
# plt.plot(tr_knset.times(), env_knset)
# plt.ylabel("KNSET")
# plt.xlabel("Time (seconds)")
# plt.title("KNSET Hilbert Envelope")
# plt.grid()

# plt.suptitle("Hilbert Envelopes")
# plt.tight_layout()
# plt.show()

# # Plot cross-correlations
# plt.figure(figsize=(12, 8))

# plt.subplot(3, 1, 1)
# plt.plot(lags_kkn_evn, corr_kkn_evn)
# plt.axvline(lag_kkn_evn, linestyle="--", label=f"Peak = {lag_kkn_evn:+.3f}s")
# plt.xlabel("Lag (s)")
# plt.ylabel("Envelope CC")
# plt.title(f"KKN -> EVN | Lag = {lag_kkn_evn:+.3f} s | CC = {cc_kkn_evn:.3f}")
# plt.legend()
# plt.grid()

# plt.subplot(3, 1, 2)
# plt.plot(lags_kkn_knset, corr_kkn_knset)
# plt.axvline(lag_kkn_knset, linestyle="--", label=f"Peak = {lag_kkn_knset:+.3f}s")
# plt.xlabel("Lag (s)")
# plt.ylabel("Envelope CC")
# plt.title(f"KKN -> KNSET | Lag = {lag_kkn_knset:+.3f} s | CC = {cc_kkn_knset:.3f}")
# plt.legend()
# plt.grid()

# plt.subplot(3, 1, 3)
# plt.plot(lags_evn_knset, corr_evn_knset)
# plt.axvline(lag_evn_knset, linestyle="--", label=f"Peak = {lag_evn_knset:+.3f}s")
# plt.xlabel("Lag (s)")
# plt.ylabel("Envelope CC")
# plt.title(f"EVN -> KNSET | Lag = {lag_evn_knset:+.3f} s | CC = {cc_evn_knset:.3f}")
# plt.legend()
# plt.grid()

# plt.suptitle("Pairwise Hilbert-Envelope Cross-Correlation")
# plt.tight_layout()
# plt.show()