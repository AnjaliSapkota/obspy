import base64
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import butter, correlate, correlation_lags, filtfilt, hilbert, windows
from pyproj import Transformer

# FDSN client
client = Client("https://seiscomp.alertnepal.online")

# UTM Projection for Nepal (Zone 45N, EPSG:32645)
transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
transformer_to_wgs = Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True)


def latlon_to_xy(origin_lat, origin_lon, lat, lon):
    origin_x, origin_y = transformer_to_utm.transform(origin_lon, origin_lat)
    target_x, target_y = transformer_to_utm.transform(lon, lat)
    return (target_x - origin_x) / 1000.0, (target_y - origin_y) / 1000.0


def fetch_waveform(station, start, end, target_fs, freqmin, freqmax):
    stream = client.get_waveforms(
        network=station["net"],
        station=station["sta"],
        location=station["loc"],
        channel=station["cha"],
        starttime=start,
        endtime=end,
    )

    stream.merge(method=1, fill_value="interpolate")
    stream.detrend("linear")
    stream.detrend("demean")
    stream.taper(max_percentage=0.05, type="hann")
    stream.interpolate(sampling_rate=target_fs, method="linear")
    stream.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)

    if len(stream) == 0:
        raise RuntimeError(f"No processed {station['sta']} waveform found.")

    return stream[0]


def trim_to_common_window(traces):
    common_start = max(trace.stats.starttime for trace in traces.values())
    common_end = min(trace.stats.endtime for trace in traces.values())
    if common_start >= common_end:
        raise RuntimeError("No common time window exists between stations.")
    for trace in traces.values():
        trace.trim(starttime=common_start, endtime=common_end)


def image_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def create_waveform_plot(traces):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, (name, trace) in zip(axes, traces.items()):
        sampling_rate = trace.stats.sampling_rate
        data = trace.data.astype(float)
        time = np.arange(len(data)) / sampling_rate
        ax.plot(time, data, linewidth=0.8)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time since common start (seconds)")
    fig.suptitle("Three-Station Seismic Waveforms")
    fig.tight_layout()
    return image_to_base64(fig)


def create_spectrogram_plot(traces, fmin=1.0, fmax=8.0):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, (name, trace) in zip(axes, traces.items()):
        sampling_rate = trace.stats.sampling_rate
        data = trace.data.astype(float)
        nperseg = min(512, len(data))

        if nperseg < 2:
            raise RuntimeError(f"Not enough samples for spectrogram at {name}.")

        noverlap = int(nperseg * 0.75)
        ax.specgram(data, NFFT=nperseg, Fs=sampling_rate, noverlap=noverlap)
        ax.set_ylabel(f"{name}\nFrequency (Hz)")
        ax.set_ylim(fmin, fmax)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time since common start (seconds)")
    fig.suptitle(f"Three-Station Frequency Spectrogram ({fmin:g}–{fmax:g} Hz)")
    fig.tight_layout()
    return image_to_base64(fig)


def get_hilbert_envelope(trace, smooth_cutoff=0.5):
    signal = trace.data.astype(float)
    signal = signal - np.mean(signal)
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)

    if smooth_cutoff is not None:
        fs = trace.stats.sampling_rate
        b, a = butter(N=2, Wn=smooth_cutoff / (0.5 * fs), btype="low")
        envelope = filtfilt(b, a, envelope)
        envelope = np.maximum(envelope, 0)
    return envelope


def calculate_envelope_lag(envelope_a, envelope_b, sampling_rate, max_shift_seconds):
    min_len = min(len(envelope_a), len(envelope_b))
    envelope_a = envelope_a[:min_len]
    envelope_b = envelope_b[:min_len]

    crop = int(0.05 * min_len)
    if crop > 0:
        envelope_a = envelope_a[crop:-crop]
        envelope_b = envelope_b[crop:-crop]

    envelope_a = envelope_a - np.mean(envelope_a)
    envelope_b = envelope_b - np.mean(envelope_b)

    taper = windows.tukey(len(envelope_a), alpha=0.05)
    a = envelope_a * taper
    b = envelope_b * taper

    correlation = correlate(a, b, mode="full")
    lags = correlation_lags(len(a), len(b), mode="full")

    norm_factor = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if norm_factor == 0:
        raise RuntimeError("Envelope has zero energy.")

    correlation = correlation / norm_factor
    lag_seconds = lags / sampling_rate

    max_shift_samples = int(round(max_shift_seconds * sampling_rate))
    valid = (lags >= -max_shift_samples) & (lags <= max_shift_samples)

    valid_corr = correlation[valid]
    valid_lags = lags[valid]

    if len(valid_corr) == 0:
        raise RuntimeError("No valid lag values found.")

    peak_index = np.argmax(valid_corr)
    lag_samples = valid_lags[peak_index]
    lag_seconds_value = lag_samples / sampling_rate
    max_correlation = valid_corr[peak_index]

    return {
        "lag": float(lag_seconds_value),
        "cc_max": float(max_correlation),
        "lags_full": lag_seconds.tolist(),
        "cc_full": correlation.tolist(),
    }


def solve_planar_slowness(station_xy, lag_ab, lag_ac, station_a, station_b, station_c):
    """Solve for horizontal slowness vector (sx, sy), velocity, and back-azimuth.
    
    Matrix formulation:
        [dt_ab] = [dx_ab  dy_ab] * [s_x]
        [dt_ac]   [dx_ac  dy_ac]   [s_y]
    """
    x_a, y_a = station_xy[station_a]
    x_b, y_b = station_xy[station_b]
    x_c, y_c = station_xy[station_c]

    # Spatial offsets in km relative to station_a
    dx_ab, dy_ab = x_b - x_a, y_b - y_a
    dx_ac, dy_ac = x_c - x_a, y_c - y_a

    # Measured time differences in seconds (lag_ab = t_b - t_a)
    dt_ab = lag_ab["lag"]
    dt_ac = lag_ac["lag"]

    # System matrix G (2x2) and data vector d (2x1)
    G = np.array([[dx_ab, dy_ab], [dx_ac, dy_ac]])
    d = np.array([dt_ab, dt_ac])

    # Check determinant to ensure stations are not collinear
    det = np.linalg.det(G)
    if abs(det) < 1e-5:
        raise ValueError("Stations are nearly collinear; cannot invert array slowness.")

    # Solve for slowness vector s = [s_x, s_y] in s/km
    s_x, s_y = np.linalg.solve(G, d)

    # Compute slowness magnitude (s/km) and apparent velocity (km/s)
    slowness_mag = np.sqrt(s_x**2 + s_y**2)
    velocity = 1.0 / slowness_mag if slowness_mag > 0 else np.nan

    # Back-Azimuth: direction FROM which the wave came (clockwise from North in degrees)
    # Note: Wave propagation vector points along (s_x, s_y), back-azimuth is opposite or atan2(s_x, s_y)
    back_azimuth = (np.degrees(np.arctan2(s_x, s_y)) + 360) % 360

    return {
        "s_x": float(s_x),
        "s_y": float(s_y),
        "slowness_km_s": float(slowness_mag),
        "velocity_km_s": float(velocity),
        "back_azimuth_deg": float(back_azimuth),
    }


def create_array_geometry_plot(station_xy, slowness_results):
    """Plot array geometry and back-azimuth wave propagation vector."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot stations
    xs = [xy[0] for xy in station_xy.values()]
    ys = [xy[1] for xy in station_xy.values()]
    
    for name, (x, y) in station_xy.items():
        ax.scatter(x, y, marker="^", color="red", s=200, zorder=5)
        ax.text(x + 3, y + 3, name, fontsize=12, fontweight="bold")

    # Center of array
    center_x, center_y = np.mean(xs), np.mean(ys)
    
    # Back-azimuth vector (points in the direction the wave is heading, opposite of back-azimuth)
    baz_rad = np.radians(slowness_results["back_azimuth_deg"])
    
    # Vector direction (propagation vector points towards back_azimuth + 180)
    prop_x = np.sin(baz_rad + np.pi)
    prop_y = np.cos(baz_rad + np.pi)
    
    vector_scale = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.4
    ax.quiver(
        center_x, center_y,
        prop_x * vector_scale, prop_y * vector_scale,
        angles="xy", scale_units="xy", scale=1,
        color="blue", width=0.008, zorder=6,
        label=f"Wave Propagation Direction\n(BAZ: {slowness_results['back_azimuth_deg']:.1f}°)"
    )

    ax.set_xlabel("East-West distance from origin station (km)")
    ax.set_ylabel("North-South distance from origin station (km)")
    ax.set_title(
        f"3-Station Planar Wave Array Solution\n"
        f"Apparent Velocity: {slowness_results['velocity_km_s']:.2f} km/s | "
        f"Back-Azimuth: {slowness_results['back_azimuth_deg']:.1f}°"
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()

    return image_to_base64(fig)


def run_velocity_calculation(
    stations,
    start,
    end,
    fs=20.0,
    fmin=1.0,
    fmax=8.0,
    smooth_cutoff=0.5,
    max_shift_seconds=50.0,
):
    """Main analysis function using Method 1 (Planar Wavefront Velocity Calculation)."""
    if len(stations) != 3:
        raise ValueError("This method needs exactly 3 stations.")

    start = UTCDateTime(start)
    end = UTCDateTime(end)

    print("Fetching waveforms...")
    traces = {}
    for station in stations:
        print(f"Fetching {station['sta']}")
        traces[station["sta"]] = fetch_waveform(station, start, end, fs, fmin, fmax)

    print("Trimming to common window...")
    trim_to_common_window(traces)

    print("Creating waveform and spectrogram plots...")
    waveform_plot = create_waveform_plot(traces)
    spectrogram_plot = create_spectrogram_plot(traces, fmin=fmin, fmax=fmax)

    print("Calculating Hilbert envelopes...")
    envelopes = {}
    for name, trace in traces.items():
        envelopes[name] = get_hilbert_envelope(trace, smooth_cutoff)

    station_a = stations[0]["sta"]
    station_b = stations[1]["sta"]
    station_c = stations[2]["sta"]

    print("Calculating pairwise TDOAs...")
    lag_ab = calculate_envelope_lag(envelopes[station_a], envelopes[station_b], fs, max_shift_seconds)
    lag_ac = calculate_envelope_lag(envelopes[station_a], envelopes[station_c], fs, max_shift_seconds)
    lag_bc = calculate_envelope_lag(envelopes[station_b], envelopes[station_c], fs, max_shift_seconds)

    closure_error = lag_ac["lag"] - (lag_ab["lag"] + lag_bc["lag"])

    print("\nHilbert Envelope TDOA Results:")
    print(f"{station_a} - {station_b}: lag = {lag_ab['lag']:+.3f} s, CC = {lag_ab['cc_max']:.3f}")
    print(f"{station_a} - {station_c}: lag = {lag_ac['lag']:+.3f} s, CC = {lag_ac['cc_max']:.3f}")
    print(f"{station_b} - {station_c}: lag = {lag_bc['lag']:+.3f} s, CC = {lag_bc['cc_max']:.3f}")
    print(f"TDOA closure error: {closure_error:+.3f} s")

    # Set up local relative spatial coordinates (origin at station_a)
    origin_lat, origin_lon = stations[0]["lat"], stations[0]["lon"]
    station_xy = {}
    for station in stations:
        station_xy[station["sta"]] = latlon_to_xy(origin_lat, origin_lon, station["lat"], station["lon"])

    # Method 1 Solution
    print("\nCalculating Seismic Wave Velocity and Back-Azimuth...")
    slowness_results = solve_planar_slowness(station_xy, lag_ab, lag_ac, station_a, station_b, station_c)

    print(f"Slowness X (s_x)    : {slowness_results['s_x']:+.6f} s/km")
    print(f"Slowness Y (s_y)    : {slowness_results['s_y']:+.6f} s/km")
    print(f"Apparent Velocity   : {slowness_results['velocity_km_s']:.3f} km/s")
    print(f"Back-Azimuth        : {slowness_results['back_azimuth_deg']:.2f}°")

    plot_png = create_array_geometry_plot(station_xy, slowness_results)

    pairs = [
        {"a": station_a, "b": station_b, **lag_ab},
        {"a": station_a, "b": station_c, **lag_ac},
        {"a": station_b, "b": station_c, **lag_bc},
    ]

    station_coordinates = {s["sta"]: {"lat": s["lat"], "lon": s["lon"]} for s in stations}

    return {
        "pairs": pairs,
        "closure_error": float(closure_error),
        "slowness": slowness_results,
        "stations": station_coordinates,
        "waveform_plot": waveform_plot,
        "spectrogram_plot": spectrogram_plot,
        "plot_png": plot_png,
        "settings": {
            "start": str(start),
            "end": str(end),
            "sampling_rate": float(fs),
            "freqmin": float(fmin),
            "freqmax": float(fmax),
            "smooth_cutoff": float(smooth_cutoff),
            "max_shift_seconds": float(max_shift_seconds),
        },
    }


# Execute direct run
if __name__ == "__main__":
    stations = [
        {"sta": "KKN", "lat": 27.8000, "lon": 85.2790, "net": "NK", "cha": "BHZ", "loc": "*"},
        {"sta": "EQM07", "lat": 27.814941, "lon": 86.713401, "net": "NP", "cha": "EHZ", "loc": "*"},
        {"sta": "EQM13", "lat": 28.256323, "lon": 85.367569, "net": "NP", "cha": "EHZ", "loc": "*"},
    ]

    start = "2026-10-06T16:52:00"
    end = "2026-10-06T17:05:00"

    result = run_velocity_calculation(
        stations=stations,
        start=start,
        end=end,
        fs=50.0,
        fmin=1.0,
        fmax=8.0,
        smooth_cutoff=0.5,
        max_shift_seconds=50.0,
    )

    print("\nAnalysis completed successfully.")