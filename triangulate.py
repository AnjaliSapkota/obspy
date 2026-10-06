import base64
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import butter, correlate, correlation_lags, filtfilt, hilbert, windows
from scipy.optimize import least_squares
from pyproj import Transformer

# FDSN client
client = Client("https://seiscomp.alertnepal.online")

# UTM projection for Nepal (Zone 45N, EPSG:32645)
transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
transformer_to_wgs = Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True)

def fetch_waveform(station,start,end,target_fs,freqmin,freqmax):
    stream = client.get_waveforms(network=station["net"],station=station["sta"],location=station["loc"],channel=station["cha"],starttime=start,endtime=end)


    if len(stream) == 0:
        raise RuntimeError(f"No {station['sta']} waveform found.")

    stream.merge(method=1, fill_value="interpolate")
    stream.detrend("linear")
    stream.detrend("demean")
    stream.taper(max_percentage=0.05, type="hann")
    stream.interpolate(sampling_rate=target_fs, method="linear")
    stream.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)

    return stream[0]


def trim_to_common_window(traces):
    common_start = max(t.stats.starttime for t in traces.values())
    common_end = min(t.stats.endtime for t in traces.values())
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
        data = trace.data.astype(float)
        time = np.arange(len(data)) / trace.stats.sampling_rate
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
        data = trace.data.astype(float)
        nperseg = min(512, len(data))
        if nperseg < 2:
            raise RuntimeError(f"Not enough samples for spectrogram at {name}.")

        ax.specgram(
            data,
            NFFT=nperseg,
            Fs=trace.stats.sampling_rate,
            noverlap=int(nperseg * 0.75),
        )
        ax.set_ylabel(f"{name}\nFrequency (Hz)")
        ax.set_ylim(fmin, fmax)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time since common start (seconds)")
    fig.suptitle(f"Three-Station Frequency Spectrogram ({fmin:g}–{fmax:g} Hz)")
    fig.tight_layout()
    return image_to_base64(fig)


def create_hyperbola_plot(station_xy, delta, source_xy):
    xs = [xy[0] for xy in station_xy.values()]
    ys = [xy[1] for xy in station_xy.values()]

    all_x = xs + [source_xy[0]]
    all_y = ys + [source_xy[1]]
    margin = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) * 0.2 + 50.0
    xmin, xmax = min(all_x) - margin, max(all_x) + margin
    ymin, ymax = min(all_y) - margin, max(all_y) + margin

    resolution = 0.5
    X, Y = np.meshgrid(np.arange(xmin, xmax, resolution), np.arange(ymin, ymax, resolution))

    D = {
        name: np.sqrt((X - sx) ** 2 + (Y - sy) ** 2)
        for name, (sx, sy) in station_xy.items()
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["blue", "red", "green"]
    legend_lines = []

    # delta[a-b] = d_b - d_a
    for i, (pair, dist_diff) in enumerate(delta.items()):
        station_a, station_b = pair.split("-")
        H = D[station_b] - D[station_a] - dist_diff
        ax.contour(X, Y, H, levels=[0], colors=colors[i], linewidths=2.5)
        legend_lines.append(
            Line2D([0], [0], color=colors[i], linewidth=2.5,
                   label=f"{station_a}–{station_b} hyperbola")
        )

    for name, (sx, sy) in station_xy.items():
        ax.scatter(sx, sy, marker=".", s=160, zorder=5)
        ax.text(sx + 5, sy + 5, name, fontsize=12, fontweight="bold")

    legend_lines.append(
        Line2D([0], [0], marker=".", color="black", linestyle="None",
               markersize=10, label="Station")
    )

    ax.scatter(source_xy[0], source_xy[1], marker="*", s=300, zorder=10)
    legend_lines.append(
        Line2D([0], [0], marker="*", color="black", linestyle="None",
               markersize=15, label="Estimated source")
    )

    ax.legend(handles=legend_lines, loc="best")
    ax.set_xlabel("East-West distance from origin station (km)")
    ax.set_ylabel("North-South distance from origin station (km)")
    ax.set_title("Three-Station TDOA Hyperbola Intersection")
    ax.grid(True)
    ax.axis("equal")
    fig.tight_layout()
    return image_to_base64(fig)


def create_velocity_plot(inv):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(inv["v_range"], inv["misfit_vs_v"], "b-", linewidth=2)
    ax.axvline(
        inv["velocity"], color="r", linestyle="--",
        label=f"Estimated velocity: {inv['velocity']:.2f} km/s",
    )
    ax.set_xlabel("Apparent velocity (km/s)")
    ax.set_ylabel("Minimum spatial TDOA misfit (s)")
    ax.set_title("Misfit vs. Velocity")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    return image_to_base64(fig)


def get_hilbert_envelope(trace, smooth_cutoff=0.5):
    signal = trace.data.astype(float)
    signal = signal - np.mean(signal)
    envelope = np.abs(hilbert(signal))

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

    # Remove 5% from both edges
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

    norm_factor = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
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

    return {
        "lag": float(valid_lags[peak_index] / sampling_rate),
        "cc_max": float(valid_corr[peak_index]),
        "lags_full": lag_seconds.tolist(),
        "cc_full": correlation.tolist(),
    }


def latlon_to_xy(origin_lat, origin_lon, lat, lon):
    ox, oy = transformer_to_utm.transform(origin_lon, origin_lat)
    tx, ty = transformer_to_utm.transform(lon, lat)
    return (tx - ox) / 1000.0, (ty - oy) / 1000.0


def xy_to_latlon(origin_lat, origin_lon, x_km, y_km):
    ox, oy = transformer_to_utm.transform(origin_lon, origin_lat)
    lon, lat = transformer_to_wgs.transform(ox + x_km * 1000.0, oy + y_km * 1000.0)
    return lat, lon


def predicted_lag(x, y, v, station_xy, a, b):
    xa, ya = station_xy[a]
    xb, yb = station_xy[b]
    d_a = np.hypot(x - xa, y - ya)
    d_b = np.hypot(x - xb, y - yb)
    return (d_a - d_b) / v


def invert_location_velocity(
    station_xy,
    lag_pairs,            # list of (station_a, station_b, lag_seconds, cc_max)
    v_min=1.0,
    v_max=3.5,
    v_step=0.025,
    grid_step_km=2.0,
    margin_km=100.0,
):
    """Grid search over (x, y, v) followed by least-squares refinement."""

    xs = [p[0] for p in station_xy.values()]
    ys = [p[1] for p in station_xy.values()]
    x_range = np.arange(min(xs) - margin_km, max(xs) + margin_km, grid_step_km)
    y_range = np.arange(min(ys) - margin_km, max(ys) + margin_km, grid_step_km)
    v_range = np.arange(v_min, v_max + v_step, v_step)

    X, Y = np.meshgrid(x_range, y_range, indexing="ij")

    D = {
        name: np.hypot(X - sx, Y - sy)
        for name, (sx, sy) in station_xy.items()
    }

    best = {"misfit": np.inf}
    misfit_vs_v = []

    for v in v_range:
        total = np.zeros_like(X)
        for a, b, lag, cc in lag_pairs:
            pred = (D[a] - D[b]) / v
            total += max(cc, 0.0) * (lag - pred) ** 2
        misfit = np.sqrt(total)

        idx = np.unravel_index(np.argmin(misfit), misfit.shape)
        misfit_vs_v.append(float(misfit[idx]))

        if misfit[idx] < best["misfit"]:
            best = {
                "misfit": float(misfit[idx]),
                "x": float(x_range[idx[0]]),
                "y": float(y_range[idx[1]]),
                "v": float(v),
            }

    def residuals(p):
        x, y, v = p
        return [
            np.sqrt(max(cc, 0.0)) * (lag - predicted_lag(x, y, v, station_xy, a, b))
            for a, b, lag, cc in lag_pairs
        ]

    refined = least_squares(
        residuals,
        [best["x"], best["y"], best["v"]],
        bounds=([-np.inf, -np.inf, v_min], [np.inf, np.inf, v_max]),
    )
    x_ref, y_ref, v_ref = refined.x

    return {
        "x_km": float(x_ref),
        "y_km": float(y_ref),
        "velocity": float(v_ref),
        "residual_norm": float(np.linalg.norm(refined.fun)),
        "grid_best": best,
        "v_range": v_range.tolist(),
        "misfit_vs_v": misfit_vs_v,
    }


def run_triangulation(
    stations,
    start,
    end,
    fs=20.0,
    fmin=1.0,
    fmax=8.0,
    smooth_cutoff=0.5,
    max_shift_seconds=50.0,
    v_min=1.0,
    v_max=3.5,
    v_step=0.025,
    grid_step_km=2.0,
):
    if len(stations) != 3:
        raise ValueError("This method needs exactly 3 stations.")

    start = UTCDateTime(start)
    end = UTCDateTime(end)

    # Fetch waveforms
    print("Fetching waveforms")
    traces = {}
    for station in stations:
        print(f"Fetching {station['sta']}")
        traces[station["sta"]] = fetch_waveform(station, start, end, fs, fmin, fmax)

    print("Trimming to common window")
    trim_to_common_window(traces)

    print("Creating waveform plots")
    waveform_plot = create_waveform_plot(traces)
    spectrogram_plot = create_spectrogram_plot(traces, fmin=fmin, fmax=fmax)

    # Envelopes
    print("Calculating Hilbert envelopes")
    envelopes = {name: get_hilbert_envelope(tr, smooth_cutoff) for name, tr in traces.items()}

    station_a = stations[0]["sta"]
    station_b = stations[1]["sta"]
    station_c = stations[2]["sta"]

    # Pairwise TDOA
    print("Calculating TDOA")
    lag_ab = calculate_envelope_lag(envelopes[station_a], envelopes[station_b], fs, max_shift_seconds)
    lag_ac = calculate_envelope_lag(envelopes[station_a], envelopes[station_c], fs, max_shift_seconds)
    lag_bc = calculate_envelope_lag(envelopes[station_b], envelopes[station_c], fs, max_shift_seconds)

    closure_error = lag_ac["lag"] - (lag_ab["lag"] + lag_bc["lag"])

    print("\nHilbert Envelope TDOA results:")
    print(f"{station_a} - {station_b}: lag = {lag_ab['lag']:+.3f} s, CC = {lag_ab['cc_max']:.3f}")
    print(f"{station_a} - {station_c}: lag = {lag_ac['lag']:+.3f} s, CC = {lag_ac['cc_max']:.3f}")
    print(f"{station_b} - {station_c}: lag = {lag_bc['lag']:+.3f} s, CC = {lag_bc['cc_max']:.3f}")
    print(f"\nTDOA closure error: {closure_error:+.3f} s")

    # Station coordinates (first station = origin)
    origin_lat = stations[0]["lat"]
    origin_lon = stations[0]["lon"]
    station_xy = {
        s["sta"]: latlon_to_xy(origin_lat, origin_lon, s["lat"], s["lon"])
        for s in stations
    }

    # Joint location + velocity inversion
    print("\nRunning joint inversion for location and velocity")
    lag_pairs = [
        (station_a, station_b, lag_ab["lag"], lag_ab["cc_max"]),
        (station_a, station_c, lag_ac["lag"], lag_ac["cc_max"]),
        (station_b, station_c, lag_bc["lag"], lag_bc["cc_max"]),
    ]

    inv = invert_location_velocity(
        station_xy,
        lag_pairs,
        v_min=v_min,
        v_max=v_max,
        v_step=v_step,
        grid_step_km=grid_step_km,
    )

    x_src = inv["x_km"]
    y_src = inv["y_km"]
    velocity = inv["velocity"]
    residual_norm = inv["residual_norm"]

    source_lat, source_lon = xy_to_latlon(origin_lat, origin_lon, x_src, y_src)

    print(f"X (East)  : {x_src:+.3f} km")
    print(f"Y (North) : {y_src:+.3f} km")
    print(f"Velocity  : {velocity:.3f} km/s")
    print(f"Residual  : {residual_norm:.4f} s")
    print(f"Latitude  : {source_lat:.6f}")
    print(f"Longitude : {source_lon:.6f}")

    # Distance differences using estimated velocity: d_b - d_a = -v * lag
    delta = {
        f"{station_a}-{station_b}": -velocity * lag_ab["lag"],
        f"{station_a}-{station_c}": -velocity * lag_ac["lag"],
        f"{station_b}-{station_c}": -velocity * lag_bc["lag"],
    }

    print("Creating hyperbola and velocity plots")
    plot_png = create_hyperbola_plot(station_xy, delta, (x_src, y_src))
    velocity_plot = create_velocity_plot(inv)

    pairs = [
        {"a": station_a, "b": station_b, **lag_ab},
        {"a": station_a, "b": station_c, **lag_ac},
        {"a": station_b, "b": station_c, **lag_bc},
    ]

    station_coordinates = {
        s["sta"]: {"lat": s["lat"], "lon": s["lon"]} for s in stations
    }

    return {
        "pairs": pairs,
        "closure_error": float(closure_error),
        "source": {
            "lat": float(source_lat),
            "lon": float(source_lon),
            "x_km": float(x_src),
            "y_km": float(y_src),
            "residual_norm": float(residual_norm),
            "velocity": float(velocity),
        },
        "stations": station_coordinates,
        "waveform_plot": waveform_plot,
        "spectrogram_plot": spectrogram_plot,
        "plot_png": plot_png,
        "velocity_plot": velocity_plot,
        "settings": {
            "start": str(start),
            "end": str(end),
            "sampling_rate": float(fs),
            "freqmin": float(fmin),
            "freqmax": float(fmax),
            "smooth_cutoff": float(smooth_cutoff),
            "max_shift_seconds": float(max_shift_seconds),
            "v_min": float(v_min),
            "v_max": float(v_max),
            "v_step": float(v_step),
            "grid_step_km": float(grid_step_km),
        },
    }


# Run directly from Python
if __name__ == "__main__":
    stations = [
        {"sta": "KKN", "lat": 27.8000, "lon": 85.2790, "net": "NK", "cha": "BHZ", "loc": "*"},
        {"sta": "EVN", "lat": 27.95865, "lon": 86.811653, "net": "IO", "cha": "BHZ", "loc": "*"},
        {"sta": "EQM10", "lat": 28.299517, "lon": 83.960148, "net": "NP", "cha": "EHZ", "loc": "*"},
    ]

    result = run_triangulation(
        stations=stations,
        start="2026-08-26T02:52:00",
        end="2026-08-26T02:56:00",
        fs=20.0,
        fmin=1.0,
        fmax=8.0,
        smooth_cutoff=0.5,
        max_shift_seconds=30.0,
        v_min=1.0,
        v_max=3.5,
    )

    print("\nAnalysis completed.")