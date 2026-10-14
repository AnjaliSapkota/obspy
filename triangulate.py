import base64
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import butter, correlate, correlation_lags, filtfilt, hilbert, windows, welch
from scipy.optimize import least_squares
from pyproj import Transformer, Geod
from obspy.taup import TauPyModel
from frequency import analyze_spectral_features, create_spectral_feature_plot

taup_model = TauPyModel(model="ak135")

# FDSN client
client = Client("https://seiscomp.alertnepal.online")

geod = Geod(ellps="WGS84")
transformer_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
transformer_to_wgs = Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True)

def fetch_waveform(station, start, end, target_fs, freqmin, freqmax):
    stream = client.get_waveforms(network=station["net"], station=station["sta"], location=station["loc"], channel=station["cha"], starttime=start, endtime=end)

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

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle("Seismic Waveforms")
    fig.tight_layout()
    return image_to_base64(fig)


def create_spectrogram_plot(traces, fmin, fmax):
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

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(f"Frequency Spectrogram ({fmin:g}–{fmax:g} Hz)")
    fig.tight_layout()
    return image_to_base64(fig)



def create_hyperbola_plot(station_xy, station_latlon, delta, source_xy, origin_lat, origin_lon,):
    xs = [xy[0] for xy in station_xy.values()]
    ys = [xy[1] for xy in station_xy.values()]

    all_x = xs + [source_xy[0]]
    all_y = ys + [source_xy[1]]

    margin = max( max(all_x) - min(all_x), max(all_y) - min(all_y),) * 2 + 50.0

    xmin, xmax = min(all_x) - margin, max(all_x) + margin
    ymin, ymax = min(all_y) - margin, max(all_y) + margin

    resolution = 1

    X, Y = np.meshgrid(np.arange(xmin, xmax, resolution),np.arange(ymin, ymax, resolution))

    # Convert the UTM grid from kilometres to geographic coordinates.
    ox, oy = transformer_to_utm.transform(origin_lon, origin_lat)

    grid_lon, grid_lat = transformer_to_wgs.transform(ox + X * 1000.0,oy + Y * 1000.0,)

    # Calculate geodesic distances from every grid point to each station.
    D = {}

    for name, (s_lat, s_lon) in station_latlon.items():
        _, _, dist_m = geod.inv(
            grid_lon,
            grid_lat,
            np.full_like(grid_lon, s_lon),
            np.full_like(grid_lat, s_lat),
        )

        D[name] = dist_m / 1000.0

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["blue", "red", "green"]
    legend_lines = []

    for i, (pair, dist_diff) in enumerate(delta.items()):
        station_a, station_b = pair.split("-")

        H = D[station_b] - D[station_a] - dist_diff

        ax.contour(X, Y, H, levels=[0], colors=colors[i], linewidths=2.5)

        legend_lines.append(Line2D([0], [0], color=colors[i % len(colors)], linewidth=2.5, label=f"{station_a}–{station_b} hyperbola", ))

    for name, (sx, sy) in station_xy.items():
        ax.scatter(sx, sy, marker=".", s=160, zorder=5)
        ax.text(sx + 5, sy + 5, name, fontsize=12, fontweight="bold")

    legend_lines.append(Line2D([0], [0], marker=".", color="black", linestyle="None",markersize=10, label="Station"))

    ax.scatter(source_xy[0], source_xy[1], marker="*", s=300, zorder=10)

    legend_lines.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            linestyle="None",
            markersize=15,
            label="Estimated source",
        )
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


def get_hilbert_envelope(trace, smooth_cutoff):
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


def invert_location_velocity_geod(
    station_latlon,
    lag_pairs,
    v_min,
    v_max,
    v_step=0.025,
    grid_step_deg=0.015,
    margin_deg=2.5,
):
    """Grid search over geographic (lat, lon, v) using true geodetic distances followed by least-squares refinement."""
    lats = [coord[0] for coord in station_latlon.values()]
    lons = [coord[1] for coord in station_latlon.values()]
    
    lat_range = np.arange(min(lats) - margin_deg, max(lats) + margin_deg, grid_step_deg)
    lon_range = np.arange(min(lons) - margin_deg, max(lons) + margin_deg, grid_step_deg)
    v_range = np.arange(v_min, v_max + v_step, v_step)

    LatGrid, LonGrid = np.meshgrid(lat_range, lon_range, indexing="ij")

    D_km = {}
    for name, (s_lat, s_lon) in station_latlon.items():
        _, _, dist_m = geod.inv(LonGrid, LatGrid, np.full_like(LonGrid, s_lon), np.full_like(LatGrid, s_lat))
        D_km[name] = dist_m / 1000.0

    best = {"misfit": np.inf}
    misfit_vs_v = []

    for v in v_range:
        total = np.zeros_like(LatGrid)
        for a, b, lag, cc in lag_pairs:
            pred = (D_km[a] - D_km[b]) / v
            total += max(cc, 0.0) * (lag - pred) ** 2
        misfit = np.sqrt(total)

        idx = np.unravel_index(np.argmin(misfit), misfit.shape)
        misfit_vs_v.append(float(misfit[idx]))

        if misfit[idx] < best["misfit"]:
            best = {
                "misfit": float(misfit[idx]),
                "lat": float(lat_range[idx[0]]),
                "lon": float(lon_range[idx[1]]),
                "v": float(v),
            }

    def residuals(p):
        lat, lon, v = p
        res = []
        for a, b, lag, cc in lag_pairs:
            _, _, da_m = geod.inv(lon, lat, station_latlon[a][1], station_latlon[a][0])
            _, _, db_m = geod.inv(lon, lat, station_latlon[b][1], station_latlon[b][0])
            pred = ((da_m - db_m) / 1000.0) / v
            res.append(np.sqrt(max(cc, 0.0)) * (lag - pred))
        return res

    refined = least_squares(
        residuals,
        [best["lat"], best["lon"], best["v"]],
        bounds=([min(lat_range), min(lon_range), v_min], [max(lat_range), max(lon_range), v_max]),
    )
    lat_ref, lon_ref, v_ref = refined.x

    return {
        "lat": float(lat_ref),
        "lon": float(lon_ref),
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
    v_min,
    v_max,
    fs,
    fmin,
    fmax,
    smooth_cutoff,
    max_shift_seconds,
    v_step=0.025,
    grid_step_deg=0.015,
):
    if len(stations) != 3:
        raise ValueError("This method needs exactly 3 stations.")

    start = UTCDateTime(start)
    end = UTCDateTime(end)
    
    if end <= start:
        raise ValueError("End time must be after start time.")
    
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

    print("Calculating spectral features")

    spectral_features = analyze_spectral_features(
        traces,
        window_seconds=10.0,
        step_seconds=5.0,
        fmin= fmin,
        fmax= fmax,
    )

    spectral_plot = None

    if spectral_features:
        print("Creating spectral feature plot")
        spectral_plot = create_spectral_feature_plot(spectral_features)
    else:
        print("No usable spectral features were calculated.")

    print(f"Calculated {len(spectral_features)} spectral feature windows.")

    print("Calculating Hilbert envelopes")
    envelopes = {name: get_hilbert_envelope(tr, smooth_cutoff) for name, tr in traces.items()}

    station_a = stations[0]["sta"]
    station_b = stations[1]["sta"]
    station_c = stations[2]["sta"]

    print("Calculating TDOA")
    lag_ab = calculate_envelope_lag(envelopes[station_a], envelopes[station_b], fs, max_shift_seconds)
    lag_ac = calculate_envelope_lag(envelopes[station_a], envelopes[station_c], fs, max_shift_seconds)
    lag_bc = calculate_envelope_lag(envelopes[station_b], envelopes[station_c], fs, max_shift_seconds)

    closure_error = lag_ac["lag"] - (lag_ab["lag"] + lag_bc["lag"])

    print(f"\nHilbert Envelope TDOA results:")
    print(f"{station_a} - {station_b}: lag = {lag_ab['lag']:+.3f} s, CC = {lag_ab['cc_max']:.3f}")
    print(f"{station_a} - {station_c}: lag = {lag_ac['lag']:+.3f} s, CC = {lag_ac['cc_max']:.3f}")
    print(f"{station_b} - {station_c}: lag = {lag_bc['lag']:+.3f} s, CC = {lag_bc['cc_max']:.3f}")
    print(f"\nTDOA closure error: {closure_error:+.3f} s")

    origin_lat = stations[0]["lat"]
    origin_lon = stations[0]["lon"]
    
    station_latlon = {
        s["sta"]: (s["lat"], s["lon"]) for s in stations
    }

    print("\nRunning geodetic joint inversion for location and velocity")
    lag_pairs = [
        (station_a, station_b, lag_ab["lag"], lag_ab["cc_max"]),
        (station_a, station_c, lag_ac["lag"], lag_ac["cc_max"]),
        (station_b, station_c, lag_bc["lag"], lag_bc["cc_max"]),
    ]

    inv = invert_location_velocity_geod(
        station_latlon,
        lag_pairs,
        v_min=v_min,
        v_max=v_max,
        v_step=v_step,
        grid_step_deg=grid_step_deg,
    )

    source_lat = inv["lat"]
    source_lon = inv["lon"]
    velocity = inv["velocity"]
    residual_norm = inv["residual_norm"]

    x_src, y_src = latlon_to_xy(origin_lat, origin_lon, source_lat, source_lon)

    station_xy = {
        s["sta"]: latlon_to_xy(origin_lat, origin_lon, s["lat"], s["lon"])
        for s in stations
    }

    print(f"Latitude  : {source_lat:.6f}")
    print(f"Longitude : {source_lon:.6f}")
    print(f"Velocity  : {velocity:.3f} km/s")
    print(f"Residual  : {residual_norm:.4f} s")

    delta = {
        f"{station_a}-{station_b}": -velocity * lag_ab["lag"],
        f"{station_a}-{station_c}": -velocity * lag_ac["lag"],
        f"{station_b}-{station_c}": -velocity * lag_bc["lag"],
    }

    print("Creating hyperbola and velocity plots")
    plot_png = create_hyperbola_plot(station_xy, station_latlon, delta, (x_src, y_src), origin_lat, origin_lon,)
    velocity_plot = create_velocity_plot(inv)

    pairs = [
        {"a": station_a, "b": station_b, **lag_ab},
        {"a": station_a, "b": station_c, **lag_ac},
        {"a": station_b, "b": station_c, **lag_bc},
    ]

    station_coordinates = {
        s["sta"]: {"lat": s["lat"], "lon": s["lon"]} for s in stations
    }

    pairs_lag_map = {
        f"{station_a}-{station_b}": lag_ab["lag"],
        f"{station_a}-{station_c}": lag_ac["lag"],
        f"{station_b}-{station_c}": lag_bc["lag"]
    }
    aligned_plot = create_aligned_envelope_plot(envelopes, pairs_lag_map, fs)
    
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
        "spectral_features": spectral_features,
        "spectral_plot": spectral_plot,
        "plot_png": plot_png,
        "velocity_plot": velocity_plot,
        "aligned_plot": aligned_plot,
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
        },

    }

def create_aligned_envelope_plot(envelopes, lags, sampling_rate):
    fig, ax = plt.subplots(figsize=(12, 5))
    
    names = list(envelopes.keys())
    ref_name = names[0]
    
    ref_env = envelopes[ref_name]
    ref_max = np.max(ref_env)
    ref_env_norm = ref_env / ref_max if ref_max > 0 else ref_env
    time_ref = np.arange(len(ref_env_norm)) / sampling_rate
    
    ax.plot(time_ref, ref_env_norm, label=f"{ref_name} (Reference)", linewidth=1.5, color="black")
    
    colors = ["blue", "green", "purple", "orange"]
    for i, name in enumerate(names[1:]):
        env = envelopes[name]
        env_max = np.max(env)
        env_norm = env / env_max if env_max > 0 else env
        time_env = np.arange(len(env_norm)) / sampling_rate
        
        lag_val = 0.0
        for pair, l_val in lags.items():
            if name in pair and ref_name in pair:
                lag_val = l_val if pair.startswith(ref_name) else -l_val
                break
                
        shifted_time = time_env + lag_val
        ax.plot(shifted_time, env_norm, label=f"{name} (shifted by {lag_val:+.2f}s)", 
                linewidth=1.2, alpha=0.8, color=colors[i % len(colors)])

    ax.set_title("Normalized Hilbert Envelopes Aligned by TDOA Lags")
    ax.set_xlabel("Reference Time (seconds)")
    ax.set_ylabel("Normalized Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return image_to_base64(fig)

