
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, correlate, correlation_lags, filtfilt, hilbert, windows, welch
from scipy.optimize import least_squares
import base64
import io

def image_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def extract_spectral_features(data, fs, fmin, fmax):
    data = np.asarray(data, dtype=float)
    data = data - np.mean(data)

    frequencies, power = welch(data, fs=fs, nperseg=min(256, len(data)),)

    mask = ((frequencies >= fmin) & (frequencies <= fmax))

    frequencies = frequencies[mask]
    power = power[mask]
    total_power = np.sum(power)

    if frequencies.size == 0 or total_power <= 0:
        raise ValueError("No usable spectral power.")

    dominant_frequency = frequencies[np.argmax(power)]
    centroid = np.sum(frequencies * power) / total_power
    bandwidth = np.sqrt(np.sum((frequencies - centroid) ** 2 * power) / total_power)

    low_band = (frequencies >= 1.0) & (frequencies < 3.0)
    high_band = (frequencies >= 3.0) & (frequencies <= 6.0)

    low_energy = np.sum(power[low_band])
    high_energy = np.sum(power[high_band])

        
    if (
        low_energy > 0
        and np.isfinite(low_energy)
        and np.isfinite(high_energy)
        and np.any(high_band)
    ):
        band_ratio = float(high_energy / low_energy)
    else:
        band_ratio = None


    return {
        "dominant_frequency_hz": float(dominant_frequency),
        "spectral_centroid_hz": float(centroid),
        "spectral_bandwidth_hz": float(bandwidth),
        "high_low_band_ratio": band_ratio,
    }


def analyze_spectral_features(traces, fmin, fmax, window_seconds=10.0, step_seconds=5.0,):
    if window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("Window and step durations must be positive.")

    results = []

    for name, trace in traces.items():
        sampling_rate = trace.stats.sampling_rate

        samples_per_window = int(round(window_seconds * sampling_rate))
        step_samples = int(round(step_seconds * sampling_rate))

        if samples_per_window < 16 or step_samples < 1:
            raise ValueError("Invalid window or step size.")

        if trace.stats.npts < samples_per_window:
            print(f"Skipping {name}: trace is shorter than one window.")
            continue

        for first_sample in range(
            0,
            trace.stats.npts - samples_per_window + 1,
            step_samples,
        ):
            last_sample = first_sample + samples_per_window
            data = trace.data[first_sample:last_sample]

            try:
                features = extract_spectral_features(
                    data,
                    sampling_rate,
                    fmin=fmin,
                    fmax=fmax,
                )
            except ValueError as error:
                print(f"Skipping {name} window: {error}")
                continue

            center_seconds = (
                first_sample + (samples_per_window - 1) / 2
            ) / sampling_rate

            results.append({
                "station": name,
                "time_seconds": float(center_seconds),
                **features,
            })

    return results


def create_spectral_feature_plot(results):
    if not results:
        raise ValueError("No spectral features available to plot.")

    feature_names = [
        ("dominant_frequency_hz", "Dominant Frequency (Hz)"),
        ("spectral_centroid_hz", "Spectral Centroid (Hz)"),
        ("spectral_bandwidth_hz", "Spectral Bandwidth (Hz)"),
        ("high_low_band_ratio", "High/Low Frequency Power Ratio"),
    ]

    stations = list(dict.fromkeys(
        row["station"] for row in results
    ))

    fig, axes = plt.subplots(
        len(feature_names),
        1,
        figsize=(12, 11),
        sharex=True,
    )

    for ax, (feature, ylabel) in zip(axes, feature_names):
        for station in stations:
            station_rows = [
                row for row in results
                if row["station"] == station
            ]

            times = [
                row["time_seconds"] for row in station_rows
            ]

            values = [
                row[feature]
                if row[feature] is not None
                else np.nan
                for row in station_rows
            ]

            ax.plot(
                times,
                values,
                marker=".",
                markersize=3,
                linewidth=1,
                label=station,
            )

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time from Common Window Start (seconds)")
    fig.suptitle("Spectral Features Across Seismic Stations")
    fig.tight_layout()

    return image_to_base64(fig)

