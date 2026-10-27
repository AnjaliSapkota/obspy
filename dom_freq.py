
import numpy as np
from obspy import UTCDateTime
from scipy.signal import butter, sosfiltfilt, hilbert, welch
from triangulate import fetch_waveform, trim_to_common_window
from itertools import combinations
from obspy.geodetics import gps2dist_azimuth

client = "https://seiscomp.alertnepal.online"

stations = [
    {"sta": "EQM10", "net": "NP", "loc": "*", "cha": "EHZ", "lat": 28.299517, "lon": 83.960148},
    {"sta": "KKN", "net": "NK", "loc": "*", "cha": "BHZ", "lat": 27.8, "lon": 85.279},
    {"sta": "EQM13", "net": "NP", "loc": "*", "cha": "EHZ", "lat": 28.256323, "lon": 85.367569},
]

start = UTCDateTime("2026-10-10T22:30:00")
end = UTCDateTime("2026-10-10T22:40:00")


def estimate_dominant_frequency_band(data, fs, fmin, fmax, power_fraction=0.90, nperseg=None):
    data = np.asarray(data, dtype=float)

    if len(data) < 2 or not np.all(np.isfinite(data)):
        raise ValueError("Waveform is empty or contains invalid values.")

    if not 0 < fmin < fmax < fs / 2:
        raise ValueError("Invalid frequency range.")

    data = data - np.mean(data)

    segment_length = min(4096, len(data)) if nperseg is None else min(int(nperseg), len(data))

    frequencies, psd = welch(
        data,
        fs=fs,
        window="hann",
        nperseg=segment_length,
        detrend="constant",
        scaling="density",
    )

    mask = (frequencies >= fmin) & (frequencies <= fmax)
    band_frequencies = frequencies[mask]
    band_psd = psd[mask]

    if len(band_frequencies) < 2:
        raise ValueError("Not enough frequency bins in the selected band.")

    df = frequencies[1] - frequencies[0]
    band_power = band_psd * df
    total_power = float(np.sum(band_power))

    if not np.isfinite(total_power) or total_power <= 0:
        raise ValueError("Signal power in the selected frequency band is zero.")

    peak_index = int(np.argmax(band_psd))
    dominant_frequency = float(band_frequencies[peak_index])

    left = peak_index
    right = peak_index
    captured_power = float(band_power[peak_index])
    target_power = power_fraction * total_power

    while captured_power < target_power:
        left_power = band_power[left - 1] if left > 0 else -np.inf
        right_power = band_power[right + 1] if right < len(band_power) - 1 else -np.inf

        if left_power == -np.inf and right_power == -np.inf:
            break

        if left_power >= right_power:
            left -= 1
            captured_power += float(band_power[left])
        else:
            right += 1
            captured_power += float(band_power[right])

    return {
        "dominant_frequency_hz": dominant_frequency,
        "band_low_hz": float(band_frequencies[left]),
        "band_high_hz": float(band_frequencies[right]),
        "band_power_fraction": float(captured_power / total_power),
        "frequencies_hz": frequencies,
        "psd": psd,
    }


def calculate_band_envelope(data, fs, low_frequency, high_frequency, filter_order=4):
    data = np.asarray(data, dtype=float)

    if not 0 < low_frequency < high_frequency < fs / 2:
        raise ValueError("Invalid frequency band for envelope calculation.")

    sos = butter(
        filter_order,
        [low_frequency, high_frequency],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    filtered = sosfiltfilt(sos, data)
    envelope = np.abs(hilbert(filtered))
    envelope_power = envelope ** 2

    peak_index = int(np.argmax(envelope_power))
    peak_time_seconds = peak_index / fs

    return {
        "filtered_waveform": filtered,
        "envelope": envelope,
        "envelope_power": envelope_power,
        "peak_index": peak_index,
        "peak_time_seconds": float(peak_time_seconds),
        "peak_envelope_amplitude": float(envelope[peak_index]),
        "peak_envelope_power": float(envelope_power[peak_index]),
    }



def calculate_snr_onset(data, fs, low_frequency, high_frequency, noise_window_seconds=10.0, moving_window_seconds=0.5, step_seconds=0.05, snr_threshold=5.0, consecutive_windows=5, filter_order=4):
    data = np.asarray(data, dtype=float)

    if fs <= 0:
        raise ValueError("Sampling rate must be positive.")

    if not np.all(np.isfinite(data)):
        raise ValueError("Waveform contains invalid values.")

    if not 0 < low_frequency < high_frequency < fs / 2:
        raise ValueError("Invalid frequency band.")

    noise_samples = int(noise_window_seconds * fs)
    window_samples = max(1, int(moving_window_seconds * fs))
    step_samples = max(1, int(step_seconds * fs))

    if len(data) <= noise_samples + window_samples:
        raise ValueError("Waveform is too short for onset detection.")

    sos = butter(
        filter_order,
        [low_frequency, high_frequency],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    filtered = sosfiltfilt(sos, data)

    noise = filtered[:noise_samples]
    noise = noise - np.median(noise)

    noise_power = float(np.mean(noise ** 2))

    if not np.isfinite(noise_power) or noise_power <= 0:
        raise ValueError("Invalid background noise power.")

    snr_values = []
    window_start_indices = []

    for i in range(0, len(filtered) - window_samples + 1, step_samples):
        segment = filtered[i:i + window_samples]
        segment = segment - np.mean(segment)
        segment_power = float(np.mean(segment ** 2))
        snr = segment_power / noise_power

        snr_values.append(snr)
        window_start_indices.append(i)

    snr_values = np.asarray(snr_values)
    window_start_indices = np.asarray(window_start_indices)

    valid = window_start_indices >= noise_samples
    candidate_indices = np.flatnonzero(valid)

    required = int(consecutive_windows)

    for j in range(len(candidate_indices) - required + 1):
        positions = candidate_indices[j:j + required]

        if not np.all(np.diff(positions) == 1):
            continue

        candidate_snr = snr_values[positions]

        if np.all(np.isfinite(candidate_snr)) and np.all(candidate_snr >= snr_threshold):
            onset_index = int(window_start_indices[positions[0]])
            onset_seconds = onset_index / fs

            return {
                "onset_index": onset_index,
                "onset_time_seconds": float(onset_seconds),
                "snr_at_onset": float(snr_values[positions[0]]),
                "noise_power": noise_power,
                "snr_values": snr_values,
                "window_start_indices": window_start_indices,
                "filtered_waveform": filtered,
                "noise_window_seconds": noise_window_seconds,
                "snr_threshold": snr_threshold,
            }

    raise ValueError(
        "No sustained onset detected. Inspect the waveform and SNR curve, "
        "then review the baseline, frequency band, and threshold."
    )


def analyze_trace(trace, fmin, fmax, power_fraction=0.90, nperseg=None):
    data = np.asarray(trace.data, dtype=float)
    fs = float(trace.stats.sampling_rate)

    band = estimate_dominant_frequency_band(
        data=data,
        fs=fs,
        fmin=fmin,
        fmax=fmax,
        power_fraction=power_fraction,
        nperseg=nperseg,
    )

    low = band["band_low_hz"]
    high = band["band_high_hz"]

    envelope_result = calculate_band_envelope(
        data=data,
        fs=fs,
        low_frequency=low,
        high_frequency=high,
    )

    peak_time_utc = trace.stats.starttime + envelope_result["peak_time_seconds"]

    onset_result = calculate_snr_onset(
        data=data,
        fs=fs,
        low_frequency=low,
        high_frequency=high,
        noise_window_seconds=10.0,
        moving_window_seconds=0.5,
        step_seconds=0.05,
        snr_threshold=10.0,
        consecutive_windows=3,
    )

    onset_time_utc = trace.stats.starttime + onset_result["onset_time_seconds"]

    result = dict(band)
    result.update(envelope_result)

    result.update({
        "peak_time_utc": str(peak_time_utc),
        "onset_time_utc": str(onset_time_utc),
        "onset_time_seconds": onset_result["onset_time_seconds"],
        "snr_at_onset": onset_result["snr_at_onset"],
        "noise_power": onset_result["noise_power"],
        "snr_values": onset_result["snr_values"],
        "snr_window_indices": onset_result["window_start_indices"],
        "trace_start_utc": str(trace.stats.starttime),
        "station_id": trace.id,
        "sampling_rate_hz": fs,
    })

    return result


def analyze_traces(traces, fmin, fmax, power_fraction=0.90, nperseg=None):
    results = []

    for name, trace in traces.items():
        try:
            result = analyze_trace(
                trace=trace,
                fmin=fmin,
                fmax=fmax,
                power_fraction=power_fraction,
                nperseg=nperseg,
            )

            result["station"] = name
            results.append(result)

            print(f"\nStation: {name}")
            print(f"Dominant frequency: {result['dominant_frequency_hz']:.3f} Hz")
            print(f"Selected band: {result['band_low_hz']:.3f}-{result['band_high_hz']:.3f} Hz")
            print(f"Peak time from trace start: {result['peak_time_seconds']:.3f} seconds")
            print(f"Peak UTC: {result['peak_time_utc']}")
            print(f"SNR onset from trace start: {result['onset_time_seconds']:.3f} seconds")
            print(f"SNR onset UTC: {result['onset_time_utc']}")
            print(f"SNR at onset: {result['snr_at_onset']:.2f}")

        except ValueError as error:
            print(f"Skipping {name}: {error}")

    return results


def calculate_onset_time_delays(results):
    if len(results) != 3:
        raise ValueError("Exactly three station onset times are required.")

    onset_times = {
        result["station"]: UTCDateTime(result["onset_time_utc"]).timestamp
        for result in results
    }

    print("\nSNR Onset Times (UTC)")
    for station, timestamp in onset_times.items():
        print(f"{station}: {UTCDateTime(timestamp)}")

    print("\nPairwise Onset-Time Delays")
    for sta_a, sta_b in combinations(onset_times, 2):
        delay = onset_times[sta_b] - onset_times[sta_a]
        print(f"{sta_b} - {sta_a}: {delay:+.3f} seconds")

    return onset_times


def triangulate_peak_times(peak_times, stations, velocity_km_s, lat_bounds=(26.5, 29.5), lon_bounds=(82.5, 87.5), grid_step_deg=0.02):
    if velocity_km_s <= 0:
        raise ValueError("Velocity must be positive.")

    if grid_step_deg <= 0:
        raise ValueError("Grid step must be positive.")

    station_lookup = {s["sta"]: s for s in stations}
    names = [name for name in peak_times if name in station_lookup]

    if len(names) != 3:
        raise ValueError("Exactly three stations with coordinates and onset times are required.")

    observed = np.array([peak_times[name] for name in names], dtype=float)
    observed_relative = observed - np.mean(observed)

    station_coordinates = [
        (station_lookup[name]["lat"], station_lookup[name]["lon"])
        for name in names
    ]

    latitudes = np.arange(lat_bounds[0], lat_bounds[1] + grid_step_deg * 0.5, grid_step_deg)
    longitudes = np.arange(lon_bounds[0], lon_bounds[1] + grid_step_deg * 0.5, grid_step_deg)

    best = None

    for lat in latitudes:
        for lon in longitudes:
            distances_km = []

            for station_lat, station_lon in station_coordinates:
                distance_m, _, _ = gps2dist_azimuth(lat, lon, station_lat, station_lon)
                distances_km.append(distance_m / 1000.0)

            predicted = np.asarray(distances_km) / velocity_km_s
            predicted_relative = predicted - np.mean(predicted)
            residuals = observed_relative - predicted_relative
            rms = float(np.sqrt(np.mean(residuals ** 2)))

            if best is None or rms < best["rms_seconds"]:
                best = {
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "rms_seconds": rms,
                    "velocity_km_s": float(velocity_km_s),
                    "station_names": names,
                    "residuals_seconds": residuals.tolist(),
                }

    return best


if __name__ == "__main__":
    traces = {}

    for s in stations:
        try:
            trace = fetch_waveform(
                station=s,
                start=start,
                end=end,
                target_fs=100.0,
                freqmin=1.0,
                freqmax=20.0,
            )
            traces[s["sta"]] = trace

        except Exception as e:
            print(f"Error fetching waveform for {s['sta']}: {e}")

    if len(traces) != 3:
        raise RuntimeError(f"Expected 3 station waveforms, received {len(traces)}.")

    trim_to_common_window(traces)

    results = analyze_traces(
        traces,
        fmin=2.0,
        fmax=8.0,
    )

    if len(results) != 3:
        raise RuntimeError(f"Expected 3 analysis results, received {len(results)}.")

    onset_times = calculate_onset_time_delays(results)

    location = triangulate_peak_times(
        peak_times=onset_times,
        stations=stations,
        velocity_km_s=1.5,
        lat_bounds=(26.5, 29.5),
        lon_bounds=(82.5, 87.5),
        grid_step_deg=0.02,
    )

    print("\nEstimated Landslide Location")
    print(f"Latitude: {location['latitude']:.4f}")
    print(f"Longitude: {location['longitude']:.4f}")
    print(f"RMS timing residual: {location['rms_seconds']:.3f} s")
    print(f"Assumed velocity: {location['velocity_km_s']:.2f} km/s")

    print("\nResiduals by station:")
    for name, residual in zip(location["station_names"], location["residuals_seconds"]):
        print(f"{name}: {residual:+.3f} s")
