import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    {"sta": "EQM07", "net": "NP", "loc": "*", "cha": "EHZ", "lat": 27.814941, "lon": 86.713401},
]

start = UTCDateTime("2026-10-10T22:37:00")
end   = UTCDateTime("2026-10-10T22:40:00")


def estimate_dominant_frequency_band(data, fs, fmin, fmax, power_fraction=0.90, nperseg=None,):
    data = np.asarray(data, dtype=float)

    # Remove the DC component.
    data = data - np.mean(data)

    frequencies, psd = welch(data, fs=fs, window="hann", nperseg= min(4096,len(data)), detrend="constant", scaling="density",)

    mask = (frequencies >= fmin) & (frequencies <= fmax)
    band_frequencies = frequencies[mask]
    band_psd = psd[mask]

    df = frequencies[1] - frequencies[0]
    bin_power = band_psd * df
    total_power = float(np.sum(bin_power))

    peak_index = int(np.argmax(band_psd))
    dominant_frequency = float(band_frequencies[peak_index])

    left = peak_index
    right = peak_index

    captured_power = float(bin_power[peak_index])
    target_power = power_fraction * total_power

    while captured_power < target_power:
        left_power = bin_power[left - 1] if left > 0 else -np.inf
        right_power = bin_power[right + 1] if right < len(bin_power) - 1 else -np.inf

        if left_power == -np.inf and right_power == -np.inf:
            break

        if left_power >= right_power:
            left -= 1
            captured_power += float(bin_power[left])
        else:
            right += 1
            captured_power += float(bin_power[right])

    low_frequency = float(band_frequencies[left])
    high_frequency = float(band_frequencies[right])

    return {
        "dominant_frequency_hz": dominant_frequency,
        "band_low_hz": low_frequency,   
        "band_high_hz": high_frequency,
        "band_power_fraction": float(captured_power / total_power),
        "frequencies_hz": frequencies,
        "psd": psd,
    }


def calculate_band_envelope( data, fs, low_frequency, high_frequency, filter_order=4,):
    data = np.asarray(data, dtype=float)
  
    sos = butter(filter_order, [low_frequency, high_frequency], btype="bandpass", fs=fs, output="sos",)

    filtered = sosfiltfilt(sos, data)

    analytic_signal = hilbert(filtered)
    envelope = np.abs(analytic_signal)
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


def analyze_trace(trace, fmin, fmax, power_fraction=0.90, nperseg=None,):
    data = np.asarray(trace.data, dtype=float)
    fs = float(trace.stats.sampling_rate)

    band = estimate_dominant_frequency_band(data=data, fs=fs, fmin=fmin, fmax=fmax, power_fraction=power_fraction, nperseg=nperseg,)

    envelope_result = calculate_band_envelope(
        data=data,
        fs=fs,
        low_frequency=band["band_low_hz"],
        high_frequency=band["band_high_hz"],
    )

    peak_time_utc = trace.stats.starttime + envelope_result["peak_time_seconds"]

    result = dict(band)
    result.update(envelope_result)

    result["peak_time_utc"] = str(peak_time_utc)
    result["trace_start_utc"] = str(trace.stats.starttime)
    result["station_id"] = trace.id
    result["sampling_rate_hz"] = fs

    return result


def analyze_traces( traces, fmin, fmax, power_fraction=0.90, nperseg=None,):
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
            print("Dominant frequency:", f'{result["dominant_frequency_hz"]:.3f} Hz')
            print("Selected band:", f'{result["band_low_hz"]:.3f}–{result["band_high_hz"]:.3f} Hz')
            print("Peak time from trace start:", f'{result["peak_time_seconds"]:.3f} seconds')
            print("Peak UTC:", result["peak_time_utc"])

        except ValueError as error:
            print(f"Skipping {name}: {error}")

    return results


def calculate_peak_time_delays(results):
    if len(results) < 3:
        raise ValueError("Need results from at least three stations.")

    peak_times = {
        result["station"]: UTCDateTime(result["peak_time_utc"]).timestamp
        for result in results
    }

    print("\n Envelope Peak Times")
    for station, timestamp in peak_times.items():
        print(f"{station}: {UTCDateTime(timestamp)}")

    print("\nPairwise Peak-Time Delays")
    for sta_a, sta_b in combinations(peak_times, 2):
        # Positive means station B peaks later than station A.
        delay = peak_times[sta_b] - peak_times[sta_a]
        print(f"{sta_b} - {sta_a}: {delay:+.3f} seconds")

    return peak_times


def triangulate_peak_times(
    peak_times,
    stations,
    velocity_km_s,
    lat_bounds=(26.5, 29.5),
    lon_bounds=(82.5, 87.5),
    grid_step_deg=0.02,
):

    if velocity_km_s <= 0:
        raise ValueError("Velocity must be positive.")

    if grid_step_deg <= 0:
        raise ValueError("Grid step must be positive.")

    station_lookup = {s["sta"]: s for s in stations}

    names = [
        name for name in peak_times
        if name in station_lookup
    ]

    if len(names) != 3:
        raise ValueError(
            "Exactly three stations with coordinates and peak times are required."
        )

    observed = np.array(
        [peak_times[name] for name in names], dtype=float
    )

    # Remove the unknown common event-time offset.
    observed_relative = observed - np.mean(observed)

    station_coordinates = [
        (
            station_lookup[name]["lat"],
            station_lookup[name]["lon"],
        )
        for name in names
    ]

    latitudes = np.arange(
        lat_bounds[0],
        lat_bounds[1] + grid_step_deg * 0.5,
        grid_step_deg,
    )
    longitudes = np.arange(
        lon_bounds[0],
        lon_bounds[1] + grid_step_deg * 0.5,
        grid_step_deg,
    )

    best = None

    for lat in latitudes:
        for lon in longitudes:
            distances_km = []

            for station_lat, station_lon in station_coordinates:
                distance_m, _, _ = gps2dist_azimuth(
                    lat, lon, station_lat, station_lon
                )
                distances_km.append(distance_m / 1000.0)

            # Predicted travel times for this candidate location.
            predicted = (
                np.asarray(distances_km) / velocity_km_s
            )

            # Remove the unknown common origin-time offset.
            predicted_relative = predicted - np.mean(predicted)

            residuals = observed_relative - predicted_relative

            rms = float(np.sqrt(np.mean(residuals**2)))

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
        raise RuntimeError(
            f"Expected 3 station waveforms, received {len(traces)}."
        )

    trim_to_common_window(traces)

    results = analyze_traces(
        traces,
        fmin=2.0,
        fmax=8.0,
    )

    if len(results) != 3:
        raise RuntimeError(
            f"Expected 3 analysis results, received {len(results)}."
        )

    # Step 1: Calculate observed peak-time differences.
    peak_times = calculate_peak_time_delays(results)

    # Step 2: Estimate location using a fixed velocity.
    location = triangulate_peak_times(
        peak_times=peak_times,
        stations=stations,
        velocity_km_s=3.0,  # Illustrative; calibrate before interpreting.
        lat_bounds=(26.5, 29.5),
        lon_bounds=(82.5, 87.5),
        grid_step_deg=0.02,
    )

    print(f"Latitude:  {location['latitude']:.4f}")
    print(f"Longitude: {location['longitude']:.4f}")
    print(f"RMS timing residual: {location['rms_seconds']:.3f} s")
    print(f"Assumed velocity: {location['velocity_km_s']:.2f} km/s")

    print("\nResiduals by station:")
    for name, residual in zip(
        location["station_names"],
        location["residuals_seconds"],
    ):
        print(f"{name}: {residual:+.3f} s")
