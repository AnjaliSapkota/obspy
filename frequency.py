import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime
from scipy.signal import butter, sosfiltfilt, hilbert, welch
from triangulate import fetch_waveform, trim_to_common_window

client = "https://seiscomp.alertnepal.online"

stations = [
    {"sta": "EQM10", "net": "NP", "loc": "*", "cha": "EHZ", "lat": 28.299517, "lon": 83.960148},
    {"sta": "KKN", "net": "NK", "loc": "*", "cha": "BHZ", "lat": 27.8, "lon": 85.279},
    {"sta": "EQM13", "net": "NP", "loc": "*", "cha": "EHZ", "lat": 28.256323, "lon": 85.367569},
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


if __name__ == "__main__":
    traces = {}

    for s in stations:
        try:
            trace = fetch_waveform(station=s, start=start, end=end, target_fs=100.0, freqmin=1.0, freqmax=20.0,)
            traces[s["sta"]] = trace
        except Exception as e:
            print(f"Error fetching waveform for {s['sta']}: {e}")

    trim_to_common_window(traces)

    results = analyze_traces(traces, fmin=2.0, fmax=8.0)
