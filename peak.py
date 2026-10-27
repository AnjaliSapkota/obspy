import csv
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.filter import envelope
from scipy.signal import butter, sosfiltfilt, windows
from pathlib import Path

client = Client("https://seiscomp.alertnepal.online")

OUTPUT_FILE = "envelope_peak_times.csv"

FMIN = 1.0
FMAX = 8.0
SMOOTH_SECONDS = 2.0

stations = [
    {
        "event_id": "LAND01",
        "source_lat": 28.4755,
        "source_lon": 84.9644,
        "net": "NP",
        "sta": "EQM10",
        "loc": "*",
        "cha": "EHZ",
        "station_lat": 28.299517,
        "station_lon": 83.960148,
        "start": "2026-10-10T22:37:00",
        "end": "2026-10-10T22:40:00",
    },
    {
        "event_id": "LAND01",
        "source_lat": 28.4755,
        "source_lon": 84.9644,
        "net": "NP",
        "sta": "EQM13",
        "loc": "*",
        "cha": "EHZ",
        "station_lat": 28.256323,
        "station_lon": 85.367569,
        "start": "2026-10-10T22:37:00",
        "end": "2026-10-10T22:40:00",
    },
    {
        "event_id": "LAND01",
        "source_lat": 28.4755,
        "source_lon": 84.9644,
        "net": "NK",
        "sta": "KKN",
        "loc": "*",
        "cha": "BHZ",
        "station_lat": 27.8000,
        "station_lon": 85.2790,
        "start": "2026-10-10T22:37:00",
        "end": "2026-10-10T22:40:00",
    },
]


def get_waveform_peak(station):
    start = UTCDateTime(station["start"])
    end = UTCDateTime(station["end"])

    stream = client.get_waveforms(
        network=station["net"],
        station=station["sta"],
        location=station["loc"],
        channel=station["cha"],
        starttime=start,
        endtime=end,
    )

    if len(stream) == 0:
        raise ValueError("No waveform data returned")

    stream.merge(method=1, fill_value="interpolate")
    stream.trim(starttime=start, endtime=end, pad=False)

    trace = max(stream, key=lambda tr: tr.stats.npts)

    if trace.stats.npts < 10:
        raise ValueError("Waveform contains too few samples")

    trace.detrend("demean")
    trace.detrend("linear")
    trace.taper(max_percentage=0.05, type="hann")

    sampling_rate = trace.stats.sampling_rate
    nyquist = sampling_rate / 2.0
    high_cutoff = min(FMAX, 0.95 * nyquist)

    if FMIN >= high_cutoff:
        raise ValueError(
            f"Sampling rate {sampling_rate} Hz is too low "
            f"for the {FMIN}-{FMAX} Hz bandpass"
        )

    sos = butter(
        4,
        [FMIN, high_cutoff],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )

    trace.data = sosfiltfilt(sos, trace.data)

    envelope_values = envelope(trace.data)

    smooth_samples = max(
        1,
        int(round(SMOOTH_SECONDS * sampling_rate))
    )

    if smooth_samples > 1:
        kernel = np.ones(smooth_samples) / smooth_samples
        envelope_values = np.convolve(
            envelope_values,
            kernel,
            mode="same",
        )

    peak_index = int(np.argmax(envelope_values))
    peak_time = trace.stats.starttime + peak_index / sampling_rate

    return {
        "event_id": station["event_id"],
        "net": trace.stats.network,
        "sta": trace.stats.station,
        "loc": trace.stats.location,
        "cha": trace.stats.channel,
        "source_lat": station["source_lat"],
        "source_lon": station["source_lon"],
        "station_lat": station["station_lat"],
        "station_lon": station["station_lon"],
        "fetch_start_utc": start.isoformat(),
        "fetch_end_utc": end.isoformat(),
        "trace_start_utc": trace.stats.starttime.isoformat(),
        "trace_end_utc": trace.stats.endtime.isoformat(),
        "sampling_rate_hz": sampling_rate,
        "peak_time_utc": peak_time.isoformat(),
        "peak_offset_seconds": float(
            peak_time - trace.stats.starttime
        ),
        "peak_envelope_amplitude": float(
            envelope_values[peak_index]
        ),
        "filter_low_hz": FMIN,
        "filter_high_hz": high_cutoff,
        "smooth_seconds": SMOOTH_SECONDS,
        "status": "success",
        "error": "",
    }


results = []

for station in stations:
    print(f"\nProcessing {station['net']}.{station['sta']}...")

    try:
        result = get_waveform_peak(station)
        results.append(result)

        print(f"  Peak time: {result['peak_time_utc']}")
        print(
            f"  Offset from trace start: "
            f"{result['peak_offset_seconds']:.3f} seconds"
        )
        print(
            f"  Peak envelope amplitude: "
            f"{result['peak_envelope_amplitude']:.6g}"
        )

    except Exception as exc:
        print(f"  ERROR: {exc}")

        results.append({
            "event_id": station["event_id"],
            "net": station["net"],
            "sta": station["sta"],
            "loc": station["loc"],
            "cha": station["cha"],
            "source_lat": station["source_lat"],
            "source_lon": station["source_lon"],
            "station_lat": station["station_lat"],
            "station_lon": station["station_lon"],
            "fetch_start_utc": station["start"],
            "fetch_end_utc": station["end"],
            "trace_start_utc": "",
            "trace_end_utc": "",
            "sampling_rate_hz": "",
            "peak_time_utc": "",
            "peak_offset_seconds": "",
            "peak_envelope_amplitude": "",
            "filter_low_hz": FMIN,
            "filter_high_hz": FMAX,
            "smooth_seconds": SMOOTH_SECONDS,
            "status": "error",
            "error": str(exc),
        })


if results:
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to: {Path(OUTPUT_FILE).resolve()}")

    successful = [
        result for result in results
        if result["status"] == "success"
    ]

    if len(successful) >= 2:
        reference = successful[0]

        print("\nPeak-time differences relative to the first successful station:")

        for result in successful[1:]:
            difference = (
                UTCDateTime(result["peak_time_utc"])
                - UTCDateTime(reference["peak_time_utc"])
            )

            print(
                f"  {result['sta']} - {reference['sta']}: "
                f"{difference:+.3f} seconds"
            )