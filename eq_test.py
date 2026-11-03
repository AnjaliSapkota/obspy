import threading

import numpy as np

from scipy.signal import (
    butter,
    detrend,
    sosfilt,
    sosfilt_zi,
    resample_poly,
    spectrogram
)

from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient

from flask import Flask, jsonify, render_template_string


app = Flask(__name__)


SERVER_URL = "ring.wscada.net:18000"

BUFFER_SECONDS = 300

UPDATE_INTERVAL = 2000

MAX_WAVEFORM_POINTS = 1500


# Preprocessing configuration

REMOVE_MEAN = True

REMOVE_LINEAR_TREND = True

ENABLE_FILTER = True

FILTER_LOW = 1.0

FILTER_HIGH = 15.0

FILTER_ORDER = 4

ENABLE_RESAMPLING = True

TARGET_FS = 100.0


# STA/LTA configuration

STA_SECONDS = 1.0

LTA_SECONDS = 30.0

TRIGGER_ON = 4.0

TRIGGER_OFF = 1.5

MIN_EVENT_DURATION = 2.0

MAX_EVENT_DURATION = 120.0

DETECTION_HISTORY_SECONDS = (
    LTA_SECONDS + MAX_EVENT_DURATION + 5.0
)

MAX_DETECTION_RECORDS = 100


# Station configuration

STATIONS = [
    {
        "net": "NP",
        "sta": "EQM10",
        "cha": "EHZ",
        "loc": "*",
        "lat": 28.299517,
        "lon": 83.960148
    },
    {
        "net": "NP",
        "sta": "EQM26",
        "cha": "EHZ",
        "loc": "*",
        "lat": 27.3936969,
        "lon": 86.6791885
    },
    {
        "net": "NP",
        "sta": "EQM07",
        "cha": "EHZ",
        "loc": "*",
        "lat": 27.814941,
        "lon": 86.713401
    },
    {
        "net": "NP",
        "sta": "EQM08",
        "cha": "EHZ",
        "loc": "*",
        "lat": 27.831000,
        "lon": 86.65
    },
    {
        "net": "NP",
        "sta": "EQM11",
        "cha": "EHZ",
        "loc": "*",
        "lat": 29.2859556,
        "lon": 81.2741972
    }
]


data_lock = threading.Lock()

station_data = {}

detection_history = []

connection_status = {
    "status": "disconnected",
    "message": "Waiting for SeedLink connection"
}


def reset_detector_state(current_data):
    """Reset the recursive detector when its input configuration changes."""

    current_data["detector_timestamps"] = np.array([], dtype=float)
    current_data["detector_waveform"] = np.array([], dtype=float)
    current_data["last_detection_input_timestamp"] = None

    current_data["sta_value"] = 0.0
    current_data["lta_value"] = 1e-99
    current_data["detector_samples_seen"] = 0

    current_data["current_sta"] = 0.0
    current_data["current_lta"] = 0.0
    current_data["current_sta_lta"] = 0.0

    current_data["lta_frozen"] = False
    current_data["lta_status"] = "UPDATING"

    current_data["trigger_active"] = False
    current_data["trigger_start_timestamp"] = None
    current_data["trigger_duration"] = 0.0
    current_data["trigger_max_ratio"] = 0.0

    current_data["active_event"] = None
    current_data["last_detection"] = None
    current_data["detection_status"] = "Normal"


for station in STATIONS:

    station_id = (
        f"{station['net']}."
        f"{station['sta']}."
        f"{station['cha']}"
    )

    station_data[station_id] = {

        "name": station["sta"],

        "network": station["net"],

        "channel": station["cha"],

        "latitude": station["lat"],

        "longitude": station["lon"],

        "raw_sampling_rate": 0,

        "processed_sampling_rate": 0,

        "raw_timestamps": np.array([], dtype=float),

        "raw_waveform": np.array([], dtype=float),

        "processed_timestamps": np.array([], dtype=float),

        "processed_waveform": np.array([], dtype=float),

        "filtered_timestamps": np.array([], dtype=float),

        "filtered_waveform": np.array([], dtype=float),

        "filter_sos": None,

        "filter_zi": None,

        "filter_signature": None,

        "latest_timestamp": None,

        "preprocessing_error": None
    }

    reset_detector_state(station_data[station_id])


def preprocess_data(station_id, waveform, timestamps, sampling_rate):

    current_data = station_data[station_id]

    waveform = np.asarray(waveform, dtype=float)

    sampling_rate = float(sampling_rate)

    if len(waveform) == 0:
        return

    filter_signature = (
        sampling_rate,
        FILTER_LOW,
        FILTER_HIGH,
        FILTER_ORDER,
        ENABLE_FILTER
    )

    if current_data["filter_signature"] != filter_signature:

        current_data["filter_sos"] = None
        current_data["filter_zi"] = None
        current_data["filter_signature"] = filter_signature

        current_data["filtered_timestamps"] = np.array([], dtype=float)
        current_data["filtered_waveform"] = np.array([], dtype=float)

        current_data["processed_timestamps"] = np.array([], dtype=float)
        current_data["processed_waveform"] = np.array([], dtype=float)

        reset_detector_state(current_data)

    # Use the updated raw buffer as detrending context.
    context_waveform = current_data["raw_waveform"].copy()

    if REMOVE_MEAN:
        context_waveform = context_waveform - np.mean(context_waveform)

    if REMOVE_LINEAR_TREND and len(context_waveform) >= 2:
        context_waveform = detrend(context_waveform, type="linear")

    # Extract only the incoming samples.
    processed_chunk = context_waveform[-len(waveform):]

    if ENABLE_FILTER:

        nyquist_frequency = sampling_rate / 2.0

        if not (0 < FILTER_LOW < FILTER_HIGH < nyquist_frequency):
            raise ValueError(
                f"Invalid filter frequencies for {station_id}. "
                f"Nyquist frequency is {nyquist_frequency} Hz."
            )

        if current_data["filter_sos"] is None:
            current_data["filter_sos"] = butter(
                FILTER_ORDER,
                [FILTER_LOW, FILTER_HIGH],
                btype="bandpass",
                fs=sampling_rate,
                output="sos"
            )

        sos = current_data["filter_sos"]

        if current_data["filter_zi"] is None:
            zi = sosfilt_zi(sos) * processed_chunk[0]
        else:
            zi = current_data["filter_zi"]

        filtered_chunk, final_zi = sosfilt(
            sos,
            processed_chunk,
            zi=zi
        )

        current_data["filter_zi"] = final_zi

    else:
        filtered_chunk = processed_chunk

    current_data["filtered_timestamps"] = np.concatenate([
        current_data["filtered_timestamps"],
        timestamps
    ])

    current_data["filtered_waveform"] = np.concatenate([
        current_data["filtered_waveform"],
        filtered_chunk
    ])

    cutoff_time = timestamps[-1] - BUFFER_SECONDS

    valid_indices = (
        current_data["filtered_timestamps"] >= cutoff_time
    )

    current_data["filtered_timestamps"] = (
        current_data["filtered_timestamps"][valid_indices]
    )

    current_data["filtered_waveform"] = (
        current_data["filtered_waveform"][valid_indices]
    )

    filtered_history = current_data["filtered_waveform"]
    filtered_times = current_data["filtered_timestamps"]

    if ENABLE_RESAMPLING and not np.isclose(sampling_rate, TARGET_FS):

        ratio = (TARGET_FS / sampling_rate)

        from fractions import Fraction

        ratio = Fraction(ratio).limit_denominator(1000)

        up = ratio.numerator
        down = ratio.denominator

        resampled_waveform = resample_poly(
            filtered_history,
            up,
            down
        )

        resampled_timestamps = (
            filtered_times[0]
            + np.arange(len(resampled_waveform)) / TARGET_FS
        )

        current_data["processed_sampling_rate"] = TARGET_FS

    else:

        resampled_waveform = filtered_history.copy()
        resampled_timestamps = filtered_times.copy()

        current_data["processed_sampling_rate"] = sampling_rate

    valid_indices = resampled_timestamps >= cutoff_time

    current_data["processed_timestamps"] = (
        resampled_timestamps[valid_indices]
    )

    current_data["processed_waveform"] = (
        resampled_waveform[valid_indices]
    )


def finish_trigger(station_id, current_data, end_timestamp):
    """Close the current trigger and save its record."""

    active_event = current_data["active_event"]

    if active_event is None:
        return

    active_event["end_timestamp"] = float(end_timestamp)

    active_event["duration"] = max(
        0.0,
        active_event["end_timestamp"] - active_event["start_timestamp"]
    )

    active_event["status"] = "ENDED"

    record = {
        "station_id": station_id,
        "start_timestamp": active_event["start_timestamp"],
        "end_timestamp": active_event["end_timestamp"],
        "duration": active_event["duration"],
        "max_sta_lta": active_event["max_sta_lta"],
        "max_abs_amplitude": active_event["max_abs_amplitude"],
        "status": (
            "POSSIBLE EARTHQUAKE"
            if MIN_EVENT_DURATION <= active_event["duration"] <= MAX_EVENT_DURATION
            else (
                "REJECTED - TOO SHORT"
                if active_event["duration"] < MIN_EVENT_DURATION
                else "REJECTED - TOO LONG"
            )
        )
    }

    detection_history.append(record)

    if len(detection_history) > MAX_DETECTION_RECORDS:
        detection_history.pop(0)

    current_data["last_detection"] = record.copy()

    print(
        f"[{station_id}] TRIGGER ENDED | "
        f"Duration: {record['duration']:.2f}s | "
        f"Max ratio: {record['max_sta_lta']:.2f} | "
        f"LTA will resume updating"
    )

    current_data["active_event"] = None
    current_data["trigger_active"] = False
    current_data["trigger_start_timestamp"] = None
    current_data["trigger_duration"] = 0.0
    current_data["trigger_max_ratio"] = 0.0

    current_data["lta_frozen"] = False
    current_data["lta_status"] = "UPDATING"

    current_data["detection_status"] = "Normal"


def run_sta_lta(station_id, new_waveform, new_timestamps, sampling_rate):

    current_data = station_data[station_id]

    if len(new_waveform) == 0:
        return

    sampling_rate = float(sampling_rate)

    nsta = int(STA_SECONDS * sampling_rate)
    nlta = int(LTA_SECONDS * sampling_rate)

    if nsta < 1 or nlta <= nsta:
        raise ValueError(
            f"Invalid STA/LTA window sizes for {station_id}"
        )

    new_waveform = np.asarray(new_waveform, dtype=float)
    new_timestamps = np.asarray(new_timestamps, dtype=float)

    # Keep the detector input history for dashboard/debugging.
    current_data["detector_timestamps"] = np.concatenate([
        current_data["detector_timestamps"],
        new_timestamps
    ])

    current_data["detector_waveform"] = np.concatenate([
        current_data["detector_waveform"],
        new_waveform
    ])

    cutoff_time = (
        new_timestamps[-1] - DETECTION_HISTORY_SECONDS
    )

    valid_indices = (
        current_data["detector_timestamps"] >= cutoff_time
    )

    current_data["detector_timestamps"] = (
        current_data["detector_timestamps"][valid_indices]
    )

    current_data["detector_waveform"] = (
        current_data["detector_waveform"][valid_indices]
    )

    sta_value = current_data["sta_value"]
    lta_value = current_data["lta_value"]
    samples_seen = current_data["detector_samples_seen"]

    alpha_sta = 1.0 / nsta
    alpha_lta = 1.0 / nlta

    for sample, timestamp in zip(new_waveform, new_timestamps):

        # Invalid samples must not corrupt the recursive state.
        if not np.isfinite(sample) or not np.isfinite(timestamp):
            continue

        squared_amplitude = float(sample) * float(sample)

        # STA always updates, including while LTA is frozen.
        sta_value = (
            alpha_sta * squared_amplitude
            + (1.0 - alpha_sta) * sta_value
        )

        # LTA updates only when this station has no active trigger.
        if not current_data["trigger_active"]:
            lta_value = (
                alpha_lta * squared_amplitude
                + (1.0 - alpha_lta) * lta_value
            )

        samples_seen += 1

        # Do not trigger until a full LTA window has been observed.
        if samples_seen <= nlta:
            current_ratio = 0.0
        else:
            if lta_value > 0.0 and np.isfinite(lta_value):
                current_ratio = sta_value / lta_value
            else:
                current_ratio = 0.0

            if not np.isfinite(current_ratio):
                current_ratio = 0.0

        current_data["current_sta"] = float(sta_value)
        current_data["current_lta"] = float(lta_value)
        current_data["current_sta_lta"] = float(current_ratio)

        # Trigger starts when the ratio crosses trigger-on.
        if (
            not current_data["trigger_active"]
            and samples_seen > nlta
            and current_ratio >= TRIGGER_ON
        ):

            current_data["trigger_active"] = True
            current_data["lta_frozen"] = True
            current_data["lta_status"] = "FROZEN"

            current_data["trigger_start_timestamp"] = float(timestamp)
            current_data["trigger_duration"] = 0.0
            current_data["trigger_max_ratio"] = float(current_ratio)

            current_data["active_event"] = {
                "station_id": station_id,
                "start_timestamp": float(timestamp),
                "end_timestamp": float(timestamp),
                "duration": 0.0,
                "max_sta_lta": float(current_ratio),
                "max_abs_amplitude": abs(float(sample)),
                "status": "ACTIVE"
            }

            current_data["detection_status"] = "Possible Earthquake"

            print(
                f"[{station_id}] TRIGGER STARTED | "
                f"Time: {timestamp:.3f} | "
                f"Ratio: {current_ratio:.2f} | "
                f"Frozen LTA: {lta_value:.6e}"
            )

        elif current_data["trigger_active"]:

            active_event = current_data["active_event"]

            active_event["end_timestamp"] = float(timestamp)

            active_event["duration"] = max(
                0.0,
                float(timestamp) - active_event["start_timestamp"]
            )

            active_event["max_sta_lta"] = max(
                active_event["max_sta_lta"],
                float(current_ratio)
            )

            active_event["max_abs_amplitude"] = max(
                active_event["max_abs_amplitude"],
                abs(float(sample))
            )

            current_data["trigger_duration"] = active_event["duration"]

            current_data["trigger_max_ratio"] = (
                active_event["max_sta_lta"]
            )

            # Trigger ends when ratio reaches or falls below trigger-off.
            if current_ratio <= TRIGGER_OFF:
                finish_trigger(
                    station_id,
                    current_data,
                    float(timestamp)
                )

    current_data["sta_value"] = float(sta_value)
    current_data["lta_value"] = float(lta_value)
    current_data["detector_samples_seen"] = samples_seen

    # Keep the displayed LTA reference synchronized with its recursive state.
    current_data["current_sta"] = float(sta_value)
    current_data["current_lta"] = float(lta_value)

    if current_data["trigger_active"]:
        current_data["detection_status"] = "Possible Earthquake"
        current_data["lta_status"] = "FROZEN"
    else:
        current_data["detection_status"] = "Normal"
        current_data["lta_status"] = "UPDATING"


def handle_data(trace):

    network = trace.stats.network
    station = trace.stats.station
    channel = trace.stats.channel

    station_id = f"{network}.{station}.{channel}"

    if station_id not in station_data:
        return

    if len(trace.data) == 0:
        return

    waveform = np.asarray(trace.data, dtype=float)
    timestamps = trace.times("timestamp")
    sampling_rate = float(trace.stats.sampling_rate)

    with data_lock:

        current_data = station_data[station_id]

        current_data["raw_timestamps"] = np.concatenate([
            current_data["raw_timestamps"],
            timestamps
        ])

        current_data["raw_waveform"] = np.concatenate([
            current_data["raw_waveform"],
            waveform
        ])

        current_data["raw_sampling_rate"] = sampling_rate
        current_data["latest_timestamp"] = float(timestamps[-1])

        cutoff_time = timestamps[-1] - BUFFER_SECONDS

        valid_indices = (
            current_data["raw_timestamps"] >= cutoff_time
        )

        current_data["raw_timestamps"] = (
            current_data["raw_timestamps"][valid_indices]
        )

        current_data["raw_waveform"] = (
            current_data["raw_waveform"][valid_indices]
        )

        try:

            preprocess_data(
                station_id,
                waveform,
                timestamps,
                sampling_rate
            )

            processed_times = current_data["processed_timestamps"]
            processed_values = current_data["processed_waveform"]

            last_timestamp = (
                current_data["last_detection_input_timestamp"]
            )

            if last_timestamp is None:
                new_mask = (
                    processed_times >= timestamps[0] - 1.0 / TARGET_FS
                )
            else:
                new_mask = (
                    processed_times > last_timestamp + 1e-8
                )

            new_times = processed_times[new_mask]
            new_values = processed_values[new_mask]

            if len(new_values) > 0:

                current_data["last_detection_input_timestamp"] = (
                    float(new_times[-1])
                )

                run_sta_lta(
                    station_id,
                    new_values,
                    new_times,
                    current_data["processed_sampling_rate"]
                )

            connection_status["status"] = "connected"
            connection_status["message"] = "Receiving live data"

            current_data["preprocessing_error"] = None

        except Exception as error:

            current_data["preprocessing_error"] = str(error)

            print(
                f"Processing error for {station_id}: {error}"
            )


# SeedLink background thread

def seedlink_worker():

    try:

        client = EasySeedLinkClient(
            SERVER_URL,
            autoconnect=False
        )

        client.conn.timeout = 10
        client.connect()
        client.on_data = handle_data

        for station in STATIONS:

            client.select_stream(
                station["net"],
                station["sta"],
                station["cha"]
            )

            print(
                f"Subscribed: {station['net']}."
                f"{station['sta']}.{station['cha']}"
            )

        with data_lock:
            connection_status["status"] = "connected"
            connection_status["message"] = "Connected, waiting for data"

        print(f"Connected to SeedLink: {SERVER_URL}")

        client.run()

    except Exception as error:

        print("SeedLink error:", error)

        with data_lock:
            connection_status["status"] = "error"
            connection_status["message"] = str(error)


# Spectrogram

def calculate_spectrogram(
    waveform,
    sampling_rate,
    start_timestamp
):

    if len(waveform) < 8 or sampling_rate <= 0:
        return {
            "times": [],
            "frequencies": [],
            "power": []
        }

    nperseg = min(256, len(waveform))

    frequencies, times, power = spectrogram(
        waveform,
        fs=sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="density",
        mode="psd"
    )

    power_db = 10 * np.log10(
        np.maximum(power, 1e-20)
    )

    absolute_times = start_timestamp + times

    time_step = max(1, len(absolute_times) // 150)
    frequency_step = max(1, len(frequencies) // 100)

    absolute_times = absolute_times[::time_step]
    frequencies = frequencies[::frequency_step]

    power_db = power_db[
        ::frequency_step,
        ::time_step
    ]

    return {
        "times": absolute_times.tolist(),
        "frequencies": frequencies.tolist(),
        "power": power_db.tolist()
    }


def get_station_output(station_id):

    with data_lock:

        current_data = station_data[station_id]

        raw_timestamps = current_data["raw_timestamps"].copy()
        raw_waveform = current_data["raw_waveform"].copy()

        processed_timestamps = (
            current_data["processed_timestamps"].copy()
        )

        processed_waveform = (
            current_data["processed_waveform"].copy()
        )

        raw_sampling_rate = current_data["raw_sampling_rate"]
        processed_sampling_rate = current_data["processed_sampling_rate"]
        latest_timestamp = current_data["latest_timestamp"]

        detection_ratio = current_data["current_sta_lta"]
        detection_status = current_data["detection_status"]

        active_event = (
            current_data["active_event"].copy()
            if current_data["active_event"] is not None
            else None
        )

        last_detection = (
            current_data["last_detection"].copy()
            if current_data["last_detection"] is not None
            else None
        )

        lta_status = current_data["lta_status"]
        current_sta = current_data["current_sta"]
        current_lta = current_data["current_lta"]
        trigger_active = current_data["trigger_active"]
        trigger_start_timestamp = current_data["trigger_start_timestamp"]
        trigger_duration = current_data["trigger_duration"]
        trigger_max_ratio = current_data["trigger_max_ratio"]

        preprocessing_error = current_data["preprocessing_error"]

        name = current_data["name"]
        network = current_data["network"]
        channel = current_data["channel"]

    raw_step = max(1, len(raw_waveform) // MAX_WAVEFORM_POINTS)
    processed_step = max(1, len(processed_waveform) // MAX_WAVEFORM_POINTS)

    raw_output = {
        "times": raw_timestamps[::raw_step].tolist(),
        "values": raw_waveform[::raw_step].tolist()
    }

    processed_output = {
        "times": processed_timestamps[::processed_step].tolist(),
        "values": processed_waveform[::processed_step].tolist()
    }

    if len(raw_waveform) > 0:
        spectrogram_output = calculate_spectrogram(
            raw_waveform,
            raw_sampling_rate,
            raw_timestamps[0]
        )
    else:
        spectrogram_output = {
            "times": [],
            "frequencies": [],
            "power": []
        }

    return {

        "station_id": station_id,

        "name": name,

        "network": network,

        "channel": channel,

        "raw_sampling_rate": raw_sampling_rate,

        "processed_sampling_rate": processed_sampling_rate,

        "raw_waveform": raw_output,

        "processed_waveform": processed_output,

        "spectrogram": spectrogram_output,

        "latest_timestamp": latest_timestamp,

        "status": "receiving" if latest_timestamp is not None else "waiting",

        "preprocessing_error": preprocessing_error,

        "detection": {

            "ratio": detection_ratio,

            "status": detection_status,

            "trigger_on": TRIGGER_ON,

            "trigger_off": TRIGGER_OFF,

            "sta_seconds": STA_SECONDS,

            "lta_seconds": LTA_SECONDS,

            "nsta": int(STA_SECONDS * processed_sampling_rate),

            "nlta": int(LTA_SECONDS * processed_sampling_rate),

            "sta_value": current_sta,

            "lta_value": current_lta,

            "lta_status": lta_status,

            "trigger_active": trigger_active,

            "trigger_start_timestamp": trigger_start_timestamp,

            "trigger_duration": trigger_duration,

            "trigger_max_ratio": trigger_max_ratio,

            "active_event": active_event,

            "last_detection": last_detection
        },

        "filter_config": {

            "enabled": ENABLE_FILTER,

            "low": FILTER_LOW,

            "high": FILTER_HIGH,

            "order": FILTER_ORDER,

            "resampling": ENABLE_RESAMPLING,

            "target_fs": TARGET_FS
        }
    }


HTML_PAGE = """

<!DOCTYPE html>
<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Seismic Monitoring Dashboard</title>

    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            padding: 20px;
            background: #f1f5f9;
            font-family: Arial, sans-serif;
            color: #1e293b;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 25px;
        }

        h1 {
            margin: 0;
            font-size: 24px;
        }

        .connection {
            padding: 9px 15px;
            border-radius: 20px;
            background: #e2e8f0;
            font-size: 13px;
        }

        .controls {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        select {
            padding: 9px 12px;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            background: white;
        }

        .station-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }

        .station-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border: 2px solid transparent;
        }

        .station-card.detected {
            border-color: #dc2626;
        }

        .station-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }

        .station-header h2 {
            margin: 0;
            font-size: 18px;
        }

        .station-info {
            color: #64748b;
            font-size: 13px;
            margin-top: 8px;
        }

        .station-status {
            font-size: 13px;
            font-weight: bold;
            color: #64748b;
        }

        .detected-text {
            color: #dc2626;
        }

        .normal-text {
            color: #16a34a;
        }

        .frozen-text {
            color: #d97706;
        }

        .updating-text {
            color: #16a34a;
        }

        .detection-panel {
            margin-top: 15px;
            padding: 15px;
            background: #f8fafc;
            border-radius: 8px;
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }

        .detection-panel.detected {
            background: #fef2f2;
        }

        .metric-label {
            font-size: 12px;
            color: #64748b;
        }

        .metric-value {
            font-size: 16px;
            font-weight: bold;
            margin-top: 5px;
            overflow-wrap: anywhere;
        }

        .plot {
            width: 100%;
            height: 300px;
        }

        .spectrogram {
            width: 100%;
            height: 300px;
        }

        .plot-title {
            margin-top: 20px;
            font-size: 14px;
            font-weight: bold;
        }

        .filter-info {
            background: #f8fafc;
            padding: 10px;
            border-radius: 6px;
            font-size: 12px;
            color: #475569;
            margin-top: 12px;
        }

        .error {
            color: #dc2626;
        }

        .history-section {
            background: white;
            padding: 20px;
            margin-top: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 13px;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }

        th {
            background: #f8fafc;
        }

        .possible {
            color: #dc2626;
            font-weight: bold;
        }

        @media (max-width: 700px) {
            .detection-panel {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (min-width: 1200px) {
            .station-grid {
                grid-template-columns: 1fr 1fr;
            }
        }

    </style>

</head>


<body>

    <header>

        <div>
            <h1>Seismic Monitoring Dashboard</h1>
            <p>Live Waveform, Preprocessing and STA/LTA Detection</p>
        </div>

        <div class="connection" id="connection-status">
            Connecting...
        </div>

    </header>


    <div class="controls">

        <label for="waveform-mode">Waveform display:</label>

        <select id="waveform-mode">
            <option value="raw">Raw waveform</option>
            <option value="processed">Processed waveform</option>
        </select>

    </div>


    <div class="station-grid">

        {% for station_id in stations %}

        <div class="station-card" id="card-{{ loop.index0 }}">

            <div class="station-header">

                <h2>{{ station_id }}</h2>

                <span class="station-status" id="status-{{ loop.index0 }}">
                    Waiting for data
                </span>

            </div>

            <div class="station-info" id="info-{{ loop.index0 }}">
                Sampling rates: Waiting...
            </div>

            <div class="filter-info" id="filter-{{ loop.index0 }}">
                Filter configuration: Waiting...
            </div>

            <div class="error" id="error-{{ loop.index0 }}"></div>

            <div class="plot-title">STA/LTA Detection and LTA Freeze</div>

            <div class="detection-panel" id="detection-panel-{{ loop.index0 }}">

                <div>
                    <div class="metric-label">Current STA/LTA ratio</div>
                    <div class="metric-value" id="ratio-{{ loop.index0 }}">0.00</div>
                </div>

                <div>
                    <div class="metric-label">Trigger status</div>
                    <div class="metric-value" id="detection-status-{{ loop.index0 }}">Normal</div>
                </div>

                <div>
                    <div class="metric-label">LTA status</div>
                    <div class="metric-value" id="lta-status-{{ loop.index0 }}">UPDATING</div>
                </div>

                <div>
                    <div class="metric-label">Current STA value</div>
                    <div class="metric-value" id="sta-value-{{ loop.index0 }}">0</div>
                </div>

                <div>
                    <div class="metric-label">LTA reference value</div>
                    <div class="metric-value" id="lta-value-{{ loop.index0 }}">0</div>
                </div>

                <div>
                    <div class="metric-label">Maximum trigger ratio</div>
                    <div class="metric-value" id="max-ratio-{{ loop.index0 }}">--</div>
                </div>

                <div>
                    <div class="metric-label">Trigger start</div>
                    <div class="metric-value" id="trigger-start-{{ loop.index0 }}">--</div>
                </div>

                <div>
                    <div class="metric-label">Trigger duration</div>
                    <div class="metric-value" id="duration-{{ loop.index0 }}">--</div>
                </div>

                <div>
                    <div class="metric-label">Trigger thresholds</div>
                    <div class="metric-value" id="thresholds-{{ loop.index0 }}">4.0 / 1.5</div>
                </div>

            </div>

            <div class="plot-title" id="waveform-title-{{ loop.index0 }}">
                Raw Waveform
            </div>

            <div class="plot" id="waveform-{{ loop.index0 }}"></div>

            <div class="plot-title">Raw Data Spectrogram</div>

            <div class="spectrogram" id="spectrogram-{{ loop.index0 }}"></div>

            <div class="station-info" id="timestamp-{{ loop.index0 }}">
                Latest data: Waiting...
            </div>

        </div>

        {% endfor %}

    </div>


    <div class="history-section">

        <h2>Recent Detection History</h2>

        <p>Completed station-level STA/LTA triggers</p>

        <div style="overflow-x:auto;">

            <table>

                <thead>
                    <tr>
                        <th>Station</th>
                        <th>Start Time</th>
                        <th>End Time</th>
                        <th>Duration (s)</th>
                        <th>Maximum STA/LTA</th>
                        <th>Maximum Amplitude</th>
                        <th>Status</th>
                    </tr>
                </thead>

                <tbody id="detection-history">
                    <tr>
                        <td colspan="7">No detections recorded</td>
                    </tr>
                </tbody>

            </table>

        </div>

    </div>


    <script>

        const stationIds = {{ stations | list | tojson }};

        const waveformMode = document.getElementById("waveform-mode");


        function formatTimestamp(timestamp) {

            if (timestamp === null || timestamp === undefined) {
                return "--";
            }

            return new Date(timestamp * 1000).toLocaleString();

        }


        function initializePlots() {

            stationIds.forEach((stationId, index) => {

                Plotly.newPlot(
                    `waveform-${index}`,
                    [{
                        x: [],
                        y: [],
                        type: "scatter",
                        mode: "lines",
                        line: {
                            color: "#2563eb",
                            width: 1
                        }
                    }],
                    {
                        title: "Amplitude vs Time",
                        xaxis: {
                            title: "Time",
                            type: "date"
                        },
                        yaxis: {
                            title: "Amplitude"
                        }
                    },
                    {
                        responsive: true
                    }
                );


                Plotly.newPlot(
                    `spectrogram-${index}`,
                    [{
                        x: [],
                        y: [],
                        z: [],
                        type: "heatmap",
                        colorscale: "Jet",
                        colorbar: {
                            title: "Power (dB)"
                        }
                    }],
                    {
                        title: "Frequency vs Time",
                        xaxis: {
                            title: "Time",
                            type: "date"
                        },
                        yaxis: {
                            title: "Frequency (Hz)"
                        }
                    },
                    {
                        responsive: true
                    }
                );

            });

        }


        function updateDetection(station, index) {

            const detection = station.detection;

            const active = detection.trigger_active;

            const detected = detection.status === "Possible Earthquake";

            const card = document.getElementById(`card-${index}`);

            const panel = document.getElementById(
                `detection-panel-${index}`
            );

            card.classList.toggle("detected", detected);
            panel.classList.toggle("detected", detected);


            const statusElement = document.getElementById(
                `detection-status-${index}`
            );

            statusElement.textContent = detection.status;

            statusElement.className =
                detected
                ? "metric-value detected-text"
                : "metric-value normal-text";


            const ltaStatusElement = document.getElementById(
                `lta-status-${index}`
            );

            ltaStatusElement.textContent = detection.lta_status;

            ltaStatusElement.className =
                detection.lta_status === "FROZEN"
                ? "metric-value frozen-text"
                : "metric-value updating-text";


            document.getElementById(
                `ratio-${index}`
            ).textContent = detection.ratio.toFixed(3);

            document.getElementById(
                `sta-value-${index}`
            ).textContent = detection.sta_value.toExponential(4);

            document.getElementById(
                `lta-value-${index}`
            ).textContent = detection.lta_value.toExponential(4);

            document.getElementById(
                `thresholds-${index}`
            ).textContent =
                `${detection.trigger_on} / ${detection.trigger_off}`;


            const lastDetection = detection.last_detection;

            if (active) {

                document.getElementById(
                    `trigger-start-${index}`
                ).textContent = formatTimestamp(
                    detection.trigger_start_timestamp
                );

                document.getElementById(
                    `duration-${index}`
                ).textContent =
                    `${detection.trigger_duration.toFixed(2)} s`;

                document.getElementById(
                    `max-ratio-${index}`
                ).textContent =
                    detection.trigger_max_ratio.toFixed(2);

            } else if (lastDetection) {

                document.getElementById(
                    `trigger-start-${index}`
                ).textContent = formatTimestamp(
                    lastDetection.start_timestamp
                );

                document.getElementById(
                    `duration-${index}`
                ).textContent =
                    `${lastDetection.duration.toFixed(2)} s`;

                document.getElementById(
                    `max-ratio-${index}`
                ).textContent =
                    lastDetection.max_sta_lta.toFixed(2);

            } else {

                document.getElementById(
                    `trigger-start-${index}`
                ).textContent = "--";

                document.getElementById(
                    `duration-${index}`
                ).textContent = "--";

                document.getElementById(
                    `max-ratio-${index}`
                ).textContent = "--";

            }

        }


        function updateDetectionHistory(history) {

            const tableBody = document.getElementById(
                "detection-history"
            );

            if (history.length === 0) {

                tableBody.innerHTML = `
                    <tr>
                        <td colspan="7">No detections recorded</td>
                    </tr>
                `;

                return;
            }

            tableBody.innerHTML = "";

            [...history].reverse().forEach(event => {

                const row = document.createElement("tr");

                const values = [
                    event.station_id,
                    formatTimestamp(event.start_timestamp),
                    formatTimestamp(event.end_timestamp),
                    event.duration.toFixed(2),
                    event.max_sta_lta.toFixed(2),
                    event.max_abs_amplitude.toExponential(4),
                    event.status
                ];

                values.forEach((value, index) => {

                    const cell = document.createElement("td");

                    cell.textContent = value;

                    if (index === 6) {
                        cell.className =
                            event.status === "POSSIBLE EARTHQUAKE"
                            ? "possible"
                            : "";
                    }

                    row.appendChild(cell);

                });

                tableBody.appendChild(row);

            });

        }


        async function updateDashboard() {

            try {

                const response = await fetch("/api/data");

                if (!response.ok) {
                    throw new Error("API request failed");
                }

                const data = await response.json();

                const connection = data.connection;

                const connectionElement = document.getElementById(
                    "connection-status"
                );

                connectionElement.textContent = connection.message;

                connectionElement.style.background =
                    connection.status === "connected"
                    ? "#dcfce7"
                    : "#fee2e2";


                stationIds.forEach((stationId, index) => {

                    const station = data.stations[stationId];

                    if (!station) {
                        return;
                    }

                    document.getElementById(
                        `status-${index}`
                    ).textContent = station.status;


                    document.getElementById(
                        `info-${index}`
                    ).textContent =
                        `Original sampling rate: ` +
                        `${station.raw_sampling_rate} Hz | ` +
                        `Processed sampling rate: ` +
                        `${station.processed_sampling_rate} Hz`;


                    const filter = station.filter_config;

                    document.getElementById(
                        `filter-${index}`
                    ).textContent =
                        `Butterworth Bandpass: ` +
                        `${filter.low}-${filter.high} Hz | ` +
                        `Order: ${filter.order} | ` +
                        `Resampling: ${filter.resampling
                            ? filter.target_fs + " Hz"
                            : "Disabled"}`;


                    document.getElementById(
                        `error-${index}`
                    ).textContent =
                        station.preprocessing_error || "";


                    document.getElementById(
                        `timestamp-${index}`
                    ).textContent = station.latest_timestamp !== null
                        ? `Latest data: ${formatTimestamp(
                            station.latest_timestamp
                          )}`
                        : "Latest data: Waiting...";


                    updateDetection(station, index);


                    const selectedWaveform =
                        waveformMode.value === "raw"
                        ? station.raw_waveform
                        : station.processed_waveform;


                    document.getElementById(
                        `waveform-title-${index}`
                    ).textContent =
                        waveformMode.value === "raw"
                        ? "Raw Waveform"
                        : "Processed Waveform";


                    const waveformTimes = selectedWaveform.times.map(
                        timestamp => new Date(timestamp * 1000)
                    );


                    Plotly.react(
                        `waveform-${index}`,
                        [{
                            x: waveformTimes,
                            y: selectedWaveform.values,
                            type: "scatter",
                            mode: "lines",
                            line: {
                                color: waveformMode.value === "raw"
                                    ? "#2563eb"
                                    : "#059669",
                                width: 1
                            }
                        }],
                        {
                            title: "Amplitude vs Time",
                            xaxis: {
                                title: "Time",
                                type: "date"
                            },
                            yaxis: {
                                title: "Amplitude"
                            }
                        },
                        {
                            responsive: true
                        }
                    );


                    const spectrogram = station.spectrogram;

                    const spectrogramTimes = spectrogram.times.map(
                        timestamp => new Date(timestamp * 1000)
                    );


                    Plotly.react(
                        `spectrogram-${index}`,
                        [{
                            x: spectrogramTimes,
                            y: spectrogram.frequencies,
                            z: spectrogram.power,
                            type: "heatmap",
                            colorscale: "Jet",
                            colorbar: {
                                title: "Power (dB)"
                            }
                        }],
                        {
                            title: "Raw Data Spectrogram",
                            xaxis: {
                                title: "Time",
                                type: "date"
                            },
                            yaxis: {
                                title: "Frequency (Hz)"
                            }
                        },
                        {
                            responsive: true
                        }
                    );

                });


                updateDetectionHistory(data.detection_history);

            } catch (error) {

                console.error(
                    "Dashboard update error:",
                    error
                );

            }

        }


        waveformMode.addEventListener("change", updateDashboard);

        initializePlots();

        updateDashboard();

        setInterval(updateDashboard, {{ update_interval }});

    </script>

</body>

</html>

"""


@app.route("/")
def dashboard():

    return render_template_string(
        HTML_PAGE,
        stations=list(station_data.keys()),
        update_interval=UPDATE_INTERVAL
    )


@app.route("/api/data")
def api_data():

    output = {}

    for station_id in station_data:
        output[station_id] = get_station_output(station_id)

    with data_lock:

        status = connection_status.copy()

        history = [
            record.copy()
            for record in detection_history
        ]

    return jsonify({

        "connection": status,

        "stations": output,

        "detection_history": history

    })


if __name__ == "__main__":

    seedlink_thread = threading.Thread(
        target=seedlink_worker,
        daemon=True
    )

    seedlink_thread.start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True
    )
