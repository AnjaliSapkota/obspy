
import threading
from fractions import Fraction

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
from obspy.signal.trigger import recursive_sta_lta, trigger_onset

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

        # Sampling rates
        "raw_sampling_rate": 0,

        "processed_sampling_rate": 0,

        # Raw buffers
        "raw_timestamps": np.array([], dtype=float),

        "raw_waveform": np.array([], dtype=float),

        # Processed buffers
        "processed_timestamps": np.array([], dtype=float),

        "processed_waveform": np.array([], dtype=float),

        # Filtered history
        "filtered_timestamps": np.array([], dtype=float),

        "filtered_waveform": np.array([], dtype=float),

        # Butterworth filter state
        "filter_sos": None,

        "filter_zi": None,

        "filter_signature": None,

        # STA/LTA input history
        "detector_timestamps": np.array([], dtype=float),

        "detector_waveform": np.array([], dtype=float),

        "last_detection_input_timestamp": None,

        # STA/LTA output
        "current_sta_lta": 0.0,

        "detection_status": "Normal",

        "active_event": None,

        "last_detection": None,

        "ignore_until_below_off": False,

        "preprocessing_error": None
    }



def preprocess_data(station_id, waveform, timestamps, sampling_rate):

    current_data = station_data[station_id]

    waveform = np.asarray(waveform, dtype=float)

    sampling_rate = float(sampling_rate)

    if len(waveform) == 0:
        return

    # Reset filter state if the sampling rate or settings change
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

        current_data["last_detection_input_timestamp"] = None

        current_data["detector_timestamps"] = np.array([], dtype=float)

        current_data["detector_waveform"] = np.array([], dtype=float)

        current_data["active_event"] = None

        current_data["detection_status"] = "Normal"

        current_data["ignore_until_below_off"] = False

    # Use the updated raw buffer as detrending context
    context_waveform = current_data["raw_waveform"].copy()

    # Demean
    if REMOVE_MEAN:

        context_waveform = (
            context_waveform - np.mean(context_waveform)
        )

    # Linear detrending
    if REMOVE_LINEAR_TREND and len(context_waveform) >= 2:

        context_waveform = detrend(
            context_waveform,
            type="linear"
        )

    # Extract only the incoming samples
    processed_chunk = context_waveform[-len(waveform):]

    # Butterworth bandpass filtering
    if ENABLE_FILTER:

        nyquist_frequency = sampling_rate / 2.0

        if not (
            0 < FILTER_LOW < FILTER_HIGH < nyquist_frequency
        ):

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

    # Append filtered samples
    current_data["filtered_timestamps"] = np.concatenate([
        current_data["filtered_timestamps"],
        timestamps
    ])

    current_data["filtered_waveform"] = np.concatenate([
        current_data["filtered_waveform"],
        filtered_chunk
    ])

    # Keep five minutes of filtered data
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

    # Optional resampling
    filtered_history = current_data["filtered_waveform"]

    filtered_times = current_data["filtered_timestamps"]

    if ENABLE_RESAMPLING and not np.isclose(
        sampling_rate,
        TARGET_FS
    ):

        ratio = Fraction(
            TARGET_FS / sampling_rate
        ).limit_denominator(1000)

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

    # Update processed buffer
    valid_indices = (
        resampled_timestamps >= cutoff_time
    )

    current_data["processed_timestamps"] = (
        resampled_timestamps[valid_indices]
    )

    current_data["processed_waveform"] = (
        resampled_waveform[valid_indices]
    )



def run_sta_lta(station_id, new_waveform, new_timestamps, sampling_rate):

    current_data = station_data[station_id]

    if len(new_waveform) == 0:
        return

    sampling_rate = float(sampling_rate)

    # Convert STA and LTA seconds into samples
    nsta = int(STA_SECONDS * sampling_rate)

    nlta = int(LTA_SECONDS * sampling_rate)

    if nsta < 1 or nlta <= nsta:

        raise ValueError(
            f"Invalid STA/LTA window sizes for {station_id}"
        )

    # Append only new processed samples
    current_data["detector_timestamps"] = np.concatenate([
        current_data["detector_timestamps"],
        new_timestamps
    ])

    current_data["detector_waveform"] = np.concatenate([
        current_data["detector_waveform"],
        new_waveform
    ])

    # Keep LTA context and maximum event history
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

    detector_times = current_data["detector_timestamps"]

    detector_waveform = current_data["detector_waveform"]

    # Wait until sufficient LTA history is available
    if len(detector_waveform) <= nlta:
        return

    # Calculate recursive STA/LTA
    characteristic_function = recursive_sta_lta(
        detector_waveform,
        nsta,
        nlta
    )

    # Current STA/LTA ratio
    current_ratio = float(characteristic_function[-1])

    current_data["current_sta_lta"] = current_ratio

    # Find where the newly received data starts
    new_start_index = np.searchsorted(
        detector_times,
        new_timestamps[0],
        side="left"
    )

    # Identify trigger intervals
    trigger_intervals = trigger_onset(
        characteristic_function,
        TRIGGER_ON,
        TRIGGER_OFF
    )

    # Handle an event that previously exceeded maximum duration
    if current_data["ignore_until_below_off"]:

        if current_ratio < TRIGGER_OFF:

            current_data["ignore_until_below_off"] = False

        else:

            current_data["detection_status"] = "Normal"

            return

    # Process only intervals that overlap newly received samples
    for start_index, end_index in trigger_intervals:

        start_index = int(start_index)

        end_index = int(end_index)

        # Ignore old completed intervals
        if end_index < new_start_index:
            continue

        # Ignore triggers in the initial LTA warm-up period
        if start_index < nlta:
            continue

        trigger_start = float(detector_times[start_index])

        trigger_end = float(detector_times[end_index])

        duration = trigger_end - trigger_start

        maximum_ratio = float(
            np.max(
                characteristic_function[
                    start_index:end_index + 1
                ]
            )
        )

        active_event = current_data["active_event"]

        # Start a new event if there is no active event
        if active_event is None:

            active_event = {
                "station_id": station_id,
                "start_timestamp": trigger_start,
                "end_timestamp": trigger_end,
                "duration": duration,
                "max_ratio": maximum_ratio,
                "status": "POSSIBLE EARTHQUAKE"
            }

            current_data["active_event"] = active_event

        else:

            # Preserve the original start time
            active_event["end_timestamp"] = trigger_end

            active_event["duration"] = (
                trigger_end - active_event["start_timestamp"]
            )

            active_event["max_ratio"] = max(
                active_event["max_ratio"],
                maximum_ratio
            )

        # Check whether the trigger has ended
        if end_index < len(characteristic_function) - 1:

            completed_event = current_data["active_event"]

            completed_duration = completed_event["duration"]

            if (
                MIN_EVENT_DURATION <= completed_duration
                <= MAX_EVENT_DURATION
            ):

                record = {
                    "station_id": station_id,
                    "start_timestamp": completed_event["start_timestamp"],
                    "end_timestamp": completed_event["end_timestamp"],
                    "duration": completed_duration,
                    "max_sta_lta": completed_event["max_ratio"],
                    "status": "POSSIBLE EARTHQUAKE"
                }

                detection_history.append(record)

                # Keep only the latest 100 records
                if len(detection_history) > MAX_DETECTION_RECORDS:

                    detection_history.pop(0)

                current_data["last_detection"] = record.copy()

                print(
                    f"POSSIBLE EARTHQUAKE: {station_id} | "
                    f"Duration: {completed_duration:.2f}s | "
                    f"Maximum STA/LTA: "
                    f"{completed_event['max_ratio']:.2f}"
                )

            current_data["active_event"] = None

        else:

            # Trigger is still active
            active_duration = (
                detector_times[-1]
                - current_data["active_event"]["start_timestamp"]
            )

            current_data["active_event"]["duration"] = active_duration

            current_data["active_event"]["end_timestamp"] = (
                float(detector_times[-1])
            )

            # Stop tracking triggers that exceed maximum duration
            if active_duration > MAX_EVENT_DURATION:

                current_data["active_event"] = None

                current_data["ignore_until_below_off"] = True

    # Update station detection status
    if current_data["active_event"] is not None:

        current_data["detection_status"] = "Possible Earthquake"

    else:

        current_data["detection_status"] = "Normal"



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

        # Update raw buffer
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

        # Keep five minutes of raw data
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

            # Step 1: Preprocess incoming data
            preprocess_data(
                station_id,
                waveform,
                timestamps,
                sampling_rate
            )

            # Step 2: Find newly available processed samples
            processed_times = current_data["processed_timestamps"]

            processed_values = current_data["processed_waveform"]

            last_timestamp = (
                current_data["last_detection_input_timestamp"]
            )

            if last_timestamp is None:

                # On startup, only analyze the current incoming chunk
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

                # Step 3: Run STA/LTA
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


# --------------------------------------------------
# SeedLink background thread
# --------------------------------------------------

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


# --------------------------------------------------
# Spectrogram
# --------------------------------------------------

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

    time_step = max(
        1,
        len(absolute_times) // 150
    )

    frequency_step = max(
        1,
        len(frequencies) // 100
    )

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

        processed_sampling_rate = (
            current_data["processed_sampling_rate"]
        )

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

        preprocessing_error = current_data["preprocessing_error"]

    # Reduce waveform points
    raw_step = max(
        1,
        len(raw_waveform) // MAX_WAVEFORM_POINTS
    )

    processed_step = max(
        1,
        len(processed_waveform) // MAX_WAVEFORM_POINTS
    )

    raw_output = {
        "times": raw_timestamps[::raw_step].tolist(),
        "values": raw_waveform[::raw_step].tolist()
    }

    processed_output = {
        "times": processed_timestamps[::processed_step].tolist(),
        "values": processed_waveform[::processed_step].tolist()
    }

    # Raw spectrogram
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

        "name": current_data["name"],

        "network": current_data["network"],

        "channel": current_data["channel"],

        "raw_sampling_rate": raw_sampling_rate,

        "processed_sampling_rate": processed_sampling_rate,

        "raw_waveform": raw_output,

        "processed_waveform": processed_output,

        "spectrogram": spectrogram_output,

        "latest_timestamp": latest_timestamp,

        "status": "receiving" if latest_timestamp else "waiting",

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

        .detection-panel {
            margin-top: 15px;
            padding: 15px;
            background: #f8fafc;
            border-radius: 8px;
            display: grid;
            grid-template-columns: repeat(2, 1fr);
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

        <label for="waveform-mode">
            Waveform display:
        </label>

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

                <span
                    class="station-status"
                    id="status-{{ loop.index0 }}">
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


            <div class="plot-title">
                STA/LTA Detection
            </div>


            <div class="detection-panel" id="detection-panel-{{ loop.index0 }}">

                <div>
                    <div class="metric-label">Current STA/LTA ratio</div>
                    <div class="metric-value" id="ratio-{{ loop.index0 }}">
                        0.00
                    </div>
                </div>

                <div>
                    <div class="metric-label">Detection status</div>
                    <div class="metric-value" id="detection-status-{{ loop.index0 }}">
                        Normal
                    </div>
                </div>

                <div>
                    <div class="metric-label">Trigger start</div>
                    <div class="metric-value" id="trigger-start-{{ loop.index0 }}">
                        --
                    </div>
                </div>

                <div>
                    <div class="metric-label">Trigger end</div>
                    <div class="metric-value" id="trigger-end-{{ loop.index0 }}">
                        --
                    </div>
                </div>

                <div>
                    <div class="metric-label">Event duration</div>
                    <div class="metric-value" id="duration-{{ loop.index0 }}">
                        --
                    </div>
                </div>

                <div>
                    <div class="metric-label">Trigger thresholds</div>
                    <div class="metric-value" id="thresholds-{{ loop.index0 }}">
                        4.0 / 1.5
                    </div>
                </div>

            </div>


            <div class="plot-title" id="waveform-title-{{ loop.index0 }}">
                Raw Waveform
            </div>

            <div
                class="plot"
                id="waveform-{{ loop.index0 }}">
            </div>


            <div class="plot-title">
                Raw Data Spectrogram
            </div>

            <div
                class="spectrogram"
                id="spectrogram-{{ loop.index0 }}">
            </div>


            <div
                class="station-info"
                id="timestamp-{{ loop.index0 }}">
                Latest data: Waiting...
            </div>

        </div>

        {% endfor %}

    </div>


    <div class="history-section">

        <h2>Recent Detection History</h2>

        <p>Completed possible earthquake detections</p>

        <div style="overflow-x:auto;">

            <table>

                <thead>

                    <tr>
                        <th>Station</th>
                        <th>Start Time</th>
                        <th>End Time</th>
                        <th>Duration (s)</th>
                        <th>Maximum STA/LTA</th>
                        <th>Status</th>
                    </tr>

                </thead>

                <tbody id="detection-history">

                    <tr>
                        <td colspan="6">No detections recorded</td>
                    </tr>

                </tbody>

            </table>

        </div>

    </div>


    <script>

        const stationIds = {{ stations | list | tojson }};

        const waveformMode = document.getElementById(
            "waveform-mode"
        );


        function formatTimestamp(timestamp) {

            if (timestamp === null || timestamp === undefined) {
                return "--";
            }

            return new Date(
                timestamp * 1000
            ).toLocaleString();

        }


        // Initialize waveform and spectrogram plots
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


        // Update one station's detection information
        function updateDetection(station, index) {

            const detection = station.detection;

            const activeEvent = detection.active_event;

            const detected = detection.status === "Possible Earthquake";

            const card = document.getElementById(
                `card-${index}`
            );

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


            document.getElementById(
                `ratio-${index}`
            ).textContent = detection.ratio.toFixed(3);


            document.getElementById(
                `thresholds-${index}`
            ).textContent =
                `${detection.trigger_on} / ${detection.trigger_off}`;


            if (activeEvent) {

                document.getElementById(
                    `trigger-start-${index}`
                ).textContent = formatTimestamp(
                    activeEvent.start_timestamp
                );

                document.getElementById(
                    `trigger-end-${index}`
                ).textContent = formatTimestamp(
                    activeEvent.end_timestamp
                );

                document.getElementById(
                    `duration-${index}`
                ).textContent =
                    `${activeEvent.duration.toFixed(2)} s`;

            } else {

                const lastDetection = detection.last_detection;

                if (lastDetection) {

                    document.getElementById(
                        `trigger-start-${index}`
                    ).textContent = formatTimestamp(
                        lastDetection.start_timestamp
                    );

                    document.getElementById(
                        `trigger-end-${index}`
                    ).textContent = formatTimestamp(
                        lastDetection.end_timestamp
                    );

                    document.getElementById(
                        `duration-${index}`
                    ).textContent =
                        `${lastDetection.duration.toFixed(2)} s`;

                } else {

                    document.getElementById(
                        `trigger-start-${index}`
                    ).textContent = "--";

                    document.getElementById(
                        `trigger-end-${index}`
                    ).textContent = "--";

                    document.getElementById(
                        `duration-${index}`
                    ).textContent = "--";

                }

            }

        }


        // Update recent detection table
        function updateDetectionHistory(history) {

            const tableBody = document.getElementById(
                "detection-history"
            );

            if (history.length === 0) {

                tableBody.innerHTML = `
                    <tr>
                        <td colspan="6">No detections recorded</td>
                    </tr>
                `;

                return;
            }

            tableBody.innerHTML = "";

            // Display newest detections first
            [...history].reverse().forEach(event => {

                const row = document.createElement("tr");

                const values = [
                    event.station_id,
                    formatTimestamp(event.start_timestamp),
                    formatTimestamp(event.end_timestamp),
                    event.duration.toFixed(2),
                    event.max_sta_lta.toFixed(2),
                    event.status
                ];

                values.forEach((value, index) => {

                    const cell = document.createElement("td");

                    cell.textContent = value;

                    if (index === 5) {
                        cell.className = "possible";
                    }

                    row.appendChild(cell);

                });

                tableBody.appendChild(row);

            });

        }


        // Fetch and update dashboard
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


                    // Station information
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


                    // Filter configuration
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


                    // Latest received data timestamp
                    document.getElementById(
                        `timestamp-${index}`
                    ).textContent = station.latest_timestamp
                        ? `Latest data: ${formatTimestamp(
                            station.latest_timestamp
                          )}`
                        : "Latest data: Waiting...";


                    // Update STA/LTA detection information
                    updateDetection(station, index);


                    // Select waveform
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


                    // Update waveform
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


                    // Update spectrogram
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


                // Update detection history
                updateDetectionHistory(data.detection_history);

            } catch (error) {

                console.error(
                    "Dashboard update error:",
                    error
                );

            }

        }


        waveformMode.addEventListener(
            "change",
            updateDashboard
        );


        initializePlots();

        updateDashboard();

        setInterval(
            updateDashboard,
            {{ update_interval }}
        );

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
