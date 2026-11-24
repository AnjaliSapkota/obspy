
import numpy as np
from obspy import UTCDateTime
from obspy.signal.trigger import recursive_sta_lta, trigger_onset

from config import DETECTION_SETTINGS
from models import Trigger


def get_detection_settings(component):

    component = component.upper()

    if component == "P":
        return {
            "sta": DETECTION_SETTINGS["sta_p"],
            "lta": DETECTION_SETTINGS["lta_p"],
            "thr_on": DETECTION_SETTINGS["thr_on"],
            "thr_off": DETECTION_SETTINGS["thr_off"],
        }

    if component == "S":
        return {
            "sta": DETECTION_SETTINGS["sta_s"],
            "lta": DETECTION_SETTINGS["lta_s"],
            "thr_on": DETECTION_SETTINGS["thr_on_s"],
            "thr_off": DETECTION_SETTINGS["thr_off_s"],
        }

    raise ValueError("Component must be P or S")


def calculate_cft(data, component="P"):

    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 1:
        raise ValueError("Waveform data must be one-dimensional")

    if len(data) == 0:
        raise ValueError("Waveform data is empty")

    if not np.all(np.isfinite(data)):
        raise ValueError("Waveform data contains NaN or infinite values")

    settings = get_detection_settings(component)

    fs = float(DETECTION_SETTINGS["fs"])

    nsta = int(settings["sta"] * fs)
    nlta = int(settings["lta"] * fs)

    if nsta <= 0 or nlta <= 0:
        raise ValueError("STA and LTA sample counts must be positive")

    if nsta >= nlta:
        raise ValueError("STA must be shorter than LTA")

    if len(data) <= nlta:
        raise ValueError(
            f"Not enough waveform samples: "
            f"need more than {nlta}, received {len(data)}"
        )

    cft = recursive_sta_lta(data, nsta, nlta)

    warmup_samples = nlta + nsta

    cft[:warmup_samples] = 0.0

    return np.asarray(cft, dtype=np.float64)


def detect_triggers(cft, component="P"):

    cft = np.asarray(cft, dtype=np.float64)

    if cft.ndim != 1:
        raise ValueError("CFT must be one-dimensional")

    if len(cft) == 0:
        return []

    if not np.all(np.isfinite(cft)):
        raise ValueError("CFT contains NaN or infinite values")

    settings = get_detection_settings(component)

    triggers = trigger_onset(
        cft,
        settings["thr_on"],
        settings["thr_off"],
    )

    return [
        (int(start), int(end))
        for start, end in triggers
    ]


def process_detection(data, start_time, station_id, component):
    """
    Detect raw STA/LTA triggers for one waveform.

    Processing:
    1. Calculate CFT.
    2. Find raw trigger intervals.
    3. Calculate trigger time, duration and amplitude.
    4. Return all Trigger objects.

    No TriggerGate filtering is applied.
    """

    data = np.asarray(data, dtype=np.float64) # convert waveform data into numpy array of type float64

    if data.ndim != 1:
        raise ValueError("Waveform data must be one-dimensional")

    if len(data) == 0:
        return []

    fs = float(DETECTION_SETTINGS["fs"])

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Invalid sampling frequency")

    if not np.all(np.isfinite(data)):
        raise ValueError("Waveform data contains NaN or infinite values")

    start_time = UTCDateTime(start_time)

    # Step 1: Calculate characteristic function.
    cft = calculate_cft(data, component)

    # Step 2: Find raw trigger intervals.
    raw_triggers = detect_triggers(cft, component)

    triggers = []

    # Step 3: Convert raw intervals into Trigger objects.
    for start_idx, end_idx in raw_triggers:

        if end_idx <= start_idx:
            continue

        trigger_time = start_time + start_idx / fs

        duration = (end_idx - start_idx) / fs

        segment = data[start_idx:end_idx + 1]

        amplitude = float(np.max(np.abs(segment)))

        cft_value = float(np.max(cft[start_idx:end_idx + 1]))

        trigger = Trigger(
            station_id=station_id,
            detection_time=trigger_time,
            cft_value=cft_value,
            duration_sec=duration,
            amplitude=amplitude,
        )

        triggers.append(trigger)

    return triggers
