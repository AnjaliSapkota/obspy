import numpy as np
from obspy import UTCDateTime

from config import DETECTION_SETTINGS
from models import Pick


def get_picker_settings(phase):

    phase = phase.upper()

    if phase == "P":
        return {
            "aic_before": DETECTION_SETTINGS["picker_p_aic_before"],
            "aic_after": DETECTION_SETTINGS["picker_p_aic_after"],
            "max_shift": DETECTION_SETTINGS["picker_p_aic_max_shift"],
        }

    if phase == "S":
        return {
            "aic_before": DETECTION_SETTINGS["picker_s_aic_before"],
            "aic_after": DETECTION_SETTINGS["picker_s_aic_after"],
            "max_shift": DETECTION_SETTINGS["picker_s_aic_max_shift"],
        }

    raise ValueError("Phase must be P or S")


def calculate_aic(data):

    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 1:
        raise ValueError("Waveform data must be one-dimensional")

    if len(data) < 3:
        raise ValueError("Not enough samples to calculate AIC")

    if not np.all(np.isfinite(data)):
        raise ValueError("Waveform data contains NaN or infinite values")

    aic = np.full(len(data), np.inf, dtype=np.float64)

    for k in range(1, len(data) - 1):

        left = data[:k]
        right = data[k:]

        var_left = np.var(left)
        var_right = np.var(right)

        if var_left <= 0 or var_right <= 0:
            continue

        aic[k] = (
            k * np.log(var_left)
            + (len(data) - k - 1) * np.log(var_right)
        )

    return aic


def pick_arrival(trace, trigger, phase):

    phase = phase.upper()

    settings = get_picker_settings(phase)

    data = np.asarray(trace.data, dtype=np.float64)

    if data.ndim != 1 or len(data) < 3:
        raise ValueError("Invalid waveform data")

    if not np.all(np.isfinite(data)):
        raise ValueError("Waveform contains NaN or infinite values")

    fs = float(trace.stats.sampling_rate)

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Invalid sampling rate")

    trigger_time = UTCDateTime(trigger.detection_time)
    trace_start = trace.stats.starttime

    # Step 1: Calculate the trigger position in the trace.
    trigger_idx = int(round((trigger_time - trace_start) * fs))

    # Step 2: Define the AIC window around the trigger.
    before_samples = int(settings["aic_before"] * fs)
    after_samples = int(settings["aic_after"] * fs)

    window_start = trigger_idx - before_samples
    window_end = trigger_idx + after_samples

    if window_start < 0 or window_end >= len(data):
        raise ValueError("Not enough waveform data around trigger")

    segment = data[window_start:window_end + 1]

    # Step 3: Calculate AIC values.
    aic = calculate_aic(segment)

    # Step 4: Restrict the search around the original trigger.
    max_shift_samples = int(settings["max_shift"] * fs)

    search_start = max(1, before_samples - max_shift_samples)
    search_end = min(
        len(segment) - 2,
        before_samples + max_shift_samples
    )

    candidate_aic = aic[search_start:search_end + 1]

    if not np.any(np.isfinite(candidate_aic)):
        raise ValueError("Could not find a valid AIC minimum")

    # Step 5: Find the minimum AIC position.
    relative_idx = int(np.nanargmin(candidate_aic))
    arrival_idx = search_start + relative_idx

    # Step 6: Convert the index into an arrival timestamp.
    arrival_time = trace_start + (window_start + arrival_idx) / fs

    aic_shift = arrival_time - trigger_time

    # Step 7: Measure how trustworthy the AIC minimum is.
    finite_aic = candidate_aic[np.isfinite(candidate_aic)]

    aic_contrast = (
        float(np.max(finite_aic) - np.min(finite_aic))
        if len(finite_aic) > 0
        else 0.0
    )

    # A minimum sitting on the search limit means the true onset is likely
    # outside the search range, so the pick is unreliable.
    aic_boundary = arrival_idx in (search_start, search_end)

    # Step 8: Create the Pick object.
    return Pick(
        station_id=trigger.station_id,
        phase=phase,
        arrival_time=arrival_time,
        trigger_time=trigger_time,
        aic_shift_sec=float(aic_shift),
        aic_contrast=aic_contrast,
        aic_boundary=aic_boundary,
    )