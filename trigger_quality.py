import numpy as np


def calculate_snr(data,pick_idx,fs,signal_window_sec=1.0,search_window_sec=2.0,noise_window_sec=1.0,noise_gap_sec=0.2,):

    # Signal-to-noise ratio around an arrival pick.

    # The signal level is the maximum RMS over every window of
    # signal_window_sec inside [pick_idx, pick_idx + search_window_sec].
    # Taking the strongest window keeps the measurement stable for emergent or
    # late-peaking arrivals, where the largest amplitude may arrive well after
    # the pick.

    # The noise level is the RMS of noise_window_secending
    # noise_gap_sec before the pick, so the noise window never overlaps the
    # onset itself.

    data = np.asarray(data, dtype=np.float64)

    fs = float(fs)

    n_signal = int(signal_window_sec * fs)
    n_search = int(search_window_sec * fs)
    n_noise = int(noise_window_sec * fs)
    n_gap = int(noise_gap_sec * fs)

    if n_signal < 2 or n_noise < 2:
        return None

    pick_idx = int(pick_idx)

    noise_end = pick_idx - n_gap
    noise_start = noise_end - n_noise
    search_end = pick_idx + n_search

    # Both windows must fit completely inside the waveform.
    if noise_start < 0 or pick_idx < 0 or search_end > len(data):
        return None

    noise = data[noise_start:noise_end]
    noise = noise - np.mean(noise)

    noise_rms = float(np.sqrt(np.mean(noise ** 2)))

    if noise_rms <= 0:
        return None

    signal = data[pick_idx:search_end]

    if len(signal) < n_signal:
        return None

    signal = signal - np.mean(signal)

    # Maximum RMS over all full-length windows in the search range.
    cumulative = np.concatenate(([0.0], np.cumsum(signal ** 2)))

    starts = np.arange(0, len(signal) - n_signal + 1)

    window_power = cumulative[starts + n_signal] - cumulative[starts]

    signal_rms = float(np.sqrt(np.max(window_power) / n_signal))

    return signal_rms / noise_rms


def evaluate_pick_quality(data, fs, pick_idx, snr_threshold=2.0, signal_window_sec=1.0, search_window_sec=2.0, noise_window_sec=1.0, noise_gap_sec=0.2, aic_boundary=False, aic_contrast=None, aic_min_contrast=50.0):
    # Evaluate an AIC pick using SNR and AIC confidence checks.

    # Parameters
    # data, fs : waveform and sampling rate.
    # pick_idx : refined arrival sample from the AIC picker (not the
    #     STA/LTA trigger index).
    # aic_boundary : True when the AIC minimum landed on the search limit.
    # aic_contrast : separation between the highest and lowest AIC values in
    #     the search range. Low values mean the AIC curve is flat and the
    #     minimum is not meaningful.

    data = np.asarray(data, dtype=np.float64)

    # Validate waveform.
    if data.ndim != 1:
        raise ValueError("Waveform data must be one-dimensional")

    if not np.all(np.isfinite(data)):
        raise ValueError("Waveform contains invalid values")

    # Validate sampling frequency.
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling frequency must be positive")

    # Validate pick index.
    if pick_idx < 0 or pick_idx >= len(data):
        raise ValueError("Pick index is outside waveform")

    # Validate window settings.
    if signal_window_sec <= 0:
        raise ValueError("Signal window must be positive")

    if search_window_sec < signal_window_sec:
        raise ValueError(
            "Search window must be at least as long as the signal window"
        )

    if noise_window_sec <= 0:
        raise ValueError("Noise window must be positive")

    if noise_gap_sec < 0:
        raise ValueError("Noise gap cannot be negative")

    if snr_threshold < 0:
        raise ValueError("SNR threshold cannot be negative")

    if aic_min_contrast < 0:
        raise ValueError("AIC contrast threshold cannot be negative")

    checks = []

    # Step 1: Calculate SNR around the refined pick.
    snr = calculate_snr(data=data, pick_idx=pick_idx, fs=fs, signal_window_sec=signal_window_sec, search_window_sec=search_window_sec, noise_window_sec=noise_window_sec, noise_gap_sec=noise_gap_sec)

    if snr is None:
        checks.append((False, "SNR could not be calculated"))
    else:
        checks.append((
            snr >= snr_threshold,
            f"SNR {snr:.2f} (threshold {snr_threshold})",
        ))

    # Step 2: Reject picks whose AIC minimum sits on the search limit.
    checks.append((
        not aic_boundary,
        "AIC minimum is not at search boundary",
    ))

    # Step 3: Reject flat AIC curves where the minimum is not meaningful.
    if aic_contrast is not None:
        checks.append((
            aic_contrast >= aic_min_contrast,
            f"AIC contrast {aic_contrast:.1f} "
            f"(minimum {aic_min_contrast})",
        ))

    # Step 4: Accept only when every check passes.
    accepted = all(passed for passed, description in checks)

    notes = [
        description
        for passed, description in checks
        if not passed
    ]

    return {
        "accepted": accepted,
        "snr": snr,
        "checks": checks,
        "notes": notes,
        "aic_boundary": bool(aic_boundary),
        "aic_contrast": aic_contrast,
    }
