
SEEDLINK_SERVER = "ring.wscada.net:18000"
FDSN_URL = "https://seiscomp.alertnepal.online"
FDSN_TIMEOUT = 60

WEB_HOST = "0.0.0.0"
WEB_PORT = 8001

STATIONS = [
    {"net": "NP", "sta": "EQM10", "cha": "EH*", "loc": "*", "lat": 28.299517, "lon": 83.960148,},
    {"net": "NP", "sta": "EQM26", "cha": "EH*", "loc": "*", "lat": 27.3936969, "lon": 86.6791885,},
    {"net": "NP", "sta": "EQM07", "cha": "EH*", "loc": "*", "lat": 27.814941, "lon": 86.713401,},
    {"net": "NP", "sta": "EQM08", "cha": "EH*", "loc": "*", "lat": 27.831000, "lon": 86.650000,},
    {"net": "NP", "sta": "EQM11", "cha": "EH*", "loc": "*", "lat": 29.2859556, "lon": 81.2741972,},
    {"net": "NP", "sta": "EQM22", "cha": "EH*", "loc": "*", "lat": 28.895805, "lon": 80.227341,},
    {"net": "AM", "sta": "R0F23", "cha": "EH*", "loc": "*", "lat": 28.045045, "lon": 84.072371,},
]

DETECTION_SETTINGS = {
    "fs": 100.0,
    "fmin": 1.0,
    "fmax": 8.0,
    "max_gap_sec": 0.5,

    # P-wave detection
    "sta_p": 1.5,
    "lta_p": 40.0,
    "thr_on": 3.5,
    "thr_off": 1.5,

    # S-wave detection
    "sta_s": 1.0,
    "lta_s": 5.0,
    "thr_on_s": 2.5,
    "thr_off_s": 1.0,

    # Phase picking
    "picker_p_sta": 1.0,
    "picker_p_lta": 10.0,
    "picker_p_thr_on": 3.5,
    "picker_p_thr_off": 1.5,
    "picker_p_aic_before": 2.0,
    "picker_p_aic_after": 1.0,
    "picker_p_aic_max_shift": 1.0,

    "picker_s_sta": 1.0,
    "picker_s_lta": 5.0,
    "picker_s_thr_on": 2.5,
    "picker_s_thr_off": 1.0,
    "picker_s_aic_before": 1.5,
    "picker_s_aic_after": 1.0,
    "picker_s_aic_max_shift": 1.5,

    # Pick quality (SNR + AIC confidence)
    "snr_threshold": 2.0,
    "snr_signal_sec": 1.0,
    "snr_search_sec": 2.0,
    "snr_noise_sec": 1.0,
    "snr_gap_sec": 0.2,
    "aic_min_contrast": 50.0,

    "min_sp_sec": 2.0,
    # Upper bound for S-P pairing. 60 s is too small once a station is
    # ~600 km from the event (S-P ~ 75 s). association.auto_max_sp_sec()
    # derives this from the network aperture; this value is the fallback
    # for callers that read the setting directly.
    "max_sp_sec": 90.0,

    "vp": 6.0,
    "vs": 3.5,

    "moveout_vmin": 6.0,
    "moveout_tolerance": 2.0,

    "min_stations": 3,
    "dead_time_sec": 15.0,
    "min_duration_sec": 0.5,
    "min_amplitude_ratio": 0.5,
}

REALTIME_SETTINGS = {
    "buffer": 300,
    "pre": 30,
    "wait_sec": 10,
    "detect_every": 2,
    "detect_window": 90,
    "dedup_sec": 5,
    "reconnect": 10,
    "onset_max_age": 60,
    "active": 60,
    "waveform": 60,
    "plt_refresh": 5,
    "counts_per_m": 1.0,
    "max_rms_km": 10.0,
    "max_rms_rel": 0.15,
    "max_log": 100,
    "max_events": 50,
}

HISTORICAL_SETTINGS = {
    "pre": 15.0,
    "post": 10.0,
    "chunk_sec": 1800,
    "detect_pad": 20.0,
    "max_range_hours": 24,
    "max_events": 20,
    "max_triggers": 2000,
    "max_rms": 1.5,
    "travel_depths": (10.0, 20.0, 30.0),
    "travel_max_km": 1400.0,
    "travel_step_km": 10.0,
}
