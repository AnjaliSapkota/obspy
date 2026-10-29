from __future__ import annotations

import argparse
import copy
import math
import threading
import time
import traceback
from fractions import Fraction
from itertools import combinations

import matplotlib

matplotlib.use("Agg")

import numpy as np
from flask import Flask, jsonify, render_template_string, request
from obspy import Stream, Trace, UTCDateTime
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from scipy.signal import find_peaks, resample_poly, spectrogram, welch

import triangulate as core


SAFETY_NOTE = (
    "Detected seismic activity is not a confirmed flood, GLOF, landslide or "
    "earthquake. Duration and waveform shape are only provisional evidence; "
    "source attribution requires historical calibration and independent "
    "information such as field reports, hydrology or imagery."
)

DEFAULT_CONFIG = {
    "server": "ring.wscada.net:18000",
    "fdsn_server": "https://seiscomp.alertnepal.online",
    "host": "0.0.0.0",
    "port": 5055,
    "fs": 20.0,
    "fmin": 1.0,
    "fmax": 8.0,
    "smooth_cutoff": 0.5,
    "keep_seconds": 900.0,
    "analysis_window": 180.0,
    "display_window": 120.0,
    "min_detection_window": 20.0,
    "cycle_seconds": 2.0,
    "gap_threshold": 2.0,
    "stale_seconds": 20.0,
    "seedlink_timeout": 15.0,
    "detection": {
        "threshold_factor": 4.0,
        "mad_factor": 6.0,
        "min_duration": 0.30,
        "merge_gap": 1.50,
        "dedupe_seconds": 2.50,
        "stale_active_seconds": 6.0,
        "max_events_per_cycle": 16,
        "short_duration": 3.0,
        "sustained_duration": 12.0,
    },
    "association": {
        "grace_seconds": 5.0,
        "deadline_seconds": 180.0,
        "max_window": 150.0,
        "pre_margin": 4.0,
        "post_margin": 6.0,
        "time_tolerance": 2.0,
        "min_cc": 0.25,
        "min_coherence": 0.30,
        "min_pair_score": 0.36,
        "min_duration_ratio": 0.20,
        "closure_tolerance": 3.5,
        "localization_refresh": 30.0,
        "max_evaluations_per_cycle": 2,
        "max_localizations_per_cycle": 1,
    },
    "localization": {
        "v_min": 1.5,
        "v_max": 3.5,
        "v_step": 0.10,
        "v_prior": 2.5,
        "grid_step_deg": 0.03,
        "margin_deg": 1.50,
        "max_pair_rmse": 1.50,
        "min_spread_km": 5.0,
        "min_geometry_score": 0.30,
    },
    "retention": {
        "detections": 500,
        "candidates": 100,
        "log": 300,
        "results_seconds": 3600.0,
    },
}

DEFAULT_STATIONS = [
    {"net": "NP", "sta": "EQM10", "cha": "EHZ", "loc": "*", "lat": 28.299517, "lon": 83.960148},
    {"net": "NP", "sta": "EQM13", "cha": "EHZ", "loc": "*", "lat": 28.256323, "lon": 85.367569},
    {"net": "NK", "sta": "KKN", "cha": "BHZ", "loc": "*", "lat": 27.800000, "lon": 85.279000},
    {"net": "NP", "sta": "EQM07", "cha": "EHZ", "loc": "*", "lat": 27.814941, "lon": 86.713401},
    {"net": "IO", "sta": "EVN", "cha": "BHZ", "loc": "*", "lat": 27.958650, "lon": 86.811653},
]

_COORDINATE_CACHE = {}

APP = Flask(__name__)
STATE = None



def utc_text(value):
    """Format a UTCDateTime as an ISO-8601 UTC string with milliseconds."""
    if value is None:
        return None
    return UTCDateTime(value).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def epoch_ms(value):
    """Return Unix epoch milliseconds for a UTCDateTime-like value."""
    return int(round(float(UTCDateTime(value).timestamp) * 1000.0))


def utc_seconds(start, end):
    """Difference between two UTC values in seconds."""
    return float(UTCDateTime(end) - UTCDateTime(start))


def jsonable(value):
    """Recursively convert numpy/ObsPy values into JSON-safe objects."""
    if isinstance(value, UTCDateTime):
        return utc_text(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def safe_float(value, default=0.0):
    """Return a finite float, otherwise ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


# ---------------------------------------------------------------------------
# State, stations and rolling buffers
# ---------------------------------------------------------------------------


def station_key(station):
    """Stable key for a station configuration."""
    return f"{station['net']}.{station['sta']}.{station['cha']}"


def station_label(station):
    """Human-readable station label."""
    return f"{station['net']}.{station['sta']}.{station['cha']}"


def make_config(overrides=None):
    """Return a deep copy of the default configuration, with overrides."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def create_state(config=None, stations=None, no_seedlink=False):
    """Create the shared application state (a plain dictionary)."""
    config = config or make_config()
    state = {
        "config": config,
        "lock": threading.RLock(),
        "stations": {},
        "buffers": {},
        "plots": {},
        "window": None,
        "detections": [],
        "candidates": [],
        "event_log": [],
        "counters": {"detection": 0, "candidate": 0, "log": 0},
        "seedlink": {"threads": {}, "no_seedlink": bool(no_seedlink)},
        "stop": threading.Event(),
        "started_utc": UTCDateTime(),
        "health": {
            "cycle_ms": None,
            "last_cycle_utc": None,
            "latest_data_lag_s": None,
            "association_ms": None,
            "localization_ms": None,
            "warnings": [],
        },
    }
    for station in stations if stations is not None else DEFAULT_STATIONS:
        add_station_to_state(state, station, start_seedlink=False)
    return state


def next_id(state, kind):
    """Return the next sequential identifier for detections/candidates/logs."""
    state["counters"][kind] += 1
    prefix = {"detection": "D", "candidate": "C", "log": "L"}[kind]
    return f"{prefix}{state['counters'][kind]:04d}"


def normalize_station(station):
    """Validate and normalize a station configuration."""
    required = ("net", "sta", "cha")
    for field in required:
        if not str(station.get(field, "")).strip():
            raise ValueError(f"Station is missing '{field}'.")

    normalized = {
        "net": str(station["net"]).strip(),
        "sta": str(station["sta"]).strip(),
        "cha": str(station["cha"]).strip(),
        "loc": str(station.get("loc") or "*").strip(),
        "lat": station.get("lat"),
        "lon": station.get("lon"),
    }
    if normalized["lat"] is not None:
        normalized["lat"] = float(normalized["lat"])
    if normalized["lon"] is not None:
        normalized["lon"] = float(normalized["lon"])
    if (normalized["lat"] is None) != (normalized["lon"] is None):
        raise ValueError("Provide both latitude and longitude, or neither.")
    return normalized


def resolve_station_coordinates(net, sta, fdsn_server):
    """Look up station coordinates from the configured FDSN service."""
    cache_key = (fdsn_server, net, sta)
    if cache_key in _COORDINATE_CACHE:
        return _COORDINATE_CACHE[cache_key]

    client = FDSNClient(fdsn_server)
    inventory = client.get_stations(network=net, station=sta, level="station")
    for network in inventory:
        for station in network:
            result = (float(station.latitude), float(station.longitude), float(station.elevation or 0.0))
            _COORDINATE_CACHE[cache_key] = result
            return result
    raise ValueError(f"No coordinates found for {net}.{sta} on {fdsn_server}")


def add_station_to_state(state, station, start_seedlink=False):
    """Add a station and initialize its empty ring buffer."""
    station = normalize_station(station)
    if station["lat"] is None:
        station["lat"], station["lon"], _ = resolve_station_coordinates(
            station["net"], station["sta"], state["config"]["fdsn_server"]
        )

    key = station_key(station)
    with state["lock"]:
        if key in state["stations"]:
            raise ValueError(f"Station {key} is already being monitored.")
        entry = dict(station)
        entry.update(
            {
                "key": key,
                "label": station_label(station),
                "added_utc": UTCDateTime(),
                "status": {
                    "connected": False,
                    "packets": 0,
                    "gaps": 0,
                    "last_packet_utc": None,
                    "last_trace_utc": None,
                    "latency_s": None,
                    "last_error": None,
                    "last_gap_utc": None,
                    "last_gap_s": None,
                    "stale": True,
                },
            }
        )
        state["stations"][key] = entry
        state["buffers"][key] = Stream()
        state["plots"][key] = {
            "available": False,
            "reason": "waiting for data",
            "start_ms": None,
            "end_ms": None,
            "t_ms": [],
            "waveform": [],
            "envelope": [],
            "spectrogram": None,
            "triggers": [],
        }

    log_event(
        state,
        "station",
        f"Started monitoring {key}",
        station=key,
        details={"lat": station["lat"], "lon": station["lon"]},
    )
    if start_seedlink:
        start_station_seedlink(state, key)
    return entry


def remove_station_from_state(state, key):
    """Stop a station listener and remove its buffer and plots."""
    with state["lock"]:
        station = state["stations"].get(key)
        if station is None:
            raise KeyError(f"Unknown station {key}")
        listener = state["seedlink"]["threads"].pop(key, None)
        del state["stations"][key]
        state["buffers"].pop(key, None)
        state["plots"].pop(key, None)

    if listener is not None:
        listener["stop"].set()
        client = listener.get("client")
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    log_event(state, "station", f"Stopped monitoring {key}", station=key)
    return station


def append_trace_to_buffer(state, key, trace):
    """Append one live packet, removing overlap and tracking gaps/latency."""
    if trace is None or trace.stats.npts == 0:
        return

    config = state["config"]
    now = UTCDateTime()
    trace = trace.copy()
    trace.data = np.asarray(trace.data, dtype=np.float64)
    trace.data = np.nan_to_num(trace.data, nan=0.0, posinf=0.0, neginf=0.0)

    with state["lock"]:
        station = state["stations"].get(key)
        if station is None:
            return
        buffer = state["buffers"].setdefault(key, Stream())
        status = station["status"]

        if len(buffer):
            last_end = max(item.stats.endtime for item in buffer)
            if trace.stats.endtime <= last_end:
                return
            if trace.stats.starttime < last_end:
                trace.trim(starttime=last_end)
            if trace.stats.npts == 0:
                return
            gap = float(trace.stats.starttime - last_end)
            if gap > config["gap_threshold"]:
                status["gaps"] += 1
                status["last_gap_utc"] = utc_text(trace.stats.starttime)
                status["last_gap_s"] = gap

        buffer.append(trace)
        newest = max(item.stats.endtime for item in buffer)
        cutoff = newest - config["keep_seconds"]
        kept = [item for item in buffer if item.stats.endtime >= cutoff]
        state["buffers"][key] = Stream(kept)

        status["packets"] += 1
        status["last_packet_utc"] = utc_text(now)
        status["last_trace_utc"] = utc_text(trace.stats.endtime)
        status["latency_s"] = max(0.0, float(now - trace.stats.endtime))
        status["stale"] = False


def snapshot_buffer(state, key):
    """Return a copy of a station's raw rolling stream."""
    with state["lock"]:
        stream = state["buffers"].get(key)
        return stream.copy() if stream is not None else Stream()


def latest_contiguous_trace(state, key):
    """Return the newest contiguous trace in a station's buffer."""
    stream = snapshot_buffer(state, key)
    if len(stream) == 0:
        return None
    stream.sort()
    try:
        stream.merge(method=1, fill_value=None)
    except Exception:
        pass
    candidates = [trace for trace in stream if trace.stats.npts > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda trace: trace.stats.endtime)


def update_station_connection(state, key, connected, error=None):
    """Update connection state from the SeedLink worker thread."""
    with state["lock"]:
        station = state["stations"].get(key)
        if station is None:
            return
        status = station["status"]
        status["connected"] = bool(connected)
        if error:
            status["last_error"] = str(error)
        if connected:
            status["last_error"] = None


# ---------------------------------------------------------------------------
# SeedLink listeners (one connection per station, so one outage is isolated)
# ---------------------------------------------------------------------------


class LiveSeedLinkClient(EasySeedLinkClient):
    """Thin EasySeedLinkClient callback; all logic lives in functions."""

    def __init__(self, state, key, server):
        super().__init__(server, autoconnect=False)
        self.state = state
        self.key = key

    def on_data(self, trace):
        try:
            append_trace_to_buffer(self.state, self.key, trace)
        except Exception as error:
            update_station_connection(self.state, self.key, False, error)


def seedlink_worker(state, key, stop_event, listener):
    """Keep one station connected, reconnecting with bounded backoff."""
    backoff = 2.0
    while not stop_event.is_set():
        station = state["stations"].get(key)
        if station is None:
            break
        client = LiveSeedLinkClient(state, key, state["config"]["server"])
        listener["client"] = client
        try:
            client.conn.timeout = float(state["config"]["seedlink_timeout"])
            client.connect()
            client.select_stream(station["net"], station["sta"], station["cha"])
            update_station_connection(state, key, True)
            client.run()
            update_station_connection(state, key, False)
        except Exception as error:
            update_station_connection(state, key, False, error)
        finally:
            try:
                client.close()
            except Exception:
                pass
            listener["client"] = None

        if stop_event.is_set():
            break
        stop_event.wait(backoff)
        backoff = min(30.0, backoff * 2.0)


def start_station_seedlink(state, key):
    """Start a per-station SeedLink worker thread."""
    if state["seedlink"].get("no_seedlink"):
        return None
    with state["lock"]:
        if key in state["seedlink"]["threads"]:
            return state["seedlink"]["threads"][key]
        stop_event = threading.Event()
        listener = {"stop": stop_event, "client": None, "thread": None}
        thread = threading.Thread(
            target=seedlink_worker,
            args=(state, key, stop_event, listener),
            daemon=True,
            name=f"seedlink-{key}",
        )
        listener["thread"] = thread
        state["seedlink"]["threads"][key] = listener
    thread.start()
    return listener


def start_all_seedlinks(state):
    """Start listeners for every configured station."""
    for key in list(state["stations"].keys()):
        start_station_seedlink(state, key)


def stop_all_seedlinks(state):
    """Ask all SeedLink workers to stop and close their sockets."""
    state["stop"].set()
    with state["lock"]:
        listeners = list(state["seedlink"]["threads"].values())
        state["seedlink"]["threads"].clear()
    for listener in listeners:
        listener["stop"].set()
        client = listener.get("client")
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Waveform preprocessing, envelopes and station-level detection
# ---------------------------------------------------------------------------


def resample_trace(trace, target_fs):
    """Anti-aliased rational resampling, preserving the start time."""
    source_fs = float(trace.stats.sampling_rate)
    target_fs = float(target_fs)
    if source_fs <= 0:
        raise ValueError("Trace has an invalid sampling rate.")
    if abs(source_fs - target_fs) < 1e-6:
        return trace.copy()

    ratio = Fraction(target_fs / source_fs).limit_denominator(2000)
    data = resample_poly(
        np.asarray(trace.data, dtype=np.float64),
        ratio.numerator,
        ratio.denominator,
    )
    output = Trace(data=np.asarray(data, dtype=np.float64))
    output.stats = trace.stats.copy()
    output.stats.sampling_rate = source_fs * ratio.numerator / ratio.denominator
    output.stats.delta = 1.0 / output.stats.sampling_rate
    output.stats.npts = len(output.data)
    return output


def prepare_trace(trace, config):
    """Detrend, taper, resample and band-pass one contiguous trace."""
    if trace is None or trace.stats.npts < 8:
        raise ValueError("Trace is too short for preprocessing.")
    work = trace.copy()
    work.data = np.asarray(work.data, dtype=np.float64)
    work.data = np.nan_to_num(work.data, nan=0.0, posinf=0.0, neginf=0.0)
    work.detrend("linear")
    work.detrend("demean")
    work.taper(max_percentage=0.02, type="hann")
    work = resample_trace(work, config["fs"])
    if work.stats.npts < 8:
        raise ValueError("Trace is too short after resampling.")
    work.filter(
        "bandpass",
        freqmin=float(config["fmin"]),
        freqmax=float(config["fmax"]),
        corners=4,
        zerophase=True,
    )
    work.data = np.nan_to_num(work.data, nan=0.0, posinf=0.0, neginf=0.0)
    return work


def envelope_of_trace(trace, config):
    """Smoothed Hilbert envelope using the shared triangulate helper."""
    if trace.stats.npts < 16:
        raise ValueError("Trace is too short for a Hilbert envelope.")
    envelope = core.get_hilbert_envelope(trace, float(config["smooth_cutoff"]))
    envelope = np.asarray(envelope, dtype=np.float64)
    if envelope.size == 0 or not np.all(np.isfinite(envelope)):
        raise ValueError("Envelope contains no finite values.")
    return envelope


def robust_noise_level(envelope):
    """Estimate the quiet baseline and its robust spread."""
    baseline = float(np.percentile(envelope, 20.0))
    median = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - median)))
    return max(baseline, 0.0), max(mad, 0.0)


def mask_to_segments(mask, gap_samples, min_samples):
    """Convert a boolean mask to inclusive sample segments."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    segments = []
    start = previous = int(indices[0])
    for index in indices[1:]:
        index = int(index)
        if index - previous > gap_samples:
            segments.append((start, previous))
            start = index
        previous = index
    segments.append((start, previous))
    return [
        (start, end)
        for start, end in segments
        if (end - start + 1) >= max(1, min_samples)
    ]


def welch_spectrum(data, fs, fmin, fmax, nperseg=None):
    """Return a band-limited Welch PSD and simple summary features."""
    data = np.asarray(data, dtype=np.float64)
    data = data - np.mean(data)
    if data.size < 16:
        return None, None, {"dominant_frequency_hz": None, "spectral_centroid_hz": None, "spectral_bandwidth_hz": None}
    if nperseg is None:
        nperseg = min(256, data.size)
    nperseg = max(16, min(int(nperseg), data.size))
    frequencies, power = welch(data, fs=fs, nperseg=nperseg)

    mask = (frequencies >= fmin) & (frequencies <= fmax)
    frequencies = frequencies[mask]
    power = power[mask]
    total = float(np.sum(power))
    if frequencies.size == 0 or total <= 0:
        return None, None, {"dominant_frequency_hz": None, "spectral_centroid_hz": None, "spectral_bandwidth_hz": None}

    dominant = float(frequencies[int(np.argmax(power))])
    centroid = float(np.sum(frequencies * power) / total)
    bandwidth = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * power) / total))
    features = {
        "dominant_frequency_hz": dominant,
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
    }
    return frequencies, power, features


def event_features(trace, envelope, start_index, end_index, config, baseline, threshold):
    """Measure one detected segment and classify it provisionally."""
    fs = float(trace.stats.sampling_rate)
    start_index = max(0, int(start_index))
    end_index = min(len(trace.data) - 1, int(end_index))
    data = np.asarray(trace.data[start_index : end_index + 1], dtype=np.float64)
    envelope_segment = np.asarray(envelope[start_index : end_index + 1], dtype=np.float64)
    if data.size < 2:
        raise ValueError("Detected segment is too short.")

    duration = data.size / fs
    peak = safe_float(np.max(np.abs(data)))
    rms = safe_float(np.sqrt(np.mean(data ** 2)))
    peak_envelope = safe_float(np.max(envelope_segment))
    baseline_safe = max(safe_float(baseline), 1e-12)
    energy = safe_float(np.sum(envelope_segment ** 2) / fs)
    mean_envelope_power = safe_float(np.mean(envelope_segment ** 2))
    energy_db = 10.0 * math.log10(max(mean_envelope_power, 1e-24) / (baseline_safe ** 2))
    snr = peak_envelope / baseline_safe
    peak_to_rms = peak / rms if rms > 0 else 0.0

    # A low detection threshold deliberately captures the exact tail of the
    # envelope, but band-pass ringing can make a short impulse look long at
    # that level. A peak-relative threshold defines the significant part of
    # the signal, which is what the trigger time and classification use.
    core_level = max(float(threshold), baseline_safe + 0.10 * max(0.0, peak_envelope - baseline_safe))
    above_core = np.flatnonzero(envelope_segment >= core_level)
    if above_core.size:
        core_first = int(above_core[0])
        core_last = int(above_core[-1])
        significant_duration = float((core_last - core_first + 1) / fs)
        core_start = trace.stats.starttime + (start_index + core_first) / fs
        core_end = trace.stats.starttime + (start_index + core_last + 1) / fs
        core_start_index = start_index + core_first
        core_end_index = start_index + core_last
    else:
        significant_duration = float(duration)
        core_start = trace.stats.starttime + start_index / fs
        core_end = trace.stats.starttime + (end_index + 1) / fs
        core_start_index = start_index
        core_end_index = end_index

    threshold_start = trace.stats.starttime + start_index / fs
    threshold_end = trace.stats.starttime + (end_index + 1) / fs

    _, _, spectral = welch_spectrum(
        data,
        fs,
        fmin=float(config["fmin"]),
        fmax=float(config["fmax"]),
    )

    if data.size >= 8:
        pulse_height = max(core_level, baseline_safe + 0.35 * max(0.0, peak_envelope - baseline_safe))
        prominence = max(baseline_safe, 0.25 * max(0.0, peak_envelope - baseline_safe))
        peaks, _ = find_peaks(
            envelope_segment,
            height=pulse_height,
            distance=max(1, int(round(0.8 * fs))),
            prominence=prominence,
        )
        pulse_count = max(1, int(len(peaks))) if peak_envelope > core_level else 0
    else:
        pulse_count = 1

    energy_trend = 0.0
    envelope_cv = 0.0
    if envelope_segment.size >= 8:
        denominator = float(np.mean(envelope_segment))
        if denominator > 0:
            envelope_cv = float(np.std(envelope_segment) / denominator)
        block_count = min(12, envelope_segment.size)
        blocks = np.array_split(envelope_segment, block_count)
        block_energy = np.array(
            [10.0 * math.log10(max(float(np.mean(block ** 2)), 1e-24)) for block in blocks if block.size]
        )
        if block_energy.size >= 2:
            times = np.linspace(0.0, max(duration, 1e-6), block_energy.size)
            energy_trend = float(np.polyfit(times, block_energy, 1)[0])

    rise_fraction = float(np.argmax(envelope_segment) / max(1, envelope_segment.size - 1))

    features = {
        "start": core_start,
        "end": core_end,
        "threshold_start": threshold_start,
        "threshold_end": threshold_end,
        "core_start": core_start,
        "core_end": core_end,
        "duration_s": float(significant_duration),
        "significant_duration_s": float(significant_duration),
        "window_duration_s": float(duration),
        "core_level": float(core_level),
        "energy": energy,
        "energy_db": float(energy_db),
        "peak_amp": float(peak),
        "rms_amp": float(rms),
        "peak_to_rms": float(peak_to_rms),
        "peak_envelope": float(peak_envelope),
        "snr": float(snr),
        "pulse_count": int(pulse_count),
        "energy_trend_db_per_s": float(energy_trend),
        "envelope_cv": float(envelope_cv),
        "rise_fraction": float(rise_fraction),
        "baseline": float(baseline_safe),
        "threshold": float(threshold),
        **spectral,
    }
    features["classification"] = classify_event(features, config)
    return features


def classify_event(features, config):
    """Provisional, explicitly non-diagnostic classification of a segment.

    Duration is only one input. Peak-to-RMS, pulse count, energy trend and
    spectral shape are combined so a short impulse is kept and a long signal is
    not automatically called a flood or landslide.
    """
    detection_config = config["detection"]
    duration = safe_float(features.get("significant_duration_s", features.get("duration_s")))
    peak_to_rms = safe_float(features.get("peak_to_rms"))
    pulse_count = int(features.get("pulse_count") or 0)
    trend = safe_float(features.get("energy_trend_db_per_s"))
    centroid = features.get("spectral_centroid_hz")
    reasons = []

    if duration < float(detection_config["short_duration"]):
        key = "short_impulsive"
        label = "Short impulsive spike"
        hypothesis = (
            "A brief transient is consistent with an impact, local rockfall, "
            "ice fall or cultural noise; it is not diagnostic of a flood or GLOF."
        )
        confidence = 0.55 + min(0.25, max(0.0, peak_to_rms - 1.5) * 0.08)
        reasons.append(f"duration {duration:.2f} s is below {detection_config['short_duration']:.1f} s")
    elif pulse_count >= 3 and duration >= float(detection_config["short_duration"]):
        key = "repeated_pulses"
        label = "Repeated pulses"
        hypothesis = (
            "Several separated pulses are consistent with repeated mass movement, "
            "rockfall sequences or a surging flow; independent evidence is required."
        )
        confidence = 0.55
        reasons.append(f"{pulse_count} envelope pulses were counted")
    elif duration >= float(detection_config["sustained_duration"]) and trend > 1.5:
        key = "emerging_sustained"
        label = "Emerging sustained activity"
        hypothesis = (
            "Sustained and increasing envelope energy may indicate an approaching "
            "or growing source, but could also be atmospheric, hydrological or "
            "instrumental. It is not a confirmed GLOF or landslide."
        )
        confidence = 0.60
        reasons.append(f"energy trend is +{trend:.1f} dB/s over {duration:.1f} s")
    elif duration >= float(detection_config["sustained_duration"]):
        key = "sustained"
        label = "Sustained activity"
        hypothesis = (
            "Prolonged ground motion is compatible with a debris flow, river surge "
            "or other sustained source, but duration alone does not identify it."
        )
        confidence = 0.58
        reasons.append(f"duration {duration:.1f} s exceeds {detection_config['sustained_duration']:.1f} s")
    elif peak_to_rms >= 5.0:
        key = "impulsive_burst"
        label = "Impulsive burst"
        hypothesis = (
            "A high crest factor suggests an impulsive source, such as a local "
            "impact or rockfall; it is not a confirmed hazard event."
        )
        confidence = 0.50
        reasons.append(f"peak-to-RMS ratio is {peak_to_rms:.1f}")
    else:
        key = "transient_activity"
        label = "Transient activity"
        hypothesis = (
            "The segment is anomalous but its shape is not specific. Treat it as "
            "detected ground motion until independent evidence is available."
        )
        confidence = 0.40
        reasons.append("no single feature is diagnostic")

    if centroid is not None:
        reasons.append(f"spectral centroid {safe_float(centroid):.2f} Hz")
    if pulse_count:
        reasons.append(f"{pulse_count} pulse(s)")
    if peak_to_rms:
        reasons.append(f"peak/RMS {peak_to_rms:.1f}")

    return {
        "key": key,
        "label": label,
        "confidence": float(max(0.0, min(0.95, confidence))),
        "hypothesis": hypothesis,
        "reasons": reasons,
        "source_confirmed": False,
        "note": "Provisional only; not a confirmed flood, GLOF, landslide or earthquake.",
    }


def detect_station_events(station, trace, config):
    """Detect envelope/energy anomalies in one prepared station trace."""
    envelope = envelope_of_trace(trace, config)
    fs = float(trace.stats.sampling_rate)
    baseline, mad = robust_noise_level(envelope)
    detection_config = config["detection"]
    threshold = (
        baseline * float(detection_config["threshold_factor"])
        + float(detection_config["mad_factor"]) * mad
    )
    if threshold <= 0:
        threshold = max(float(np.max(envelope)) * 0.25, 1e-12)

    mask = envelope > threshold
    gap_samples = max(1, int(round(float(detection_config["merge_gap"]) * fs)))
    min_samples = max(1, int(round(float(detection_config["min_duration"]) * fs)))
    segments = mask_to_segments(mask, gap_samples, min_samples)

    events = []
    for start_index, end_index in segments:
        try:
            event = event_features(
                trace,
                envelope,
                start_index,
                end_index,
                config,
                baseline,
                threshold,
            )
        except ValueError:
            continue
        event.update(
            {
                "station": station["key"],
                "station_label": station["label"],
                "active": bool(event["end"] >= trace.stats.endtime - 1.0 / fs),
                "start_uncertain": bool(event["start"] <= trace.stats.starttime + 1.0 / fs),
                "end_uncertain": bool(event["end"] >= trace.stats.endtime - 1.0 / fs),
                "detected_utc": UTCDateTime(),
            }
        )
        events.append(event)

    events.sort(key=lambda item: item["peak_amp"], reverse=True)
    events = events[: int(detection_config["max_events_per_cycle"])]
    events.sort(key=lambda item: item["start"])
    diagnostics = {
        "baseline": float(baseline),
        "mad": float(mad),
        "threshold": float(threshold),
        "segments": len(segments),
    }
    return events, diagnostics


def find_matching_detection(state, event, tolerance):
    """Find an existing detection for the same station and overlapping signal."""
    for detection in reversed(state["detections"]):
        if detection["station"] != event["station"]:
            continue
        if detection.get("retired"):
            continue
        if (
            detection["start"] <= event["end"] + tolerance
            and event["start"] <= detection["end"] + tolerance
        ):
            return detection
    return None


def detection_view(detection, state):
    """JSON-safe detection description used by the web page."""
    station = state["stations"].get(detection["station"], {})
    classification = detection.get("classification") or {}
    return {
        "id": detection.get("id"),
        "station": detection.get("station"),
        "station_label": detection.get("station_label") or station.get("label"),
        "candidate_id": detection.get("candidate_id"),
        "start_utc": utc_text(detection.get("start")),
        "end_utc": utc_text(detection.get("end")),
        "duration_s": safe_float(detection.get("duration_s")),
        "significant_duration_s": safe_float(detection.get("significant_duration_s", detection.get("duration_s"))),
        "active": bool(detection.get("active")),
        "energy": safe_float(detection.get("energy")),
        "energy_db": safe_float(detection.get("energy_db")),
        "peak_amp": safe_float(detection.get("peak_amp")),
        "rms_amp": safe_float(detection.get("rms_amp")),
        "peak_to_rms": safe_float(detection.get("peak_to_rms")),
        "snr": safe_float(detection.get("snr")),
        "pulse_count": int(detection.get("pulse_count") or 0),
        "energy_trend_db_per_s": safe_float(detection.get("energy_trend_db_per_s")),
        "dominant_frequency_hz": detection.get("dominant_frequency_hz"),
        "spectral_centroid_hz": detection.get("spectral_centroid_hz"),
        "spectral_bandwidth_hz": detection.get("spectral_bandwidth_hz"),
        "classification": classification,
        "start_uncertain": bool(detection.get("start_uncertain")),
        "end_uncertain": bool(detection.get("end_uncertain")),
    }


def upsert_detection(state, station, event, config, now):
    """Insert a new detection or merge it with an overlapping one.

    Returns ``(detection, created)``. The stable detection id is what candidates
    reference, so re-detecting an ongoing event never creates duplicate triggers.
    """
    tolerance = float(config["detection"]["dedupe_seconds"])
    with state["lock"]:
        if station["key"] not in state["stations"]:
            return None, False
        existing = find_matching_detection(state, event, tolerance)
        if existing is None:
            detection = dict(event)
            detection["id"] = next_id(state, "detection")
            detection["station"] = station["key"]
            detection["station_label"] = station["label"]
            detection["candidate_id"] = None
            detection["created"] = now
            detection["last_seen"] = now
            state["detections"].append(detection)
            log_event(
                state,
                "detection",
                (
                    f"{station['label']}: {event['classification']['label']} "
                    f"for {event.get('significant_duration_s', event['duration_s']):.2f} s"
                ),
                station=station["key"],
                detection=detection["id"],
                details=detection_view(detection, state),
            )
            return detection, True

        old_status = bool(existing.get("active"))
        old_class = (existing.get("classification") or {}).get("key")

        # When an event is already running and the analysis window is trimmed,
        # a new segment can appear to start at the window edge. Keep the real
        # earlier onset in that case.
        if event.get("start_uncertain") and existing["start"] < event["start"]:
            start = existing["start"]
        else:
            start = min(existing["start"], event["start"])
        end = max(existing["end"], event["end"])
        if event.get("start_uncertain") and existing["start"] < event["start"]:
            start = existing["start"]

        existing["start"] = start
        existing["end"] = end
        existing["duration_s"] = float(end - start)
        existing["significant_duration_s"] = max(
            safe_float(existing.get("significant_duration_s")),
            safe_float(event.get("significant_duration_s")),
        )
        existing["core_start"] = event.get("core_start", existing.get("core_start"))
        existing["core_end"] = event.get("core_end", existing.get("core_end"))
        existing["core_level"] = event.get("core_level", existing.get("core_level"))
        existing["active"] = bool(event.get("active"))
        existing["start_uncertain"] = bool(existing.get("start_uncertain")) and bool(event.get("start_uncertain"))
        existing["end_uncertain"] = bool(event.get("end_uncertain"))
        existing["last_seen"] = now
        existing["peak_amp"] = max(safe_float(existing.get("peak_amp")), safe_float(event.get("peak_amp")))
        existing["rms_amp"] = max(safe_float(existing.get("rms_amp")), safe_float(event.get("rms_amp")))
        existing["peak_envelope"] = max(safe_float(existing.get("peak_envelope")), safe_float(event.get("peak_envelope")))
        existing["energy"] = max(safe_float(existing.get("energy")), safe_float(event.get("energy")))
        existing["energy_db"] = max(safe_float(existing.get("energy_db")), safe_float(event.get("energy_db")))
        existing["peak_to_rms"] = (
            existing["peak_amp"] / existing["rms_amp"] if existing["rms_amp"] > 0 else safe_float(event.get("peak_to_rms"))
        )
        existing["pulse_count"] = max(int(existing.get("pulse_count") or 0), int(event.get("pulse_count") or 0))
        existing["energy_trend_db_per_s"] = safe_float(event.get("energy_trend_db_per_s"))
        existing["dominant_frequency_hz"] = event.get("dominant_frequency_hz")
        existing["spectral_centroid_hz"] = event.get("spectral_centroid_hz")
        existing["spectral_bandwidth_hz"] = event.get("spectral_bandwidth_hz")
        existing["classification"] = classify_event(existing, config)

        if existing.get("candidate_id"):
            candidate = find_candidate(state, existing["candidate_id"])
            if candidate is not None:
                candidate["last_update"] = now
                if old_status != existing["active"] or old_class != existing["classification"]["key"]:
                    candidate["dirty"] = True

        if old_status and not existing["active"]:
            log_event(
                state,
                "detection_end",
                f"{station['label']}: activity ended after {existing.get('significant_duration_s', existing['duration_s']):.2f} s",
                station=station["key"],
                detection=existing["id"],
                details=detection_view(existing, state),
            )
        return existing, False


def close_stale_active_detections(state, station, trace_end, config, now):
    """Close active detections when the station stops reporting the signal."""
    stale_after = float(config["detection"]["stale_active_seconds"])
    with state["lock"]:
        for detection in state["detections"]:
            if detection["station"] != station["key"] or not detection.get("active"):
                continue
            if detection.get("retired"):
                continue
            quiet_for = float(now - detection["last_seen"])
            behind = float(trace_end - detection["end"]) if trace_end is not None else 0.0
            if quiet_for >= stale_after or behind >= stale_after:
                detection["active"] = False
                detection["classification"] = classify_event(detection, config)
                if detection.get("candidate_id"):
                    candidate = find_candidate(state, detection["candidate_id"])
                    if candidate is not None:
                        candidate["dirty"] = True
                log_event(
                    state,
                    "detection_end",
                    f"{station['label']}: activity ended after {detection.get('significant_duration_s', detection['duration_s']):.2f} s",
                    station=station["key"],
                    detection=detection["id"],
                    details=detection_view(detection, state),
                )


# ---------------------------------------------------------------------------
# Event log and candidate bookkeeping
# ---------------------------------------------------------------------------


def log_event(state, kind, message, **fields):
    """Append one entry to the in-memory live event log."""
    with state["lock"]:
        entry = {
            "id": next_id(state, "log"),
            "time": UTCDateTime(),
            "kind": kind,
            "message": message,
        }
        entry.update(fields)
        state["event_log"].append(entry)
        limit = int(state["config"]["retention"]["log"])
        if len(state["event_log"]) > limit:
            state["event_log"] = state["event_log"][-limit:]
        return entry


def find_candidate(state, candidate_id):
    """Return a candidate by id, or None."""
    for candidate in state["candidates"]:
        if candidate["id"] == candidate_id:
            return candidate
    return None


def geodesic_km(station_a, station_b):
    """Great-circle distance between two station configurations, in km."""
    _, _, metres = core.geod.inv(
        float(station_a["lon"]),
        float(station_a["lat"]),
        float(station_b["lon"]),
        float(station_b["lat"]),
    )
    return float(metres) / 1000.0


def time_compatibility(detection_a, detection_b, station_a, station_b, config):
    """Check onset times against the fastest plausible propagation."""
    distance = geodesic_km(station_a, station_b)
    v_min = float(config["localization"]["v_min"])
    tolerance = float(config["association"]["time_tolerance"])
    max_dt = distance / max(v_min, 1e-6) + tolerance
    observed = abs(float(detection_a["start"] - detection_b["start"]))
    return {
        "ok": bool(observed <= max_dt),
        "observed_s": float(observed),
        "max_dt_s": float(max_dt),
        "score": float(max(0.0, 1.0 - observed / max(max_dt, 1e-6))),
        "distance_km": float(distance),
    }


def candidate_deadline(candidate, state, config):
    """Last time to wait for more stations, based on array size and v_min."""
    detections = [
        detection
        for detection in state["detections"]
        if detection.get("id") in set(candidate.get("detection_ids", []))
    ]
    stations = [state["stations"][detection["station"]] for detection in detections if detection["station"] in state["stations"]]
    if len(stations) >= 2:
        max_distance = max(
            (geodesic_km(a, b) for a, b in combinations(stations, 2)),
            default=0.0,
        )
        max_duration = max((safe_float(item.get("duration_s")) for item in detections), default=0.0)
        wait = max_distance / max(float(config["localization"]["v_min"]), 1e-6)
        wait += max_duration + float(config["association"]["grace_seconds"]) + 15.0
        return candidate["created"] + min(300.0, max(45.0, wait))
    return candidate["created"] + float(config["association"]["deadline_seconds"])


def ensure_detection_candidate(state, detection, config, now):
    """Attach a detection to a compatible candidate, or open a new candidate."""
    if detection.get("candidate_id"):
        return find_candidate(state, detection["candidate_id"])
    if detection["station"] not in state["stations"]:
        return None

    station = state["stations"][detection["station"]]
    best_candidate = None
    best_score = -1.0

    for candidate in state["candidates"]:
        if candidate.get("status") in ("expired", "retired"):
            continue
        if now > candidate_deadline(candidate, state, config) + 10.0:
            continue
        candidate_detections = [
            item
            for item in state["detections"]
            if item.get("id") in set(candidate.get("detection_ids", []))
        ]
        if any(item["station"] == detection["station"] for item in candidate_detections):
            continue
        scores = []
        compatible = True
        for other in candidate_detections:
            other_station = state["stations"].get(other["station"])
            if other_station is None:
                compatible = False
                break
            result = time_compatibility(detection, other, station, other_station, config)
            if not result["ok"]:
                compatible = False
                break
            scores.append(result["score"])
        if compatible and scores:
            score = float(np.mean(scores))
            if score > best_score:
                best_score = score
                best_candidate = candidate

    if best_candidate is None:
        candidate = {
            "id": next_id(state, "candidate"),
            "created": now,
            "last_new_detection": now,
            "last_update": now,
            "last_evaluated": None,
            "next_refresh": None,
            "deadline": now + float(config["association"]["deadline_seconds"]),
            "detection_ids": [detection["id"]],
            "status": "pending",
            "dirty": True,
            "association": None,
            "localization": None,
            "status_message": "Waiting for another station",
            "notes": [],
        }
        candidate["deadline"] = candidate_deadline(candidate, state, config)
        state["candidates"].append(candidate)
        detection["candidate_id"] = candidate["id"]
        log_event(
            state,
            "candidate",
            f"Candidate {candidate['id']} opened by {detection['station_label']}",
            candidate=candidate["id"],
            station=detection["station"],
            detection=detection["id"],
        )
        return candidate

    detection["candidate_id"] = best_candidate["id"]
    best_candidate["detection_ids"].append(detection["id"])
    best_candidate["last_new_detection"] = now
    best_candidate["last_update"] = now
    best_candidate["dirty"] = True
    best_candidate["status"] = "verifying"
    best_candidate["status_message"] = "Checking cross-station compatibility"
    best_candidate["deadline"] = candidate_deadline(best_candidate, state, config)
    log_event(
        state,
        "candidate",
        (
            f"Candidate {best_candidate['id']} now includes "
            f"{detection['station_label']}"
        ),
        candidate=best_candidate["id"],
        station=detection["station"],
        detection=detection["id"],
        details={"time_score": best_score},
    )
    return best_candidate


def candidate_detections(state, candidate):
    """Return live detection dictionaries referenced by a candidate."""
    wanted = set(candidate.get("detection_ids", []))
    return [
        detection
        for detection in state["detections"]
        if detection.get("id") in wanted and not detection.get("retired")
    ]


# ---------------------------------------------------------------------------
# Cross-station association
# ---------------------------------------------------------------------------


def load_event_trace(state, detection, start, end, config):
    """Extract and preprocess the buffer segment covering one detection."""
    stream = snapshot_buffer(state, detection["station"])
    if len(stream) == 0:
        return None
    try:
        stream.merge(method=1, fill_value=None)
    except Exception:
        pass

    overlapping = [
        trace
        for trace in stream
        if trace.stats.endtime >= start and trace.stats.starttime <= end and trace.stats.npts > 0
    ]
    if not overlapping:
        return None

    def overlap(trace):
        return max(0.0, float(min(trace.stats.endtime, end) - max(trace.stats.starttime, start)))

    best = max(overlapping, key=overlap)
    if overlap(best) < min(2.0, max(0.5, safe_float(detection.get("duration_s")) * 0.6)):
        return None
    try:
        cut = best.copy().trim(
            starttime=max(start, best.stats.starttime),
            endtime=min(end, best.stats.endtime),
        )
        if cut.stats.npts < 8:
            return None
        return prepare_trace(cut, config)
    except Exception:
        return None


def align_traces(traces):
    """Trim prepared traces to exactly the same sample grid."""
    if not traces:
        return {}
    fs = min(float(trace.stats.sampling_rate) for trace in traces.values())
    start = max(trace.stats.starttime for trace in traces.values())
    end = min(trace.stats.endtime for trace in traces.values())
    if end <= start:
        return {}

    trimmed = {}
    for name, trace in traces.items():
        work = trace.copy()
        offset = int(round(float(start - work.stats.starttime) * fs))
        if offset < 0:
            continue
        if offset:
            work.data = work.data[offset:]
        work.stats.starttime = start
        work.stats.sampling_rate = fs
        work.stats.delta = 1.0 / fs
        work.stats.npts = len(work.data)
        trimmed[name] = work

    common_end = min(item.stats.endtime for item in trimmed.values())
    count = int(math.floor(float(common_end - start) * fs))
    if count < int(2.0 * fs):
        return {}
    for work in trimmed.values():
        work.data = np.asarray(work.data[:count], dtype=np.float64)
        work.stats.npts = count
        work.stats.starttime = start
        work.stats.sampling_rate = fs
    return trimmed


def pair_coherence(trace_a, trace_b, config, span_seconds):
    """Mean and 90th percentile magnitude-squared coherence."""
    fs = float(trace_a.stats.sampling_rate)
    window = max(1.0, min(10.0, span_seconds / 2.0))
    try:
        results = core.calculate_cross_coherence(
            {"a": trace_a, "b": trace_b},
            fs=fs,
            fmin=float(config["fmin"]),
            fmax=float(config["fmax"]),
            window_seconds=window,
        )
        values = np.asarray(results["a-b"]["coherence"], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0, 0.0
        return float(np.mean(values)), float(np.percentile(values, 90.0))
    except Exception:
        return 0.0, 0.0


def spectral_similarity(data_a, data_b, fs, config):
    """Cosine similarity and dominant-frequency ratio for two event windows."""
    nperseg = min(256, len(data_a), len(data_b))
    frequencies_a, power_a, features_a = welch_spectrum(
        data_a, fs, float(config["fmin"]), float(config["fmax"]), nperseg=nperseg
    )
    frequencies_b, power_b, features_b = welch_spectrum(
        data_b, fs, float(config["fmin"]), float(config["fmax"]), nperseg=nperseg
    )
    if frequencies_a is None or frequencies_b is None or len(power_a) != len(power_b):
        return 0.0, None, features_a, features_b
    norm = float(np.linalg.norm(power_a) * np.linalg.norm(power_b))
    if norm <= 0:
        return 0.0, None, features_a, features_b
    cosine = float(np.dot(power_a, power_b) / norm)
    dominant_a = safe_float(features_a.get("dominant_frequency_hz"))
    dominant_b = safe_float(features_b.get("dominant_frequency_hz"))
    ratio = None
    if dominant_a > 0 and dominant_b > 0:
        ratio = max(dominant_a, dominant_b) / min(dominant_a, dominant_b)
    return cosine, ratio, features_a, features_b


def analyze_pair(station_a, station_b, trace_a, trace_b, detection_a, detection_b, config):
    """Compare one station pair using time, envelope, coherence and spectra."""
    association_config = config["association"]
    localization_config = config["localization"]
    fs = float(trace_a.stats.sampling_rate)
    span = float(min(trace_a.stats.endtime, trace_b.stats.endtime) - max(trace_a.stats.starttime, trace_b.stats.starttime))
    distance = geodesic_km(station_a, station_b)
    v_min = float(localization_config["v_min"])
    tolerance = float(association_config["time_tolerance"])

    max_lag = min(distance / max(v_min, 1e-6) + 2.0, max(0.5, 0.90 * span))
    max_lag = max(max_lag, 2.0 / fs)

    try:
        envelope_a = envelope_of_trace(trace_a, config)
        envelope_b = envelope_of_trace(trace_b, config)
        lag_result = core.calculate_envelope_lag(envelope_a, envelope_b, fs, max_lag)
    except Exception as error:
        return {
            "a": station_a["key"],
            "b": station_b["key"],
            "supported": False,
            "error": str(error),
            "distance_km": distance,
        }

    observed_dt = float(detection_a["start"] - detection_b["start"])
    max_dt = distance / max(v_min, 1e-6) + tolerance
    time_ok = abs(observed_dt) <= max_dt
    lag_ok = abs(lag_result["lag"]) <= distance / max(v_min, 1e-6) + tolerance
    lag_dt_diff = abs(lag_result["lag"] - observed_dt)
    lag_dt_ok = lag_dt_diff <= max(1.2, 0.10 * max_lag)
    lag_at_limit = abs(lag_result["lag"]) >= 0.98 * max_lag
    coherence_mean, coherence_p90 = pair_coherence(trace_a, trace_b, config, span)

    n = min(len(trace_a.data), len(trace_b.data))
    spectral_cosine, dominant_ratio, features_a, features_b = spectral_similarity(
        trace_a.data[:n], trace_b.data[:n], fs, config
    )
    duration_a = max(safe_float(detection_a.get("duration_s")), 1e-3)
    duration_b = max(safe_float(detection_b.get("duration_s")), 1e-3)
    duration_ratio = min(duration_a, duration_b) / max(duration_a, duration_b)
    duration_ok = duration_ratio >= float(association_config["min_duration_ratio"])

    time_score = max(0.0, 1.0 - abs(observed_dt) / max(max_dt, 1e-6))
    lag_score = max(0.0, 1.0 - lag_dt_diff / max(1.2, 0.10 * max_lag))
    cc = max(0.0, safe_float(lag_result.get("cc_max")))
    cc_score = min(1.0, cc)
    coherence_score = min(1.0, max(0.0, coherence_mean))
    spectral_score = max(0.0, spectral_cosine)
    duration_score = duration_ratio

    score = (
        0.25 * time_score
        + 0.20 * lag_score
        + 0.25 * cc_score
        + 0.10 * coherence_score
        + 0.10 * spectral_score
        + 0.10 * duration_score
    )
    supported = (
        time_ok
        and lag_ok
        and lag_dt_ok
        and duration_ok
        and not lag_at_limit
        and (cc >= float(association_config["min_cc"]) or coherence_mean >= float(association_config["min_coherence"]))
        and score >= float(association_config["min_pair_score"])
    )

    return {
        "a": station_a["key"],
        "b": station_b["key"],
        "a_label": station_a["label"],
        "b_label": station_b["label"],
        "distance_km": float(distance),
        "max_lag_s": float(max_lag),
        "lag_s": float(lag_result["lag"]),
        "cc_max": float(lag_result["cc_max"]),
        "observed_dt_s": observed_dt,
        "max_dt_s": float(max_dt),
        "lag_dt_diff_s": float(lag_dt_diff),
        "max_lag_hit": bool(lag_at_limit),
        "time_ok": bool(time_ok),
        "lag_physical": bool(lag_ok),
        "lag_time_consistent": bool(lag_dt_ok),
        "coherence_mean": float(coherence_mean),
        "coherence_p90": float(coherence_p90),
        "spectral_cosine": float(spectral_cosine),
        "dominant_frequency_ratio": dominant_ratio,
        "dominant_a_hz": features_a.get("dominant_frequency_hz"),
        "dominant_b_hz": features_b.get("dominant_frequency_hz"),
        "duration_ratio": float(duration_ratio),
        "time_score": float(time_score),
        "lag_score": float(lag_score),
        "score": float(score),
        "supported": bool(supported),
    }


def lag_lookup(lag_map, station_a, station_b):
    """Return lag(station_a - station_b) from a directed lag map."""
    if (station_a, station_b) in lag_map:
        return lag_map[(station_a, station_b)]
    if (station_b, station_a) in lag_map:
        return -lag_map[(station_b, station_a)]
    return None


def pair_closure_scores(lag_map, station_names):
    """Maximum closure error for every valid station triplet."""
    closures = []
    for station_a, station_b, station_c in combinations(station_names, 3):
        lag_ab = lag_lookup(lag_map, station_a, station_b)
        lag_ac = lag_lookup(lag_map, station_a, station_c)
        lag_bc = lag_lookup(lag_map, station_b, station_c)
        if None in (lag_ab, lag_ac, lag_bc):
            continue
        error = lag_ac - lag_ab - lag_bc
        closures.append(
            {
                "stations": [station_a, station_b, station_c],
                "closure_s": float(error),
            }
        )
    return closures


def evaluate_association(state, candidate, config):
    """Check all candidate pairs and report whether cross-station support exists."""
    detections = candidate_detections(state, candidate)
    result = {
        "status": "insufficient",
        "verified": False,
        "reason": "Fewer than two stations are available.",
        "station_keys": [item["station"] for item in detections],
        "pairs": [],
        "missing": [],
        "closure": [],
        "max_closure_s": None,
        "mean_cc": None,
        "mean_coherence": None,
        "mean_pair_score": None,
        "support_stations": [],
        "bad_time_pairs": [],
        "notes": [],
    }
    if len(detections) < 2:
        return result

    now = UTCDateTime()
    association_config = config["association"]
    starts = [item["start"] for item in detections]
    ends = [min(now, item["end"]) for item in detections]
    window_start = min(starts) - float(association_config["pre_margin"])
    window_end = max(ends) + float(association_config["post_margin"])
    max_window = float(association_config["max_window"])
    if float(window_end - window_start) > max_window:
        window_start = window_end - max_window
    if window_end <= window_start:
        result["reason"] = "No usable common time window."
        return result

    traces = {}
    loaded_detections = {}
    for detection in detections:
        station = state["stations"].get(detection["station"])
        if station is None:
            result["missing"].append(detection["station"])
            continue
        trace = load_event_trace(state, detection, window_start, window_end, config)
        if trace is None:
            result["missing"].append(detection["station"])
            continue
        traces[detection["station"]] = trace
        loaded_detections[detection["station"]] = detection

    traces = align_traces(traces)
    if len(traces) < 2:
        result["reason"] = "Not enough contiguous event waveforms for a pair check."
        return result

    pairs = []
    lag_map = {}
    station_names = list(traces.keys())
    for station_a, station_b in combinations(station_names, 2):
        pair = analyze_pair(
            state["stations"][station_a],
            state["stations"][station_b],
            traces[station_a],
            traces[station_b],
            loaded_detections[station_a],
            loaded_detections[station_b],
            config,
        )
        pairs.append(pair)
        if "lag_s" in pair:
            lag_map[(station_a, station_b)] = pair["lag_s"]

    supported_pairs = [pair for pair in pairs if pair.get("supported")]
    support_stations = sorted(
        {
            name
            for pair in supported_pairs
            for name in (pair["a"], pair["b"])
        }
    )
    closures = pair_closure_scores(lag_map, station_names)
    max_closure = max((abs(item["closure_s"]) for item in closures), default=None)
    closure_limit = float(association_config["closure_tolerance"])

    times = [pair for pair in pairs if not pair.get("time_ok", True)]
    if times:
        result["bad_time_pairs"] = [f"{pair['a']}–{pair['b']}" for pair in times]
    if max_closure is not None and max_closure > closure_limit:
        result["notes"].append(
            f"Pairwise TDOA closure is {max_closure:.2f} s, above the {closure_limit:.1f} s tolerance."
        )

    if supported_pairs:
        result["mean_cc"] = float(np.mean([pair["cc_max"] for pair in supported_pairs]))
        result["mean_coherence"] = float(np.mean([pair["coherence_mean"] for pair in supported_pairs]))
        result["mean_pair_score"] = float(np.mean([pair["score"] for pair in supported_pairs]))

    verified = (
        bool(supported_pairs)
        and len(support_stations) >= 2
        and (max_closure is None or max_closure <= closure_limit)
        and not times
    )
    if verified:
        result["status"] = "supported"
        result["verified"] = True
        result["reason"] = (
            f"{len(support_stations)} stations passed time, envelope, coherence "
            "and duration checks."
        )
    elif times or (max_closure is not None and max_closure > closure_limit):
        result["status"] = "inconsistent"
        result["reason"] = "Time or pairwise-delay consistency checks failed."
    elif supported_pairs:
        result["status"] = "weak"
        result["reason"] = "Only one station has a supported pair, or support is too weak."
    else:
        result["status"] = "unverified"
        result["reason"] = "No pair passed the association checks."

    result.update(
        {
            "pairs": pairs,
            "closure": closures,
            "max_closure_s": float(max_closure) if max_closure is not None else None,
            "support_stations": support_stations,
            "station_keys": station_names,
        }
    )
    return result


# ---------------------------------------------------------------------------
# Envelope-TDOA source localization
# ---------------------------------------------------------------------------


def station_geometry(stations, source_lat=None, source_lon=None):
    """Describe array geometry and azimuthal coverage around a source."""
    if len(stations) < 2:
        return {
            "stations": len(stations),
            "spread_km": 0.0,
            "linearity": 0.0,
            "azimuth_gap_deg": None,
            "score": 0.0,
        }

    origin = stations[0]
    points = np.array(
        [
            core.latlon_to_xy(
                float(origin["lat"]),
                float(origin["lon"]),
                float(station["lat"]),
                float(station["lon"]),
            )
            for station in stations
        ],
        dtype=np.float64,
    )
    spread = 0.0
    for index_a, index_b in combinations(range(len(points)), 2):
        spread = max(spread, float(np.linalg.norm(points[index_a] - points[index_b])))

    centered = points - np.mean(points, axis=0)
    linearity = 0.0
    if centered.shape[0] >= 2:
        eigenvalues = np.linalg.eigvalsh(np.cov(centered.T) if centered.shape[0] > 1 else np.zeros((2, 2)))
        eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))
        if eigenvalues[-1] > 0:
            linearity = float(eigenvalues[0] / eigenvalues[-1])

    if source_lat is None or source_lon is None:
        source_lat = float(np.mean([station["lat"] for station in stations]))
        source_lon = float(np.mean([station["lon"] for station in stations]))
    bearings = []
    for station in stations:
        azimuth, _, _ = core.geod.inv(
            float(source_lon),
            float(source_lat),
            float(station["lon"]),
            float(station["lat"]),
        )
        bearings.append(float(azimuth) % 360.0)
    bearings.sort()
    gaps = [
        (bearings[(index + 1) % len(bearings)] - bearings[index]) % 360.0
        for index in range(len(bearings))
    ]
    azimuth_gap = max(gaps) if gaps else None

    score = 0.25
    if len(stations) >= 3:
        score += 0.15
    if len(stations) >= 4:
        score += 0.20
    if spread >= 10.0:
        score += 0.10
    if linearity >= 0.10:
        score += 0.10
    if azimuth_gap is not None and azimuth_gap <= 180.0:
        score += 0.10
    if azimuth_gap is not None and azimuth_gap <= 120.0:
        score += 0.10

    return {
        "stations": len(stations),
        "spread_km": float(spread),
        "linearity": float(linearity),
        "azimuth_gap_deg": float(azimuth_gap) if azimuth_gap is not None else None,
        "score": float(min(1.0, score)),
    }


def invert_lags(stations, lag_pairs, config, v_min=None, v_max=None, fix_velocity=None):
    """Call the existing geodetic joint inversion in triangulate.py.

    Realtime keeps its own candidate-level outlier rejection, so the core
    inversion is asked not to silently drop pairs that the caller still reports.
    """
    station_latlon = {
        station["key"]: (float(station["lat"]), float(station["lon"]))
        for station in stations
    }
    localization_config = config["localization"]
    v_min = float(localization_config["v_min"] if v_min is None else v_min)
    v_max = float(localization_config["v_max"] if v_max is None else v_max)
    return core.invert_location_velocity_geod(
        station_latlon,
        lag_pairs,
        v_min=v_min,
        v_max=v_max,
        v_step=float(localization_config["v_step"]),
        grid_step_deg=float(localization_config["grid_step_deg"]),
        margin_deg=float(localization_config["margin_deg"]),
        fix_velocity=fix_velocity,
        reject_outliers=False,
    )


def predicted_lag_seconds(station_a, station_b, lat, lon, velocity):
    """Predicted envelope TDOA for station_a minus station_b."""
    _, _, distance_a = core.geod.inv(float(lon), float(lat), float(station_a["lon"]), float(station_a["lat"]))
    _, _, distance_b = core.geod.inv(float(lon), float(lat), float(station_b["lon"]), float(station_b["lat"]))
    return ((float(distance_a) - float(distance_b)) / 1000.0) / max(float(velocity), 1e-6)


def lag_residuals(stations_by_key, lag_pairs, lat, lon, velocity):
    """Per-pair residuals (observed minus predicted), in seconds."""
    residuals = []
    for station_a, station_b, lag, cc in lag_pairs:
        predicted = predicted_lag_seconds(
            stations_by_key[station_a],
            stations_by_key[station_b],
            lat,
            lon,
            velocity,
        )
        residuals.append(
            {
                "a": station_a,
                "b": station_b,
                "lag_s": float(lag),
                "predicted_s": float(predicted),
                "residual_s": float(lag - predicted),
                "weighted_residual_s": float(safe_float(cc, 0.0) * (lag - predicted)),
            }
        )
    return residuals


def localize_envelope_lags(stations, lag_pairs, start_times, config):
    """Estimate a source from envelope TDOAs with explicit quality handling."""
    localization_config = config["localization"]
    result = {
        "status": "insufficient_stations",
        "reliable": False,
        "quality": "insufficient",
        "confidence": 0.0,
        "source": None,
        "velocity": None,
        "velocity_fixed": False,
        "velocity_range": [
            float(localization_config["v_min"]),
            float(localization_config["v_max"]),
        ],
        "station_count": len(stations),
        "pair_count": len(lag_pairs),
        "notes": [],
        "pairs": [],
    }
    if len(stations) < 3:
        result["notes"].append("At least three associated stations are required for localization.")
        return result
    if len(lag_pairs) < 3:
        result["notes"].append("At least three usable station-pair TDOAs are required.")
        return result

    stations_by_key = {station["key"]: station for station in stations}
    underdetermined = len(stations) == 3
    rejected = []
    velocity_fixed = False
    fixed_velocity = None

    try:
        if underdetermined:
            velocity_fixed = True
            fixed_velocity = float(localization_config["v_prior"])
            inversion = invert_lags(
                stations,
                lag_pairs,
                config,
                fix_velocity=fixed_velocity,
            )
            inversion["velocity"] = fixed_velocity
            result["notes"].append(
                "Only three stations: velocity and location trade off. "
                f"Location uses a fixed {fixed_velocity:.2f} km/s prior and is indicative only."
            )
        else:
            inversion = invert_lags(stations, lag_pairs, config)
            first_residuals = lag_residuals(
                stations_by_key,
                lag_pairs,
                inversion["lat"],
                inversion["lon"],
                inversion["velocity"],
            )
            absolute = np.array([abs(item["residual_s"]) for item in first_residuals])
            if len(lag_pairs) > 4 and absolute.size:
                worst_index = int(np.argmax(absolute))
                median = float(np.median(absolute))
                if absolute[worst_index] > max(2.0, 3.0 * median):
                    rejected.append(
                        f"{lag_pairs[worst_index][0]}–{lag_pairs[worst_index][1]}"
                    )
                    reduced_pairs = [
                        pair for index, pair in enumerate(lag_pairs) if index != worst_index
                    ]
                    if len(reduced_pairs) >= 4:
                        lag_pairs = reduced_pairs
                        inversion = invert_lags(stations, lag_pairs, config)
    except Exception as error:
        result.update(
            {
                "status": "localization_failed",
                "quality": "failed",
                "notes": result["notes"] + [f"Inversion failed: {type(error).__name__}: {error}"],
            }
        )
        return result

    lat = float(inversion["lat"])
    lon = float(inversion["lon"])
    velocity = float(inversion["velocity"])
    residuals = lag_residuals(stations_by_key, lag_pairs, lat, lon, velocity)
    weights = np.array([max(0.0, safe_float(item[3])) for item in lag_pairs], dtype=np.float64)
    residual_values = np.array([item["residual_s"] for item in residuals], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(residual_values)
    weighted_rmse = float(
        np.sqrt(np.sum(weights * residual_values ** 2) / max(weights.sum(), 1e-12))
    )
    max_abs_residual = float(np.max(np.abs(residual_values))) if residual_values.size else 0.0
    cc_mean = float(np.mean(weights)) if weights.size else 0.0

    geometry = station_geometry(stations, source_lat=lat, source_lon=lon)
    # The core inversion reports whether the refined solution is pinned to the
    # edge of its search grid, which is a warning that the array cannot reconcile
    # the observed TDOAs.
    boundary_hit = bool(inversion.get("boundary_hit", False))
    if not boundary_hit:
        margin = float(localization_config["margin_deg"])
        lats = [float(station["lat"]) for station in stations]
        lons = [float(station["lon"]) for station in stations]
        edge = max(0.02, margin * 0.08)
        boundary_hit = bool(
            lat <= min(lats) - margin + edge
            or lat >= max(lats) + margin - edge
            or lon <= min(lons) - margin + edge
            or lon >= max(lons) + margin - edge
        )

    residual_score = max(
        0.0,
        1.0 - weighted_rmse / max(float(localization_config["max_pair_rmse"]), 1e-6),
    )
    confidence = 0.45 * residual_score + 0.25 * min(1.0, cc_mean) + 0.30 * geometry["score"]
    if underdetermined:
        confidence = min(0.40, confidence)
    if boundary_hit:
        confidence *= 0.65
    if geometry["spread_km"] < float(localization_config["min_spread_km"]):
        confidence *= 0.7
    confidence = float(max(0.0, min(1.0, confidence)))

    if underdetermined:
        status = "underdetermined"
        quality = "low"
        reliable = False
        result["notes"].append("Source coordinates are reported only as an indicative estimate.")
    elif weighted_rmse > float(localization_config["max_pair_rmse"]):
        status = "inconsistent"
        quality = "inconsistent"
        reliable = False
        result["notes"].append("TDOA residuals exceed the configured tolerance.")
    elif boundary_hit:
        status = "inconsistent"
        quality = "low"
        reliable = False
        result["notes"].append("The solution sits on the inversion boundary; the array may be too small.")
    elif geometry["score"] < float(localization_config["min_geometry_score"]):
        status = "underdetermined"
        quality = "low"
        reliable = False
        result["notes"].append("Station geometry is poor for a unique source location.")
    else:
        status = "localized"
        reliable = True
        quality = "high" if confidence >= 0.70 else "medium" if confidence >= 0.45 else "low"
        result["notes"].append("Source location is an envelope-TDOA estimate, not a confirmed hazard location.")

    origin_times = []
    for key, start in start_times.items():
        station = stations_by_key.get(key)
        if station is None:
            continue
        _, _, distance = core.geod.inv(
            lon,
            lat,
            float(station["lon"]),
            float(station["lat"]),
        )
        origin_times.append(float(start - (float(distance) / 1000.0) / max(velocity, 1e-6)))
    if origin_times:
        origin_timestamp = float(np.median(origin_times))
        origin_spread = float(max(origin_times) - min(origin_times))
    else:
        origin_timestamp = None
        origin_spread = None

    x_km, y_km = core.latlon_to_xy(
        float(stations[0]["lat"]),
        float(stations[0]["lon"]),
        lat,
        lon,
    )

    result.update(
        {
            "status": status,
            "reliable": bool(reliable),
            "quality": quality,
            "confidence": confidence,
            "station_count": len(stations),
            "pair_count": len(lag_pairs),
            "underdetermined": bool(underdetermined),
            "velocity_fixed": bool(velocity_fixed),
            "velocity": velocity,
            "velocity_prior": fixed_velocity,
            "source": {
                "lat": lat,
                "lon": lon,
                "x_km": float(x_km),
                "y_km": float(y_km),
                "origin_time_utc": utc_text(UTCDateTime(origin_timestamp)) if origin_timestamp is not None else None,
                "origin_time_spread_s": origin_spread,
            },
            "geometry": geometry,
            "boundary_hit": bool(boundary_hit),
            "weighted_rmse_s": weighted_rmse,
            "max_abs_residual_s": max_abs_residual,
            "residual_norm": safe_float(inversion.get("residual_norm")),
            "rejected_pairs": rejected,
            "pairs": residuals,
            "lag_pairs_used": len(lag_pairs),
        }
    )
    for item in residuals:
        item["a_label"] = stations_by_key[item["a"]]["label"]
        item["b_label"] = stations_by_key[item["b"]]["label"]
    return result


# ---------------------------------------------------------------------------
# Candidate evaluation and results
# ---------------------------------------------------------------------------


def summarize_classifications(detections):
    """Combine per-station provisional labels without overstating a source."""
    if not detections:
        return {
            "key": "unknown",
            "label": "Unknown",
            "hypothesis": "No classification available.",
            "reasons": [],
            "source_confirmed": False,
        }
    ranked = sorted(
        detections,
        key=lambda item: safe_float(item.get("significant_duration_s", item.get("duration_s"))),
        reverse=True,
    )
    longest = ranked[0]
    classification = longest.get("classification") or {}
    durations = [
        safe_float(item.get("significant_duration_s", item.get("duration_s")))
        for item in detections
    ]
    labels = [((item.get("classification") or {}).get("label")) for item in detections]
    reasons = list(classification.get("reasons") or [])
    reasons.append(
        f"{len(detections)} station(s); duration range {min(durations):.2f}–{max(durations):.2f} s"
    )
    return {
        "key": classification.get("key", "unknown"),
        "label": classification.get("label", "Unknown"),
        "hypothesis": classification.get("hypothesis"),
        "confidence": classification.get("confidence", 0.0),
        "reasons": reasons,
        "station_labels": labels,
        "duration_min_s": float(min(durations)),
        "duration_max_s": float(max(durations)),
        "source_confirmed": False,
        "note": (
            "Provisional classification only. It does not confirm a flood, GLOF, "
            "landslide, earthquake or other source."
        ),
    }


def candidate_ready(candidate, state, config, now):
    """Decide whether a candidate has had enough time to receive other stations."""
    detections = candidate_detections(state, candidate)
    association_config = config["association"]
    grace = float(association_config["grace_seconds"])
    deadline = candidate.get("deadline") or (
        candidate["created"] + float(association_config["deadline_seconds"])
    )

    if candidate.get("dirty"):
        if float(now - candidate["last_new_detection"]) >= grace:
            return True
        if now >= deadline:
            return True

    if candidate.get("status") in ("pending", "verifying", "unverified") and now >= deadline:
        return True

    if candidate.get("status") in ("verified", "localized", "underdetermined", "inconsistent", "unverified"):
        refresh_at = candidate.get("next_refresh")
        if refresh_at is not None and now >= refresh_at:
            if any(detection.get("active") for detection in detections):
                return True

    return False


def apply_candidate_result(state, candidate, association, config, budget, now):
    """Turn association evidence into candidate status and localization."""
    candidate["association"] = association
    candidate["last_evaluated"] = now
    candidate["dirty"] = False
    previous_status = candidate.get("status")
    previous_localization = candidate.get("localization")
    localization = None
    status = candidate.get("status")
    message = association.get("reason") or "Candidate evaluated."
    notes = list(association.get("notes") or [])

    if association.get("verified"):
        support_keys = [
            key for key in association.get("support_stations", []) if key in state["stations"]
        ]
        if len(support_keys) < 3:
            status = "verified"
            message = (
                f"Cross-station support from {len(support_keys)} stations; "
                "at least three are needed for localization."
            )
            notes.append("Localization withheld because fewer than three stations are associated.")
        elif budget["localizations"] >= int(config["association"]["max_localizations_per_cycle"]):
            candidate["dirty"] = True
            status = "verified"
            message = "Association verified; localization queued for the next processing cycle."
        else:
            lag_pairs = [
                (pair["a"], pair["b"], pair["lag_s"], pair["cc_max"])
                for pair in association.get("pairs", [])
                if pair.get("supported")
                and pair.get("a") in support_keys
                and pair.get("b") in support_keys
            ]
            detections = [
                detection
                for detection in candidate_detections(state, candidate)
                if detection["station"] in support_keys
            ]
            start_times = {detection["station"]: detection["start"] for detection in detections}
            budget["localizations"] += 1
            localization = localize_envelope_lags(
                [state["stations"][key] for key in support_keys],
                lag_pairs,
                start_times,
                config,
            )
            candidate["localization"] = localization
            status = localization["status"]
            message = {
                "localized": "Cross-station verification and localization succeeded.",
                "underdetermined": "Cross-station support exists, but the source is underdetermined.",
                "inconsistent": "Cross-station verification passed but the location is inconsistent.",
                "insufficient_stations": "Not enough associated stations for localization.",
                "localization_failed": "Localization failed.",
            }.get(status, "Localization finished.")
            notes.extend(localization.get("notes") or [])
    else:
        if association.get("status") == "inconsistent":
            status = "inconsistent"
        elif now >= candidate.get("deadline", now):
            status = "expired"
            message = "Candidate expired before enough compatible stations arrived."
        else:
            status = "unverified"
        localization = candidate.get("localization")

    if localization is None and previous_localization is not None:
        localization = previous_localization
    candidate["localization"] = localization
    candidate["status"] = status
    candidate["status_message"] = message
    candidate["notes"] = notes
    candidate["next_refresh"] = now + float(config["association"]["localization_refresh"])
    candidate["classification"] = summarize_classifications(candidate_detections(state, candidate))

    status_changed = status != previous_status
    localization_changed = (
        isinstance(localization, dict)
        and isinstance(previous_localization, dict)
        and (
            localization.get("status") != previous_localization.get("status")
            or localization.get("confidence") != previous_localization.get("confidence")
        )
    )
    if status_changed or (status in ("localized", "underdetermined", "inconsistent") and localization_changed):
        details = candidate_view(state, candidate)
        if status == "localized":
            source = (localization or {}).get("source") or {}
            log_event(
                state,
                "localization",
                (
                    f"Candidate {candidate['id']} localized near "
                    f"{source.get('lat'):.4f}, {source.get('lon'):.4f} "
                    f"(quality {localization.get('quality')})"
                ),
                candidate=candidate["id"],
                details=details,
            )
        elif status in ("underdetermined", "inconsistent"):
            log_event(
                state,
                "localization",
                f"Candidate {candidate['id']}: {message}",
                candidate=candidate["id"],
                details=details,
            )
        elif status == "verified":
            log_event(
                state,
                "association",
                f"Candidate {candidate['id']} verified by {len(association.get('support_stations', []))} stations",
                candidate=candidate["id"],
                details=details,
            )
        elif status in ("unverified", "expired"):
            log_event(
                state,
                "association",
                f"Candidate {candidate['id']}: {message}",
                candidate=candidate["id"],
                details=details,
            )
    return candidate


def candidate_view(state, candidate):
    """JSON-safe candidate summary for the log and detail panel."""
    detections = candidate_detections(state, candidate)
    association = candidate.get("association") or {}
    localization = candidate.get("localization") or {}
    return {
        "id": candidate["id"],
        "created_utc": utc_text(candidate.get("created")),
        "status": candidate.get("status"),
        "status_message": candidate.get("status_message"),
        "dirty": bool(candidate.get("dirty")),
        "deadline_utc": utc_text(candidate.get("deadline")),
        "station_keys": sorted({detection["station"] for detection in detections}),
        "station_labels": sorted({detection["station_label"] for detection in detections}),
        "detections": [detection_view(detection, state) for detection in detections],
        "classification": candidate.get("classification") or summarize_classifications(detections),
        "association": {
            "status": association.get("status"),
            "verified": association.get("verified"),
            "reason": association.get("reason"),
            "mean_cc": association.get("mean_cc"),
            "mean_coherence": association.get("mean_coherence"),
            "mean_pair_score": association.get("mean_pair_score"),
            "max_closure_s": association.get("max_closure_s"),
            "support_stations": association.get("support_stations", []),
            "bad_time_pairs": association.get("bad_time_pairs", []),
            "missing": association.get("missing", []),
            "pairs": association.get("pairs", []),
            "closure": association.get("closure", []),
            "notes": association.get("notes", []),
        },
        "localization": localization,
        "notes": candidate.get("notes", []),
    }


def evaluate_ready_candidates(state, config, now):
    """Evaluate only settled/updated candidates, with a per-cycle budget."""
    budget = {"localizations": 0}
    evaluations = 0
    candidates = sorted(
        list(state["candidates"]),
        key=lambda item: (
            len(candidate_detections(state, item)),
            item.get("last_new_detection") or item["created"],
        ),
        reverse=True,
    )
    for candidate in candidates:
        if evaluations >= int(config["association"]["max_evaluations_per_cycle"]):
            break
        if not candidate_ready(candidate, state, config, now):
            continue
        started = time.perf_counter()
        try:
            association = evaluate_association(state, candidate, config)
            apply_candidate_result(state, candidate, association, config, budget, now)
        except Exception as error:
            candidate["status"] = "error"
            candidate["status_message"] = f"{type(error).__name__}: {error}"
            candidate["dirty"] = False
            candidate["last_evaluated"] = now
            log_event(
                state,
                "error",
                f"Candidate {candidate['id']} evaluation failed: {type(error).__name__}: {error}",
                candidate=candidate["id"],
            )
        evaluations += 1
        state["health"]["association_ms"] = (time.perf_counter() - started) * 1000.0
    return evaluations


# ---------------------------------------------------------------------------
# Plot snapshots and the analysis loop
# ---------------------------------------------------------------------------


def downsample_xy(x_values, y_values, max_points):
    """Downsample a line while always retaining the final point."""
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)
    if x_values.size <= max_points:
        return x_values.tolist(), y_values.tolist()
    indices = np.linspace(0, x_values.size - 1, max_points).astype(int)
    indices = np.unique(indices)
    return x_values[indices].tolist(), y_values[indices].tolist()


def build_plot_snapshot(state, station, trace, window_start, window_end, config):
    """Build decimated waveform/envelope/spectrogram arrays and triggers."""
    empty = {
        "available": False,
        "reason": "waiting for contiguous data",
        "start_ms": epoch_ms(window_start),
        "end_ms": epoch_ms(window_end),
        "t_ms": [],
        "waveform": [],
        "envelope": [],
        "spectrogram": None,
        "triggers": [],
    }
    if trace is None or trace.stats.npts < 8:
        return empty

    try:
        display = trace.copy().trim(starttime=window_start, endtime=window_end)
    except Exception:
        return empty
    if display.stats.npts < 8:
        return empty

    fs = float(display.stats.sampling_rate)
    try:
        envelope = envelope_of_trace(display, config)
    except Exception:
        return empty

    times = np.arange(display.stats.npts) / fs
    t_ms = (float(display.stats.starttime.timestamp) + times) * 1000.0
    waveform_t, waveform_y = downsample_xy(t_ms, display.data, 900)
    envelope_t, envelope_y = downsample_xy(t_ms, envelope, 500)

    spectrogram_data = None
    if display.stats.npts >= 32:
        nperseg = min(256, display.stats.npts)
        try:
            frequencies, spec_times, power = spectrogram(
                display.data,
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=int(nperseg * 0.75),
            )
            frequency_limit = max(float(config["fmax"]) * 1.5, float(config["fmax"]) + 1.0)
            frequency_mask = frequencies <= frequency_limit
            frequencies = frequencies[frequency_mask]
            power = power[frequency_mask, :]
            power_db = 10.0 * np.log10(np.maximum(power, 1e-20))
            z_max = float(np.max(power_db)) if power_db.size else 0.0
            z_min = z_max - 60.0
            power_db = np.clip(power_db, z_min, z_max)
            spec_t_ms = (float(display.stats.starttime.timestamp) + spec_times) * 1000.0
            spectrogram_data = {
                "t_ms": spec_t_ms.tolist(),
                "f": frequencies.tolist(),
                "z": power_db.tolist(),
                "zmin": z_min,
                "zmax": z_max,
            }
        except Exception:
            spectrogram_data = None

    triggers = []
    with state["lock"]:
        detections = [item for item in state["detections"] if item["station"] == station["key"]]
        for detection in detections[-60:]:
            if detection.get("retired"):
                continue
            start = detection["start"]
            end = detection["end"]
            if start > window_end or end < window_start:
                continue
            candidate = find_candidate(state, detection["candidate_id"]) if detection.get("candidate_id") else None
            status = (candidate or {}).get("status", "pending")
            triggers.append(
                {
                    "detection_id": detection["id"],
                    "candidate_id": detection.get("candidate_id"),
                    "start_ms": epoch_ms(start),
                    "end_ms": epoch_ms(end),
                    "start_utc": utc_text(start),
                    "end_utc": utc_text(end),
                    "duration_s": safe_float(detection.get("duration_s")),
                    "active": bool(detection.get("active")),
                    "status": status,
                    "label": (detection.get("classification") or {}).get("label", "Activity"),
                }
            )

    return {
        "available": True,
        "start_ms": epoch_ms(window_start),
        "end_ms": epoch_ms(window_end),
        "t_ms": waveform_t,
        "waveform": waveform_y,
        "envelope": envelope_y,
        "spectrogram": spectrogram_data,
        "triggers": triggers,
    }


def run_cycle(state):
    """One lightweight cycle of detection, association and visualization."""
    started = time.perf_counter()
    config = state["config"]
    now = UTCDateTime()
    station_snapshot = []

    with state["lock"]:
        stations = list(state["stations"].values())

    latest_traces = {}
    for station in stations:
        key = station["key"]
        latest_traces[key] = latest_contiguous_trace(state, key)
        with state["lock"]:
            status = state["stations"].get(key, {}).get("status")
            if status is not None:
                last_trace = status.get("last_trace_utc")
                if last_trace is None:
                    status["stale"] = True
                else:
                    status["stale"] = float(now - UTCDateTime(last_trace)) > float(config["stale_seconds"])
        station_snapshot.append((station, latest_traces[key]))

    ends = [
        float(trace.stats.endtime.timestamp)
        for trace in latest_traces.values()
        if trace is not None
    ]
    global_end = UTCDateTime(max(ends)) if ends else now
    window_start = global_end - float(config["display_window"])
    window_end = global_end

    for station, raw_trace in station_snapshot:
        key = station["key"]
        with state["lock"]:
            if key not in state["stations"]:
                # The station was removed while this cycle was running.
                continue
        if raw_trace is None:
            snapshot = build_plot_snapshot(
                state, station, None, window_start, window_end, config
            )
            with state["lock"]:
                if key in state["stations"]:
                    state["plots"][key] = snapshot
            continue

        prepared = None
        try:
            raw_start = max(
                raw_trace.stats.starttime,
                global_end - max(float(config["analysis_window"]), float(config["display_window"])) - 10.0,
            )
            raw = raw_trace.copy().trim(starttime=raw_start, endtime=global_end)
            prepared = prepare_trace(raw, config)
        except Exception as error:
            with state["lock"]:
                if key in state["stations"]:
                    state["plots"][key] = {
                        "available": False,
                        "reason": f"preprocessing failed: {error}",
                        "start_ms": epoch_ms(window_start),
                        "end_ms": epoch_ms(window_end),
                        "t_ms": [],
                        "waveform": [],
                        "envelope": [],
                        "spectrogram": None,
                        "triggers": [],
                    }
            continue

        detection_end = min(prepared.stats.endtime, now)
        detection_start = detection_end - float(config["analysis_window"])
        detection_trace = prepared.copy().trim(starttime=detection_start, endtime=detection_end)
        with state["lock"]:
            present = key in state["stations"]
        if present and float(detection_trace.stats.endtime - detection_trace.stats.starttime) >= float(config["min_detection_window"]):
            try:
                events, diagnostics = detect_station_events(station, detection_trace, config)
                station["status"]["baseline"] = diagnostics["baseline"]
                station["status"]["threshold"] = diagnostics["threshold"]
                for event in events:
                    detection, created = upsert_detection(state, station, event, config, now)
                    if detection is None:
                        break
                    if created or detection.get("candidate_id") is None:
                        ensure_detection_candidate(state, detection, config, now)
                close_stale_active_detections(state, station, detection_trace.stats.endtime, config, now)
            except Exception as error:
                log_event(
                    state,
                    "error",
                    f"{station['label']} detection failed: {type(error).__name__}: {error}",
                    station=key,
                )
        elif present:
            close_stale_active_detections(state, station, prepared.stats.endtime, config, now)

        snapshot = build_plot_snapshot(
            state, station, prepared, window_start, window_end, config
        )
        with state["lock"]:
            if key in state["stations"]:
                state["plots"][key] = snapshot

    evaluate_ready_candidates(state, config, now)
    prune_state(state, now)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    all_ends = [
        float(trace.stats.endtime.timestamp)
        for trace in latest_traces.values()
        if trace is not None
    ]
    data_lag = (now.timestamp - max(all_ends)) if all_ends else None
    connected = sum(1 for station in stations if station["status"].get("connected"))
    pending = sum(
        1
        for candidate in state["candidates"]
        if candidate.get("status") in ("pending", "verifying", "verified")
    )
    warnings = []
    if elapsed_ms > float(config["cycle_seconds"]) * 1000.0:
        warnings.append(f"Processing cycle took {elapsed_ms:.0f} ms, longer than the target interval.")
    if data_lag is not None and data_lag > float(config["stale_seconds"]):
        warnings.append(f"Newest station data is {data_lag:.1f} s old.")
    with state["lock"]:
        state["window"] = {
            "start_ms": epoch_ms(window_start),
            "end_ms": epoch_ms(window_end),
            "start_utc": utc_text(window_start),
            "end_utc": utc_text(window_end),
        }
        state["health"].update(
            {
                "cycle_ms": elapsed_ms,
                "last_cycle_utc": utc_text(now),
                "latest_data_lag_s": data_lag,
                "connected_stations": connected,
                "station_count": len(stations),
                "pending_candidates": pending,
                "warnings": warnings,
            }
        )


def pipeline_loop(state):
    """Run detection/association cycles without blocking the web server."""
    while not state["stop"].is_set():
        started = time.perf_counter()
        try:
            run_cycle(state)
        except Exception:
            log_event(state, "error", f"Pipeline cycle failed:\n{traceback.format_exc(limit=4)}")
        wait = max(0.05, float(state["config"]["cycle_seconds"]) - (time.perf_counter() - started))
        state["stop"].wait(wait)


def prune_state(state, now):
    """Bound memory while keeping the recent live event log intact."""
    retention = state["config"]["retention"]
    with state["lock"]:
        max_detections = int(retention["detections"])
        max_candidates = int(retention["candidates"])
        max_log = int(retention["log"])
        state["detections"] = state["detections"][-max_detections:]
        state["candidates"] = state["candidates"][-max_candidates:]
        state["event_log"] = state["event_log"][-max_log:]

        referenced = {
            detection_id
            for candidate in state["candidates"]
            for detection_id in candidate.get("detection_ids", [])
        }
        for detection in state["detections"]:
            if (
                not detection.get("active")
                and detection["id"] not in referenced
                and float(now - detection["start"]) > float(retention["results_seconds"])
            ):
                detection["retired"] = True


# ---------------------------------------------------------------------------
# Web API
# ---------------------------------------------------------------------------


def station_view(station, now, stale_seconds=20.0):
    """JSON-safe station status."""
    status = station.get("status", {})
    last_trace = status.get("last_trace_utc")
    stale = status.get("stale", True)
    if last_trace is not None:
        stale = float(now - UTCDateTime(last_trace)) > float(stale_seconds)
    return {
        "key": station["key"],
        "net": station["net"],
        "sta": station["sta"],
        "cha": station["cha"],
        "loc": station.get("loc", "*"),
        "label": station["label"],
        "lat": station.get("lat"),
        "lon": station.get("lon"),
        "connected": bool(status.get("connected")),
        "stale": bool(stale),
        "packets": int(status.get("packets", 0)),
        "gaps": int(status.get("gaps", 0)),
        "latency_s": status.get("latency_s"),
        "last_packet_utc": status.get("last_packet_utc"),
        "last_trace_utc": status.get("last_trace_utc"),
        "last_error": status.get("last_error"),
        "last_gap_utc": status.get("last_gap_utc"),
        "last_gap_s": status.get("last_gap_s"),
        "baseline": status.get("baseline"),
        "threshold": status.get("threshold"),
        "added_utc": utc_text(station.get("added_utc")),
    }


def build_state_response(state):
    """Snapshot the current state for the browser."""
    now = UTCDateTime()
    with state["lock"]:
        stations = [
            station_view(station, now, float(state["config"]["stale_seconds"]))
            for station in state["stations"].values()
        ]
        plots = {key: value for key, value in state["plots"].items()}
        detections = [
            detection_view(detection, state)
            for detection in state["detections"][-120:]
            if not detection.get("retired")
        ]
        candidates = [
            candidate_view(state, candidate)
            for candidate in state["candidates"][-30:]
        ]
        candidates.sort(key=lambda item: item.get("created_utc") or "", reverse=True)
        latest_localization = None
        for candidate in reversed(state["candidates"]):
            localization = candidate.get("localization")
            if localization and localization.get("source"):
                latest_localization = {
                    "candidate_id": candidate["id"],
                    "status": candidate.get("status"),
                    **localization,
                }
                break
        event_log = [
            jsonable(entry)
            for entry in reversed(state["event_log"][-160:])
        ]
        window = state.get("window")
        health = dict(state["health"])
        config = state["config"]

    return {
        "server_utc": utc_text(now),
        "started_utc": utc_text(state["started_utc"]),
        "window": window,
        "stations": stations,
        "plots": plots,
        "detections": detections,
        "candidates": candidates,
        "latest_localization": latest_localization,
        "event_log": event_log,
        "health": health,
        "safety": SAFETY_NOTE,
        "config": {
            "server": config["server"],
            "fs": config["fs"],
            "fmin": config["fmin"],
            "fmax": config["fmax"],
            "smooth_cutoff": config["smooth_cutoff"],
            "analysis_window": config["analysis_window"],
            "display_window": config["display_window"],
            "cycle_seconds": config["cycle_seconds"],
            "v_min": config["localization"]["v_min"],
            "v_max": config["localization"]["v_max"],
            "v_prior": config["localization"]["v_prior"],
            "min_stations": config["localization"].get("min_stations", 3),
        },
    }


@APP.route("/api/state")
def api_state():
    try:
        return jsonify(build_state_response(STATE))
    except Exception as error:
        return jsonify(error=f"{type(error).__name__}: {error}"), 500


@APP.route("/api/stations", methods=["POST"])
def api_add_station():
    payload = request.get_json(silent=True) or {}
    try:
        station = normalize_station(payload)
        if station["lat"] is None:
            station["lat"], station["lon"], _ = resolve_station_coordinates(
                station["net"], station["sta"], STATE["config"]["fdsn_server"]
            )
        entry = add_station_to_state(STATE, station, start_seedlink=True)
        return jsonify(station=station_view(entry, UTCDateTime()), message=f"Added {entry['label']}.")
    except ValueError as error:
        return jsonify(error=str(error)), 400
    except Exception as error:
        return jsonify(error=f"{type(error).__name__}: {error}"), 500


@APP.route("/api/stations/<path:key>", methods=["DELETE"])
def api_remove_station(key):
    try:
        remove_station_from_state(STATE, key)
        return jsonify(removed=key, message=f"Removed {key}.")
    except KeyError:
        return jsonify(error=f"Unknown station {key}."), 404
    except Exception as error:
        return jsonify(error=f"{type(error).__name__}: {error}"), 500


@APP.route("/api/health")
def api_health():
    with STATE["lock"]:
        return jsonify(jsonable(STATE["health"]))


@APP.route("/")
def index():
    return render_template_string(PAGE)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Live multi-station seismic anomaly monitor")
    parser.add_argument("--server", default=None, help="SeedLink server, host:port")
    parser.add_argument("--host", default=None, help="Web server bind address")
    parser.add_argument("--port", type=int, default=None, help="Web server port")
    parser.add_argument(
        "--no-seedlink",
        action="store_true",
        help="Run the web UI and detector without opening SeedLink connections",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Run the detection pipeline without starting the web server",
    )
    return parser.parse_args(argv)


def main(argv=None):
    global STATE
    args = parse_args(argv)
    overrides = {}
    if args.server:
        overrides["server"] = args.server
    if args.host:
        overrides["host"] = args.host
    if args.port:
        overrides["port"] = args.port
    config = make_config(overrides)
    STATE = create_state(config=config, no_seedlink=args.no_seedlink)

    if not args.no_seedlink:
        start_all_seedlinks(STATE)
    else:
        log_event(STATE, "system", "SeedLink disabled; waiting for externally supplied data.")

    pipeline = threading.Thread(target=pipeline_loop, args=(STATE,), daemon=True, name="realtime-pipeline")
    pipeline.start()

    print("Live seismic anomaly monitor")
    print(f"  SeedLink: {config['server'] if not args.no_seedlink else 'disabled'}")
    print(f"  Stations: {', '.join(STATE['stations'].keys())}")
    print(f"  Web UI:   http://{config['host']}:{config['port']}/")
    print("  Press Ctrl+C to stop.")

    try:
        if args.no_web:
            while not STATE["stop"].is_set():
                time.sleep(1.0)
        else:
            APP.run(
                host=config["host"],
                port=int(config["port"]),
                debug=False,
                use_reloader=False,
                threaded=True,
            )
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        STATE["stop"].set()
        stop_all_seedlinks(STATE)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live seismic anomaly monitor</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#eef2f4;--panel:#fff;--ink:#14232b;--mute:#5b6b74;--line:#ccd7dc;--accent:#0b6e75;--good:#177a55;--warn:#b96a09;--bad:#b3342a;--info:#1d5d8f;--soft:#f4f8f9}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font:14px/1.45 "Segoe UI",system-ui,-apple-system,sans-serif}
header{display:flex;justify-content:space-between;align-items:flex-start;gap:18px;padding:18px 24px 8px;flex-wrap:wrap}
h1{margin:0;font-size:24px;letter-spacing:-.02em;font-weight:700}
header p{margin:4px 0 0;color:var(--mute);max-width:86ch}
.statusbar{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}
.chip{border:1px solid var(--line);background:#fff;border-radius:999px;padding:5px 10px;font-size:12px;white-space:nowrap}
.chip.good{border-color:#a9d8c5;background:#ecf8f2;color:var(--good)}
.chip.warn{border-color:#efd0a4;background:#fff6e8;color:var(--warn)}
.chip.bad{border-color:#efb8b2;background:#fdf0ee;color:var(--bad)}
.chip.info{border-color:#b8d3e8;background:#eef6fc;color:var(--info)}
.notice{margin:4px 24px 0;padding:11px 14px;border-left:5px solid var(--warn);background:#fff8ec;color:#62430f;border-radius:4px;font-size:13px}
main{display:grid;grid-template-columns:330px minmax(0,1fr);gap:16px;padding:16px 24px 36px;align-items:start}
@media(max-width:1050px){main{grid-template-columns:1fr}header{padding:16px 16px 6px}.notice{margin:4px 16px 0}main{padding:14px 16px 30px}}
.sidebar{display:flex;flex-direction:column;gap:14px;position:sticky;top:12px;max-height:calc(100vh - 24px);overflow:auto;padding-right:2px}
@media(max-width:1050px){.sidebar{position:static;max-height:none}}
.content{display:flex;flex-direction:column;gap:14px;min-width:0}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:13px 14px;box-shadow:0 1px 2px rgba(20,35,43,.04)}
.panel h2{font-size:14px;margin:0 0 9px;display:flex;justify-content:space-between;align-items:center;gap:8px}
.panel h3{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute);margin:12px 0 5px}
label{display:block;font-size:12px;color:var(--mute);margin:8px 0 3px}
input[type=text],input[type=number]{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:5px;font:inherit;background:#fff;color:var(--ink)}
button{font:inherit;font-size:12.5px;font-weight:600;border-radius:5px;border:1px solid var(--line);background:#fff;color:var(--ink);padding:6px 9px;cursor:pointer}
button:hover{background:var(--soft)}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.danger{color:var(--bad);border-color:#e7b9b4}
button.tiny{padding:3px 6px;font-size:11.5px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:7px}
.station{display:flex;align-items:flex-start;gap:8px;border-bottom:1px dashed #dce5e8;padding:7px 0}
.station:last-child{border-bottom:0}
.station input[type=checkbox]{margin-top:2px}
.station .meta{flex:1;min-width:0}
.station .name{font-weight:650;font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.station .sub{color:var(--mute);font-size:11px;margin-top:1px}
.badge{display:inline-block;border-radius:999px;padding:1px 7px;font-size:10.5px;font-weight:650;border:1px solid transparent;white-space:nowrap}
.badge.good{background:#e7f6ee;color:var(--good);border-color:#b7e0cc}
.badge.warn{background:#fff4e2;color:var(--warn);border-color:#efd0a4}
.badge.bad{background:#fdeceb;color:var(--bad);border-color:#efb8b2}
.badge.info{background:#eaf4fb;color:var(--info);border-color:#b8d3e8}
.badge.muted{background:#eef2f4;color:var(--mute);border-color:#d7e0e4}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.metric{background:var(--soft);border-radius:5px;padding:6px 8px;min-width:0}
.metric .k{font-size:10px;color:var(--mute);text-transform:uppercase;letter-spacing:.05em}
.metric .v{font-size:13px;font-weight:650;overflow-wrap:anywhere}
.chart{width:100%;min-height:150px}
.chartnote{font-size:11px;color:var(--mute);margin-top:5px}
.tablewrap{max-height:310px;overflow:auto;border:1px solid #e0e7ea;border-radius:6px}
table{width:100%;border-collapse:collapse;font-size:11.5px}
th,td{text-align:left;padding:5px 6px;border-bottom:1px solid #e6edef;vertical-align:top}
th{position:sticky;top:0;background:#f6f9fa;color:var(--mute);font-weight:650;z-index:1}
tr:hover td{background:#fbfdfd}
tr.clickable{cursor:pointer}
.grid2{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);gap:14px}
@media(max-width:1050px){.grid2{grid-template-columns:1fr}}
#map{height:310px;border-radius:6px;border:1px solid #d6e0e4}
.empty{color:var(--mute);font-size:12px;padding:8px 2px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 9px;font-size:12px}
.kv .k{color:var(--mute)}
.kv .v{font-weight:600;overflow-wrap:anywhere}
.candidate-card{border:1px solid #e0e7ea;border-radius:6px;padding:8px;margin-bottom:7px;background:#fcfefe}
.candidate-card.sel{border-color:var(--accent);box-shadow:0 0 0 2px rgba(11,110,117,.10)}
.mini{font-size:11px;color:var(--mute)}
details summary{cursor:pointer;color:var(--accent);font-size:12px;font-weight:600}
pre{white-space:pre-wrap;word-break:break-word;background:#f7fafb;border:1px solid #e0e7ea;border-radius:5px;padding:7px;font-size:11px;max-height:260px;overflow:auto;margin:6px 0}
.footer{color:var(--mute);font-size:11px;padding:0 24px 20px}
</style>
</head>
<body>
<header>
  <div>
    <h1>Live seismic anomaly monitor</h1>
    <p>Envelope/energy detection, cross-station association and envelope-TDOA localization from live SeedLink data. Green markers on the plots are station triggers; they are not confirmed hazard events.</p>
  </div>
  <div class="statusbar" id="statusbar"></div>
</header>
<div class="notice" id="safety">Detected seismic activity is not a confirmed flood, GLOF, landslide or earthquake. Classification is provisional.</div>
<main>
  <aside class="sidebar">
    <section class="panel">
      <h2>Stations <span class="mini" id="stationCount"></span></h2>
      <div id="stationList" class="empty">Waiting for state…</div>
      <h3>Add station</h3>
      <form id="addForm">
        <div class="row3">
          <div><label for="net">Network</label><input id="net" type="text" placeholder="NP" required></div>
          <div><label for="sta">Station</label><input id="sta" type="text" placeholder="EQM10" required></div>
          <div><label for="cha">Channel</label><input id="cha" type="text" placeholder="EHZ" required></div>
        </div>
        <div class="row">
          <div><label for="lat">Latitude</label><input id="lat" type="number" step="any" placeholder="optional"></div>
          <div><label for="lon">Longitude</label><input id="lon" type="number" step="any" placeholder="optional"></div>
        </div>
        <div style="margin-top:9px;display:flex;gap:8px;align-items:center">
          <button class="primary" type="submit">Add</button>
          <span class="mini" id="addMessage"></span>
        </div>
      </form>
    </section>
    <section class="panel">
      <h2>Selected candidate <span class="mini" id="selectedId"></span></h2>
      <div id="candidateDetail" class="empty">Select a candidate from the table or wait for an association.</div>
    </section>
    <section class="panel">
      <h2>Pipeline health</h2>
      <div id="health" class="empty">Waiting for the first analysis cycle…</div>
    </section>
  </aside>
  <section class="content">
    <div class="panel">
      <h2>Live waveform <span class="mini" id="waveTitle"></span></h2>
      <div id="waveform" class="chart"></div>
      <div class="chartnote">Band-passed waveform. Vertical markers are station detections placed at their exact UTC trigger time.</div>
    </div>
    <div class="panel">
      <h2>Hilbert envelope</h2>
      <div id="envelope" class="chart"></div>
      <div class="chartnote">Smoothed envelope and event markers on the same synchronized UTC axis.</div>
    </div>
    <div class="panel">
      <h2>Spectrogram</h2>
      <div id="spectrogram" class="chart"></div>
      <div class="chartnote">Trigger markers are drawn on the spectrogram at the same trigger timestamps.</div>
    </div>
    <div class="grid2">
      <div class="panel">
        <h2>Estimated source</h2>
        <div id="map"></div>
        <div id="sourceSummary" class="chartnote">No localized candidate yet.</div>
      </div>
      <div class="panel">
        <h2>Candidate associations</h2>
        <div class="tablewrap"><table><thead><tr><th>UTC</th><th>Status</th><th>Stations</th><th>Class / duration</th><th>Quality</th></tr></thead><tbody id="candidateRows"></tbody></table></div>
      </div>
    </div>
    <div class="panel">
      <h2>Live event log</h2>
      <div class="tablewrap"><table><thead><tr><th>UTC</th><th>Kind</th><th>Message</th></tr></thead><tbody id="logRows"></tbody></table></div>
    </div>
  </section>
</main>
<div class="footer">Detected activity ≠ confirmed hazard. Use this tool for monitoring and triage, then validate with independent observations.</div>
<script>
const POLL_MS = 2000;
const PLOT_CONFIG = {responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"]};
const state = {
  data: null,
  display: new Set(),
  known: new Set(),
  selectedCandidate: null,
  map: null,
  mapLayer: null,
  polling: false,
};

const STATUS_COLORS = {
  pending: "#b96a09",
  verifying: "#b96a09",
  verified: "#1d5d8f",
  localized: "#177a55",
  underdetermined: "#b96a09",
  inconsistent: "#b3342a",
  unverified: "#5b6b74",
  expired: "#5b6b74",
  error: "#b3342a",
  default: "#b96a09",
};

function esc(value) {
  if (value === null || value === undefined) return "";
  return String(value).replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
}

function fmtUTC(iso) {
  if (!iso) return "—";
  return String(iso).replace("T", " ").replace("Z", " UTC");
}

function fmtMsUTC(ms) {
  if (!Number.isFinite(ms)) return "—";
  return new Date(ms).toISOString().replace("T", " ").replace("Z", " UTC");
}

function fmtDuration(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value)) return "—";
  if (value < 60) return `${value.toFixed(value < 10 ? 2 : 1)} s`;
  const minutes = Math.floor(value / 60);
  return `${minutes}m ${Math.round(value % 60)}s`;
}

function statusClass(status) {
  if (["localized", "verified", "supported", "high"].includes(status)) return "good";
  if (["underdetermined", "pending", "verifying", "medium", "weak", "low"].includes(status)) return "warn";
  if (["inconsistent", "error", "failed"].includes(status)) return "bad";
  if (["unverified", "expired", "insufficient"].includes(status)) return "muted";
  return "info";
}

function badge(status, text) {
  return `<span class="badge ${statusClass(status)}">${esc(text || status || "—")}</span>`;
}

function triggerColor(status) {
  return STATUS_COLORS[status] || STATUS_COLORS.default;
}

function tickData(startMs, endMs, count = 6) {
  const span = Math.max(1, endMs - startMs);
  const values = [];
  const texts = [];
  for (let i = 0; i < count; i += 1) {
    const ms = startMs + (span * i) / (count - 1);
    values.push(ms);
    texts.push(new Date(ms).toISOString().slice(11, 19));
  }
  return {values, texts};
}

function commonLayout(rows, window, hovermode) {
  const startMs = (window && window.start_ms) || Date.now() - 120000;
  const endMs = (window && window.end_ms) || Date.now();
  const ticks = tickData(startMs, endMs);
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#ffffff",
    font: {family: "Segoe UI, system-ui, sans-serif", size: 11, color: "#14232b"},
    margin: {l: 82, r: 18, t: 18, b: 42},
    showlegend: false,
    hovermode: hovermode || "x unified",
    grid: {rows: rows, columns: 1, pattern: "independent", roworder: "top to bottom", ygap: 0.18},
    xaxis: {
      type: "linear",
      range: [startMs, endMs],
      tickvals: ticks.values,
      ticktext: ticks.texts,
      tickformat: "",
      showgrid: true,
      gridcolor: "#e6edef",
      zeroline: false,
      showticklabels: rows === 1,
      tickfont: {size: 10},
    },
    yaxis: {gridcolor: "#eef3f4", zeroline: false, tickfont: {size: 10}},
  };
  for (let row = 2; row <= rows; row += 1) {
    layout[`xaxis${row}`] = {
      type: "linear",
      range: [startMs, endMs],
      matches: "x",
      tickvals: ticks.values,
      ticktext: ticks.texts,
      showgrid: true,
      gridcolor: "#e6edef",
      zeroline: false,
      showticklabels: row === rows,
      tickfont: {size: 10},
    };
    layout[`yaxis${row}`] = {gridcolor: "#eef3f4", zeroline: false, tickfont: {size: 10}};
  }
  return layout;
}

function axisSuffix(row) {
  return row === 1 ? "" : String(row);
}

function triggerLine(trigger, row, y0, y1) {
  const suffix = axisSuffix(row);
  const color = triggerColor(trigger.status);
  const text = `${trigger.label || "Trigger"}<br>${fmtMsUTC(trigger.start_ms)}<br>Candidate ${trigger.candidate_id || "—"} · ${trigger.status} · ${fmtDuration(trigger.duration_s)}`;
  return {
    x: [trigger.start_ms, trigger.start_ms],
    y: [y0, y1],
    type: "scatter",
    mode: "lines",
    line: {color, width: 1.6, dash: "dot"},
    hoverinfo: "skip",
    xaxis: `x${suffix}`,
    yaxis: `y${suffix}`,
    showlegend: false,
    _triggerText: text,
  };
}

function triggerMarker(trigger, row, y) {
  const suffix = axisSuffix(row);
  const color = triggerColor(trigger.status);
  const text = `${trigger.label || "Trigger"}<br>${fmtMsUTC(trigger.start_ms)}<br>Candidate ${trigger.candidate_id || "—"} · ${trigger.status} · ${fmtDuration(trigger.duration_s)}`;
  return {
    x: [trigger.start_ms],
    y: [y],
    type: "scatter",
    mode: "markers+text",
    marker: {symbol: "triangle-up", size: 9, color, line: {color: "#ffffff", width: 0.6}},
    text: ["TRIG"],
    textposition: "top center",
    textfont: {size: 8, color},
    hovertext: [text],
    hovertemplate: "%{hovertext}<extra></extra>",
    xaxis: `x${suffix}`,
    yaxis: `y${suffix}`,
    showlegend: false,
  };
}

function selectedKeys() {
  if (!state.data) return [];
  return Array.from(state.display).filter(key => state.data.plots && state.data.plots[key]);
}

function maxAbs(values) {
  let max = 0;
  for (const value of values || []) {
    const abs = Math.abs(Number(value));
    if (Number.isFinite(abs) && abs > max) max = abs;
  }
  return max || 1e-9;
}

function maxValue(values) {
  let max = 0;
  for (const value of values || []) {
    const number = Number(value);
    if (Number.isFinite(number) && number > max) max = number;
  }
  return max || 1e-9;
}

function renderWaveform(keys, window) {
  const target = document.getElementById("waveform");
  if (!keys.length) {
    Plotly.purge(target);
    target.innerHTML = '<div class="empty">No station selected or no contiguous waveform available yet.</div>';
    return;
  }
  const traces = [];
  keys.forEach((key, index) => {
    const plot = state.data.plots[key] || {t_ms: [], waveform: [], envelope: [], triggers: []};
    const station = state.data.stations.find(item => item.key === key) || {label: key};
    const row = index + 1;
    const suffix = axisSuffix(row);
    const limit = maxAbs(plot.waveform) * 1.08;
    traces.push({
      x: plot.t_ms,
      y: plot.waveform,
      type: "scatter",
      mode: "lines",
      line: {color: "#0b6e75", width: 0.9},
      hovertemplate: "%{x|%H:%M:%S}<br>%{y:.4g}<extra></extra>",
      xaxis: `x${suffix}`,
      yaxis: `y${suffix}`,
      showlegend: false,
    });
    (plot.triggers || []).forEach(trigger => {
      traces.push(triggerLine(trigger, row, -limit, limit));
      traces.push(triggerMarker(trigger, row, limit));
    });
    const layoutKey = row === 1 ? "yaxis" : `yaxis${row}`;
    // Added after layout creation below.
    traces[traces.length - 1]._stationTitle = `${station.label}<br><span style="font-size:9px">amplitude</span>`;
  });
  const layout = commonLayout(keys.length, window);
  keys.forEach((key, index) => {
    const station = state.data.stations.find(item => item.key === key) || {label: key};
    const axis = index === 0 ? "yaxis" : `yaxis${index + 1}`;
    layout[axis].title = {text: `${station.label}<br><span style="font-size:9px">amplitude</span>`, font: {size: 10}};
    layout[axis].showticklabels = true;
  });
  Plotly.react(target, traces, layout, PLOT_CONFIG);
}

function renderEnvelope(keys, window) {
  const target = document.getElementById("envelope");
  if (!keys.length) {
    Plotly.purge(target);
    target.innerHTML = '<div class="empty">No station selected or no envelope available yet.</div>';
    return;
  }
  const traces = [];
  keys.forEach((key, index) => {
    const plot = state.data.plots[key] || {t_ms: [], waveform: [], envelope: [], triggers: []};
    const station = state.data.stations.find(item => item.key === key) || {label: key};
    const row = index + 1;
    const suffix = axisSuffix(row);
    const limit = maxValue(plot.envelope) * 1.12;
    traces.push({
      x: plot.t_ms,
      y: plot.envelope,
      type: "scatter",
      mode: "lines",
      fill: "tozeroy",
      fillcolor: "rgba(11,110,117,0.10)",
      line: {color: "#0b6e75", width: 1.1},
      hovertemplate: "%{x|%H:%M:%S}<br>envelope %{y:.4g}<extra></extra>",
      xaxis: `x${suffix}`,
      yaxis: `y${suffix}`,
      showlegend: false,
    });
    (plot.triggers || []).forEach(trigger => {
      traces.push(triggerLine(trigger, row, 0, limit));
    });
  });
  const layout = commonLayout(keys.length, window);
  keys.forEach((key, index) => {
    const station = state.data.stations.find(item => item.key === key) || {label: key};
    const axis = index === 0 ? "yaxis" : `yaxis${index + 1}`;
    layout[axis].title = {text: `${station.label}<br><span style="font-size:9px">envelope</span>`, font: {size: 10}};
    layout[axis].showticklabels = true;
  });
  Plotly.react(target, traces, layout, PLOT_CONFIG);
}

function renderSpectrogram(keys, window) {
  const target = document.getElementById("spectrogram");
  if (!keys.length) {
    Plotly.purge(target);
    target.innerHTML = '<div class="empty">No station selected or no spectrogram available yet.</div>';
    return;
  }
  const traces = [];
  keys.forEach((key, index) => {
    const plot = state.data.plots[key] || {spectrogram: null, triggers: []};
    const station = state.data.stations.find(item => item.key === key) || {label: key};
    const row = index + 1;
    const suffix = axisSuffix(row);
    const spec = plot.spectrogram;
    if (!spec || !spec.z || !spec.z.length) {
      traces.push({
        x: [0, 1],
        y: [0, 1],
        type: "scatter",
        mode: "lines",
        line: {color: "rgba(0,0,0,0)"},
        xaxis: `x${suffix}`,
        yaxis: `y${suffix}`,
        showlegend: false,
        hoverinfo: "skip",
      });
      return;
    }
    traces.push({
      x: spec.t_ms,
      y: spec.f,
      z: spec.z,
      type: "heatmap",
      colorscale: "Viridis",
      zmin: spec.zmin,
      zmax: spec.zmax,
      showscale: index === 0,
      colorbar: {thickness: 8, len: 0.7, title: {text: "dB", side: "right"}},
      hovertemplate: "%{x}<br>%{y:.2f} Hz<br>%{z:.1f} dB<extra></extra>",
      xaxis: `x${suffix}`,
      yaxis: `y${suffix}`,
    });
    const top = maxValue(spec.f);
    (plot.triggers || []).forEach(trigger => {
      traces.push(triggerLine(trigger, row, 0, top));
    });
  });
  const layout = commonLayout(keys.length, window, "closest");
  keys.forEach((key, index) => {
    const station = state.data.stations.find(item => item.key === key) || {label: key};
    const axis = index === 0 ? "yaxis" : `yaxis${index + 1}`;
    layout[axis].title = {text: `${station.label}<br><span style="font-size:9px">Hz</span>`, font: {size: 10}};
    layout[axis].showticklabels = true;
  });
  Plotly.react(target, traces, layout, PLOT_CONFIG);
}

function renderCharts() {
  if (!state.data) return;
  const keys = selectedKeys();
  document.getElementById("waveTitle").textContent = keys.length
    ? `${keys.length} station${keys.length === 1 ? "" : "s"} · ${fmtUTC(state.data.window && state.data.window.start_utc)} → ${fmtUTC(state.data.window && state.data.window.end_utc)}`
    : "no stations selected";
  renderWaveform(keys, state.data.window);
  renderEnvelope(keys, state.data.window);
  renderSpectrogram(keys, state.data.window);
}

function renderStations() {
  const container = document.getElementById("stationList");
  if (!state.data) return;
  const stations = state.data.stations || [];
  const keys = stations.map(station => station.key);
  keys.forEach(key => {
    if (!state.known.has(key)) {
      state.display.add(key);
      state.known.add(key);
    }
  });
  for (const key of Array.from(state.known)) {
    if (!keys.includes(key)) {
      state.known.delete(key);
      state.display.delete(key);
    }
  }
  if (!stations.length) {
    container.innerHTML = '<div class="empty">No stations configured.</div>';
  } else {
    container.innerHTML = stations.map(station => {
      const connected = station.connected ? "connected" : "disconnected";
      const stale = station.stale ? " · stale" : "";
      const gaps = station.gaps ? ` · ${station.gaps} gap(s)` : "";
      const latency = station.latency_s === null || station.latency_s === undefined ? "" : ` · ${Number(station.latency_s).toFixed(1)} s latency`;
      const badgeClass = station.connected && !station.stale ? "good" : station.connected ? "warn" : "bad";
      return `<div class="station" data-key="${esc(station.key)}">
        <input type="checkbox" data-key="${esc(station.key)}" ${state.display.has(station.key) ? "checked" : ""}>
        <div class="meta">
          <div class="name">${esc(station.label)} ${badge(connected, connected)}</div>
          <div class="sub">${esc(connected)}${esc(stale)}${esc(gaps)}${esc(latency)}</div>
          ${station.last_error ? `<div class="sub" style="color:var(--bad)">${esc(station.last_error)}</div>` : ""}
        </div>
        <button class="tiny danger" data-remove="${esc(station.key)}">Remove</button>
      </div>`;
    }).join("");
  }
  document.getElementById("stationCount").textContent = `${stations.filter(item => item.connected && !item.stale).length}/${stations.length} live`;

  container.querySelectorAll('input[type="checkbox"]').forEach(input => {
    input.addEventListener("change", event => {
      const key = event.target.dataset.key;
      if (event.target.checked) state.display.add(key);
      else state.display.delete(key);
      renderCharts();
    });
  });
  container.querySelectorAll("button[data-remove]").forEach(button => {
    button.addEventListener("click", () => removeStation(button.dataset.remove));
  });
}

async function removeStation(key) {
  if (!window.confirm(`Stop monitoring ${key}?`)) return;
  try {
    const response = await fetch(`/api/stations/${encodeURIComponent(key)}`, {method: "DELETE"});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "remove failed");
    state.display.delete(key);
    state.known.delete(key);
    pollOnce();
  } catch (error) {
    window.alert(`Could not remove ${key}: ${error.message}`);
  }
}

function renderStatusBar() {
  const data = state.data;
  if (!data) return;
  const health = data.health || {};
  const connected = (data.stations || []).filter(station => station.connected && !station.stale).length;
  const total = (data.stations || []).length;
  const pending = data.candidates ? data.candidates.filter(item => ["pending", "verifying", "verified"].includes(item.status)).length : 0;
  const cycle = health.cycle_ms === null || health.cycle_ms === undefined ? "—" : `${Math.round(health.cycle_ms)} ms`;
  const lag = health.latest_data_lag_s === null || health.latest_data_lag_s === undefined ? "—" : `${Number(health.latest_data_lag_s).toFixed(1)} s`;
  const chips = [
    `<span class="chip info">UTC ${esc(fmtUTC(data.server_utc))}</span>`,
    `<span class="chip ${connected === total && total ? "good" : "warn"}">${connected}/${total} stations live</span>`,
    `<span class="chip ${pending ? "warn" : "muted"}">${pending} open candidate${pending === 1 ? "" : "s"}</span>`,
    `<span class="chip muted">cycle ${esc(cycle)}</span>`,
    `<span class="chip ${Number(health.latest_data_lag_s) > 20 ? "warn" : "muted"}">data lag ${esc(lag)}</span>`,
    `<span class="chip muted">SeedLink ${esc(data.config && data.config.server || "—")}</span>`,
  ];
  document.getElementById("statusbar").innerHTML = chips.join("");
  if (data.safety) document.getElementById("safety").textContent = data.safety;
}

function metric(label, value) {
  return `<div class="metric"><div class="k">${esc(label)}</div><div class="v">${esc(value)}</div></div>`;
}

function renderCandidateDetail() {
  const container = document.getElementById("candidateDetail");
  const idLabel = document.getElementById("selectedId");
  if (!state.data || !state.data.candidates || !state.data.candidates.length) {
    idLabel.textContent = "";
    container.innerHTML = '<div class="empty">No candidate events yet. The detector is waiting for sustained or impulsive envelope anomalies.</div>';
    return;
  }
  if (!state.selectedCandidate || !state.data.candidates.some(item => item.id === state.selectedCandidate)) {
    const preferred = state.data.candidates.find(item => ["localized", "underdetermined", "verified", "inconsistent"].includes(item.status));
    state.selectedCandidate = (preferred || state.data.candidates[0]).id;
  }
  const candidate = state.data.candidates.find(item => item.id === state.selectedCandidate);
  if (!candidate) return;
  idLabel.textContent = candidate.id;
  const classification = candidate.classification || {};
  const association = candidate.association || {};
  const localization = candidate.localization || {};
  const source = localization.source || {};
  const detectionRows = (candidate.detections || []).map(detection => `<tr><td>${esc(fmtUTC(detection.start_utc))}</td><td>${esc(detection.station_label || detection.station)}</td><td>${esc(fmtDuration(detection.significant_duration_s ?? detection.duration_s))}</td><td>${esc((detection.classification || {}).label || "-")}</td><td>${esc(Number(detection.energy_db || 0).toFixed(1))} dB</td></tr>`).join("");
  const pairRows = (association.pairs || []).map(pair => `<tr><td>${esc(pair.a_label || pair.a)}–${esc(pair.b_label || pair.b)}</td><td>${Number(pair.cc_max || 0).toFixed(2)}</td><td>${Number(pair.coherence_mean || 0).toFixed(2)}</td><td>${Number(pair.score || 0).toFixed(2)}</td><td>${pair.supported ? badge("good", "support") : badge("muted", "no")}</td></tr>`).join("");
  container.innerHTML = `
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:7px">${badge(candidate.status, candidate.status)} ${badge("info", classification.label || "unclassified")}</div>
    <div class="mini" style="margin-bottom:7px">${esc(candidate.status_message || "")}</div>
    <div class="metrics">
      ${metric("Stations", `${(candidate.station_keys || []).length}`)}
      ${metric("Duration", classification.duration_max_s !== undefined ? fmtDuration(classification.duration_max_s) : "—")}
      ${metric("Mean CC", association.mean_cc === null || association.mean_cc === undefined ? "—" : Number(association.mean_cc).toFixed(2))}
      ${metric("Coherence", association.mean_coherence === null || association.mean_coherence === undefined ? "—" : Number(association.mean_coherence).toFixed(2))}
      ${metric("Closure", association.max_closure_s === null || association.max_closure_s === undefined ? "—" : `${Number(association.max_closure_s).toFixed(2)} s`)}
      ${metric("Confidence", localization.confidence === undefined ? "—" : `${Math.round(Number(localization.confidence) * 100)}%`)}
    </div>
    <h3>Source estimate</h3>
    <div class="kv">
      <div class="k">Coordinates</div><div class="v">${source.lat !== undefined ? `${Number(source.lat).toFixed(5)}, ${Number(source.lon).toFixed(5)}` : "not available"}</div>
      <div class="k">Velocity</div><div class="v">${localization.velocity !== undefined ? `${Number(localization.velocity).toFixed(2)} km/s${localization.velocity_fixed ? " (fixed prior)" : ""}` : "—"}</div>
      <div class="k">Quality</div><div class="v">${esc(localization.quality || "—")}${localization.underdetermined ? " · underdetermined" : ""}</div>
      <div class="k">RMSE</div><div class="v">${localization.weighted_rmse_s === undefined ? "—" : `${Number(localization.weighted_rmse_s).toFixed(2)} s`}</div>
      <div class="k">Geometry</div><div class="v">${localization.geometry ? `${localization.geometry.stations} stations · spread ${Number(localization.geometry.spread_km).toFixed(1)} km · gap ${localization.geometry.azimuth_gap_deg === null ? "—" : Number(localization.geometry.azimuth_gap_deg).toFixed(0) + "°"}` : "—"}</div>
    </div>
    <div class="mini" style="margin-top:6px">${esc((classification.note || ""))}</div>
    <details style="margin-top:7px"><summary>Detections (${(candidate.detections || []).length})</summary>
      <div class="tablewrap" style="margin-top:5px"><table><thead><tr><th>UTC</th><th>Station</th><th>Duration</th><th>Class</th><th>Energy</th></tr></thead><tbody>${detectionRows}</tbody></table></div>
    </details>
    <details style="margin-top:7px"><summary>Pair checks (${(association.pairs || []).length})</summary>
      <div class="tablewrap" style="margin-top:5px"><table><thead><tr><th>Pair</th><th>CC</th><th>Coh</th><th>Score</th><th>Support</th></tr></thead><tbody>${pairRows}</tbody></table></div>
    </details>
    ${(localization.notes || []).length ? `<div class="mini" style="margin-top:7px">${localization.notes.map(esc).join("<br>")}</div>` : ""}
  `;
}

function renderCandidateRows() {
  const body = document.getElementById("candidateRows");
  if (!state.data || !state.data.candidates || !state.data.candidates.length) {
    body.innerHTML = '<tr><td colspan="5" class="empty">No candidates yet.</td></tr>';
    return;
  }
  body.innerHTML = state.data.candidates.slice(0, 25).map(candidate => {
    const classification = candidate.classification || {};
    const localization = candidate.localization || {};
    return `<tr class="clickable" data-candidate="${esc(candidate.id)}">
      <td>${esc(fmtUTC(candidate.created_utc))}</td>
      <td>${badge(candidate.status, candidate.status)}</td>
      <td>${esc((candidate.station_labels || []).join(", ") || "—")}</td>
      <td>${esc(classification.label || "—")}<div class="mini">${esc(fmtDuration(classification.duration_max_s))}</div></td>
      <td>${esc(localization.quality || "—")}<div class="mini">${localization.confidence === undefined ? "" : Math.round(Number(localization.confidence) * 100) + "%"}</div></td>
    </tr>`;
  }).join("");
  body.querySelectorAll("tr[data-candidate]").forEach(row => {
    row.addEventListener("click", () => {
      state.selectedCandidate = row.dataset.candidate;
      renderCandidateDetail();
      renderMap();
    });
  });
}

function renderLog() {
  const body = document.getElementById("logRows");
  if (!state.data || !state.data.event_log || !state.data.event_log.length) {
    body.innerHTML = '<tr><td colspan="3" class="empty">No events logged yet.</td></tr>';
    return;
  }
  body.innerHTML = state.data.event_log.slice(0, 90).map(entry => `<tr>
    <td>${esc(fmtUTC(entry.time))}</td>
    <td>${badge(entry.kind === "localization" ? "localized" : entry.kind === "error" ? "error" : entry.kind === "association" ? "verified" : "info", entry.kind)}</td>
    <td>${esc(entry.message)}</td>
  </tr>`).join("");
}

function renderHealth() {
  const container = document.getElementById("health");
  const health = state.data && state.data.health;
  if (!health) {
    container.innerHTML = '<div class="empty">Waiting for the first analysis cycle…</div>';
    return;
  }
  const warnings = health.warnings || [];
  container.innerHTML = `
    <div class="kv">
      <div class="k">Last cycle</div><div class="v">${esc(fmtUTC(health.last_cycle_utc))}</div>
      <div class="k">Cycle time</div><div class="v">${health.cycle_ms === null || health.cycle_ms === undefined ? "—" : Math.round(health.cycle_ms) + " ms"}</div>
      <div class="k">Association</div><div class="v">${health.association_ms === null || health.association_ms === undefined ? "—" : Math.round(health.association_ms) + " ms"}</div>
      <div class="k">Data lag</div><div class="v">${health.latest_data_lag_s === null || health.latest_data_lag_s === undefined ? "—" : Number(health.latest_data_lag_s).toFixed(1) + " s"}</div>
      <div class="k">Open candidates</div><div class="v">${health.pending_candidates === undefined ? "—" : health.pending_candidates}</div>
      <div class="k">Stations</div><div class="v">${health.connected_stations === undefined ? "—" : health.connected_stations}/${health.station_count === undefined ? "—" : health.station_count} connected</div>
    </div>
    ${warnings.length ? `<div class="mini" style="color:var(--warn);margin-top:7px">${warnings.map(esc).join("<br>")}</div>` : ""}
  `;
}

function initMap() {
  if (state.map) return;
  if (!window.L) return;
  const element = document.getElementById("map");
  state.map = L.map(element, {zoomControl: true, attributionControl: true}).setView([28.0, 85.0], 7);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(state.map);
  state.mapLayer = L.layerGroup().addTo(state.map);
}

function renderMap() {
  if (!state.data) return;
  initMap();
  if (!state.map) return;
  state.mapLayer.clearLayers();
  const stations = state.data.stations || [];
  for (const station of stations) {
    if (station.lat === null || station.lon === null || station.lat === undefined || station.lon === undefined) continue;
    const color = station.connected && !station.stale ? "#0b6e75" : "#b3342a";
    L.circleMarker([station.lat, station.lon], {radius: 6, color, weight: 2, fillColor: color, fillOpacity: 0.75})
      .bindTooltip(`${station.label}<br>${station.connected ? "connected" : "disconnected"}`, {direction: "top"})
      .addTo(state.mapLayer);
  }

  const candidate = (state.data.candidates || []).find(item => item.id === state.selectedCandidate);
  const localization = candidate && candidate.localization;
  const source = localization && localization.source;
  if (source && Number.isFinite(Number(source.lat)) && Number.isFinite(Number(source.lon))) {
    const color = localization.status === "localized" ? "#177a55" : "#b96a09";
    L.marker([source.lat, source.lon], {
      icon: L.divIcon({
        className: "",
        html: `<div style="font-size:26px;line-height:26px;color:${color};text-shadow:0 0 3px #fff">★</div>`,
        iconSize: [26, 26],
        iconAnchor: [13, 13],
      }),
    }).bindTooltip(`Candidate ${candidate.id}<br>${localization.status} · ${localization.quality}<br>${Number(source.lat).toFixed(5)}, ${Number(source.lon).toFixed(5)}`).addTo(state.mapLayer);
    for (const detection of candidate.detections || []) {
      const station = stations.find(item => item.key === detection.station);
      if (!station) continue;
      L.polyline([[source.lat, source.lon], [station.lat, station.lon]], {color, weight: 1.2, dashArray: "4 5", opacity: 0.65}).addTo(state.mapLayer);
    }
    state.map.setView([source.lat, source.lon], Math.max(state.map.getZoom(), 7));
  }
  const summary = document.getElementById("sourceSummary");
  if (source) {
    summary.innerHTML = `Candidate ${esc(candidate.id)} · ${badge(localization.status, localization.status)} · ${esc(localization.quality || "")} · confidence ${Math.round(Number(localization.confidence || 0) * 100)}% · velocity ${localization.velocity === undefined ? "—" : Number(localization.velocity).toFixed(2) + " km/s"}${localization.underdetermined ? " · underdetermined" : ""}`;
  } else {
    summary.textContent = "No localized candidate yet.";
  }
  setTimeout(() => state.map.invalidateSize(), 20);
}

function applyState(data) {
  state.data = data;
  renderStatusBar();
  renderStations();
  renderCharts();
  renderCandidateDetail();
  renderCandidateRows();
  renderLog();
  renderHealth();
  renderMap();
}

async function pollOnce() {
  if (state.polling) return;
  state.polling = true;
  try {
    const response = await fetch("/api/state", {cache: "no-store"});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    applyState(payload);
  } catch (error) {
    document.getElementById("statusbar").innerHTML = `<span class="chip bad">state refresh failed: ${esc(error.message)}</span>`;
  } finally {
    state.polling = false;
  }
}

function schedulePoll() {
  pollOnce();
  setTimeout(schedulePoll, POLL_MS);
}

document.getElementById("addForm").addEventListener("submit", async event => {
  event.preventDefault();
  const message = document.getElementById("addMessage");
  const payload = {
    net: document.getElementById("net").value.trim(),
    sta: document.getElementById("sta").value.trim(),
    cha: document.getElementById("cha").value.trim(),
  };
  const lat = document.getElementById("lat").value.trim();
  const lon = document.getElementById("lon").value.trim();
  if (lat && lon) {
    payload.lat = Number(lat);
    payload.lon = Number(lon);
  }
  message.textContent = "Adding…";
  try {
    const response = await fetch("/api/stations", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "add failed");
    message.textContent = result.message || "Added.";
    document.getElementById("sta").value = "";
    document.getElementById("lat").value = "";
    document.getElementById("lon").value = "";
    pollOnce();
  } catch (error) {
    message.textContent = error.message;
    message.style.color = "var(--bad)";
  }
});

window.addEventListener("resize", () => {
  for (const id of ["waveform", "envelope", "spectrogram"]) {
    const element = document.getElementById(id);
    if (element && element.data) Plotly.Plots.resize(element);
  }
  if (state.map) state.map.invalidateSize();
});

schedulePoll();


</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
