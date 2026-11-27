from __future__ import annotations
import io
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, Response, jsonify, request
from matplotlib.ticker import FuncFormatter
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from association import (AssociationParams, auto_max_sp_sec, build_candidates, geometry_precheck, pair_ps, prepare_picks)
from config import (DETECTION_SETTINGS as D, FDSN_TIMEOUT, FDSN_URL, HISTORICAL_SETTINGS as H, REALTIME_SETTINGS as R, STATIONS, WEB_HOST)
from detection import process_detection
from geometry import StationGeometry
from location import grid_search_location, locate_event_from_picks
from magnitude import calculate_epicentral_distance, calculate_ml
from pickers import pick_arrival
from preprocess import prepare_components, prepare_stream
from travel_times import TravelTimeTable
from trigger_quality import evaluate_pick_quality


FDSN_WEB_PORT = int(os.environ.get("FDSN_WEB_PORT", "8003"))

EVENT_DIR = Path("fdsn_events")
FDSN_WORKERS = 4
GRID_PADDING_KM = 200.0
P_REFINE_SEARCHES = (12.0, 6.0)
S_PICK_SEARCH_SEC = 10.0
P_OUTLIER_SEC = 3.0
EVENT_COINCIDENCE_CAP_SEC = 180.0
EVENT_SUPPRESS_MIN_SEC = 30.0
EVENT_SUPPRESS_MAX_SEC = 90.0
MAX_TRIGGERS_PER_TRACE = 200

ST_COLORS = [
    "#0F766E", "#2563EB", "#7C3AED", "#92400E",
    "#0891B2", "#4D7C0F", "#475569", "#C026D3",
]

JOBS = {}
JOBS_LOCK = threading.RLock()

_T_TABLE = None
_T_TABLE_LOCK = threading.Lock()


def fmt_time(value):
    return UTCDateTime(value).strftime("%Y-%m-%d %H:%M:%S") + " UTC"


def station_id(station):
    return f"{station['net']}.{station['sta']}"


def travel_table():
    global _T_TABLE
    with _T_TABLE_LOCK:
        if _T_TABLE is None:
            print("Building travel-time table")
            _T_TABLE = TravelTimeTable(
                H["travel_depths"], H["travel_max_km"], H["travel_step_km"]
            )
        return _T_TABLE


def is_cancelled(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return bool(job and job.get("cancel"))


def job_update(job_id, **fields):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is not None:
            job.update(fields)


def public_event(event):
    item = {key: value for key, value in event.items() if key != "waveform_png"}
    item["has_waveform"] = bool(event.get("waveform_png"))
    return item


# FDSN access
def make_client():
    return Client(FDSN_URL, timeout=FDSN_TIMEOUT)


def fetch_waveforms(station, start, end, client=None):
    """Fetch one station's data, trying the configured channel first."""
    client = client or make_client()
    loc = str(station.get("loc") or "*")
    cha = str(station.get("cha") or "*")

    attempts = [(loc, cha)]
    for fallback in (("*", "?H?"), ("*", "*")):
        if fallback not in attempts:
            attempts.append(fallback)

    last_error = None
    for location, channel in attempts:
        try:
            stream = client.get_waveforms(
                station["net"], station["sta"], location, channel, start, end
            )
        except Exception as error:  # FDSNNoData, HTTP errors, timeouts, ...
            last_error = error
            continue
        if len(stream) > 0:
            return stream
        last_error = ValueError("No data returned")

    raise ValueError(
        f"No FDSN data for {station_id(station)} between "
        f"{fmt_time(start)} and {fmt_time(end)}: {last_error}"
    )


def fetch_components(station, start, end):
    stream = fetch_waveforms(station, start, end)
    processed = prepare_stream(
        stream, D["fs"], D["fmin"], D["fmax"], D["max_gap_sec"]
    )
    return prepare_components(processed)


# waveform helpers
def newest_trace(stream):
    traces = [trace for trace in stream if trace.stats.npts > 0]
    return max(traces, key=lambda trace: trace.stats.endtime) if traces else None


def trace_covering(stream, target, duration=0.0):
    """Best trace around ``target`` (a trace covering it is preferred)."""
    if stream is None or len(stream) == 0:
        return None

    traces = [trace for trace in stream if trace.stats.npts > 0]
    if not traces:
        return None

    target = UTCDateTime(target)
    wanted_end = target + max(0.0, float(duration))
    covering = [
        trace for trace in traces
        if trace.stats.starttime <= target
        and trace.stats.endtime >= min(wanted_end, target + 1e-6)
    ]
    pool = covering or traces
    return max(pool, key=lambda trace: trace.stats.npts)


def pick_quality(pick):
    snr = pick.snr if pick.snr is not None else float("-inf")
    contrast = pick.aic_contrast if pick.aic_contrast is not None else float("-inf")
    return snr, contrast


def pick_from_trace(trace, sid, phase, after=None, until=None, near=None, search=None):
    if trace is None or trace.stats.npts == 0:
        return [], []

    try:
        triggers = process_detection(trace.data, trace.stats.starttime, sid, phase)
    except ValueError:
        return [], []

    selected = []
    for trigger in triggers:
        stamp = trigger.detection_time
        if after is not None and stamp < after:
            continue
        if until is not None and stamp >= until:
            continue
        if near is not None and abs(stamp - near) > search:
            continue
        selected.append(trigger)

    selected = selected[:MAX_TRIGGERS_PER_TRACE]
    picks = []
    fs = float(trace.stats.sampling_rate)

    for trigger in selected:
        try:
            pick = pick_arrival(trace, trigger, phase)
            idx = int(round((pick.arrival_time - trace.stats.starttime) * fs))
            quality = evaluate_pick_quality(
                trace.data,
                fs,
                idx,
                snr_threshold=D["snr_threshold"],
                signal_window_sec=D["snr_signal_sec"],
                search_window_sec=D["snr_search_sec"],
                noise_window_sec=D["snr_noise_sec"],
                noise_gap_sec=D["snr_gap_sec"],
                aic_boundary=pick.aic_boundary,
                aic_contrast=pick.aic_contrast,
                aic_min_contrast=D["aic_min_contrast"],
            )
        except ValueError:
            continue

        pick.snr = quality["snr"]
        pick.accepted = quality["accepted"]
        pick.rejection_reason = "; ".join(quality["notes"]) or None
        picks.append(pick)

    return selected, picks


def pick_near(trace, sid, phase, target, search):
    """Pick the accepted onset nearest a predicted arrival."""
    if trace is None:
        return None
    _triggers, picks = pick_from_trace(
        trace, sid, phase, near=target, search=search
    )
    accepted = [pick for pick in picks if pick.accepted]
    if not accepted:
        return None
    return min(accepted, key=lambda pick: abs(pick.arrival_time - target))


def dedupe_trigger_entries(entries, gap=5.0):
    ordered = sorted(entries, key=lambda item: item["t"])
    result = []
    for entry in ordered:
        if result and abs(entry["t"] - result[-1]["t"]) < gap:
            if entry["cft"] > result[-1]["cft"]:
                result[-1] = entry
        else:
            result.append(entry)
    return result


# station scan
def detect_station_range(station, start, end, cancel=None):
    """Scan one station in padded chunks and return P picks and triggers."""
    sid = station_id(station)
    client = make_client()
    fs = float(D["fs"])
    pad = max(
        float(H["detect_pad"]),
        float(D["lta_p"]) + float(D["sta_p"]) + 2.0,
    )
    chunk_sec = float(H["chunk_sec"])

    triggers = []
    picks = []
    ok_chunks = 0
    failed_chunks = 0
    last_error = None
    cancelled = False

    position = UTCDateTime(start)
    end = UTCDateTime(end)

    while position < end:
        if cancel is not None and cancel():
            cancelled = True
            break

        chunk_end = min(position + chunk_sec, end)
        wave_start = position - pad
        wave_end = chunk_end + pad

        try:
            stream = fetch_waveforms(station, wave_start, wave_end, client=client)
            processed = prepare_stream(
                stream, fs, D["fmin"], D["fmax"], D["max_gap_sec"]
            )
            components = prepare_components(processed)
            vertical = components["Z"]

            if len(vertical) == 0:
                raise ValueError("No vertical component in chunk")

            chunk_triggers = []
            chunk_picks = []

            for trace in vertical:
                if trace.stats.npts < int((D["lta_p"] + D["sta_p"]) * fs):
                    continue
                found_triggers, found_picks = pick_from_trace(trace, sid, "P", after=position, until=chunk_end)
                chunk_triggers.extend(found_triggers)
                chunk_picks.extend(found_picks)

            accepted_times = {
                round(float(pick.trigger_time.timestamp), 3)
                for pick in chunk_picks if pick.accepted
            }

            for trigger in chunk_triggers:
                stamp = float(trigger.detection_time.timestamp)
                triggers.append({
                    "station": sid,
                    "t": stamp,
                    "time": fmt_time(trigger.detection_time),
                    "clock": trigger.detection_time.strftime("%H:%M:%S"),
                    "cft": round(float(trigger.cft_value), 2),
                    "duration": round(float(trigger.duration_sec), 2),
                    "amplitude": round(float(trigger.amplitude), 2),
                    "accepted": round(stamp, 3) in accepted_times,
                })

            picks.extend(chunk_picks)
            ok_chunks += 1
        except Exception as error:
            failed_chunks += 1
            last_error = str(error)

        position = chunk_end

    accepted = [pick for pick in picks if pick.accepted]
    accepted = prepare_picks(
        accepted, max(float(D["dead_time_sec"]), 5.0)
    )

    if cancelled:
        state = "cancelled"
    elif ok_chunks:
        state = "ok"
    else:
        state = "error"

    return {
        "picks": accepted,
        "triggers": dedupe_trigger_entries(triggers, 5.0),
        "stats": {
            "station": sid,
            "state": state,
            "triggers": len(triggers),
            "picks": len(accepted),
            "chunks": ok_chunks,
            "failed_chunks": failed_chunks,
            "error": last_error,
        },
    }


# grouping
def coincidence_window(geometry):
    if not geometry.moveout_limits:
        return 30.0
    moveout = max(geometry.moveout_limits.values())
    return float(min(EVENT_COINCIDENCE_CAP_SEC, max(20.0, moveout)))


def event_suppress_seconds(params):
    return float(
        max(
            EVENT_SUPPRESS_MIN_SEC,
            min(EVENT_SUPPRESS_MAX_SEC, params.max_sp_sec),
        )
    )


def group_picks(picks, geometry, min_stations, coincidence_sec, suppress_sec):
    """Group P picks into candidate event windows, chronologically."""
    ordered = sorted(picks, key=lambda pick: pick.arrival_time)
    groups = []
    used_until = {}
    index = 0

    while index < len(ordered):
        anchor = ordered[index]
        anchor_ts = float(anchor.arrival_time.timestamp)

        if anchor_ts < used_until.get(anchor.station_id, -1e18):
            index += 1
            continue

        window = []
        cursor = index
        while (
            cursor < len(ordered)
            and float(ordered[cursor].arrival_time.timestamp) - anchor_ts
            <= coincidence_sec
        ):
            pick = ordered[cursor]
            stamp = float(pick.arrival_time.timestamp)
            if stamp >= used_until.get(pick.station_id, -1e18):
                window.append(pick)
            cursor += 1

        by_station = {}
        for pick in window:
            best = by_station.get(pick.station_id)
            if best is None or pick_quality(pick) > pick_quality(best):
                by_station[pick.station_id] = pick

        candidate = None
        if len(by_station) >= int(min_stations):
            candidates = build_candidates(
                list(by_station.values()), geometry, int(min_stations)
            )
            if candidates:
                candidate = max(
                    candidates,
                    key=lambda group: (
                        len(group),
                        -min(float(pick.arrival_time.timestamp) for pick in group),
                    ),
                )

        if candidate is None:
            index += 1
            continue

        groups.append(candidate)
        last_ts = max(float(pick.arrival_time.timestamp) for pick in candidate)
        for pick in candidate:
            used_until[pick.station_id] = last_ts + suppress_sec

        while (
            index < len(ordered)
            and float(ordered[index].arrival_time.timestamp) <= last_ts
        ):
            index += 1

    return groups


# location and magnitude
def padded_box(lats, lons, pad_km):
    dlat = pad_km / 110.574
    dlon = pad_km / (111.32 * np.cos(np.radians(np.mean(lats))))
    return (
        (max(-90.0, min(lats) - dlat), min(90.0, max(lats) + dlat)),
        (max(-180.0, min(lons) - dlon), min(180.0, max(lons) + dlon)),
    )


def search_box(geometry, pad_km=GRID_PADDING_KM):
    lats = [station["latitude"] for station in geometry.stations.values()]
    lons = [station["longitude"] for station in geometry.stations.values()]
    return padded_box(lats, lons, pad_km)


def predict_station(station, location, table):
    station_lat = station.get("latitude", station.get("lat"))
    station_lon = station.get("longitude", station.get("lon"))
    if station_lat is None or station_lon is None:
        raise ValueError("Station has no coordinates")

    distance = calculate_epicentral_distance(
        station_lat,
        station_lon,
        location["latitude"],
        location["longitude"],
    )
    depth = float(location["depth_km"])
    p_time = location["origin_time"] + table.get_time(distance, depth, "P")
    s_time = location["origin_time"] + table.get_time(distance, depth, "S")
    return distance, p_time, s_time


def locate_p_only(p_picks, geometry, params, table, lat_range, lon_range, grid_km, depths=None):
    """Locate from P picks, allowing a three-station solution."""
    p_picks = list(p_picks)
    depths = tuple(depths or params.travel_depths)

    if len(p_picks) >= 4:
        return locate_event_from_picks(
            p_picks,
            geometry.stations,
            table,
            depths,
            lat_range,
            lon_range,
            grid_km,
        )

    if len(p_picks) < 3:
        raise ValueError("At least three P picks are required for a location")

    best = None
    for depth in depths:
        try:
            result = grid_search_location(
                p_picks,
                geometry.stations,
                table,
                lat_range,
                lon_range,
                float(depth),
                grid_km,
            )
        except ValueError:
            continue
        if best is None or result["rms_sec"] < best["rms_sec"]:
            best = result

    if best is None:
        raise ValueError("Grid search could not evaluate any depth")

    best["n_observations"] = len(best["picks"])
    best["n_p"] = len(best["picks"])
    best["n_s"] = 0
    return best


def locate_with_p_filter(p_picks, geometry, params, table, lat_range, lon_range, grid_km):
    """Locate while dropping the worst P pick until residuals are sane."""
    picks = list(p_picks)
    if not picks:
        raise ValueError("No P picks")

    best = None
    while True:
        try:
            result = locate_p_only(
                picks, geometry, params, table,
                lat_range, lon_range, grid_km,
            )
        except ValueError:
            if best is not None:
                break
            raise

        best = result
        details = [detail for detail in result["picks"] if detail["phase"] == "P"]
        if not details:
            break

        worst = max(details, key=lambda detail: abs(detail["residual_sec"]))
        if abs(worst["residual_sec"]) <= P_OUTLIER_SEC or len(details) <= 3:
            break

        picks = [
            pick for pick in picks
            if not (
                pick.station_id == worst["station_id"]
                and pick.phase.upper() == "P"
            )
        ]
        if len(picks) < 3:
            break

    return best, picks


def final_locate(picks, geometry, params, table, lat_range, lon_range, grid_km, depths=None):
    depths = tuple(depths or params.travel_depths)
    if len(picks) >= 4:
        return locate_event_from_picks(
            picks,
            geometry.stations,
            table,
            depths,
            lat_range,
            lon_range,
            grid_km,
        )

    p_picks = [pick for pick in picks if pick.phase.upper() == "P"]
    return locate_p_only(
        p_picks, geometry, params, table,
        lat_range, lon_range, grid_km, depths,
    )


def refine_p_picks(stations, location, components, table, search):
    """Re-pick P on each station near the predicted arrival."""
    refined = {}
    for station in stations:
        sid = station_id(station)
        component = components.get(sid)
        if component is None:
            continue

        try:
            _distance, predicted_p, _predicted_s = predict_station(
                station, location, table
            )
        except ValueError:
            continue

        trace = trace_covering(component.get("Z"), predicted_p, 2.0)
        pick = pick_near(trace, sid, "P", predicted_p, search)
        if pick is not None:
            refined[sid] = pick

    return refined


def estimate_period(data, fs):
    data = data - np.mean(data)
    crossings = np.count_nonzero(np.diff(np.signbit(data)))
    if crossings < 2:
        return 1.0 / np.sqrt(D["fmin"] * D["fmax"])
    period = 2.0 * len(data) / fs / crossings
    return float(np.clip(period, 1.0 / D["fmax"], 1.0 / D["fmin"]))


def station_ml(traces, onset, delta_km):
    """Approximate station ML from the 15 s after the S (or P) arrival."""
    amp_trace = traces.get("H") or traces.get("Z")
    per_trace = traces.get("N") or traces.get("Z")
    if amp_trace is None or per_trace is None:
        return None

    amp = amp_trace.slice(onset, onset + 15).data
    per = per_trace.slice(onset, onset + 15).data
    if len(amp) < 20 or len(per) < 20:
        return None

    fs = float(amp_trace.stats.sampling_rate)
    period = estimate_period(per, fs)
    peak_velocity = np.max(np.abs(amp)) / R["counts_per_m"]
    displacement_nm = peak_velocity * period / (2 * np.pi) * 1e9

    try:
        return calculate_ml(displacement_nm, period, delta_km)
    except ValueError:
        return None


def component_traces(component, onset):
    return {
        "Z": trace_covering(component.get("Z"), onset, 15.0),
        "H": trace_covering(component.get("H"), onset, 15.0),
        "N": trace_covering(component.get("N"), onset, 15.0),
    }


# ---------------------------------------------------------------------------
# event plot
# ---------------------------------------------------------------------------
def make_event_waveform_png(components, rows, start, end):
    names = [row["name"] for row in rows if row["name"] in components]
    if not names:
        return None

    by_name = {row["name"]: row for row in rows}
    fig, axes = plt.subplots(
        len(names), 1, sharex=True,
        figsize=(8, max(3.2, 2.0 * len(names) + 0.8)),
    )
    axes = np.atleast_1d(axes)

    for index, (ax, sid) in enumerate(zip(axes, names)):
        color = ST_COLORS[index % len(ST_COLORS)]
        row = by_name[sid]
        ax.set_ylabel("Normalized\namplitude", fontsize=8.5)
        ax.text(
            0.006, 0.92, sid, transform=ax.transAxes, va="top",
            fontsize=10, fontweight="bold", color=color,
        )

        trace = trace_covering(components[sid].get("Z"), start, float(end - start))
        if trace is None:
            ax.set_yticks([])
            ax.text(
                0.5, 0.5, "no vertical data", transform=ax.transAxes,
                ha="center", va="center", color="#8a96a3",
            )
            continue

        trace = trace.copy().trim(start, end, pad=True, fill_value=0)
        data = trace.data.astype(float)
        scale = (
            float(np.percentile(np.abs(data), 99))
            or float(np.max(np.abs(data)))
            or 1.0
        )
        x = np.arange(len(data)) / float(trace.stats.sampling_rate) + (
            trace.stats.starttime - end
        )
        ax.plot(x, data / scale, color=color, linewidth=0.9)
        ax.set_ylim(-1.05, 1.05)

        for key, line_color, style in (
            ("p_time", "#D6361F", "--"),
            ("s_time", "#1F7A55", "-"),
        ):
            value = row.get(key)
            if not value:
                continue
            arrival = UTCDateTime(value)
            if start <= arrival <= end:
                ax.axvline(
                    float(arrival - end), color=line_color,
                    linestyle=style, linewidth=1.1,
                )

    for ax in axes:
        ax.set_xlim(float(start - end), 0.0)

    axes[-1].xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: (end + float(value)).strftime("%H:%M:%S"))
    )
    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle(
        "Event waveforms, each trace normalized (red = P pick, green = S pick)",
        fontsize=10.5, fontweight="bold",
    )
    fig.subplots_adjust(left=0.135, right=0.995, top=0.94, bottom=0.075, hspace=0.28)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


# candidate event
def failed_event(group, start, end, error, index):
    first = min((pick.arrival_time for pick in group), default=start)
    return {
        "id": f"e{index}",
        "located": False,
        "error": str(error),
        "time": fmt_time(first),
        "origin_time": None,
        "lat": None,
        "lon": None,
        "depth_km": None,
        "magnitude": None,
        "rms_sec": None,
        "n_stations": 0,
        "n_p": 0,
        "n_s": 0,
        "stations": [],
        "trigger_stations": sorted({pick.station_id for pick in group}),
        "window_start": fmt_time(start),
        "window_end": fmt_time(end),
        "method": None,
        "waveform_png": None,
        "has_waveform": False,
    }


def process_candidate(job_id, group, index, geometry, params, table, start_limit, end_limit, stations):
    first = min(pick.arrival_time for pick in group)
    last = max(pick.arrival_time for pick in group)
    start = max(UTCDateTime(start_limit), first - float(H["pre"]))
    end = min(UTCDateTime(end_limit), last + float(params.max_sp_sec) + float(H["post"]),)
    if end <= start:
        end = start + 120

    # Fetch a focused window for every selected station.
    components = {}
    errors = {}
    workers = max(1, min(FDSN_WORKERS, len(stations)))

    def fetch_one(station):
        return station_id(station), fetch_components(station, start, end)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, station): station for station in stations}
        for future in as_completed(futures):
            station = futures[future]
            sid = station_id(station)
            try:
                sid, component = future.result()
                components[sid] = component
            except Exception as error:
                errors[sid] = str(error)

    if len(components) < 3:
        return failed_event(group, start, end, "Fewer than three stations returned waveform data", index,)

    initial_picks = prepare_picks([pick for pick in group if pick.station_id in components], params.dedup_sec,)
    if len(initial_picks) < 3:
        return failed_event(group, start, end, "Fewer than three triggered stations returned data", index,)

    lat_range, lon_range = search_box(geometry)

    try:
        location, p_used = locate_with_p_filter(
            initial_picks, geometry, params, table,
            lat_range, lon_range, params.grid_coarse_km,
        )
    except ValueError as error:
        return failed_event(group, start, end, f"Location failed: {error}", index)

    # Re-pick P near the prediction, then re-locate.
    usable_stations = [station for station in stations if station_id(station) in components]
    for search in P_REFINE_SEARCHES:
        refined = refine_p_picks(usable_stations, location, components, table, search)
        if len(refined) < 3:
            break

        merged = {pick.station_id: pick for pick in p_used}
        merged.update(refined)

        try:
            trial, trial_used = locate_with_p_filter(
                list(merged.values()), geometry, params, table,
                lat_range, lon_range, params.grid_coarse_km,
            )
        except ValueError:
            break

        if trial["rms_sec"] <= location["rms_sec"] + 0.25:
            location, p_used = trial, trial_used
        else:
            break

    p_by_station = {pick.station_id: pick for pick in p_used}
    candidates = build_candidates(
        list(p_by_station.values()), geometry, params.min_stations
    )
    original_stations = {pick.station_id for pick in group}

    if candidates:
        candidate = max(
            candidates,
            key=lambda items: (
                len({pick.station_id for pick in items} & original_stations),
                len(items),
            ),
        )
    else:
        candidate = list(p_by_station.values())

    if len({pick.station_id for pick in candidate}) < params.min_stations:
        return failed_event(
            group, start, end,
            "Not enough mutually consistent P picks",
            index,
        )

    # Pick S on the horizontal envelope (or Z when no horizontals exist).
    s_picks = []
    candidate_by_station = {pick.station_id: pick for pick in candidate}

    for sid, p_pick in candidate_by_station.items():
        station = geometry.stations.get(sid)
        component = components.get(sid)
        if station is None or component is None:
            continue

        try:
            _distance, _predicted_p, predicted_s = predict_station(
                station, location, table
            )
        except ValueError:
            continue

        stream = component.get("H")
        if stream is None or len(stream) == 0:
            stream = component.get("Z")
        trace = trace_covering(stream, predicted_s, 2.0)
        pick = pick_near(trace, sid, "S", predicted_s, S_PICK_SEARCH_SEC)
        if pick is None:
            continue

        sp_time = float(pick.arrival_time - p_pick.arrival_time)
        if params.min_sp_sec <= sp_time <= params.max_sp_sec:
            s_picks.append(pick)

    all_picks = prepare_picks(candidate + s_picks, params.dedup_sec)
    pairs, _orphan_s = pair_ps(
        all_picks, params.min_sp_sec, params.max_sp_sec
    )
    check = geometry_precheck(
        candidate, pairs, geometry, params.vp, params.vs
    )
    suspicious = set(check["suspicious_stations"])

    used = list(candidate)
    for p_pick, s_pick in pairs:
        if p_pick in candidate and s_pick.station_id not in suspicious:
            used.append(s_pick)
    used = prepare_picks(used, params.dedup_sec)

    try:
        result = final_locate(
            used, geometry, params, table,
            lat_range, lon_range, params.grid_coarse_km,
        )
    except ValueError as error:
        return failed_event(group, start, end, f"Location failed: {error}", index)

    if result["rms_sec"] > params.max_rms_sec:
        return failed_event(
            group, start, end,
            f"RMS {result['rms_sec']:.2f} s is above the limit "
            f"({params.max_rms_sec} s)",
            index,
        )

    # Fine grid around the coarse solution.
    fine_lat, fine_lon = padded_box(
        [result["latitude"]], [result["longitude"]],
        2.0 * params.grid_coarse_km,
    )
    try:
        fine = final_locate(
            used, geometry, params, table,
            fine_lat, fine_lon, params.grid_fine_km,
            depths=[result["depth_km"]],
        )
        if fine["rms_sec"] <= result["rms_sec"]:
            result = fine
    except ValueError:
        pass

    # Station rows and magnitude.
    by_station = {}
    for detail in result["picks"]:
        by_station.setdefault(detail["station_id"], {})[detail["phase"]] = detail

    snr = {(pick.station_id, pick.phase): pick.snr for pick in used}
    rows = []
    magnitudes = []

    for sid, phases in by_station.items():
        station = geometry.stations[sid]
        onset = (phases.get("S") or phases["P"])["observed_arrival"]
        distance = calculate_epicentral_distance(
            station["latitude"],
            station["longitude"],
            result["latitude"],
            result["longitude"],
        )

        component = components.get(sid)
        ml = None
        if component is not None:
            ml = station_ml(component_traces(component, onset), onset, distance)
        if ml is not None:
            magnitudes.append(ml)

        p_detail = phases.get("P")
        s_detail = phases.get("S")
        p_snr = snr.get((sid, "P"))
        s_snr = snr.get((sid, "S"))
        quality_snr = p_snr if p_snr is not None else s_snr

        rows.append({
            "name": sid,
            "quality": "-" if quality_snr is None else f"SNR {quality_snr:.1f}",
            "p_time": str(p_detail["observed_arrival"]) if p_detail else None,
            "s_time": str(s_detail["observed_arrival"]) if s_detail else None,
            "s_picked": s_detail is not None,
            "ml": None if ml is None else round(float(ml), 1),
            "distance_km": round(float(distance), 1),
            "p_residual_sec": (
                None if p_detail is None else round(float(p_detail["residual_sec"]), 2)
            ),
            "s_residual_sec": (
                None if s_detail is None else round(float(s_detail["residual_sec"]), 2)
            ),
        })

    rows.sort(key=lambda row: row["name"])
    event = {
        "id": f"e{index}",
        "located": True,
        "error": None,
        "detected": fmt_time(UTCDateTime()),
        "time": fmt_time(result["origin_time"]),
        "origin_time": fmt_time(result["origin_time"]),
        "lat": round(float(result["latitude"]), 4),
        "lon": round(float(result["longitude"]), 4),
        "depth_km": round(float(result["depth_km"]), 1),
        "magnitude": (
            round(float(np.median(magnitudes)), 1) if magnitudes else None
        ),
        "rms_sec": round(float(result["rms_sec"]), 3),
        "n_stations": len(by_station),
        "n_p": int(result.get("n_p", 0)),
        "n_s": int(result.get("n_s", 0)),
        "stations": rows,
        "trigger_stations": sorted({pick.station_id for pick in group}),
        "window_start": fmt_time(start),
        "window_end": fmt_time(end),
        "method": "P + S" if result.get("n_s", 0) else "P",
        "data_errors": errors or None,
        "waveform_png": None,
        "has_waveform": False,
    }

    try:
        event["waveform_png"] = make_event_waveform_png(
            components, rows, start, end
        )
        event["has_waveform"] = bool(event["waveform_png"])
    except Exception as error:
        event["waveform_png"] = None
        event["has_waveform"] = False
        errors.setdefault("plot", str(error))

    event["data_errors"] = errors or None
    return event


# ---------------------------------------------------------------------------
# job storage / API helpers
# ---------------------------------------------------------------------------
def save_event(job_id, event):
    try:
        EVENT_DIR.mkdir(parents=True, exist_ok=True)
        payload = public_event(event)
        payload["job_id"] = job_id
        path = EVENT_DIR / f"event_{job_id}_{event['id']}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as error:
        print(f"[save] event {event.get('id')}: {error}")


def save_job(job_id):
    try:
        payload = public_job(JOBS[job_id])
    except Exception:
        return
    try:
        EVENT_DIR.mkdir(parents=True, exist_ok=True)
        path = EVENT_DIR / f"job_{job_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as error:
        print(f"[save] job {job_id}: {error}")


def public_station_rows(job):
    stats = {item["station"]: item for item in job.get("station_stats", [])}
    rows = []
    for station in job["stations"]:
        sid = station_id(station)
        item = stats.get(sid, {})
        rows.append({
            "name": sid,
            "net": station["net"],
            "sta": station["sta"],
            "lat": station.get("lat"),
            "lon": station.get("lon"),
            "state": item.get("state", "pending"),
            "triggers": item.get("triggers", 0),
            "picks": item.get("picks", 0),
            "chunks": item.get("chunks", 0),
            "failed_chunks": item.get("failed_chunks", 0),
            "error": item.get("error"),
        })
    return rows


def public_job(job):
    with JOBS_LOCK:
        events = [public_event(event) for event in job.get("events", [])]
        triggers = sorted(
            job.get("triggers", []), key=lambda item: item["t"], reverse=True
        )[:200]
        station_rows = public_station_rows(job)
        snapshot = {
            "id": job["id"],
            "status": job.get("status", "queued"),
            "phase": job.get("phase", ""),
            "progress": round(float(job.get("progress", 0.0)), 1),
            "message": job.get("message", ""),
            "error": job.get("error"),
            "created": job.get("created"),
            "elapsed_s": round(
                (job.get("finished") or time.time()) - job["created_ts"], 1
            ),
            "start": fmt_time(job["start"]),
            "end": fmt_time(job["end"]),
            "stations": station_rows,
            "min_stations": job.get("min_stations", D["min_stations"]),
            "max_events": job.get("max_events", H["max_events"]),
            "trigger_count": job.get("trigger_count", len(job.get("triggers", []))),
            "picks_count": job.get("picks_count", 0),
            "candidate_count": job.get("candidate_count", 0),
        }

    events.sort(
        key=lambda event: event.get("origin_time") or event.get("time") or "",
        reverse=True,
    )

    return {
        **snapshot,
        "fdsn_url": FDSN_URL,
        "max_range_hours": H["max_range_hours"],
        "triggers": triggers,
        "events": events,
        "counts": {
            "stations_total": len(station_rows),
            "stations_ok": sum(1 for row in station_rows if row["state"] == "ok"),
            "stations_error": sum(1 for row in station_rows if row["state"] == "error"),
            "triggers": snapshot["trigger_count"],
            "candidates": snapshot["candidate_count"],
            "events": len(events),
            "located": sum(1 for event in events if event.get("located")),
        },
    }


def build_request(payload):
    start_value = str(payload.get("start") or "").strip()
    end_value = str(payload.get("end") or "").strip()
    if not start_value or not end_value:
        raise ValueError("Start and end are required")

    try:
        start = UTCDateTime(start_value)
        end = UTCDateTime(end_value)
    except Exception:
        raise ValueError("Start and end must be valid UTC times")

    now = UTCDateTime()
    if end <= start:
        raise ValueError("End must be after start")
    if end > now + 60:
        raise ValueError("End cannot be in the future")
    if end - start > float(H["max_range_hours"]) * 3600:
        raise ValueError(
            f"Range is limited to {H['max_range_hours']} hours"
        )

    names = payload.get("stations") or []
    if not isinstance(names, list):
        raise ValueError("Stations must be a list")

    selected = [
        station for station in STATIONS
        if station_id(station) in names or station["sta"] in names
    ]
    if len(selected) < 3:
        raise ValueError("Select at least 3 stations")

    try:
        min_stations = int(payload.get("min_stations", D["min_stations"]))
    except (TypeError, ValueError):
        raise ValueError("Minimum stations must be an integer")
    if min_stations < 3 or min_stations > len(selected):
        raise ValueError(
            f"Minimum stations must be between 3 and {len(selected)}"
        )

    try:
        max_events = int(payload.get("max_events", H["max_events"]))
    except (TypeError, ValueError):
        raise ValueError("Maximum events must be an integer")
    max_events = max(1, min(max_events, int(H["max_events"])))

    return {
        "start": start,
        "end": end,
        "stations": selected,
        "min_stations": min_stations,
        "max_events": max_events,
    }


# job runner
def run_job(job_id):
    # retrieve job parameters
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        stations = list(job["stations"])
        start = job["start"]
        end = job["end"]
        min_stations = job["min_stations"]
        max_events = job["max_events"]

    try:
        geometry = StationGeometry(stations=stations)
        params = AssociationParams.from_config(min_stations=min_stations, max_events=max_events)
        params.max_sp_sec = auto_max_sp_sec(geometry, params.vp, params.vs)
        params.validate()
        table = travel_table()

        job_update(job_id, status="running", phase="detecting", progress=2, message=f"Scanning {len(stations)} stations", error=None,)

        all_picks = []
        all_triggers = []
        stats = []
        total = len(stations)
        completed = 0

        with ThreadPoolExecutor(max_workers=max(1, min(FDSN_WORKERS, total))) as pool:
            futures = {
                pool.submit(detect_station_range, station, start, end, lambda: is_cancelled(job_id), ): station
                for station in stations
            }

            for future in as_completed(futures):
                station = futures[future]
                sid = station_id(station)
                try:
                    scan = future.result()
                except Exception as error:
                    scan = {
                        "picks": [],
                        "triggers": [],
                        "stats": {
                            "station": sid,
                            "state": "error",
                            "triggers": 0,
                            "picks": 0,
                            "chunks": 0,
                            "failed_chunks": 0,
                            "error": str(error),
                        },
                    }

                all_picks.extend(scan["picks"])
                all_triggers.extend(scan["triggers"])
                stats.append(scan["stats"])
                completed += 1

                job_update(
                    job_id,
                    station_stats=list(stats),
                    triggers=list(all_triggers[-int(H["max_triggers"]):]),
                    trigger_count=len(all_triggers),
                    picks_count=len(all_picks),
                    progress=2 + 50.0 * completed / max(1, total),
                    message=f"Scanned {completed} of {total} stations",
                )

                if is_cancelled(job_id):
                    break

        if is_cancelled(job_id):
            job_update(job_id, status="cancelled", phase="cancelled",message="Cancelled", progress=min(60.0, 2 + 50.0),)
            return

        job_update(job_id, phase="grouping", progress=54, message="Grouping station picks into candidate events",)

        grouping_picks = prepare_picks(all_picks, max(float(params.dedup_sec), float(D["dead_time_sec"]))
)
        groups = group_picks(
            grouping_picks,
            geometry,
            params.min_stations,
            coincidence_window(geometry),
            event_suppress_seconds(params),
        )

        job_update(
            job_id,
            candidate_count=len(groups),
            progress=58,
            message=f"Found {len(groups)} candidate event(s)",
        )

        groups = groups[:max_events]
        if not groups:
            job_update(job_id, status="done", phase="done", progress=100, message="Completed - no candidate events found")
            return

        for index, group in enumerate(groups):
            if is_cancelled(job_id):
                break

            job_update(
                job_id,
                phase="locating",
                progress=60 + 36.0 * index / max(1, len(groups)),
                message=f"Locating event {index + 1} of {len(groups)}",
            )

            try:
                event = process_candidate(job_id, group, index, geometry, params, table, start, end, stations,)
            except Exception as error:
                event = failed_event(group, start, end, f"Event processing failed: {error}", index)

            if event is None:
                event = failed_event(group, start, end, "Event processing returned no result", index)

            with JOBS_LOCK:
                current = JOBS.get(job_id)
                if current is not None:
                    current["events"].append(event)

            save_event(job_id, event)

            job_update(job_id, progress=60 + 36.0 * (index + 1) / max(1, len(groups)),)

        if is_cancelled(job_id):
            job_update(job_id, status="cancelled", phase="cancelled", message="Cancelled",)
            return

        with JOBS_LOCK:
            current = JOBS.get(job_id)
            located = sum(
                1 for event in (current or {}).get("events", [])
                if event.get("located")
            )
            total_events = len((current or {}).get("events", []))

        job_update(
            job_id,
            status="done",
            phase="done",
            progress=100,
            message=(
                f"Completed - {located} located event(s) "
                f"out of {total_events} candidate(s)"
            ),
        )
    except Exception as error:
        job_update(
            job_id,
            status="failed",
            phase="failed",
            error=str(error),
            message=str(error),
        )
    finally:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job["finished"] = time.time()
        save_job(job_id)


# website
# ---------------------------------------------------------------------------
PAGE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Historical Earthquake Detection (FDSN)</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  :root { --bg:#f1f4f7; --panel:#fff; --ink:#17212b; --muted:#63707d; --line:#dbe2e8;
          --ok:#1f7a55; --warn:#b7791f; --bad:#d6361f; --blue:#2563eb; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui, sans-serif;
         font-variant-numeric:tabular-nums; }
  header { background:var(--panel); border-bottom:1px solid var(--line); padding:14px 20px;
           display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; }
  h1 { margin:0; font-size:19px; }
  h2 { margin:0 0 10px; font-size:15px; }
  .muted { color:var(--muted); font-size:12.5px; }
  main { max-width:1180px; margin:0 auto; padding:18px; display:grid;
         grid-template-columns:340px minmax(0,1fr); gap:16px; align-items:start; }
  @media (max-width:900px) { main { grid-template-columns:1fr; } }
  .panel, section { background:var(--panel); border:1px solid var(--line); border-radius:7px; }
  .panel { padding:14px 16px; }
  section { padding:14px 16px; margin-bottom:16px; overflow:auto; }
  #results > * + * { margin-top:0; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; }
  .card { border:1px solid var(--line); border-radius:6px; padding:10px 12px; }
  .card span { display:block; color:var(--muted); font-size:12px; }
  .card b { font-size:21px; }
  label { display:block; font-size:12.5px; color:var(--muted); margin-top:10px; }
  input[type=datetime-local], input[type=number] {
    width:100%; padding:6px 8px; border:1px solid var(--line); border-radius:5px;
    font:inherit; color:var(--ink); background:#fff; margin-top:3px;
  }
  fieldset { border:1px solid var(--line); border-radius:6px; margin:12px 0 0; padding:8px 10px; }
  legend { color:var(--muted); font-size:12.5px; padding:0 4px; }
  .stations { display:grid; grid-template-columns:1fr 1fr; gap:3px 10px; }
  .stations label { display:flex; align-items:center; gap:7px; margin:3px 0; color:var(--ink);
                    font-size:13px; }
  .stations input { width:auto; }
  .quick { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
  button { font:inherit; border:1px solid var(--line); background:var(--panel); color:var(--ink);
           border-radius:5px; padding:6px 11px; cursor:pointer; }
  button:hover { background:#f4f6f8; }
  button.primary { background:var(--blue); border-color:var(--blue); color:#fff; font-weight:600;
                   width:100%; margin-top:14px; padding:9px; }
  button.primary:disabled { opacity:.55; cursor:wait; }
  .progress { height:8px; background:#e6ebf0; border-radius:4px; margin-top:12px; overflow:hidden; }
  #bar { height:100%; width:0; background:var(--blue); transition:width .25s ease; }
  .maphead { display:flex; align-items:center; justify-content:space-between; gap:10px;
             flex-wrap:wrap; margin-bottom:10px; }
  .maphead h2 { margin:0; }
  .linkbtn { font:inherit; font-size:12.5px; background:none; border:1px solid var(--line);
             border-radius:4px; padding:3px 9px; color:var(--ink); cursor:pointer; }
  .linkbtn:hover { background:#f4f6f8; }
  #map { height:430px; border:1px solid var(--line); border-radius:6px; background:#e3e8ed; }
  .legend { display:flex; gap:16px; flex-wrap:wrap; margin-top:8px; color:var(--muted); font-size:12.5px; }
  .legend span { display:inline-flex; align-items:center; gap:6px; }
  .legend .tri { width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent;
                 border-bottom:11px solid var(--ok); }
  .legend .epi { width:11px; height:11px; border-radius:50%; background:var(--bad); border:2px solid #fff;
                 box-shadow:0 0 0 1px rgba(20,32,43,.4); }
  .st-mark { position:relative; }
  .st-mark svg { display:block; filter:drop-shadow(0 0 1px rgba(20,32,43,.55)); }
  .st-mark .st-label { position:absolute; left:19px; top:-1px; font-size:11.5px; font-weight:600;
                       white-space:nowrap; background:rgba(255,255,255,.9); padding:0 4px;
                       border-radius:3px; color:var(--ink); }
  .leaflet-popup-content { font:13px/1.45 system-ui, sans-serif; margin:10px 12px; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; margin-top:8px; }
  th, td { text-align:left; padding:5px 6px 5px 0; border-bottom:1px solid #edf1f4;
           white-space:nowrap; }
  th { color:var(--muted); font-weight:500; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:#9aa5b1;
         margin-right:5px; }
  .ok .dot { background:var(--ok); } .error .dot { background:var(--bad); }
  .cancelled .dot { background:var(--warn); } .pending .dot { background:#9aa5b1; }
  .badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11.5px;
           font-weight:600; background:#ddefe6; color:#17603f; }
  .badge.bad { background:#fbeae6; color:#8a2412; }
  .badge.warn { background:#fbf3df; color:#7a5411; }
  .event { border-top:1px solid var(--line); padding:10px 0; }
  .event:first-child { border-top:0; }
  .eventhead { display:flex; gap:9px; align-items:baseline; flex-wrap:wrap; cursor:pointer; }
  .eventhead b { font-size:15px; }
  .eventhead span:last-child { margin-left:auto; }
  .plot { display:block; width:100%; max-width:760px; margin-top:10px;
          border:1px solid var(--line); border-radius:6px; }
  details { margin-top:7px; }
  summary { cursor:pointer; color:var(--muted); font-size:12.5px; }
  #error { display:none; margin-bottom:14px; padding:10px 13px; border:1px solid #e8b4aa;
           background:#fbeae6; color:#8a2412; border-radius:6px; }
  .err { color:var(--bad); }
  @media (max-width:600px) { .stations { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <div><h1>Historical Earthquake Detection</h1><div class="muted" id="server">FDSN service</div></div>
  <div class="muted" id="clock">Ready</div>
</header>
<main>
  <section class="panel">
    <form id="run-form">
      <label>Start (UTC) <input type="datetime-local" id="start" step="1"></label>
      <label>End (UTC) <input type="datetime-local" id="end" step="1"></label>
      <div class="quick">
        <button type="button" data-hours="1">1 h</button>
        <button type="button" data-hours="3">3 h</button>
        <button type="button" data-hours="6">6 h</button>
        <button type="button" data-hours="12">12 h</button>
        <button type="button" data-hours="24">24 h</button>
      </div>

      <fieldset>
        <legend>Stations</legend>
        <div class="stations" id="station-list"></div>
      </fieldset>

      <details>
        <summary>Advanced settings</summary>
        <label>Minimum stations <input type="number" id="min-stations" min="3" step="1"></label>
        <label>Maximum events <input type="number" id="max-events" min="1" step="1"></label>
      </details>

      <button class="primary" id="runbtn" type="submit">Run detection</button>
    </form>
    <div class="muted" id="jobmsg" style="margin-top:10px">Ready.</div>
    <div class="progress"><div id="bar"></div></div>
    <button type="button" id="cancelbtn" class="linkbtn" style="margin-top:8px;display:none">Cancel</button>
  </section>

  <div id="results">
    <div id="error"></div>
    <section><div class="cards" id="cards"></div></section>

    <section>
      <div class="maphead">
        <h2>Station map</h2>
        <button class="linkbtn" id="fit">Fit to map</button>
      </div>
      <div id="map"></div>
      <div class="legend">
        <span><i class="tri"></i>Station</span>
        <span><i class="epi"></i>Located epicenter</span>
        <span class="muted" id="mapnote"></span>
      </div>
    </section>

    <section>
      <h2>Station scan</h2>
      <table>
        <thead><tr><th>Station</th><th>State</th><th>Triggers</th><th>Picks</th><th>Chunks</th><th>Error</th></tr></thead>
        <tbody id="station-rows"><tr><td colspan="6" class="muted">No scan yet.</td></tr></tbody>
      </table>
    </section>

    <section>
      <h2>Detected events</h2>
      <div id="events"><div class="muted">No events yet.</div></div>
    </section>

    <section>
      <h2>Station triggers (newest first)</h2>
      <div id="triggers"><div class="muted">No triggers yet.</div></div>
    </section>
  </div>
</main>
<script>
const STATIONS = __STATIONS__;
const DEFAULTS = __DEFAULTS__;
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => (
  {"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]
));
const num = (value, digits = 1) => value == null ? "-" : Number(value).toFixed(digits);
const shortTime = value => value ? String(value).replace("T", " ").slice(0, 19) + " UTC" : "-";

function utcInput(date) {
  return date.toISOString().slice(0, 19);
}

function setRange(hours) {
  const end = new Date(Date.now() - 300000);
  const start = new Date(end.getTime() - hours * 3600000);
  $("end").value = utcInput(end);
  $("start").value = utcInput(start);
}

$("station-list").innerHTML = STATIONS.map(station => `
  <label><input type="checkbox" value="${esc(station.name)}" checked>
    <span>${esc(station.sta)}</span><span class="muted">${esc(station.net)}</span></label>
`).join("");

$("min-stations").value = DEFAULTS.min_stations;
$("max-events").value = DEFAULTS.max_events;

/* ---------- map ---------- */
const STATE_COLORS = { ok:"#1f7a55", error:"#d6361f", cancelled:"#b7791f", pending:"#9aa5b1" };
let map = null, stationMarkers = {}, epicenterMarkers = {}, fitted = false;
let lastData = null, pollTimer = null;

function initMap() {
  if (typeof L === "undefined") {
    $("map").innerHTML = '<div class="muted" style="padding:24px">The map needs internet access to load Leaflet and the map tiles.</div>';
    $("fit").hidden = true;
    return;
  }
  const street = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  });
  const terrain = L.tileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", {
    maxZoom: 17, subdomains: "abc",
    attribution: 'Map data &copy; OpenStreetMap contributors, SRTM | Style &copy; <a href="https://opentopomap.org">OpenTopoMap</a>'
  });
  const satellite = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    maxZoom: 18, attribution: "Tiles &copy; Esri"
  });
  map = L.map("map", { layers: [street] }).setView([28.2, 85.3], 7);
  L.control.layers({ "Street": street, "Terrain": terrain, "Satellite": satellite }, null,
                   { position: "topright" }).addTo(map);
  $("fit").onclick = fitMap;
}

function fitMap() {
  if (!map) return;
  const points = [...Object.values(stationMarkers), ...Object.values(epicenterMarkers)]
    .map(marker => marker.getLatLng());
  if (points.length) map.fitBounds(L.latLngBounds(points), { padding: [50, 50], maxZoom: 9 });
}

function stationIcon(station) {
  const color = STATE_COLORS[station.state] || "#9aa5b1";
  return L.divIcon({
    className: "", iconSize: [18, 18], iconAnchor: [9, 9],
    html: `<div class="st-mark">
      <svg width="18" height="18" viewBox="0 0 18 18">
        <polygon points="9,1.5 16.8,16.2 1.2,16.2" fill="${color}" stroke="#fff" stroke-width="1.7"/>
      </svg><span class="st-label">${esc(station.sta)}</span></div>`
  });
}

function stationPopup(station) {
  const lines = [`<b>${esc(station.name)}</b>`, `State: ${esc(station.state)}`];
  if (station.triggers != null) lines.push(`Triggers: ${esc(station.triggers)}`);
  if (station.picks != null) lines.push(`Accepted P picks: ${esc(station.picks)}`);
  if (station.chunks != null) lines.push(`Chunks scanned: ${esc(station.chunks)}`);
  if (station.error) lines.push(`<span class="err">${esc(station.error)}</span>`);
  lines.push(`<span class="muted">${num(station.lat, 4)}, ${num(station.lon, 4)}</span>`);
  return lines.join("<br>");
}

function epicenterPopup(event) {
  return `<b style="color:#d6361f">M ${esc(event.magnitude ?? "?")}</b><br>
    <b>Estimated epicenter</b><br>
    Latitude: ${num(event.lat, 4)}<br>
    Longitude: ${num(event.lon, 4)}<br>
    Depth: ${num(event.depth_km, 0)} km<br>
    Origin time: ${esc(event.origin_time || "-")}<br>
    <span class="muted">RMS ${num(event.rms_sec, 2)} s &middot; ${esc(event.n_stations)} stations &middot; ${esc(event.method || "-")}</span>`;
}

function updateMap(data) {
  if (!map) return;

  const placed = (data.stations || []).filter(station => station.lat != null && station.lon != null);
  placed.forEach(station => {
    const icon = stationIcon(station);
    const popup = stationPopup(station);
    if (stationMarkers[station.name]) {
      stationMarkers[station.name]
        .setLatLng([station.lat, station.lon])
        .setIcon(icon)
        .setPopupContent(popup);
    } else {
      stationMarkers[station.name] = L.marker([station.lat, station.lon], {
        icon, title: station.name
      }).addTo(map).bindPopup(popup);
    }
  });

  const located = (data.events || []).filter(event => event.located && event.lat != null && event.lon != null);
  const seen = new Set();
  located.forEach(event => {
    seen.add(event.id);
    const popup = epicenterPopup(event);
    if (epicenterMarkers[event.id]) {
      epicenterMarkers[event.id].setLatLng([event.lat, event.lon]).setPopupContent(popup);
    } else {
      epicenterMarkers[event.id] = L.circleMarker([event.lat, event.lon], {
        radius: 10, color: "#8a2412", weight: 2, fillColor: "#d6361f", fillOpacity: 0.85
      }).addTo(map).bindPopup(popup);
    }
  });
  Object.keys(epicenterMarkers).forEach(id => {
    if (!seen.has(id)) {
      map.removeLayer(epicenterMarkers[id]);
      delete epicenterMarkers[id];
    }
  });

  const counts = data.counts || {};
  $("mapnote").textContent =
    `${counts.located || 0} located event${counts.located === 1 ? "" : "s"} · ` +
    `${counts.stations_ok || 0}/${counts.stations_total || 0} stations with data`;

  if (!fitted && (placed.length || located.length)) { fitted = true; fitMap(); }
}

function focusEpicenter(id) {
  const marker = epicenterMarkers[id];
  if (!map || !marker) return;
  map.setView(marker.getLatLng(), Math.max(map.getZoom(), 8));
  marker.openPopup();
  $("map").scrollIntoView({ behavior: "smooth", block: "center" });
}

/* ---------- rendering ---------- */
function renderCards(data) {
  const counts = data.counts || {};
  $("cards").innerHTML = [
    ["Stations with data", `${counts.stations_ok || 0} / ${counts.stations_total || 0}`],
    ["Station triggers", counts.triggers || 0],
    ["Candidate events", counts.candidates || 0],
    ["Located events", counts.located || 0],
  ].map(([label, value]) => `<div class="card"><span>${esc(label)}</span><b>${esc(value)}</b></div>`).join("");
}

function renderStations(data) {
  const rows = data.stations || [];
  if (!rows.length) {
    $("station-rows").innerHTML = '<tr><td colspan="6" class="muted">No scan yet.</td></tr>';
    return;
  }
  $("station-rows").innerHTML = rows.map(station => `
    <tr class="${esc(station.state)}">
      <td><span class="dot"></span>${esc(station.name)}</td>
      <td>${esc(station.state)}</td>
      <td>${esc(station.triggers)}</td>
      <td>${esc(station.picks)}</td>
      <td>${esc(station.chunks)}${station.failed_chunks ? ` <span class="muted">(${esc(station.failed_chunks)} failed)</span>` : ""}</td>
      <td class="muted">${esc(station.error || "-")}</td>
    </tr>`).join("");
}

function eventCard(jobId, event) {
  const head = event.located
    ? `<b>M ${esc(event.magnitude ?? "?")}</b><span>${esc(event.origin_time)}</span>
       <span class="badge">LOCATED</span>`
    : `<b>Detection</b><span>${esc(event.time || "-")}</span>
       <span class="badge bad">NOT LOCATED</span>`;

  const summary = event.located
    ? `${num(event.lat, 3)}, ${num(event.lon, 3)} · ${num(event.depth_km, 0)} km deep ·
       RMS ${num(event.rms_sec, 2)} s · ${esc(event.n_stations)} stations ·
       ${esc(event.method || "-")}`
    : `<span class="err">${esc(event.error || "No location could be computed")}</span>`;

  const images = event.has_waveform
    ? `<img class="plot" loading="lazy" src="api/jobs/${encodeURIComponent(jobId)}/events/${encodeURIComponent(event.id)}/waveform.png" alt="Event waveforms">`
    : "";

  const rows = (event.stations || []).map(station => `
    <tr><td>${esc(station.name)}</td><td>${esc(station.quality || "-")}</td>
      <td>${esc((station.p_time || "-").slice(11, 23))}</td>
      <td>${esc((station.s_time || "-").slice(11, 23))}</td>
      <td>${esc(station.ml ?? "-")}</td>
      <td>${num(station.distance_km, 1)}</td>
      <td>${num(station.p_residual_sec, 2)}</td></tr>
  `).join("");

  const table = rows ? `
    <table><thead><tr><th>Station</th><th>SNR</th><th>P</th><th>S</th><th>ML</th><th>Dist (km)</th><th>P res (s)</th></tr></thead>
    <tbody>${rows}</tbody></table>` : "";

  const target = event.located ? ` data-event-id="${esc(event.id)}" title="Show on map"` : "";
  const triggerStations = (event.trigger_stations || []).join(", ");

  return `<div class="event">
    <div class="eventhead"${target}>${head}</div>
    <div class="muted">${summary}</div>
    <div class="muted">Triggered: ${esc(triggerStations || "-")}</div>
    <details><summary>Waveforms and station picks</summary>${images}${table}</details>
  </div>`;
}

function renderEvents(data) {
  const events = data.events || [];
  $("events").innerHTML = events.length
    ? events.map(event => eventCard(data.id, event)).join("")
    : '<div class="muted">No events yet.</div>';

  document.querySelectorAll("#events .eventhead[data-event-id]").forEach(element => {
    element.onclick = () => focusEpicenter(element.dataset.eventId);
  });
}

function renderTriggers(data) {
  const triggers = data.triggers || [];
  if (!triggers.length) {
    $("triggers").innerHTML = '<div class="muted">No triggers yet.</div>';
    return;
  }
  $("triggers").innerHTML = `
    <div class="muted">Showing ${triggers.length} of ${esc(data.trigger_count || triggers.length)} triggers.</div>
    <table><thead><tr><th>Time (UTC)</th><th>Station</th><th>CFT</th><th>Duration (s)</th><th>Status</th></tr></thead>
    <tbody>${triggers.map(trigger => `
      <tr><td>${esc(trigger.time)}</td><td>${esc(trigger.station)}</td>
        <td>${esc(trigger.cft)}</td><td>${esc(trigger.duration)}</td>
        <td>${trigger.accepted
          ? '<span class="badge">PICKED</span>'
          : '<span class="badge warn">REJECTED</span>'}</td></tr>
    `).join("")}</tbody></table>`;
}

function render(data) {
  lastData = data;
  const running = data.status === "running" || data.status === "queued";

  $("server").textContent = (data.fdsn_url || "FDSN") + " · " + (data.status || "");
  $("clock").textContent = `${data.start || "-"} → ${data.end || "-"} · ${num(data.elapsed_s, 0)} s`;
  $("error").style.display = data.error ? "block" : "none";
  $("error").textContent = data.error || "";

  let message = `${data.status} · ${data.phase || ""} · ${data.message || ""}`;
  if (data.error) message = `<span class="err">${esc(data.error)}</span>`;
  $("jobmsg").innerHTML = message;
  $("bar").style.width = `${data.progress || 0}%`;
  $("runbtn").disabled = running;
  $("cancelbtn").style.display = running ? "inline-block" : "none";

  renderCards(data);
  renderStations(data);
  renderEvents(data);
  renderTriggers(data);
  updateMap(data);
}

async function poll(jobId) {
  if (pollTimer) clearTimeout(pollTimer);
  try {
    const response = await fetch(`api/jobs/${jobId}`, { cache: "no-store" });
    if (!response.ok) {
      if (response.status === 404) localStorage.removeItem("fdsn_job");
      throw new Error("HTTP " + response.status);
    }
    const data = await response.json();
    render(data);
    if (data.status === "running" || data.status === "queued") {
      pollTimer = setTimeout(() => poll(jobId), 1500);
    }
  } catch (error) {
    $("jobmsg").innerHTML = `<span class="err">Cannot read job: ${esc(error.message)}</span>`;
  }
}

/* ---------- form ---------- */
$("run-form").onsubmit = async event => {
  event.preventDefault();
  const stations = [...document.querySelectorAll("#station-list input:checked")]
    .map(input => input.value);
  if (stations.length < 3) {
    $("jobmsg").textContent = "Select at least 3 stations.";
    return;
  }

  $("runbtn").disabled = true;
  $("jobmsg").textContent = "Starting job...";
  try {
    const response = await fetch("api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        start: $("start").value,
        end: $("end").value,
        stations: stations,
        min_stations: Number($("min-stations").value),
        max_events: Number($("max-events").value),
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      if (response.status === 409 && data.job_id) {
        localStorage.setItem("fdsn_job", data.job_id);
        poll(data.job_id);
        return;
      }
      throw new Error(data.error || "Could not start job");
    }
    localStorage.setItem("fdsn_job", data.job_id);
    poll(data.job_id);
  } catch (error) {
    $("runbtn").disabled = false;
    $("jobmsg").innerHTML = `<span class="err">${esc(error.message)}</span>`;
  }
};

$("cancelbtn").onclick = async () => {
  const jobId = localStorage.getItem("fdsn_job");
  if (!jobId) return;
  await fetch(`api/jobs/${jobId}/cancel`, { method: "POST" });
};

document.querySelectorAll(".quick button").forEach(button => {
  button.onclick = () => setRange(Number(button.dataset.hours));
});

initMap();
setRange(6);

const savedJob = localStorage.getItem("fdsn_job");
if (savedJob) poll(savedJob);

</script>
</body>
</html>
"""

PAGE = PAGE_HEAD


def page_html():
    station_payload = [
        {
            "name": station_id(station),
            "net": station["net"],
            "sta": station["sta"],
            "lat": station.get("lat"),
            "lon": station.get("lon"),
            "cha": station.get("cha"),
        }
        for station in STATIONS
    ]
    defaults = {
        "min_stations": D["min_stations"],
        "max_events": H["max_events"],
        "max_range_hours": H["max_range_hours"],
    }
    return (
        PAGE
        .replace("__STATIONS__", json.dumps(station_payload))
        .replace("__DEFAULTS__", json.dumps(defaults))
    )


app = Flask(__name__)


@app.route("/")
def home():
    return Response(page_html(), mimetype="text/html")


@app.route("/api/run", methods=["POST"])
def api_run():
    payload = request.get_json(silent=True) or {}
    try:
        params = build_request(payload)
    except ValueError as error:
        return jsonify(error=str(error)), 400

    with JOBS_LOCK:
        active = [
            job_id for job_id, job in JOBS.items()
            if job.get("status") in ("running", "queued")
        ]
        if active:
            return jsonify(
                error="A historical job is already running",
                job_id=active[0],
            ), 409

        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "phase": "queued",
            "progress": 0.0,
            "message": "Queued",
            "error": None,
            "created": fmt_time(UTCDateTime()),
            "created_ts": time.time(),
            "finished": None,
            "start": params["start"],
            "end": params["end"],
            "stations": params["stations"],
            "min_stations": params["min_stations"],
            "max_events": params["max_events"],
            "triggers": [],
            "trigger_count": 0,
            "picks_count": 0,
            "events": [],
            "station_stats": [],
            "candidate_count": 0,
            "cancel": False,
        }

    trim_jobs()
    threading.Thread(target=run_job, args=(job_id,), daemon=True).start()
    return jsonify(job_id=job_id)


@app.route("/api/jobs/<job_id>")
def api_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify(error="Job not found"), 404
    return jsonify(public_job(job))


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def api_cancel(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify(error="Job not found"), 404
        job["cancel"] = True
    return jsonify(ok=True)


@app.route("/api/jobs/<job_id>/events/<event_id>/waveform.png")
def event_waveform(job_id, event_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        event = None
        if job is not None:
            event = next(
                (item for item in job.get("events", [])
                 if item.get("id") == event_id),
                None,
            )
        png = event.get("waveform_png") if event else None

    if png is None:
        return Response("No waveform image for this event", status=404,
                        mimetype="text/plain")
    return Response(
        png,
        mimetype="image/png",
        headers={"Cache-Control": "no-store"},
    )


def trim_jobs():
    """Keep only a few old jobs in memory."""
    with JOBS_LOCK:
        if len(JOBS) <= 8:
            return
        finished = [
            (job_id, job) for job_id, job in JOBS.items()
            if job.get("status") not in ("running", "queued")
        ]
        finished.sort(key=lambda item: item[1].get("created_ts", 0))
        for job_id, _job in finished[:len(JOBS) - 8]:
            JOBS.pop(job_id, None)


def main():
    travel_table()
    app.run(host=WEB_HOST, port=FDSN_WEB_PORT, debug=False, use_reloader=False, threaded=True,)


if __name__ == "__main__":
    main()
