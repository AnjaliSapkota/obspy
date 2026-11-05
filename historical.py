"""Historical earthquake detection and location.

This module is loaded by realtime_earthquake.py and mounted at /historical/.
It scans archived FDSN data, groups station triggers, and locates events from
their P arrival times.
"""

import base64
import io
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Blueprint, Response, jsonify, request
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from obspy.taup import TauPyModel
from scipy.signal import hilbert

from magnitude import calculate_ml
from plotting import circle_map_png


FDSN_URL = "https://seiscomp.alertnepal.online"
FDSN_TIMEOUT = 60
CHUNK_SEC = 1800
DETECT_PAD = 10
MAX_RANGE_HOURS = 24
MAX_EVENTS = 20
MAX_TRIGGERS = 2000
COUNTS_PER_M = 1.0
TRAVEL_DEPTHS = (10.0, 20.0, 30.0)
TRAVEL_MAX_KM = 1400.0
TRAVEL_STEP_KM = 10.0
# S-P separations shorter than this are dominated by the P coda, so they are
# not used as distance constraints.
S_MIN_SEPARATION = 30.0

DEFAULT_SETTINGS = {
    "fs": 50.0,
    "fmin": 1.0,
    "fmax": 8.0,
    "sta_p": 1.0,
    "lta_p": 10.0,
    "thr_on": 4.0,
    "thr_off": 1.5,
    "sta_s": 1.0,
    "lta_s": 8.0,
    "min_sp_sec": 1.0,
    "max_sp_sec": 90.0,
    "thr_on_s": 2.5,
    "thr_off_s": 1.0,
    "coincidence": 90.0,
    "min_stations": 3,
    "pre": 15.0,
    "post": 10.0,
    "vp": 6.0,
    "vs": 3.5,
}

JOBS = {}
JOBS_LOCK = threading.Lock()


def station_id(station):
    return f"{station['net']}.{station['sta']}"


def fmt_time(value):
    try:
        return UTCDateTime(value).strftime("%Y-%m-%d %H:%M:%S") + " UTC"
    except Exception:
        return "-"


def preprocess(stream, settings):
    """Merge, detrend, bandpass, and resample a waveform stream."""
    fs = float(settings["fs"])
    st = stream.copy()
    st.merge(method=1, fill_value=0)
    st.detrend("linear")
    st.detrend("demean")

    try:
        st.taper(0.02)
    except Exception:
        pass

    for trace in list(st):
        if trace.stats.npts < 20:
            st.remove(trace)
            continue
        if abs(float(trace.stats.sampling_rate) - fs) > 1e-6:
            try:
                trace.resample(fs)
            except Exception:
                st.remove(trace)

    if len(st) == 0:
        raise ValueError("No usable samples")

    try:
        st.filter(
            "bandpass",
            freqmin=float(settings["fmin"]),
            freqmax=float(settings["fmax"]),
            corners=4,
            zerophase=True,
        )
    except Exception as error:
        raise ValueError(f"Bandpass filter failed: {error}")

    return st


def vertical_trace(stream):
    traces = stream.select(component="Z")
    if len(traces) == 0:
        traces = [tr for tr in stream if tr.stats.channel.endswith("Z")]
    if not traces:
        return None
    return max(traces, key=lambda tr: tr.stats.npts)


def dedup_triggers(triggers, gap=5.0):
    ordered = sorted(triggers, key=lambda item: item["t"])
    result = []
    for trigger in ordered:
        if result and abs(trigger["t"] - result[-1]["t"]) < gap:
            if trigger.get("cft", 0) > result[-1].get("cft", 0):
                result[-1] = trigger
        else:
            result.append(trigger)
    return result


def detect_station_range(station, start, end, settings, cancel=None):
    """Scan one station in chunks and return all STA/LTA triggers."""
    client = Client(FDSN_URL, timeout=FDSN_TIMEOUT)
    fs = float(settings["fs"])
    sta = float(settings["sta_p"])
    lta = float(settings["lta_p"])
    triggers = []
    ok_chunks = 0
    failed_chunks = 0
    last_error = None

    position = start
    while position < end:
        if cancel is not None and cancel():
            break

        chunk_end = min(position + CHUNK_SEC, end)
        try:
            stream = client.get_waveforms(
                station["net"],
                station["sta"],
                "*",
                "?H?",
                position - DETECT_PAD,
                chunk_end + DETECT_PAD,
            )
            if len(stream) == 0:
                raise ValueError("No data")

            stream = preprocess(stream, settings)
            z = vertical_trace(stream)
            if z is None:
                raise ValueError("No vertical component")

            z.trim(position, chunk_end, pad=True, fill_value=0)
            if z.stats.npts < int((lta + sta + 3) * fs):
                raise ValueError("Not enough samples in chunk")

            cft = classic_sta_lta(z.data.astype(float), int(sta * fs), int(lta * fs))
            cft = np.nan_to_num(cft, nan=0.0, posinf=0.0, neginf=0.0)

            for onset, offset in trigger_onset(cft, float(settings["thr_on"]), float(settings["thr_off"])):
                trigger_time = z.stats.starttime + onset / fs
                if trigger_time < position or trigger_time >= chunk_end:
                    continue
                triggers.append({
                    "station": station_id(station),
                    "t": float(trigger_time.timestamp),
                    "time": fmt_time(trigger_time),
                    "cft": round(float(cft[onset]), 2),
                    "duration": round(float((offset - onset) / fs), 2),
                })
            ok_chunks += 1
        except Exception as error:
            failed_chunks += 1
            last_error = str(error)

        position = chunk_end

    return {
        "triggers": dedup_triggers(triggers, 5.0)[:MAX_TRIGGERS],
        "ok": ok_chunks,
        "failed": failed_chunks,
        "error": last_error,
    }


def group_triggers(triggers, coincidence, min_stations, suppress=60.0):
    """Group triggers from different stations into candidate events.

    For every possible start time in the current window, count the distinct
    stations that trigger within ``coincidence`` seconds.  Accept the start
    that covers the most stations, then ignore those stations for ``suppress``
    seconds so their S/coda triggers cannot start a second bogus group.
    """
    ordered = sorted(triggers, key=lambda item: item["t"])
    groups = []
    last_used = {}
    index = 0

    while index < len(ordered):
        horizon = ordered[index]["t"] + coincidence
        best = None
        start = index

        while start < len(ordered) and ordered[start]["t"] <= horizon:
            anchor = ordered[start]
            if anchor["t"] - last_used.get(anchor["station"], -1e18) >= suppress:
                by_station = {}
                end = start
                while end < len(ordered) and ordered[end]["t"] - anchor["t"] <= coincidence:
                    item = ordered[end]
                    if item["t"] - last_used.get(item["station"], -1e18) >= suppress:
                        by_station.setdefault(item["station"], item)
                    end += 1
                if len(by_station) >= int(min_stations):
                    if best is None or len(by_station) > best[0]:
                        best = (len(by_station), end, by_station)
            start += 1

        if best is None:
            index += 1
            continue

        _, end, by_station = best
        group = sorted(by_station.values(), key=lambda item: item["t"])
        groups.append(group)
        for item in group:
            last_used[item["station"]] = item["t"]
        index = end

    return groups


def fetch_event_streams(stations, start, end, settings):
    """Fetch the short window needed to locate one candidate event."""
    def fetch_one(station):
        client = Client(FDSN_URL, timeout=FDSN_TIMEOUT)
        stream = client.get_waveforms(
            station["net"], station["sta"], "*", "?H?", start, end
        )
        if len(stream) == 0:
            raise ValueError("No data")
        return station_id(station), preprocess(stream, settings)

    streams = {}
    errors = {}
    workers = max(1, min(8, len(stations)))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, station): station for station in stations}
        for future in as_completed(futures):
            station = futures[future]
            try:
                sid, stream = future.result()
                streams[sid] = stream
            except Exception as error:
                errors[station_id(station)] = str(error)

    return streams, errors


def make_fetch(streams):
    """Build the component callback used for magnitude and plots."""
    def fetch_components(client, station, start, end, fs, fmin, fmax, wait_sec):
        del client, wait_sec
        sid = station_id(station)
        stream = streams.get(sid)
        if stream is None:
            raise ValueError("No historical data for this station")

        st = stream.copy()
        st.trim(start, end, pad=True, fill_value=0)

        z = vertical_trace(st)
        if z is None or z.stats.npts < 10:
            raise ValueError("No vertical data in this window")
        z.data = z.data.astype(float)

        horizontals = st.select(component="N") + st.select(component="E")
        if len(horizontals) >= 2:
            horizontals.trim(start, end, pad=True, fill_value=0)
            length = min([len(z.data)] + [len(tr.data) for tr in horizontals])
            if length > 10:
                envelope = np.zeros(length, dtype=float)
                max_horizontal = np.zeros(length, dtype=float)
                for trace in horizontals:
                    data = trace.data[:length].astype(float)
                    envelope += data * data
                    max_horizontal = np.maximum(max_horizontal, np.abs(data))
                z.data = z.data[:length]
                return z, np.sqrt(envelope), max_horizontal

        envelope = np.abs(hilbert(z.data.astype(float)))
        max_horizontal = np.abs(z.data.astype(float)) / max(COUNTS_PER_M, 1e-9)
        return z, envelope, max_horizontal

    return fetch_components


def overview_png(streams, result, start, end):
    """One compact waveform image per located event."""
    names = sorted(streams)
    if not names:
        return None

    picks = {item.get("name"): item for item in result.get("stations", []) if item.get("name")}
    fig, axes = plt.subplots(
        len(names), 1,
        figsize=(10, max(2.4, 1.25 * len(names))),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for ax, sid in zip(axes, names):
        z = vertical_trace(streams[sid])
        if z is None:
            continue
        z = z.copy().trim(start, end, pad=True, fill_value=0)
        x = z.times() + float(z.stats.starttime - start)
        data = z.data.astype(float)
        scale = float(np.max(np.abs(data))) if len(data) else 0.0
        if scale > 0:
            data = data / scale
        ax.plot(x, data, color="#2563EB", linewidth=0.7)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Z norm", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)
        ax.text(0.005, 0.9, sid, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top")

        info = picks.get(sid) or {}
        if info.get("p_time"):
            ax.axvline(float(UTCDateTime(info["p_time"]) - start),
                       color="#D6361F", linestyle="--", linewidth=1)
        if info.get("s_time"):
            ax.axvline(float(UTCDateTime(info["s_time"]) - start),
                       color="#1F7A55", linestyle="--", linewidth=1)

    axes[-1].set_xlabel("Seconds from event window start")
    fig.suptitle(
        "Event waveforms, each trace normalized (red = P arrival, green = predicted S)\n"
        "Quiet lead-ins are low background noise, not data gaps",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


EARTH_RADIUS_KM = 6371.0


def epicentral_km(lat, lon, station_lat, station_lon):
    """Great-circle distance in km, vectorized over the first two arguments."""
    p1 = np.radians(lat)
    p2 = np.radians(station_lat)
    dp = p2 - p1
    dl = np.radians(station_lon - lon)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


TAUP_MODEL = None
TAUP_CACHE = {}
TAUP_LOCK = threading.Lock()


def travel_curves(depth):
    """P and S travel times on a distance grid, cached per source depth."""
    global TAUP_MODEL
    key = float(depth)
    with TAUP_LOCK:
        cached = TAUP_CACHE.get(key)
        if cached is not None:
            return cached
        if TAUP_MODEL is None:
            TAUP_MODEL = TauPyModel(model="iasp91")
        distances = np.arange(
            0.0, TRAVEL_MAX_KM + TRAVEL_STEP_KM, TRAVEL_STEP_KM
        )
        p_times = np.empty(len(distances))
        s_times = np.empty(len(distances))
        for index, km in enumerate(distances):
            arrivals = TAUP_MODEL.get_travel_times(
                source_depth_in_km=key,
                distance_in_degree=float(km) / 111.19,
                phase_list=["P", "p", "S", "s"],
            )
            p = [a.time for a in arrivals if a.name in ("P", "p")]
            s = [a.time for a in arrivals if a.name in ("S", "s")]
            p_times[index] = min(p) if p else np.nan
            s_times[index] = min(s) if s else np.nan
        curves = (distances, p_times, s_times)
        TAUP_CACHE[key] = curves
        return curves


def locate_event(stations, p_times, s_times, depths=TRAVEL_DEPTHS, step=0.05):
    """Grid-search the epicenter and origin time that fit the arrivals best.

    Each depth in ``depths`` is tried and the solution with the lowest RMS
    arrival-time residual wins.  ``s_times`` may be empty; P times alone are
    enough to locate, while S-P times add absolute distance information.
    """
    lat0 = np.array([station["lat"] for station in stations])
    lon0 = np.array([station["lon"] for station in stations])
    grid_lat, grid_lon = np.meshgrid(
        np.arange(lat0.min() - 5.0, lat0.max() + 5.0 + step, step),
        np.arange(lon0.min() - 5.0, lon0.max() + 5.0 + step, step),
        indexing="ij",
    )

    observed_p = np.array([
        UTCDateTime(p_times[station_id(s)]).timestamp for s in stations
    ])
    s_used = [s for s in stations if station_id(s) in s_times]
    observed_s = np.array([
        UTCDateTime(s_times[station_id(s)]).timestamp for s in s_used
    ])
    shape = (-1,) + (1,) * grid_lat.ndim

    best = None
    for depth in depths:
        distances, p_curve, s_curve = travel_curves(depth)
        predicted_p = np.empty((len(stations),) + grid_lat.shape)
        for index, station in enumerate(stations):
            km = epicentral_km(
                grid_lat, grid_lon, station["lat"], station["lon"]
            )
            predicted_p[index] = np.interp(km, distances, p_curve)

        origin = (observed_p.reshape(shape) - predicted_p).mean(axis=0)
        residual = observed_p.reshape(shape) - (origin + predicted_p)
        total = np.sum(residual ** 2, axis=0)
        count = len(stations)

        for index, station in enumerate(s_used):
            km = epicentral_km(
                grid_lat, grid_lon, station["lat"], station["lon"]
            )
            predicted_s = origin + np.interp(km, distances, s_curve)
            total = total + (observed_s[index] - predicted_s) ** 2
            count += 1

        rms = np.sqrt(total / count)
        chosen = np.unravel_index(np.argmin(rms), rms.shape)
        value = float(rms[chosen])
        if best is None or value < best["rms"]:
            best = {
                "rms": value,
                "depth": float(depth),
                "lat": float(grid_lat[chosen]),
                "lon": float(grid_lon[chosen]),
                "origin": float(origin[chosen]),
                "distances": distances,
                "p_curve": p_curve,
                "s_curve": s_curve,
            }
    return best


def predict_times(location, station):
    """Epicentral distance plus predicted P and S times for one station."""
    km = float(epicentral_km(
        location["lat"], location["lon"], station["lat"], station["lon"]
    ))
    p_time = location["origin"] + float(
        np.interp(km, location["distances"], location["p_curve"])
    )
    s_time = location["origin"] + float(
        np.interp(km, location["distances"], location["s_curve"])
    )
    return km, p_time, s_time


def choose_location(stations, p_times, s_times):
    """Locate with P times, then add S-P times that survive an outlier check."""
    p_only = locate_event(stations, p_times, {})
    used = {
        name: UTCDateTime(value).timestamp for name, value in s_times.items()
    }
    if len(used) < 2:
        return p_only, {}

    location = locate_event(stations, p_times, used)
    for _ in range(2):
        station_by_name = {station_id(s): s for s in stations}
        residuals = {}
        for name in used:
            _km, _p, predicted_s = predict_times(location, station_by_name[name])
            residuals[name] = used[name] - predicted_s
        worst = max(residuals, key=lambda name: abs(residuals[name]))

        if len(used) > 2 and abs(residuals[worst]) > 6.0:
            del used[worst]
            location = locate_event(stations, p_times, used)
        elif abs(residuals[worst]) > 10.0:
            return p_only, {}
        else:
            break

    return location, used


def failed_event(group, start, end, error, index):
    return {
        "id": f"e{index}",
        "located": False,
        "error": error,
        "origin_time": None,
        "time": fmt_time(min(item["t"] for item in group)),
        "lat": None,
        "lon": None,
        "magnitude": None,
        "rms_km": None,
        "n_stations": 0,
        "window_start": str(start),
        "window_end": str(end),
        "trigger_stations": [item["station"] for item in group],
        "stations": [],
        "map_png": None,
        "overview_png": None,
    }


def s_candidates(p_times, triggers, settings):
    """First later trigger per station, far enough after P to be an S phase."""
    by_station = {}
    for item in triggers or []:
        by_station.setdefault(item["station"], []).append(item["t"])

    lower = max(float(settings["min_sp_sec"]), S_MIN_SEPARATION)
    upper = float(settings["max_sp_sec"])
    candidates = {}
    for name, p_time in p_times.items():
        for value in sorted(by_station.get(name, [])):
            separation = value - p_time
            if separation < lower:
                continue
            if separation <= upper:
                candidates[name] = value
            break
    return candidates


def locate_candidate(stations, group, settings, index, triggers=None):
    p_times = {item["station"]: item["t"] for item in group}
    names = set(p_times)
    selected = [station for station in stations if station_id(station) in names]
    first = min(p_times.values())
    last = max(p_times.values())
    start = UTCDateTime(first - float(settings["pre"]))
    tail = float(settings["max_sp_sec"]) + float(settings["post"])
    end = UTCDateTime(min(time.time(), last + tail))
    if end <= start:
        end = start + 60

    try:
        streams, errors = fetch_event_streams(selected, start, end, settings)
        if len(streams) < 3:
            return failed_event(
                group, start, end,
                "Fewer than 3 stations returned data", index,
            )

        usable = [s for s in selected if station_id(s) in streams]
        fs = float(settings["fs"])
        s_times = {
            name: value for name, value in s_candidates(
                p_times, triggers, settings
            ).items()
            if name in {station_id(s) for s in usable}
        }

        location, used_s = choose_location(usable, p_times, s_times)
        residuals = {}
        for station in usable:
            sid = station_id(station)
            _km, predicted, _s = predict_times(location, station)
            residuals[sid] = p_times[sid] - predicted

        results = []
        station_magnitudes = []
        fetch = make_fetch(streams)

        for station in usable:
            sid = station_id(station)
            distance, predicted_p, predicted_s = predict_times(location, station)
            s_time = UTCDateTime(predicted_s)
            sp_diff = predicted_s - predicted_p

            try:
                z, env, max_hor = fetch(
                    None, station, start, end, fs,
                    float(settings["fmin"]), float(settings["fmax"]), 0.0,
                )
                s_index = int((s_time - z.stats.starttime) * fs)
                window = max_hor[max(0, s_index):max(0, s_index) + int(2 * fs)]
                ml = None
                if len(window) > 0:
                    amplitude = float(np.max(window)) * 1e9
                    ml = calculate_ml(amplitude, 0.2, distance, 0.0)
            except Exception:
                ml = None

            if ml is not None:
                station_magnitudes.append(ml)

            residual_km = residuals[sid] * float(settings["vp"])
            if abs(residuals[sid]) <= 2.0:
                quality = "GOOD"
            elif abs(residuals[sid]) <= 5.0:
                quality = "FAIR"
            else:
                quality = "REJECT"

            results.append({
                "name": sid,
                "quality": quality,
                "error": None,
                "p_time": UTCDateTime(p_times[sid]).isoformat(),
                "s_time": s_time.isoformat(),
                "sp_diff": sp_diff,
                "hypo_distance_km": distance,
                "ml": round(ml, 2) if ml is not None else None,
                "residual_km": residual_km,
            })

        vp = float(settings["vp"])
        rms_km = float(np.sqrt(np.mean(
            [(residuals[station_id(s)] * vp) ** 2 for s in usable]
        )))
        magnitude = (
            float(np.median(station_magnitudes)) if station_magnitudes else None
        )
        obs = [
            (s["lat"], s["lon"],
             predict_times(location, s)[0])
            for s in usable
        ]
        method = "P and S-P" if used_s else "P"
        map_png = circle_map_png(
            obs,
            [station_id(s) for s in usable],
            (location["lat"], location["lon"]),
            title=f"Location from {method} arrival times",
        )

        event = {
            "id": f"e{index}",
            "located": True,
            "error": None,
            "origin_time": UTCDateTime(location["origin"]).isoformat(),
            "time": fmt_time(first),
            "lat": location["lat"],
            "lon": location["lon"],
            "magnitude": magnitude,
            "rms_km": round(rms_km, 2),
            "depth_km": round(location["depth"], 1),
            "location_method": method,
            "s_stations": sorted(used_s),
            "n_stations": len(usable),
            "window_start": str(start),
            "window_end": str(end),
            "trigger_stations": [item["station"] for item in group],
            "stations": results,
            "map_png": map_png,
            "overview_png": None,
        }
        try:
            event["overview_png"] = overview_png(
                streams, {"stations": results}, start, end
            )
        except Exception:
            event["overview_png"] = None
        if errors:
            event["data_errors"] = errors
        return event
    except Exception as error:
        return failed_event(group, start, end, str(error), index)


def run_job(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return

    try:
        job["status"] = "running"
        job["phase"] = "detecting"
        job["progress"] = 2
        triggers = []
        total = max(1, len(job["stations"]))

        for index, station in enumerate(job["stations"]):
            if job.get("cancel"):
                break
            sid = station_id(station)
            job["message"] = f"Scanning {sid} ({index + 1}/{total})"
            try:
                scan = detect_station_range(
                    station, job["start"], job["end"], job["settings"],
                    cancel=lambda: job.get("cancel", False),
                )
                triggers.extend(scan["triggers"])
                job["station_stats"].append({
                    "station": sid,
                    "state": "ok" if scan["ok"] else "error",
                    "triggers": len(scan["triggers"]),
                    "chunks": scan["ok"],
                    "failed_chunks": scan["failed"],
                    "error": scan["error"],
                })
            except Exception as error:
                job["station_stats"].append({
                    "station": sid,
                    "state": "error",
                    "triggers": 0,
                    "error": str(error),
                })
            job["triggers"] = sorted(triggers, key=lambda item: item["t"])
            job["progress"] = 5 + 45 * (index + 1) / total

        if job.get("cancel"):
            job["status"] = "cancelled"
            job["phase"] = "cancelled"
            job["message"] = "Cancelled"
            return

        job["phase"] = "grouping"
        job["message"] = "Grouping station triggers"
        job["progress"] = 52
        all_groups = group_triggers(
            job["triggers"],
            float(job["settings"]["coincidence"]),
            int(job["settings"]["min_stations"]),
            suppress=float(job["settings"]["max_sp_sec"]),
        )
        job["candidate_count"] = len(all_groups)
        groups = all_groups[:MAX_EVENTS]

        for index, group in enumerate(groups):
            if job.get("cancel"):
                break
            job["phase"] = "locating"
            job["message"] = f"Locating event {index + 1} of {len(groups)}"
            job["events"].append(
                locate_candidate(
                    job["stations"], group, job["settings"], index,
                    job["triggers"],
                )
            )
            job["progress"] = 55 + 40 * (index + 1) / max(1, len(groups))

        if job.get("cancel"):
            job["status"] = "cancelled"
            job["phase"] = "cancelled"
            job["message"] = "Cancelled"
            return

        job["status"] = "done"
        job["phase"] = "done"
        job["progress"] = 100
        job["message"] = "Completed"
    except Exception as error:
        job["status"] = "failed"
        job["phase"] = "failed"
        job["error"] = str(error)
        job["message"] = str(error)
    finally:
        job["finished"] = time.time()


def build_params(payload, stations, defaults):
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
    if end - start > MAX_RANGE_HOURS * 3600:
        raise ValueError(f"Range is limited to {MAX_RANGE_HOURS} hours")

    names = payload.get("stations") or []
    selected = [s for s in stations if station_id(s) in names or s["sta"] in names]
    if len(selected) < 3:
        raise ValueError("Select at least 3 stations")

    settings = dict(defaults)
    incoming = payload.get("settings") or {}
    for key in settings:
        if key in incoming and incoming[key] not in (None, ""):
            try:
                settings[key] = float(incoming[key])
            except (TypeError, ValueError):
                raise ValueError(f"Invalid value for {key}")

    settings["min_stations"] = int(settings["min_stations"])
    if settings["fs"] < 10 or settings["fs"] > 200:
        raise ValueError("Sample rate must be between 10 and 200 Hz")
    if settings["fmin"] <= 0 or settings["fmax"] <= settings["fmin"]:
        raise ValueError("Frequency band is invalid")
    if settings["fmax"] >= settings["fs"] / 2:
        raise ValueError("High frequency must be below half the sample rate")
    if settings["coincidence"] <= 0:
        raise ValueError("Coincidence window must be positive")
    if settings["min_stations"] < 3:
        raise ValueError("At least 3 stations are required")

    return {"start": start, "end": end, "stations": selected, "settings": settings}


def public_job(job):
    triggers = sorted(job.get("triggers", []), key=lambda item: item["t"], reverse=True)
    events = []
    for event in job.get("events", []):
        item = {key: value for key, value in event.items()
                if key not in ("map_png", "overview_png")}
        item["has_map"] = bool(event.get("map_png"))
        item["has_overview"] = bool(event.get("overview_png"))
        events.append(item)

    station_stats = job.get("station_stats", [])
    return {
        "id": job["id"],
        "status": job.get("status", "queued"),
        "phase": job.get("phase", ""),
        "progress": round(float(job.get("progress", 0)), 1),
        "message": job.get("message", ""),
        "error": job.get("error"),
        "elapsed_s": round((job.get("finished") or time.time()) - job["created_ts"], 1),
        "params": {
            "start": str(job["start"]),
            "end": str(job["end"]),
            "stations": [station_id(station) for station in job["stations"]],
            "settings": job["settings"],
        },
        "counts": {
            "stations": len(station_stats),
            "stations_ok": sum(1 for item in station_stats if item.get("state") == "ok"),
            "triggers": len(triggers),
            "candidates": job.get("candidate_count", 0),
            "events": len(events),
            "located": sum(1 for event in events if event.get("located")),
        },
        "stations": station_stats,
        "triggers": triggers[:200],
        "trigger_count": len(triggers),
        "events": events,
    }


def trim_jobs():
    """Keep only a few old jobs in memory."""
    with JOBS_LOCK:
        if len(JOBS) <= 5:
            return
        finished = [
            (job_id, job) for job_id, job in JOBS.items()
            if job.get("status") != "running"
        ]
        finished.sort(key=lambda item: item[1].get("created_ts", 0))
        for job_id, _ in finished[:len(JOBS) - 5]:
            JOBS.pop(job_id, None)


def page_html(stations, settings):
    station_payload = [
        {
            "name": station_id(station),
            "sta": station["sta"],
            "net": station["net"],
            "lat": station.get("lat"),
            "lon": station.get("lon"),
        }
        for station in stations
    ]
    return (
        PAGE
        .replace("__STATIONS__", json.dumps(station_payload))
        .replace("__DEFAULTS__", json.dumps(settings))
    )


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Historical Earthquake Detection</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  :root { --bg:#f1f4f7; --panel:#fff; --ink:#17212b; --muted:#63707d; --line:#dbe2e8;
          --ok:#1f7a55; --bad:#d6361f; --blue:#2563eb; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui, sans-serif;
         font-variant-numeric:tabular-nums; }
  header { background:var(--panel); border-bottom:1px solid var(--line); padding:14px 20px;
           display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; }
  h1 { margin:0; font-size:19px; }
  h2 { margin:0 0 10px; font-size:15px; }
  a { color:var(--blue); text-decoration:none; }
  .navbtn { display:inline-block; border:1px solid var(--line); border-radius:5px;
            background:var(--panel); color:var(--ink); padding:6px 11px; font-weight:600; }
  .navbtn:hover { background:#f4f6f8; }
  main { max-width:1180px; margin:0 auto; padding:18px; display:grid;
         grid-template-columns:340px minmax(0,1fr); gap:16px; align-items:start; }
  @media (max-width:900px) { main { grid-template-columns:1fr; } }
  .panel, section { background:var(--panel); border:1px solid var(--line); border-radius:7px;
                    padding:14px 16px; }
  #results > * + * { margin-top:16px; }
  .muted { color:var(--muted); font-size:12.5px; }
  label { display:block; font-size:12.5px; color:var(--muted); margin-top:10px; }
  input[type=datetime-local], input[type=number], select {
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
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; }
  .card { border:1px solid var(--line); border-radius:6px; padding:10px 12px; }
  .card span { display:block; color:var(--muted); font-size:12px; }
  .card b { font-size:21px; }
  #map { height:380px; border:1px solid var(--line); border-radius:6px; background:#e3e8ed; }
  .event { border-top:1px solid var(--line); padding:10px 0; }
  .event:first-child { border-top:0; }
  .eventhead { display:flex; gap:9px; align-items:baseline; flex-wrap:wrap; }
  .eventhead b { font-size:15px; }
  .badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11.5px;
           font-weight:600; background:#ddefe6; color:#17603f; }
  .badge.bad { background:#fbeae6; color:#8a2412; }
  .plot { display:block; width:100%; max-width:720px; margin-top:10px; border:1px solid var(--line);
          border-radius:6px; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; margin-top:8px; }
  th, td { text-align:left; padding:4px 6px 4px 0; border-bottom:1px solid #edf1f4;
           white-space:nowrap; }
  th { color:var(--muted); font-weight:500; }
  details { margin-top:8px; }
  summary { cursor:pointer; color:var(--muted); font-size:12.5px; }
  .err { color:var(--bad); }
</style>
</head>
<body>
<header>
  <div><h1>Historical Earthquake Detection</h1>
    <div class="muted">Scan archived FDSN data for triggers, then locate events.</div></div>
  <a class="navbtn" href="/">Live monitor &rarr;</a>
</header>
<main>
  <section class="panel">
    <form id="run-form">
      <label>Start (UTC) <input type="datetime-local" id="start" step="1"></label>
      <label>End (UTC) <input type="datetime-local" id="end" step="1"></label>
      <div class="quick">
        <button type="button" data-hours="1">1 h</button>
        <button type="button" data-hours="6">6 h</button>
        <button type="button" data-hours="24">24 h</button>
      </div>
      <fieldset>
        <legend>Stations</legend>
        <div class="stations" id="station-list"></div>
      </fieldset>
      <details>
        <summary>Advanced settings</summary>
        <label>Sample rate (Hz) <input type="number" id="set-fs" step="1"></label>
        <label>Band low (Hz) <input type="number" id="set-fmin" step="0.1"></label>
        <label>Band high (Hz) <input type="number" id="set-fmax" step="0.1"></label>
        <label>P STA (s) <input type="number" id="set-sta_p" step="0.1"></label>
        <label>P LTA (s) <input type="number" id="set-lta_p" step="0.1"></label>
        <label>Trigger on <input type="number" id="set-thr_on" step="0.1"></label>
        <label>Trigger off <input type="number" id="set-thr_off" step="0.1"></label>
        <label>Coincidence (s) <input type="number" id="set-coincidence" step="1"></label>
        <label>Minimum stations <input type="number" id="set-min_stations" step="1"></label>
        <label>S-P max (s) <input type="number" id="set-max_sp_sec" step="1"></label>
        <label>Vp (km/s) <input type="number" id="set-vp" step="0.1"></label>
        <label>Vs (km/s) <input type="number" id="set-vs" step="0.1"></label>
      </details>
      <button class="primary" id="runbtn" type="submit">Run detection</button>
    </form>
    <div class="muted" id="jobmsg" style="margin-top:10px">Ready.</div>
    <div class="progress"><div id="bar"></div></div>
    <button type="button" id="cancelbtn" style="margin-top:8px;display:none">Cancel</button>
  </section>

  <div id="results">
    <section><div class="cards" id="cards"></div></section>
    <section><h2>Map</h2><div id="map"></div></section>
    <section><h2>Detected events</h2>
      <div class="muted" style="margin-bottom:8px">Located from P arrival times; S times are predicted from the best-fit epicenter. Magnitude is approximate because archived data are used as counts, not response-corrected velocity.</div>
      <div id="events"><div class="muted">No events yet.</div></div></section>
    <section><h2>Station triggers</h2><div id="triggers"><div class="muted">No triggers yet.</div></div></section>
  </div>
</main>

<script>
const STATIONS = __STATIONS__;
const DEFAULTS = __DEFAULTS__;
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => (
  {"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]
));
const num = (value, digits = 2) => value == null ? "-" : Number(value).toFixed(digits);
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

Object.entries(DEFAULTS).forEach(([key, value]) => {
  const input = $("set-" + key);
  if (input) input.value = value;
});

let map = null;
let eventLayer = null;

function initMap() {
  if (typeof L === "undefined") {
    $("map").innerHTML = '<div class="muted" style="padding:24px">The map needs internet access to load Leaflet and the tiles.</div>';
    return;
  }
  map = L.map("map").setView([28.2, 85.3], 7);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors"
  }).addTo(map);
  eventLayer = L.layerGroup().addTo(map);

  const points = [];
  STATIONS.forEach(station => {
    if (station.lat == null || station.lon == null) return;
    points.push([station.lat, station.lon]);
    L.circleMarker([station.lat, station.lon], {
      radius: 6, color: "#1f7a55", fillColor: "#1f7a55", fillOpacity: 0.9
    }).bindPopup(`<b>${esc(station.name)}</b>`).addTo(map);
  });
  if (points.length) map.fitBounds(points, { padding: [40, 40], maxZoom: 8 });
}

function updateMap(events) {
  if (!map || !eventLayer) return;
  eventLayer.clearLayers();
  events.filter(event => event.located).forEach(event => {
    L.circleMarker([event.lat, event.lon], {
      radius: 9, color: "#8a2412", weight: 2, fillColor: "#d6361f", fillOpacity: 0.85
    }).bindPopup(
      `<b>M ${esc(event.magnitude ?? "?")}</b><br>` +
      `${esc(shortTime(event.origin_time))}<br>` +
      `${num(event.lat, 4)}, ${num(event.lon, 4)}<br>` +
      `<span class="muted">RMS ${num(event.rms_km)} km</span>`
    ).addTo(eventLayer);
  });
}

function readSettings() {
  const settings = {};
  Object.keys(DEFAULTS).forEach(key => {
    const input = $("set-" + key);
    if (input) settings[key] = Number(input.value);
  });
  return settings;
}

function renderCards(data) {
  const counts = data.counts;
  $("cards").innerHTML = [
    ["Stations with data", `${counts.stations_ok} / ${counts.stations}`],
    ["Station triggers", counts.triggers],
    ["Candidate events", counts.candidates],
    ["Located events", counts.located],
  ].map(([label, value]) => `<div class="card"><span>${esc(label)}</span><b>${esc(value)}</b></div>`).join("");
}

function eventCard(jobId, event) {
  const head = event.located
    ? `<b>M ${esc(event.magnitude ?? "?")}</b><span>${esc(shortTime(event.origin_time))}</span>
       <span class="badge">LOCATED</span>`
    : `<b>Detection</b><span>${esc(event.time || "-")}</span>
       <span class="badge bad">NOT LOCATED</span>`;

  const summary = event.located
    ? `${num(event.lat, 3)}, ${num(event.lon, 3)} &middot; RMS ${num(event.rms_km)} km
       &middot; ${esc(event.n_stations)} stations`
    : `<span class="err">${esc(event.error || "No location could be computed")}</span>`;

  const images = [];
  if (event.has_map) {
    images.push(`<img class="plot" loading="lazy" src="api/jobs/${jobId}/events/${event.id}/map.png" alt="Event map">`);
  }
  if (event.has_overview) {
    images.push(`<img class="plot" loading="lazy" src="api/jobs/${jobId}/events/${event.id}/overview.png" alt="Event waveforms">`);
  }

  const rows = (event.stations || []).map(station => `
    <tr><td>${esc(station.name)}</td><td>${esc(station.quality || "-")}</td>
      <td>${esc((station.p_time || "-").slice(11, 23))}</td>
      <td>${esc((station.s_time || "-").slice(11, 23))}</td>
      <td>${esc(station.ml ?? "-")}</td>
      <td>${num(station.residual_km, 1)}</td></tr>
  `).join("");

  const table = rows ? `
    <table><thead><tr><th>Station</th><th>Quality</th><th>P</th><th>S (pred)</th><th>ML</th><th>Residual</th></tr></thead>
    <tbody>${rows}</tbody></table>` : "";

  return `<div class="event">
    <div class="eventhead">${head}</div>
    <div class="muted">${summary}</div>
    <div class="muted">Triggered: ${esc((event.trigger_stations || []).join(", "))}</div>
    <details><summary>Map, waveforms and station picks</summary>${images.join("")}${table}</details>
  </div>`;
}

function renderTriggers(data) {
  const triggers = data.triggers || [];
  if (!triggers.length) {
    $("triggers").innerHTML = '<div class="muted">No triggers yet.</div>';
    return;
  }
  $("triggers").innerHTML = `
    <div class="muted">Showing ${triggers.length} of ${data.trigger_count} triggers.</div>
    <table><thead><tr><th>Time (UTC)</th><th>Station</th><th>CFT</th><th>Duration (s)</th></tr></thead>
    <tbody>${triggers.map(trigger => `
      <tr><td>${esc(trigger.time)}</td><td>${esc(trigger.station)}</td>
        <td>${esc(trigger.cft)}</td><td>${esc(trigger.duration)}</td></tr>
    `).join("")}</tbody></table>`;
}

function render(data) {
  const running = data.status === "running";
  $("runbtn").disabled = running;
  $("cancelbtn").style.display = running ? "inline-block" : "none";
  $("bar").style.width = `${data.progress || 0}%`;

  let message = `${data.status} &middot; ${data.phase || ""} &middot; ${data.message || ""}`;
  if (data.error) message = `<span class="err">${esc(data.error)}</span>`;
  $("jobmsg").innerHTML = message;

  renderCards(data);
  $("events").innerHTML = data.events && data.events.length
    ? data.events.map(event => eventCard(data.id, event)).join("")
    : '<div class="muted">No events yet.</div>';
  renderTriggers(data);
  updateMap(data.events || []);
}

async function poll(jobId) {
  try {
    const response = await fetch(`api/jobs/${jobId}`, { cache: "no-store" });
    if (!response.ok) {
      if (response.status === 404) localStorage.removeItem("historical_job");
      throw new Error("HTTP " + response.status);
    }
    const data = await response.json();
    render(data);
    if (data.status === "running") {
      setTimeout(() => poll(jobId), 1500);
    }
  } catch (error) {
    $("jobmsg").innerHTML = `<span class="err">Cannot read job: ${esc(error.message)}</span>`;
  }
}

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
        settings: readSettings(),
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not start job");
    localStorage.setItem("historical_job", data.job_id);
    poll(data.job_id);
  } catch (error) {
    $("runbtn").disabled = false;
    $("jobmsg").innerHTML = `<span class="err">${esc(error.message)}</span>`;
  }
};

$("cancelbtn").onclick = async () => {
  const jobId = localStorage.getItem("historical_job");
  if (!jobId) return;
  await fetch(`api/jobs/${jobId}/cancel`, { method: "POST" });
};

document.querySelectorAll(".quick button").forEach(button => {
  button.onclick = () => setRange(Number(button.dataset.hours));
});

initMap();
setRange(6);

const savedJob = localStorage.getItem("historical_job");
if (savedJob) poll(savedJob);
</script>
</body>
</html>
"""


def create_blueprint(stations, defaults=None):
    station_list = [dict(station) for station in stations]
    base_settings = dict(DEFAULT_SETTINGS)
    if defaults:
        for key in base_settings:
            if key in defaults:
                try:
                    base_settings[key] = float(defaults[key])
                except (TypeError, ValueError):
                    pass
    base_settings["min_stations"] = int(base_settings["min_stations"])

    bp = Blueprint("historical", __name__, url_prefix="/historical")

    @bp.route("/")
    def historical_page():
        return Response(page_html(station_list, base_settings), mimetype="text/html")

    @bp.route("/api/run", methods=["POST"])
    def api_run():
        payload = request.get_json(silent=True) or {}
        try:
            params = build_params(payload, station_list, base_settings)
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
                "progress": 0,
                "message": "Queued",
                "error": None,
                "created": fmt_time(UTCDateTime()),
                "created_ts": time.time(),
                "finished": None,
                "start": params["start"],
                "end": params["end"],
                "stations": params["stations"],
                "settings": params["settings"],
                "triggers": [],
                "events": [],
                "station_stats": [],
                "candidate_count": 0,
            }

        trim_jobs()
        threading.Thread(target=run_job, args=(job_id,), daemon=True).start()
        return jsonify(job_id=job_id)

    @bp.route("/api/jobs/<job_id>")
    def api_job(job_id):
        job = JOBS.get(job_id)
        if job is None:
            return jsonify(error="Job not found"), 404
        return jsonify(public_job(job))

    @bp.route("/api/jobs/<job_id>/cancel", methods=["POST"])
    def api_cancel(job_id):
        job = JOBS.get(job_id)
        if job is None:
            return jsonify(error="Job not found"), 404
        job["cancel"] = True
        return jsonify(ok=True)

    @bp.route("/api/jobs/<job_id>/events/<event_id>/<kind>.png")
    def event_image(job_id, event_id, kind):
        job = JOBS.get(job_id)
        if job is None:
            return Response("Job not found", status=404, mimetype="text/plain")
        event = next(
            (item for item in job.get("events", []) if item.get("id") == event_id),
            None,
        )
        if event is None:
            return Response("Event not found", status=404, mimetype="text/plain")

        key = "map_png" if kind == "map" else "overview_png" if kind == "overview" else None
        png = event.get(key) if key else None
        if not png:
            return Response("No image for this event", status=404, mimetype="text/plain")
        return Response(base64.b64decode(png), mimetype="image/png",
                        headers={"Cache-Control": "no-store"})

    return bp
