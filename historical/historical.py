import base64
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from obspy.taup import TauPyModel
from scipy.signal import hilbert

from realtime.magnitude import calculate_ml
from past.plotting import circle_map_png


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

# A trigger shorter than this is treated as a spike, not a phase arrival.
MIN_TRIGGER_SEC = 0.5
# A station joins an event only if its P pick is this close to the prediction.
P_ASSOC_SEC = 5.0
# Search half-width when re-picking P and S near the predicted arrival.
P_PICK_WINDOW_SEC = 12.0
S_PICK_WINDOW_SEC = 10.0
# S onsets farther than this from the final S prediction are not trusted.
S_ASSOC_SEC = 3.0
# Candidate S onsets are collected over a wider window because the P-only
# location still has an unconstrained depth (and therefore S-P error).
S_CANDIDATE_WINDOW_SEC = S_PICK_WINDOW_SEC + 5.0
# A located event whose P picks still scatter more than this is not trusted.
MAX_P_RMS_SEC = 4.0

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

    # Keep enough data before each chunk so the LTA (and therefore the CFT)
    # is already warm when the chunk starts.  Without this the filter and
    # STA/LTA start cold at every chunk boundary and can fire spuriously.
    pad = max(DETECT_PAD, lta + sta + 2.0)

    position = start
    while position < end:
        if cancel is not None and cancel():
            break

        chunk_end = min(position + CHUNK_SEC, end)
        wave_start = position - pad
        try:
            stream = client.get_waveforms(
                station["net"],
                station["sta"],
                "*",
                "?H?",
                wave_start,
                chunk_end + pad,
            )
            if len(stream) == 0:
                raise ValueError("No data")

            stream = preprocess(stream, settings)
            z = vertical_trace(stream)
            if z is None:
                raise ValueError("No vertical component")

            z.trim(wave_start, chunk_end, pad=True, fill_value=0)
            if z.stats.npts < int((lta + sta + 3) * fs):
                raise ValueError("Not enough samples in chunk")

            cft = classic_sta_lta(z.data.astype(float), int(sta * fs), int(lta * fs))
            cft = np.nan_to_num(cft, nan=0.0, posinf=0.0, neginf=0.0)

            for onset, offset in trigger_onset(cft, float(settings["thr_on"]), float(settings["thr_off"])):
                trigger_time = z.stats.starttime + onset / fs
                if trigger_time < position or trigger_time >= chunk_end:
                    continue
                duration = (offset - onset) / fs
                if duration < MIN_TRIGGER_SEC:
                    continue
                triggers.append({
                    "station": station_id(station),
                    "t": float(trigger_time.timestamp),
                    "time": fmt_time(trigger_time),
                    "cft": round(float(cft[onset]), 2),
                    "duration": round(float(duration), 2),
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
                           color="#1F7A55",
                           linestyle="-" if info.get("s_picked") else ":",
                           linewidth=1)

    axes[-1].set_xlabel("Seconds from event window start")
    fig.suptitle(
        "Event waveforms, each trace normalized (red = P pick, green = S pick)\n"
        "Dotted green is a predicted S where no S onset was picked",
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


def distance_from_sp(sp_seconds, depth):
    """Epicentral distance whose predicted S-P time matches ``sp_seconds``.

    This is independent of the trial epicenter, so circles drawn with it make
    a genuine consistency check of the located epicenter.
    """
    distances, p_curve, s_curve = travel_curves(depth)
    sp_curve = s_curve - p_curve
    order = np.argsort(sp_curve)
    return float(np.interp(
        float(sp_seconds), sp_curve[order], distances[order],
    ))


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


def sta_lta_cft(data, fs, sta, lta):
    """STA/LTA characteristic function, or None when the window is too short."""
    values = np.asarray(data, dtype=float)
    nsta = max(1, int(round(sta * fs)))
    nlta = max(nsta + 1, int(round(lta * fs)))
    if len(values) <= nlta + 1:
        return None
    cft = classic_sta_lta(values, nsta, nlta)
    return np.nan_to_num(cft, nan=0.0, posinf=0.0, neginf=0.0)


def aic_pick(data, fs, index, search_before=1.5, search_after=0.75, max_shift=1.0):
    """Refine an STA/LTA onset sample with the Akaike information criterion."""
    values = np.asarray(data, dtype=float)
    count = len(values)
    index = int(index)
    if count < 12 or not np.isfinite(fs) or fs <= 0:
        return index

    start = max(1, index - int(round(search_before * fs)))
    stop = min(count - 2, index + int(round(search_after * fs)))
    if stop - start < 12:
        return index

    segment = values[start:stop + 1]
    segment = segment - np.mean(segment)
    length = len(segment)

    cumulative = np.cumsum(segment)
    cumulative_sq = np.cumsum(segment * segment)

    k = np.arange(1, length - 1)
    sum_left = cumulative[k - 1]
    sum_right = cumulative[-1] - sum_left
    sq_left = cumulative_sq[k - 1]
    sq_right = cumulative_sq[-1] - sq_left

    var_left = sq_left / k - (sum_left / k) ** 2
    right_length = length - k
    var_right = sq_right / right_length - (sum_right / right_length) ** 2

    valid = (var_left > 0) & (var_right > 0)
    if not np.any(valid):
        return index

    aic = np.full(k.shape, np.inf, dtype=float)
    aic[valid] = (
        k[valid] * np.log(var_left[valid])
        + right_length[valid] * np.log(var_right[valid])
    )

    best = int(np.argmin(aic))
    pick = start + int(k[best])
    edge = max(2, int(0.05 * length))
    if best < edge or best >= len(k) - edge:
        return index
    if abs(pick - index) > max_shift * fs:
        return index
    return pick


def pick_phase(data, starttime, target, fs, settings, phase, search):
    """Pick the onset nearest a predicted arrival and refine it with AIC."""
    phase = phase.upper()
    if phase == "P":
        sta = float(settings["sta_p"])
        lta = float(settings["lta_p"])
        thr_on = float(settings["thr_on"])
        thr_off = float(settings["thr_off"])
        before, after, max_shift = 1.5, 0.75, 1.0
    else:
        sta = float(settings["sta_s"])
        lta = float(settings["lta_s"])
        thr_on = float(settings["thr_on_s"])
        thr_off = float(settings["thr_off_s"])
        before, after, max_shift = 1.5, 3.0, 1.5

    cft = sta_lta_cft(data, fs, sta, lta)
    if cft is None:
        return None

    target_ts = float(target)
    best = None
    for onset, offset in trigger_onset(cft, thr_on, thr_off):
        onset_time = starttime + onset / fs
        delta = abs(float(onset_time.timestamp) - target_ts)
        if delta > search:
            continue
        if (offset - onset) / fs < MIN_TRIGGER_SEC:
            continue
        if best is None or delta < best[0]:
            best = (delta, int(onset))

    if best is None:
        return None

    index = aic_pick(
        data, fs, best[1],
        search_before=before, search_after=after, max_shift=max_shift,
    )
    return starttime + index / fs


def robust_locate(stations, p_times, tolerance=P_ASSOC_SEC):
    """Locate from the most mutually consistent subset of P picks.

    Every subset of stations is located and scored by how many of the
    remaining P picks it predicts within ``tolerance`` seconds.  A single
    false trigger (for example an early noise burst on one station) then
    cannot drag the epicenter away from the picks that agree.
    """
    station_by_name = {station_id(station): station for station in stations}
    active = {
        name: float(value) for name, value in p_times.items()
        if name in station_by_name
    }
    names = sorted(active)
    if len(names) < 3:
        return None, {}, {}

    best = None
    for size in range(len(names), 2, -1):
        for combo in combinations(names, size):
            subset = [station_by_name[name] for name in combo]
            picks = {name: active[name] for name in combo}
            location = locate_event(subset, picks, {})
            residuals = {
                name: active[name] - predict_times(location, station_by_name[name])[1]
                for name in names
            }
            inliers = {
                name: active[name] for name in names
                if abs(residuals[name]) <= tolerance
            }
            if len(inliers) < size:
                continue
            rms = float(np.sqrt(np.mean(
                [residuals[name] ** 2 for name in inliers]
            )))
            score = (len(inliers), -rms)
            if best is None or score > best[0]:
                best = (score, location, inliers, residuals)
        if best is not None and best[0][0] >= 4:
            # Four or more agreeing stations make the solution redundant
            # enough that smaller subsets do not need to be tried.
            break

    if best is None:
        return None, {}, {}

    _score, location, inliers, residuals = best
    inlier_residuals = {
        name: residuals[name] for name in inliers
    }
    return location, inliers, inlier_residuals


def refine_p_picks(stations, location, streams, settings, search):
    """Re-pick P on every station near the arrival predicted by ``location``."""
    fs = float(settings["fs"])
    refined = {}
    for station in stations:
        sid = station_id(station)
        stream = streams.get(sid)
        if stream is None:
            continue
        z = vertical_trace(stream)
        if z is None or z.stats.npts < 10:
            continue
        _km, predicted_p, _s = predict_times(location, station)
        picked = pick_phase(
            z.data.astype(float), z.stats.starttime, predicted_p,
            fs, settings, "P", search,
        )
        if picked is not None:
            refined[sid] = float(picked.timestamp)
    return refined


def s_onsets(envelope, starttime, p_time, predicted_s, fs, settings):
    """All STA/LTA onsets that could be an S arrival for one station.

    Returns ``(timestamp, sample_index)`` pairs, limited to triggers between
    the minimum and maximum plausible S-P time and near the S prediction.
    """
    cft = sta_lta_cft(
        envelope, fs, float(settings["sta_s"]), float(settings["lta_s"]),
    )
    if cft is None:
        return []

    lower = float(settings["min_sp_sec"])
    upper = float(settings["max_sp_sec"])
    onsets = []
    for onset, offset in trigger_onset(
        cft, float(settings["thr_on_s"]), float(settings["thr_off_s"]),
    ):
        if (offset - onset) / fs < MIN_TRIGGER_SEC:
            continue
        timestamp = float((starttime + onset / fs).timestamp)
        if not lower <= timestamp - p_time <= upper:
            continue
        if abs(timestamp - predicted_s) > S_CANDIDATE_WINDOW_SEC:
            continue
        onsets.append((timestamp, int(onset)))
    return onsets


def pick_s_picks(stations, p_times, components, settings, location=None):
    """Pick S arrivals and refine the location with the consistent S-P times.

    S is often emergent and the P-only location still has an unconstrained
    depth, so the single onset nearest one prediction can easily land on the P
    coda.  This alternates between choosing the onset nearest the current
    prediction and relocating with P + S-P times, then keeps only the S picks
    that stay within ``S_ASSOC_SEC`` of the converged prediction.

    Returns ``(s_picks, location, used_s)``.
    """
    fs = float(settings["fs"])
    station_by_name = {station_id(station): station for station in stations}
    active = [s for s in stations if station_id(s) in p_times]
    if location is None:
        location = locate_event(active, p_times, {})
    if location is None or len(active) < 3:
        return {}, location, set()

    candidates = {}
    for station in active:
        sid = station_id(station)
        if components.get(sid) is None:
            continue
        z, envelope, _max_hor = components[sid]
        _km, _predicted_p, predicted_s = predict_times(location, station)
        onsets = s_onsets(
            envelope, z.stats.starttime, p_times[sid], predicted_s, fs, settings,
        )
        if onsets:
            candidates[sid] = onsets

    if len(candidates) < 2:
        return {}, location, set()

    def nearest(loc):
        chosen = {}
        for sid, onsets in candidates.items():
            _km, _p, predicted_s = predict_times(loc, station_by_name[sid])
            chosen[sid] = min(onsets, key=lambda item: abs(item[0] - predicted_s))
        return chosen

    def fit(seed):
        chosen = dict(seed)
        best = None
        for _ in range(4):
            s_times = {sid: item[0] for sid, item in chosen.items()}
            trial = locate_event(active, p_times, s_times)
            residuals = {
                sid: s_times[sid] - predict_times(trial, station_by_name[sid])[2]
                for sid in s_times
            }
            p_residuals = [
                p_times[sid] - predict_times(trial, station)[1]
                for sid, station in station_by_name.items()
                if sid in p_times
            ]
            rms = float(np.sqrt(np.mean([value ** 2 for value in residuals.values()])))
            p_rms = float(np.sqrt(np.mean(np.square(p_residuals))))
            inlier_count = sum(1 for value in residuals.values() if abs(value) <= S_ASSOC_SEC)
            score = (inlier_count, -(rms + p_rms))
            if best is None or score > best[0]:
                best = (score, trial, dict(chosen), residuals)
            if inlier_count < 2:
                break
            following = nearest(trial)
            if {sid: item[0] for sid, item in following.items()} == s_times:
                break
            chosen = following
        return best

    # The nearest trigger can be P coda when two onsets are nearly equidistant
    # from an uncertain prediction, so also start from each alternative trigger
    # and keep the solution that explains the most arrivals with the least
    # scatter.
    seeds = [nearest(location)]
    for sid, onsets in candidates.items():
        if len(onsets) < 2:
            continue
        for candidate in onsets:
            seed = nearest(location)
            seed[sid] = candidate
            seeds.append(seed)

    best = None
    for seed in seeds:
        result = fit(seed)
        if result is not None and (best is None or result[0] > best[0]):
            best = result

    if best is None:
        return {}, location, set()

    _score, trial, chosen, residuals = best
    inliers = {
        sid: item for sid, item in chosen.items()
        if abs(residuals[sid]) <= S_ASSOC_SEC
    }
    if len(inliers) < 2:
        return {}, location, set()

    # AIC-refine the accepted triggers and relocate once with the sharpened
    # onsets; onsets that no longer fit are dropped.
    refined = {}
    for sid, (_timestamp, index) in inliers.items():
        z, envelope, _max_hor = components[sid]
        pick = aic_pick(envelope, fs, index, 1.5, 3.0, 1.5)
        refined[sid] = float((z.stats.starttime + pick / fs).timestamp)

    located = locate_event(active, p_times, refined)
    refined_residuals = {
        sid: refined[sid] - predict_times(located, station_by_name[sid])[2]
        for sid in refined
    }
    used = {
        sid: value for sid, value in refined.items()
        if abs(refined_residuals[sid]) <= S_ASSOC_SEC
    }
    if len(used) < 2:
        return {}, location, set()
    return used, located, set(used)



def locate_candidate(stations, group, settings, index, triggers=None):
    del triggers  # S is picked from the waveforms, not from detection triggers.
    raw_p_times = {item["station"]: float(item["t"]) for item in group}
    names = set(raw_p_times)
    selected = [station for station in stations if station_id(station) in names]
    first = min(raw_p_times.values())
    last = max(raw_p_times.values())
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

        # Locate from the detection triggers first, dropping gross outliers,
        # then re-pick every P on the waveform near the predicted arrival.
        location, p_times, _residuals = robust_locate(usable, raw_p_times)
        if location is None or len(p_times) < 3:
            return failed_event(
                group, start, end,
                "Fewer than 3 consistent P picks", index,
            )

        for search in (P_PICK_WINDOW_SEC, 6.0):
            refined = refine_p_picks(usable, location, streams, settings, search)
            if len(refined) < 3:
                break
            p_times = refined
            location, p_times, _residuals = robust_locate(usable, p_times)
            if location is None or len(p_times) < 3:
                return failed_event(
                    group, start, end,
                    "Fewer than 3 consistent P picks", index,
                )

        p_rms_sec = float(np.sqrt(np.mean(
            [value ** 2 for value in _residuals.values()]
        )))
        if p_rms_sec > MAX_P_RMS_SEC:
            return failed_event(
                group, start, end,
                f"P picks are inconsistent (RMS {p_rms_sec:.1f} s)", index,
            )

        fetch = make_fetch(streams)
        components = {}
        for station in usable:
            sid = station_id(station)
            try:
                components[sid] = fetch(
                    None, station, start, end, fs,
                    float(settings["fmin"]), float(settings["fmax"]), 0.0,
                )
            except Exception:
                components[sid] = None

        consistent = [s for s in usable if station_id(s) in p_times]
        s_picks, location, used_s = pick_s_picks(
            consistent, p_times, components, settings, location,
        )

        residuals = {}
        for station in usable:
            sid = station_id(station)
            if sid in p_times:
                _km, predicted_p, _predicted_s = predict_times(location, station)
                residuals[sid] = p_times[sid] - predicted_p

        results = []
        station_magnitudes = []
        vp = float(settings["vp"])
        for station in usable:
            sid = station_id(station)
            distance, predicted_p, predicted_s = predict_times(location, station)
            s_pred_time = UTCDateTime(predicted_s)
            s_value = s_picks.get(sid, predicted_s)
            s_time = UTCDateTime(s_value)
            if sid in p_times:
                sp_diff = s_value - p_times[sid]
            else:
                sp_diff = predicted_s - predicted_p

            ml = None
            if components.get(sid) is not None:
                try:
                    z, _env, max_hor = components[sid]
                    s_index = int((s_time - z.stats.starttime) * fs)
                    window = max_hor[max(0, s_index):max(0, s_index) + int(2 * fs)]
                    if len(window) > 0:
                        amplitude = float(np.max(window)) * 1e9
                        ml = calculate_ml(amplitude, 0.2, distance, 0.0)
                except Exception:
                    ml = None

            if ml is not None:
                station_magnitudes.append(ml)

            if sid in residuals:
                residual_km = residuals[sid] * vp
                if abs(residuals[sid]) <= 2.0:
                    quality = "GOOD"
                elif abs(residuals[sid]) <= 5.0:
                    quality = "FAIR"
                else:
                    quality = "REJECT"
            else:
                residual_km = None
                quality = "REJECT"

            results.append({
                "name": sid,
                "quality": quality,
                "error": None if sid in p_times else "No consistent P pick",
                "p_time": (
                    UTCDateTime(p_times[sid]).isoformat() if sid in p_times else None
                ),
                "trigger_time": (
                    fmt_time(raw_p_times[sid]) if sid in raw_p_times else None
                ),
                "s_time": s_time.isoformat(),
                "s_pred_time": s_pred_time.isoformat(),
                "s_picked": sid in s_picks,
                "sp_diff": sp_diff,
                "hypo_distance_km": distance,
                "ml": round(ml, 2) if ml is not None else None,
                "residual_km": residual_km,
            })

        rms_km = float(np.sqrt(np.mean([
            (residuals[station_id(s)] * vp) ** 2 for s in consistent
        ])))
        magnitude = (
            float(np.median(station_magnitudes)) if station_magnitudes else None
        )
        method = "P and S-P" if used_s else "P"
        depth = float(location["depth"])
        sp_obs = []
        sp_names = []
        sp_labels = []
        for station in consistent:
            sid = station_id(station)
            if sid not in used_s:
                continue
            sp = float(s_picks[sid] - p_times[sid])
            km = distance_from_sp(sp, depth)
            sp_obs.append((station["lat"], station["lon"], km))
            sp_names.append(sid)
            sp_labels.append(f"{sid} S-P {sp:.1f}s ({km:.1f} km)")

        if len(sp_obs) >= 2:
            map_png = circle_map_png(
                sp_obs,
                sp_names,
                (location["lat"], location["lon"]),
                title=(
                    "Epicentral distance from observed S-P times "
                    f"(depth {depth:.0f} km)"
                ),
                legend_labels=sp_labels,
            )
        else:
            obs = [
                (s["lat"], s["lon"], predict_times(location, s)[0])
                for s in consistent
            ]
            map_png = circle_map_png(
                obs,
                [station_id(s) for s in consistent],
                (location["lat"], location["lon"]),
                title=f"Station distances from the {method} solution",
            )

        event = {
            "id": f"e{index}",
            "located": True,
            "error": None,
            "origin_time": UTCDateTime(location["origin"]).isoformat(),
            "time": fmt_time(min(p_times.values())),
            "lat": location["lat"],
            "lon": location["lon"],
            "magnitude": magnitude,
            "rms_km": round(rms_km, 2),
            "depth_km": round(location["depth"], 1),
            "location_method": method,
            "s_stations": sorted(used_s),
            "n_stations": len(consistent),
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

