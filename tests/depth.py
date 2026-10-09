
import threading
import time
import numpy as np

from obspy import UTCDateTime, Stream
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from scipy.optimize import least_squares
from scipy.stats import kurtosis
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient

SEEDLINK_HOST = "ring.wscada.net:18000"

fs = 20.0
fmin = 1.0
fmax = 8.0
buffer = 60
scan = 2

sta_p = 1.0
lta_p = 10.0
p_min_threshold = 2.3
p_max_threshold = 8.0
p_percentile = 99.0
p_off_ratio = 0.4

sta_s = 1.0
lta_s = 8.0
s_min_threshold = 1.7
s_max_threshold = 6.0
s_percentile = 99.0
s_off_ratio = 0.4

background_exclude = 10.0
threshold_update_interval = 30.0

min_sp = 2.0
max_sp = 120.0

vp = 6.0
vs = 3.5

min_st = 3
event_origin_tolerance = 30.0
max_location_rms = 3.0

stations = [
    {"sta": "EQM06", "lat": 29.540123, "lon": 82.081913, "net": "NP", "cha": "EHZ", "loc": "*"},
    {"sta": "EQM08", "lat": 27.831, "lon": 86.65, "net": "NP", "cha": "EHZ", "loc": "*"},
    {"sta": "EQM11", "lat": 29.2859556, "lon": 81.2741972, "net": "NP", "cha": "EHZ", "loc": ""},
]

buffers = {}
buffer_lock = threading.Lock()
station_state = {}
last_location_time = None

for station in stations:
    name = f"{station['net']}.{station['sta']}"
    buffers[name] = Stream()
    station_state[name] = {
        "p_time": None,
        "p_trigger": None,
        "s_time": None,
        "s_trigger": None,
        "quality": None,
        "distance_km": None,
        "threshold_last_update": None,
        "p_threshold": p_min_threshold,
        "p_off_threshold": max(1.2, p_min_threshold * p_off_ratio),
        "p_background_level": 1.0,
        "s_threshold": s_min_threshold,
        "s_off_threshold": max(1.1, s_min_threshold * s_off_ratio),
        "s_background_level": 1.0,
    }


class LiveSeedLinkClient(EasySeedLinkClient):
    def on_data(self, trace):
        name = f"{trace.stats.network}.{trace.stats.station}"

        if name not in buffers:
            return

        try:
            tr = trace.copy()

            with buffer_lock:
                buffers[name] += tr
                buffers[name].merge(method=1, fill_value="interpolate")

                if len(buffers[name]) > 0:
                    latest_end = max(t.stats.endtime for t in buffers[name])
                    cutoff = latest_end - buffer
                    buffers[name].trim(
                        starttime=cutoff,
                        endtime=latest_end,
                        pad=False
                    )
                for buffered_trace in buffers[name]:
                    buffered_trace.stats.processing = []
                        
        except Exception as e:
            print(f"[SeedLink] Error processing {name}: {e}")

    def on_seedlink_error(self, err):
        print(f"[SeedLink] Error: {err}")

    def on_seedlink_connection_error(self):
        print("[SeedLink] Connection error")


def prepare_station_stream(station_name, fs=20.0, fmin=1.0, fmax=8.0):
    with buffer_lock:
        if len(buffers[station_name]) == 0:
            return None, None

        st = buffers[station_name].copy()

    z_stream = st.select(component="Z")

    if len(z_stream) == 0:
        return None, None

    north = st.select(component="N")
    east = st.select(component="E")
    horizontals = []

    if len(north) > 0:
        horizontals.append(north[0])

    if len(east) > 0:
        horizontals.append(east[0])

    if len(horizontals) < 2:
        one = st.select(component="1")
        two = st.select(component="2")
        horizontals = []

        if len(one) > 0:
            horizontals.append(one[0])

        if len(two) > 0:
            horizontals.append(two[0])

    if len(horizontals) < 2:
        return None, None

    z = z_stream[0].copy()
    h1 = horizontals[0].copy()
    h2 = horizontals[1].copy()

    start = max(
        z.stats.starttime,
        h1.stats.starttime,
        h2.stats.starttime
    )

    end = min(
        z.stats.endtime,
        h1.stats.endtime,
        h2.stats.endtime
    )

    if end <= start:
        return None, None

    z.trim(start, end)
    h1.trim(start, end)
    h2.trim(start, end)

    for tr in [z, h1, h2]:
        tr.stats.processing = []
        tr.interpolate(sampling_rate=fs, method="linear")

    n = min(len(z.data), len(h1.data), len(h2.data))

    if n < int(15 * fs):
        return None, None

    z.data = z.data[:n]
    h1.data = h1.data[:n]
    h2.data = h2.data[:n]

    for tr in [z, h1, h2]:
        tr.detrend("linear")
        tr.detrend("demean")
        tr.taper(max_percentage=0.05, type="hann")
        tr.filter(
            "bandpass",
            freqmin=fmin,
            freqmax=fmax,
            corners=4,
            zerophase=False
        )

    h1_data = h1.data.astype(float)
    h2_data = h2.data.astype(float)
    env = np.sqrt(h1_data ** 2 + h2_data ** 2)

    return z, env


def calculate_adaptive_threshold(
    cft,
    fs,
    percentile=99.0,
    min_thresh=2.0,
    max_thresh=8.0,
    exclude_tail_sec=10.0
):
    if len(cft) == 0:
        return min_thresh, 1.0

    exclude_samples = int(exclude_tail_sec * fs)

    if len(cft) > exclude_samples:
        background = cft[:-exclude_samples]
    else:
        background = cft

    startup_samples = int(10.0 * fs)

    if len(background) > startup_samples:
        background = background[startup_samples:]

    background = background[np.isfinite(background)]
    background = background[background > 0]

    if len(background) < 50:
        return min_thresh, 1.0

    background_level = float(np.median(background))
    percentile_value = float(np.percentile(background, percentile))

    adaptive_thresh = percentile_value * 1.2
    adaptive_thresh = float(
        np.clip(adaptive_thresh, min_thresh, max_thresh)
    )

    return adaptive_thresh, background_level


def update_station_thresholds(station_name, z, env, fs):
    state = station_state[station_name]
    now = z.stats.endtime

    if state["p_time"] is not None:
        return

    if state["threshold_last_update"] is not None:
        elapsed = now - state["threshold_last_update"]

        if elapsed < threshold_update_interval:
            return

    nsta_p = int(sta_p * fs)
    nlta_p = int(lta_p * fs)

    if len(z.data) > nlta_p + 10:
        p_cft = recursive_sta_lta(z.data, nsta_p, nlta_p)

        p_threshold, p_background = calculate_adaptive_threshold(
            p_cft,
            fs,
            p_percentile,
            p_min_threshold,
            p_max_threshold,
            background_exclude
        )

        state["p_threshold"] = p_threshold
        state["p_off_threshold"] = max(
            1.2,
            p_threshold * p_off_ratio
        )
        state["p_background_level"] = p_background

    nsta_s = int(sta_s * fs)
    nlta_s = int(lta_s * fs)

    if len(env) > nlta_s + 10:
        s_cft = recursive_sta_lta(env, nsta_s, nlta_s)

        s_threshold, s_background = calculate_adaptive_threshold(
            s_cft,
            fs,
            s_percentile,
            s_min_threshold,
            s_max_threshold,
            background_exclude
        )

        state["s_threshold"] = s_threshold
        state["s_off_threshold"] = max(
            1.1,
            s_threshold * s_off_ratio
        )
        state["s_background_level"] = s_background

    state["threshold_last_update"] = now

    print(
        f"[THRESHOLD] {station_name} "
        f"P={state['p_threshold']:.2f}/{state['p_off_threshold']:.2f} "
        f"S={state['s_threshold']:.2f}/{state['s_off_threshold']:.2f}"
    )


def aic_pick(trace_data, fs, approximate_idx, search_before=2.0, search_after=4.0):
    n = len(trace_data)

    i0 = max(1, approximate_idx - int(search_before * fs))
    i1 = min(n - 2, approximate_idx + int(search_after * fs))

    if i1 <= i0 + 10:
        return approximate_idx

    x = np.asarray(trace_data[i0:i1], dtype=float)
    aic = np.full(len(x), np.nan)

    for k in range(1, len(x) - 1):
        var1 = np.var(x[:k])
        var2 = np.var(x[k:])

        if var1 <= 0 or var2 <= 0:
            continue

        aic[k] = (
            k * np.log10(var1)
            + (len(x) - k) * np.log10(var2)
        )

    valid = np.isfinite(aic)

    if not np.any(valid):
        return approximate_idx

    local_idx = int(np.nanargmin(aic))

    return int(i0 + local_idx)


def pick_p(
    z,
    fs,
    search_start,
    sta=1.0,
    lta=10.0,
    thr_on=3.5,
    thr_off=1.5
):
    nsta = max(1, int(sta * fs))
    nlta = max(nsta + 1, int(lta * fs))

    cft = recursive_sta_lta(
        z.data.astype(np.float64),
        nsta,
        nlta
    )

    finite_cft = cft[np.isfinite(cft)]

    if len(finite_cft) == 0:
        print("[P DEBUG] No valid STA/LTA values")
        return None

    max_cft = float(np.max(finite_cft))

    trigger_count = 0
    recent_trigger_count = 0

    triggers = trigger_onset(
        cft,
        thr_on,
        thr_off
    )

    trigger_count = len(triggers)

    for on, _off in triggers:
        trigger_time = z.stats.starttime + on / fs

        if trigger_time < search_start:
            continue

        recent_trigger_count += 1

        p_idx = int(on)

        p_idx_aic = aic_pick(
            z.data,
            fs,
            p_idx,
            search_before=2.0,
            search_after=4.0
        )

        p_time = z.stats.starttime + p_idx_aic / fs

        return {
            "time": p_time,
            "trigger": float(cft[p_idx]),
            "sta_time": trigger_time,
            "cft": cft,
        }

    print(
        f"[P DEBUG] {z.stats.network}.{z.stats.station} "
        f"max_STA_LTA={max_cft:.2f} "
        f"on={thr_on:.2f} "
        f"off={thr_off:.2f} "
        f"all_triggers={trigger_count} "
        f"recent_triggers={recent_trigger_count}"
    )

    return None

def pick_s(
    env,
    fs,
    t0,
    p_time,
    sta=1.0,
    lta=8.0,
    min_sp_sec=2.0,
    max_sp_sec=120.0,
    thr_on=1.7,
    thr_off=1.0
):
    nsta = max(1, int(sta * fs))
    nlta = max(nsta + 1, int(lta * fs))

    cft = recursive_sta_lta(
        env.astype(np.float64),
        nsta,
        nlta
    )

    p_idx = int((p_time - t0) * fs)

    i0 = p_idx + int(min_sp_sec * fs)
    i1 = min(
        p_idx + int(max_sp_sec * fs),
        len(cft) - 1
    )

    if i0 >= i1:
        return None

    search_cft = cft[i0:i1 + 1]

    finite_cft = search_cft[np.isfinite(search_cft)]

    if len(finite_cft) == 0:
        return None

    max_s_cft = float(np.max(finite_cft))

    triggers = trigger_onset(
        cft,
        thr_on,
        thr_off
    )

    triggers_in_window = 0

    for on, _off in triggers:

        if i0 <= on <= i1:

            triggers_in_window += 1

            s_idx = int(on)

            s_idx_aic = aic_pick(
                env,
                fs,
                s_idx,
                search_before=1.5,
                search_after=4.0
            )

            s_time = t0 + s_idx_aic / fs

            if s_time <= p_time:
                continue

            sp = s_time - p_time

            if sp < min_sp_sec or sp > max_sp_sec:
                continue

            return {
                "time": s_time,
                "trigger": float(cft[s_idx]),
                "sta_time": t0 + s_idx / fs,
                "cft": cft,
            }

    print(
        f"[S DEBUG] max_S_STA_LTA={max_s_cft:.2f} "
        f"threshold={thr_on:.2f} "
        f"triggers_in_window={triggers_in_window}"
    )

    return None

def calculate_quality_metrics(
    z,
    env,
    fs,
    p_time,
    s_time,
    p_sta_trigger,
    s_sta_trigger,
    p_threshold,
    s_threshold,
    max_sp=max_sp
):
    t0 = z.stats.starttime

    p_idx = int((p_time - t0) * fs)
    s_idx = int((s_time - t0) * fs)

    window = int(5.0 * fs)

    p0 = max(0, p_idx - window // 2)
    p1 = min(len(z.data), p_idx + window // 2)

    s0 = max(0, s_idx - window // 2)
    s1 = min(len(env), s_idx + window // 2)

    pz = z.data[p0:p1].astype(float)
    sz = z.data[s0:s1].astype(float)
    ph = env[p0:p1].astype(float)
    sh = env[s0:s1].astype(float)

    if len(pz) < 10 or len(sz) < 10:
        raise RuntimeError("Not enough samples for quality analysis")

    p_rms_z = float(np.sqrt(np.mean(pz ** 2)))
    p_rms_h = float(np.sqrt(np.mean(ph ** 2)))
    s_rms_z = float(np.sqrt(np.mean(sz ** 2)))
    s_rms_h = float(np.sqrt(np.mean(sh ** 2)))

    eps = 1e-12

    p_hv = p_rms_h / (p_rms_z + eps)
    s_hv = s_rms_h / (s_rms_z + eps)

    p_kurt = float(kurtosis(pz, fisher=True, bias=False))
    s_kurt = float(kurtosis(sz, fisher=True, bias=False))

    def outlier_ratio(data):
        median = np.median(data)
        mad = np.median(np.abs(data - median))

        if mad <= eps:
            return 0.0

        zscore = np.abs(data - median) / mad

        return float(np.mean(zscore > 4.45))

    p_outliers = outlier_ratio(pz)
    s_outliers = outlier_ratio(sz)

    sp = float(s_time - p_time)
    checks = []

    if sp < min_sp:
        checks.append("S-P interval is very short")

    if sp > max_sp:
        checks.append("S-P interval exceeds search range")

    if p_sta_trigger < p_threshold:
        checks.append("P trigger is below the adaptive threshold")

    if s_sta_trigger < s_threshold:
        checks.append("S trigger is below the adaptive threshold")

    if p_outliers > 0.15:
        checks.append("P window contains many outliers")

    if s_outliers > 0.15:
        checks.append("S window contains many outliers")

    score = 0

    if p_sta_trigger >= p_threshold:
        score += 1

    if s_sta_trigger >= s_threshold:
        score += 1

    if min_sp <= sp <= max_sp:
        score += 1

    if p_outliers <= 0.15:
        score += 1

    if s_outliers <= 0.15:
        score += 1

    if p_hv < 2.0:
        score += 1

    if s_hv > p_hv:
        score += 1

    if score >= 6:
        quality = "GOOD"
    elif score >= 4:
        quality = "REVIEW"
    else:
        quality = "REJECT"

    return {
        "quality": quality,
        "quality_score": score,
        "quality_notes": checks,
        "p_kurtosis": p_kurt,
        "s_kurtosis": s_kurt,
        "p_outlier_ratio": p_outliers,
        "s_outlier_ratio": s_outliers,
        "p_horizontal_vertical_ratio": p_hv,
        "s_horizontal_vertical_ratio": s_hv,
        "p_sta_trigger": float(p_sta_trigger),
        "s_sta_trigger": float(s_sta_trigger),
    }


def latlon_to_xy(lat, lon, lat0, lon0):
    x = (lon - lon0) * 111.32 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110.574

    return x, y


def xy_to_latlon(x, y, lat0, lon0):
    lat = lat0 + y / 110.574
    lon = lon0 + x / (111.32 * np.cos(np.radians(lat0)))

    return lat, lon


def locate_hypocenter(obs, vp=6.0, vs=3.5):
    n = len(obs)

    if n < 3:
        raise ValueError("Need at least 3 stations")

    lat0 = float(np.mean([o["lat"] for o in obs]))
    lon0 = float(np.mean([o["lon"] for o in obs]))

    station_xy = []

    for o in obs:
        x, y = latlon_to_xy(
            o["lat"],
            o["lon"],
            lat0,
            lon0
        )

        station_xy.append((x, y))

    station_xy = np.asarray(station_xy, dtype=float)
    sx = station_xy[:, 0]
    sy = station_xy[:, 1]

    if len(set(zip(np.round(sx, 4), np.round(sy, 4)))) < n:
        raise ValueError("Two stations have identical coordinates")

    earliest_p = min(e["p_time"] for e in obs)

    observed_p = np.asarray(
        [
            e["p_time"].timestamp - earliest_p.timestamp
            for e in obs
        ],
        dtype=float
    )

    observed_s = np.asarray(
        [
            e["s_time"].timestamp - earliest_p.timestamp
            for e in obs
        ],
        dtype=float
    )

    def residuals(params):
        x, y, depth, origin_relative = params

        distances = np.sqrt(
            (x - sx) ** 2
            + (y - sy) ** 2
            + depth ** 2
        )

        predicted_p = origin_relative + distances / vp
        predicted_s = origin_relative + distances / vs

        p_residuals = predicted_p - observed_p
        s_residuals = predicted_s - observed_s

        return np.concatenate([
            p_residuals,
            s_residuals
        ])

    initial = np.array([
        float(np.mean(sx)),
        float(np.mean(sy)),
        10.0,
        -10.0
    ])

    lower_bounds = np.array([
        -10000.0,
        -10000.0,
        0.0,
        -120.0
    ])

    upper_bounds = np.array([
        10000.0,
        10000.0,
        300.0,
        10.0
    ])

    result = least_squares(
        residuals,
        initial,
        bounds=(lower_bounds, upper_bounds),
        method="trf"
    )

    x, y, depth, origin_relative = result.x

    lat, lon = xy_to_latlon(
        x,
        y,
        lat0,
        lon0
    )

    origin_timestamp = earliest_p.timestamp + origin_relative
    origin_time = UTCDateTime(origin_timestamp)

    final_residuals = residuals(result.x)

    p_residuals = final_residuals[:n]
    s_residuals = final_residuals[n:]

    rms_seconds = float(
        np.sqrt(np.mean(final_residuals ** 2))
    )

    return {
        "lat": float(lat),
        "lon": float(lon),
        "depth_km": float(depth),
        "origin_time": origin_time,
        "rms_seconds": rms_seconds,
        "p_residuals": [float(r) for r in p_residuals],
        "s_residuals": [float(r) for r in s_residuals],
        "success": bool(result.success),
        "message": result.message,
        "cost": float(result.cost),
    }


def reset_station_state(name):
    state = station_state[name]

    state["p_time"] = None
    state["p_trigger"] = None
    state["s_time"] = None
    state["s_trigger"] = None
    state["quality"] = None
    state["distance_km"] = None
    state["threshold_last_update"] = None

    state["p_threshold"] = p_min_threshold
    state["p_off_threshold"] = max(
        1.2,
        p_min_threshold * p_off_ratio
    )

    state["s_threshold"] = s_min_threshold
    state["s_off_threshold"] = max(
        1.1,
        s_min_threshold * s_off_ratio
    )


def process_station(station):
    name = f"{station['net']}.{station['sta']}"

    try:
        z, env = prepare_station_stream(
            name,
            fs,
            fmin,
            fmax
        )

        if z is None:
            return

        print(
            f"[SCAN] {name} "
            f"end={z.stats.endtime.isoformat()} "
            f"samples={len(z.data)} "
            f"p_threshold={station_state[name]['p_threshold']:.2f} "
            f"s_threshold={station_state[name]['s_threshold']:.2f}"
        )
        
        now = z.stats.endtime
        state = station_state[name]

        update_station_thresholds(
            name,
            z,
            env,
            fs
        )

        if state["p_time"] is None:
            search_start = z.stats.endtime - scan

            result = pick_p(
                z,
                fs,
                search_start,
                sta_p,
                lta_p,
                state["p_threshold"],
                state["p_off_threshold"]
            )

            if result is not None:
                state["p_time"] = result["time"]
                state["p_trigger"] = result["trigger"]

                print(
                    f"[P] {name} "
                    f"{result['time'].isoformat()} "
                    f"STA/LTA={result['trigger']:.2f} "
                    f"threshold={state['p_threshold']:.2f}"
                )

        if state["p_time"] is not None and state["s_time"] is None:
            if now - state["p_time"] >= min_sp:
                result = pick_s(
                    env,
                    fs,
                    z.stats.starttime,
                    state["p_time"],
                    sta_s,
                    lta_s,
                    min_sp,
                    max_sp,
                    state["s_threshold"],
                    state["s_off_threshold"]
                )

                if result is not None:
                    state["s_time"] = result["time"]
                    state["s_trigger"] = result["trigger"]

                    sp = state["s_time"] - state["p_time"]

                    print(
                        f"[S] {name} "
                        f"{result['time'].isoformat()} "
                        f"S-P={sp:.2f}s "
                        f"STA/LTA={result['trigger']:.2f} "
                        f"threshold={state['s_threshold']:.2f}"
                    )

                    q = calculate_quality_metrics(
                        z,
                        env,
                        fs,
                        state["p_time"],
                        state["s_time"],
                        state["p_trigger"],
                        state["s_trigger"],
                        state["p_threshold"],
                        state["s_threshold"],
                        max_sp
                    )

                    state["quality"] = q

                    k = vp * vs / (vp - vs)
                    distance = sp * k
                    state["distance_km"] = distance

                    print(
                        f"[DIST-INFO] {name} "
                        f"{distance:.2f} km "
                        f"quality={q['quality']} "
                        f"score={q['quality_score']}/7"
                    )

                    if q["quality_notes"]:
                        print(
                            f"[QUALITY NOTES] {name}: "
                            + "; ".join(q["quality_notes"])
                        )

        if (
            state["p_time"] is not None
            and now - state["p_time"] > max_sp + 30
        ):
            reset_station_state(name)
            print(f"[RESET] {name}")

    except Exception as e:
        print(f"[PROCESS ERROR] {name}: {e}")


def get_current_event_stations():
    entries = []
    k = vp * vs / (vp - vs)

    for station in stations:
        name = f"{station['net']}.{station['sta']}"
        state = station_state[name]

        if (
            state["p_time"] is None
            or state["s_time"] is None
            or state["quality"] is None
        ):
            continue

        if state["quality"]["quality"] == "REJECT":
            continue

        sp = state["s_time"] - state["p_time"]
        distance_km = sp * k

        estimated_origin = state["p_time"] - distance_km / vp

        entries.append({
            "name": name,
            "lat": station["lat"],
            "lon": station["lon"],
            "p_time": state["p_time"],
            "s_time": state["s_time"],
            "distance_km": state["distance_km"],
            "estimated_origin": estimated_origin,
            "quality": state["quality"]["quality"],
        })

    return entries


def group_event_stations(entries):
    if not entries:
        return []

    best_group = []

    for anchor in entries:
        group = [
            entry for entry in entries
            if abs(
                entry["estimated_origin"]
                - anchor["estimated_origin"]
            ) <= event_origin_tolerance
        ]

        if len(group) > len(best_group):
            best_group = group

    unique_group = {}
    for entry in best_group:
        unique_group[entry["name"]] = entry

    return list(unique_group.values())


def check_and_locate_event():
    global last_location_time

    entries = get_current_event_stations()
    event_entries = group_event_stations(entries)

    if len(event_entries) < min_st:
        return

    p_times = [e["p_time"] for e in event_entries]
    earliest = min(p_times)
    latest = max(p_times)

    if latest - earliest > 90:
        return

    event_time = min(
        e["estimated_origin"] for e in event_entries
    )

    if (
        last_location_time is not None
        and abs(event_time - last_location_time) < 5
    ):
        return

    names = [e["name"] for e in event_entries]

    try:
        result = locate_hypocenter(
            event_entries,
            vp=vp,
            vs=vs
        )

        last_location_time = event_time

        if not result["success"]:
            print(
                f"[LOCATION REJECTED] Solver did not converge: "
                f"{result['message']}"
            )

        elif result["rms_seconds"] > max_location_rms:
            print(
                f"[LOCATION REJECTED] RMS error "
                f"{result['rms_seconds']:.3f} s exceeds "
                f"{max_location_rms:.2f} s"
            )

            print(
                "The picks may be inconsistent, the velocity "
                "assumptions may be unsuitable, or the station "
                "geometry may be weak."
            )

        else:
            print("\nLIVE EARTHQUAKE LOCATION")
            print(f"Origin time : {result['origin_time'].isoformat()}")
            print(f"Latitude    : {result['lat']:.5f} N")
            print(f"Longitude   : {result['lon']:.5f} E")
            print(f"Depth       : {result['depth_km']:.2f} km")
            print(f"RMS error   : {result['rms_seconds']:.3f} s")
            print(f"Stations    : {len(event_entries)}")
            print(f"Solver      : {result['success']}")

            print("\nP arrival residuals:")

            for name, residual in zip(
                names,
                result["p_residuals"]
            ):
                print(f"  {name:10s} {residual:+.3f} s")

            print("\nS arrival residuals:")

            for name, residual in zip(
                names,
                result["s_residuals"]
            ):
                print(f"  {name:10s} {residual:+.3f} s")

            print("\nObserved arrivals:")

            for entry in event_entries:
                print(
                    f"  {entry['name']:10s} "
                    f"P={entry['p_time'].isoformat()} "
                    f"S={entry['s_time'].isoformat()} "
                    f"quality={entry['quality']}"
                )

            print()

        for entry in event_entries:
            reset_station_state(entry["name"])

    except Exception as e:
        print(f"[LOCATION ERROR] {e}")


def processing_loop():
    print("\nStarting real-time processing...")
    print(f"Buffer: {buffer}s | Scan: {scan}s | Sampling rate: {fs} Hz")
    print(f"Vp: {vp} km/s | Vs: {vs} km/s")
    print(f"Minimum stations: {min_st}")
    print(f"Event origin tolerance: {event_origin_tolerance}s")
    print(f"Maximum location RMS: {max_location_rms}s")
    print("Locator: 3D P/S arrival-time nonlinear least squares\n")

    while True:
        loop_start = time.time()

        for station in stations:
            process_station(station)

        check_and_locate_event()

        elapsed = time.time() - loop_start
        time.sleep(max(0, scan - elapsed))


def start_seedlink():
    print(f"Connecting to SeedLink: {SEEDLINK_HOST}")

    client = LiveSeedLinkClient(
        SEEDLINK_HOST,
        autoconnect=False
    )

    client.conn.timeout = 10
    client.connect()

    for station in stations:
        client.select_stream(
            station["net"],
            station["sta"],
            "EH?"
        )

    print("Waiting for waveform data...")
    client.run()


if __name__ == "__main__":
    print("Stations:")

    for station in stations:
        print(
            f"  {station['net']}.{station['sta']} "
            f"({station['lat']}, {station['lon']})"
        )

    processing_thread = threading.Thread(
        target=processing_loop,
        daemon=True
    )

    processing_thread.start()

    try:
        start_seedlink()

    except KeyboardInterrupt:
        print("Stopping real-time system.")

    except Exception as e:
        print(f"SeedLink stopped: {e}")