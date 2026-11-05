import io
import json
import math
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, Response, jsonify, request
from matplotlib.ticker import FuncFormatter, MultipleLocator
from obspy import UTCDateTime
from obspy.clients.seedlink import Client as SeedLinkClient
from obspy.taup import TauPyModel
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import butter, sosfilt


# Pipeline (a function-only port of GlobalQuake's detection chain):
#   1. detector   per-station filtered STA/LTA with short/medium/long/third/special averages -> events
#   2. P picker   walk back from the trigger until the signal is quiet again                -> picks
#   3. clusters   picks whose time gaps fit the station spacing (5 km/s + 2.5 s)            -> clusters
#   4. locator    golden-angle spiral grid search + depth scan, median origin, TauP table  -> hypocenter
#   5. checks     correct-pick count, azimuth coverage, S-wave rejection                   -> confirm / dismiss

SERVER_HOST = "ring.wscada.net"
SERVER_PORT = 18000

STATIONS = [
    {"net": "NP", "sta": "EQM10", "cha": "EHZ", "loc": "*", "lat": 28.299517, "lon": 83.960148, "elev": 0},
    {"net": "NP", "sta": "EQM26", "cha": "EHZ", "loc": "*", "lat": 27.3936969, "lon": 86.6791885, "elev": 0},
    {"net": "NP", "sta": "EQM07", "cha": "EHZ", "loc": "*", "lat": 27.814941, "lon": 86.713401, "elev": 0},
    {"net": "NP", "sta": "EQM08", "cha": "EHZ", "loc": "*", "lat": 27.831000, "lon": 86.65, "elev": 0},
    {"net": "NP", "sta": "EQM11", "cha": "EHZ", "loc": "*", "lat": 29.2859556, "lon": 81.2741972, "elev": 0},
    {"net": "NP", "sta": "EQM22", "cha": "EHZ", "loc": "*", "lat": 28.895805, "lon": 80.227341, "elev": 0},
    {"net": "AM", "sta": "R0F23", "cha": "EHZ", "loc": "*", "lat": 28.045045, "lon": 84.072371, "elev": 0},
    {"net": "AM", "sta": "R02C2", "cha": "EHZ", "loc": "*", "lat": 27.621622, "lon": 85.320148, "elev": 0},
]
COORDS = {s["sta"]: (s["lat"], s["lon"], s.get("elev", 0.0)) for s in STATIONS}   # elevation in metres

WINDOW_SECONDS = 420      # data re-analysed each cycle (the detector needs ~80 s of warm-up and pick context)
LATENCY_SECONDS = 5
UPDATE_SECONDS = 20
ACTIVE_MAX_LATENCY = 60
FS = 50.0                 # working sampling rate (Hz)

# --- detector (BetterAnalysis) ---
BAND = (2.0, 5.0)              # Hz, causal 3rd-order Butterworth
INIT_OFFSET_SECONDS = 4.0      # DC offset estimated from the first 4 s
INIT_AVERAGE_SECONDS = 10.0    # then 10 s to settle the long average
EVENT_THRESHOLD = 4.75         # trigger at 1.3x this (or 2.05x for a fast rise)
EVENT_END_DURATION = 7.0
EVENT_TOO_LONG = 300.0
EVENT_EXTENSION = 90.0         # s of context kept before a trigger
MIN_EVENT_DIFF = 3.0           # picks closer than this at one station are duplicates
MAX_PICKS_PER_WINDOW = 8       # more than this and the station is treated as noisy and ignored

# --- P picker (Event.findPWaveMethod1) ---
PICK_CONTEXT_SECONDS = 65.0
RECALC_THRESHOLDS = (16.0, 32.0, 64.0, 128.0, 512.0, 2048.0)
SPECIAL_PERCENTILE = (0.08, 0.12, 0.18, 0.24, 0.32, 0.40, 0.48)
SLOW_MULTIPLIERS = (1.12, 1.5, 1.9, 2.2, 2.4, 2.5, 2.6)

# --- association (ClusterAnalysis / GlobalStationManager) ---
PICK_VELOCITY = 5.0            # km/s, window = distance / 5 + 2.5 s
PICK_SLACK = 2.5
CLUSTER_MIN_SIZE = 4
NEARBY_RAYS, NEARBY_PER_RAY, NEARBY_MAX_KM, NEARBY_FAR_KM = 9, 3, 4000.0, 1000.0
PICK_MEMORY = 900.0            # s, picks older than this are forgotten

# --- location (EarthquakeAnalysis) ---
P_WAVE_INACCURACY = 1.8        # s, a pick is "correct" if it fits within this
SEARCH_STAGES = ((2500.0, 4000), (100.0, 3000), (10.0, 2000))   # (radius km, spiral points)
SEARCH_DEPTHS = [0, 5, 10, 15, 20, 30, 40, 60, 80, 120, 200, 300, 500, 700]
MAX_DEPTH = 750.0
PHI = 1.61803398875
QUADRANTS = 26

# --- confirmation ---
MIN_STATIONS = 5               # with lat, lon, depth and origin free, 4 picks always fit
MIN_RATIO = 16.0               # strongest pick must reach this STA/LTA
CORRECTNESS_THRESHOLD = 0.5
INVESTIGATE_SECONDS = 180.0    # candidates unconfirmed this long after first seen are dismissed
MATCH_SECONDS = 30.0
MAX_LOG = 300

EARTH_RADIUS, EARTH_CIRCUMFERENCE = 6379.0, 40082.0
TABLE_FILE = Path("travel_table_iasp91.npz")
HISTORY_FILE = Path("quake_history.json")
PORT = 8000

STATION_PALETTE = ["#0F766E", "#2563EB", "#7C3AED", "#92400E", "#0891B2", "#4D7C0F", "#475569", "#C026D3"]
STATION_COLORS = {s["sta"]: STATION_PALETTE[i % len(STATION_PALETTE)] for i, s in enumerate(STATIONS)}
INK, SLATE, GRID, ALERT = "#14202B", "#5B6B7A", "#E3E8ED", "#D6361F"

plt.rcParams.update({
    "font.size": 9, "figure.facecolor": "white", "savefig.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#C9D1D9", "axes.labelcolor": SLATE, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": SLATE, "ytick.color": SLATE,
})


# ----------------------------------------------------------------------------- helpers

def fmt_time(t):
    return UTCDateTime(t).strftime("%Y-%m-%d %H:%M:%S") + " UTC"


def fmt_clock(t):
    return UTCDateTime(t).strftime("%H:%M:%S")


def fmt_npt(t):
    return (UTCDateTime(t) + 5.75 * 3600).strftime("%H:%M:%S") + " NPT"


def great_circle_distance(lat1, lon1, lat2, lon2):
    """Haversine distance in km (works on arrays)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return EARTH_RADIUS * 2 * np.arctan2(np.sqrt(np.clip(a, 0, 1)), np.sqrt(np.clip(1 - a, 0, 1)))


def calculate_angle(lat1, lon1, lat2, lon2):
    """Bearing in degrees, 0..360."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def move_on_globe(lat, lon, distance_km, angle_deg):
    """Point reached from (lat, lon) after distance_km along a bearing (works on arrays)."""
    delta = np.asarray(distance_km, float) / EARTH_RADIUS
    theta, phi, gamma = np.radians(lat), np.radians(lon), np.radians(np.asarray(angle_deg, float))
    x = np.cos(delta) * np.cos(theta) * np.cos(phi) - np.sin(delta) * (
        np.sin(theta) * np.cos(phi) * np.cos(gamma) + np.sin(phi) * np.sin(gamma))
    y = np.cos(delta) * np.cos(theta) * np.sin(phi) - np.sin(delta) * (
        np.sin(theta) * np.sin(phi) * np.cos(gamma) - np.cos(phi) * np.sin(gamma))
    z = np.sin(delta) * np.cos(theta) * np.cos(gamma) + np.cos(delta) * np.sin(theta)
    return np.degrees(np.arcsin(np.clip(z, -1, 1))), np.degrees(np.arctan2(y, x))


def to_angle(km):
    return np.asarray(km, float) / EARTH_CIRCUMFERENCE * 360.0


def elevation_correction(elev_m):
    return np.asarray(elev_m, float) / 6000.0       # metres -> seconds, as in GlobalQuake


# ----------------------------------------------------------------------------- travel times (TauP)

P_PHASES = ("P", "Pn", "Pdiff", "PKP", "PKIKP", "PKiKP")
S_PHASES = ("S", "Sn", "Sdiff")
DEPTH_AXIS = np.array([0, 2, 5, 8, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160, 200, 250, 300, 350,
                       400, 450, 500, 550, 600, 650, 700, 750], float)
ANGLE_AXIS = np.concatenate([[0.01], np.arange(0.1, 2.0001, 0.1), np.arange(2.5, 10.0001, 0.5),
                             np.arange(11.0, 180.0001, 1.0)])
TRAVEL = {}


def build_travel_table():
    """First-arrival P and S times from ObsPy's TauP (iasp91), cached on disk."""
    print(f"Building iasp91 travel-time table ({len(DEPTH_AXIS)} depths x {len(ANGLE_AXIS)} distances), "
          f"this happens once...")
    model = TauPyModel(model="iasp91")
    p = np.full((len(DEPTH_AXIS), len(ANGLE_AXIS)), np.nan)
    s = p.copy()
    for i, depth in enumerate(DEPTH_AXIS):
        for j, angle in enumerate(ANGLE_AXIS):
            try:
                arrivals = model.get_travel_times(float(depth), float(angle), phase_list=P_PHASES + S_PHASES)
            except Exception:
                continue
            for a in arrivals:
                table = p if a.name in P_PHASES else s
                if np.isnan(table[i, j]) or a.time < table[i, j]:
                    table[i, j] = a.time
        if i % 4 == 3:
            print(f"  travel table {i + 1}/{len(DEPTH_AXIS)} depths")
    np.savez_compressed(TABLE_FILE, p=p, s=s)
    return p, s


def load_travel_tables():
    if TABLE_FILE.exists():
        data = np.load(TABLE_FILE)
        p, s = data["p"], data["s"]
    else:
        p, s = build_travel_table()
    for name, table in (("p", p), ("s", s)):
        TRAVEL[name] = RegularGridInterpolator((DEPTH_AXIS, ANGLE_AXIS), table, bounds_error=False, fill_value=np.nan)


def travel_time(kind, depth, angle):
    """Travel time in seconds, NaN where the phase does not exist. Broadcasts depth and angle."""
    depth, angle = np.broadcast_arrays(np.asarray(depth, float), np.asarray(angle, float))
    points = np.stack([depth, np.maximum(angle, ANGLE_AXIS[0])], axis=-1)
    return TRAVEL[kind](points).reshape(depth.shape)


# ----------------------------------------------------------------------------- data

def fetch_waveform(station, start, end):
    client = SeedLinkClient(SERVER_HOST, port=SERVER_PORT, timeout=20)
    loc = "??" if station["loc"] == "*" else station["loc"]
    stream = client.get_waveforms(station["net"], station["sta"], loc, station["cha"], start, end)
    if len(stream) == 0:
        raise RuntimeError("no data")

    # GlobalQuake restarts its detector after a gap, so only the newest gap-free piece is used
    pieces = stream.merge(method=1).split()
    trace = sorted(pieces, key=lambda t: t.stats.starttime)[-1]
    trace.data = trace.data.astype(float)
    trace.resample(FS)
    needed = INIT_OFFSET_SECONDS + INIT_AVERAGE_SECONDS + PICK_CONTEXT_SECONDS + 30
    if trace.stats.endtime - trace.stats.starttime < needed:
        raise RuntimeError(f"only {trace.stats.endtime - trace.stats.starttime:.0f} s of gap-free data")
    return trace


def fetch_all(stations, start, end):
    def one(station):
        try:
            return station["sta"], fetch_waveform(station, start, end), None
        except Exception as error:
            return station["sta"], None, str(error)

    with ThreadPoolExecutor(max_workers=len(stations)) as pool:
        results = list(pool.map(one, stations))
    traces = {n: t for n, t, _ in results if t is not None}
    errors = {n: e for n, t, e in results if t is None}
    return traces, errors


# ----------------------------------------------------------------------------- 1. detector

def run_detector(trace):
    """Five moving averages of |filtered velocity| and a two-tier STA/LTA trigger (BetterAnalysis)."""
    fs = trace.stats.sampling_rate
    x = trace.data.astype(float)
    n_off = int(INIT_OFFSET_SECONDS * fs)
    n_init = int((INIT_OFFSET_SECONDS + INIT_AVERAGE_SECONDS) * fs)

    nyq = fs / 2
    sos = butter(3, [BAND[0] / nyq, BAND[1] / nyq], btype="bandpass", output="sos")
    filtered = sosfilt(sos, x - x[:n_off].mean())          # causal, so no future samples leak in
    a = np.abs(filtered)

    # calibration: long average from the first seconds, then settled with a 6 s time constant
    long_avg = a[n_off // 4:n_off].mean()
    for i in range(n_off, n_init):
        long_avg -= (long_avg - a[i]) / (fs * 6.0)
    long_avg = max(long_avg, 1e-9)
    short = medium = third = long_avg
    special = long_avg * 2.5
    long_avg *= 0.75

    n = len(x)
    ratio, medium_ratio, special_ratio = np.zeros(n), np.zeros(n), np.zeros(n)
    events, open_event, event_timer = [], None, 0.0
    ks, km, kt, ksp, kl = 1 / (fs * 0.5), 1 / (fs * 6.0), 1 / (fs * 30.0), 1 / (fs * 40.0), 1 / (fs * 200.0)

    for i in range(n_init, n):
        t, v = i / fs, a[i]
        short += (v - short) * ks
        medium += (v - medium) * km
        third += (v - third) * kt
        special = v if v > special else special + (v - special) * ksp
        if short / long_avg < 4.0:                            # LTA is frozen during strong signal
            long_avg += (v - long_avg) * kl
        r = short / long_avg
        ratio[i], medium_ratio[i], special_ratio[i] = r, medium / long_avg, special / long_avg

        if open_event is None:
            fast = r >= EVENT_THRESHOLD * 2.05 and t - event_timer > 0.1
            slow = r >= EVENT_THRESHOLD * 1.3 and t - event_timer > 0.2
            if short / third > 3.0 and (fast or slow):
                open_event = {"start": i, "end": None, "valid": True, "max_ratio": r}
                events.append(open_event)
        if r < EVENT_THRESHOLD:
            event_timer = t

        for e in events[-5:]:
            if e["valid"] and (e["end"] is None or i - e["end"] < EVENT_EXTENSION * fs):
                e["max_ratio"] = max(e["max_ratio"], r)

        if open_event is not None:
            dt = t - open_event["start"] / fs
            if dt >= EVENT_END_DURATION and medium < third * 0.95:                 # ended normally
                open_event["end"], open_event = i, None
            elif dt >= EVENT_TOO_LONG:                                             # never ended
                open_event["valid"], open_event["end"], open_event = False, i, None
            elif dt >= 1.0 and ((dt < 7.5 and short < long_avg * 1.25) or short < medium * 0.12):
                open_event["valid"], open_event["end"], open_event = False, i, None   # ended badly

    return {"fs": fs, "n_init": n_init, "filtered": filtered, "ratio": ratio, "medium_ratio": medium_ratio,
            "special_ratio": special_ratio, "events": events}


def pick_p_wave(det, event):
    """Walk back from the trigger to the last sample where both the short and the 'special' ratio are quiet."""
    fs, start = det["fs"], event["start"]
    first = max(det["n_init"], start - int(EVENT_EXTENSION * fs))
    if start - first < PICK_CONTEXT_SECONDS * fs:                 # GlobalQuake needs 65 s of context
        return None

    strength = min(sum(event["max_ratio"] >= th for th in RECALC_THRESHOLDS), len(SPECIAL_PERCENTILE) - 1)
    look = first if strength == 0 else max(first, start - int(60.0 / strength * fs))
    window = slice(look, start + 1)
    ratio, medium, special = det["ratio"][window], det["medium_ratio"][window], det["special_ratio"][window]

    positive = special[special > 0]
    min_special = float(positive.min()) if positive.size else 1.0
    max_special = max(min_special * 5.0, float(positive.max()) if positive.size else 1.0)
    slow_percentile = float(np.sort(medium)[int((medium.size - 1) * 0.175)])
    mul = SPECIAL_PERCENTILE[strength] * 1.1
    special_limit = max_special * mul + (1.0 - mul) * min_special

    quiet = (ratio <= slow_percentile * SLOW_MULTIPLIERS[strength] * 1.25) & (special <= special_limit)
    idx = np.nonzero(quiet)[0]
    return None if idx.size == 0 else look + int(idx[-1])


def detect_picks(name, trace):
    det = run_detector(trace)
    t0, fs = trace.stats.starttime.timestamp, det["fs"]
    picks = []
    for event in det["events"]:
        if not event["valid"]:
            continue
        idx = pick_p_wave(det, event)
        if idx is None:
            continue
        end = event["end"] if event["end"] is not None else len(trace.data) - 1
        if picks and t0 + idx / fs - picks[-1]["t"] < MIN_EVENT_DIFF:
            continue                                              # later duplicate is dropped
        picks.append({"station": name, "t": t0 + idx / fs, "duration": (end - event["start"]) / fs,
                      "max_ratio": float(event["max_ratio"])})
    noisy = len(picks) > MAX_PICKS_PER_WINDOW
    return det, ([] if noisy else picks), noisy


def update_pick_log(log, picks, recompute_from, now):
    """Picks inside a station's re-analysed region are replaced, older ones are kept."""
    kept = [p for p in log if now - p["t"] < PICK_MEMORY
            and not (p["station"] in recompute_from and p["t"] >= recompute_from[p["station"]])]
    for p in picks:
        if any(k["station"] == p["station"] and abs(k["t"] - p["t"]) < 20 for k in kept):
            continue
        if not any(k["station"] == p["station"] and abs(k["t"] - p["t"]) < 20 for k in log):
            print(f"  >> Pick {p['station']} {fmt_time(p['t'])} (STA/LTA {p['max_ratio']:.1f})")
        kept.append({"station": p["station"], "t": p["t"], "time": fmt_time(p["t"]), "clock": fmt_clock(p["t"]),
                     "npt": fmt_npt(p["t"]), "duration": round(p["duration"], 1),
                     "peak_ratio": round(p["max_ratio"], 1), "max_ratio": p["max_ratio"]})
    kept.sort(key=lambda p: p["t"], reverse=True)
    log[:] = kept[:MAX_LOG]


# ----------------------------------------------------------------------------- 2. clusters

def build_nearby_graph():
    """Per station: up to 3 neighbours in each of 9 rays, at most one beyond 1000 km (GlobalStationManager)."""
    graph = {}
    for a, (alat, alon, _) in COORDS.items():
        rays = [[] for _ in range(NEARBY_RAYS)]
        for b, (blat, blon, _) in COORDS.items():
            if a == b:
                continue
            dist = float(great_circle_distance(alat, alon, blat, blon))
            if dist > NEARBY_MAX_KM:
                continue
            ray = int(calculate_angle(alat, alon, blat, blon) / 360.0 * (NEARBY_RAYS - 1))
            for k in (ray, (ray + 1) % NEARBY_RAYS, (ray - 1) % NEARBY_RAYS):
                rays[k].append((dist, b))
        found = {}
        for ray in rays:
            count = 0
            for dist, b in sorted(ray):
                if count >= NEARBY_PER_RAY:
                    break
                if b not in found:
                    found[b] = dist
                    count += 1
                if dist > NEARBY_FAR_KM:
                    break
        graph[a] = list(found.items())
    return graph


NEARBY = build_nearby_graph()


def pick_in_window(candidate, origin, dist_km):
    return abs(candidate["t"] - origin["t"]) <= dist_km / PICK_VELOCITY + PICK_SLACK


def find_free_pick(by_station, station, used, origin, dist_km):
    return next((c for c in by_station.get(station, []) if id(c) not in used and pick_in_window(c, origin, dist_km)), None)


def expand_cluster(group, by_station, used):
    """Keep adding picks of neighbouring stations until nothing more fits."""
    members = {p["station"] for p in group}
    frontier = list(group)
    while frontier:
        added = []
        for p in frontier:
            for other, dist in NEARBY[p["station"]]:
                if other in members:
                    continue
                c = find_free_pick(by_station, other, used, p, dist)
                if c is not None:
                    added.append(c)
                    members.add(other)
        group += added
        frontier = added
    return group


def find_clusters(picks):
    picks = sorted(picks, key=lambda p: p["t"])
    by_station = {}
    for p in picks:
        by_station.setdefault(p["station"], []).append(p)

    used, clusters = set(), []
    for p in picks:
        if id(p) in used:
            continue
        group = [p]
        for other, dist in NEARBY[p["station"]]:
            c = find_free_pick(by_station, other, used, p, dist)
            if c is not None:
                group.append(c)
        if len(group) < CLUSTER_MIN_SIZE:
            continue
        group = expand_cluster(group, by_station, used)
        used.update(id(g) for g in group)
        clusters.append(group)
    return clusters


# ----------------------------------------------------------------------------- 3. location

def heuristic(correct, err):
    """GlobalQuake's score: correct^2 / err^2."""
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.asarray(correct, float) ** 2 / np.asarray(err, float) ** 2
    return np.where(np.isfinite(h), h, np.where((np.asarray(err) == 0) & (np.asarray(correct) > 0), np.inf, 0.0))


def best_index(correct, err):
    """Highest score, but a candidate with 30 % fewer correct picks than the best cannot win."""
    ok = np.isfinite(err) & (correct > 0)
    if not ok.any():
        return None
    contenders = ok & (correct * 1.3 >= correct[ok].max())
    return int(np.argmax(np.where(contenders, heuristic(correct, err), -1.0)))


def better(a, b):
    if a is None or b is None:
        return a or b
    if a["correct"] > int(b["correct"] * 1.3):
        return a
    if b["correct"] > int(a["correct"] * 1.3):
        return b
    return a if heuristic(a["correct"], a["err"]) > heuristic(b["correct"], b["err"]) else b


def analyse_grid(angles, depths, p_waves, elev):
    """For every (point, depth): origin = median(pick - travel), then count and score the fitting picks.

    angles is (..., events), depths broadcasts against it. Picks without a P arrival count as wrong.
    """
    origins = p_waves - (travel_time("p", depths, angles) + elev)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        origin = np.nanmedian(origins, axis=-1)
    signed = origins - origin[..., None]
    absolute = np.where(np.isnan(signed), 1e3, np.abs(signed))
    correct = (absolute < P_WAVE_INACCURACY).sum(axis=-1)
    penalised = np.where(absolute < P_WAVE_INACCURACY, absolute, (absolute - P_WAVE_INACCURACY) * 0.2 + P_WAVE_INACCURACY)
    err = (penalised ** 2).sum(axis=-1)
    dead = np.isnan(origins).sum(axis=-1) * 2 >= origins.shape[-1]       # median would be an unknown arrival
    return origin, signed, np.where(dead, 0, correct), np.where(dead, np.inf, err)


def scan_area(group, lat, lon, max_dist, points):
    """Candidate epicentres on a golden-angle spiral around (lat, lon); every depth is tried at each one."""
    ev_lat = np.array([COORDS[p["station"]][0] for p in group])
    ev_lon = np.array([COORDS[p["station"]][1] for p in group])
    elev = elevation_correction([COORDS[p["station"]][2] for p in group])
    p_waves = np.array([p["t"] for p in group])

    n = np.arange(points, dtype=float)
    lats, lons = move_on_globe(lat, lon, np.sqrt(n) * max_dist / math.sqrt(points), 360.0 / PHI ** 2 * n)
    angles = to_angle(great_circle_distance(lats[:, None], lons[:, None], ev_lat, ev_lon))        # (P, E)
    depths = np.array(SEARCH_DEPTHS, float)

    origin, _, correct, err = analyse_grid(angles[:, None, :], depths[None, :, None], p_waves, elev)
    idx = best_index(correct.ravel(), err.ravel())
    if idx is None:
        return None
    i, j = divmod(idx, len(depths))
    return {"lat": float(lats[i]), "lon": float(lons[i]), "depth": float(depths[j]),
            "origin": float(origin[i, j]), "err": float(err[i, j]), "correct": int(correct[i, j])}


def cluster_centre(group):
    """Spherical mean of the picking stations."""
    lat = np.mean([COORDS[p["station"]][0] for p in group])
    lons = np.radians([COORDS[p["station"]][1] for p in group])
    return float(lat), float(np.degrees(np.arctan2(np.sin(lons).mean(), np.cos(lons).mean())))


def locate(group):
    lat, lon = cluster_centre(group)
    best = None
    for radius, points in SEARCH_STAGES:                    # coarse to fine, each stage centred on the best so far
        candidate = scan_area(group, lat, lon, radius, points)
        best = better(candidate, best)
        if best is None:
            return None
        lat, lon = best["lat"], best["lon"]

    ev_lat = np.array([COORDS[p["station"]][0] for p in group])
    ev_lon = np.array([COORDS[p["station"]][1] for p in group])
    elev = elevation_correction([COORDS[p["station"]][2] for p in group])
    angles = to_angle(great_circle_distance(lat, lon, ev_lat, ev_lon))
    origin, signed, correct, err = analyse_grid(angles[None, None, :], np.array([[[best["depth"]]]]),
                                                np.array([p["t"] for p in group]), elev)
    best.update(origin=float(origin[0, 0]), residuals=signed[0, 0], correct=int(correct[0, 0]), err=float(err[0, 0]))
    return best


def quadrant_score(lat, lon, group):
    """Azimuth coverage: each of 26 sectors gives 0.5 for its first two stations."""
    counts, score = [0] * QUADRANTS, 0.0
    for p in group:
        sta_lat, sta_lon, _ = COORDS[p["station"]]
        k = min(QUADRANTS - 1, int(calculate_angle(lat, lon, sta_lat, sta_lon) * QUADRANTS / 360.0))
        score += 0.5 if counts[k] < 2 else 0.0
        counts[k] += 1
    return score


def evaluate_event(event):
    n = event["used"]
    if n < MIN_STATIONS:
        return "investigating", f"{n} of {MIN_STATIONS} stations so far, waiting for more"

    problems = []
    if event["max_ratio"] < MIN_RATIO:
        problems.append(f"strongest pick only reaches STA/LTA {event['max_ratio']}")
    if event["correct"] < MIN_STATIONS:
        problems.append(f"only {event['correct']} of {n} picks fit the solution")
    if event["correct"] / n < CORRECTNESS_THRESHOLD:
        problems.append(f"{round(100 * event['correct'] / n)} % of picks fit")
    if event["quadrants"] < 2.0:
        problems.append("stations cover too narrow an azimuth range")
    if event["depth"] > MAX_DEPTH - 5:
        problems.append("depth ran to the search limit")

    if problems:
        text = "; ".join(problems)
        return "investigating", text[0].upper() + text[1:]
    return "confirmed", f"{event['correct']} of {n} picks fit, RMS {event['rms']} s"


def build_event(group):
    group = sorted(group, key=lambda p: p["t"])
    sol = locate(group)
    if sol is None:
        return None
    t0 = sol["origin"]
    event = {
        "id": str(int(t0)), "t0": t0, "time": fmt_time(t0), "clock": fmt_clock(t0), "npt": fmt_npt(t0),
        "lat": round(sol["lat"], 3), "lon": round(sol["lon"], 3), "depth": round(sol["depth"], 1),
        "rms": round(float(np.sqrt(np.nanmean(np.minimum(np.abs(sol["residuals"]), 30.0) ** 2))), 2),
        "correct": sol["correct"], "used": len(group),
        "max_ratio": round(max(p["max_ratio"] for p in group), 1),
        "quadrants": quadrant_score(sol["lat"], sol["lon"], group),
        "stations": [p["station"] for p in group],
        "picks": {p["station"]: p["t"] for p in group},
        "residuals": {p["station"]: round(float(r), 2) for p, r in zip(group, sol["residuals"])},
    }
    event["status"], event["reason"] = evaluate_event(event)
    return event


def is_s_arrival(pick, event):
    """True when a pick fits the S wave of a known event better than its P wave."""
    lat, lon, elev = COORDS[pick["station"]]
    angle = float(to_angle(great_circle_distance(event["lat"], event["lon"], lat, lon)))
    actual = pick["t"] - event["t0"]
    s_time = float(travel_time("s", event["depth"], angle))
    p_time = float(travel_time("p", event["depth"], angle))
    if np.isnan(s_time):
        return False
    diff = actual - (s_time + float(elevation_correction(elev)) * 1.5)
    s_fits = (-2.0 - s_time * 0.03) < diff < (6.0 + s_time * 0.05)
    p_fits = not np.isnan(p_time) and abs(p_time + float(elevation_correction(elev)) - actual) < max(10.0, 1.0 + p_time * 0.01)
    return s_fits and not p_fits


def usable_picks(log, events, now):
    """Fresh picks that are not the S wave of an event we already know."""
    live = [e for e in events if e["status"] != "dismissed"]
    return [p for p in log if now - p["t"] < PICK_MEMORY and not any(is_s_arrival(p, e) for e in live)]


def apply_events(events, found, now):
    for new in found:
        match = next((i for i, e in enumerate(events) if abs(e["t0"] - new["t0"]) <= MATCH_SECONDS), None)

        if match is None:
            new["first_seen"] = new["last_seen"] = now
            events.insert(0, new)
            print(f"  >> {new['status'].upper()} candidate {new['time']} at {new['lat']}, {new['lon']}, "
                  f"depth {new['depth']:g} km ({new['reason']})")
            continue

        old = events[match]
        old["last_seen"] = now
        if old["status"] == "dismissed":
            continue
        if old["status"] == "confirmed":
            replace = new["status"] == "confirmed" and new["used"] > old["used"]
        else:
            replace = new["used"] >= old["used"]
        if replace:
            if new["status"] == "confirmed" and old["status"] != "confirmed":
                print(f"  >> EARTHQUAKE CONFIRMED {new['time']} at {new['lat']}, {new['lon']}, depth {new['depth']:g} km")
            new["id"], new["first_seen"], new["last_seen"] = old["id"], old["first_seen"], now
            events[match] = new

    for e in events:
        if e["status"] == "investigating" and now - e["first_seen"] > INVESTIGATE_SECONDS:
            e["status"] = "dismissed"
            e["reason"] = "Not confirmed in time. " + e["reason"]
            print(f"  >> Candidate {e['time']} dismissed")
    del events[MAX_LOG:]


# ----------------------------------------------------------------------------- plots

def image_to_bytes(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def create_empty_plot(message):
    fig, ax = plt.subplots(figsize=(12, 2.2))
    ax.text(0.5, 0.5, message, ha="center", va="center", color=SLATE, fontsize=11)
    ax.axis("off")
    return image_to_bytes(fig)


def time_axis(trace, end):
    return trace.times() + (trace.stats.starttime - end)


def style_time_axis(ax, end):
    ax.set_xlim(-WINDOW_SECONDS, 0)
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: (end + float(x)).strftime("%H:%M:%S")))
    ax.set_xlabel("Time (UTC)")


def create_waveform_plot(traces, dets, picks, end):
    fig, axes = plt.subplots(len(traces), 1, figsize=(12, 1.25 * len(traces) + 0.7), sharex=True,
                             gridspec_kw={"hspace": 0.12})
    axes = np.atleast_1d(axes)
    for ax, (name, tr) in zip(axes, traces.items()):
        data = dets[name]["filtered"]
        ax.plot(time_axis(tr, end), data, color=STATION_COLORS[name], linewidth=0.55)
        peak = float(np.max(np.abs(data[dets[name]["n_init"]:]))) or 1.0
        ax.set_ylim(-peak * 1.15, peak * 1.15)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        for p in picks:
            if p["station"] == name:
                ax.axvline(p["t"] - end.timestamp, color=ALERT, linewidth=1.3)
        ax.text(0.006, 0.93, name, transform=ax.transAxes, va="top", fontweight="bold", color=STATION_COLORS[name])
    style_time_axis(axes[-1], end)
    fig.suptitle(f"Waveforms {BAND[0]:g}-{BAND[1]:g} Hz (red = P pick)", x=0.125, ha="left", y=0.995, color=INK)
    return image_to_bytes(fig)


def create_ratio_plot(traces, dets, picks, end):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for name, tr in traces.items():
        ax.plot(time_axis(tr, end), dets[name]["ratio"], color=STATION_COLORS[name], linewidth=1, label=name)
    ax.axhline(EVENT_THRESHOLD * 1.3, color=ALERT, linestyle="--", linewidth=1.2)
    ax.axvspan(-WINDOW_SECONDS, -WINDOW_SECONDS + INIT_OFFSET_SECONDS + INIT_AVERAGE_SECONDS, color="#EEF1F4", linewidth=0)
    for p in picks:
        ax.axvline(p["t"] - end.timestamp, color=STATION_COLORS[p["station"]], linestyle=":", linewidth=1)
    top = max([EVENT_THRESHOLD * 4] + [float(d["ratio"].max()) for d in dets.values()])
    ax.set_ylim(0, min(top * 1.1, 200))
    style_time_axis(ax, end)
    ax.set_ylabel("Short average / long average")
    ax.set_title("Detector ratio (grey = calibration, dashed = trigger)", loc="left")
    ax.legend(ncol=len(traces), loc="upper right", frameon=False)
    return image_to_bytes(fig)


# ----------------------------------------------------------------------------- cycle

def run_cycle():
    end = UTCDateTime() - LATENCY_SECONDS
    start = end - WINDOW_SECONDS
    print(f"\n[{fmt_time(UTCDateTime())}] cycle, window {start.strftime('%H:%M:%S')} -> {end.strftime('%H:%M:%S')}")

    traces, errors = fetch_all(STATIONS, start, end)
    for name, error in errors.items():
        print(f"  {name}: {error}")

    picks, dets, recompute_from, station_status = [], {}, {}, []
    for station in STATIONS:
        name = station["sta"]
        info = {"station": name, "net": station["net"], "cha": station["cha"], "color": STATION_COLORS[name],
                "lat": station["lat"], "lon": station["lon"]}
        if name not in traces:
            info.update(state="nodata", active=False, note=errors.get(name, "Station did not respond"))
            station_status.append(info)
            continue

        trace = traces[name]
        det, found, noisy = detect_picks(name, trace)
        dets[name] = det
        picks += found
        recompute_from[name] = trace.stats.starttime.timestamp + det["n_init"] / det["fs"] + PICK_CONTEXT_SECONDS - 5

        latency = UTCDateTime() - trace.stats.endtime
        active = latency <= ACTIVE_MAX_LATENCY
        recent = any(p["t"] > end.timestamp - 120 for p in found)
        state = "stale" if not active else "triggered" if recent else "quiet"
        info.update(state=state, active=active, last_sample=fmt_clock(trace.stats.endtime), latency_s=round(latency, 1),
                    max_ratio=round(float(det["ratio"][-int(60 * FS):].max()), 1),
                    note="Too many picks, station ignored" if noisy else "")
        station_status.append(info)

    if traces:
        plots = {"waveforms": create_waveform_plot(traces, dets, picks, end),
                 "ratio": create_ratio_plot(traces, dets, picks, end)}
    else:
        plots = {k: create_empty_plot("No station returned data in this cycle") for k in ("waveforms", "ratio")}
    return {"plots": plots, "picks": picks, "stations": station_status, "recompute_from": recompute_from}


# ----------------------------------------------------------------------------- state / loop

STATE = {
    "id": 0, "updated": None, "updated_ts": None, "error": None, "cycle_seconds": None,
    "stations": [], "pick_log": [], "events": [],
    "settings": {"server": f"{SERVER_HOST}:{SERVER_PORT}", "update_seconds": UPDATE_SECONDS,
                 "window_seconds": WINDOW_SECONDS, "thresh_on": round(EVENT_THRESHOLD * 1.3, 2),
                 "min_stations": MIN_STATIONS, "cluster_min": CLUSTER_MIN_SIZE, "colors": STATION_COLORS},
}
PLOTS = {}
STATE_LOCK = threading.Lock()


def load_history():
    try:
        saved = json.loads(HISTORY_FILE.read_text())
        STATE["pick_log"] = saved.get("pick_log", [])
        STATE["events"] = saved.get("events", [])
        print(f"Loaded {len(STATE['pick_log'])} picks and {len(STATE['events'])} events from {HISTORY_FILE}")
    except FileNotFoundError:
        pass
    except Exception as error:
        print(f"Could not read {HISTORY_FILE}: {error}")


def save_history():
    try:
        HISTORY_FILE.write_text(json.dumps({"pick_log": STATE["pick_log"], "events": STATE["events"]}))
    except Exception as error:
        print(f"Could not write {HISTORY_FILE}: {error}")


def locate_clusters(log, events, now):
    found = []
    for group in find_clusters(usable_picks(log, events, now)):
        try:
            event = build_event(group)
        except Exception as error:
            print(f"  location failed: {error}")
            continue
        if event is not None:
            found.append(event)
    return found


def processing_loop():
    while True:
        started = time.time()
        try:
            result = run_cycle()
            now = time.time()
            with STATE_LOCK:
                update_pick_log(STATE["pick_log"], result["picks"], result["recompute_from"], now)
                log, events = list(STATE["pick_log"]), list(STATE["events"])
            found = locate_clusters(log, events, now)                 # heavy part, runs outside the lock
            with STATE_LOCK:
                apply_events(STATE["events"], found, now)
                PLOTS.update(result["plots"])
                STATE.update(id=STATE["id"] + 1, updated=fmt_time(UTCDateTime()), updated_ts=now, error=None,
                             stations=result["stations"], cycle_seconds=round(time.time() - started, 1))
                save_history()
        except Exception as error:
            print(f"  cycle failed: {error}")
            with STATE_LOCK:
                STATE.update(id=STATE["id"] + 1, error=str(error))
        time.sleep(max(1, UPDATE_SECONDS - (time.time() - started)))


def build_payload():
    now = time.time()
    events = STATE["events"]
    counts = {
        "active": sum(1 for s in STATE["stations"] if s.get("active")), "total": len(STATIONS),
        "investigating": sum(1 for e in events if e["status"] == "investigating"),
        "confirmed_24h": sum(1 for e in events if e["status"] == "confirmed" and now - e["t0"] < 86400),
        "picks_1h": sum(1 for p in STATE["pick_log"] if now - p["t"] < 3600),
    }
    payload = dict(STATE)
    payload.update(counts=counts, server_time=now)
    return payload


# ----------------------------------------------------------------------------- website

app = Flask(__name__)

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Earthquake monitor</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  body { margin: 0; background: #EEF1F4; color: #14202B; font: 14px/1.45 system-ui, sans-serif; }
  header { background: #fff; border-bottom: 1px solid #D9E0E6; padding: 14px 20px; display: flex; justify-content: space-between; }
  h1 { font-size: 18px; margin: 0; } h2 { font-size: 14px; margin: 0; padding: 12px 16px; border-bottom: 1px solid #D9E0E6; }
  main { max-width: 1300px; margin: 0 auto; padding: 16px 20px 40px; }
  .muted { color: #5B6B7A; } .small { font-size: 12.5px; }
  .panel { background: #fff; border: 1px solid #D9E0E6; border-radius: 6px; margin-bottom: 14px; }
  .banner { padding: 12px 16px; border-radius: 6px; background: #fff; border: 1px solid #D9E0E6; margin-bottom: 14px; }
  .banner.event { background: #FBEAE6; border-left: 5px solid #D6361F; font-weight: 600; }
  .banner.watch { background: #FBF3E0; border-left: 5px solid #B7791F; }
  .grid { display: grid; grid-template-columns: 1.6fr 1fr; gap: 14px; }
  @media (max-width: 960px) { .grid { grid-template-columns: 1fr; } }
  #map { height: 460px; border-radius: 0 0 6px 6px; }
  .item { padding: 10px 16px; border-top: 1px solid #EEF1F4; } .item:first-child { border-top: 0; }
  .item.confirmed { border-left: 3px solid #D6361F; } .item.investigating { border-left: 3px solid #B7791F; }
  .scroll { max-height: 520px; overflow: auto; }
  img { width: 100%; display: block; padding: 10px 14px 14px; box-sizing: border-box; }
  table { border-collapse: collapse; font-size: 12.5px; margin-top: 4px; } td, th { padding: 2px 10px 2px 0; text-align: left; }
</style></head><body>
<header><div><h1>Earthquake monitor</h1><div class="muted small" id="server"></div></div><div class="muted" id="live">Connecting</div></header>
<main>
  <div class="banner" id="banner">Waiting for the first cycle (about 20 s).</div>
  <div class="grid">
    <div class="panel"><h2>Stations and epicentres <span class="muted small" id="mapnote"></span></h2><div id="map"></div></div>
    <div class="panel"><h2>Events</h2><div class="scroll" id="events"></div></div>
  </div>
  <div class="grid">
    <div><div class="panel"><h2>Waveforms</h2><img id="p1" alt="waveforms"></div>
         <div class="panel"><h2>STA/LTA</h2><img id="p2" alt="sta/lta"></div></div>
    <div class="panel"><h2>Station picks</h2><div class="scroll" id="picks"></div></div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const STATE_COLOR = { triggered: "#D6361F", quiet: "#1F7A55", stale: "#B7791F", nodata: "#9AA5B1", noisy: "#7C3AED" };
let lastId = -1, data = null, serverClock = { t: 0, at: performance.now() };
const now = () => serverClock.t + (performance.now() - serverClock.at) / 1000;
const ago = s => { s = Math.max(0, Math.round(s)); return s < 60 ? s + " s ago" : s < 3600 ? Math.floor(s / 60) + " min ago" : Math.floor(s / 3600) + " h ago"; };

const map = L.map("map").setView([28.2, 84.5], 7);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "&copy; OpenStreetMap contributors" }).addTo(map);
const layer = L.layerGroup().addTo(map);
let fitted = false;

function renderMap(d) {
  layer.clearLayers();
  const pts = [];
  d.stations.forEach(s => {
    pts.push([s.lat, s.lon]);
    L.circleMarker([s.lat, s.lon], { radius: 7, color: "#fff", weight: 2, fillColor: STATE_COLOR[s.state], fillOpacity: 1 })
      .bindTooltip(s.station, { permanent: true, direction: "right", offset: [6, 0] }).addTo(layer);
  });
  d.events.filter(e => e.status !== "dismissed").forEach(e => {
    const col = e.status === "confirmed" ? "#D6361F" : "#B7791F";
    L.circleMarker([e.lat, e.lon], { radius: 11, color: col, weight: 3, fillColor: col, fillOpacity: .25 })
      .bindPopup(`<b>${e.status}</b><br>${e.clock} UTC<br>${e.lat}, ${e.lon}, ${e.depth} km<br>RMS ${e.rms} s`).addTo(layer);
    e.stations.forEach(n => { const s = d.stations.find(x => x.station === n);
      if (s) L.polyline([[e.lat, e.lon], [s.lat, s.lon]], { color: col, weight: 1, dashArray: "4 4" }).addTo(layer); });
  });
  $("mapnote").textContent = `${d.counts.active}/${d.counts.total} stations active`;
  if (!fitted && pts.length) { fitted = true; map.fitBounds(pts, { padding: [50, 50], maxZoom: 9 }); }
}

function renderEvents(d) {
  const list = d.events.filter(e => e.status !== "dismissed");
  $("events").innerHTML = list.length ? list.map(e => `<div class="item ${e.status}">
      <b>${e.status === "confirmed" ? "Confirmed" : "Investigating"}</b> ${e.clock} UTC <span class="muted small">(${ago(now() - e.t0)})</span><br>
      ${e.lat}, ${e.lon}, depth ${e.depth} km<br><span class="small muted">${e.reason}</span>
      <details><summary class="small muted">Picks</summary><table><tr><th>Station</th><th>Pick (UTC)</th><th>Residual</th></tr>
      ${e.stations.map(n => `<tr><td>${n}</td><td>${new Date(e.picks[n] * 1000).toISOString().substr(11, 8)}</td><td>${e.residuals[n]} s</td></tr>`).join("")}</table></details>
    </div>`).join("") : `<div class="item muted small">No events. A candidate needs ${d.settings.cluster_min} compatible picks, confirmation needs ${d.settings.min_stations} with a good fit.</div>`;
}

function renderPicks(d) {
  $("picks").innerHTML = d.pick_log.length ? d.pick_log.map(p => `<div class="item">
      <span style="color:${d.settings.colors[p.station]}"><b>${p.station}</b></span> ${p.clock} UTC
      <span class="muted small">${ago(now() - p.t)}</span><br>
      <span class="small muted">STA/LTA ${p.peak_ratio}, lasted ${p.duration} s</span></div>`).join("")
    : `<div class="item muted small">No picks yet. Trigger at STA/LTA above ${d.settings.thresh_on}.</div>`;
}

function renderBanner() {
  if (!data || !data.updated) return;
  const b = $("banner"), t = now();
  const conf = data.events.find(e => e.status === "confirmed" && t - e.t0 < 900);
  const inv = data.events.filter(e => e.status === "investigating");
  if (conf) { b.className = "banner event"; b.textContent = `Earthquake confirmed at ${conf.clock} UTC, ${conf.lat}, ${conf.lon}, depth ${conf.depth} km (${conf.stations.length} stations)`; }
  else if (inv.length) { b.className = "banner watch"; b.textContent = `${inv.length} candidate(s) under investigation: ${inv[0].reason}`; }
  else { b.className = "banner"; b.textContent = "No earthquakes in the last 15 minutes."; }
  const age = t - data.updated_ts;
  $("live").textContent = (age > data.settings.update_seconds * 2.5 ? "Stale, updated " : "Live, updated ") + ago(age);
}

function setImg(el, url) { const n = new Image(); n.onload = () => el.src = url; n.src = url; }

function render(d) {
  data = d;
  $("server").textContent = "SeedLink " + d.settings.server;
  if (!d.updated) return;
  renderMap(d); renderEvents(d); renderPicks(d); renderBanner();
  setImg($("p1"), "/plot/waveforms?v=" + d.id); setImg($("p2"), "/plot/ratio?v=" + d.id);
}

async function poll() {
  try {
    const d = await (await fetch("/api/latest?since=" + lastId)).json();
    serverClock = { t: d.server_time, at: performance.now() };
    if (!d.unchanged) { lastId = d.id; render(d); }
  } catch (e) { $("live").textContent = "Server unreachable"; }
}
poll(); setInterval(poll, 5000); setInterval(renderBanner, 1000);
</script></body></html>"""


@app.route("/")
def index():
    return PAGE


@app.route("/api/latest")
def latest():
    since = request.args.get("since", -1, type=int)
    with STATE_LOCK:
        if STATE["id"] == since:
            return jsonify({"unchanged": True, "server_time": time.time()})
        payload = build_payload()
    return jsonify(payload)


@app.route("/plot/<name>")
def plot(name):
    with STATE_LOCK:
        png = PLOTS.get(name)
    if png is None:
        return Response("plot not ready", status=404)
    return Response(png, mimetype="image/png", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    load_travel_tables()
    load_history()
    threading.Thread(target=processing_loop, daemon=True).start()
    print(f"Open http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)