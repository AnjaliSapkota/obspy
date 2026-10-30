import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, Response, jsonify, request
from matplotlib.ticker import FuncFormatter, MultipleLocator
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.seedlink import Client as SeedLinkClient
from scipy.signal import butter, correlate, correlation_lags, filtfilt, hilbert, spectrogram, windows


SERVER_HOST = "ring.wscada.net"
SERVER_PORT = 18000
FDSN_URL = "https://seiscomp.alertnepal.online"   # only used to look up missing station coordinates

# lat/lon are needed for the map. Stations with lat/lon = None are looked up from the FDSN
# service at startup; if that fails they are simply left off the map.
STATIONS = [
    {"net": "NP", "sta": "EQM10", "cha": "EHZ", "loc": "*", "lat": 28.299517, "lon": 83.960148},
    {"net": "NP", "sta": "EQM26", "cha": "EHZ", "loc": "*", "lat": 27.3936969, "lon": 86.6791885},
    {"net": "NP", "sta": "EQM07", "cha": "EHZ", "loc": "*", "lat": 27.814941, "lon": 86.713401},
    {"net": "NP", "sta": "EQM08", "cha": "EHZ", "loc": "*", "lat": 27.831000, "lon": 86.65},
    {"net": "NP", "sta": "EQM11", "cha": "EHZ", "loc": "*", "lat": 29.2859556, "lon": 81.2741972},
    {"net": "NP", "sta": "EQM22", "cha": "EHZ", "loc": "*", "lat": 28.895805, "lon": 80.227341},
    {"net": "AM", "sta": "R0F23", "cha": "EHZ", "loc": "*", "lat": 28.045045, "lon": 84.072371},
    {"net": "AM", "sta": "R02C2", "cha": "EHZ", "loc": "*", "lat": 27.621622, "lon": 85.320148},

]

WINDOW_SECONDS = 300      # data shown / analysed each cycle
LATENCY_SECONDS = 5       # stay slightly behind "now" so data has arrived
UPDATE_SECONDS = 20       # how often to refresh
ACTIVE_MAX_LATENCY = 60   # a station is "active" if its newest sample is at most this old (s)

FS = 50.0                 # common sampling rate (Hz)
FMIN, FMAX = 1.0, 8.0     # bandpass (Hz), FMAX must be < FS/2
ENVELOPE_CUTOFF = 0.5     # envelope smoothing (Hz)

THRESHOLD_RATIO = 8.0     # trigger when energy > THRESHOLD_RATIO x baseline energy
MIN_DURATION = 20.0        # s, ignore shorter triggers
MERGE_GAP = 5.0           # s, merge triggers closer than this
EDGE_SECONDS = 5.0        # s, ignore window edges (filter transients)
DEDUP_SECONDS = 20.0      # same station + onset within this = same event

COINCIDENCE_SECONDS = 60.0   # onsets within this window belong to the same candidate
SEGMENT_PAD = 10.0        # s added around the detections before cross-correlating

MIN_STATIONS = 3              # stations needed before a candidate can be confirmed
INVESTIGATE_SECONDS = 120.0   # unconfirmed candidates are dismissed this long after their onset
CONFIRM_MIN_MEAN_CC = 0.5     # mean envelope correlation over all pairs
CONFIRM_MIN_PAIR_CC = 0.3     # weakest pair must still correlate
CONFIRM_MAX_CLOSURE = 3.0     # s, lag(a,c) - lag(a,b) - lag(b,c) must be consistent

MAX_LOG = 100
MAX_EVENT_PLOTS = 15
HISTORY_FILE = Path("event_history.json")   # detections + candidates survive restarts
PORT = 8000

# One fixed colour per station, used in the plots and in the web UI.
STATION_PALETTE = ["#0F766E", "#2563EB", "#7C3AED", "#92400E", "#0891B2", "#4D7C0F", "#475569", "#C026D3"]
STATION_COLORS = {s["sta"]: STATION_PALETTE[i % len(STATION_PALETTE)] for i, s in enumerate(STATIONS)}

INK, SLATE, GRID, ALERT = "#14202B", "#5B6B7A", "#E3E8ED", "#D6361F"

plt.rcParams.update({
    "font.size": 9,
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#C9D1D9", "axes.labelcolor": SLATE, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": SLATE, "ytick.color": SLATE,
})


def fmt_time(t):
    return UTCDateTime(t).strftime("%Y-%m-%d %H:%M:%S") + " UTC"


def fmt_clock(t):
    return UTCDateTime(t).strftime("%H:%M:%S")


def fmt_npt(t):
    return (UTCDateTime(t) + 5.75 * 3600).strftime("%H:%M:%S") + " NPT"

def fill_missing_coordinates():
    missing = [s for s in STATIONS if s.get("lat") is None or s.get("lon") is None]
    if not missing:
        return
    try:
        client = FDSNClient(FDSN_URL, timeout=15)
    except Exception as error:
        print(f"Coordinate lookup unavailable: {error}")
        return
    for station in missing:
        try:
            inventory = client.get_stations(network=station["net"], station=station["sta"], level="station")
            station["lat"] = float(inventory[0][0].latitude)
            station["lon"] = float(inventory[0][0].longitude)
            print(f"Found coordinates for {station['sta']}: {station['lat']:.4f}, {station['lon']:.4f}")
        except Exception as error:
            print(f"No coordinates for {station['sta']} ({error}); add lat/lon to STATIONS to show it on the map.")


def fetch_waveform(station, start, end):
    client = SeedLinkClient(SERVER_HOST, port=SERVER_PORT, timeout=20)
    loc = "??" if station["loc"] == "*" else station["loc"]
    stream = client.get_waveforms(station["net"], station["sta"], loc, station["cha"], start, end)
    if len(stream) == 0:
        raise RuntimeError("no data")

    stream.merge(method=1, fill_value=0)
    stream.detrend("linear")
    stream.detrend("demean")
    stream.taper(max_percentage=0.05, type="hann", side = "left")
    stream.resample(sampling_rate=FS)
    stream.filter("bandpass", freqmin=FMIN, freqmax=FMAX, corners=4, zerophase=False)
    return stream[0]


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

def get_hilbert_envelope(trace, smooth_cutoff):
    signal = trace.data.astype(float)
    signal = signal - np.mean(signal)
    envelope = np.abs(hilbert(signal))
    if smooth_cutoff is not None:
        fs = trace.stats.sampling_rate
        b, a = butter(N=2, Wn=smooth_cutoff / (0.5 * fs), btype="low")
        envelope = np.maximum(filtfilt(b, a, envelope), 0)
    return envelope

def find_segments(mask):
    """Return [start, end) index pairs of contiguous True runs."""
    padded = np.concatenate(([0], mask.astype(int), [0]))
    d = np.diff(padded)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))

def merge_segments(segments, max_gap):
    merged = []
    for start, end in segments:
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged

def detect_events(name, trace, envelope):
    empty = {"detections": [], "threshold_amp": None, "max_ratio": None, "ratio": None}
    fs = trace.stats.sampling_rate
    energy = envelope ** 2
    edge = int(EDGE_SECONDS * fs)
    if len(energy) < 4 * edge:
        return empty

    baseline = np.median(energy[edge:-edge])
    if baseline <= 0:
        return empty

    ratio = energy / baseline
    mask = ratio > THRESHOLD_RATIO
    mask[:edge] = False
    mask[-edge:] = False

    detections = []
    for s, e in merge_segments(find_segments(mask), int(MERGE_GAP * fs)):
        if (e - s) / fs < MIN_DURATION:
            continue
        detections.append({
            "station": name,
            "onset": trace.stats.starttime + s / fs,
            "end": trace.stats.starttime + e / fs,
            "peak_ratio": float(ratio[s:e].max()),
        })

    return {
        "detections": detections,
        "threshold_amp": float(np.sqrt(baseline * THRESHOLD_RATIO)),
        "max_ratio": float(ratio[edge:-edge].max()),
        "ratio": ratio,
    }


def update_detection_log(log, detections):
    new_entries = []
    for d in detections:
        t = d["onset"].timestamp
        if any(x["station"] == d["station"] and abs(x["t"] - t) < DEDUP_SECONDS for x in log):
            continue
        entry = {
            "station": d["station"],
            "t": t,
            "time": fmt_time(d["onset"]),
            "clock": fmt_clock(d["onset"]),
            "npt": fmt_npt(d["onset"]),
            "duration": round(d["end"] - d["onset"], 1),
            "peak_ratio": round(d["peak_ratio"], 1),
        }
        log.insert(0, entry)
        new_entries.append(entry)
        print(f"  >> Station {entry['station']} detected event at {entry['time']} "
              f"(peak energy ratio {entry['peak_ratio']})")
    del log[MAX_LOG:]
    return new_entries


def find_coincident_groups(detections, min_stations):
    dets = sorted(detections, key=lambda d: d["onset"])
    groups = []
    i = 0
    while i < len(dets):
        group = [d for d in dets[i:] if d["onset"] - dets[i]["onset"] <= COINCIDENCE_SECONDS]
        if len({d["station"] for d in group}) >= min_stations:
            groups.append(group)
            i += len(group)
        else:
            i += 1
    return groups


def calculate_envelope_lag(env_a, env_b, fs, max_shift_seconds):
    n = min(len(env_a), len(env_b))
    env_a, env_b = env_a[:n], env_b[:n]

    crop = int(0.05 * n)
    if crop > 0:
        env_a, env_b = env_a[crop:-crop], env_b[crop:-crop]

    env_a = env_a - np.mean(env_a)
    env_b = env_b - np.mean(env_b)
    taper = windows.tukey(len(env_a), alpha=0.05)
    a, b = env_a * taper, env_b * taper

    norm = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if norm == 0:
        raise RuntimeError("envelope has zero energy")

    corr = correlate(a, b, mode="full") / norm
    lags = correlation_lags(len(a), len(b), mode="full")

    max_shift = int(round(max_shift_seconds * fs))
    valid = (lags >= -max_shift) & (lags <= max_shift)
    peak = np.argmax(corr[valid])
    return {"lag": float(lags[valid][peak] / fs), "cc_max": float(corr[valid][peak])}


def lag_closure_error(names, pairs):
    lag = {(p["a"], p["b"]): p["lag"] for p in pairs}
    worst = None
    for x, y, z in combinations(names, 3):
        if (x, y) in lag and (y, z) in lag and (x, z) in lag:
            value = abs(lag[(x, z)] - lag[(x, y)] - lag[(y, z)])
            worst = value if worst is None else max(worst, value)
    return worst


def evaluate_candidate(candidate):
    n = len(candidate["stations"])
    if n < MIN_STATIONS:
        return "investigating", f"{n} of {MIN_STATIONS} stations so far, waiting for more"

    problems = []
    if candidate["mean_cc"] < CONFIRM_MIN_MEAN_CC:
        problems.append(f"mean correlation {candidate['mean_cc']} is below {CONFIRM_MIN_MEAN_CC}")
    if candidate["min_cc"] < CONFIRM_MIN_PAIR_CC:
        problems.append(f"weakest pair {candidate['min_cc']} is below {CONFIRM_MIN_PAIR_CC}")
    if candidate["closure_error"] is not None and candidate["closure_error"] > CONFIRM_MAX_CLOSURE:
        problems.append(f"lags disagree by {candidate['closure_error']} s")
    if candidate["lag_at_limit"]:
        problems.append("a lag sits at the search limit")

    if problems:
        text = "; ".join(problems)
        return "investigating", text[0].upper() + text[1:]
    return "confirmed", f"{n} stations, mean correlation {candidate['mean_cc']}, lags consistent"


def analyze_candidate(group, traces, envelopes, end):
    first = {}
    for d in group:                      # group is sorted by onset
        first.setdefault(d["station"], d)
    names = list(first.keys())

    t0 = min(d["onset"] for d in first.values()) - SEGMENT_PAD
    t1 = max(d["end"] for d in first.values()) + SEGMENT_PAD

    segments = {}
    for name in names:
        tr = traces[name]
        fs = tr.stats.sampling_rate
        i0 = max(0, int((t0 - tr.stats.starttime) * fs))
        i1 = min(tr.stats.npts, int((t1 - tr.stats.starttime) * fs))
        segments[name] = (i0, envelopes[name][i0:i1])

    pairs = []
    for a, b in combinations(names, 2):
        if min(len(segments[a][1]), len(segments[b][1])) < 40:
            continue
        try:
            r = calculate_envelope_lag(segments[a][1], segments[b][1], FS, COINCIDENCE_SECONDS)
        except RuntimeError:
            continue
        pairs.append({"pair": f"{a}-{b}", "a": a, "b": b, "lag": r["lag"], "cc": r["cc_max"]})

    if not pairs:
        return None, None

    onset0 = min(d["onset"] for d in first.values())
    closure = lag_closure_error(names, pairs)
    candidate = {
        "id": str(int(onset0.timestamp)),
        "t0": onset0.timestamp,
        "time": fmt_time(onset0),
        "clock": fmt_clock(onset0),
        "npt": fmt_npt(onset0),
        "stations": names,                                   # in onset order
        "station_onsets": {n: fmt_clock(first[n]["onset"]) for n in names},
        "pairs": [{"pair": p["pair"], "lag": round(p["lag"], 2), "cc": round(p["cc"], 2)} for p in pairs],
        "mean_cc": round(float(np.mean([p["cc"] for p in pairs])), 2),
        "min_cc": round(float(np.min([p["cc"] for p in pairs])), 2),
        "closure_error": None if closure is None else round(closure, 2),
        "lag_at_limit": any(abs(p["lag"]) >= 0.95 * COINCIDENCE_SECONDS for p in pairs),
    }
    candidate["status"], candidate["reason"] = evaluate_candidate(candidate)
    plot = create_aligned_envelope_plot(names, segments, traces, pairs, end)
    return candidate, plot


def apply_candidates(candidates, found, now):
    changed = []
    for new, plot in found:
        match = next((i for i, c in enumerate(candidates)
                      if abs(c["t0"] - new["t0"]) <= COINCIDENCE_SECONDS), None)

        if match is None:
            new["first_seen"] = new["last_seen"] = now
            candidates.insert(0, new)
            changed.append((new, plot))
            print(f"  >> CANDIDATE {new['status'].upper()} at {new['time']}: {', '.join(new['stations'])} "
                  f"({new['reason']})")
            if new["status"] == "confirmed":
                print(f"  >> EVENT CONFIRMED at {new['time']}")
            continue

        old = candidates[match]
        old["last_seen"] = now
        if old["status"] == "confirmed":
            replace = new["status"] == "confirmed" and len(new["stations"]) > len(old["stations"])
        elif old["status"] == "dismissed":
            replace = False
        else:
            replace = len(new["stations"]) >= len(old["stations"])

        if replace:
            if new["status"] == "confirmed" and old["status"] != "confirmed":
                print(f"  >> EVENT CONFIRMED at {new['time']} ({', '.join(new['stations'])})")
            new["id"] = old["id"]
            new["first_seen"] = old["first_seen"]
            new["last_seen"] = now
            candidates[match] = new
            changed.append((new, plot))

    for c in candidates:
        if c["status"] == "investigating" and now - c["t0"] > INVESTIGATE_SECONDS:
            c["status"] = "dismissed"
            c["reason"] = "Not confirmed in time. " + c["reason"]
            print(f"  >> Candidate at {c['time']} dismissed")

    del candidates[MAX_LOG:]
    return changed

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
    """Seconds relative to 'now' (end of requested window); 0 = now."""
    return trace.times() + (trace.stats.starttime - end)


def clock_formatter(end):
    return FuncFormatter(lambda x, _: (end + float(x)).strftime("%H:%M:%S"))


def style_time_axis(ax, end):
    """Fixed live window with UTC clock labels."""
    ax.set_xlim(-WINDOW_SECONDS, 0)
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.xaxis.set_major_formatter(clock_formatter(end))


def make_rows(n, height_per_row):
    fig, axes = plt.subplots(n, 1, figsize=(12, height_per_row * n + 0.7), sharex=True,
                             gridspec_kw={"hspace": 0.12})
    return fig, np.atleast_1d(axes)


def label_station(ax, name, extra=None):
    ax.text(0.006, 0.93, name, transform=ax.transAxes, va="top", ha="left", fontweight="bold",
            fontsize=10, color=STATION_COLORS.get(name, INK),
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8})
    if extra:
        ax.text(0.994, 0.93, extra, transform=ax.transAxes, va="top", ha="right", fontsize=8, color=SLATE,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8})


def shade_detections(ax, name, detections, end):
    for d in detections:
        if d["station"] == name:
            ax.axvspan(d["onset"] - end, d["end"] - end, color=ALERT, alpha=0.16, linewidth=0)
            ax.axvline(d["onset"] - end, color=ALERT, linewidth=1.2)


def create_waveform_plot(traces, detections, end):
    fig, axes = make_rows(len(traces), 1.25)
    for ax, (name, tr) in zip(axes, traces.items()):
        ax.plot(time_axis(tr, end), tr.data, color=STATION_COLORS[name], linewidth=0.55)
        peak = float(np.max(np.abs(tr.data))) or 1.0
        ax.set_ylim(-peak * 1.15, peak * 1.15)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", visible=False)
        shade_detections(ax, name, detections, end)
        label_station(ax, name, f"peak {peak:.3g}")
    style_time_axis(axes[-1], end)
    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle(f"Waveforms, {FMIN:g}-{FMAX:g} Hz", x=0.125, ha="left", y=0.995, fontsize=11,
                 fontweight="semibold", color=INK)
    return image_to_bytes(fig)


def create_spectrogram_plot(traces, detections, end):
    fig, axes = make_rows(len(traces), 1.5)
    for ax, (name, tr) in zip(axes, traces.items()):
        fs = tr.stats.sampling_rate
        data = tr.data.astype(float)
        nperseg = int(min(256, max(16, len(data) // 4)))
        f, t, power = spectrogram(data, fs=fs, nperseg=nperseg, noverlap=int(nperseg * 0.8))
        band = (f >= FMIN) & (f <= FMAX)
        db = 10 * np.log10(power[band] + 1e-20)
        x0 = time_axis(tr, end)[0]
        ax.pcolormesh(t + x0, f[band], db, shading="auto", cmap="viridis", rasterized=True,
                      vmin=np.percentile(db, 5), vmax=np.percentile(db, 99.5))
        ax.grid(False)
        ax.set_ylim(FMIN, FMAX)
        ax.set_ylabel("Hz", fontsize=8)
        for d in detections:
            if d["station"] == name:
                ax.axvline(d["onset"] - end, color="white", linewidth=2.4)
                ax.axvline(d["onset"] - end, color=ALERT, linewidth=1.2)
        label_station(ax, name)
    style_time_axis(axes[-1], end)
    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle("Spectrograms (dB, colour scaled per station; red line = detection onset)",
                 x=0.125, ha="left", y=0.995, fontsize=11, fontweight="semibold", color=INK)
    return image_to_bytes(fig)


def create_envelope_plot(traces, envelopes, thresholds, detections, end):
    fig, axes = make_rows(len(traces), 1.25)
    for ax, (name, tr) in zip(axes, traces.items()):
        x = time_axis(tr, end)
        color = STATION_COLORS[name]
        ax.fill_between(x, 0, envelopes[name], color=color, alpha=0.22, linewidth=0)
        ax.plot(x, envelopes[name], color=color, linewidth=1)
        top = max(float(np.max(envelopes[name])), thresholds.get(name) or 0) * 1.2 or 1.0
        ax.set_ylim(0, top)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", visible=False)
        if thresholds.get(name) is not None:
            ax.axhline(thresholds[name], color=ALERT, linestyle="--", linewidth=1)
            ax.text(0.994, thresholds[name] / top, "threshold", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=7.5, color=ALERT)
        shade_detections(ax, name, detections, end)
        label_station(ax, name)
    style_time_axis(axes[-1], end)
    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle("Hilbert envelopes (smoothed)", x=0.125, ha="left", y=0.995, fontsize=11,
                 fontweight="semibold", color=INK)
    return image_to_bytes(fig)


def create_ratio_plot(traces, ratios, detections, end):
    fig, ax = plt.subplots(figsize=(12, 4.4))
    drawn = 0
    for name, tr in traces.items():
        ratio = ratios.get(name)
        if ratio is None:
            continue
        ax.semilogy(time_axis(tr, end), np.maximum(ratio, 1e-2), color=STATION_COLORS[name],
                    linewidth=1.1, label=name)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return create_empty_plot("Not enough data to compute the trigger ratio")

    ax.axhline(THRESHOLD_RATIO, color=ALERT, linestyle="--", linewidth=1.2)
    ax.text(-WINDOW_SECONDS + 2, THRESHOLD_RATIO * 1.12, f"trigger at {THRESHOLD_RATIO:g}x median energy",
            color=ALERT, fontsize=8.5, va="bottom")
    ax.axvspan(-WINDOW_SECONDS, -WINDOW_SECONDS + EDGE_SECONDS, color="#EEF1F4", linewidth=0)
    ax.axvspan(-EDGE_SECONDS, 0, color="#EEF1F4", linewidth=0)
    for d in detections:
        ax.axvline(d["onset"] - end, color=STATION_COLORS[d["station"]], linestyle=":", linewidth=1)

    top = max([float(np.max(r)) for r in ratios.values() if r is not None] + [100.0]) * 1.5
    ax.set_ylim(0.2, top)
    style_time_axis(ax, end)
    ax.set_xlabel("Time (UTC), grey = ignored edges")
    ax.set_ylabel("Energy / median energy")
    ax.set_title("Trigger ratio")
    ax.legend(ncol=len(traces), loc="upper right", frameon=False, fontsize=9)
    return image_to_bytes(fig)


def create_aligned_envelope_plot(names, segments, traces, pairs, end):
    n = len(names)
    ref = names[0]
    shifts = {ref: 0.0}
    for p in pairs:
        if p["a"] == ref:
            shifts[p["b"]] = p["lag"]       # shifting b by +lag aligns it with the reference

    fig, axes = plt.subplots(2, 1, figsize=(12, 1.0 * n * 2 + 2.6), gridspec_kw={"hspace": 0.55})
    all_x = []
    for ax, aligned in zip(axes, (False, True)):
        positions, labels, colors = [], [], []
        for i, name in enumerate(names):
            i0, env = segments[name]
            tr = traces[name]
            fs = tr.stats.sampling_rate
            shift = shifts.get(name, 0.0) if aligned else 0.0
            x = (tr.stats.starttime - end) + (i0 + np.arange(len(env))) / fs + shift
            y = env / max(float(np.max(env)), 1e-12)
            base = n - 1 - i
            ax.fill_between(x, base, base + 0.88 * y, color=STATION_COLORS[name], alpha=0.28, linewidth=0)
            ax.plot(x, base + 0.88 * y, color=STATION_COLORS[name], linewidth=1.1)
            all_x += [x[0], x[-1]]
            positions.append(base + 0.4)
            labels.append(f"{name}  {shift:+.1f} s" if aligned and name != ref else name)
            colors.append(STATION_COLORS[name])
        order = np.argsort(positions)
        ax.set_yticks([positions[k] for k in order])
        ax.set_yticklabels([labels[k] for k in order])
        for tick, k in zip(ax.get_yticklabels(), order):
            tick.set_color(colors[k])
            tick.set_fontweight("bold")
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title("As recorded" if not aligned else f"Aligned to {ref} using envelope cross-correlation")
        ax.xaxis.set_major_formatter(clock_formatter(end))

    for ax in axes:
        ax.set_xlim(min(all_x), max(all_x))
        step = max(5, int(round((max(all_x) - min(all_x)) / 8 / 5)) * 5)
        ax.xaxis.set_major_locator(MultipleLocator(step))
    axes[-1].set_xlabel("Time (UTC, reference station clock)")
    return image_to_bytes(fig)


def station_info(station):
    return {"station": station["sta"], "net": station["net"], "cha": station["cha"],
            "color": STATION_COLORS[station["sta"]], "lat": station.get("lat"), "lon": station.get("lon")}


def run_cycle():
    end = UTCDateTime() - LATENCY_SECONDS
    start = end - WINDOW_SECONDS
    print(f"\n[{fmt_time(UTCDateTime())}] cycle, window {start.strftime('%H:%M:%S')} -> {end.strftime('%H:%M:%S')}")

    traces, errors = fetch_all(STATIONS, start, end)
    for name, error in errors.items():
        print(f"  {name}: {error}")

    envelopes = {n: get_hilbert_envelope(t, ENVELOPE_CUTOFF) for n, t in traces.items()}

    detections, thresholds, ratios, station_status = [], {}, {}, []
    for station in STATIONS:
        name = station["sta"]
        info = station_info(station)
        if name not in traces:
            info.update(state="nodata", active=False, note=errors.get(name, "Station did not respond"))
            station_status.append(info)
            continue

        result = detect_events(name, traces[name], envelopes[name])
        detections += result["detections"]
        thresholds[name] = result["threshold_amp"]
        ratios[name] = result["ratio"]

        latency = UTCDateTime() - traces[name].stats.endtime
        active = latency <= ACTIVE_MAX_LATENCY
        if not active:
            state = "stale"
        else:
            state = "triggered" if result["detections"] else "quiet"
        info.update(
            state=state, active=active,
            last_sample=fmt_clock(traces[name].stats.endtime),
            latency_s=round(latency, 1),
            max_ratio=None if result["max_ratio"] is None else round(result["max_ratio"], 1),
            note="",
        )
        station_status.append(info)

    keys = ("waveforms", "spectrograms", "envelopes", "ratio")
    if traces:
        plots = {
            "waveforms": create_waveform_plot(traces, detections, end),
            "spectrograms": create_spectrogram_plot(traces, detections, end),
            "envelopes": create_envelope_plot(traces, envelopes, thresholds, detections, end),
            "ratio": create_ratio_plot(traces, ratios, detections, end),
        }
    else:
        plots = {k: create_empty_plot("No station returned data in this cycle") for k in keys}

    # Candidates: two or more coincident stations. evaluate_candidate() decides the status.
    found = []
    for group in find_coincident_groups(detections, min_stations=2):
        candidate, plot = analyze_candidate(group, traces, envelopes, end)
        if candidate is not None:
            found.append((candidate, plot))

    return {"plots": plots, "detections": detections, "stations": station_status, "candidates": found}


STATE = {
    "id": 0, "updated": None, "updated_ts": None, "error": None, "cycle_seconds": None,
    "stations": [], "detection_log": [], "candidates": [],
    "settings": {
        "server": f"{SERVER_HOST}:{SERVER_PORT}",
        "threshold_ratio": THRESHOLD_RATIO, "min_stations": MIN_STATIONS,
        "coincidence_seconds": COINCIDENCE_SECONDS, "update_seconds": UPDATE_SECONDS,
        "window_seconds": WINDOW_SECONDS, "fmin": FMIN, "fmax": FMAX, "colors": STATION_COLORS,
        "investigate_seconds": INVESTIGATE_SECONDS,
    },
}
PLOTS = {}   # name -> PNG bytes, served at /plot/<name>
STATE_LOCK = threading.Lock()


def load_history():
    try:
        saved = json.loads(HISTORY_FILE.read_text())
        STATE["detection_log"] = saved.get("detection_log", [])
        STATE["candidates"] = saved.get("candidates", [])
        print(f"Loaded {len(STATE['detection_log'])} detections and {len(STATE['candidates'])} candidates "
              f"from {HISTORY_FILE}")
    except FileNotFoundError:
        pass
    except Exception as error:
        print(f"Could not read {HISTORY_FILE}: {error}")


def save_history():
    try:
        HISTORY_FILE.write_text(json.dumps({"detection_log": STATE["detection_log"],
                                            "candidates": STATE["candidates"]}))
    except Exception as error:
        print(f"Could not write {HISTORY_FILE}: {error}")


def prune_event_plots():
    keep = {f"event-{c['id']}" for c in STATE["candidates"][:MAX_EVENT_PLOTS]}
    for name in [k for k in PLOTS if k.startswith("event-") and k not in keep]:
        del PLOTS[name]


def processing_loop():
    while True:
        started = time.time()
        try:
            result = run_cycle()
            now = time.time()
            with STATE_LOCK:
                update_detection_log(STATE["detection_log"], result["detections"])
                PLOTS.update(result["plots"])
                for candidate, plot in apply_candidates(STATE["candidates"], result["candidates"], now):
                    PLOTS[f"event-{candidate['id']}"] = plot
                prune_event_plots()
                STATE.update(id=STATE["id"] + 1, updated=fmt_time(UTCDateTime()), updated_ts=now,
                             error=None, stations=result["stations"],
                             cycle_seconds=round(time.time() - started, 1))
                save_history()
        except Exception as error:
            print(f"  cycle failed: {error}")
            with STATE_LOCK:
                STATE.update(id=STATE["id"] + 1, error=str(error))
        time.sleep(max(1, UPDATE_SECONDS - (time.time() - started)))


def build_payload():
    now = time.time()
    candidates = [dict(c, has_plot=f"event-{c['id']}" in PLOTS) for c in STATE["candidates"]]
    stations = STATE["stations"]
    counts = {
        "active": sum(1 for s in stations if s.get("active")),
        "total": len(STATIONS),
        "investigating": sum(1 for c in candidates if c["status"] == "investigating"),
        "confirmed_24h": sum(1 for c in candidates if c["status"] == "confirmed" and now - c["t0"] < 86400),
        "confirmed_total": sum(1 for c in candidates if c["status"] == "confirmed"),
        "dismissed": sum(1 for c in candidates if c["status"] == "dismissed"),
        "triggers_1h": sum(1 for e in STATE["detection_log"] if now - e["t"] < 3600),
    }
    payload = dict(STATE)
    payload.update(candidates=candidates, counts=counts, server_time=now)
    return payload


# Website
app = Flask(__name__)

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Seismic monitor</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  :root {
    --paper: #EEF1F4; --panel: #FFFFFF; --ink: #14202B; --slate: #5B6B7A; --line: #D9E0E6;
    --alert: #D6361F; --alert-bg: #FBEAE6; --live: #1F7A55; --warn: #B7791F; --warn-bg: #FBF3E0; --grey: #9AA5B1;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--paper); color: var(--ink); font: 14px/1.45 "IBM Plex Sans", system-ui, sans-serif;
         font-variant-numeric: tabular-nums; }
  h1 { font-size: 18px; margin: 0; font-weight: 600; }
  h2 { font-size: 14px; font-weight: 600; margin: 0; }
  .muted { color: var(--slate); }
  .small { font-size: 12.5px; }

  header { background: var(--panel); border-bottom: 1px solid var(--line); }
  .bar { max-width: 1360px; margin: 0 auto; padding: 14px 20px; display: flex; gap: 16px; align-items: center;
         justify-content: space-between; flex-wrap: wrap; }
  .live { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--slate); }
  .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--slate); flex: none; }
  .live.ok .dot { background: var(--live); animation: pulse 2s ease-out infinite; }
  .live.stale .dot { background: var(--warn); }
  @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(31,122,85,.45); } 100% { box-shadow: 0 0 0 9px rgba(31,122,85,0); } }

  main { max-width: 1360px; margin: 0 auto; padding: 18px 20px 40px; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; }
  .panelhead { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 12px 16px;
               border-bottom: 1px solid var(--line); }
  #error { display: none; margin-bottom: 14px; padding: 10px 14px; border: 1px solid #E8B4AA; background: var(--alert-bg);
           border-radius: 6px; color: #8A2412; }

  /* KPI band */
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 14px; }
  .kpi { background: var(--panel); border: 1px solid var(--line); border-left: 4px solid var(--grey); border-radius: 6px;
         padding: 12px 16px; }
  .kpi.good { border-left-color: var(--live); } .kpi.warn { border-left-color: var(--warn); }
  .kpi.bad, .kpi.event { border-left-color: var(--alert); }
  .klabel { color: var(--slate); font-size: 13px; }
  .kval { font-size: 32px; font-weight: 600; line-height: 1.15; margin: 2px 0; }
  .kpi.good .kval { color: var(--live); } .kpi.warn .kval { color: var(--warn); }
  .kpi.bad .kval, .kpi.event .kval { color: var(--alert); }
  .ksub { color: var(--slate); font-size: 12.5px; }

  .banner { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; padding: 13px 16px; margin-bottom: 14px;
            border-radius: 6px; border: 1px solid var(--line); background: var(--panel); color: var(--slate); }
  .banner.event { background: var(--alert-bg); border-color: #E8B4AA; border-left: 5px solid var(--alert); color: var(--ink); }
  .banner.watch { background: var(--warn-bg); border-color: #EBD6A4; border-left: 5px solid var(--warn); color: var(--ink); }
  .banner .big { font-size: 19px; font-weight: 600; }
  .banner.event .big { color: var(--alert); } .banner.watch .big { color: var(--warn); }

  /* map + candidates */
  .top { display: grid; grid-template-columns: minmax(0, 1.65fr) minmax(300px, 1fr); gap: 14px; margin-bottom: 14px;
         align-items: start; }
  #map { height: 470px; background: #E3E8ED; border-radius: 0 0 6px 6px; }
  .sidescroll { max-height: 530px; overflow: auto; }
  .stack > .panel + .panel { margin-top: 14px; }
  .linkbtn { font: inherit; font-size: 12.5px; background: none; border: 1px solid var(--line); border-radius: 4px;
             padding: 3px 9px; color: var(--ink); cursor: pointer; }
  .linkbtn:hover { background: #F4F6F8; }
  button:focus-visible, summary:focus-visible { outline: 2px solid #2563EB; outline-offset: 2px; }

  /* map markers */
  .mk { position: relative; width: 16px; height: 16px; }
  .mk i { display: block; width: 16px; height: 16px; border-radius: 50%; background: var(--live); border: 2px solid #fff;
          box-shadow: 0 0 0 1px rgba(20,32,43,.4); }
  .mk.s-stale i { background: var(--warn); } .mk.s-nodata i { background: var(--grey); }
  .mk.s-triggered i { background: var(--alert); animation: ring 1.6s ease-out infinite; }
  .mk.inev i { box-shadow: 0 0 0 3px rgba(214,54,31,.55); }
  .mk b { position: absolute; left: 21px; top: -1px; font-size: 12px; font-weight: 600; white-space: nowrap;
          background: rgba(255,255,255,.9); padding: 0 5px; border-radius: 3px; color: var(--ink); }
  @keyframes ring { 0% { box-shadow: 0 0 0 0 rgba(214,54,31,.6), 0 0 0 1px rgba(20,32,43,.4); }
                    100% { box-shadow: 0 0 0 14px rgba(214,54,31,0), 0 0 0 1px rgba(20,32,43,.4); } }
  .legend { display: flex; gap: 14px; flex-wrap: wrap; padding: 8px 16px; border-top: 1px solid var(--line); font-size: 12.5px;
            color: var(--slate); }
  .legend span { display: inline-flex; align-items: center; gap: 5px; }
  .legend i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .leaflet-popup-content { font: 13px/1.45 "IBM Plex Sans", system-ui, sans-serif; margin: 10px 12px; }

  /* station strip */
  .strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1px; background: var(--line);
           border: 1px solid var(--line); border-radius: 6px; overflow: hidden; margin-bottom: 14px; }
  .cell { background: var(--panel); padding: 12px 14px; }
  .cell .tp { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
  .swatch { width: 10px; height: 10px; border-radius: 2px; flex: none; }
  .cell .name { font-weight: 600; }
  .chip { margin-left: auto; font-size: 12px; padding: 1px 8px; border-radius: 10px; background: #E9EEF2; color: var(--slate); }
  .chip.triggered { background: var(--alert); color: #fff; }
  .chip.quiet { background: #DDEFE6; color: #17603F; }
  .chip.stale { background: #F3E9D2; color: #7A5A12; } .chip.nodata { background: #E9EEF2; color: var(--slate); }
  .gauge { position: relative; height: 6px; background: #E6EBF0; border-radius: 3px; }
  .gauge .fill { position: absolute; inset: 0 auto 0 0; border-radius: 3px; }
  .gauge .mark { position: absolute; top: -4px; bottom: -4px; width: 2px; background: var(--ink); }
  .cell .meta { display: flex; justify-content: space-between; margin-top: 8px; font-size: 12.5px; color: var(--slate); gap: 8px; }

  /* plots */
  .layout { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 14px; align-items: start; }
  @media (max-width: 1020px) { .layout, .top { grid-template-columns: 1fr; } #map { height: 380px; } }
  .tabs { display: flex; gap: 2px; border-bottom: 1px solid var(--line); padding: 0 8px; overflow-x: auto; }
  .tabs button { font: inherit; background: none; border: 0; padding: 12px 14px; color: var(--slate); cursor: pointer;
                 border-bottom: 2px solid transparent; margin-bottom: -1px; white-space: nowrap; }
  .tabs button[aria-selected="true"] { color: var(--ink); font-weight: 600; border-bottom-color: var(--ink); }
  .tabs button:hover { color: var(--ink); }
  .plotwrap { padding: 12px 14px 16px; min-height: 320px; }
  .plotwrap img { width: 100%; height: auto; display: block; }
  .empty { padding: 30px 16px; text-align: center; color: var(--slate); }
  #eventpanel { margin-top: 14px; }

  /* lists */
  .item { padding: 11px 16px; border-top: 1px solid var(--line); }
  .item:first-child { border-top: 0; }
  .item.investigating { border-left: 3px solid var(--warn); } .item.confirmed { border-left: 3px solid var(--alert); }
  .chips { display: flex; flex-wrap: wrap; gap: 5px; margin: 6px 0; }
  .tag { font-size: 12px; padding: 1px 7px; border-radius: 3px; color: #fff; }
  .row { display: flex; align-items: baseline; gap: 8px; }
  .row .when { margin-left: auto; color: var(--slate); font-size: 12.5px; }
  .status { font-size: 12px; font-weight: 600; padding: 1px 8px; border-radius: 10px; }
  .status.investigating { background: var(--warn-bg); color: #7A5A12; } .status.confirmed { background: var(--alert); color: #fff; }
  details summary { cursor: pointer; color: var(--slate); font-size: 12.5px; }
  table { width: 100%; border-collapse: collapse; margin-top: 6px; font-size: 12.5px; }
  th, td { text-align: left; padding: 3px 4px 3px 0; border-bottom: 1px solid #EEF1F4; }
  th { color: var(--slate); font-weight: 500; }
  .feed { max-height: 460px; overflow: auto; }
  @media (prefers-reduced-motion: reduce) { .live.ok .dot, .mk.s-triggered i { animation: none; } }
</style></head>
<body>
<header><div class="bar">
  <div><h1>Seismic monitor</h1><div class="muted small" id="server"></div></div>
  <div class="live" id="live"><span class="dot"></span><span id="livetext">Connecting</span></div>
</div></header>

<main>
  <div id="error"></div>
  <section class="kpis" id="kpis"></section>
  <div class="banner" id="banner">Waiting for the first data. The first cycle can take about 20 seconds.</div>

  <div class="top">
    <div class="panel">
      <div class="panelhead"><h2>Station map</h2>
        <span class="muted small" id="mapnote"></span>
        <button class="linkbtn" id="fit">Fit to stations</button></div>
      <div id="map"></div>
      <div class="legend">
        <span><i style="background:var(--live)"></i>Active</span>
        <span><i style="background:var(--alert)"></i>Triggered</span>
        <span><i style="background:var(--warn)"></i>Stale data</span>
        <span><i style="background:var(--grey)"></i>No data</span>
        <span><i style="background:#fff;box-shadow:0 0 0 2px rgba(214,54,31,.6)"></i>In a current candidate</span>
      </div>
    </div>
    <div class="stack">
      <div class="panel">
        <div class="panelhead"><h2>Under investigation</h2><span class="muted small" id="invcount"></span></div>
        <div class="sidescroll" style="max-height:230px" id="investigating"></div>
      </div>
      <div class="panel">
        <div class="panelhead"><h2>Confirmed detections</h2><span class="muted small" id="confcount"></span></div>
        <div class="sidescroll" style="max-height:230px" id="confirmed"></div>
      </div>
    </div>
  </div>

  <div class="strip" id="strip"></div>

  <div class="layout">
    <section>
      <div class="panel">
        <div class="tabs" role="tablist" id="tabs"></div>
        <div class="plotwrap">
          <div class="muted small" id="tabnote" style="margin-bottom:8px"></div>
          <img id="plot" alt="" hidden>
          <div class="empty" id="plotempty">Plots appear after the first cycle.</div>
        </div>
      </div>
      <div class="panel" id="eventpanel" hidden>
        <div class="panelhead"><h2 id="eventtitle"></h2></div>
        <div class="plotwrap" style="min-height:0"><img id="eventimg" alt="Envelopes of the selected candidate, as recorded and aligned"></div>
      </div>
    </section>
    <aside class="panel">
      <div class="panelhead"><h2>Station detections</h2></div>
      <div class="feed" id="feed"></div>
    </aside>
  </div>
</main>

<script>
const $ = id => document.getElementById(id);
const TABS = [["waveforms", "Waveforms"], ["spectrograms", "Spectrograms"],
              ["envelopes", "Envelopes"], ["ratio", "Trigger ratio"]];
const STATE_LABEL = { triggered: "Triggered", quiet: "Active", stale: "Stale", nodata: "No data" };
let lastId = -1, data = null, tab = "waveforms", selectedEvent = null;
let clock = { serverTime: 0, at: performance.now() };

const serverNow = () => clock.serverTime + (performance.now() - clock.at) / 1000;
function ago(s) {
  s = Math.max(0, Math.round(s));
  if (s < 60) return s + " s ago";
  if (s < 3600) return Math.floor(s / 60) + " min ago";
  if (s < 86400) return Math.floor(s / 3600) + " h ago";
  return Math.floor(s / 86400) + " d ago";
}
function setImg(el, url) {                      // swap the image only once the new one has loaded
  const next = new Image();
  next.onload = () => { el.src = url; el.hidden = false; };
  next.src = url;
}

/* ---------- map ---------- */
let map = null, markers = {}, route = null, fitted = false;

function initMap() {
  if (typeof L === "undefined") {
    $("map").innerHTML = '<div class="empty">The map needs internet access to load Leaflet and the map tiles.</div>';
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
  L.control.layers({ "Street": street, "Terrain": terrain, "Satellite": satellite }, null, { position: "topright" }).addTo(map);
  $("fit").onclick = fitMap;
}
function fitMap() {
  const pts = Object.values(markers).map(m => m.getLatLng());
  if (map && pts.length) map.fitBounds(L.latLngBounds(pts), { padding: [60, 60], maxZoom: 9 });
}
function currentCandidates(d) {   // candidates worth highlighting on the map
  return d.candidates.filter(c => c.status === "investigating" ||
                                  (c.status === "confirmed" && serverNow() - c.t0 < 900));
}
function popupHtml(s) {
  const lines = [`<b>${s.net}.${s.station}</b> ${s.cha}`, `State: ${STATE_LABEL[s.state]}`];
  if (s.state === "nodata") lines.push(`<span class="muted">${s.note}</span>`);
  else lines.push(`Last sample ${s.last_sample} UTC (${s.latency_s} s behind)`, `Peak energy ${s.max_ratio ?? "–"}× median`);
  lines.push(`<span class="muted">${s.lat.toFixed(4)}, ${s.lon.toFixed(4)}</span>`);
  return lines.join("<br>");
}
function updateMap(d) {
  if (!map) return;
  const inEvent = new Set();
  const cands = currentCandidates(d);
  cands.forEach(c => c.stations.forEach(n => inEvent.add(n)));

  const placed = d.stations.filter(s => s.lat != null && s.lon != null);
  placed.forEach(s => {
    const icon = L.divIcon({ className: "", iconSize: [16, 16], iconAnchor: [8, 8],
      html: `<div class="mk s-${s.state} ${inEvent.has(s.station) ? "inev" : ""}"><i></i><b>${s.station}</b></div>` });
    if (markers[s.station]) {
      markers[s.station].setLatLng([s.lat, s.lon]).setIcon(icon).setPopupContent(popupHtml(s));
    } else {
      markers[s.station] = L.marker([s.lat, s.lon], { icon }).addTo(map).bindPopup(popupHtml(s));
    }
  });

  // dashed line through the stations of the newest candidate, in order of onset
  if (route) { map.removeLayer(route); route = null; }
  const latest = cands[0];
  if (latest) {
    const byName = Object.fromEntries(placed.map(s => [s.station, s]));
    const pts = latest.stations.filter(n => byName[n]).map(n => [byName[n].lat, byName[n].lon]);
    if (pts.length > 1) route = L.polyline(pts, { color: "#D6361F", weight: 2.5, dashArray: "6 6", opacity: .8 }).addTo(map);
  }

  const missing = d.stations.filter(s => s.lat == null || s.lon == null).map(s => s.station);
  $("mapnote").textContent = `${d.counts.active} of ${d.counts.total} stations active` +
    (missing.length ? `, no location for ${missing.join(", ")}` : "");
  if (!fitted && placed.length) { fitted = true; fitMap(); }
}

/* ---------- dashboard ---------- */
function kpi(label, value, sub, cls) {
  return `<div class="kpi ${cls}"><div class="klabel">${label}</div><div class="kval">${value}</div><div class="ksub">${sub}</div></div>`;
}
function renderKpis(d) {
  const c = d.counts, s = d.settings;
  const down = c.total - c.active;
  $("kpis").innerHTML =
    kpi("Stations active", `${c.active} / ${c.total}`, down ? `${down} without recent data` : "All reporting",
        c.active === c.total ? "good" : (c.active === 0 ? "bad" : "warn")) +
    kpi("Under investigation", c.investigating,
        c.investigating ? "Candidates waiting for confirmation" : "No pending candidates",
        c.investigating ? "warn" : "") +
    kpi("Confirmed events", c.confirmed_24h, `last 24 h, ${c.confirmed_total} in history`,
        c.confirmed_24h ? "event" : "") +
    kpi("Station triggers", c.triggers_1h, "single-station triggers in the last hour", "") +
    kpi("Last cycle", (d.cycle_seconds ?? "–") + " s", `every ${s.update_seconds} s, ${s.window_seconds / 60} min window`, "");
}

function renderStrip(d) {
  const thr = d.settings.threshold_ratio;
  const pos = r => Math.min(100, Math.max(0, Math.log10(Math.max(r, 1)) / 2 * 100));   // 1× to 100×
  $("strip").innerHTML = d.stations.map(s => {
    const head = `<div class="tp"><span class="swatch" style="background:${s.color}"></span>
      <span class="name">${s.station}</span><span class="chip ${s.state}">${STATE_LABEL[s.state]}</span></div>`;
    if (s.state === "nodata") {
      return `<div class="cell">${head}<div class="gauge"></div><div class="meta"><span>${s.note}</span></div></div>`;
    }
    const hot = s.state === "triggered";
    const fill = hot ? "var(--alert)" : (s.state === "stale" ? "var(--grey)" : s.color);
    return `<div class="cell">${head}
      <div class="gauge" title="Peak energy ratio ${s.max_ratio}× (trigger at ${thr}×)">
        <div class="fill" style="width:${pos(s.max_ratio)}%;background:${fill}"></div>
        <div class="mark" style="left:${pos(thr)}%"></div></div>
      <div class="meta"><span>Peak ${s.max_ratio ?? "–"}× of ${thr}×</span><span>${s.latency_s} s behind</span></div></div>`;
  }).join("");
}

function candidateCard(e, colors) {
  return `<div class="item ${e.status}">
    <div class="row"><span class="status ${e.status}">${e.status === "confirmed" ? "Confirmed" : "Investigating"}</span>
      <b title="${e.time} / ${e.npt}">${e.clock} UTC</b><span class="when" data-t="${e.t0}"></span></div>
    <div class="chips">${e.stations.map(n => `<span class="tag" style="background:${colors[n]}">${n}</span>`).join("")}</div>
    <div class="small ${e.status === "investigating" ? "" : "muted"}">${e.reason}</div>
    <div class="row" style="margin-top:4px">
      <details style="flex:1"><summary>Lags and onsets</summary>
        <table><tr><th>Pair</th><th>Lag (s)</th><th>CC</th></tr>
        ${e.pairs.map(p => `<tr><td>${p.pair}</td><td>${p.lag}</td><td>${p.cc}</td></tr>`).join("")}</table>
        <table><tr><th>Station</th><th>Onset (UTC)</th></tr>
        ${Object.entries(e.station_onsets).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("")}</table>
        <div class="small muted" style="margin-top:6px">Mean CC ${e.mean_cc}, weakest pair ${e.min_cc}, lag closure ${e.closure_error ?? "n/a"} s</div>
      </details>
      ${e.has_plot ? `<button class="linkbtn" data-event="${e.id}">Show plot</button>` : ""}
    </div></div>`;
}
function renderCandidates(d) {
  const colors = d.settings.colors, c = d.counts;
  const inv = d.candidates.filter(x => x.status === "investigating");
  const conf = d.candidates.filter(x => x.status === "confirmed");
  $("invcount").textContent = c.dismissed ? `${c.dismissed} dismissed` : "";
  $("confcount").textContent = conf.length ? `${conf.length} total` : "";
  $("investigating").innerHTML = inv.length ? inv.map(e => candidateCard(e, colors)).join("")
    : `<div class="item muted small">Nothing pending. Two or more stations triggering within ${d.settings.coincidence_seconds} s start a candidate.</div>`;
  $("confirmed").innerHTML = conf.length ? conf.map(e => candidateCard(e, colors)).join("")
    : `<div class="item muted small">No confirmed events yet. Confirmation needs ${d.settings.min_stations} stations with consistent envelope lags.</div>`;
  document.querySelectorAll("[data-event]").forEach(b => b.onclick = () => {
    selectedEvent = b.dataset.event; renderEventPlot(d);
    $("eventpanel").scrollIntoView({ behavior: "smooth", block: "nearest" });
  });
}
function renderEventPlot(d) {
  const withPlot = d.candidates.filter(c => c.has_plot);
  const chosen = withPlot.find(c => c.id === selectedEvent) || withPlot[0];
  $("eventpanel").hidden = !chosen;
  if (!chosen) return;
  $("eventtitle").textContent = (chosen.status === "confirmed" ? "Confirmed event, " : "Candidate, ") + chosen.time;
  setImg($("eventimg"), `/plot/event-${chosen.id}?v=${d.id}`);
}

function renderFeed(d) {
  const c = d.settings.colors;
  $("feed").innerHTML = d.detection_log.length ? d.detection_log.map(e => `
    <div class="item">
      <div class="row"><span class="swatch" style="background:${c[e.station] || "#999"}"></span><b>${e.station}</b>
        <span title="${e.time} / ${e.npt}">${e.clock} UTC</span><span class="when" data-t="${e.t}"></span></div>
      <div class="small muted">Lasted ${e.duration} s, peak energy ${e.peak_ratio}× the median</div>
    </div>`).join("")
    : `<div class="item muted small">No station has triggered. The threshold is ${d.settings.threshold_ratio}× the median energy.</div>`;
}

function renderBanner() {
  if (!data || !data.updated) return;
  const b = $("banner"), now = serverNow();
  const confirmed = data.candidates.find(c => c.status === "confirmed" && now - c.t0 < 900);
  const inv = data.candidates.filter(c => c.status === "investigating");
  if (confirmed) {
    b.className = "banner event";
    b.innerHTML = `<span class="big">Confirmed event at ${confirmed.clock} UTC</span>
      <span>${confirmed.stations.length} stations: ${confirmed.stations.join(", ")}</span>
      <span class="muted">${ago(now - confirmed.t0)}, mean correlation ${confirmed.mean_cc}</span>`;
  } else if (inv.length) {
    b.className = "banner watch";
    b.innerHTML = `<span class="big">${inv.length} candidate${inv.length > 1 ? "s" : ""} under investigation</span>
      <span>Latest at ${inv[0].clock} UTC: ${inv[0].stations.join(", ")}</span>
      <span class="muted">${inv[0].reason}</span>`;
  } else {
    b.className = "banner";
    b.textContent = "No events in the last 15 minutes and no candidates under investigation.";
  }
}

function buildTabs() {
  $("tabs").innerHTML = TABS.map(([key, label]) =>
    `<button role="tab" data-tab="${key}" aria-selected="${key === tab}">${label}</button>`).join("");
  $("tabs").querySelectorAll("button").forEach(b => b.onclick = () => { tab = b.dataset.tab; buildTabs(); showPlot(); });
}
function showPlot() {
  if (!data || !data.updated) return;
  const s = data.settings;
  const notes = {
    waveforms: `Band-passed ${s.fmin}–${s.fmax} Hz. Red shading marks detections.`,
    spectrograms: "Power in dB, scaled per station. Red lines mark detection onsets.",
    envelopes: "Smoothed Hilbert envelope. The dashed line is the detection threshold.",
    ratio: `Energy relative to each station's median. A station triggers above ${s.threshold_ratio}×.`,
  };
  $("tabnote").textContent = notes[tab];
  $("plotempty").hidden = true;
  setImg($("plot"), `/plot/${tab}?v=${data.id}`);
}

function tick() {
  const live = $("live");
  if (data && data.updated_ts) {
    const age = serverNow() - data.updated_ts;
    const stale = age > data.settings.update_seconds * 2.5;
    live.className = "live " + (stale ? "stale" : "ok");
    $("livetext").textContent = (stale ? "Stale, updated " : "Live, updated ") + ago(age);
  }
  document.querySelectorAll("[data-t]").forEach(el => el.textContent = ago(serverNow() - parseFloat(el.dataset.t)));
  renderBanner();
}

function render(d) {
  data = d;
  $("server").textContent = "SeedLink " + d.settings.server;
  const err = $("error");
  err.style.display = d.error ? "block" : "none";
  err.textContent = d.error ? "The last cycle failed: " + d.error : "";
  if (!d.updated) return;
  renderKpis(d); renderStrip(d); renderCandidates(d); renderEventPlot(d); renderFeed(d); updateMap(d); showPlot();
  tick();
}

async function poll() {
  try {
    const res = await fetch("/api/latest?since=" + lastId);
    const d = await res.json();
    clock = { serverTime: d.server_time, at: performance.now() };
    if (!d.unchanged) { lastId = d.id; render(d); }
  } catch (e) {
    $("live").className = "live stale";
    $("livetext").textContent = "Server unreachable";
  }
}

initMap(); buildTabs();
poll(); setInterval(poll, 5000); setInterval(tick, 1000);
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
    load_history()
    threading.Thread(target=fill_missing_coordinates, daemon=True).start()
    threading.Thread(target=processing_loop, daemon=True).start()
    print(f"Open http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)