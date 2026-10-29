import base64
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, request
from obspy import UTCDateTime
from obspy.clients.seedlink import Client as SeedLinkClient
from scipy.signal import butter, correlate, correlation_lags, filtfilt, hilbert, windows

SERVER_HOST = "ring.wscada.net"
SERVER_PORT = 18000

STATIONS = [
    {"net": "NP", "sta": "EQM10", "cha": "EHZ", "loc": "*"},
    {"net": "NP", "sta": "EQM13", "cha": "EHZ", "loc": "*"},
    {"net": "NK", "sta": "KKN",   "cha": "BHZ", "loc": "*"},
    {"net": "NP", "sta": "EQM07", "cha": "EHZ", "loc": "*"},
    {"net": "NP", "sta": "EQM08", "cha": "EHZ", "loc": "*"},
]

WINDOW_SECONDS = 300      # data shown / analysed each cycle
LATENCY_SECONDS = 5       # stay slightly behind "now" so data has arrived
UPDATE_SECONDS = 20       # how often to refresh

FS = 50.0                 # common sampling rate (Hz)
FMIN, FMAX = 1.0, 8.0     # bandpass (Hz), FMAX must be < FS/2
ENVELOPE_CUTOFF = 0.5     # envelope smoothing (Hz)

THRESHOLD_RATIO = 8.0     # trigger when energy > THRESHOLD_RATIO x median energy of the window
MIN_DURATION = 3.0        # s, ignore shorter triggers
MERGE_GAP = 5.0           # s, merge triggers closer than this
EDGE_SECONDS = 5.0        # s, ignore window edges (filter transients)
DEDUP_SECONDS = 20.0      # same station + onset within this = same event

MIN_STATIONS = 3          # 1 + 2 others
COINCIDENCE_SECONDS = 60.0   # onsets within this window count as the same event
SEGMENT_PAD = 10.0        # s added around the detections before cross-correlating

MAX_LOG = 50
PORT = 8000


def fmt_time(t):
    return UTCDateTime(t).strftime("%Y-%m-%d %H:%M:%S") + " UTC"



def fetch_waveform(station, start, end):
    client = SeedLinkClient(SERVER_HOST, port=SERVER_PORT, timeout=20)
    loc = "??" if station["loc"] == "*" else station["loc"]
    stream = client.get_waveforms(station["net"], station["sta"], loc,
                                  station["cha"], start, end)
    if len(stream) == 0:
        raise RuntimeError("no data")

    stream.merge(method=1, fill_value="interpolate")
    stream.detrend("linear")
    stream.detrend("demean")
    stream.taper(max_percentage=0.05, type="hann")
    stream.interpolate(sampling_rate=FS, method="linear")
    stream.filter("bandpass", freqmin=FMIN, freqmax=FMAX, corners=4, zerophase=True)
    return stream[0]


def fetch_all(stations, start, end):
    """Fetch all stations in parallel. Returns (traces, errors)."""
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
    """
    Energy = envelope^2. Baseline = median energy of the window (robust to the event).
    Trigger where energy / baseline > THRESHOLD_RATIO.
    Returns (detections, threshold_amplitude, max_ratio).
    """
    fs = trace.stats.sampling_rate
    energy = envelope ** 2
    edge = int(EDGE_SECONDS * fs)
    if len(energy) < 4 * edge:
        return [], None, None

    baseline = np.median(energy[edge:-edge])
    if baseline <= 0:
        return [], None, None

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

    threshold_amp = float(np.sqrt(baseline * THRESHOLD_RATIO))
    max_ratio = float(ratio[edge:-edge].max())
    return detections, threshold_amp, max_ratio


def update_detection_log(log, detections):
    """Add detections not already logged; print + return the new ones."""
    new_entries = []
    for d in detections:
        t = d["onset"].timestamp
        if any(x["station"] == d["station"] and abs(x["t"] - t) < DEDUP_SECONDS for x in log):
            continue
        entry = {
            "station": d["station"],
            "t": t,
            "time": fmt_time(d["onset"]),
            "duration": round(d["end"] - d["onset"], 1),
            "peak_ratio": round(d["peak_ratio"], 1),
        }
        log.insert(0, entry)
        new_entries.append(entry)
        print(f"  >> Station {entry['station']} detected event at {entry['time']} "
              f"(peak energy ratio {entry['peak_ratio']})")
    del log[MAX_LOG:]
    return new_entries


def find_coincident_groups(detections):
    """Group detections whose onsets fall within COINCIDENCE_SECONDS; keep groups with enough stations."""
    dets = sorted(detections, key=lambda d: d["onset"])
    groups = []
    i = 0
    while i < len(dets):
        group = [d for d in dets[i:] if d["onset"] - dets[i]["onset"] <= COINCIDENCE_SECONDS]
        if len({d["station"] for d in group}) >= MIN_STATIONS:
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


def analyze_candidate(group, traces, envelopes, end):
    """Envelope cross-correlation between the stations that detected the same event."""
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
    candidate = {
        "id": str(int(onset0.timestamp)),
        "t0": onset0.timestamp,
        "time": fmt_time(onset0),
        "stations": names,
        "station_onsets": {n: fmt_time(first[n]["onset"]) for n in names},
        "pairs": [{"pair": p["pair"], "lag": round(p["lag"], 2), "cc": round(p["cc"], 2)} for p in pairs],
        "mean_cc": round(float(np.mean([p["cc"] for p in pairs])), 2),
    }
    plot = create_aligned_envelope_plot(names, segments, traces, pairs, end)
    return candidate, plot


def update_candidates(candidates, candidate):
    """Insert a new candidate, or replace an existing one if it now has more stations.
    Returns True if the candidate was added/updated."""
    for i, existing in enumerate(candidates):
        if abs(existing["t0"] - candidate["t0"]) <= COINCIDENCE_SECONDS:
            if len(candidate["stations"]) > len(existing["stations"]):
                candidates[i] = candidate
                return True
            return False
    candidates.insert(0, candidate)
    del candidates[MAX_LOG:]
    print(f"  >> POSSIBLE EVENT at {candidate['time']} seen by {', '.join(candidate['stations'])} "
          f"(mean envelope CC {candidate['mean_cc']})")
    return True


def image_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def create_empty_plot(message):
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    return image_to_base64(fig)


def time_axis(trace, end):
    """Seconds relative to 'now' (end of requested window); 0 = now."""
    return trace.times() + (trace.stats.starttime - end)


def make_rows(n, height_per_row):
    fig, axes = plt.subplots(n, 1, figsize=(12, height_per_row * n + 1), sharex=True)
    return fig, np.atleast_1d(axes)


def create_waveform_plot(traces, detections, end):
    fig, axes = make_rows(len(traces), 1.5)
    for ax, (name, tr) in zip(axes, traces.items()):
        ax.plot(time_axis(tr, end), tr.data, linewidth=0.6)
        for d in detections:
            if d["station"] == name:
                ax.axvspan(d["onset"] - end, d["end"] - end, color="red", alpha=0.25)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[0].set_xlim(-WINDOW_SECONDS, 0)
    axes[-1].set_xlabel("Seconds relative to now")
    fig.suptitle(f"Live waveforms ({FMIN:g}-{FMAX:g} Hz)")
    fig.tight_layout()
    return image_to_base64(fig)


def create_spectrogram_plot(traces, detections, end):
    fig, axes = make_rows(len(traces), 1.7)
    for ax, (name, tr) in zip(axes, traces.items()):
        x = time_axis(tr, end)
        nfft = min(256, tr.stats.npts)
        ax.specgram(tr.data.astype(float), NFFT=nfft, Fs=tr.stats.sampling_rate,
                    noverlap=int(nfft * 0.75), xextent=(x[0], x[-1]))
        for d in detections:
            if d["station"] == name:
                ax.axvline(d["onset"] - end, color="red", linewidth=1.2)
        ax.set_ylim(FMIN, FMAX)
        ax.set_ylabel(f"{name}\nHz")
    axes[0].set_xlim(-WINDOW_SECONDS, 0)
    axes[-1].set_xlabel("Seconds relative to now")
    fig.suptitle("Live spectrograms (red line = detection onset)")
    fig.tight_layout()
    return image_to_base64(fig)


def create_envelope_plot(traces, envelopes, thresholds, detections, end):
    fig, axes = make_rows(len(traces), 1.5)
    for ax, (name, tr) in zip(axes, traces.items()):
        ax.plot(time_axis(tr, end), envelopes[name], color="tab:green", linewidth=1)
        if thresholds.get(name) is not None:
            ax.axhline(thresholds[name], color="red", linestyle="--", linewidth=1)
        for d in detections:
            if d["station"] == name:
                ax.axvspan(d["onset"] - end, d["end"] - end, color="red", alpha=0.25)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[0].set_xlim(-WINDOW_SECONDS, 0)
    axes[-1].set_xlabel("Seconds relative to now")
    fig.suptitle("Live Hilbert envelopes (dashed = energy threshold, red = detection)")
    fig.tight_layout()
    return image_to_base64(fig)


def create_aligned_envelope_plot(names, segments, traces, pairs, end):
    """Event segment envelopes, aligned to the first station using the measured lags."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ref = names[0]

    for name in names:
        i0, env = segments[name]
        tr = traces[name]
        fs = tr.stats.sampling_rate
        env = env / max(np.max(env), 1e-12)

        shift = 0.0
        for p in pairs:
            if p["a"] == ref and p["b"] == name:
                shift = p["lag"]          # shifting b by +lag aligns it with a
        x = (tr.stats.starttime - end) + (i0 + np.arange(len(env))) / fs + shift
        label = f"{name} (reference)" if name == ref else f"{name} ({shift:+.2f}s)"
        ax.plot(x, env, linewidth=1.5 if name == ref else 1.1, label=label)

    ax.set_title("Possible event: envelopes aligned by cross-correlation lag")
    ax.set_xlabel("Seconds relative to now")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return image_to_base64(fig)


def run_cycle():
    end = UTCDateTime() - LATENCY_SECONDS
    start = end - WINDOW_SECONDS
    print(f"\n[{fmt_time(UTCDateTime())}] cycle, window {start.strftime('%H:%M:%S')} -> {end.strftime('%H:%M:%S')}")

    traces, errors = fetch_all(STATIONS, start, end)
    for name, error in errors.items():
        print(f"  {name}: {error}")

    # --- live views: always drawn with whatever data we have ---
    envelopes = {n: get_hilbert_envelope(t, ENVELOPE_CUTOFF) for n, t in traces.items()}

    detections = []
    thresholds = {}
    station_status = []
    for station in STATIONS:
        name = station["sta"]
        if name not in traces:
            station_status.append({"station": name, "status": "no data", "note": errors.get(name, "")})
            continue
        dets, threshold, max_ratio = detect_events(name, traces[name], envelopes[name])
        detections += dets
        thresholds[name] = threshold
        station_status.append({
            "station": name,
            "status": "triggered" if dets else "ok",
            "last_sample": fmt_time(traces[name].stats.endtime),
            "latency_s": round(UTCDateTime() - traces[name].stats.endtime, 1),
            "max_ratio": None if max_ratio is None else round(max_ratio, 1),
            "note": "",
        })

    if traces:
        plots = {
            "Waveforms": create_waveform_plot(traces, detections, end),
            "Spectrograms": create_spectrogram_plot(traces, detections, end),
            "Envelopes": create_envelope_plot(traces, envelopes, thresholds, detections, end),
        }
    else:
        message = "No station returned data in this cycle"
        plots = {k: create_empty_plot(message) for k in ("Waveforms", "Spectrograms", "Envelopes")}

    # --- candidate events (>= MIN_STATIONS stations) ---
    new_candidates = []
    for group in find_coincident_groups(detections):
        candidate, plot = analyze_candidate(group, traces, envelopes, end)
        if candidate is not None:
            new_candidates.append((candidate, plot))

    return {"plots": plots, "detections": detections, "stations": station_status,
            "new_candidates": new_candidates}



STATE = {
    "id": 0, "updated": None, "error": None,
    "stations": [], "plots": {},
    "detection_log": [], "candidates": [], "candidate_plot": None,
}
STATE_LOCK = threading.Lock()


def processing_loop():
    while True:
        started = time.time()
        try:
            result = run_cycle()
            with STATE_LOCK:
                update_detection_log(STATE["detection_log"], result["detections"])
                for candidate, plot in result["new_candidates"]:
                    if update_candidates(STATE["candidates"], candidate):
                        STATE["candidate_plot"] = {"id": candidate["id"], "time": candidate["time"],
                                                   "image": plot}
                STATE.update(id=STATE["id"] + 1, updated=fmt_time(UTCDateTime()), error=None,
                             stations=result["stations"], plots=result["plots"])
        except Exception as error:
            print(f"  cycle failed: {error}")
            with STATE_LOCK:
                STATE.update(id=STATE["id"] + 1, error=str(error))
        time.sleep(max(1, UPDATE_SECONDS - (time.time() - started)))


# Website

app = Flask(__name__)

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Realtime Seismic Event Detection</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #f4f5f7; color: #222; }
  header { background: #1f2937; color: #fff; padding: 14px 24px; }
  header h1 { margin: 0; font-size: 20px; }
  #status { font-size: 13px; opacity: .8; margin-top: 4px; }
  main { max-width: 1100px; margin: 16px auto; padding: 0 16px; }
  .card { background: #fff; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;
          box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  .error { background: #fee2e2; color: #991b1b; }
  .possible { border-left: 5px solid #dc2626; }
  table { border-collapse: collapse; width: 100%; font-size: 14px; }
  td, th { padding: 4px 8px; border-bottom: 1px solid #eee; text-align: left; }
  .triggered { color: #dc2626; font-weight: 600; }
  .nodata { color: #999; }
  img { max-width: 100%; }
  h2 { font-size: 16px; margin: 0 0 8px; }
  .muted { color: #666; font-size: 13px; }
</style></head>
<body>
<header><h1>Realtime Seismic Event Detection</h1><div id="status">Waiting for first result...</div></header>
<main>
  <div id="error"></div>
  <div class="card"><h2>Stations</h2><div id="stations"></div></div>
  <div class="card"><h2>Possible events (3+ stations)</h2><div id="candidates"></div></div>
  <div id="candidate_plot"></div>
  <div class="card"><h2>Station detections</h2><div id="log"></div></div>
  <div id="plots"></div>
</main>
<script>
let lastId = 0;

function render(d) {
  document.getElementById("status").textContent =
    d.updated ? "Last update: " + d.updated : "Waiting for first result...";
  document.getElementById("error").innerHTML =
    d.error ? `<div class="card error">Last cycle failed: ${d.error}</div>` : "";

  document.getElementById("stations").innerHTML = d.stations.length ? `<table>
    <tr><th>Station</th><th>Status</th><th>Last sample</th><th>Latency (s)</th><th>Max energy ratio</th></tr>
    ${d.stations.map(s => s.status === "no data"
      ? `<tr class="nodata"><td>${s.station}</td><td>no data</td><td colspan="3">${s.note}</td></tr>`
      : `<tr><td>${s.station}</td><td class="${s.status === 'triggered' ? 'triggered' : ''}">${s.status}</td>
         <td>${s.last_sample}</td><td>${s.latency_s}</td><td>${s.max_ratio ?? "-"}</td></tr>`).join("")}
    </table>` : "";

  document.getElementById("candidates").innerHTML = d.candidates.length
    ? d.candidates.map(c => `<div class="card possible">
        <b>POSSIBLE EVENT at ${c.time}</b> &mdash; ${c.stations.join(", ")}
        <span class="muted">(mean envelope CC ${c.mean_cc})</span>
        <table style="margin-top:6px"><tr><th>Pair</th><th>Lag (s)</th><th>CC</th></tr>
        ${c.pairs.map(p => `<tr><td>${p.pair}</td><td>${p.lag}</td><td>${p.cc}</td></tr>`).join("")}</table>
        <div class="muted">Onsets: ${Object.entries(c.station_onsets).map(([k, v]) => k + " " + v.slice(11)).join(" | ")}</div>
      </div>`).join("")
    : '<span class="muted">None yet.</span>';

  document.getElementById("candidate_plot").innerHTML = d.candidate_plot
    ? `<div class="card"><h2>Latest possible event (${d.candidate_plot.time})</h2>
       <img src="data:image/png;base64,${d.candidate_plot.image}"></div>` : "";

  document.getElementById("log").innerHTML = d.detection_log.length
    ? "<table><tr><th>Time</th><th>Message</th></tr>" + d.detection_log.map(e =>
        `<tr><td>${e.time}</td><td>Station <b>${e.station}</b> detected event
         <span class="muted">(${e.duration}s, peak ratio ${e.peak_ratio})</span></td></tr>`).join("") + "</table>"
    : '<span class="muted">No detections yet.</span>';

  document.getElementById("plots").innerHTML = Object.entries(d.plots).map(
    ([title, b64]) => `<div class="card"><h2>${title}</h2><img src="data:image/png;base64,${b64}"></div>`).join("");
}

async function poll() {
  try {
    const res = await fetch("/api/latest?since=" + lastId);
    const data = await res.json();
    if (!data.unchanged) { lastId = data.id; render(data); }
  } catch (e) { document.getElementById("status").textContent = "Server unreachable..."; }
}
poll(); setInterval(poll, 5000);
</script></body></html>"""


@app.route("/")
def index():
    return PAGE


@app.route("/api/latest")
def latest():
    since = request.args.get("since", 0, type=int)
    with STATE_LOCK:
        if STATE["id"] == since:
            return jsonify({"unchanged": True})
        return jsonify(STATE)


if __name__ == "__main__":
    threading.Thread(target=processing_loop, daemon=True).start()
    print(f"Open http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)