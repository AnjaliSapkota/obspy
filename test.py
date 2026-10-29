"""
Realtime TDOA triangulation from a SeedLink server, shown on a simple website.

Workflow (repeats every UPDATE_SECONDS):
    fetch last WINDOW_SECONDS of data -> preprocess -> trim to common window
    -> cross-coherence & band selection -> Hilbert envelopes -> envelope lags
    -> joint location + velocity inversion -> plots -> website

Install:  pip install obspy numpy scipy matplotlib pyproj flask
Run:      python triangulate_realtime.py      then open http://localhost:8000
"""
import base64
import io
import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import matplotlib
matplotlib.use("Agg")  # no GUI, safe for background thread
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, request
from matplotlib.lines import Line2D
from obspy import UTCDateTime
from obspy.clients.seedlink import Client as SeedLinkClient
from pyproj import Geod, Transformer
from scipy.optimize import least_squares
from scipy.signal import (butter, coherence, correlate, correlation_lags,
                          filtfilt, hilbert, windows)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
SERVER_HOST = "ring.wscada.net"
SERVER_PORT = 18000

STATIONS = [
    {"net": "NP", "sta": "EQM10", "cha": "EHZ", "loc": "*", "lat": 28.299517, "lon": 83.960148},
    {"net": "NP", "sta": "EQM13", "cha": "EHZ", "loc": "*", "lat": 28.256323, "lon": 85.367569},
    {"net": "NK", "sta": "KKN",   "cha": "BHZ", "loc": "*", "lat": 27.800000, "lon": 85.279000},
    {"net": "NP", "sta": "EQM07", "cha": "EHZ", "loc": "*", "lat": 27.814941, "lon": 86.713401},
    {"net": "IO", "sta": "EVN",   "cha": "BHZ", "loc": "*", "lat": 27.958650, "lon": 86.811653},
]

WINDOW_SECONDS = 180      # length of data analysed each cycle
LATENCY_SECONDS = 5       # stay slightly behind "now" so data has arrived
UPDATE_SECONDS = 60       # how often to recompute

FS = 20.0                 # common sampling rate (Hz)
FMIN, FMAX = 1.0, 8.0     # initial bandpass (Hz); must be < FS/2
SMOOTH_CUTOFF = 0.2       # envelope smoothing (Hz)
MAX_SHIFT_SECONDS = 40.0  # max envelope lag searched
V_MIN, V_MAX, V_STEP = 0.5, 5.0, 0.1   # km/s
GRID_STEP_DEG = 0.05      # coarse grid keeps realtime cycle fast
GRID_MARGIN_DEG = 1.0
COHERENCE_THRESHOLD = 0.5

PORT = 8000

# ----------------------------------------------------------------------------
# Geodesy helpers
# ----------------------------------------------------------------------------
geod = Geod(ellps="WGS84")
to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
to_wgs = Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True)


def latlon_to_xy(origin_lat, origin_lon, lat, lon):
    ox, oy = to_utm.transform(origin_lon, origin_lat)
    tx, ty = to_utm.transform(lon, lat)
    return (tx - ox) / 1000.0, (ty - oy) / 1000.0


def xy_to_latlon(origin_lat, origin_lon, x_km, y_km):
    ox, oy = to_utm.transform(origin_lon, origin_lat)
    lon, lat = to_wgs.transform(ox + x_km * 1000.0, oy + y_km * 1000.0)
    return lat, lon


def fetch_waveform(station, start, end, fs, fmin, fmax):
    client = SeedLinkClient(SERVER_HOST, port=SERVER_PORT, timeout=20)
    loc = "??" if station["loc"] == "*" else station["loc"]
    stream = client.get_waveforms(station["net"], station["sta"], loc,
                                  station["cha"], start, end)
    if len(stream) == 0:
        raise RuntimeError(f"No data for {station['sta']}")

    stream.merge(method=1, fill_value="interpolate")
    stream.detrend("linear")
    stream.detrend("demean")
    stream.taper(max_percentage=0.05, type="hann")
    stream.interpolate(sampling_rate=fs, method="linear")
    stream.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
    return stream[0]


def fetch_all(stations, start, end, fs, fmin, fmax):
    """Fetch all stations in parallel; skip the ones that fail."""
    def one(station):
        try:
            return station["sta"], fetch_waveform(station, start, end, fs, fmin, fmax)
        except Exception as error:
            print(f"  {station['sta']} failed: {error}")
            return station["sta"], None

    with ThreadPoolExecutor(max_workers=len(stations)) as pool:
        results = list(pool.map(one, stations))
    return {name: tr for name, tr in results if tr is not None}


def trim_to_common_window(traces):
    common_start = max(t.stats.starttime for t in traces.values())
    common_end = min(t.stats.endtime for t in traces.values())
    if common_start >= common_end:
        raise RuntimeError("No common time window between stations.")
    for trace in traces.values():
        trace.trim(starttime=common_start, endtime=common_end)

def calculate_cross_coherence(traces, fs, fmin, fmax, window_seconds=10.0):
    nperseg = min(int(window_seconds * fs), min(len(t.data) for t in traces.values()))
    if nperseg < 16:
        raise RuntimeError("Not enough samples for cross-coherence.")

    results = {}
    for a, b in combinations(traces.keys(), 2):
        freqs, values = coherence(traces[a].data.astype(float),
                                  traces[b].data.astype(float),
                                  fs=fs, window="hann",
                                  nperseg=nperseg, noverlap=nperseg // 2)
        mask = (freqs >= fmin) & (freqs <= fmax)
        results[f"{a}-{b}"] = {"frequencies": freqs[mask], "coherence": values[mask]}
    return results


def select_coherent_band(pair_results, fmin, fmax, threshold=0.5, min_bandwidth=0.5):
    freqs = next(iter(pair_results.values()))["frequencies"]
    if len(freqs) < 2:
        return fmin, fmax

    mean_coh = np.mean([r["coherence"] for r in pair_results.values()], axis=0)
    indices = np.where(mean_coh >= threshold)[0]
    if len(indices) == 0:
        return fmin, fmax

    groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    bands = [(freqs[g[0]], freqs[g[-1]]) for g in groups
             if len(g) >= 2 and freqs[g[-1]] - freqs[g[0]] >= min_bandwidth]
    if not bands:
        return fmin, fmax

    low, high = max(bands, key=lambda b: b[1] - b[0])
    return float(low), float(high)



def get_hilbert_envelope(trace, smooth_cutoff):
    signal = trace.data.astype(float)
    signal = signal - np.mean(signal)
    envelope = np.abs(hilbert(signal))
    if smooth_cutoff is not None:
        fs = trace.stats.sampling_rate
        b, a = butter(N=2, Wn=smooth_cutoff / (0.5 * fs), btype="low")
        envelope = np.maximum(filtfilt(b, a, envelope), 0)
    return envelope


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
        raise RuntimeError("Envelope has zero energy.")

    corr = correlate(a, b, mode="full") / norm
    lags = correlation_lags(len(a), len(b), mode="full")

    max_shift = int(round(max_shift_seconds * fs))
    valid = (lags >= -max_shift) & (lags <= max_shift)
    peak = np.argmax(corr[valid])

    return {"lag": float(lags[valid][peak] / fs), "cc_max": float(corr[valid][peak])}



def invert_location_velocity(station_latlon, lag_pairs, v_min, v_max, v_step,
                             grid_step_deg, margin_deg):
    lats = [c[0] for c in station_latlon.values()]
    lons = [c[1] for c in station_latlon.values()]
    lat_range = np.arange(min(lats) - margin_deg, max(lats) + margin_deg, grid_step_deg)
    lon_range = np.arange(min(lons) - margin_deg, max(lons) + margin_deg, grid_step_deg)
    v_range = np.arange(v_min, v_max + v_step, v_step)
    lat_grid, lon_grid = np.meshgrid(lat_range, lon_range, indexing="ij")

    dist_km = {}
    for name, (s_lat, s_lon) in station_latlon.items():
        _, _, d = geod.inv(lon_grid, lat_grid,
                           np.full_like(lon_grid, s_lon), np.full_like(lat_grid, s_lat))
        dist_km[name] = d / 1000.0

    best = {"misfit": np.inf}
    misfit_vs_v = []
    for v in v_range:
        total = np.zeros_like(lat_grid)
        for a, b, lag, cc in lag_pairs:
            pred = (dist_km[a] - dist_km[b]) / v
            total += max(cc, 0.0) * (lag - pred) ** 2
        misfit = np.sqrt(total)
        idx = np.unravel_index(np.argmin(misfit), misfit.shape)
        misfit_vs_v.append(float(misfit[idx]))
        if misfit[idx] < best["misfit"]:
            best = {"misfit": float(misfit[idx]), "lat": float(lat_range[idx[0]]),
                    "lon": float(lon_range[idx[1]]), "v": float(v)}

    def residuals(p):
        lat, lon, v = p
        res = []
        for a, b, lag, cc in lag_pairs:
            _, _, da = geod.inv(lon, lat, station_latlon[a][1], station_latlon[a][0])
            _, _, db = geod.inv(lon, lat, station_latlon[b][1], station_latlon[b][0])
            pred = ((da - db) / 1000.0) / v
            res.append(np.sqrt(max(cc, 0.0)) * (lag - pred))
        return res

    refined = least_squares(
        residuals, [best["lat"], best["lon"], best["v"]],
        bounds=([lat_range.min(), lon_range.min(), v_min],
                [lat_range.max(), lon_range.max(), v_max]))
    lat, lon, v = refined.x

    return {"lat": float(lat), "lon": float(lon), "velocity": float(v),
            "residual_norm": float(np.linalg.norm(refined.fun)),
            "v_range": v_range.tolist(), "misfit_vs_v": misfit_vs_v}



def image_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def create_waveform_plot(traces):
    fig, axes = plt.subplots(len(traces), 1, figsize=(12, 1.6 * len(traces) + 1), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, tr) in zip(axes, traces.items()):
        t = np.arange(tr.stats.npts) / tr.stats.sampling_rate
        ax.plot(t, tr.data, linewidth=0.7)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Filtered waveforms")
    fig.tight_layout()
    return image_to_base64(fig)


def create_spectrogram_plot(traces, fmin, fmax):
    fig, axes = plt.subplots(len(traces), 1, figsize=(12, 1.8 * len(traces) + 1), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, tr) in zip(axes, traces.items()):
        nfft = min(256, tr.stats.npts)
        ax.specgram(tr.data.astype(float), NFFT=nfft, Fs=tr.stats.sampling_rate,
                    noverlap=int(nfft * 0.75))
        ax.set_ylabel(f"{name}\nHz")
        ax.set_ylim(fmin, fmax)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Spectrograms ({fmin:.2f}-{fmax:.2f} Hz)")
    fig.tight_layout()
    return image_to_base64(fig)


def create_coherence_plot(pair_results, threshold):
    fig, ax = plt.subplots(figsize=(12, 5))
    for pair, r in pair_results.items():
        ax.plot(r["frequencies"], r["coherence"], linewidth=1.1, label=pair)
    ax.axhline(threshold, color="black", linestyle="--", label=f"threshold {threshold:.2f}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Coherence")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-coherence between station pairs")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    return image_to_base64(fig)


def create_aligned_envelope_plot(envelopes, pair_lags, fs):
    fig, ax = plt.subplots(figsize=(12, 5))
    names = list(envelopes.keys())
    ref = names[0]
    ref_env = envelopes[ref] / max(np.max(envelopes[ref]), 1e-12)
    ax.plot(np.arange(len(ref_env)) / fs, ref_env, color="black", linewidth=1.5,
            label=f"{ref} (reference)")

    for name in names[1:]:
        env = envelopes[name] / max(np.max(envelopes[name]), 1e-12)
        lag = 0.0
        for pair, value in pair_lags.items():
            if pair == f"{ref}-{name}":
                lag = value
            elif pair == f"{name}-{ref}":
                lag = -value
        ax.plot(np.arange(len(env)) / fs + lag, env, linewidth=1.1, alpha=0.8,
                label=f"{name} ({lag:+.2f}s)")

    ax.set_title("Envelopes aligned by TDOA lags")
    ax.set_xlabel("Reference time (s)")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return image_to_base64(fig)


def create_hyperbola_plot(station_xy, station_latlon, delta, source_xy, origin):
    origin_lat, origin_lon = origin
    xs = [p[0] for p in station_xy.values()] + [source_xy[0]]
    ys = [p[1] for p in station_xy.values()] + [source_xy[1]]
    margin = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys)) + 20.0

    step = 2.0  # km
    X, Y = np.meshgrid(np.arange(min(xs) - margin, max(xs) + margin, step),
                       np.arange(min(ys) - margin, max(ys) + margin, step))
    ox, oy = to_utm.transform(origin_lon, origin_lat)
    glon, glat = to_wgs.transform(ox + X * 1000.0, oy + Y * 1000.0)

    dist = {}
    for name, (s_lat, s_lon) in station_latlon.items():
        _, _, d = geod.inv(glon, glat, np.full_like(glon, s_lon), np.full_like(glat, s_lat))
        dist[name] = d / 1000.0

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = plt.cm.tab10.colors
    handles = []
    for i, (pair, diff) in enumerate(delta.items()):
        a, b = pair.split("-")
        ax.contour(X, Y, dist[b] - dist[a] - diff, levels=[0],
                   colors=[colors[i % 10]], linewidths=1.8)
        handles.append(Line2D([0], [0], color=colors[i % 10], linewidth=2, label=pair))

    for name, (sx, sy) in station_xy.items():
        ax.scatter(sx, sy, marker="^", s=90, color="black", zorder=5)
        ax.text(sx + 3, sy + 3, name, fontsize=10, fontweight="bold")
    ax.scatter(*source_xy, marker="*", s=350, color="red", zorder=10)
    handles.append(Line2D([0], [0], marker="*", color="red", linestyle="None",
                          markersize=14, label="Estimated source"))

    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.set_xlabel("East (km) from first station")
    ax.set_ylabel("North (km) from first station")
    ax.set_title("TDOA hyperbolas and estimated source")
    ax.set_aspect("equal")
    ax.grid(True)
    fig.tight_layout()
    return image_to_base64(fig)


def create_velocity_plot(inv):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(inv["v_range"], inv["misfit_vs_v"], "b-", linewidth=2)
    ax.axvline(inv["velocity"], color="r", linestyle="--",
               label=f"Estimated: {inv['velocity']:.2f} km/s")
    ax.set_xlabel("Apparent velocity (km/s)")
    ax.set_ylabel("Min. TDOA misfit (s)")
    ax.set_title("Misfit vs velocity")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    return image_to_base64(fig)


# ----------------------------------------------------------------------------
# One full processing cycle
# ----------------------------------------------------------------------------
def run_triangulation():
    end = UTCDateTime() - LATENCY_SECONDS
    start = end - WINDOW_SECONDS
    print(f"\n[{UTCDateTime()}] window {start} -> {end}")

    traces = fetch_all(STATIONS, start, end, FS, FMIN, FMAX)
    if len(traces) < 3:
        raise RuntimeError(f"Only {len(traces)} station(s) returned data; need 3.")
    trim_to_common_window(traces)

    # Coherence + band selection
    coh = calculate_cross_coherence(traces, FS, FMIN, FMAX)
    coherence_plot = create_coherence_plot(coh, COHERENCE_THRESHOLD)
    fmin, fmax = select_coherent_band(coh, FMIN, FMAX, COHERENCE_THRESHOLD)
    if fmax > fmin:
        for tr in traces.values():
            tr.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)

    waveform_plot = create_waveform_plot(traces)
    spectrogram_plot = create_spectrogram_plot(traces, fmin, fmax)

    # Envelopes and lags
    envelopes = {n: get_hilbert_envelope(t, SMOOTH_CUTOFF) for n, t in traces.items()}
    names = list(traces.keys())
    pair_keys = list(combinations(names, 2))
    lags = {f"{a}-{b}": calculate_envelope_lag(envelopes[a], envelopes[b], FS, MAX_SHIFT_SECONDS)
            for a, b in pair_keys}
    lag_pairs = [(a, b, lags[f"{a}-{b}"]["lag"], lags[f"{a}-{b}"]["cc_max"])
                 for a, b in pair_keys]

    closure = max((lags[f"{x}-{z}"]["lag"] - lags[f"{x}-{y}"]["lag"] - lags[f"{y}-{z}"]["lag"]
                   for x, y, z in itertools.combinations(names, 3)), key=abs)

    # Inversion (only stations that returned data)
    used = [s for s in STATIONS if s["sta"] in traces]
    station_latlon = {s["sta"]: (s["lat"], s["lon"]) for s in used}
    origin = (used[0]["lat"], used[0]["lon"])

    inv = invert_location_velocity(station_latlon, lag_pairs, V_MIN, V_MAX, V_STEP,
                                   GRID_STEP_DEG, GRID_MARGIN_DEG)
    src_xy = latlon_to_xy(*origin, inv["lat"], inv["lon"])
    station_xy = {n: latlon_to_xy(*origin, lat, lon) for n, (lat, lon) in station_latlon.items()}
    delta = {k: -inv["velocity"] * v["lag"] for k, v in lags.items()}

    hyperbola_plot = create_hyperbola_plot(station_xy, station_latlon, delta, src_xy, origin)
    velocity_plot = create_velocity_plot(inv)
    aligned_plot = create_aligned_envelope_plot(
        envelopes, {k: v["lag"] for k, v in lags.items()}, FS)

    ref_trace = next(iter(traces.values()))
    return {
        "window": f"{ref_trace.stats.starttime} to {ref_trace.stats.endtime}",
        "stations_used": names,
        "band_hz": [round(fmin, 2), round(fmax, 2)],
        "source": {"lat": inv["lat"], "lon": inv["lon"], "velocity": inv["velocity"],
                   "residual_norm": inv["residual_norm"]},
        "closure_error": float(closure),
        "pairs": [{"pair": k, "lag": v["lag"], "cc": v["cc_max"]} for k, v in lags.items()],
        "plots": {
            "Location (hyperbolas)": hyperbola_plot,
            "Aligned envelopes": aligned_plot,
            "Cross-coherence": coherence_plot,
            "Velocity misfit": velocity_plot,
            "Waveforms": waveform_plot,
            "Spectrograms": spectrogram_plot,
        },
    }


# ----------------------------------------------------------------------------
# Background loop + shared state
# ----------------------------------------------------------------------------
STATE = {"id": 0, "updated": None, "error": None, "result": None}
STATE_LOCK = threading.Lock()


def processing_loop():
    while True:
        started = time.time()
        try:
            result = run_triangulation()
            with STATE_LOCK:
                STATE.update(id=STATE["id"] + 1, updated=str(UTCDateTime()),
                             error=None, result=result)
            print("  done.")
        except Exception as error:
            print(f"  cycle failed: {error}")
            with STATE_LOCK:
                STATE.update(id=STATE["id"] + 1, error=str(error))
        time.sleep(max(1, UPDATE_SECONDS - (time.time() - started)))


# ----------------------------------------------------------------------------
# Website
# ----------------------------------------------------------------------------
app = Flask(__name__)

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Realtime Seismic Triangulation</title>
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
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
  .kpi b { display: block; font-size: 20px; }
  .kpi span { font-size: 12px; color: #666; }
  table { border-collapse: collapse; width: 100%; font-size: 14px; }
  td, th { padding: 4px 8px; border-bottom: 1px solid #eee; text-align: left; }
  img { max-width: 100%; }
  h2 { font-size: 16px; margin: 0 0 8px; }
</style></head>
<body>
<header><h1>Realtime Seismic Triangulation</h1><div id="status">Waiting for first result...</div></header>
<main><div id="error"></div><div id="summary"></div><div id="plots"></div></main>
<script>
let lastId = 0;
function kpi(label, value) { return `<div class="kpi"><b>${value}</b><span>${label}</span></div>`; }

function render(data) {
  document.getElementById("status").textContent =
    data.updated ? "Last update: " + data.updated + " UTC" : "Waiting for first result...";
  document.getElementById("error").innerHTML =
    data.error ? `<div class="card error">Last cycle failed: ${data.error}</div>` : "";
  const r = data.result;
  if (!r) return;
  const s = r.source;
  document.getElementById("summary").innerHTML = `
    <div class="card"><div class="grid">
      ${kpi("Latitude", s.lat.toFixed(4))}
      ${kpi("Longitude", s.lon.toFixed(4))}
      ${kpi("Velocity (km/s)", s.velocity.toFixed(2))}
      ${kpi("Residual norm (s)", s.residual_norm.toFixed(3))}
      ${kpi("Closure error (s)", r.closure_error.toFixed(3))}
      ${kpi("Band (Hz)", r.band_hz[0] + " - " + r.band_hz[1])}
    </div>
    <p style="font-size:13px;color:#666">Window: ${r.window}<br>Stations: ${r.stations_used.join(", ")}</p></div>
    <div class="card"><h2>Envelope lags</h2><table><tr><th>Pair</th><th>Lag (s)</th><th>CC</th></tr>
      ${r.pairs.map(p => `<tr><td>${p.pair}</td><td>${p.lag.toFixed(2)}</td><td>${p.cc.toFixed(2)}</td></tr>`).join("")}
    </table></div>`;
  document.getElementById("plots").innerHTML = Object.entries(r.plots).map(
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