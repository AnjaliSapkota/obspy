import base64
import io
import json
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, Response, jsonify
from matplotlib.ticker import FuncFormatter
from obspy import Stream, UTCDateTime
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy.signal import hilbert

from earthquake import run_location

SERVER_URL = "ring.wscada.net:18000"
WEB_HOST = "0.0.0.0"
WEB_PORT = 8001

STATIONS = [
    {"net": "NP", "sta": "EQM10", "cha": "EHZ", "loc": "*", "lat": 28.299517, "lon": 83.960148},
    {"net": "NP", "sta": "EQM26", "cha": "EHZ", "loc": "*", "lat": 27.3936969, "lon": 86.6791885},
    {"net": "NP", "sta": "EQM07", "cha": "EHZ", "loc": "*", "lat": 27.814941, "lon": 86.713401},
    {"net": "NP", "sta": "EQM08", "cha": "EHZ", "loc": "*", "lat": 27.831000, "lon": 86.65},
    {"net": "NP", "sta": "EQM11", "cha": "EHZ", "loc": "*", "lat": 29.2859556, "lon": 81.2741972},
    {"net": "NP", "sta": "EQM22", "cha": "EHZ", "loc": "*", "lat": 28.895805, "lon": 80.227341},
    {"net": "AM", "sta": "R0F23", "cha": "EHZ", "loc": "*", "lat": 28.045045, "lon": 84.072371},
]

# Analysis settings
FS = 100.0                
FMIN, FMAX = 1.0, 8.0    
STA_P, LTA_P = 1.0, 10.0 
THR_ON, THR_OFF = 4, 1.5

# Detection settings
BUFFER_SEC = 300          
PRE_SEC = 30             
WAIT_SEC = 40            
MIN_STATIONS = 3         
COINCIDENCE_SEC = 30     
DETECT_EVERY = 2.0      
DETECT_WINDOW = 90       
DEDUP_SEC = 60            
RECONNECT_SEC = 10       
ACTIVE_SEC = 60           
WAVEFORM_SEC = 60      
PLOT_REFRESH = 5

COUNTS_PER_M = 1.0

MAX_LOG = 100
MAX_EVENTS = 50

STATION_COLORS = [
    "#0F766E", "#2563EB", "#7C3AED", "#92400E",
    "#0891B2", "#4D7C0F", "#475569", "#C026D3",
]

LOCK = threading.Lock()
PLOT_LOCK = threading.Lock()
PLOT_CACHE = {"png": None, "time": 0.0}
BUFFERS = {}              # "NET.STA" -> obspy Stream
STATE = {
    "connected": False,
    "pending": False,
    "status": "starting",
    "error": None,
    "ignore_before": None,
    "last_scan": None,
    "last_data": None,
    "started": time.time(),
    "event_seq": 0,
    "detections": [],
    "events": [],
}


def station_id(station):
    return f"{station['net']}.{station['sta']}"


def fmt_time(value):
    return UTCDateTime(value).strftime("%Y-%m-%d %H:%M:%S") + " UTC"


def buffer_copy(sid):
    with LOCK:
        stream = BUFFERS.get(sid)
        return stream.copy() if stream is not None else None


def prepare_trace(stream, fs=FS, fmin=FMIN, fmax=FMAX):
    if stream is None or len(stream) == 0:
        raise ValueError("No data in buffer")

    stream = stream.copy()
    stream.merge(method=1, fill_value=0)
    trace = max(stream, key=lambda item: item.stats.npts)

    if trace.stats.npts < 20:
        raise ValueError("Not enough samples")

    trace.detrend("linear")
    trace.taper(0.02)
    trace.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)

    if abs(trace.stats.sampling_rate - fs) > 1e-6:
        trace.resample(fs)

    return trace


def fetch_components(client, station, start, end, fs=FS, fmin=FMIN, fmax=FMAX, wait_sec=0.0):

    del client, wait_sec  # kept so this matches the get_three_components signature

    stream = buffer_copy(station_id(station))
    if stream is None:
        raise ValueError("No live data")

    z = prepare_trace(stream, fs, fmin, fmax)
    z.trim(start, end, pad=True, fill_value=0)

    if z.stats.npts < 10:
        raise ValueError("No data in the requested window")

    envelope = np.abs(hilbert(z.data))
    max_horizontal = np.abs(z.data) / max(COUNTS_PER_M, 1e-9)
    return z, envelope, max_horizontal


def latest_data_time():
    with LOCK:
        ends = [trace.stats.endtime for stream in BUFFERS.values() for trace in stream]
    return max(ends) if ends else None


def common_end_time():
    with LOCK:
        ends = []
        for station in STATIONS:
            stream = BUFFERS.get(station_id(station))
            if stream is not None and len(stream) > 0:
                ends.append(max(trace.stats.endtime for trace in stream))

    if not ends:
        return None

    # Ignore a station clock that is far in the future.
    newest = min(max(ends), UTCDateTime())
    recent = [end for end in ends if 0 <= newest - end <= ACTIVE_SEC]
    return min(recent) if recent else newest


def find_onset(sid, now):
    stream = buffer_copy(sid)
    if stream is None:
        return None

    try:
        trace = prepare_trace(stream)
    except Exception:
        return None

    if now - trace.stats.endtime > DETECT_WINDOW:
        return None  # station data is too old to be useful

    trace.trim(starttime=now - DETECT_WINDOW)
    if trace.stats.npts < int((LTA_P + 5) * FS):
        return None

    cft = classic_sta_lta(trace.data, int(STA_P * FS), int(LTA_P * FS))
    onsets = trigger_onset(cft, THR_ON, THR_OFF)
    if len(onsets) == 0:
        return None

    ignore_before = STATE["ignore_before"]
    times = []
    for onset, _ in onsets:
        onset_time = trace.stats.starttime + onset / FS
        if onset_time > now:
            continue
        if now - onset_time > COINCIDENCE_SEC:
            continue
        if ignore_before is not None and onset_time <= ignore_before:
            continue
        times.append(onset_time)

    return max(times) if times else None


def record_detection(sid, onset):
    stamp = onset.timestamp
    with LOCK:
        for old in STATE["detections"]:
            if old["station"] == sid and abs(old["t"] - stamp) < DEDUP_SEC:
                return
        entry = {
            "station": sid,
            "t": stamp,
            "time": fmt_time(onset),
            "clock": onset.strftime("%H:%M:%S"),
        }
        STATE["detections"].insert(0, entry)
        del STATE["detections"][MAX_LOG:]
    print(f"[trigger] {sid} at {entry['time']}")


def scan_for_event(now):
    onsets = {}
    for station in STATIONS:
        sid = station_id(station)
        onset = find_onset(sid, now)
        if onset is not None:
            onsets[sid] = onset
            record_detection(sid, onset)

    if len(onsets) < MIN_STATIONS:
        return

    times = sorted(onsets.values())
    for index in range(len(times) - MIN_STATIONS + 1):
        if times[index + MIN_STATIONS - 1] - times[index] > COINCIDENCE_SEC:
            continue

        with LOCK:
            if STATE["pending"]:
                return
            STATE["pending"] = True
            STATE["status"] = "event detected, waiting for more data"

        start = times[index] - PRE_SEC
        print(f"[detect] {len(onsets)} stations triggered, first at {times[index].isoformat()}")

        timer = threading.Timer(WAIT_SEC, process_event, args=(start,))
        timer.daemon = True
        timer.start()
        return


def detection_loop():
    while True:
        time.sleep(DETECT_EVERY)
        with LOCK:
            pending = STATE["pending"]
        if pending:
            continue

        now = latest_data_time()
        if now is None:
            continue
        now = min(now, UTCDateTime())  # ignore future clock skew

        with LOCK:
            STATE["last_scan"] = time.time()

        try:
            scan_for_event(now)
        except Exception as error:
            print(f"[detect] scan failed: {error}")


def print_result(result):
    if not result.get("located"):
        print(f"[result] NOT LOCATED: {result.get('error')}")
        return

    epicenter = result["epicenter"]
    print("[result] LOCATED")
    print(f"  origin time : {result['origin_time']}")
    print(f"  epicenter   : {epicenter['lat']:.3f}, {epicenter['lon']:.3f}")
    print(f"  ML          : {result['magnitude_ml']}")
    print(f"  RMS         : {result['rms_km']:.2f} km ({result['n_stations_used']} stations)")
    for station in result["stations"]:
        print(f"  {station['name']:<12} {station.get('quality', '-'):<8} {station.get('error', '')}")


def new_event_id():
    with LOCK:
        STATE["event_seq"] += 1
        return f"{int(time.time())}-{STATE['event_seq']}"


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def make_event(result, start, end):
    epicenter = result.get("epicenter") or {}
    latitude = as_float(epicenter.get("lat"))
    longitude = as_float(epicenter.get("lon"))
    located = bool(result.get("located")) and latitude is not None and longitude is not None
    stations = []

    for station in result.get("stations", []):
        stations.append({
            "name": station.get("name"),
            "quality": station.get("quality"),
            "error": station.get("error"),
            "p_time": station.get("p_time"),
            "s_time": station.get("s_time"),
            "residual_km": station.get("residual_km"),
            "ml": station.get("ml"),
        })

    return {
        "id": new_event_id(),
        "detected": fmt_time(UTCDateTime()),
        "time": fmt_time(UTCDateTime(start) + PRE_SEC),
        "window_start": str(start),
        "window_end": None if end is None else str(end),
        "located": located,
        "error": result.get("error"),
        "origin_time": result.get("origin_time"),
        "lat": latitude if located else None,
        "lon": longitude if located else None,
        "magnitude": as_float(result.get("magnitude_ml")),
        "rms_km": None if result.get("rms_km") is None else round(float(result["rms_km"]), 2),
        "n_stations": result.get("n_stations_used", 0),
        "stations": stations,
        "map_png": result.get("circle_map_png"),
    }


def add_event(event):
    with LOCK:
        STATE["events"].insert(0, event)
        del STATE["events"][MAX_EVENTS:]
    if event["located"] and event["lat"] is not None and event["lon"] is not None:
        print(f"[event] M {event['magnitude']} at {event['origin_time']} "
              f"({event['lat']:.3f}, {event['lon']:.3f})")
    else:
        print(f"[event] detection not located: {event['error']}")


def save_result(result):
    """Write one slim JSON file per located event (no base64 images)."""
    slim = {key: value for key, value in result.items() if key != "circle_map_png"}
    slim["stations"] = [
        {key: value for key, value in station.items() if not key.endswith("_png")}
        for station in result.get("stations", [])
    ]

    stamp = str(result.get("origin_time", "unknown"))[:19].replace(":", "-")
    name = Path(f"event_{stamp}.json")
    with open(name, "w", encoding="utf-8") as handle:
        json.dump(slim, handle, indent=2, default=str)
    print(f"[saved] {name}")


def process_event(start):
    end = None
    with LOCK:
        STATE["status"] = "locating"
        STATE["error"] = None

    try:
        end = common_end_time()
        if end is None or end <= start:
            raise RuntimeError("Not enough live data to locate the event")

        print(f"[locate] window {start.isoformat()} -> {end.isoformat()}")
        result = run_location(
            None, STATIONS, start, end,
            fs=FS, fmin=FMIN, fmax=FMAX,
            sta_p=STA_P, lta_p=LTA_P, thr_on=THR_ON, thr_off=THR_OFF,
            fetch=fetch_components,
        )

        print_result(result)
        add_event(make_event(result, start, end))

        if result.get("located"):
            save_result(result)

        with LOCK:
            STATE["ignore_before"] = end
    except Exception as error:
        print(f"[locate] failed: {error}")
        add_event(make_event({"located": False, "error": str(error)}, start, end))
        with LOCK:
            STATE["error"] = str(error)
    finally:
        with LOCK:
            STATE["pending"] = False
            STATE["status"] = "listening"


def on_data(trace):
    if trace is None or trace.stats.npts == 0:
        return

    trace.stats.location = ""
    sid = f"{trace.stats.network}.{trace.stats.station}"

    with LOCK:
        stream = BUFFERS.setdefault(sid, Stream())
        stream += trace
        try:
            stream.merge(method=1, fill_value=0)
            newest = max(item.stats.endtime for item in stream)
            stream.trim(starttime=newest - BUFFER_SEC)
        except Exception:
            pass
        STATE["last_data"] = time.time()

    with LOCK:
        STATE["connected"] = True


def make_seedlink_client():

    client = EasySeedLinkClient(SERVER_URL, autoconnect=False)
    client.on_data = on_data
    client.conn.timeout = 10
    client.connect()
    return client


def run_seedlink():
    while True:
        client = None
        try:
            client = make_seedlink_client()
            for station in STATIONS:
                client.select_stream(station["net"], station["sta"], station["cha"])

            with LOCK:
                STATE["connected"] = True
                STATE["status"] = "listening"
                STATE["error"] = None

            print(f"Connected to {SERVER_URL}, listening on {len(STATIONS)} stations...")
            client.run()
        except KeyboardInterrupt:
            raise
        except Exception as error:
            print(f"[seedlink] {error}; reconnecting in {RECONNECT_SEC}s")
            with LOCK:
                STATE["status"] = "reconnecting"
                STATE["error"] = str(error)
        finally:
            with LOCK:
                STATE["connected"] = False
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

        time.sleep(RECONNECT_SEC)


# ---------------- website ----------------
app = Flask(__name__)


def station_rows():
    now = UTCDateTime()
    now_ts = time.time()
    rows = []

    with LOCK:
        detections = list(STATE["detections"])

        for station in STATIONS:
            sid = station_id(station)
            stream = BUFFERS.get(sid)

            if stream is None or len(stream) == 0:
                rows.append({
                    "name": sid,
                    "net": station["net"],
                    "sta": station["sta"],
                    "lat": station["lat"],
                    "lon": station["lon"],
                    "state": "no data",
                    "last_sample": None,
                    "latency_s": None,
                    "buffered_s": 0.0,
                })
                continue

            end = max(trace.stats.endtime for trace in stream)
            start = min(trace.stats.starttime for trace in stream)
            latency = round(float(now - end), 1)
            triggered = any(
                item["station"] == sid and now_ts - item["t"] < ACTIVE_SEC
                for item in detections
            )

            if latency > ACTIVE_SEC:
                state = "stale"
            elif triggered:
                state = "triggered"
            else:
                state = "live"

            rows.append({
                "name": sid,
                "net": station["net"],
                "sta": station["sta"],
                "lat": station["lat"],
                "lon": station["lon"],
                "state": state,
                "last_sample": end.strftime("%H:%M:%S"),
                "latency_s": latency,
                "buffered_s": round(float(end - start), 1),
            })

    return rows


def public_event(event):
    return {key: value for key, value in event.items() if key != "map_png"}


def build_status():
    now = time.time()
    rows = station_rows()

    with LOCK:
        detections = [dict(item) for item in STATE["detections"][:50]]
        events = [public_event(event) for event in STATE["events"]]
        for event, original in zip(events, STATE["events"]):
            event["has_map"] = bool(original.get("map_png"))
        status = STATE["status"]
        connected = STATE["connected"]
        error = STATE["error"]
        last_scan = STATE["last_scan"]
        last_data = STATE["last_data"]
        started = STATE["started"]

    return {
        "server": SERVER_URL,
        "now": fmt_time(UTCDateTime()),
        "status": status,
        "connected": connected,
        "error": error,
        "last_scan": None if last_scan is None else fmt_time(UTCDateTime(last_scan)),
        "last_data": None if last_data is None else fmt_time(UTCDateTime(last_data)),
        "uptime_s": round(now - started, 1),
        "stations": rows,
        "detections": detections,
        "events": events,
        "counts": {
            "stations_live": sum(1 for row in rows if row["state"] in ("live", "triggered")),
            "stations_total": len(STATIONS),
            "triggers": len(detections),
            "events": len(events),
            "located": sum(1 for event in events if event["located"]),
        },
    }


def make_waveform_png(window_sec=WAVEFORM_SEC):
    with LOCK:
        streams = {}
        for station in STATIONS:
            sid = station_id(station)
            stream = BUFFERS.get(sid)
            streams[sid] = stream.copy() if stream is not None else None
        detections = list(STATE["detections"])

    ends = [
        max(trace.stats.endtime for trace in stream)
        for stream in streams.values()
        if stream is not None and len(stream) > 0
    ]
    end = max(ends) if ends else UTCDateTime()
    start = end - window_sec

    fig, axes = plt.subplots(
        len(STATIONS), 1,
        figsize=(12, max(3.2, 1.02 * len(STATIONS) + 0.7)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for index, (ax, station) in enumerate(zip(axes, STATIONS)):
        sid = station_id(station)
        color = STATION_COLORS[index % len(STATION_COLORS)]
        stream = streams[sid]

        ax.set_yticks([])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.text(0.006, 0.92, sid, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, fontweight="bold", color=color,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8})

        if stream is None or len(stream) == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="#8a96a3", fontsize=9)
            continue

        try:
            trace = prepare_trace(stream)
            trace.trim(start, end, pad=True, fill_value=0)
            data = trace.data.astype(float)
        except Exception:
            ax.text(0.5, 0.5, "not enough data", transform=ax.transAxes,
                    ha="center", va="center", color="#8a96a3", fontsize=9)
            continue

        peak = float(np.percentile(np.abs(data), 99.5)) if len(data) else 0.0
        if not np.isfinite(peak) or peak <= 0:
            peak = 1.0

        x = np.arange(len(data)) / trace.stats.sampling_rate + (trace.stats.starttime - end)
        ax.plot(x, data / peak, color=color, linewidth=0.65)
        ax.set_ylim(-1.3, 1.3)
        ax.text(0.994, 0.92, f"peak {peak:.3g}", transform=ax.transAxes,
                va="top", ha="right", fontsize=7.5, color="#63707d",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8})

        for detection in detections:
            if detection["station"] != sid:
                continue
            detected_at = UTCDateTime(detection["t"])
            if start <= detected_at <= end:
                ax.axvline(detected_at - end, color="#D6361F", linewidth=1.1, alpha=0.9)

    for ax in axes:
        ax.set_xlim(-window_sec, 0)

    axes[-1].xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: (end + float(value)).strftime("%H:%M:%S"))
    )
    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle(f"Live waveforms, last {window_sec:g} s (red line = station trigger)",
                 x=0.5, y=0.995, fontsize=11, fontweight="bold", color="#17212b")
    fig.subplots_adjust(left=0.07, right=0.995, top=0.94, bottom=0.07, hspace=0.16)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def get_waveform_png():
    now = time.time()
    with PLOT_LOCK:
        if PLOT_CACHE["png"] is not None and now - PLOT_CACHE["time"] < PLOT_REFRESH:
            return PLOT_CACHE["png"]

        png = make_waveform_png()
        PLOT_CACHE["png"] = png
        PLOT_CACHE["time"] = now
        return png


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live Earthquake Monitor</title>
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
  main { max-width:1180px; margin:0 auto; padding:18px; }
  .muted { color:var(--muted); font-size:12.5px; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:16px; }
  .card, section { background:var(--panel); border:1px solid var(--line); border-radius:7px; }
  .card { padding:12px 15px; }
  .card span { display:block; color:var(--muted); font-size:12.5px; }
  .card b { font-size:22px; }
  section { padding:14px 16px; margin-bottom:16px; overflow:auto; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:6px 8px 6px 0; border-bottom:1px solid #edf1f4; white-space:nowrap; }
  th { color:var(--muted); font-weight:500; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:#9aa5b1; margin-right:5px; }
  .live .dot { background:var(--ok); } .triggered .dot { background:var(--bad); }
  .stale .dot { background:var(--warn); }
  .badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11.5px; font-weight:600; }
  .badge.ok { background:#ddefe6; color:#17603f; }
  .badge.bad { background:#fbeae6; color:#8a2412; }
  .event { border-top:1px solid var(--line); padding:10px 0; }
  .event:first-child { border-top:0; }
  .event .head { display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; }
  .event .head b { font-size:15px; }
  .event .head span:last-child { margin-left:auto; }
  details { margin-top:7px; }
  summary { cursor:pointer; color:var(--muted); font-size:12.5px; }
  .map { display:block; width:100%; max-width:680px; margin-top:10px; border:1px solid var(--line); border-radius:6px; }
  .waveforms { display:block; width:100%; height:auto; border:1px solid var(--line); border-radius:6px; background:#fff; }
  #map { height:440px; border:1px solid var(--line); border-radius:6px; background:#e3e8ed; }
  .maphead { display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap; margin-bottom:10px; }
  .maphead h2 { margin:0; }
  .linkbtn { font:inherit; font-size:12.5px; background:none; border:1px solid var(--line); border-radius:4px;
             padding:3px 9px; color:var(--ink); cursor:pointer; }
  .linkbtn:hover { background:#f4f6f8; }
  .st-mark { position:relative; }
  .st-mark svg { display:block; filter:drop-shadow(0 0 1px rgba(20,32,43,.55)); }
  .st-mark .st-label { position:absolute; left:19px; top:-1px; font-size:11.5px; font-weight:600; white-space:nowrap;
                      background:rgba(255,255,255,.9); padding:0 4px; border-radius:3px; color:var(--ink); }
  .st-mark.triggered svg { filter:drop-shadow(0 0 4px rgba(214,54,31,.95)); }
  .legend { display:flex; gap:16px; flex-wrap:wrap; margin-top:8px; color:var(--muted); font-size:12.5px; }
  .legend span { display:inline-flex; align-items:center; gap:6px; }
  .legend .tri { width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent;
                border-bottom:11px solid var(--ok); }
  .legend .epi { width:11px; height:11px; border-radius:50%; background:var(--bad); border:2px solid #fff;
                box-shadow:0 0 0 1px rgba(20,32,43,.4); }
  .leaflet-popup-content { font:13px/1.45 system-ui, sans-serif; margin:10px 12px; }
  .event .head { cursor:pointer; }
  #error { display:none; margin-bottom:14px; padding:10px 13px; border:1px solid #e8b4aa;
           background:#fbeae6; color:#8a2412; border-radius:6px; }
</style>
</head>
<body>
<header>
  <div><h1>Live Earthquake Monitor</h1><div class="muted" id="server">Connecting...</div></div>
  <div class="muted" id="clock">Loading...</div>
</header>
<main>
  <div id="error"></div>
  <div class="cards" id="cards"></div>

  <section>
    <div class="maphead">
      <h2>Station map</h2>
      <button class="linkbtn" id="fit">Fit to map</button>
    </div>
    <div class="muted" style="margin-bottom:10px">
      Triangles are stations, coloured by state. Red circles mark located epicenters &mdash;
      click one for magnitude, coordinates and origin time. Click an event below to show it here.
    </div>
    <div id="map"></div>
    <div class="legend">
      <span><i class="tri"></i>Station</span>
      <span><i class="epi"></i>Estimated epicenter</span>
      <span class="muted" id="mapnote"></span>
    </div>
  </section>

  <section>
    <h2>Live waveforms</h2>
    <div class="muted" style="margin-bottom:10px">
      Last 60 seconds, normalised per station. Red lines mark station triggers.
    </div>
    <img id="waveforms" class="waveforms" src="/waveforms.png" alt="Live waveforms">
  </section>

  <section>
    <h2>Stations</h2>
    <table>
      <thead><tr><th>Station</th><th>State</th><th>Last sample</th><th>Latency</th><th>Buffered</th></tr></thead>
      <tbody id="stations"><tr><td colspan="5" class="muted">Waiting for data...</td></tr></tbody>
    </table>
  </section>

  <section>
    <h2>Detected events</h2>
    <div id="events"><div class="muted">No events yet.</div></div>
  </section>

  <section>
    <h2>Station triggers (newest first)</h2>
    <table>
      <thead><tr><th>Time (UTC)</th><th>Station</th><th>Detected</th></tr></thead>
      <tbody id="triggers"><tr><td colspan="3" class="muted">No triggers yet.</td></tr></tbody>
    </table>
  </section>
</main>

<script>
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => (
  {"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]
));
const num = (value, digits = 1) => value == null ? "-" : Number(value).toFixed(digits);
const isoTime = value => value ? String(value).replace("T", " ").slice(0, 19) + " UTC" : "-";

/* ---------- map ---------- */
const STATE_COLORS = { live:"#1f7a55", triggered:"#d6361f", stale:"#b7791f", "no data":"#9aa5b1" };
let map = null, stationMarkers = {}, epicenterMarkers = {}, fitted = false;

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
  const pts = [...Object.values(stationMarkers), ...Object.values(epicenterMarkers)].map(m => m.getLatLng());
  if (pts.length) map.fitBounds(L.latLngBounds(pts), { padding: [50, 50], maxZoom: 9 });
}

function stationIcon(s) {
  const color = STATE_COLORS[s.state] || "#9aa5b1";
  const cls = s.state === "triggered" ? " triggered" : "";
  return L.divIcon({
    className: "", iconSize: [18, 18], iconAnchor: [9, 9],
    html: `<div class="st-mark${cls}">
      <svg width="18" height="18" viewBox="0 0 18 18">
        <polygon points="9,1.5 16.8,16.2 1.2,16.2" fill="${color}" stroke="#fff" stroke-width="1.7"/>
      </svg><span class="st-label">${esc(s.sta)}</span></div>`
  });
}

function stationPopup(s) {
  const lines = [`<b>${esc(s.name)}</b>`, `State: ${esc(s.state)}`];
  if (s.last_sample) lines.push(`Last sample ${esc(s.last_sample)} UTC`);
  if (s.latency_s != null) lines.push(`${num(s.latency_s)} s behind`);
  lines.push(`<span class="muted">${num(s.lat, 4)}, ${num(s.lon, 4)}</span>`);
  return lines.join("<br>");
}

function epicenterPopup(e) {
  return `<b style="color:#d6361f">M ${esc(e.magnitude ?? "?")}</b><br>
    <b>Estimated epicenter</b><br>
    Latitude: ${num(e.lat, 4)}<br>
    Longitude: ${num(e.lon, 4)}<br>
    Origin time: ${esc(isoTime(e.origin_time))}<br>
    <span class="muted">RMS ${num(e.rms_km, 2)} km &middot; ${esc(e.n_stations)} stations</span>`;
}

function updateMap(d) {
  if (!map) return;

  const placed = d.stations.filter(s => s.lat != null && s.lon != null);
  placed.forEach(s => {
    const icon = stationIcon(s), popup = stationPopup(s);
    if (stationMarkers[s.sta]) {
      stationMarkers[s.sta].setLatLng([s.lat, s.lon]).setIcon(icon).setPopupContent(popup);
    } else {
      stationMarkers[s.sta] = L.marker([s.lat, s.lon], { icon, title: s.name })
        .addTo(map).bindPopup(popup);
    }
  });

  // One red circle per located event. Clicking it opens magnitude, coordinates and origin time.
  const located = d.events.filter(e => e.located && e.lat != null && e.lon != null);
  const seen = new Set();
  located.forEach(e => {
    seen.add(e.id);
    const popup = epicenterPopup(e);
    if (epicenterMarkers[e.id]) {
      epicenterMarkers[e.id].setLatLng([e.lat, e.lon]).setPopupContent(popup);
    } else {
      epicenterMarkers[e.id] = L.circleMarker([e.lat, e.lon], {
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

  const missing = d.stations.filter(s => s.lat == null || s.lon == null).map(s => s.name);
  $("mapnote").textContent =
    `${d.counts.located} located event${d.counts.located === 1 ? "" : "s"} · ` +
    `${d.counts.stations_live}/${d.counts.stations_total} stations live` +
    (missing.length ? ` · no coordinates for ${missing.join(", ")}` : "");

  if (!fitted && (placed.length || located.length)) { fitted = true; fitMap(); }
}

function focusEpicenter(id) {
  const marker = epicenterMarkers[id];
  if (!map || !marker) return;
  map.setView(marker.getLatLng(), Math.max(map.getZoom(), 8));
  marker.openPopup();
  $("map").scrollIntoView({ behavior: "smooth", block: "center" });
}

function render(d) {
  $("server").textContent = d.server + " · " + d.status;
  $("clock").textContent = d.now + (d.connected ? "" : " · reconnecting");
  $("error").style.display = d.error ? "block" : "none";
  $("error").textContent = d.error || "";

  $("cards").innerHTML = [
    ["Station status", d.counts.stations_live + " / " + d.counts.stations_total],
    ["Station triggers", d.counts.triggers],
    ["Located events", d.counts.located],
    ["Last scan", d.last_scan ? d.last_scan.slice(11, 19) : "waiting"],
  ].map(([label, value]) => `<div class="card"><span>${esc(label)}</span><b>${esc(value)}</b></div>`).join("");

  $("stations").innerHTML = d.stations.map(s => `
    <tr class="${esc(s.state.replace(" ", "-"))}">
      <td><span class="dot"></span>${esc(s.name)}</td>
      <td>${esc(s.state)}</td>
      <td>${esc(s.last_sample || "-")}</td>
      <td>${s.latency_s == null ? "-" : num(s.latency_s) + " s"}</td>
      <td>${num(s.buffered_s)} s</td>
    </tr>`).join("");

  $("triggers").innerHTML = d.detections.length ? d.detections.map(t => `
    <tr><td>${esc(t.clock)}</td><td>${esc(t.station)}</td><td class="muted">${esc(t.time)}</td></tr>`
  ).join("") : `<tr><td colspan="3" class="muted">No triggers yet.</td></tr>`;

  $("events").innerHTML = d.events.length ? d.events.map(e => {
    const head = e.located
      ? `<b>M ${esc(e.magnitude ?? "?")}</b><span>${esc(e.origin_time)}</span><span class="badge ok">LOCATED</span>`
      : `<b>Detection</b><span>${esc(e.time)}</span><span class="badge bad">NOT LOCATED</span>`;
    const summary = e.located
      ? `${num(e.lat, 3)}, ${num(e.lon, 3)} · RMS ${num(e.rms_km, 2)} km · ${esc(e.n_stations)} stations`
      : esc(e.error || "No location could be computed");
    const map = e.has_map ? `<img class="map" src="/events/${encodeURIComponent(e.id)}/map.png" alt="Event map">` : "";
    const stations = e.stations.length ? `
      <table><thead><tr><th>Station</th><th>Quality</th><th>P</th><th>S</th><th>ML</th></tr></thead><tbody>
      ${e.stations.map(s => `<tr><td>${esc(s.name)}</td><td>${esc(s.quality || "-")}</td>
        <td>${esc((s.p_time || "-").slice(11, 23))}</td><td>${esc((s.s_time || "-").slice(11, 23))}</td>
        <td>${esc(s.ml ?? "-")}</td></tr>`).join("")}
      </tbody></table>` : "";
    const target = e.located ? ` data-event-id="${esc(e.id)}" title="Show on map"` : "";
    return `<div class="event"><div class="head"${target}>${head}</div>
      <div class="muted">${summary}</div>
      <details><summary>Map and station details</summary>${map}${stations}</details></div>`;
  }).join("") : `<div class="muted">No events yet.</div>`;

  document.querySelectorAll("#events .head[data-event-id]").forEach(el => {
    el.onclick = () => focusEpicenter(el.dataset.eventId);
  });

  updateMap(d);
}

async function refresh() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error("HTTP " + response.status);
    render(await response.json());
  } catch (error) {
    $("server").textContent = "Website cannot reach the detector: " + error.message;
  }
}

function refreshWaveforms() {
  const image = new Image();
  image.onload = () => { $("waveforms").src = image.src; };
  image.src = "/waveforms.png?t=" + Date.now();
}

initMap();
refresh();
refreshWaveforms();
setInterval(refresh, 5000);
setInterval(refreshWaveforms, 5000);
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return Response(PAGE, mimetype="text/html")


@app.route("/api/status")
def api_status():
    return jsonify(build_status())


@app.route("/waveforms.png")
def waveforms_png():
    try:
        return Response(get_waveform_png(), mimetype="image/png")
    except Exception as error:
        return Response(f"Could not draw waveforms: {error}", status=500, mimetype="text/plain")


@app.route("/events/<event_id>/map.png")
def event_map(event_id):
    with LOCK:
        event = next((item for item in STATE["events"] if item["id"] == event_id), None)
        png = event.get("map_png") if event else None

    if not png:
        return Response("No map for this event", status=404, mimetype="text/plain")
    return Response(base64.b64decode(png), mimetype="image/png")


def start_web_server():
    print(f"Website: http://127.0.0.1:{WEB_PORT}")
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, use_reloader=False, threaded=True)


def start_detection():
    threading.Thread(target=detection_loop, daemon=True).start()


def main():
    print("Realtime earthquake detector")
    print(f"  SeedLink : {SERVER_URL}")
    print(f"  Website  : http://127.0.0.1:{WEB_PORT}")
    start_detection()
    threading.Thread(target=start_web_server, daemon=True).start()

    try:
        run_seedlink()
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
