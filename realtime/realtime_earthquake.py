import io
import json
import threading
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, Response, jsonify, request
from matplotlib.ticker import FuncFormatter
from obspy import Stream, UTCDateTime
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient

from association import (AssociationParams, auto_max_sp_sec, build_candidates, geometry_precheck, pair_ps, prepare_picks,)
from config import ( DETECTION_SETTINGS as D, REALTIME_SETTINGS as R, SEEDLINK_SERVER, STATIONS, WEB_HOST, WEB_PORT,)
from detection import process_detection
from geometry import StationGeometry
from location import locate_event_from_picks
from magnitude import calculate_epicentral_distance, calculate_ml
from pickers import pick_arrival
from preprocess import prepare_components, prepare_stream
from travel_times import TravelTimeTable
from trigger_quality import evaluate_pick_quality

fs = D["fs"]
st_ids = [f"{s['net']}.{s['sta']}" for s in STATIONS]
st_color = ["#0F766E", "#2563EB", "#7C3AED", "#92400E", "#0891B2", "#4D7C0F", "#475569", "#C026D3",]

geometry = StationGeometry()

params = AssociationParams.from_config()
params.max_sp_sec = auto_max_sp_sec(geometry, params.vp, params.vs)
params.validate()

print("Building travel-time table")
t_table = TravelTimeTable(params.travel_depths, params.travel_max_km, params.travel_step_km)

LOCK = threading.Lock()
BUFFERS = {}
plt_cache = {}
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


def fmt_time(value):
    return UTCDateTime(value).strftime("%Y-%m-%d %H:%M:%S") + " UTC"


def buffer_copy(sid):
    with LOCK:
        stream = BUFFERS.get(sid)
        return stream.copy() if stream is not None else None


def latest_data_time():
    with LOCK:
        ends = [t.stats.endtime for s in BUFFERS.values() for t in s]
    return max(ends) if ends else None


# ---------------- waveform processing ----------------
def station_traces(sid, window=None):
    """Filtered Z, horizontal-envelope H and N traces for one station."""
    stream = buffer_copy(sid)
    if stream is None:
        return None

    try:
        parts = prepare_components(
            prepare_stream(stream, fs, D["fmin"], D["fmax"])
        )
    except Exception:
        return None

    if len(parts["Z"]) == 0:
        return None

    traces = {
        "Z": parts["Z"][0],
        "H": parts["H"][0] if parts["H"] else None,
        "N": parts["N"][0] if len(parts["N"]) else None,
    }

    if window:
        for trace in traces.values():
            if trace is not None:
                trace.trim(starttime=trace.stats.endtime - window)

    return traces


def pick_phase(trace, sid, phase, after=None):
    # STA/LTA trigger -> AIC pick -> SNR/AIC quality check.
    try:
        triggers = process_detection(trace.data, trace.stats.starttime, sid, phase)
    except ValueError:
        return []

    picks = []

    for trigger in triggers:
        if after is not None and trigger.detection_time < after:
            continue

        try:
            pick = pick_arrival(trace, trigger, phase)

            idx = int(round((pick.arrival_time - trace.stats.starttime) * fs))

            quality = evaluate_pick_quality(
                trace.data, fs, idx,
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

    return picks


# ---------------- live detection ----------------
def record_detection(sid, onset):
    stamp = onset.timestamp
    with LOCK:
        for old in STATE["detections"]:
            if old["station"] == sid and abs(old["t"] - stamp) < R["dedup_sec"]:
                return
        entry = {
            "station": sid,
            "t": stamp,
            "time": fmt_time(onset),
            "clock": onset.strftime("%H:%M:%S"),
        }
        STATE["detections"].insert(0, entry)
        del STATE["detections"][R["max_log"]:]
    print(f"[trigger] {sid} at {entry['time']}")


def scan_for_event(now):
    with LOCK:
        ignore = STATE["ignore_before"]

    after = now - R["onset_max_age"]
    if ignore is not None and ignore > after:
        after = ignore

    picks = []
    for sid in st_ids:
        traces = station_traces(sid, R["detect_window"])
        if traces is None:
            continue
        for pick in pick_phase(traces["Z"], sid, "P", after):
            if pick.accepted:
                record_detection(sid, pick.arrival_time)
                picks.append(pick)

    picks = prepare_picks(picks, params.dedup_sec)
    candidates = build_candidates(picks, geometry, params.min_stations)
    if not candidates:
        return

    first = candidates[0][0].arrival_time

    with LOCK:
        if STATE["pending"]:
            return
        STATE["pending"] = True
        STATE["status"] = "event detected, waiting for more data"

    print(f"[detect] {len(candidates[0])} stations, first P at {first.isoformat()}")

    timer = threading.Timer(params.wait_sec, process_event, args=(first - R["pre"],))
    timer.daemon = True
    timer.start()

# check for events every 2 seconds, but only if no other event is pending
def detection_loop():
    while True:
        time.sleep(R["detect_every"])

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


# ---------------- location + magnitude ----------------
def padded_box(lats, lons, pad_km):
    dlat = pad_km / 110.574
    dlon = pad_km / (111.32 * np.cos(np.radians(np.mean(lats))))
    return (
        (max(-90.0, min(lats) - dlat), min(90.0, max(lats) + dlat)),
        (max(-180.0, min(lons) - dlon), min(180.0, max(lons) + dlon)),
    )


def estimate_period(data, fs):
    """Dominant period from zero crossings, kept inside the filter band."""
    data = data - np.mean(data)
    crossings = np.count_nonzero(np.diff(np.signbit(data)))
    if crossings < 2:
        return 1.0 / np.sqrt(D["fmin"] * D["fmax"])
    period = 2.0 * len(data) / fs / crossings
    return float(np.clip(period, 1.0 / D["fmax"], 1.0 / D["fmin"]))


def station_ml(traces, onset, delta_km):
    """Approximate station ML from the 15 s after the S (or P) arrival."""
    amp_trace = traces["H"] or traces["Z"]
    per_trace = traces["N"] or traces["Z"]

    amp = amp_trace.slice(onset, onset + 15).data
    per = per_trace.slice(onset, onset + 15).data
    if len(amp) < 20 or len(per) < 20:
        return None

    period = estimate_period(per, fs)
    peak_velocity = np.max(np.abs(amp)) / R["counts_per_m"]  # m/s
    displacement_nm = peak_velocity * period / (2 * np.pi) * 1e9

    try:
        return calculate_ml(displacement_nm, period, delta_km)
    except ValueError:
        return None


def new_event(start, **fields):
    with LOCK:
        STATE["event_seq"] += 1
        seq = STATE["event_seq"]

    event = {
        "id": f"{int(time.time())}-{seq}",
        "detected": fmt_time(UTCDateTime()),
        "time": fmt_time(UTCDateTime(start) + R["pre"]),
        "located": False,
        "error": None,
        "origin_time": None,
        "lat": None,
        "lon": None,
        "depth_km": None,
        "magnitude": None,
        "rms_sec": None,
        "n_stations": 0,
        "stations": [],
        "has_map": False,
    }
    event.update(fields)
    return event


def build_event(start):
    # 1. Pick P (vertical) and S (horizontal envelope) at every station.
    traces = {}
    picks = []

    for sid in st_ids:
        t = station_traces(sid)
        if t is None:
            continue

        p_picks = pick_phase(t["Z"], sid, "P", start)
        good_p = [p for p in p_picks if p.accepted]
        if not good_p:
            continue

        s_picks = []
        if t["H"] is not None:
            s_start = min(p.arrival_time for p in good_p) + params.min_sp_sec
            s_picks = pick_phase(t["H"], sid, "S", s_start)

        traces[sid] = t
        picks += p_picks + s_picks

    # 2. Associate: group compatible P picks, pair S picks, sanity-check geometry.
    picks = prepare_picks(picks, params.dedup_sec)
    candidates = build_candidates(picks, geometry, params.min_stations)
    if not candidates:
        raise RuntimeError("No group of stations with compatible P arrivals")

    candidate = max(candidates, key=len)
    pairs, _ = pair_ps(picks, params.min_sp_sec, params.max_sp_sec)
    check = geometry_precheck(candidate, pairs, geometry, params.vp, params.vs)
    bad = set(check["suspicious_stations"])

    used = list(candidate)
    for p_pick, s_pick in pairs:
        if p_pick in candidate and s_pick.station_id not in bad:
            used.append(s_pick)

    # 3. Locate: coarse grid, then a fine grid around the best point.
    lats = [s["latitude"] for s in geometry.stations.values()]
    lons = [s["longitude"] for s in geometry.stations.values()]
    lat_range, lon_range = padded_box(lats, lons, params.grid_padding_km)

    result = locate_event_from_picks(
        used, geometry.stations, t_table, params.travel_depths,
        lat_range, lon_range, params.grid_coarse_km,
    )

    lat_range, lon_range = padded_box(
        [result["latitude"]], [result["longitude"]], 2 * params.grid_coarse_km
    )
    fine = locate_event_from_picks(
        used, geometry.stations, t_table, [result["depth_km"]],
        lat_range, lon_range, params.grid_fine_km,
    )
    if fine["rms_sec"] <= result["rms_sec"]:
        result = fine

    if result["rms_sec"] > params.max_rms_sec:
        raise RuntimeError(
            f"RMS {result['rms_sec']:.2f} s is above the limit "
            f"({params.max_rms_sec} s)"
        )

    # 4. Magnitude: median of the station values.
    by_station = {}
    for detail in result["picks"]:
        by_station.setdefault(detail["station_id"], {})[detail["phase"]] = detail

    snr = {(p.station_id, p.phase): p.snr for p in used}
    rows, magnitudes = [], []

    for sid, phases in by_station.items():
        onset = (phases.get("S") or phases["P"])["observed_arrival"]
        station = geometry.stations[sid]

        delta = calculate_epicentral_distance(
            station["latitude"], station["longitude"],
            result["latitude"], result["longitude"],
        )
        ml = station_ml(traces[sid], onset, delta)
        if ml is not None:
            magnitudes.append(ml)

        p_snr = snr.get((sid, "P"))
        rows.append({
            "name": sid,
            "quality": "-" if p_snr is None else f"SNR {p_snr:.1f}",
            "p_time": str(phases["P"]["observed_arrival"]) if "P" in phases else None,
            "s_time": str(phases["S"]["observed_arrival"]) if "S" in phases else None,
            "ml": None if ml is None else round(ml, 1),
        })

    return new_event(
        start,
        located=True,
        time=fmt_time(result["origin_time"]),
        origin_time=fmt_time(result["origin_time"]),
        lat=round(result["latitude"], 4),
        lon=round(result["longitude"], 4),
        depth_km=result["depth_km"],
        magnitude=round(float(np.median(magnitudes)), 1) if magnitudes else None,
        rms_sec=round(result["rms_sec"], 3),
        n_stations=len(by_station),
        stations=rows,
    )


def process_event(start):
    with LOCK:
        STATE["status"] = "locating"
        STATE["error"] = None

    try:
        event = build_event(start)
        print(f"[event] M {event['magnitude']} at {event['origin_time']} "
              f"({event['lat']:.3f}, {event['lon']:.3f}, {event['depth_km']:.0f} km)")
        with open(f"event_{event['id']}.json", "w", encoding="utf-8") as handle:
            json.dump(event, handle, indent=2)
    except Exception as error:
        print(f"[locate] not located: {error}")
        event = new_event(start, error=str(error))
        with LOCK:
            STATE["error"] = str(error)

    latest = latest_data_time()

    with LOCK:
        STATE["events"].insert(0, event)
        del STATE["events"][R["max_events"]:]
        if latest is not None:
            # Skip the rest of this event's coda so it is not detected twice.
            STATE["ignore_before"] = min(latest, UTCDateTime()) + params.dead_time_sec
        STATE["pending"] = False
        STATE["status"] = "listening"


# ---------------- SeedLink ----------------
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
            newest = max(t.stats.endtime for t in stream)
            stream.trim(starttime=newest - R["buffer"])
        except Exception:
            pass
        STATE["last_data"] = time.time()
        STATE["connected"] = True


def run_seedlink():
    while True:
        client = None
        try:
            client = EasySeedLinkClient(SEEDLINK_SERVER, autoconnect=False)
            client.on_data = on_data
            client.conn.timeout = 10
            client.connect()

            for station in STATIONS:
                # SeedLink selectors use "?" as the wildcard.
                client.select_stream(
                    station["net"], station["sta"], station["cha"].replace("*", "?")
                )

            with LOCK:
                STATE["connected"] = True
                STATE["status"] = "listening"
                STATE["error"] = None

            print(f"Connected to {SEEDLINK_SERVER}, listening on {len(STATIONS)} stations...")
            client.run()
        except KeyboardInterrupt:
            raise
        except Exception as error:
            print(f"[seedlink] {error}; reconnecting in {R['reconnect']}s")
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

        time.sleep(R["reconnect"])


# ---------------- website data ----------------
def station_rows():
    now = UTCDateTime()
    now_ts = time.time()
    rows = []

    with LOCK:
        detections = list(STATE["detections"])

        for station in STATIONS:
            sid = f"{station['net']}.{station['sta']}"
            stream = BUFFERS.get(sid)

            row = {
                "name": sid, "net": station["net"], "sta": station["sta"],
                "lat": station["lat"], "lon": station["lon"],
                "state": "no data", "last_sample": None,
                "latency_s": None, "buffered_s": 0.0,
            }

            if stream is not None and len(stream) > 0:
                end = max(t.stats.endtime for t in stream)
                start = min(t.stats.starttime for t in stream)
                latency = round(float(now - end), 1)
                triggered = any(
                    d["station"] == sid and now_ts - d["t"] < R["active"]
                    for d in detections
                )

                if latency > R["active"]:
                    state = "stale"
                elif triggered:
                    state = "triggered"
                else:
                    state = "live"

                row.update(
                    state=state,
                    last_sample=end.strftime("%H:%M:%S"),
                    latency_s=latency,
                    buffered_s=round(float(end - start), 1),
                )

            rows.append(row)

    return rows


def build_status():
    rows = station_rows()

    with LOCK:
        detections = [dict(d) for d in STATE["detections"][:50]]
        events = [dict(e) for e in STATE["events"]]
        snapshot = {k: STATE[k] for k in
                    ("status", "connected", "error", "last_scan", "last_data", "started")}

    return {
        "server": SEEDLINK_SERVER,
        "now": fmt_time(UTCDateTime()),
        "status": snapshot["status"],
        "connected": snapshot["connected"],
        "error": snapshot["error"],
        "last_scan": None if snapshot["last_scan"] is None else fmt_time(UTCDateTime(snapshot["last_scan"])),
        "last_data": None if snapshot["last_data"] is None else fmt_time(UTCDateTime(snapshot["last_data"])),
        "uptime_s": round(time.time() - snapshot["started"], 1),
        "stations": rows,
        "detections": detections,
        "events": events,
        "counts": {
            "stations_live": sum(1 for r in rows if r["state"] in ("live", "triggered")),
            "stations_total": len(STATIONS),
            "triggers": len(detections),
            "events": len(events),
            "located": sum(1 for e in events if e["located"]),
        },
    }


# ---------------- waveform plot ----------------
def make_waveform_png(sids, window_sec=R["waveform"]):
    with LOCK:
        detections = list(STATE["detections"])

    traces = {sid: station_traces(sid) for sid in sids}
    ends = [t["Z"].stats.endtime for t in traces.values() if t is not None]
    end = max(ends) if ends else UTCDateTime()
    start = end - window_sec

    fig, axes = plt.subplots(
        len(sids), 1, sharex=True,
        figsize=(8, max(3.6, 2.1 * len(sids) + 0.9)),
    )
    axes = np.atleast_1d(axes)

    for i, (ax, sid) in enumerate(zip(axes, sids)):
        color = st_color[i % len(st_color)]
        ax.set_ylabel("Normalized\namplitude", fontsize=8.5)
        ax.text(0.006, 0.92, sid, transform=ax.transAxes, va="top",
                fontsize=10, fontweight="bold", color=color)

        try:
            z = traces[sid]["Z"].copy().trim(start, end, pad=True, fill_value=0)
            data = z.data.astype(float) / max(R["counts_per_m"], 1e-9)
            scale = float(np.percentile(np.abs(data), 99)) or float(np.max(np.abs(data))) or 1.0
        except Exception:
            ax.set_yticks([])
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="#8a96a3")
            continue

        x = np.arange(len(data)) / z.stats.sampling_rate + (z.stats.starttime - end)
        ax.plot(x, data / scale, color=color, linewidth=0.9)
        ax.set_ylim(-1.05, 1.05)
        ax.text(0.994, 0.92, f"p99 {scale:.2e}", transform=ax.transAxes,
                va="top", ha="right", fontsize=8, color="#63707d")

        for d in detections:
            if d["station"] == sid and start <= UTCDateTime(d["t"]) <= end:
                ax.axvline(UTCDateTime(d["t"]) - end, color="#D6361F", linewidth=1.1)

    for ax in axes:
        ax.set_xlim(-window_sec, 0)

    axes[-1].xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: (end + float(v)).strftime("%H:%M:%S"))
    )
    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle(f"Live waveforms, last {window_sec:g} s (red line = station trigger)",
                 fontsize=10.5, fontweight="bold")
    fig.subplots_adjust(left=0.135, right=0.995, top=0.95, bottom=0.075, hspace=0.28)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def get_waveform_png(sids):
    key = tuple(sids)
    with LOCK:
        cached = plt_cache.get(key)
    if cached and time.time() - cached[0] < R["plt_refresh"]:
        return cached[1]

    png = make_waveform_png(sids)
    with LOCK:
        plt_cache[key] = (time.time(), png)
    return png


def parse_station_selection(value):
    """'NET.STA,NET.STA' -> list of known station ids (all stations if empty)."""
    if not value:
        return list(st_ids)
    wanted = {part.strip() for part in value.split(",") if part.strip()}
    return [sid for sid in st_ids if sid in wanted or sid.split(".")[1] in wanted]


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
  .st-mark.selected svg polygon { stroke:#2563eb; stroke-width:2.6; }
  .legend { display:flex; gap:16px; flex-wrap:wrap; margin-top:8px; color:var(--muted); font-size:12.5px; }
  .legend span { display:inline-flex; align-items:center; gap:6px; }
  .legend .tri { width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent;
                border-bottom:11px solid var(--ok); }
  .legend .epi { width:11px; height:11px; border-radius:50%; background:var(--bad); border:2px solid #fff;
                box-shadow:0 0 0 1px rgba(20,32,43,.4); }
  .leaflet-popup-content { font:13px/1.45 system-ui, sans-serif; margin:10px 12px; }
  .event .head { cursor:pointer; }
  .mapactions { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }

  /* right-side live waveform drawer */
  .drawer { position:fixed; top:0; right:0; bottom:0; width:min(560px, 94vw); background:var(--panel);
            border-left:1px solid var(--line); box-shadow:-18px 0 40px rgba(20,32,43,.18);
            transform:translateX(105%); transition:transform .25s ease; z-index:1200;
            display:flex; flex-direction:column; }
  .drawer.open { transform:translateX(0); }
  .drawerhead { display:flex; align-items:flex-start; justify-content:space-between; gap:10px;
                padding:14px 16px; border-bottom:1px solid var(--line); }
  .drawerhead h2 { margin:0 0 3px; }
  .drawerbody { padding:12px 16px 18px; overflow:auto; }
  .wavechips { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px; }
  .wavechip { display:inline-flex; align-items:center; gap:6px; font-size:12px; padding:2px 6px 2px 9px;
              border-radius:12px; background:#eef1f4; color:var(--ink); }
  .wavechip button { border:0; background:none; cursor:pointer; font-size:14px; line-height:1;
                     color:var(--muted); padding:0 1px; }
  .wavechip button:hover { color:var(--bad); }
  #waveimg { display:block; width:100%; height:auto; border:1px solid var(--line); border-radius:6px; background:#fff; }

  /* compare stations modal */
  .modal-backdrop { position:fixed; inset:0; z-index:1300; background:rgba(20,32,43,.45);
                    display:flex; align-items:center; justify-content:center; padding:18px; }
  .modal-backdrop[hidden] { display:none; }
  .modal { background:var(--panel); border-radius:8px; width:min(520px, 100%); max-height:85vh;
           display:flex; flex-direction:column; box-shadow:0 20px 60px rgba(20,32,43,.35); }
  .modalhead, .modalfoot { display:flex; align-items:center; gap:8px; padding:13px 16px; }
  .modalhead { justify-content:space-between; border-bottom:1px solid var(--line); }
  .modalhead h2 { margin:0; }
  .modalfoot { justify-content:flex-end; border-top:1px solid var(--line); }
  .modalbody { padding:12px 16px; overflow:auto; }
  .station-picker { display:grid; grid-template-columns:1fr 1fr; gap:4px 12px; margin-top:8px; }
  .station-picker label { display:flex; align-items:center; gap:8px; padding:5px 6px; border-radius:4px; cursor:pointer; }
  .station-picker label:hover { background:#f4f6f8; }
  .station-picker .sid { font-weight:600; }
  .btn { font:inherit; font-size:13px; border-radius:5px; padding:6px 14px; cursor:pointer;
         border:1px solid var(--line); background:var(--panel); color:var(--ink); }
  .btn.primary { background:var(--blue); border-color:var(--blue); color:#fff; font-weight:600; }
  .btn.primary:hover { filter:brightness(1.05); }
  @media (max-width:600px) { .station-picker { grid-template-columns:1fr; } }
  #error { display:none; margin-bottom:14px; padding:10px 13px; border:1px solid #e8b4aa;
           background:#fbeae6; color:#8a2412; border-radius:6px; }
</style>
</head>
<body>
<header>
  <div><h1>Live Earthquake Monitor</h1><div class="muted" id="server">Connecting...</div></div>
  <div class="muted" style="display:flex;gap:14px;align-items:center;flex-wrap:wrap">
    <span id="clock">Loading...</span>
  </div>
</header>
<main>
  <div id="error"></div>
  <div class="cards" id="cards"></div>

  <section>
    <div class="maphead">
      <h2>Station map</h2>
      <div class="mapactions">
        <button class="linkbtn" id="compare">Compare stations</button>
        <button class="linkbtn" id="fit">Fit to map</button>
      </div>
    </div>
   
    <div id="map"></div>
    <div class="legend">
      <span><i class="tri"></i>Station</span>
      <span><i class="epi"></i>Estimated epicenter</span>
      <span class="muted" id="mapnote"></span>
    </div>
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

<aside class="drawer" id="wavepanel" aria-label="Live waveforms">
  <div class="drawerhead">
    <div>
      <h2>Live waveforms</h2>
      <div class="muted" id="wavenote"></div>
    </div>
    <button class="linkbtn" id="waveclose">Close</button>
  </div>
  <div class="drawerbody">
    <div class="wavechips" id="wavechips"></div>
    <img id="waveimg" alt="Live waveforms for the selected stations">
  </div>
</aside>

<div class="modal-backdrop" id="compare-backdrop" hidden>
  <div class="modal" role="dialog" aria-modal="true" aria-labelledby="compare-title">
    <div class="modalhead">
      <h2 id="compare-title">Compare stations</h2>
      <button class="linkbtn" id="compare-close">Close</button>
    </div>
    <div class="modalbody">
      <div class="muted">Choose the stations to show in the live waveform panel.</div>
      <div class="station-picker" id="station-picker"></div>
    </div>
    <div class="modalfoot">
      <button class="btn" id="compare-all">Select all</button>
      <button class="btn" id="compare-none">Clear</button>
      <button class="btn primary" id="compare-apply">Show waveforms</button>
    </div>
  </div>
</div>

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
let selectedStations = new Set(), stationInfo = {}, lastStatus = null, waveRequest = 0;

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
  const cls = (s.state === "triggered" ? " triggered" : "") +
              (selectedStations.has(s.name) ? " selected" : "");
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
    <span class="muted">RMS ${num(e.rms_sec, 2)} s &middot; ${esc(e.n_stations)} stations</span>`;
}

function updateMap(d) {
  if (!map) return;

  d.stations.forEach(s => { stationInfo[s.name] = s; });

  const placed = d.stations.filter(s => s.lat != null && s.lon != null);
  placed.forEach(s => {
    const icon = stationIcon(s), popup = stationPopup(s);
    if (stationMarkers[s.sta]) {
      stationMarkers[s.sta].setLatLng([s.lat, s.lon]).setIcon(icon).setPopupContent(popup);
    } else {
      const marker = L.marker([s.lat, s.lon], { icon, title: s.name })
        .addTo(map).bindPopup(popup);
      marker.on("click", () => toggleStation(s.name));
      stationMarkers[s.sta] = marker;
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

function stationLabel(name) {
  return stationInfo[name] ? stationInfo[name].sta : name;
}

function selectionNote(names) {
  return `${names.length} station${names.length === 1 ? "" : "s"} selected · each trace normalized to its 99th-percentile amplitude`;
}

function refreshStationIcons() {
  if (!lastStatus) return;
  lastStatus.stations.forEach(s => {
    const marker = stationMarkers[s.sta];
    if (marker) marker.setIcon(stationIcon(s));
  });
}

function refreshWaveforms() {
  const names = [...selectedStations];
  if (!names.length) return;
  const request = ++waveRequest;
  const url = "/waveforms.png?stations=" + encodeURIComponent(names.join(",")) + "&t=" + Date.now();
  const image = new Image();
  image.onload = () => {
    if (request === waveRequest) {
      $("waveimg").src = image.src;
      $("wavenote").textContent = selectionNote(names);
    }
  };
  image.onerror = () => {
    if (request === waveRequest) $("wavenote").textContent = "Could not load waveforms for the selected stations.";
  };
  image.src = url;
}

function renderWaveSelection() {
  const names = [...selectedStations];
  $("wavepanel").classList.toggle("open", names.length > 0);
  $("wavechips").innerHTML = names.map(name => `
    <span class="wavechip">${esc(stationLabel(name))}
      <button type="button" data-remove="${esc(name)}" title="Remove ${esc(stationLabel(name))}">&times;</button>
    </span>`).join("");
  $("wavechips").querySelectorAll("[data-remove]").forEach(button => {
    button.onclick = () => {
      selectedStations.delete(button.dataset.remove);
      renderWaveSelection();
    };
  });
  $("wavenote").textContent = names.length ? selectionNote(names) : "";
  if (names.length) refreshWaveforms();
  refreshStationIcons();
}

function toggleStation(name) {
  if (selectedStations.has(name)) selectedStations.delete(name);
  else selectedStations.add(name);
  renderWaveSelection();
}

function openCompare() {
  const stations = lastStatus ? lastStatus.stations : [];
  $("station-picker").innerHTML = stations.length ? stations.map(s => `
    <label><input type="checkbox" value="${esc(s.name)}"${selectedStations.has(s.name) ? " checked" : ""}>
      <span class="sid">${esc(s.sta)}</span><span class="muted">${esc(s.name)}</span></label>`).join("")
    : `<div class="muted">The station list is not available yet.</div>`;
  $("compare-backdrop").hidden = false;
}

function closeCompare() { $("compare-backdrop").hidden = true; }

function applyCompare() {
  selectedStations = new Set(
    [...$("station-picker").querySelectorAll("input:checked")].map(input => input.value)
  );
  closeCompare();
  renderWaveSelection();
}

function render(d) {
  lastStatus = d;
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
      ? `${num(e.lat, 3)}, ${num(e.lon, 3)} · ${num(e.depth_km, 0)} km deep · RMS ${num(e.rms_sec, 2)} s · ${esc(e.n_stations)} stations`
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

$("compare").onclick = openCompare;
$("compare-close").onclick = closeCompare;
$("compare-apply").onclick = applyCompare;
$("compare-all").onclick = () =>
  $("station-picker").querySelectorAll("input").forEach(input => input.checked = true);
$("compare-none").onclick = () =>
  $("station-picker").querySelectorAll("input").forEach(input => input.checked = false);
$("compare-backdrop").addEventListener("click", event => {
  if (event.target === $("compare-backdrop")) closeCompare();
});
$("waveclose").onclick = () => { selectedStations.clear(); renderWaveSelection(); };
document.addEventListener("keydown", event => { if (event.key === "Escape") closeCompare(); });

initMap();
refresh();
setInterval(refresh, 5000);
setInterval(refreshWaveforms, 5000);
</script>
</body>
</html>
"""


# ---------------- routes ----------------
app = Flask(__name__)


@app.route("/")
def home():
    return Response(PAGE, mimetype="text/html")


@app.route("/api/status")
def api_status():
    return jsonify(build_status())


@app.route("/waveforms.png")
def waveforms_png():
    raw = request.args.get("stations", "")
    sids = parse_station_selection(raw)
    if raw and not sids:
        return Response("No known stations in the selection", status=400, mimetype="text/plain")
    try:
        return Response(get_waveform_png(sids), mimetype="image/png",
                        headers={"Cache-Control": "no-store"})
    except Exception as error:
        return Response(f"Could not draw waveforms: {error}", status=500, mimetype="text/plain")


def main():
    threading.Thread(target=detection_loop, daemon=True).start()
    threading.Thread(target=lambda: app.run(host=WEB_HOST, port=WEB_PORT, debug=False,use_reloader=False, threaded=True),daemon=True,).start()

    try:
        run_seedlink()
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
