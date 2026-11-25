import base64
import json
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Blueprint, Flask, Response, jsonify, request
from obspy import UTCDateTime

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realtime.config import STATIONS, WEB_HOST
from historical import (DEFAULT_SETTINGS, JOBS, JOBS_LOCK, build_params, fmt_time, public_job, run_job, station_id, trim_jobs,)


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
  <div><h1>Historical Earthquake Detection</h1></div>
</header>
<main>
  <section class="panel">
    <form id="run-form">
      <label>Start (UTC) <input type="datetime-local" id="start" step="1"></label>
      <label>End (UTC) <input type="datetime-local" id="end" step="1"></label>

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
      <td>${esc((station.s_time || "-").slice(11, 23))}${station.s_picked === false ? ' <span class="muted">pred</span>' : ''}</td>
      <td>${esc(station.ml ?? "-")}</td>
      <td>${num(station.residual_km, 1)}</td></tr>
  `).join("");

  const table = rows ? `
    <table><thead><tr><th>Station</th><th>Quality</th><th>P</th><th>S</th><th>ML</th><th>Residual</th></tr></thead>
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


def create_blueprint(stations, defaults=None, url_prefix="/historical"):
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

    bp = Blueprint("historical", __name__, url_prefix=url_prefix)

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


def create_app(stations=None, defaults=None):
    """Build the standalone historical detection website."""
    flask_app = Flask(__name__)
    flask_app.register_blueprint(
        create_blueprint(
            STATIONS if stations is None else stations,
            defaults,
            url_prefix="",
        )
    )
    return flask_app


app = create_app()


def main():
    print("Historical detection: http://127.0.0.1:8002/")
    app.run(host=WEB_HOST, port=8002, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
