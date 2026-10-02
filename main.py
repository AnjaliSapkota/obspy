from flask import Flask, jsonify, render_template_string, request
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

import test3 as core

app = Flask(__name__)

DEFAULT_BASE_URL = "https://seiscomp.alertnepal.online"
DEFAULT_STATIONS = [
    {"sta": "EQM13", "lat": 28.256323, "lon": 85.367569, "net": "NP", "cha": "EHZ", "loc": "*"},
    {"sta": "EQM08", "lat": 27.831,    "lon": 86.65,     "net": "NP", "cha": "EHZ", "loc": "*"},
    {"sta": "EQM10", "lat": 28.299517, "lon": 83.960148, "net": "NP", "cha": "EHZ", "loc": "*"},
]

_coord_cache = {}


def lookup_station(base_url, net, sta):
    """Return (lat, lon, elevation) for a station from the FDSN server."""
    key = (base_url, net, sta)
    if key not in _coord_cache:
        inv = Client(base_url).get_stations(network=net, station=sta, level="station")
        found = None
        for network in inv:
            for station in network:
                found = (station.latitude, station.longitude, station.elevation)
                break
            if found:
                break
        if not found:
            raise ValueError(f"Station {net}.{sta} not found on {base_url}")
        _coord_cache[key] = found
    return _coord_cache[key]


@app.route("/api/station")
def station_lookup():
    net = request.args.get("net", "").strip()
    sta = request.args.get("sta", "").strip()
    base_url = request.args.get("base_url") or DEFAULT_BASE_URL
    if not net or not sta:
        return jsonify(error="Network and station codes are required."), 400
    try:
        lat, lon, elev = lookup_station(base_url, net, sta)
        return jsonify(net=net, sta=sta, lat=lat, lon=lon, elevation=elev)
    except ValueError as e:
        return jsonify(error=str(e)), 404
    except Exception as e:
        # obspy raises FDSNNoDataException (204) when nothing matches
        return jsonify(error=f"No coordinates found for {net}.{sta} ({type(e).__name__})"), 404


@app.route("/api/run", methods=["POST"])
def run():
    try:
        p = request.get_json()
        base_url = p.get("base_url") or DEFAULT_BASE_URL

        # fill in any station that still has no coordinates
        for s in p["stations"]:
            if s.get("lat") in (None, "") or s.get("lon") in (None, ""):
                s["lat"], s["lon"], _ = lookup_station(base_url, s["net"], s["sta"])

        result = core.run_location(
            base_url=base_url,
            stations=p["stations"],
            start=UTCDateTime(p["start"]),
            end=UTCDateTime(p["end"]),
            fs=float(p["fs"]),
            fmin=float(p["fmin"]),
            fmax=float(p["fmax"]),
            wait_sec=float(p["wait_sec"]),
            vp=float(p["vp"]),
            vs=float(p["vs"]),
            sta_p=float(p["sta_p"]),
            lta_p=float(p["lta_p"]),
            thr_on=float(p["thr_on"]),
            thr_off=float(p["thr_off"]),
            sta_s=float(p["sta_s"]),
            lta_s=float(p["lta_s"]),
            min_sp_sec=float(p["min_sp_sec"]),
            max_sp_sec=float(p["max_sp_sec"]),
            thr_on_s=float(p["thr_on_s"]),
            thr_off_s=float(p["thr_off_s"]),
        )
        return jsonify(**result)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


@app.route("/")
def index():
    return render_template_string(PAGE, stations=DEFAULT_STATIONS, base_url=DEFAULT_BASE_URL)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Earthquake locator</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root{--bg:#eef1f3;--panel:#fff;--ink:#16232b;--mute:#5d6c75;--line:#cdd6db;--accent:#0b6e75;--hot:#d1432b;--good:#1f9e6b;--review:#d97706}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Segoe UI",system-ui,sans-serif}
header{padding:22px 28px 6px}
h1{margin:0;font-size:24px;font-weight:650;letter-spacing:-.01em}
header p{margin:4px 0 0;color:var(--mute);max-width:72ch}
main{display:grid;grid-template-columns:340px 1fr;gap:20px;padding:18px 28px 40px}
@media(max-width:980px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px}
h2{font-size:15px;margin:0 0 10px}
label{display:block;font-size:13px;color:var(--mute);margin:10px 0 3px}
input[type=text],input[type=number]{width:100%;padding:7px 9px;border:1px solid var(--line);border-radius:5px;font:inherit;color:var(--ink);background:#fff}
input:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.sta{border:1px solid var(--line);border-radius:6px;padding:8px 10px;margin-bottom:8px;position:relative}
.sta .name{font-weight:600;font-size:13.5px;margin-bottom:6px}
.sta .row3 input{font-size:12.5px;padding:5px 6px}
.sta .rm{position:absolute;top:6px;right:8px;background:none;color:var(--mute);padding:0 4px;font-size:16px;border:none;cursor:pointer}
details{margin-top:12px}
summary{cursor:pointer;font-size:13px;color:var(--mute)}
button{font:inherit;border:0;border-radius:5px;padding:8px 12px;cursor:pointer;background:var(--ink);color:#fff}
button.add{width:100%;margin-top:4px;background:#fff;color:var(--ink);border:1px dashed var(--line)}
button.go{width:100%;margin-top:14px;background:var(--accent);padding:11px;font-weight:600}
button:disabled{opacity:.55;cursor:wait}
#status{margin-top:10px;font-size:13px;color:var(--mute)}
#status.err{color:var(--hot)}
.result{display:none}
.big{display:flex;gap:26px;flex-wrap:wrap;margin-bottom:12px}
.big div span{display:block;font-size:12px;color:var(--mute)}
.big div b{font-size:22px;font-weight:650}
#map{height:320px;border-radius:6px;border:1px solid var(--line);margin-bottom:14px}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin:10px 0}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{color:var(--mute);font-weight:500}
.stack{display:grid;gap:16px}
img.plot{max-width:100%;border-radius:6px;border:1px solid var(--line);display:block;margin-top:8px}
.stacard{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px}
.stacard h3{margin:0 0 8px;font-size:15px;display:flex;justify-content:space-between;align-items:center}
.badge{padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;color:#fff}
.badge.GOOD{background:var(--good)}
.badge.REVIEW{background:var(--review)}
.badge.REJECT{background:var(--hot)}
.stacard .err{color:var(--hot);font-size:13px;margin:4px 0}
.notes{font-size:12px;color:var(--mute);margin-top:6px}
</style></head><body>
<header><h1>Earthquake locator</h1>
<p>Picks P and S arrivals with recursive STA/LTA, converts S&ndash;P time to distance, and trilaterates the epicenter from circle intersections across three stations.</p></header>
<main>
<div class="panel">
  <h2>Data source</h2>
  <label>FDSN base URL</label><input type="text" id="base_url" value="{{ base_url }}">
  <h2 style="margin-top:14px">Stations</h2>
  <div id="stations"></div>
  <button class="add" onclick="addStation()">+ Add station</button>

  <label style="margin-top:14px">Start (UTC)</label><input type="text" id="start" value="2026-09-22T07:45:00">
  <label>End (UTC)</label><input type="text" id="end" value="2026-09-22T07:55:00">
  <div class="row"><div><label>Band low (Hz)</label><input type="number" id="fmin" value="1.0" step="0.1"></div>
  <div><label>Band high (Hz)</label><input type="number" id="fmax" value="8" step="0.1"></div></div>
  <div class="row"><div><label>Sample rate (Hz)</label><input type="number" id="fs" value="20" step="1"></div>
  <div><label>Wait (s)</label><input type="number" id="wait_sec" value="30" step="5"></div></div>
  <div class="row"><div><label>Vp (km/s)</label><input type="number" id="vp" value="6.0" step="0.1"></div>
  <div><label>Vs (km/s)</label><input type="number" id="vs" value="3.5" step="0.1"></div></div>

  <details>
    <summary>P-pick STA/LTA settings</summary>
    <div class="row"><div><label>STA (s)</label><input type="number" id="sta_p" value="1.0" step="0.1"></div>
    <div><label>LTA (s)</label><input type="number" id="lta_p" value="10.0" step="0.5"></div></div>
    <div class="row"><div><label>Trigger on</label><input type="number" id="thr_on" value="3.5" step="0.1"></div>
    <div><label>Trigger off</label><input type="number" id="thr_off" value="1.5" step="0.1"></div></div>
  </details>
  <details>
    <summary>S-pick settings</summary>
    <div class="row"><div><label>STA (s)</label><input type="number" id="sta_s" value="1.0" step="0.1"></div>
    <div><label>LTA (s)</label><input type="number" id="lta_s" value="8.0" step="0.5"></div></div>
    <div class="row"><div><label>Min S&ndash;P (s)</label><input type="number" id="min_sp_sec" value="4.0" step="0.5"></div>
    <div><label>Max S&ndash;P (s)</label><input type="number" id="max_sp_sec" value="50.0" step="1"></div></div>
    <div class="row"><div><label>Trigger on</label><input type="number" id="thr_on_s" value="2.5" step="0.1"></div>
    <div><label>Trigger off</label><input type="number" id="thr_off_s" value="1.0" step="0.1"></div></div>
  </details>

  <button class="go" id="go" onclick="run()">Locate event</button>
  <div id="status"></div>
</div>
<div class="stack">
  <div class="panel result" id="res">
    <div class="big" id="big"></div>
    <div id="map"></div>
    <table id="tbl"></table>
    <h2>Circle intersection (trilateration)</h2>
    <img class="plot" id="circ">
  </div>
  <div class="stack" id="stacards"></div>
</div>
</main>
<script>
let stations = {{ stations|tojson }};
let map, layer;

function drawStations(){
  document.getElementById('stations').innerHTML = stations.map((s,i) => `
    <div class="sta">
      <button class="rm" onclick="removeStation(${i})" title="Remove">\u00d7</button>
      <div class="name">Station ${i+1}</div>
      <div class="row3">
        <input type="text" value="${s.sta}" onchange="stations[${i}].sta=this.value.trim().toUpperCase();lookup(${i})" placeholder="STA">
        <input type="text" value="${s.net}" onchange="stations[${i}].net=this.value.trim().toUpperCase();lookup(${i})" placeholder="NET">
        <input type="text" value="${s.loc}" onchange="stations[${i}].loc=this.value" placeholder="LOC">
      </div>
      <div class="row3" style="margin-top:6px">
        <input type="text" value="${s.cha}" onchange="stations[${i}].cha=this.value" placeholder="CHA">
        <input type="number" step="0.0001" value="${s.lat ?? ''}" onchange="stations[${i}].lat=parseFloat(this.value)" placeholder="lat (auto)">
        <input type="number" step="0.0001" value="${s.lon ?? ''}" onchange="stations[${i}].lon=parseFloat(this.value)" placeholder="lon (auto)">
      </div>
    </div>`).join('');
}
drawStations();

async function lookup(i){
  const s = stations[i];
  if(!s.sta || !s.net) return;
  setStatus(`Looking up ${s.net}.${s.sta}\u2026`);
  try{
    const q = new URLSearchParams({net:s.net, sta:s.sta, base_url:v('base_url')});
    const r = await fetch('/api/station?' + q);
    const d = await r.json();
    if(!r.ok) throw new Error(d.error);
    s.lat = d.lat; s.lon = d.lon;
    drawStations();
    setStatus(`${s.net}.${s.sta}: ${d.lat.toFixed(4)}, ${d.lon.toFixed(4)}`);
  }catch(e){
    s.lat = null; s.lon = null;
    drawStations();
    setStatus(e.message, true);
  }
}

function addStation(){
  stations.push({sta:"", net:"", loc:"*", cha:"EHZ", lat:null, lon:null});
  drawStations();
}
function removeStation(i){
  if(stations.length<=3){ setStatus('At least 3 stations are needed.', true); return; }
  stations.splice(i,1); drawStations();
}
function setStatus(t, err){ const s=document.getElementById('status'); s.textContent=t; s.className=err?'err':''; }
const v = id => document.getElementById(id).value;

async function run(){
  const btn = document.getElementById('go'); btn.disabled = true;
  const bad = stations.findIndex(s => !s.sta || !s.net || !Number.isFinite(s.lat) || !Number.isFinite(s.lon));
  if(bad >= 0){ setStatus(`Station ${bad+1} is missing a code or coordinates.`, true); btn.disabled = false; return; }
  setStatus('Downloading waveforms and processing. This can take a minute.');
  try{
    const r = await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        base_url:v('base_url'), stations, start:v('start'), end:v('end'),
        fmin:v('fmin'), fmax:v('fmax'), fs:v('fs'), wait_sec:v('wait_sec'),
        vp:v('vp'), vs:v('vs'),
        sta_p:v('sta_p'), lta_p:v('lta_p'), thr_on:v('thr_on'), thr_off:v('thr_off'),
        sta_s:v('sta_s'), lta_s:v('lta_s'), min_sp_sec:v('min_sp_sec'), max_sp_sec:v('max_sp_sec'),
        thr_on_s:v('thr_on_s'), thr_off_s:v('thr_off_s')
      })});
    const d = await r.json();
    if(!r.ok) throw new Error(d.error);
    show(d);
    setStatus(d.located ? 'Done.' : (d.error || 'Could not locate the event.'), !d.located);
  }catch(e){ setStatus(e.message, true); }
  btn.disabled = false;
}

function show(d){
  document.getElementById('res').style.display='block';

  document.getElementById('tbl').innerHTML =
    '<tr><th>Station</th><th>P time (UTC)</th><th>S time (UTC)</th><th>S\u2013P (s)</th><th>Distance (km)</th><th>Fit residual (km)</th></tr>' +
    d.stations.map(s => s.p_time
      ? `<tr><td>${s.name}</td><td>${s.p_time.slice(11,23)}</td><td>${s.s_time.slice(11,23)}</td><td>${s.sp_diff.toFixed(2)}</td><td>${s.distance_km.toFixed(1)}</td><td${Math.abs(s.residual_km||0)>15?' style="color:var(--hot)"':''}>${s.residual_km!==undefined?s.residual_km.toFixed(1):'\u2013'}</td></tr>`
      : `<tr><td>${s.name}</td><td colspan="5" style="color:var(--hot)">${s.error||'no pick'}</td></tr>`
    ).join('');

  if(!map){ map = L.map('map'); L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    {attribution:'&copy; OpenStreetMap'}).addTo(map); }
  if(layer) layer.remove();
  layer = L.layerGroup().addTo(map);
  const pts = [];
  for(const s of d.stations){
    pts.push([s.lat, s.lon]);
    L.circleMarker([s.lat, s.lon], {radius:6, color:'#0b6e75', fillColor:'#0b6e75', fillOpacity:.9})
      .bindTooltip(s.name, {permanent:true, direction:'right'}).addTo(layer);
  }

  if(d.located){
    const e = d.epicenter;
    document.getElementById('big').innerHTML =
      `<div><span>Epicenter latitude</span><b>${e.lat.toFixed(4)}\u00b0 N</b></div>
       <div><span>Epicenter longitude</span><b>${e.lon.toFixed(4)}\u00b0 E</b></div>
       <div><span>Origin time (UTC)</span><b>${d.origin_time.slice(11,23)}</b></div>
       <div><span>RMS misfit</span><b>${d.rms_km.toFixed(1)} km</b></div>`;
    pts.push([e.lat, e.lon]);
    L.circleMarker([e.lat, e.lon], {radius:9, color:'#d1432b', fillColor:'#d1432b', fillOpacity:.85})
      .bindTooltip('Epicenter', {permanent:true, direction:'top'}).addTo(layer);
    document.getElementById('circ').src = 'data:image/png;base64,' + d.circle_map_png;
  } else {
    document.getElementById('big').innerHTML = `<div><span>Status</span><b style="color:var(--hot)">Not located</b></div>`;
    document.getElementById('circ').removeAttribute('src');
  }
  if(pts.length) { map.fitBounds(pts, {padding:[40,40]}); setTimeout(()=>map.invalidateSize(),50); }

  const box = document.getElementById('stacards'); box.innerHTML = '';
  for(const s of d.stations){
    const q = s.quality || 'REJECT';
    let html = `<div class="stacard">
      <h3><span>${s.name}</span><span class="badge ${q}">${q}</span></h3>`;
    if(s.error){
      html += `<div class="err">${s.error}</div>`;
    }
    if(s.quality_notes && s.quality_notes.length){
      html += `<div class="notes"><b>Notes:</b> ${s.quality_notes.join(', ')}</div>`;
    }
    if(s.waveform_png){
      html += `<img class="plot" src="data:image/png;base64,${s.waveform_png}">`;
    }
    if(s.spectrogram_png){
      html += `<img class="plot" src="data:image/png;base64,${s.spectrogram_png}">`;
    }
    html += `</div>`;
    box.innerHTML += html;
  }
}
</script></body></html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)