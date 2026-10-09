from flask import Flask, jsonify, render_template_string, request
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import earthquake as core_sta
import triangulate as core_tdoa

app = Flask(__name__)

DEFAULT_BASE_URL = "https://seiscomp.alertnepal.online"

STA_STATIONS = [
    {"sta": "EQM13", "lat": 28.256323, "lon": 85.367569, "net": "NP", "cha": "EHZ", "loc": "*"},
    {"sta": "EQM08", "lat": 27.831,    "lon": 86.65,     "net": "NP", "cha": "EHZ", "loc": "*"},
    {"sta": "EQM10", "lat": 28.299517, "lon": 83.960148, "net": "NP", "cha": "EHZ", "loc": "*"},
]

TDOA_STATIONS = [
    {"sta": "KKN",   "net": "NK", "loc": "*", "cha": "BHZ", "lat": 27.8000,   "lon": 85.2790},
    {"sta": "EVN",   "net": "IO", "loc": "*", "cha": "BHZ", "lat": 27.95865,  "lon": 86.811653},
    {"sta": "EQM10", "net": "NP", "loc": "*", "cha": "EHZ", "lat": 28.299517, "lon": 83.960148},
]

_coord_cache = {}


def _query(server, net, sta):
    inv = Client(server).get_stations(network=net, station=sta, level="station")
    for network in inv:
        for station in network:
            return (station.latitude, station.longitude, station.elevation)
    return None


def lookup_station(base_url, net, sta):
    """Return (lat, lon, elevation) for a station. Tries base_url, then fallbacks."""
    key = (base_url, net, sta)
    if key in _coord_cache:
        return _coord_cache[key]
    found = None
    for server in [base_url]:
        try:
            found = _query(server, net, sta)
        except Exception:
            found = None  # FDSNNoDataException, connection error, etc.
        if found:
            break
    if not found:
        raise ValueError(f"Station {net}.{sta} not found on {base_url}")
    _coord_cache[key] = found
    return found


def fill_coords(stations, base_url):
    for s in stations:
        if s.get("lat") in (None, "") or s.get("lon") in (None, ""):
            s["lat"], s["lon"], _ = lookup_station(base_url, s["net"], s["sta"])


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
        return jsonify(error=f"No coordinates found for {net}.{sta} ({type(e).__name__})"), 404


@app.route("/api/run", methods=["POST"])
def run_stalta():
    try:
        p = request.get_json()
        base_url = p.get("base_url") or DEFAULT_BASE_URL
        fill_coords(p["stations"], base_url)

        result = core_sta.run_location(
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


@app.route("/api/run_tdoa", methods=["POST"])
def run_tdoa():
    try:
        p = request.get_json()
        base_url = p.get("base_url") or DEFAULT_BASE_URL
        fill_coords(p["stations"], base_url)

        v_min = float(p["v_min"])
        v_max = float(p["v_max"])
        if v_min <= 0 or v_min >= v_max:
            raise ValueError("Min velocity must be positive and smaller than max velocity.")

        result = core_tdoa.run_triangulation(
            stations=p["stations"],
            start=UTCDateTime(p["start"]),
            end=UTCDateTime(p["end"]),
            fs=float(p["fs"]),
            fmin=float(p["fmin"]),
            fmax=float(p["fmax"]),
            smooth_cutoff=float(p["smooth_cutoff"]),
            max_shift_seconds=float(p["max_shift"]),
            v_min=v_min,
            v_max=v_max,
        )
        return jsonify(**result)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


@app.route("/")
def index():
    return render_template_string(
        PAGE, sta_stations=STA_STATIONS, tdoa_stations=TDOA_STATIONS, base_url=DEFAULT_BASE_URL
    )


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Earthquake &amp; tremor locator</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#eef1f3;--panel:#fff;--ink:#16232b;--mute:#5d6c75;--line:#cdd6db;--accent:#0b6e75;--hot:#d1432b;--good:#1f9e6b;--review:#d97706}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Segoe UI",system-ui,sans-serif}
header{padding:22px 28px 6px}
h1{margin:0;font-size:24px;font-weight:650;letter-spacing:-.01em}
header p{margin:4px 0 0;color:var(--mute);max-width:80ch}
.tabs{display:flex;gap:8px;padding:12px 28px 0}
.tabs button{background:#fff;color:var(--ink);border:1px solid var(--line);padding:9px 18px;font-weight:600}
.tabs button.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.view{display:none}
.view.active{display:grid}
main{grid-template-columns:340px 1fr;gap:20px;padding:18px 28px 40px}
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
.status{margin-top:10px;font-size:13px;color:var(--mute)}
.status.err{color:var(--hot)}
.result{display:none}
.big{display:flex;gap:26px;flex-wrap:wrap;margin-bottom:12px}
.big div span{display:block;font-size:12px;color:var(--mute)}
.big div b{font-size:22px;font-weight:650}
.map{height:330px;border-radius:6px;border:1px solid var(--line);margin-bottom:14px}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin:10px 0}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{color:var(--mute);font-weight:500}
.stack{display:grid;gap:16px;align-content:start}
img.plot{max-width:100%;border-radius:6px;border:1px solid var(--line);display:block;margin-top:8px}
.stacard{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px}
.stacard h3{margin:0 0 8px;font-size:15px;display:flex;justify-content:space-between;align-items:center}
.badge{padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;color:#fff}
.badge.GOOD{background:var(--good)}
.badge.REVIEW{background:var(--review)}
.badge.REJECT{background:var(--hot)}
.stacard .err{color:var(--hot);font-size:13px;margin:4px 0}
.notes{font-size:12px;color:var(--mute);margin-top:6px}
.plot-section{margin-top:18px}
.cc{height:200px}
</style></head><body>
<header><h1>Earthquake &amp; tremor locator</h1>
<p>Two independent methods: STA/LTA picks with S&ndash;P trilateration for earthquakes, and envelope cross-correlation (TDOA) for tremor.</p></header>

<div class="tabs">
  <button id="tabbtn_a" class="active" onclick="showTab('a')">STA/LTA method</button>
  <button id="tabbtn_t" onclick="showTab('t')">TDOA method</button>
</div>

<!--  STA/LTA  -->
<main class="view active" id="view_a">
<div class="panel">
  <h2>Data source</h2>
  <label>FDSN base URL</label><input type="text" id="a_base_url" value="{{ base_url }}">
  <h2 style="margin-top:14px">Stations</h2>
  <div id="stations_a"></div>
  <button class="add" onclick="addStation()">+ Add station</button>

  <label style="margin-top:14px">Start (UTC)</label><input type="text" id="a_start" value="2026-09-22T07:45:00">
  <label>End (UTC)</label><input type="text" id="a_end" value="2026-09-22T07:55:00">
  <div class="row"><div><label>Band low (Hz)</label><input type="number" id="a_fmin" value="1.0" step="0.1"></div>
  <div><label>Band high (Hz)</label><input type="number" id="a_fmax" value="8" step="0.1"></div></div>
  <div class="row"><div><label>Sample rate (Hz)</label><input type="number" id="a_fs" value="20" step="1"></div>
  <div><label>Wait (s)</label><input type="number" id="a_wait_sec" value="30" step="5"></div></div>
  <div class="row"><div><label>Vp (km/s)</label><input type="number" id="a_vp" value="6.0" step="0.1"></div>
  <div><label>Vs (km/s)</label><input type="number" id="a_vs" value="3.5" step="0.1"></div></div>

  <details>
    <summary>P-pick STA/LTA settings</summary>
    <div class="row"><div><label>STA (s)</label><input type="number" id="a_sta_p" value="1.0" step="0.1"></div>
    <div><label>LTA (s)</label><input type="number" id="a_lta_p" value="10.0" step="0.5"></div></div>
    <div class="row"><div><label>Trigger on</label><input type="number" id="a_thr_on" value="3.5" step="0.1"></div>
    <div><label>Trigger off</label><input type="number" id="a_thr_off" value="1.5" step="0.1"></div></div>
  </details>
  <details>
    <summary>S-pick settings</summary>
    <div class="row"><div><label>STA (s)</label><input type="number" id="a_sta_s" value="1.0" step="0.1"></div>
    <div><label>LTA (s)</label><input type="number" id="a_lta_s" value="8.0" step="0.5"></div></div>
    <div class="row"><div><label>Min S&ndash;P (s)</label><input type="number" id="a_min_sp_sec" value="4.0" step="0.5"></div>
    <div><label>Max S&ndash;P (s)</label><input type="number" id="a_max_sp_sec" value="50.0" step="1"></div></div>
    <div class="row"><div><label>Trigger on</label><input type="number" id="a_thr_on_s" value="2.5" step="0.1"></div>
    <div><label>Trigger off</label><input type="number" id="a_thr_off_s" value="1.0" step="0.1"></div></div>
  </details>

  <button class="go" id="go_a" onclick="runSta()">Locate event</button>
  <div class="status" id="status_a"></div>
</div>
<div class="stack">
  <div class="panel result" id="res_a">
    <div class="big" id="big_a"></div>
    <div class="map" id="map_a"></div>
    <table id="tbl_a"></table>
    <h2>Circle intersection (trilateration)</h2>
    <img class="plot" id="circ_a">
  </div>
  <div class="stack" id="stacards_a"></div>
</div>
</main>

<!-- TDOA -->
<main class="view" id="view_t">
<div class="panel">
  <h2>Data source</h2>
  <label>FDSN base URL (for coordinate lookup)</label><input type="text" id="t_base_url" value="{{ base_url }}">
  <h2 style="margin-top:14px">Stations (exactly 3)</h2>
  <div id="stations_t"></div>

  <label>Start (UTC)</label><input type="text" id="t_start" value="2026-08-26T02:52:00">
  <label>End (UTC)</label><input type="text" id="t_end" value="2026-08-26T02:56:00">
  <div class="row"><div><label>Band low (Hz)</label><input type="number" id="t_fmin" value="1.0" step="0.1"></div>
  <div><label>Band high (Hz)</label><input type="number" id="t_fmax" value="8" step="0.1"></div></div>
  <div class="row"><div><label>Sample rate (Hz)</label><input type="number" id="t_fs" value="20" step="1"></div>
  <div><label>Envelope smooth cutoff (Hz)</label><input type="number" id="t_smooth_cutoff" value="0.5" step="0.1"></div></div>

  <div class="row"><div><label>Max lag (s)</label><input type="number" id="t_max_shift" value="30"></div>
  <div></div></div>
  <div class="row"><div><label>Min velocity (km/s)</label><input type="number" id="t_v_min" value="1.0" step="0.1"></div>
  <div><label>Max velocity (km/s)</label><input type="number" id="t_v_max" value="3.5" step="0.1"></div></div>

  <button class="go" id="go_t" onclick="runTdoa()">Triangulate</button>
  <div class="status" id="status_t"></div>
</div>
<div class="stack">
  <div class="panel result" id="res_t">
    <div class="big" id="big_t"></div>
    <div class="map" id="map_t"></div>
    <table id="tbl_t"></table>
    <div class="plot-section"><h2>Normal waveforms</h2><img id="wf_t" class="plot" alt="Waveforms"></div>
    <div class="plot-section"><h2>Frequency spectrogram</h2><img id="sp_t" class="plot" alt="Spectrogram"></div>
    <div class="plot-section"><h2>Hyperbola intersection</h2><img id="hyp_t" class="plot" alt="TDOA hyperbolas"></div>
    <div class="plot-section"><h2>Misfit vs. velocity</h2><img id="vel_t" class="plot" alt="Velocity misfit"></div>
  </div>
  <div class="panel result" id="ccs_t">
    <h2>Envelope cross-correlation</h2>
    <div id="ccplots_t"></div>
  </div>
</div>
</main>

<script>
const S = { a: {{ sta_stations|tojson }}, t: {{ tdoa_stations|tojson }} };
const maps = {a:null, t:null}, layers = {a:null, t:null};
const v = id => document.getElementById(id).value;

function setStatus(k, t, err){
  const s = document.getElementById('status_'+k); s.textContent = t; s.className = 'status' + (err?' err':'');
}

function showTab(k){
  for(const x of ['a','t']){
    document.getElementById('view_'+x).classList.toggle('active', x===k);
    document.getElementById('tabbtn_'+x).classList.toggle('active', x===k);
  }
  if(maps[k]) setTimeout(()=>maps[k].invalidateSize(), 50);
}

function setF(k,i,f,val){ S[k][i][f] = val; }
function setNum(k,i,f,val){ const n = parseFloat(val); S[k][i][f] = Number.isFinite(n) ? n : null; }

function drawStations(k){
  const removable = k === 'a';
  document.getElementById('stations_'+k).innerHTML = S[k].map((s,i) => `
    <div class="sta">
      ${removable ? `<button class="rm" onclick="removeStation(${i})" title="Remove">\u00d7</button>` : ''}
      <div class="name">Station ${i+1}${k==='t' && i===0 ? ' \u2014 origin' : ''}</div>
      <div class="row3">
        <input type="text" value="${s.sta}" onchange="setF('${k}',${i},'sta',this.value.trim().toUpperCase());lookup('${k}',${i})" placeholder="STA">
        <input type="text" value="${s.net}" onchange="setF('${k}',${i},'net',this.value.trim().toUpperCase());lookup('${k}',${i})" placeholder="NET">
        <input type="text" value="${s.loc}" onchange="setF('${k}',${i},'loc',this.value)" placeholder="LOC">
      </div>
      <div class="row3" style="margin-top:6px">
        <input type="text" value="${s.cha}" onchange="setF('${k}',${i},'cha',this.value)" placeholder="CHA">
        <input type="number" step="0.0001" value="${s.lat ?? ''}" onchange="setNum('${k}',${i},'lat',this.value)" placeholder="lat (auto)">
        <input type="number" step="0.0001" value="${s.lon ?? ''}" onchange="setNum('${k}',${i},'lon',this.value)" placeholder="lon (auto)">
      </div>
    </div>`).join('');
}

async function lookup(k,i){
  const s = S[k][i];
  if(!s.sta || !s.net) return;
  setStatus(k, `Looking up ${s.net}.${s.sta}\u2026`);
  try{
    const q = new URLSearchParams({net:s.net, sta:s.sta, base_url:v(k+'_base_url')});
    const r = await fetch('/api/station?' + q);
    const d = await r.json();
    if(!r.ok) throw new Error(d.error);
    s.lat = d.lat; s.lon = d.lon;
    drawStations(k);
    setStatus(k, `${s.net}.${s.sta}: ${d.lat.toFixed(4)}, ${d.lon.toFixed(4)}`);
  }catch(e){
    s.lat = null; s.lon = null;
    drawStations(k);
    setStatus(k, e.message, true);
  }
}

function addStation(){
  S.a.push({sta:"", net:"", loc:"*", cha:"EHZ", lat:null, lon:null});
  drawStations('a');
}
function removeStation(i){
  if(S.a.length<=3){ setStatus('a','At least 3 stations are needed.', true); return; }
  S.a.splice(i,1); drawStations('a');
}

function getMap(k){
  if(!maps[k]){
    maps[k] = L.map('map_'+k);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {attribution:'&copy; OpenStreetMap'}).addTo(maps[k]);
  }
  if(layers[k]) layers[k].remove();
  layers[k] = L.layerGroup().addTo(maps[k]);
  return [maps[k], layers[k]];
}

function checkCoords(k){
  const bad = S[k].findIndex(s => !s.sta || !s.net || !Number.isFinite(s.lat) || !Number.isFinite(s.lon));
  if(bad >= 0){ setStatus(k, `Station ${bad+1} is missing a code or coordinates.`, true); return false; }
  return true;
}

/* STA/LTA */
async function runSta(){
  const btn = document.getElementById('go_a'); btn.disabled = true;
  if(!checkCoords('a')){ btn.disabled = false; return; }
  setStatus('a','Downloading waveforms and processing. This can take a minute.');
  try{
    const r = await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        base_url:v('a_base_url'), stations:S.a, start:v('a_start'), end:v('a_end'),
        fmin:v('a_fmin'), fmax:v('a_fmax'), fs:v('a_fs'), wait_sec:v('a_wait_sec'),
        vp:v('a_vp'), vs:v('a_vs'),
        sta_p:v('a_sta_p'), lta_p:v('a_lta_p'), thr_on:v('a_thr_on'), thr_off:v('a_thr_off'),
        sta_s:v('a_sta_s'), lta_s:v('a_lta_s'), min_sp_sec:v('a_min_sp_sec'), max_sp_sec:v('a_max_sp_sec'),
        thr_on_s:v('a_thr_on_s'), thr_off_s:v('a_thr_off_s')
      })});
    const d = await r.json();
    if(!r.ok) throw new Error(d.error);
    showSta(d);
    setStatus('a', d.located ? 'Done.' : (d.error || 'Could not locate the event.'), !d.located);
  }catch(e){ setStatus('a', e.message, true); }
  btn.disabled = false;
}

function showSta(d){
  document.getElementById('res_a').style.display = 'block';

  // Safe Table Headers and Rows
  document.getElementById('tbl_a').innerHTML =
    '<tr>' +
      '<th>Station</th>' +
      '<th>P time (UTC)</th>' +
      '<th>S time (UTC)</th>' +
      '<th>S–P (s)</th>' +
      '<th>Epi. Dist (km)</th>' +
      '<th>Hypo. Dist (km)</th>' +
      '<th>Local Mag (M<sub>L</sub>)</th>' +
      '<th>Residual (km)</th>' +
    '</tr>' +
    d.stations.map(s => s.p_time
      ? `<tr>
          <td>${s.name}</td>
          <td>${s.p_time.slice(11,23)}</td>
          <td>${s.s_time.slice(11,23)}</td>
          <td>${Number.isFinite(s.sp_diff) ? s.sp_diff.toFixed(2) : '–'}</td>
          <td>${Number.isFinite(s.epicentral_distance_km) ? s.epicentral_distance_km.toFixed(1) : '–'}</td>
          <td>${Number.isFinite(s.hypo_distance_km) ? s.hypo_distance_km.toFixed(1) : '–'}</td>
          <td><b>${Number.isFinite(s.ml) ? s.ml.toFixed(2) : '–'}</b></td>
          <td${Math.abs(s.residual_km||0)>15?' style="color:var(--hot)"':''}>${Number.isFinite(s.residual_km) ? s.residual_km.toFixed(1) : '–'}</td>
         </tr>`
      : `<tr><td>${s.name}</td><td colspan="7" style="color:var(--hot)">${s.error||'no pick'}</td></tr>`
    ).join('');

  const [map, layer] = getMap('a');
  const pts = [];
  for(const s of d.stations){
    if(Number.isFinite(s.lat) && Number.isFinite(s.lon)) {
      pts.push([s.lat, s.lon]);
      L.circleMarker([s.lat, s.lon], {radius:6, color:'#0b6e75', fillColor:'#0b6e75', fillOpacity:.9})
        .bindTooltip(s.name, {permanent:true, direction:'right'}).addTo(layer);
    }
  }

  //  Safe Summary Banner with Type Guards
  if(d.located && d.epicenter){
    const e = d.epicenter;
    const magStr = Number.isFinite(d.magnitude_ml) ? d.magnitude_ml.toFixed(2) : 'N/A';
    const rmsStr = Number.isFinite(d.rms_km) ? d.rms_km.toFixed(1) + ' km' : 'N/A';
    const timeStr = d.origin_time ? d.origin_time.slice(11,23) : 'N/A';

    document.getElementById('big_a').innerHTML =
      `<div><span>Epicenter Latitude</span><b>${e.lat.toFixed(4)}° N</b></div>
       <div><span>Epicenter Longitude</span><b>${e.lon.toFixed(4)}° E</b></div>
       <div><span>Magnitude (M<sub>L</sub>)</span><b style="color:var(--accent)">${magStr}</b></div>
       <div><span>Origin Time (UTC)</span><b>${timeStr}</b></div>
       <div><span>RMS Misfit</span><b>${rmsStr}</b></div>
       <div><span>Stations Used</span><b>${d.n_stations_used || 0}</b></div>`;
    
    pts.push([e.lat, e.lon]);
    L.circleMarker([e.lat, e.lon], {radius:9, color:'#d1432b', fillColor:'#d1432b', fillOpacity:.85})
      .bindTooltip('Epicenter', {permanent:true, direction:'top'}).addTo(layer);
    if(d.circle_map_png) {
      document.getElementById('circ_a').src = 'data:image/png;base64,' + d.circle_map_png;
    }
  } else {
    document.getElementById('big_a').innerHTML = `<div><span>Status</span><b style="color:var(--hot)">${d.error || 'Not located'}</b></div>`;
    document.getElementById('circ_a').removeAttribute('src');
  }

  if(pts.length){ map.fitBounds(pts, {padding:[40,40]}); setTimeout(()=>map.invalidateSize(),50); }

  // Safe Station Card Rendering
  const box = document.getElementById('stacards_a'); box.innerHTML = '';
  for(const s of d.stations){
    const q = s.quality || 'REJECT';
    const mlLabel = Number.isFinite(s.ml) ? `(M<sub>L</sub> ${s.ml.toFixed(2)})` : '';
    let html = `<div class="stacard"><h3><span>${s.name} ${mlLabel}</span><span class="badge ${q}">${q}</span></h3>`;
    if(s.error) html += `<div class="err">${s.error}</div>`;
    if(s.quality_notes && s.quality_notes.length) html += `<div class="notes"><b>Notes:</b> ${s.quality_notes.join(', ')}</div>`;
    if(s.waveform_png) html += `<img class="plot" src="data:image/png;base64,${s.waveform_png}">`;
    if(s.spectrogram_png) html += `<img class="plot" src="data:image/png;base64,${s.spectrogram_png}">`;
    html += `</div>`;
    box.innerHTML += html;
  }
}

/* TDOA */
async function runTdoa(){
  const btn = document.getElementById('go_t'); btn.disabled = true;
  if(!checkCoords('t')){ btn.disabled = false; return; }
  setStatus('t','Downloading waveforms and processing. This can take a minute.');
  try{
    const r = await fetch('/api/run_tdoa',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        base_url:v('t_base_url'), stations:S.t, start:v('t_start'), end:v('t_end'),
        fmin:v('t_fmin'), fmax:v('t_fmax'), fs:v('t_fs'),
        smooth_cutoff:v('t_smooth_cutoff'), max_shift:v('t_max_shift'),
        v_min:v('t_v_min'), v_max:v('t_v_max')
      })});
    const d = await r.json();
    if(!r.ok) throw new Error(d.error || 'Analysis failed.');
    showTdoa(d);
    setStatus('t','Analysis completed successfully.');
  }catch(e){ setStatus('t', e.message, true); }
  btn.disabled = false;
}

function showTdoa(d){
  const s = d.source;
  document.getElementById('res_t').style.display = 'block';
  document.getElementById('ccs_t').style.display = 'block';

  document.getElementById('big_t').innerHTML = `
    <div><span>Estimated latitude</span><b>${s.lat.toFixed(4)}&deg; N</b></div>
    <div><span>Estimated longitude</span><b>${s.lon.toFixed(4)}&deg; E</b></div>
    <div><span>Velocity</span><b>${s.velocity.toFixed(2)} km/s</b></div>
    <div><span>TDOA closure error</span><b>${d.closure_error.toFixed(3)} s</b></div>`;

  const [map, layer] = getMap('t');
  const pts = [[s.lat, s.lon]];
  L.circleMarker([s.lat, s.lon], {radius:9, color:'#d1432b', fillColor:'#d1432b', fillOpacity:.8})
    .bindTooltip('Estimated source', {permanent:true, direction:'top'}).addTo(layer);
  for(const [n,c] of Object.entries(d.stations)){
    pts.push([c.lat, c.lon]);
    L.circleMarker([c.lat, c.lon], {radius:6, color:'#0b6e75', fillColor:'#0b6e75', fillOpacity:.9})
      .bindTooltip(n, {permanent:true, direction:'right'}).addTo(layer);
    L.polyline([[c.lat,c.lon],[s.lat,s.lon]], {color:'#0b6e75', weight:1, dashArray:'4'}).addTo(layer);
  }
  map.fitBounds(pts, {padding:[40,40]});
  setTimeout(()=>map.invalidateSize(), 50);

  document.getElementById('tbl_t').innerHTML =
    '<tr><th>Pair</th><th>Lag (s)</th><th>Peak CC</th></tr>' +
    d.pairs.map(p => `<tr><td>${p.a} &ndash; ${p.b}</td><td>${p.lag.toFixed(3)}</td><td>${p.cc_max.toFixed(2)}</td></tr>`).join('') +
    `<tr><td colspan="3" style="color:var(--mute)">Residual norm of the least-squares fit: ${s.residual_norm.toFixed(4)} km</td></tr>`;

  document.getElementById('wf_t').src = 'data:image/png;base64,' + d.waveform_plot;
  document.getElementById('sp_t').src = 'data:image/png;base64,' + d.spectrogram_plot;
  document.getElementById('hyp_t').src = 'data:image/png;base64,' + d.plot_png;
  document.getElementById('vel_t').src = 'data:image/png;base64,' + d.velocity_plot;

  const box = document.getElementById('ccplots_t'); box.innerHTML = '';
  d.pairs.forEach((p,i) => {
    const div = document.createElement('div'); div.className = 'cc'; box.appendChild(div);
    Plotly.newPlot(div, [{x:p.lags_full, y:p.cc_full, mode:'lines', line:{color:'#16232b', width:1.5}, hoverinfo:'x+y'}], {
      title:{text:`${p.a} \u2013 ${p.b}`, font:{size:13}},
      margin:{l:50, r:20, t:32, b:36},
      xaxis:{title: i === d.pairs.length - 1 ? 'Lag (s)' : ''},
      yaxis:{title:'Normalized CC'},
      shapes:[{type:'line', x0:p.lag, x1:p.lag, y0:0, y1:1, yref:'paper', line:{color:'#d1432b', dash:'dash'}}],
      annotations:[{x:p.lag, y:1, yref:'paper', text:`${p.lag.toFixed(2)} s, CC ${p.cc_max.toFixed(2)}`,
                    showarrow:false, xanchor:'left', font:{color:'#d1432b', size:12}}]
    }, {displayModeBar:false, responsive:true});
  });
}

drawStations('a');
drawStations('t');
</script></body></html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)