from flask import Flask, request, jsonify, render_template_string
from obspy import UTCDateTime
import triangulate as core

app = Flask(__name__)

DEFAULT_STATIONS = [
    {"sta": "KKN","net": "NK","loc": "*","cha": "BHZ","lat": 27.8000,"lon": 85.2790},
    {"sta": "EVN","net": "IO","loc": "*","cha": "BHZ","lat": 27.95865,"lon": 86.811653},
    {"sta": "EQM10","net": "NP","loc": "*","cha": "EHZ","lat": 28.299517,"lon": 83.960148},
]

@app.route("/api/run", methods=["POST"])
def run():
    try:
        p = request.get_json()

        result = core.run_triangulation(
            stations=p["stations"],
            start=UTCDateTime(p["start"]),
            end=UTCDateTime(p["end"]),
            fs=float(p["fs"]),
            fmin=float(p["fmin"]),
            fmax=float(p["fmax"]),
            smooth_cutoff=float(p["smooth_cutoff"]),
            max_shift_seconds=float(p["max_shift"]),
            velocity=float(p["velocity"]),
        )

        return jsonify(**result)

    except ValueError as e:
        return jsonify(error=str(e)), 400

    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


@app.route("/")
def index():
    return render_template_string(PAGE, stations=DEFAULT_STATIONS)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TDOA Tremor Triangulation</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

<style>
:root{
    --bg:#eef1f3;
    --panel:#fff;
    --ink:#16232b;
    --mute:#5d6c75;
    --line:#cdd6db;
    --accent:#0b6e75;
    --hot:#d1432b;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Segoe UI",system-ui,sans-serif}
header{padding:22px 28px 6px}
h1{margin:0;font-size:24px;font-weight:650;letter-spacing:-.01em}
header p{margin:4px 0 0;color:var(--mute);max-width:80ch}
main{display:grid;grid-template-columns:330px 1fr;gap:20px;padding:18px 28px 40px}
@media(max-width:940px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px}
h2{font-size:15px;margin:0 0 10px}
label{display:block;font-size:13px;color:var(--mute);margin:10px 0 3px}
input[type=text],input[type=number]{width:100%;padding:7px 9px;border:1px solid var(--line);border-radius:5px;font:inherit;color:var(--ink);background:#fff}
input:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.sta{border:1px solid var(--line);border-radius:6px;padding:8px 10px;margin-bottom:8px}
.sta .name{font-weight:600;font-size:13.5px;margin-bottom:6px}
.sta .row3 input{font-size:12.5px;padding:5px 6px}
button{font:inherit;border:0;border-radius:5px;padding:8px 12px;cursor:pointer;background:var(--ink);color:#fff}
button.go{width:100%;margin-top:14px;background:var(--accent);padding:11px;font-weight:600}
button:disabled{opacity:.55;cursor:wait}
#status{margin-top:10px;font-size:13px;color:var(--mute)}
#status.err{color:var(--hot)}
.result{display:none}
.big{display:flex;gap:26px;flex-wrap:wrap;margin-bottom:12px}
.big div span{display:block;font-size:12px;color:var(--mute)}
.big div b{font-size:22px;font-weight:650}
#map{height:340px;border-radius:6px;border:1px solid var(--line);margin-bottom:14px}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin:10px 0}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{color:var(--mute);font-weight:500}
.stack{display:grid;gap:16px}
.plot-section{margin-top:18px}
.plot-section h2{margin-bottom:10px}
.analysis-plot{width:100%;display:block;border:1px solid var(--line);border-radius:6px;background:#fff}
.cc{height:200px}
#hyp{max-width:100%;display:block;border-radius:6px;border:1px solid var(--line)}
</style>
</head>

<body>
<header>
  <h1>TDOA Tremor Triangulation</h1>
</header>

<main>
<div class="panel">
  <h2>Stations (exactly 3)</h2>
  <div id="stations"></div>

  <label>Start (UTC)</label>
  <input type="text" id="start" value="2026-08-26T02:52:00">

  <label>End (UTC)</label>
  <input type="text" id="end" value="2026-08-26T02:56:00">

  <div class="row">
    <div>
      <label>Band low (Hz)</label>
      <input type="number" id="fmin" value="1.0" step="0.1">
    </div>
    <div>
      <label>Band high (Hz)</label>
      <input type="number" id="fmax" value="8" step="0.1">
    </div>
  </div>

  <div class="row">
    <div>
      <label>Sample rate (Hz)</label>
      <input type="number" id="fs" value="20" step="1">
    </div>
    <div>
      <label>Envelope smooth cutoff (Hz)</label>
      <input type="number" id="smooth_cutoff" value="0.5" step="0.1">
    </div>
  </div>

  <div class="row">
    <div>
      <label>Max lag (s)</label>
      <input type="number" id="max_shift" value="30">
    </div>
    <div>
      <label>Velocity (km/s)</label>
      <input type="number" id="velocity" value="2.4" step="0.1">
    </div>
  </div>

  <button class="go" id="go" onclick="run()">Triangulate</button>
  <div id="status"></div>
</div>

<div class="stack">
  <div class="panel result" id="res">
    <div class="big" id="big"></div>
    <div id="map"></div>
    <table id="tbl"></table>

    <div class="plot-section">
      <h2>Normal waveforms</h2>
      <img id="waveformPlot" class="analysis-plot" alt="Three-station seismic waveforms">
    </div>

    <div class="plot-section">
      <h2>Frequency spectrogram</h2>
      <img id="spectrogramPlot" class="analysis-plot" alt="Three-station frequency spectrogram">
    </div>

    <div class="plot-section">
      <h2>Hyperbola intersection</h2>
      <img id="hyp" alt="TDOA hyperbola intersection">
    </div>
  </div>

  <div class="panel result" id="ccs">
    <h2>Envelope cross-correlation</h2>
    <div id="ccplots"></div>
  </div>
</div>
</main>

<script>
let stations = {{ stations|tojson }};
let map, layer;

function drawStations(){
    document.getElementById('stations').innerHTML = stations.map((s,i) => `
        <div class="sta">
            <div class="name">Station ${i+1}${i===0?' &mdash; origin':''}</div>
            <div class="row3">
                <input type="text" value="${s.sta}" onchange="stations[${i}].sta=this.value" placeholder="STA">
                <input type="text" value="${s.net}" onchange="stations[${i}].net=this.value" placeholder="NET">
                <input type="text" value="${s.loc}" onchange="stations[${i}].loc=this.value" placeholder="LOC">
            </div>
            <div class="row3" style="margin-top:6px">
                <input type="text" value="${s.cha}" onchange="stations[${i}].cha=this.value" placeholder="CHA">
                <input type="number" step="0.0001" value="${s.lat}" onchange="stations[${i}].lat=parseFloat(this.value)" placeholder="lat">
                <input type="number" step="0.0001" value="${s.lon}" onchange="stations[${i}].lon=parseFloat(this.value)" placeholder="lon">
            </div>
        </div>
    `).join('');
}

function setStatus(t, err){
    const s = document.getElementById('status');
    s.textContent = t;
    s.className = err ? 'err' : '';
}

const v = id => document.getElementById(id).value;

async function run(){
    const btn = document.getElementById('go');
    btn.disabled = true;
    setStatus('Downloading waveforms and processing. This can take a minute.');

    try{
        const r = await fetch('/api/run', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({
                stations,
                start:v('start'),
                end:v('end'),
                fmin:v('fmin'),
                fmax:v('fmax'),
                fs:v('fs'),
                smooth_cutoff:v('smooth_cutoff'),
                max_shift:v('max_shift'),
                velocity:v('velocity')
            })
        });

        const d = await r.json();
        if(!r.ok){
            throw new Error(d.error || 'Analysis failed.');
        }

        show(d);
        setStatus('Analysis completed successfully.');
    }catch(e){
        setStatus(e.message, true);
    }
    btn.disabled = false;
}

function show(d){
    const s = d.source;

    document.getElementById('res').style.display = 'block';
    document.getElementById('ccs').style.display = 'block';

    document.getElementById('big').innerHTML = `
        <div><span>Estimated latitude</span><b>${s.lat.toFixed(4)}&deg; N</b></div>
        <div><span>Estimated longitude</span><b>${s.lon.toFixed(4)}&deg; E</b></div>
        <div><span>Velocity</span><b>${s.velocity.toFixed(2)} km/s</b></div>
        <div><span>TDOA closure error</span><b>${d.closure_error.toFixed(3)} s</b></div>
    `;

    if(!map){
        map = L.map('map');
        L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution:'&copy; OpenStreetMap'
        }).addTo(map);
    }

    if(layer){ layer.remove(); }
    layer = L.layerGroup().addTo(map);

    const pts = [[s.lat, s.lon]];

    L.circleMarker([s.lat, s.lon], {
        radius:9, color:'#d1432b', fillColor:'#d1432b', fillOpacity:.8
    }).bindTooltip('Estimated source', {permanent:true, direction:'top'}).addTo(layer);

    for(const [n,c] of Object.entries(d.stations)){
        pts.push([c.lat, c.lon]);
        L.circleMarker([c.lat, c.lon], {
            radius:6, color:'#0b6e75', fillColor:'#0b6e75', fillOpacity:.9
        }).bindTooltip(n, {permanent:true, direction:'right'}).addTo(layer);

        L.polyline([[c.lat,c.lon], [s.lat,s.lon]], {
            color:'#0b6e75', weight:1, dashArray:'4'
        }).addTo(layer);
    }

    map.fitBounds(pts, {padding:[40,40]});
    setTimeout(() => map.invalidateSize(), 50);

    document.getElementById('tbl').innerHTML =
        '<tr><th>Pair</th><th>Lag (s)</th><th>Peak CC</th></tr>' +
        d.pairs.map(p => `
            <tr>
                <td>${p.a} &ndash; ${p.b}</td>
                <td>${p.lag.toFixed(3)}</td>
                <td>${p.cc_max.toFixed(2)}</td>
            </tr>
        `).join('') +
        `<tr><td colspan="3" style="color:var(--mute)">Residual norm of the least-squares fit: ${s.residual_norm.toFixed(4)} km</td></tr>`;

    document.getElementById('waveformPlot').src = 'data:image/png;base64,' + d.waveform_plot;
    document.getElementById('spectrogramPlot').src = 'data:image/png;base64,' + d.spectrogram_plot;
    document.getElementById('hyp').src = 'data:image/png;base64,' + d.plot_png;

    const box = document.getElementById('ccplots');
    box.innerHTML = '';

    d.pairs.forEach((p,i) => {
        const div = document.createElement('div');
        div.className = 'cc';
        box.appendChild(div);

        Plotly.newPlot(div, [
            {
                x:p.lags_full,
                y:p.cc_full,
                mode:'lines',
                line:{color:'#16232b', width:1.5},
                hoverinfo:'x+y'
            }
        ], {
            title:{text:`${p.a} \u2013 ${p.b}`, font:{size:13}},
            margin:{l:50, r:20, t:32, b:36},
            xaxis:{title: i === d.pairs.length - 1 ? 'Lag (s)' : ''},
            yaxis:{title:'Normalized CC'},
            shapes:[{
                type:'line', x0:p.lag, x1:p.lag, y0:0, y1:1, yref:'paper',
                line:{color:'#d1432b', dash:'dash'}
            }],
            annotations:[{
                x:p.lag, y:1, yref:'paper',
                text:`${p.lag.toFixed(2)} s, CC ${p.cc_max.toFixed(2)}`,
                showarrow:false, xanchor:'left', font:{color:'#d1432b', size:12}
            }]
        }, {
            displayModeBar:false, responsive:true
        });
    });
}

drawStations();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=False, port=5000)