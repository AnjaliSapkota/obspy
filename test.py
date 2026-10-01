from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
import matplotlib.pyplot as plt
import numpy as np

client = Client("https://seiscomp.alertnepal.online")

stations = [
    {"sta": "EQM13", "lat": 28.256323, "lon": 85.367569, "net": "NP", "cha": "EHZ", "loc": "*"},
    {"sta": "EQM08", "lat": 27.831, "lon": 86.65, "net": "NP", "cha": "EHZ", "loc": "*"},
    {"sta": "EQM10", "lat": 28.299517, "lon": 83.960148, "net": "NP", "cha": "EHZ", "loc": "*"},
]

start = UTCDateTime("2026-09-22T07:45:00")
end = UTCDateTime("2026-09-22T07:55:00")

fs = 20.0
fmin = 1.0
fmax = 8.0
warmup_sec = 30.0

vp, vs = 6.0, 3.5
K = vp * vs / (vp - vs)


def preprocess(station, start, end):
    padded_start = start - warmup_sec
    stream = client.get_waveforms(
        network=station["net"],
        station=station["sta"],
        location=station["loc"],
        channel=station["cha"],
        starttime=padded_start,
        endtime=end,
    )

    if len(stream) == 0:
        raise RuntimeError(f"No waveform found for {station['sta']}")

    stream.merge(method=1, fill_value="interpolate")
    stream.detrend("linear")
    stream.detrend("demean")
    stream.taper(max_percentage=0.05, type="hann")
    stream.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
    stream.interpolate(sampling_rate=fs, method="linear")
    return stream[0]


def get_three_components(st_info):
    st = client.get_waveforms(
        network=st_info["net"],
        station=st_info["sta"],
        location=st_info["loc"],
        channel=st_info["cha"][:2] + "?",
        starttime=start - warmup_sec,
        endtime=end,
    )
    if len(st) == 0:
        raise RuntimeError(f"No waveforms found for {st_info['sta']}")

    st.merge(method=1, fill_value="interpolate")
    st.detrend("linear")
    st.detrend("demean")
    st.taper(max_percentage=0.05, type="hann")
    st.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
    st.interpolate(sampling_rate=fs, method="linear")

    t0 = max(tr.stats.starttime for tr in st)
    t1 = min(tr.stats.endtime for tr in st)
    st.trim(t0, t1)

    n = min(tr.stats.npts for tr in st)
    for tr in st:
        tr.data = tr.data[:n]

    z_list = st.select(component="Z")
    if len(z_list) == 0:
        raise RuntimeError("Vertical component missing")
    z = z_list[0]

    hors = [tr for tr in st if tr.stats.channel[-1] != "Z"]
    if len(hors) < 2:
        raise RuntimeError("Need two horizontal components for S-picking")

    hor_energy = np.zeros(n, dtype=float)
    for tr in hors:
        hor_energy += tr.data.astype(float) ** 2
    env = np.sqrt(hor_energy)

    return z, env


def pick_p(z, sta=1.0, lta=10.0, thr_on=3.5, thr_off=1.5):
    cft = recursive_sta_lta(z.data, int(sta * fs), int(lta * fs))
    for on, _ in trigger_onset(cft, thr_on, thr_off):
        t = z.stats.starttime + on / fs
        if t >= start:
            return t
    return None


def pick_s(env, t0, p_time, sta=1.0, lta=8.0, min_sp_sec=4.0, max_sp_sec=50.0):
    cft = recursive_sta_lta(env, int(sta * fs), int(lta * fs))
    p_idx = int((p_time - t0) * fs)

    i0 = p_idx + int(min_sp_sec * fs)
    i1 = min(p_idx + int(max_sp_sec * fs), len(cft))

    if i0 >= i1 or i0 >= len(cft):
        return None

    seg = cft[i0:i1]
    peak_rel_idx = np.argmax(seg)
    return t0 + (i0 + peak_rel_idx) / fs


def latlon_to_xy(lat, lon, lat0, lon0):
    x = (lon - lon0) * 111.32 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110.574
    return x, y


def xy_to_latlon(x, y, lat0, lon0):
    lat = lat0 + y / 110.574
    lon = lon0 + x / (111.32 * np.cos(np.radians(lat0)))
    return lat, lon


def circle_intersections(x1, y1, r1, x2, y2, r2):
    dx, dy = x2 - x1, y2 - y1
    d = np.sqrt(dx**2 + dy**2)
    if d == 0 or d > r1 + r2 or d < abs(r1 - r2):
        return []

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h_sq = max(0.0, r1**2 - a**2)
    h = np.sqrt(h_sq)

    xm = x1 + a * dx / d
    ym = y1 + a * dy / d
    rx = -dy * h / d
    ry = dx * h / d

    p1 = (xm + rx, ym + ry)
    p2 = (xm - rx, ym - ry)
    return [p1] if h < 1e-8 else [p1, p2]


def locate_epicenter_intersection(obs):
    if len(obs) < 3:
        raise ValueError("At least 3 stations required for trilateration")

    obs = obs[:3]
    lat0 = np.mean([o[0] for o in obs])
    lon0 = np.mean([o[1] for o in obs])

    xy = [latlon_to_xy(lat, lon, lat0, lon0) + (r,) for lat, lon, r in obs]
    pairs = [(0, 1), (0, 2), (1, 2)]

    all_points = []
    for i, j in pairs:
        x1, y1, r1 = xy[i]
        x2, y2, r2 = xy[j]
        pts = circle_intersections(x1, y1, r1, x2, y2, r2)
        all_points.extend([(x, y, (i, j)) for x, y in pts])

    if not all_points:
        raise RuntimeError("No circle intersections found. Check picks and velocity model.")

    scored_candidates = []
    for x, y, pair in all_points:
        residuals = [np.sqrt((x - sx) ** 2 + (y - sy) ** 2) - r for sx, sy, r in xy]
        rms = np.sqrt(np.mean(np.array(residuals) ** 2))
        scored_candidates.append((rms, x, y, pair, residuals))

    scored_candidates.sort(key=lambda item: item[0])
    rms, x_epi, y_epi, _, _ = scored_candidates[0]
    lat_epi, lon_epi = xy_to_latlon(x_epi, y_epi, lat0, lon0)

    return lat_epi, lon_epi, rms, scored_candidates


def plot_map(obs, names, epi):
    fig, ax = plt.subplots(figsize=(8, 8))
    ang = np.linspace(0, 2 * np.pi, 360)

    for (lat, lon, d), nm in zip(obs, names):
        lat_circle = lat + (d / 110.574) * np.sin(ang)
        lon_circle = lon + (d / (111.32 * np.cos(np.radians(lat)))) * np.cos(ang)
        ax.plot(lon_circle, lat_circle, lw=1.2, label=f"{nm} ({d:.1f} km)")
        ax.plot(lon, lat, "^", color="k", ms=9)
        ax.annotate(nm, (lon, lat), textcoords="offset points", xytext=(6, 6))

    ax.plot(epi[1], epi[0], "r*", ms=18, label=f"Epicenter ({epi[0]:.3f}N, {epi[1]:.3f}E)")
    ax.set_aspect(1 / np.cos(np.radians(epi[0])))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(True)
    ax.legend(loc="upper right")
    ax.set_title("Trilateration from S-P Travel Times")
    plt.show()


# Processing pipeline
obs, names, picks = [], [], {}
for st_info in stations:
    name = f"{st_info['net']}.{st_info['sta']}"
    try:
        z, env = get_three_components(st_info)
        tp = pick_p(z)
        if tp is None:
            print(f"[{name}] No P pick")
            continue

        ts = pick_s(env, z.stats.starttime, tp)
        if ts is None:
            print(f"[{name}] No S pick")
            continue

        dist = (ts - tp) * K
        picks[name] = (z, env, tp, ts)
        obs.append((st_info["lat"], st_info["lon"], dist))
        names.append(name)
        print(f"[{name}] P={tp} | S={ts} | S-P={ts - tp:.2f}s -> {dist:.1f} km")
    except Exception as e:
        print(f"Skipping {name}: {e}")

if len(obs) >= 3:
    lat, lon, rms, candidates = locate_epicenter_intersection(obs)

    # Correct Origin Time Calculation using S-P travel time
    origin_times = [tp - (ts - tp) * (vs / (vp - vs)) for name, (z, env, tp, ts) in picks.items()]
    t_origin = UTCDateTime(np.mean([t.timestamp for t in origin_times]))

    print(f"\nEpicenter Solution: {lat:.3f}°N, {lon:.3f}°E (RMS Misfit: {rms:.1f} km)")
    print(f"Estimated Origin Time: {t_origin.isoformat()}")

    plot_map(obs, names, (lat, lon))

    # QC Visualisation Plot
    fig, axes = plt.subplots(len(picks), 1, figsize=(12, 3 * len(picks)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (nm, (z, env, tp, ts)) in zip(axes, picks.items()):
        t = z.times("matplotlib")
        ax.plot(t, env, "gray", lw=0.6, label="Horizontal Energy")
        ax.plot(t, z.data, "k", lw=0.6, label="Z")
        ax.axvline(tp.matplotlib_date, color="b", ls="--", label="P pick")
        ax.axvline(ts.matplotlib_date, color="r", ls="--", label="S pick")
        ax.xaxis_date()
        ax.set_ylabel(nm)
        ax.legend(loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()
else:
    print("Insufficient station picks (< 3) to perform trilateration.")