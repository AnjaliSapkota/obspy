from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
import matplotlib.pyplot as plt
import numpy as np

client = Client("https://seiscomp.alertnepal.online")

stations = [
    {"sta": "EQM13", "lat": 28.256323,"lon": 85.367569,"net": "NP","cha": "EHZ","loc": "*"},
    {"sta": "EQM08","lat": 27.831,"lon": 86.65,"net": "NP","cha": "EHZ","loc": "*",},
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
R_EARTH = 6371.0


def preprocess(station, start, end):
    padded_start = start - warmup_sec
    stream = client.get_waveforms(network=station["net"],station=station["sta"],location=station["loc"],channel=station["cha"],starttime=padded_start,endtime=end)

    if len(stream) == 0:
        raise RuntimeError(f"No waveform found for {station['sta']}")

    stream.merge(method=1, fill_value="interpolate")
    stream.detrend("linear")
    stream.detrend("demean")
    stream.taper(max_percentage=0.05, type="hann")
    stream.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
    stream.interpolate(sampling_rate=fs, method="linear")
    return stream[0]

def fetch_all_traces(stations, start, end):
    traces = {}
    for st_info in stations:
        name = f"{st_info['net']}.{st_info['sta']}"
        tr = preprocess(st_info, start, end)
        traces[name] = tr
    return traces

def sta_lta(trace, sta=1.0, lta=10.0, thr_on=2.5, thr_off=1.5):
    df = trace.stats.sampling_rate
    nsta = int(sta * df)
    nlta = int(lta * df)

    cft = recursive_sta_lta(trace.data, nsta, nlta)
    warmup_samples = int(warmup_sec * df)

    tr_trimmed = trace.copy()
    tr_trimmed.trim(starttime=start)
    cft_trimmed = cft[warmup_samples:]

    onsets = trigger_onset(cft_trimmed, thr_on, thr_off)

    arrivals = []
    for onset in onsets:
        on_sample = onset[0]
        rel_time_sec = on_sample / df
        abs_time = tr_trimmed.stats.starttime + rel_time_sec
        arrivals.append((rel_time_sec, abs_time))

    return tr_trimmed, cft_trimmed, arrivals


def plot_waveforms_and_triggers(traces, sta_sec=1.0, lta_sec=10.0, thr_on=3.5, thr_off=1.5):

    fig, axes = plt.subplots(len(traces), 2, figsize=(14, 3 * len(traces)), sharex="col")
    if len(traces) == 1:
        axes = [axes]

    arrival_results = {}

    for idx, (name, trace) in enumerate(traces.items()):
        tr_trimmed, cft_trimmed, arrivals = sta_lta(trace, sta_sec, lta_sec, thr_on, thr_off)

        sampling_rate = tr_trimmed.stats.sampling_rate
        time_sec = np.arange(len(tr_trimmed.data)) / sampling_rate

        ax_wave = axes[idx][0]
        ax_wave.plot(time_sec, tr_trimmed.data, color="k", linewidth=0.7, label="Waveform")
        ax_wave.set_ylabel(f"{name}\nAmp")
        ax_wave.grid(True)

        ax_cft = axes[idx][1]
        ax_cft.plot(time_sec, cft_trimmed, color="blue", linewidth=0.8, label="STA/LTA")
        ax_cft.axhline(thr_on, color="red", linestyle="--", label=f"On ({thr_on})")
        ax_cft.axhline(thr_off, color="orange", linestyle=":", label=f"Off ({thr_off})")
        ax_cft.set_ylabel("STA/LTA Ratio")
        ax_cft.grid(True)

        if len(arrivals) > 0:
            first_arrival = arrivals[0]
            arrival_results[name] = first_arrival[1]

            for rel_t, abs_t in arrivals:
                ax_wave.axvline(rel_t, color="red", linestyle="--", linewidth=1.2)
                ax_cft.axvline(rel_t, color="red", linestyle="--", linewidth=1.2)
            print(f"[{name}] First P-arrival: {first_arrival[1]} (Rel: {first_arrival[0]:.2f} s)")
        else:
            print(f"[{name}] No arrivals detected above threshold = {thr_on}")

    axes[-1][0].set_xlabel("Time (s)")
    axes[-1][1].set_xlabel("Time (s)")
    fig.suptitle("STA/LTA Arrival Detection", fontsize=14)
    fig.tight_layout()
    plt.show()

    return arrival_results

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
        raise RuntimeError("No waveform found")

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

    z = st.select(component="Z")[0]
    hors = [tr for tr in st if tr.stats.channel[-1] != "Z"]
    if len(hors) < 2:
        raise RuntimeError("Need horizontal components for S picking")

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
    """Pick S-wave by finding the max horizontal energy ratio AFTER the P-wave window."""
    cft = recursive_sta_lta(env, int(sta * fs), int(lta * fs))

    p_idx = int((p_time - t0) * fs)
    # Enforce minimum delay after P arrival to skip P-wave energy packet
    i0 = p_idx + int(min_sp_sec * fs)
    i1 = min(p_idx + int(max_sp_sec * fs), len(cft))

    if i0 >= i1 or i0 >= len(cft):
        return None

    seg = cft[i0:i1]
    # Pick the peak of the horizontal STA/LTA ratio in the S window
    peak_rel_idx = np.argmax(seg)

    return t0 + (i0 + peak_rel_idx) / fs

def latlon_to_xy(lat, lon, lat0, lon0):
    x = ((lon - lon0) * 111.32 * np.cos(np.radians(lat0)))
    y = ((lat - lat0) * 110.574)
    return x, y

def xy_to_latlon(x, y, lat0, lon0):
    lat = (lat0 + y / 110.574)
    lon = (lon0 + x / (111.32 * np.cos(np.radians(lat0))))
    return lat, lon

def circle_intersections(x1, y1, r1, x2, y2, r2):
    dx = x2 - x1
    dy = y2 - y1
    d = np.sqrt(dx**2 + dy**2)
    if d == 0:
        return []

    if d > r1 + r2:
        return []

    if d < abs(r1 - r2):
        return []

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h_sq = r1**2 - a**2
    if h_sq < 0:
        h_sq = 0

    h = np.sqrt(h_sq)
    xm = x1 + a * dx / d
    ym = y1 + a * dy / d
    rx = -dy * h / d
    ry = dx * h / d
    p1 = (xm + rx,ym + ry)
    p2 = (xm - rx,ym - ry)
    if h < 1e-8:
        return [p1]

    return [p1, p2]

def locate_epicenter_intersection(obs):

    if len(obs) < 3:
        raise ValueError("At least 3 stations are required")

    obs = obs[:3]

    lat0 = np.mean([o[0] for o in obs])
    lon0 = np.mean([o[1] for o in obs])

    xy = []

    for lat, lon, radius in obs:
        x, y = latlon_to_xy(
            lat,
            lon,
            lat0,
            lon0
        )

        xy.append((x, y, radius))

    pair_intersections = {}

    pairs = [
        (0, 1),
        (0, 2),
        (1, 2)
    ]

    for i, j in pairs:

        x1, y1, r1 = xy[i]
        x2, y2, r2 = xy[j]

        points = circle_intersections(
            x1, y1, r1,
            x2, y2, r2
        )

        pair_intersections[(i, j)] = points

        print(
            f"\nCircle intersection "
            f"{i + 1}-{j + 1}:"
        )

        if len(points) == 0:

            print("  No intersection")

        else:

            for k, (x, y) in enumerate(points):

                lat, lon = xy_to_latlon(
                    x,
                    y,
                    lat0,
                    lon0
                )

                print(
                    f"  Candidate {k + 1}: "
                    f"{lat:.5f} N, "
                    f"{lon:.5f} E"
                )

    all_points = []

    for pair, points in pair_intersections.items():

        for x, y in points:

            all_points.append(
                (x, y, pair)
            )

    if len(all_points) == 0:
        raise RuntimeError(
            "No circle intersections found. "
            "Check P/S picks, velocity model, "
            "and station distances."
        )

    # Evaluate every intersection candidate
    # against all three measured circles.

    scored_candidates = []

    for x, y, pair in all_points:

        residuals = []

        for sx, sy, radius in xy:

            calculated_distance = np.sqrt(
                (x - sx) ** 2
                + (y - sy) ** 2
            )

            residuals.append(
                calculated_distance - radius
            )

        rms = np.sqrt(
            np.mean(
                np.array(residuals) ** 2
            )
        )

        scored_candidates.append(
            (
                rms,
                x,
                y,
                pair,
                residuals
            )
        )

    # Candidate whose distances best agree
    # with all three S-P distances.

    scored_candidates.sort(
        key=lambda item: item[0]
    )

    rms, x_epi, y_epi, pair, residuals = (
        scored_candidates[0]
    )

    lat_epi, lon_epi = xy_to_latlon(
        x_epi,
        y_epi,
        lat0,
        lon0
    )

    return (
        lat_epi,
        lon_epi,
        rms,
        scored_candidates
    )

# def haversine(lat1, lon1, lat2, lon2):
#     lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
#     a = (
#         np.sin((lat2 - lat1) / 2) ** 2
#         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
#     )
#     return 2 * R_EARTH * np.arcsin(np.sqrt(a))

# def locate_epicenter(obs):
#     lats = np.array([o[0] for o in obs])
#     lons = np.array([o[1] for o in obs])
#     d_obs = np.array([o[2] for o in obs])

#     def search(lat_rng, lon_rng, step):
#         la, lo = np.meshgrid(
#             np.arange(*lat_rng, step),
#             np.arange(*lon_rng, step),
#             indexing="ij",
#         )
#         misfit = sum(
#             (haversine(la, lo, s_la, s_lo) - d) ** 2
#             for s_la, s_lo, d in zip(lats, lons, d_obs)
#         )
#         i, j = np.unravel_index(np.argmin(misfit), misfit.shape)
#         return la[i, j], lo[i, j], misfit[i, j]

#     pad = 3.0
#     la, lo, _ = search(
#         (lats.min() - pad, lats.max() + pad),
#         (lons.min() - pad, lons.max() + pad),
#         0.05,
#     )
#     la, lo, m = search((la - 0.1, la + 0.1), (lo - 0.1, lo + 0.1), 0.002)
#     rms = np.sqrt(m / len(obs))
#     return la, lo, rms


def plot_map(obs, names, epi):
    fig, ax = plt.subplots(figsize=(8, 8))
    ang = np.linspace(0, 2 * np.pi, 360)
    for (lat, lon, d), nm in zip(obs, names):
        ax.plot(
            lon + d / (111.19 * np.cos(np.radians(lat))) * np.cos(ang),
            lat + d / 111.19 * np.sin(ang),
            lw=1.2,
            label=f"{nm} ({d:.1f} km)",
        )
        ax.plot(lon, lat, "^", color="k", ms=9)
        ax.annotate(nm, (lon, lat), textcoords="offset points", xytext=(6, 6))
    ax.plot(
        epi[1],
        epi[0],
        "r*",
        ms=18,
        label=f"Epicenter {epi[0]:.3f}, {epi[1]:.3f}",
    )
    ax.set_aspect(1 / np.cos(np.radians(epi[0])))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)
    ax.legend()
    ax.set_title("Trilateration from S-P distances")
    plt.show()


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
        print(f"[{name}] P={tp}  S={ts}  S-P={ts - tp:.2f}s  ->  {dist:.1f} km")
    except Exception as e:
        print(f"Skipping {name}: {e}")

if len(obs) >= 3:
    lat, lon, rms, candidates = locate_epicenter_intersection(obs)

    # FIXED: Clean unpacking of tp and distance d to compute origin time
    origin_times = [
        tp - (dist / vp)
        for (name, (z, env, tp, ts)), (_, _, dist) in zip(
            picks.items(), obs
        )
    ]
    t_origin = origin_times[0]  # First station origin estimate

    print(f"\nEpicenter: {lat:.3f}N, {lon:.3f}E   (misfit RMS {rms:.1f} km)")
    print(f"Approx. origin time: {t_origin}")
    plot_map(obs, names, (lat, lon))

    # QC visual inspection plot
    fig, axes = plt.subplots(
        len(picks), 1, figsize=(12, 3 * len(picks)), sharex=True
    )
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
    plt.show()
else:
    print("Need at least 3 stations with both P and S picks")

# Diagnostic single-component STA/LTA run
traces = fetch_all_traces(stations, start, end)
detected_times = plot_waveforms_and_triggers(
    traces, sta_sec=1.0, lta_sec=10.0, thr_on=3.5, thr_off=1.5
)