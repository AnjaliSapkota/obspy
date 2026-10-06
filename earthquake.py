import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from scipy.stats import kurtosis


def get_three_components(client, station, start, end, fs, fmin, fmax, wait_sec):
    st = client.get_waveforms(
        network=station["net"],
        station=station["sta"],
        location=station["loc"],
        channel=station["cha"][:2] + "?",
        starttime=start - wait_sec,
        endtime=end,
    )

    if len(st) == 0:
        raise RuntimeError(f"No waveforms found for {station['sta']}")

    zs = st.select(component="Z")
    if len(zs) == 0:
        raise RuntimeError("Vertical component missing")

    # prefer the band/instrument code the user typed (e.g. "BH"), else take the first Z found
    wanted = station["cha"][:2]
    z_ref = next((t for t in zs if t.stats.channel[:2] == wanted), zs[0])
    loc_ref = z_ref.stats.location
    band_ref = z_ref.stats.channel[:2]

    # keep one location and one band/instrument code only
    st = st.select(location=loc_ref, channel=band_ref + "?")

    # one Z + two horizontals (N/E preferred, otherwise 1/2)
    horiz = st.select(component="N") + st.select(component="E")

    if len(horiz) < 2:
        raise RuntimeError(f"Need two horizontal components for S-picking ({band_ref}*)")
    st = st.select(component="Z") + horiz

    st.merge(method=1, fill_value="interpolate")
    st.detrend("linear")
    st.detrend("demean")
    st.taper(max_percentage=0.05, type="hann")
    st.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True,)
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


def aic_pick(trace_data, fs, approximate_idx, search_before=2.0, search_after=4.0):
    # Refine an approximate arrival using the AIC picker
    n = len(trace_data)

    i0 = max(1, approximate_idx - int(search_before * fs))
    i1 = min(n - 2, approximate_idx + int(search_after * fs))

    if i1 <= i0 + 10:
        return approximate_idx

    x = np.asarray(trace_data[i0:i1], dtype=float)
    aic = np.full(len(x), np.nan)

    for k in range(1, len(x) - 1):
        var1 = np.var(x[:k])
        var2 = np.var(x[k:])

        if var1 <= 0 or var2 <= 0:
            continue

        aic[k] = k * np.log10(var1) + (len(x) - k) * np.log10(var2)

    valid = np.isfinite(aic)

    if not np.any(valid):
        return approximate_idx

    local_idx = np.nanargmin(aic)
    refined_idx = i0 + local_idx

    return int(refined_idx)


def pick_p(z,fs,start,sta=1.0,lta=10.0,thr_on=3.5,thr_off=1.5,aic_before=2.0,aic_after=4.0,):
    nsta = int(sta * fs)
    nlta = int(lta * fs)

    cft = recursive_sta_lta(z.data, nsta, nlta)
    p_idx = None

    for on, _off in trigger_onset(cft, thr_on, thr_off):
        t = z.stats.starttime + on / fs
        if t >= start:
            p_idx = int(on)
            break

    if p_idx is None:
        return None, cft, None, None

    p_idx_aic = aic_pick(
        z.data,
        fs,
        p_idx,
        search_before=aic_before,
        search_after=aic_after,
    )

    p_time_sta = z.stats.starttime + p_idx / fs
    p_time_aic = z.stats.starttime + p_idx_aic / fs

    trigger_value = float(cft[p_idx])

    return (
        p_time_aic,
        cft,
        p_time_sta,
        trigger_value,
    )


def pick_s(
    env,
    fs,
    t0,
    p_time,
    sta=1.0,
    lta=8.0,
    min_sp_sec=4.0,
    max_sp_sec=50.0,
    thr_on=2.5,
    thr_off=1.0,
    aic_before=1.5,
    aic_after=4.0,
):
    """Detect S after P."""
    nsta = max(1, int(sta * fs))
    nlta = max(nsta + 1, int(lta * fs))

    cft = recursive_sta_lta(env, nsta, nlta)
    p_idx = int((p_time - t0) * fs)

    i0 = p_idx + int(min_sp_sec * fs)
    i1 = min(p_idx + int(max_sp_sec * fs), len(cft) - 1)

    if i0 >= i1 or i0 >= len(cft):
        return None, cft, None, None

    s_idx = None
    all_triggers = trigger_onset(cft, thr_on, thr_off)

    for on, _off in all_triggers:
        if i0 <= on <= i1:
            s_idx = int(on)
            break

    if s_idx is None:
        return None, cft, None, None

    s_idx_aic = aic_pick(
        env,
        fs,
        s_idx,
        search_before=aic_before,
        search_after=aic_after,
    )

    s_time_sta = t0 + s_idx / fs
    s_time_aic = t0 + s_idx_aic / fs

    trigger_value = float(cft[s_idx])

    return (
        s_time_aic,
        cft,
        s_time_sta,
        trigger_value,
    )


def calculate_quality_metrics(
    z,
    env,
    fs,
    p_time,
    s_time,
    p_sta_trigger,
    s_sta_trigger,
):
    t0 = z.stats.starttime
    p_idx = int((p_time - t0) * fs)
    s_idx = int((s_time - t0) * fs)

    p_window = int(5.0 * fs)
    s_window = int(5.0 * fs)

    p0 = max(0, p_idx - p_window // 2)
    p1 = min(len(z.data), p_idx + p_window // 2)

    s0 = max(0, s_idx - s_window // 2)
    s1 = min(len(env), s_idx + s_window // 2)

    pz = z.data[p0:p1].astype(float)
    sz = z.data[s0:s1].astype(float)

    ph = env[p0:p1].astype(float)
    sh = env[s0:s1].astype(float)

    if len(pz) < 10 or len(sz) < 10:
        raise RuntimeError("Not enough samples for quality analysis")

    p_rms_z = float(np.sqrt(np.mean(pz**2)))
    p_rms_h = float(np.sqrt(np.mean(ph**2)))
    s_rms_z = float(np.sqrt(np.mean(sz**2)))
    s_rms_h = float(np.sqrt(np.mean(sh**2)))

    eps = 1e-12
    p_hv_ratio = p_rms_h / (p_rms_z + eps)
    s_hv_ratio = s_rms_h / (s_rms_z + eps)

    p_kurt = float(kurtosis(pz, fisher=True, bias=False))
    s_kurt = float(kurtosis(sz, fisher=True, bias=False))

    def outlier_ratio(data):
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad <= eps:
            return 0.0
        zscore = np.abs(data - median) / mad
        return float(np.mean(zscore > 4.45))

    p_outliers = outlier_ratio(pz)
    s_outliers = outlier_ratio(sz)

    sp = float(s_time - p_time)
    checks = []

    if sp < 4.0:
        checks.append("S-P interval is very short")
    if sp > 50.0:
        checks.append("S-P interval exceeds search range")
    if p_sta_trigger < 3.0:
        checks.append("Weak P STA/LTA trigger")
    if s_sta_trigger < 2.0:
        checks.append("Weak S STA/LTA trigger")
    if p_outliers > 0.15:
        checks.append("P window contains many outliers")
    if s_outliers > 0.15:
        checks.append("S window contains many outliers")

    score = 0
    if p_sta_trigger >= 3.5:
        score += 1
    if s_sta_trigger >= 2.5:
        score += 1
    if sp >= 4.0:
        score += 1
    if p_outliers <= 0.15:
        score += 1
    if s_outliers <= 0.15:
        score += 1
    if p_hv_ratio < 2.0:
        score += 1
    if s_hv_ratio > p_hv_ratio:
        score += 1

    if score >= 6:
        quality = "GOOD"
    elif score >= 4:
        quality = "REVIEW"
    else:
        quality = "REJECT"

    return {
        "quality": quality,
        "quality_score": score,
        "quality_notes": checks,
        "p_kurtosis": p_kurt,
        "s_kurtosis": s_kurt,
        "p_outlier_ratio": p_outliers,
        "s_outlier_ratio": s_outliers,
        "p_horizontal_vertical_ratio": p_hv_ratio,
        "s_horizontal_vertical_ratio": s_hv_ratio,
        "p_sta_trigger": float(p_sta_trigger),
        "s_sta_trigger": float(s_sta_trigger),
    }


def latlon_to_xy(lat, lon, lat0, lon0):
    x = (lon - lon0) * 111.32 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110.574
    return x, y


def xy_to_latlon(x, y, lat0, lon0):
    lat = lat0 + y / 110.574
    lon = lon0 + x / (111.32 * np.cos(np.radians(lat0)))
    return lat, lon
def locate_epicenter_lsq(obs, source_depth_km=0.0):
    if len(obs) != 3:
        raise ValueError("Intersection method requires exactly 3 stations")

    lat0 = np.mean([o[0] for o in obs])
    lon0 = np.mean([o[1] for o in obs])

    xy = []
    for lat, lon, hypocentral_distance in obs:
        if hypocentral_distance <= source_depth_km:
            epicentral_distance = 0.1
        else:
            epicentral_distance = np.sqrt(
                hypocentral_distance**2 - source_depth_km**2
            )

        x, y = latlon_to_xy(lat, lon, lat0, lon0)
        xy.append((x, y, epicentral_distance))

    x1, y1, r1 = xy[0]
    x2, y2, r2 = xy[1]
    x3, y3, r3 = xy[2]

    dx12 = x2 - x1
    dy12 = y2 - y1
    d12 = np.sqrt(dx12**2 + dy12**2)

    if d12 == 0:
        raise ValueError("Two stations have identical coordinates")

    ex = dx12 / d12
    ey = dy12 / d12

    # Chord midpoint along baseline
    a = (r1**2 - r2**2 + d12**2) / (2 * d12)

    # Perpendicular offset to intersection points
    h_squared = r1**2 - a**2
    h = np.sqrt(max(0.0, h_squared))

    xm = x1 + a * ex
    ym = y1 + a * ey

    px = -ey
    py = ex

    p1 = (xm + h * px, ym + h * py)
    p2 = (xm - h * px, ym - h * py)

    # Calculate total RMS residual across all 3 station circles for both roots
    def calc_rms(point):
        px_c, py_c = point
        res = []
        for sx, sy, r in xy:
            dist = np.sqrt((px_c - sx)**2 + (py_c - sy)**2)
            res.append(dist - r)
        return np.sqrt(np.mean(np.array(res)**2)), res

    rms1, res1 = calc_rms(p1)
    rms2, res2 = calc_rms(p2)

    # Choose the candidate root that minimizes overall residual RMS across all stations
    if rms1 <= rms2:
        x_epi, y_epi = p1
        rms = rms1
        residuals = res1
    else:
        x_epi, y_epi = p2
        rms = rms2
        residuals = res2

    lat_epi, lon_epi = xy_to_latlon(x_epi, y_epi, lat0, lon0)

    return float(lat_epi), float(lon_epi), float(rms), residuals

    def third_circle_error(point):
        x, y = point
        distance = np.sqrt((x - x3)**2 + (y - y3)**2)
        return abs(distance - r3)

    # Select candidate closest to station 3 circle
    if third_circle_error(p1) <= third_circle_error(p2):
        x_epi, y_epi = p1
    else:
        x_epi, y_epi = p2

    residuals = [
        float(np.sqrt((x_epi - sx)**2 + (y_epi - sy)**2) - r)
        for sx, sy, r in xy
    ]

    lat_epi, lon_epi = xy_to_latlon(x_epi, y_epi, lat0, lon0)
    rms = float(np.sqrt(np.mean(np.asarray(residuals)**2)))

    return float(lat_epi), float(lon_epi), rms, residuals

    # Select the intersection that is closest to the
    # third station's circle radius
    def third_circle_error(point):
        x, y = point
        distance = np.sqrt(
            (x - x3)**2 +
            (y - y3)**2
        )
        return abs(distance - r3)

    error1 = third_circle_error(p1)
    error2 = third_circle_error(p2)

    if error1 <= error2:
        x_epi, y_epi = p1
    else:
        x_epi, y_epi = p2

    # Calculate residuals against all three observed circles
    residuals = []

    for sx, sy, r in xy:
        calculated_distance = np.sqrt(
            (x_epi - sx)**2 +
            (y_epi - sy)**2
        )

        residuals.append(
            float(calculated_distance - r)
        )

    lat_epi, lon_epi = xy_to_latlon(
        x_epi,
        y_epi,
        lat0,
        lon0,
    )

    rms = float(
        np.sqrt(
            np.mean(
                np.asarray(residuals)**2
            )
        )
    )

    return (
        float(lat_epi),
        float(lon_epi),
        rms,
        residuals,
    )

def _fig_to_png_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def waveform_stalta_png(name,z,env,cft_p,cft_s,tp,ts,p_sta,s_sta,thr_on,thr_off,thr_on_s,thr_off_s,):
    t = z.times("matplotlib")

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(t, env, color="gray", lw=0.6, label="Horizontal energy")
    axes[0].plot(t, z.data, color="black", lw=0.6, label="Z")
    axes[0].axvline(tp.matplotlib_date, color="#2f6fed", ls="--", label="P AIC pick")

    if ts is not None:
        axes[0].axvline(ts.matplotlib_date, color="#d1432b", ls="--", label="S AIC pick")

    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"{name} — waveform")

    axes[1].plot(t, cft_p, color="#2f6fed", lw=0.8)
    axes[1].axhline(thr_on, color="#1f9e6b", ls=":", lw=1, label=f"on={thr_on}")
    axes[1].axhline(thr_off, color="#999", ls=":", lw=1, label=f"off={thr_off}")
    axes[1].axvline(tp.matplotlib_date, color="#2f6fed", ls="--")

    if p_sta is not None:
        axes[1].axvline(p_sta.matplotlib_date, color="orange", ls=":", label="P STA/LTA trigger")

    axes[1].set_ylabel("STA/LTA (P)")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(t, cft_s, color="#d1432b", lw=0.8)
    axes[2].axhline(thr_on_s, color="#1f9e6b", ls=":", lw=1, label=f"on={thr_on_s}")
    axes[2].axhline(thr_off_s, color="#999", ls=":", lw=1, label=f"off={thr_off_s}")
    axes[2].axvline(tp.matplotlib_date, color="#2f6fed", ls="--", alpha=0.5)

    if ts is not None:
        axes[2].axvline(ts.matplotlib_date, color="#d1432b", ls="--")

    if s_sta is not None:
        axes[2].axvline(s_sta.matplotlib_date, color="orange", ls=":", label="S STA/LTA trigger")

    axes[2].set_ylabel("STA/LTA (S)")
    axes[2].legend(loc="upper right", fontsize=8)

    axes[2].xaxis_date()
    axes[2].set_xlabel("Time")

    if ts is not None:
        sp = ts - tp
        pad_before = max(3.0, 0.3 * sp)
        pad_after = max(5.0, 0.5 * sp)
        xlim = (
            (tp - pad_before).matplotlib_date,
            (ts + pad_after).matplotlib_date,
        )
    else:
        xlim = (
            (tp - 5).matplotlib_date,
            (tp + 60).matplotlib_date,
        )

    for ax in axes:
        ax.set_xlim(*xlim)
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    return _fig_to_png_b64(fig)


def spectrogram_png(name, z):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.specgram(z.data, Fs=z.stats.sampling_rate, cmap="viridis")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s, from trace start)")
    ax.set_title(f"{name} — spectrogram")
    fig.tight_layout()
    return _fig_to_png_b64(fig)


def circle_map_png(obs, names, epicenter):
    fig, ax = plt.subplots(figsize=(7, 7))
    ang = np.linspace(0, 2 * np.pi, 360)

    for (lat, lon, d), nm in zip(obs, names):
        lat_circle = lat + (d / 110.574) * np.sin(ang)
        lon_circle = lon + (d / (111.32 * np.cos(np.radians(lat)))) * np.cos(ang)

        ax.plot(lon_circle, lat_circle, lw=1.2, label=f"{nm} ({d:.1f} km)")
        ax.plot(lon, lat, "^", color="k", ms=9)
        ax.annotate(nm, (lon, lat), textcoords="offset points", xytext=(6, 6))

    ax.plot(
        epicenter[1],
        epicenter[0],
        "r*",
        ms=18,
        label=f"Epicenter ({epicenter[0]:.3f}N, {epicenter[1]:.3f}E)",
    )

    ax.set_aspect(1 / np.cos(np.radians(epicenter[0])))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Trilateration from S-P travel times")

    fig.tight_layout()
    return _fig_to_png_b64(fig)


def run_location(
    base_url,
    stations,
    start,
    end,
    fs=20.0,
    fmin=1.0,
    fmax=8.0,
    wait_sec=30.0,
    vp=6.0,
    vs=3.5,
    sta_p=1.0,
    lta_p=10.0,
    thr_on=3.5,
    thr_off=1.5,
    sta_s=1.0,
    lta_s=8.0,
    min_sp_sec=4.0,
    max_sp_sec=50.0,
    thr_on_s=2.5,
    thr_off_s=1.0,
):
    client = Client(base_url)
    k = vp * vs / (vp - vs)

    station_results = []
    obs = []
    names = []

    for station in stations:
        name = f"{station['net']}.{station['sta']}"
        entry = {
            "name": name,
            "lat": station["lat"],
            "lon": station["lon"],
        }

        try:
            z, env = get_three_components(client,station,start,end,fs,fmin,fmax,wait_sec,)
            tp, cft_p, p_sta_time, p_trigger = pick_p(z,fs,start,sta_p,lta_p,thr_on,thr_off,)

            if tp is None:
                entry["error"] = "No P trigger found"
                entry["quality"] = "REJECT"
                station_results.append(entry)
                continue

            ts, cft_s, s_sta_time, s_trigger = pick_s(env,fs,z.stats.starttime,tp,sta_s,lta_s,min_sp_sec,max_sp_sec,thr_on_s,thr_off_s,)

            if ts is None:
                entry["error"] = f"No S trigger found in window (P at {tp.isoformat()[11:23]})"
                entry["quality"] = "REJECT"
                entry["waveform_png"] = waveform_stalta_png(
                    name, z, env, cft_p, cft_s, tp, None,
                    p_sta_time, None, thr_on, thr_off, thr_on_s, thr_off_s,
                )
                entry["spectrogram_png"] = spectrogram_png(name, z)
                station_results.append(entry)
                continue

            sp_diff = float(ts - tp)
            dist_km = float(sp_diff * k)

            q_info = calculate_quality_metrics(z,env,fs,tp,ts,p_trigger,s_trigger,)

            entry.update(
                {
                    "p_time": tp.isoformat(),
                    "s_time": ts.isoformat(),
                    "sp_diff": sp_diff,
                    "distance_km": dist_km,
                    "waveform_png": waveform_stalta_png(
                        name,
                        z,
                        env,
                        cft_p,
                        cft_s,
                        tp,
                        ts,
                        p_sta_time,
                        s_sta_time,
                        thr_on,
                        thr_off,
                        thr_on_s,
                        thr_off_s,
                    ),
                    "spectrogram_png": spectrogram_png(name, z),
                    **q_info,
                }
            )

            obs.append((station["lat"], station["lon"], dist_km))
            names.append(name)

        except Exception as e:
            entry["error"] = str(e)
            entry["quality"] = "REJECT"

        station_results.append(entry)

    if len(obs) >= 3:
        try:
            lat_epi, lon_epi, rms, residuals = locate_epicenter_lsq(obs)

            obs_idx = 0
            for sr in station_results:
                if "p_time" in sr and sr.get("quality") != "REJECT":
                    sr["residual_km"] = residuals[obs_idx]
                    obs_idx += 1

            origin_times = []
            for sr in station_results:
                if "p_time" in sr and sr.get("quality") != "REJECT":
                    p_utc = UTCDateTime(sr["p_time"])
                    origin_times.append(p_utc - (sr["distance_km"] / vp))

            avg_origin_time = (
                UTCDateTime(np.mean([t.timestamp for t in origin_times])).isoformat()
                if origin_times
                else None
            )

            return {
                "located": True,
                "epicenter": {"lat": lat_epi, "lon": lon_epi},
                "rms_km": rms,
                "origin_time": avg_origin_time,
                "circle_map_png": circle_map_png(
                    obs, names, (lat_epi, lon_epi)
                ),
                "stations": station_results,
            }
        except Exception as e:
            return {
                "located": False,
                "error": f"Trilateration failed: {e}",
                "stations": station_results,
            }

    return {
        "located": False,
        "error": f"Insufficient valid station picks ({len(obs)}/3 required)",
        "stations": station_results,
    }