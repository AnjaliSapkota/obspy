import pprint
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from scipy.optimize import minimize
from scipy.stats import kurtosis
import numpy as np


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
        raise RuntimeError(f"Vertical component missing for {station['sta']}")

    wanted = station["cha"][:2]
    z_ref = next((t for t in zs if t.stats.channel[:2] == wanted), zs[0])
    loc_ref = z_ref.stats.location
    band_ref = z_ref.stats.channel[:2]

    st = st.select(location=loc_ref, channel=band_ref + "?")
    horiz = st.select(component="N") + st.select(component="E")

    if len(horiz) < 2:
        horiz = st.select(component="1") + st.select(component="2")

    if len(horiz) < 2:
        raise RuntimeError(f"Need two horizontal components for {station['sta']} ({band_ref}*)")

    st = st.select(component="Z") + horiz

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

    hor_energy = np.zeros(n, dtype=float)
    for tr in hors:
        hor_energy += tr.data.astype(float) ** 2

    env = np.sqrt(hor_energy)
    return z, env


def aic_pick(trace_data, fs, approximate_idx, search_before=2.0, search_after=4.0):
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

    if not np.any(np.isfinite(aic)):
        return approximate_idx

    return int(i0 + np.nanargmin(aic))


def pick_p(z, fs, start, sta=1.0, lta=10.0, thr_on=3.5, thr_off=1.5, aic_before=2.0, aic_after=4.0):
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

    p_idx_aic = aic_pick(z.data, fs, p_idx, search_before=aic_before, search_after=aic_after)
    p_time_sta = z.stats.starttime + p_idx / fs
    p_time_aic = z.stats.starttime + p_idx_aic / fs

    return p_time_aic, cft, p_time_sta, float(cft[p_idx])


def pick_s(env, fs, t0, p_time, sta=1.0, lta=8.0, min_sp_sec=4.0, max_sp_sec=50.0, thr_on=2.5, thr_off=1.0, aic_before=1.5, aic_after=4.0):
    nsta = max(1, int(sta * fs))
    nlta = max(nsta + 1, int(lta * fs))

    cft = recursive_sta_lta(env, nsta, nlta)
    p_idx = int((p_time - t0) * fs)

    i0 = p_idx + int(min_sp_sec * fs)
    i1 = min(p_idx + int(max_sp_sec * fs), len(cft) - 1)

    if i0 >= i1 or i0 >= len(cft):
        return None, cft, None, None

    s_idx = None
    for on, _off in trigger_onset(cft, thr_on, thr_off):
        if i0 <= on <= i1:
            s_idx = int(on)
            break

    if s_idx is None:
        return None, cft, None, None

    s_idx_aic = aic_pick(env, fs, s_idx, search_before=aic_before, search_after=aic_after)
    s_time_sta = t0 + s_idx / fs
    s_time_aic = t0 + s_idx_aic / fs

    return s_time_aic, cft, s_time_sta, float(cft[s_idx])


def calculate_quality_metrics(z, env, fs, p_time, s_time, p_sta_trigger, s_sta_trigger):
    t0 = z.stats.starttime
    p_idx = int((p_time - t0) * fs)
    s_idx = int((s_time - t0) * fs)

    p_window = int(5.0 * fs)
    s_window = int(5.0 * fs)

    p0, p1 = max(0, p_idx - p_window // 2), min(len(z.data), p_idx + p_window // 2)
    s0, s1 = max(0, s_idx - s_window // 2), min(len(env), s_idx + s_window // 2)

    pz, sz = z.data[p0:p1].astype(float), z.data[s0:s1].astype(float)
    ph, sh = env[p0:p1].astype(float), env[s0:s1].astype(float)

    if len(pz) < 10 or len(sz) < 10:
        raise RuntimeError("Not enough samples for quality analysis")

    eps = 1e-12
    p_rms_z, p_rms_h = np.sqrt(np.mean(pz**2)), np.sqrt(np.mean(ph**2))
    s_rms_z, s_rms_h = np.sqrt(np.mean(sz**2)), np.sqrt(np.mean(sh**2))

    p_hv_ratio = float(p_rms_h / (p_rms_z + eps))
    s_hv_ratio = float(s_rms_h / (s_rms_z + eps))

    def outlier_ratio(data):
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad <= eps:
            return 0.0
        return float(np.mean((np.abs(data - median) / mad) > 4.45))

    p_outliers = outlier_ratio(pz)
    s_outliers = outlier_ratio(sz)
    sp = float(s_time - p_time)

    checks = []
    if sp < 4.0: checks.append("S-P interval is short")
    if p_sta_trigger < 3.0: checks.append("Weak P STA/LTA trigger")
    if s_sta_trigger < 2.0: checks.append("Weak S STA/LTA trigger")

    score = sum([
        p_sta_trigger >= 3.5,
        s_sta_trigger >= 2.5,
        sp >= 4.0,
        p_outliers <= 0.15,
        s_outliers <= 0.15,
        p_hv_ratio < 2.0,
        s_hv_ratio > p_hv_ratio,
    ])

    quality = "GOOD" if score >= 6 else ("REVIEW" if score >= 4 else "REJECT")

    return {
        "quality": quality,
        "quality_score": score,
        "quality_notes": checks,
        "p_kurtosis": float(kurtosis(pz, fisher=True, bias=False)),
        "s_kurtosis": float(kurtosis(sz, fisher=True, bias=False)),
        "p_outlier_ratio": p_outliers,
        "s_outlier_ratio": s_outliers,
        "p_horizontal_vertical_ratio": p_hv_ratio,
        "s_horizontal_vertical_ratio": s_hv_ratio,
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
    if len(obs) < 3:
        raise ValueError("Trilateration requires at least 3 stations")

    lat0 = np.mean([o[0] for o in obs])
    lon0 = np.mean([o[1] for o in obs])

    xy = []
    for lat, lon, hypocentral_distance in obs:
        epicentral_distance = 0.1 if hypocentral_distance <= source_depth_km else np.sqrt(hypocentral_distance**2 - source_depth_km**2)
        x, y = latlon_to_xy(lat, lon, lat0, lon0)
        xy.append((x, y, epicentral_distance))

    def objective(point):
        px, py = point
        return np.sum([(np.sqrt((px - sx) ** 2 + (py - sy) ** 2) - r) ** 2 for sx, sy, r in xy])

    res_opt = minimize(objective, x0=[0.0, 0.0], method="Nelder-Mead")
    x_epi, y_epi = res_opt.x

    residuals = [float(np.sqrt((x_epi - sx) ** 2 + (y_epi - sy) ** 2) - r) for sx, sy, r in xy]
    rms = float(np.sqrt(np.mean(np.asarray(residuals) ** 2)))
    lat_epi, lon_epi = xy_to_latlon(x_epi, y_epi, lat0, lon0)

    return float(lat_epi), float(lon_epi), rms, residuals


def calculate_station_magnitude():
    ...


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
):
    print(f"Connecting to FDSN Client at: {base_url}")
    client = Client(base_url)
    k = vp * vs / (vp - vs)

    station_results = []
    obs = []

    for station in stations:
        name = f"{station['net']}.{station['sta']}"
        print(f"--> Fetching and picking waveforms for {name}...")
        entry = {"name": name, "lat": station["lat"], "lon": station["lon"]}

        try:
            z, env = get_three_components(client, station, start, end, fs, fmin, fmax, wait_sec)
            tp, _, _, p_trigger = pick_p(z, fs, start)

            if tp is None:
                entry["error"] = "No P trigger found"
                entry["quality"] = "REJECT"
                station_results.append(entry)
                continue

            ts, _, _, s_trigger = pick_s(env, fs, z.stats.starttime, tp)

            if ts is None:
                entry["error"] = f"No S trigger found (P at {tp.isoformat()[11:23]})"
                entry["quality"] = "REJECT"
                station_results.append(entry)
                continue

            sp_diff = float(ts - tp)
            dist_km = float(sp_diff * k)
            q_info = calculate_quality_metrics(z, env, fs, tp, ts, p_trigger, s_trigger)
            ml, mw, amp_mm = calculate_station_magnitude(z, env, dist_km)

            entry.update({
                "p_time": tp.isoformat(),
                "s_time": ts.isoformat(),
                "sp_diff": sp_diff,
                "distance_km": dist_km,
                "ml": ml,
                "mw": mw,
                **q_info,
            })

            obs.append((station["lat"], station["lon"], dist_km))

        except Exception as e:
            entry["error"] = str(e)
            entry["quality"] = "REJECT"

        station_results.append(entry)

    if len(obs) >= 3:
        try:
            lat_epi, lon_epi, rms, residuals = locate_epicenter_lsq(obs)

            obs_idx = 0
            for sr in station_results:
                if "p_time" in sr and "s_time" in sr:
                    sr["residual_km"] = residuals[obs_idx]
                    obs_idx += 1

            origin_times = [
                UTCDateTime(sr["p_time"]) - (sr["distance_km"] / vp)
                for sr in station_results if "p_time" in sr and "s_time" in sr
            ]

            avg_origin = UTCDateTime(np.mean([t.timestamp for t in origin_times])).isoformat() if origin_times else None

            # Summary network magnitude average
            ml_vals = [sr["ml"] for sr in station_results if sr.get("ml") is not None]
            mw_vals = [sr["mw"] for sr in station_results if sr.get("mw") is not None]

            mean_ml = float(np.mean(ml_vals)) if ml_vals else None
            mean_mw = float(np.mean(mw_vals)) if mw_vals else None

            return {
                "located": True,
                "epicenter": {"lat": lat_epi, "lon": lon_epi},
                "rms_km": rms,
                "origin_time": avg_origin,
                "network_ml": mean_ml,
                "network_mw": mean_mw,
                "stations": station_results,
            }
        except Exception as e:
            return {"located": False, "error": f"Trilateration failed: {e}", "stations": station_results}

    return {"located": False, "error": f"Insufficient valid station picks ({len(obs)}/3 required)", "stations": station_results}


if __name__ == "__main__":
    stations = [
        {"sta": "EQM13", "lat": 28.256323, "lon": 85.367569, "net": "NP", "cha": "EHZ", "loc": "*"},
        {"sta": "EQM08", "lat": 27.831, "lon": 86.65, "net": "NP", "cha": "EHZ", "loc": "*"},
        {"sta": "EQM10", "lat": 28.299517, "lon": 83.960148, "net": "NP", "cha": "EHZ", "loc": "*"},
    ]

    server_url = "https://seiscomp.alertnepal.online"
    start_time = UTCDateTime("2026-09-22T07:45:00")
    end_time = UTCDateTime("2026-09-22T07:55:00")

    print("\n" + "=" * 60)
    print("STARTING SEISMIC LOCATION & MAGNITUDE CALCULATION")
    print("=" * 60)

    result = run_location(
        base_url=server_url,
        stations=stations,
        start=start_time,
        end=end_time
    )

    print("PROCESSING SUMMARY RESULT")
    pprint.pprint(result)