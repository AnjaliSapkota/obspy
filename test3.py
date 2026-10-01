import base64
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from scipy.optimize import least_squares

def get_three_components(client, station, start, end, fs, fmin, fmax, warmup_sec):
    st = client.get_waveforms(
        network=station["net"],
        station=station["sta"],
        location=station["loc"],
        channel=station["cha"][:2] + "?",
        starttime=start - warmup_sec,
        endtime=end,
    )
    if len(st) == 0:
        raise RuntimeError(f"No waveforms found for {station['sta']}")

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


# P/S picking (matches pick_p() / pick_s())
def pick_p(z, fs, start, sta=1.0, lta=10.0, thr_on=3.5, thr_off=1.5):
    cft = recursive_sta_lta(z.data, int(sta * fs), int(lta * fs))
    p_time = None
    for on, _ in trigger_onset(cft, thr_on, thr_off):
        t = z.stats.starttime + on / fs
        if t >= start:
            p_time = t
            break
    return p_time, cft


def pick_s(env, fs, t0, p_time, sta=1.0, lta=8.0, min_sp_sec=4.0, max_sp_sec=50.0,
           thr_on=2.5, thr_off=1.0):
    """
    Detect the S arrival as the first STA/LTA trigger onset within the search
    window [p_time + min_sp_sec, p_time + max_sp_sec].

    Onset detection runs on the FULL cft array, not a slice of it. P-wave
    energy leaks into the horizontal components too, so the STA/LTA(S) curve
    is often still elevated from that when the search window opens at
    p_time + min_sp_sec. Slicing first and then looking for "the first sample
    above thr_on" mistakes that still-active trigger's tail for a brand new
    onset right at the window's start - which is what produced S-P times
    barely longer than min_sp_sec. Running trigger_onset on the whole curve
    finds where that earlier trigger genuinely dropped below thr_off, so the
    first onset actually starting inside the window is the next real rise -
    the S arrival - not a windowing artifact.
    """
    cft = recursive_sta_lta(env, int(sta * fs), int(lta * fs))
    p_idx = int((p_time - t0) * fs)

    i0 = p_idx + int(min_sp_sec * fs)
    i1 = min(p_idx + int(max_sp_sec * fs), len(cft))

    if i0 >= i1 or i0 >= len(cft):
        return None, cft

    s_idx = None
    for on, _off in trigger_onset(cft, thr_on, thr_off):
        if i0 <= on <= i1:
            s_idx = on
            break

    if s_idx is None:
        # No fresh onset found in the window (e.g. the curve never dropped
        # back below thr_off) - fall back to the window's peak so a pick
        # still comes out, but this case is worth a second look by eye.
        seg = cft[i0:i1]
        s_idx = i0 + int(np.argmax(seg))

    return t0 + s_idx / fs, cft


# --------------------------------------------------------------------------
# Geometry (matches latlon_to_xy / xy_to_latlon / circle_intersections)
# --------------------------------------------------------------------------
def latlon_to_xy(lat, lon, lat0, lon0):
    x = (lon - lon0) * 111.32 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110.574
    return x, y


def xy_to_latlon(x, y, lat0, lon0):
    lat = lat0 + y / 110.574
    lon = lon0 + x / (111.32 * np.cos(np.radians(lat0)))
    return lat, lon


def circle_intersections(x1, y1, r1, x2, y2, r2):
    """Kept for the visualization plot (drawing the distance circles)."""
    dx, dy = x2 - x1, y2 - y1
    d = np.sqrt(dx ** 2 + dy ** 2)
    if d == 0 or d > r1 + r2 or d < abs(r1 - r2):
        return []

    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h_sq = max(0.0, r1 ** 2 - a ** 2)
    h = np.sqrt(h_sq)

    xm = x1 + a * dx / d
    ym = y1 + a * dy / d
    rx = -dy * h / d
    ry = dx * h / d

    p1 = (xm + rx, ym + ry)
    p2 = (xm - rx, ym - ry)
    return [p1] if h < 1e-8 else [p1, p2]


def locate_epicenter_lsq(obs):
    """
    obs: list of (lat, lon, distance_km), at least 3 — uses ALL of them.

    A pairwise circle-intersection trilateration only ever considers the
    handful of exact intersection points between two circles at a time, then
    picks whichever scores lowest RMS against the third. With any pick noise
    that discrete choice is brittle and can lock onto the wrong one. Instead,
    fit (lat, lon) directly by nonlinear least squares against every station's
    observed distance at once, started from several points around the array
    so the optimizer doesn't get stuck in the wrong local minimum.

    Returns (lat, lon, rms_km, per_station_residuals_km) where a positive
    residual means the fitted point is farther from that station than its
    S-P distance implied (a quick way to spot a bad pick).
    """
    if len(obs) < 3:
        raise ValueError("At least 3 stations required for trilateration")

    lat0 = np.mean([o[0] for o in obs])
    lon0 = np.mean([o[1] for o in obs])
    xy = [latlon_to_xy(lat, lon, lat0, lon0) + (r,) for lat, lon, r in obs]
    max_r = max(r for _, _, r in xy)

    def resid(p):
        x, y = p
        return [np.sqrt((x - sx) ** 2 + (y - sy) ** 2) - r for sx, sy, r in xy]

    # Multi-start grid around the array centroid, out to roughly the largest
    # observed radius, so the fit isn't sensitive to the initial guess.
    offsets = np.linspace(-0.6 * max_r, 0.6 * max_r, 5)
    best = None
    for dx in offsets:
        for dy in offsets:
            res = least_squares(resid, [dx, dy])
            if best is None or res.cost < best.cost:
                best = res

    x_epi, y_epi = best.x
    lat_epi, lon_epi = xy_to_latlon(x_epi, y_epi, lat0, lon0)
    residuals = resid([x_epi, y_epi])
    rms = float(np.sqrt(np.mean(np.array(residuals) ** 2)))

    return float(lat_epi), float(lon_epi), rms, [float(r) for r in residuals]


# --------------------------------------------------------------------------
# Plotting helpers (all return base64-encoded PNGs)
# --------------------------------------------------------------------------
def _fig_to_png_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def waveform_stalta_png(name, z, env, cft_p, cft_s, tp, ts, thr_on, thr_off, thr_on_s, thr_off_s):
    t = z.times("matplotlib")
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(t, env, color="gray", lw=0.6, label="Horizontal energy")
    axes[0].plot(t, z.data, color="black", lw=0.6, label="Z")
    axes[0].axvline(tp.matplotlib_date, color="#2f6fed", ls="--", label="P pick")
    if ts is not None:
        axes[0].axvline(ts.matplotlib_date, color="#d1432b", ls="--", label="S pick")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"{name} \u2014 waveform")

    axes[1].plot(t, cft_p, color="#2f6fed", lw=0.8)
    axes[1].axhline(thr_on, color="#1f9e6b", ls=":", lw=1, label=f"on={thr_on}")
    axes[1].axhline(thr_off, color="#999", ls=":", lw=1, label=f"off={thr_off}")
    axes[1].axvline(tp.matplotlib_date, color="#2f6fed", ls="--")
    axes[1].set_ylabel("STA/LTA (P)")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(t, cft_s, color="#d1432b", lw=0.8)
    axes[2].axhline(thr_on_s, color="#1f9e6b", ls=":", lw=1, label=f"on={thr_on_s}")
    axes[2].axhline(thr_off_s, color="#999", ls=":", lw=1, label=f"off={thr_off_s}")
    axes[2].axvline(tp.matplotlib_date, color="#2f6fed", ls="--", alpha=0.5)
    if ts is not None:
        axes[2].axvline(ts.matplotlib_date, color="#d1432b", ls="--")
    axes[2].set_ylabel("STA/LTA (S)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].xaxis_date()
    axes[2].set_xlabel("Time")

    # Zoom to a window around the picks. Showing the whole fetched trace
    # (often 10+ minutes) compresses a 15-30 s S-P interval into a sliver a
    # few pixels wide, making it impossible to see whether a pick actually
    # sits on the true onset or a couple seconds off it.
    if ts is not None:
        sp = ts - tp
        pad_before = max(3.0, 0.3 * sp)
        pad_after = max(5.0, 0.5 * sp)
        xlim = ((tp - pad_before).matplotlib_date, (ts + pad_after).matplotlib_date)
    else:
        xlim = ((tp - 5).matplotlib_date, (tp + 20).matplotlib_date)
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
    ax.set_title(f"{name} \u2014 spectrogram")
    fig.tight_layout()
    return _fig_to_png_b64(fig)


def circle_map_png(obs, names, epicenter):
    """obs/names: any number of stations (>=3), not limited to 3."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ang = np.linspace(0, 2 * np.pi, 360)

    for (lat, lon, d), nm in zip(obs, names):
        lat_circle = lat + (d / 110.574) * np.sin(ang)
        lon_circle = lon + (d / (111.32 * np.cos(np.radians(lat)))) * np.cos(ang)
        ax.plot(lon_circle, lat_circle, lw=1.2, label=f"{nm} ({d:.1f} km)")
        ax.plot(lon, lat, "^", color="k", ms=9)
        ax.annotate(nm, (lon, lat), textcoords="offset points", xytext=(6, 6))

    ax.plot(epicenter[1], epicenter[0], "r*", ms=18,
            label=f"Epicenter ({epicenter[0]:.3f}N, {epicenter[1]:.3f}E)")
    ax.set_aspect(1 / np.cos(np.radians(epicenter[0])))
    ax.set_xlabel("Longitude (\u00b0E)")
    ax.set_ylabel("Latitude (\u00b0N)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Trilateration from S-P travel times")
    fig.tight_layout()
    return _fig_to_png_b64(fig)


# --------------------------------------------------------------------------
# Top-level orchestration: the full pipeline as one call
# --------------------------------------------------------------------------
def run_location(base_url, stations, start, end, fs=20.0, fmin=1.0, fmax=8.0,
                  warmup_sec=30.0, vp=6.0, vs=3.5,
                  sta_p=1.0, lta_p=10.0, thr_on=3.5, thr_off=1.5,
                  sta_s=1.0, lta_s=8.0, min_sp_sec=4.0, max_sp_sec=50.0,
                  thr_on_s=2.5, thr_off_s=1.0):
    client = Client(base_url)
    k = vp * vs / (vp - vs)

    station_results = []
    obs, names = [], []

    for station in stations:
        name = f"{station['net']}.{station['sta']}"
        entry = {"name": name, "lat": station["lat"], "lon": station["lon"]}
        try:
            z, env = get_three_components(client, station, start, end, fs, fmin, fmax, warmup_sec)
            tp, cft_p = pick_p(z, fs, start, sta_p, lta_p, thr_on, thr_off)
            if tp is None:
                entry["error"] = "No P pick"
                station_results.append(entry)
                continue

            ts, cft_s = pick_s(env, fs, z.stats.starttime, tp, sta_s, lta_s, min_sp_sec, max_sp_sec,
                               thr_on_s, thr_off_s)
            if ts is None:
                entry["error"] = "No S pick"
                station_results.append(entry)
                continue

            sp_diff = ts - tp
            dist = sp_diff * k

            entry.update({
                "p_time": tp.isoformat(),
                "s_time": ts.isoformat(),
                "sp_diff": float(sp_diff),
                "distance_km": float(dist),
                "waveform_png": waveform_stalta_png(name, z, env, cft_p, cft_s, tp, ts,
                                                     thr_on, thr_off, thr_on_s, thr_off_s),
                "spectrogram_png": spectrogram_png(name, z),
            })
            station_results.append(entry)
            obs.append((station["lat"], station["lon"], dist))
            names.append(name)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            station_results.append(entry)

    result = {"stations": station_results, "located": False}

    if len(obs) >= 3:
        lat, lon, rms, residuals = locate_epicenter_lsq(obs)

        # Attach each station's residual (predicted-minus-observed distance,
        # km) so a consistently large residual at one station flags a bad
        # pick rather than being silently absorbed into the fit.
        picked = [s for s in station_results if "distance_km" in s]
        for s, res in zip(picked, residuals):
            s["residual_km"] = res

        used = [s for s in station_results if "p_time" in s and "s_time" in s]
        origin_times = [
            UTCDateTime(s["p_time"]) - s["sp_diff"] * (vs / (vp - vs)) for s in used
        ]
        t_origin = UTCDateTime(np.mean([t.timestamp for t in origin_times]))

        result.update({
            "located": True,
            "epicenter": {"lat": lat, "lon": lon},
            "rms_km": rms,
            "origin_time": t_origin.isoformat(),
            "circle_map_png": circle_map_png(obs, names, (lat, lon)),
        })
    else:
        result["error"] = "Fewer than 3 stations produced valid P/S picks; cannot trilaterate."

    return result