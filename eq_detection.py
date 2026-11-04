import csv
import os
import queue
import threading
import time

import numpy as np
from scipy.signal import butter, hilbert, lfilter, sosfilt
from obspy import Trace, UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth


CONFIG = {
    "server": "ring.wscada.net:18000",
    "stations": [
        {"net": "NP", "sta": "EQM10", "cha": "EHZ", "loc": "*", "lat": 28.299517, "lon": 83.960148},
        {"net": "NP", "sta": "EQM26", "cha": "EHZ", "loc": "*", "lat": 27.3936969, "lon": 86.6791885},
        {"net": "NP", "sta": "EQM07", "cha": "EHZ", "loc": "*", "lat": 27.814941, "lon": 86.713401},
        {"net": "NP", "sta": "EQM08", "cha": "EHZ", "loc": "*", "lat": 27.831000, "lon": 86.65},
    ],

    "socket_timeout_sec": 10,
    "reconnect_delay_sec": 10,
    "status_every_sec": 30,
    "queue_size": 500,

    # Preprocessing
    "mean_tau_sec": 30.0,
    "fmin": 1.0,
    "fmax": 10.0,
    "bp_order": 4,

    # Adaptive moving averages
    "tau_fast": 0.5,
    "tau_medium": 5.0,
    "tau_bg": 100.0,
    "warmup_sec": 100.0,
    "level_floor": 1e-20,

    # Event detection
    "r1_on": 3.0,
    "r2_on": 1.5,
    "persist_sec": 0.5,
    "end_ratio": 2.0,
    "end_hold_sec": 5.0,
    "min_event_sec": 3.0,
    "max_event_sec": 120.0,
    "refractory_sec": 30.0,

    # Waveform buffer
    "ring_sec": 300.0,
    "pre_trigger_sec": 20.0,
    "post_event_sec": 3.0,

    # P-wave picking
    "sta_sec": 0.5,
    "lta_sec": 5.0,
    "peak_back_sec": 10.0,
    "peak_fwd_sec": 2.0,
    "ratio_peak_min": 3.0,
    "ratio_low": 2.0,
    "gap_tol_sec": 1.0,
    "aic_before_sec": 2.0,
    "aic_after_sec": 1.0,
    "aic_max_shift_sec": 1.5,

    # Network coincidence
    "coincidence_window_sec": 40.0,
    "coincidence_min_stations": 3,
    "min_p_velocity_km_s": 4.0,
    "coincidence_tolerance_sec": 2.0,

    # Output
    "save_events": True,
    "event_dir": "events",
    "log_csv": "events_log.csv",
}


# 1. DATA ACQUISITION

def acquisition_loop(q, cfg, stop, dropped):
    from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient

    while not stop.is_set():
        client = None
        try:
            client = EasySeedLinkClient(cfg["server"], autoconnect=False)
            client.conn.timeout = cfg["socket_timeout_sec"]
            client.connect()

            for s in cfg["stations"]:
                selector = ("??" if s["loc"] == "*" else s["loc"]) + s["cha"]
                client.select_stream(s["net"], s["sta"], selector)

            def on_data(trace):
                try:
                    q.put(trace, timeout=1)
                except queue.Full:
                    dropped["records"] += 1

            client.on_data = on_data

            print("Connected to", cfg["server"])
            client.run()
            print("SeedLink stream ended.")

        except Exception as exc:
            print("SeedLink error:", exc)

        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

        if not stop.is_set():
            print("Reconnecting in %d s ..." % cfg["reconnect_delay_sec"])
            stop.wait(cfg["reconnect_delay_sec"])


# 2. STATION STATE

def new_station_state(sid, fs, t0, cfg):
    f_hi = min(cfg["fmax"], 0.4 * fs)
    if fs <= 0 or f_hi <= cfg["fmin"]:
        raise ValueError("%s: invalid sampling rate or bandpass" % sid)

    sos = butter(cfg["bp_order"], [cfg["fmin"], f_hi],
                 btype="band", fs=fs, output="sos")
    n = int(cfg["ring_sec"] * fs)

    return {
        "id": sid,
        "fs": fs,
        "t0": t0,              # time of sample 0
        "n_total": 0,          # samples received so far

        "ring": np.zeros(n),
        "ring_n": n,

        "sos": sos,
        "zi": np.zeros((sos.shape[0], 2)),

        "mean": None,
        "fast": 0.0,
        "med": 0.0,
        "bg": 0.0,
        "bg_count": 0,

        "state": "IDLE",
        "hits": 0,
        "trigger_idx": None,
        "last_active_idx": None,
        "quiet_n": 0,
        "quiet_until": 0,
    }


# 3. WAVEFORM BUFFER

def ring_write(st, filtered):
    i0 = st["n_total"]
    idx = np.arange(i0, i0 + len(filtered)) % st["ring_n"]
    st["ring"][idx] = filtered
    st["n_total"] += len(filtered)
    return i0


def ring_read(st, i_start, i_stop):
    i_start = max(i_start, st["n_total"] - st["ring_n"], 0)
    i_stop = min(i_stop, st["n_total"])
    if i_stop <= i_start:
        return np.zeros(0), i_start
    idx = np.arange(i_start, i_stop) % st["ring_n"]
    return st["ring"][idx].copy(), i_start


# 4. MOVING AVERAGE + PREPROCESSING

def ema(x, tau, fs, previous):
    a = 1.0 - np.exp(-1.0 / (tau * fs))
    y, _ = lfilter([a], [1.0, -(1.0 - a)], x,
                   zi=np.array([(1.0 - a) * previous]))
    return y


def preprocess_block(st, x, cfg):
    if st["mean"] is None:
        st["mean"] = x[0]

    mean = ema(x, cfg["mean_tau_sec"], st["fs"], st["mean"])
    st["mean"] = mean[-1]

    y, st["zi"] = sosfilt(st["sos"], x - mean, zi=st["zi"])
    return y


# 5. TRACE HANDLING (data is assumed to arrive contiguously)

def process_trace(states, tr, cfg):
    data = np.ma.asarray(tr.data)
    if np.ma.getmaskarray(data).any():
        return []                       # skip records with masked samples

    x = np.asarray(data, dtype=float)
    fs = float(tr.stats.sampling_rate)
    if len(x) == 0 or fs <= 0 or not np.all(np.isfinite(x)):
        return []

    sid = tr.id
    st = states.get(sid)
    if st is None:
        st = new_station_state(sid, fs, tr.stats.starttime.timestamp, cfg)
        states[sid] = st

    y = preprocess_block(st, x, cfg)
    return detect_block(st, y, cfg)


# 6. EVENT DETECTION

def trigger_condition(f, m, bg, cfg):
    floor = cfg["level_floor"]
    return (f > cfg["r1_on"] * max(bg, floor)
            and f > cfg["r2_on"] * max(m, floor))


def detect_block(st, y, cfg):
    events = []
    fs = st["fs"]
    cf = np.abs(y)

    fast = ema(cf, cfg["tau_fast"], fs, st["fast"])
    med = ema(cf, cfg["tau_medium"], fs, st["med"])
    st["fast"], st["med"] = fast[-1], med[-1]

    i0 = ring_write(st, y)

    a_bg = 1.0 - np.exp(-1.0 / (cfg["tau_bg"] * fs))
    warm_n = int(cfg["warmup_sec"] * fs)
    persist_n = max(1, int(cfg["persist_sec"] * fs))
    hold_n = max(1, int(cfg["end_hold_sec"] * fs))
    refr_n = int(cfg["refractory_sec"] * fs)

    for j, (c, f, m) in enumerate(zip(cf, fast, med)):
        i = i0 + j

        if st["state"] == "IDLE":
            # background level: cumulative mean during warmup, then EMA
            if st["bg_count"] < warm_n:
                st["bg_count"] += 1
                st["bg"] += (c - st["bg"]) / st["bg_count"]
                if st["bg_count"] < warm_n:
                    continue
            else:
                st["bg"] += a_bg * (c - st["bg"])

            if i < st["quiet_until"]:
                continue

            # trigger must persist for persist_n consecutive samples
            if trigger_condition(f, m, st["bg"], cfg):
                if st["hits"] == 0:
                    st["trigger_idx"] = i
                st["hits"] += 1
                if st["hits"] >= persist_n:
                    st["state"] = "ACTIVE"
                    st["last_active_idx"] = i
                    st["quiet_n"] = 0
            else:
                st["hits"] = 0

        else:  # ACTIVE
            if f > cfg["end_ratio"] * max(st["bg"], cfg["level_floor"]):
                st["last_active_idx"] = i
                st["quiet_n"] = 0
            else:
                st["quiet_n"] += 1

            duration = (i - st["trigger_idx"]) / fs
            ended = None

            if st["quiet_n"] >= hold_n:
                ended = (st["last_active_idx"], "quiet")
            elif duration >= cfg["max_event_sec"]:
                ended = (i, "max_duration")

            if ended is not None:
                end_idx, reason = ended
                if (end_idx - st["trigger_idx"]) / fs >= cfg["min_event_sec"]:
                    ev = finalize_event(st, end_idx, reason, cfg)
                    if ev is not None:
                        events.append(ev)

                st["state"] = "IDLE"
                st["hits"] = 0
                st["quiet_until"] = i + refr_n

    return events


# 7. EVENT FINALIZATION

def finalize_event(st, end_idx, reason, cfg):
    fs = st["fs"]
    pre_n = int(cfg["pre_trigger_sec"] * fs)
    post_n = int(cfg["post_event_sec"] * fs)

    wf, i_start = ring_read(st, st["trigger_idx"] - pre_n, end_idx + post_n + 1)
    if len(wf) < int(5 * fs):
        return None

    def to_time(idx):
        return st["t0"] + idx / fs

    wf_start = to_time(i_start)
    trig_rel = st["trigger_idx"] - i_start

    k = pick_p_wave(wf, fs, trig_rel, cfg)
    p_time = None if k is None else wf_start + k / fs

    trigger_time = to_time(st["trigger_idx"])

    return {
        "id": st["id"],
        "station": ".".join(st["id"].split(".")[:2]),
        "fs": fs,
        "trigger_time": trigger_time,
        "p_time": p_time,
        "end_time": to_time(end_idx),
        "end_reason": reason,
        "wf_start": wf_start,
        "waveform": wf,
        "snr": simple_snr(wf, fs, k),
        "coincident": False,
    }


# 8. P-WAVE PICKING (STA/LTA backward search + AIC refinement)

def sta_lta(cf, nsta, nlta):
    n = len(cf)
    ratio = np.zeros(n)
    if n < nsta + nlta + 1:
        return ratio

    c = np.concatenate(([0.0], np.cumsum(cf)))
    end = np.arange(nsta + nlta, n + 1)

    sta = (c[end] - c[end - nsta]) / nsta
    lta = (c[end - nsta] - c[end - nsta - nlta]) / nlta

    floor = 1e-6 * cf.mean() + 1e-30
    ratio[end - 1] = sta / np.maximum(lta, floor)
    return ratio


def backward_search(ratio, fs, trig_idx, cfg):
    a = max(0, trig_idx - int(cfg["peak_back_sec"] * fs))
    b = min(len(ratio), trig_idx + int(cfg["peak_fwd_sec"] * fs) + 1)
    if b <= a:
        return None

    k_peak = a + int(np.argmax(ratio[a:b]))
    if ratio[k_peak] < cfg["ratio_peak_min"]:
        return None

    # walk back from the peak while the ratio stays elevated
    gap_tol = int(cfg["gap_tol_sec"] * fs)
    onset = k_peak
    misses = 0
    k = k_peak
    while k >= 0:
        if ratio[k] >= cfg["ratio_low"]:
            onset = k
            misses = 0
        else:
            misses += 1
            if misses > gap_tol:
                break
        k -= 1

    return onset


def aic_pick(x):
    n = len(x)
    if n < 10:
        return None

    c1 = np.cumsum(x)
    c2 = np.cumsum(x * x)

    k = np.arange(2, n - 2)
    n2 = n - k

    m1 = c1[k - 1] / k
    v1 = c2[k - 1] / k - m1 ** 2
    m2 = (c1[-1] - c1[k - 1]) / n2
    v2 = (c2[-1] - c2[k - 1]) / n2 - m2 ** 2

    tiny = 1e-12 * np.var(x) + 1e-30
    aic = k * np.log(np.maximum(v1, tiny)) + n2 * np.log(np.maximum(v2, tiny))
    return int(k[np.argmin(aic)])


def aic_refine(wf, fs, k_back, cfg):
    a = max(0, k_back - int(cfg["aic_before_sec"] * fs))
    b = min(len(wf), k_back + int(cfg["aic_after_sec"] * fs))
    if b - a < 20:
        return None

    k = aic_pick(wf[a:b])
    if k is None:
        return None

    edge = max(2, int(0.05 * (b - a)))
    if k < edge or k >= (b - a) - edge:
        return None

    return a + k


def pick_p_wave(wf, fs, trig_rel, cfg):
    """Return P-pick sample index within wf, or None."""
    cf = np.abs(hilbert(wf))
    nsta = max(2, int(cfg["sta_sec"] * fs))
    nlta = int(cfg["lta_sec"] * fs)
    ratio = sta_lta(cf, nsta, nlta)

    k_back = backward_search(ratio, fs, trig_rel, cfg)
    if k_back is None:
        return None

    k_aic = aic_refine(wf, fs, k_back, cfg)
    if k_aic is not None and abs(k_aic - k_back) <= cfg["aic_max_shift_sec"] * fs:
        return k_aic
    return k_back


def simple_snr(wf, fs, k):
    """Signal RMS (1 s after pick) / noise RMS (5 s before pick)."""
    if k is None:
        return None

    noise = wf[max(0, k - int(5 * fs)):max(0, k - int(0.3 * fs))]
    sig = wf[k:k + int(1 * fs)]
    if len(noise) < int(fs) or len(sig) == 0:
        return None

    noise_rms = np.sqrt(np.mean(noise ** 2))
    if noise_rms <= 0:
        return None
    return float(np.sqrt(np.mean(sig ** 2)) / noise_rms)


# 9. NETWORK COINCIDENCE

def station_coordinates(cfg):
    return {"%s.%s" % (s["net"], s["sta"]): (s["lat"], s["lon"])
            for s in cfg["stations"]}


def compatible_picks(a, b, coords, cfg):
    ca, cb = coords.get(a["station"]), coords.get(b["station"])
    if ca is None or cb is None:
        return False

    dist_km = gps2dist_azimuth(ca[0], ca[1], cb[0], cb[1])[0] / 1000.0
    max_dt = dist_km / cfg["min_p_velocity_km_s"] + cfg["coincidence_tolerance_sec"]
    return abs(a["p_time"] - b["p_time"]) <= max_dt


def check_coincidence(ev, all_events, cfg):
    if ev["p_time"] is None:
        return

    coords = station_coordinates(cfg)

    candidates = [e for e in all_events
                  if e is not ev and e["p_time"] is not None
                  and abs(e["p_time"] - ev["p_time"]) <= cfg["coincidence_window_sec"]]
    candidates.sort(key=lambda e: abs(e["p_time"] - ev["p_time"]))

    cluster = [ev]
    used = {ev["station"]}

    for cand in candidates:
        if cand["station"] in used:
            continue
        if all(compatible_picks(cand, m, coords, cfg) for m in cluster):
            cluster.append(cand)
            used.add(cand["station"])

    if len(used) >= cfg["coincidence_min_stations"]:
        for m in cluster:
            m["coincident"] = True


# 10. OUTPUT

def fmt(t):
    return "-" if t is None else str(UTCDateTime(t))[:23]


def print_event(ev):
    snr = ev["snr"]
    print("-" * 60)
    print("STATION EVENT %s%s" % (
        ev["id"], "  [COINCIDENT]" if ev["coincident"] else ""))
    print("  trigger : %s" % fmt(ev["trigger_time"]))
    print("  P pick  : %s  snr=%s" % (
        fmt(ev["p_time"]), "-" if snr is None else "%.1f" % snr))
    print("  end     : %s (%s)" % (fmt(ev["end_time"]), ev["end_reason"]))


def save_event(ev, cfg):
    os.makedirs(cfg["event_dir"], exist_ok=True)

    net, sta, loc, cha = (ev["id"].split(".") + ["", "", "", ""])[:4]
    stamp = UTCDateTime(ev["trigger_time"]).strftime("%Y%m%dT%H%M%S")

    tr = Trace(
        data=ev["waveform"].astype(np.float32),
        header={
            "network": net, "station": sta, "location": loc, "channel": cha,
            "sampling_rate": ev["fs"],
            "starttime": UTCDateTime(ev["wf_start"]),
        },
    )
    tr.write(os.path.join(cfg["event_dir"], "%s_%s.mseed" % (ev["station"], stamp)),
             format="MSEED")

    new_file = not os.path.exists(cfg["log_csv"])
    with open(cfg["log_csv"], "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["id", "trigger", "p_time", "end", "snr", "coincident"])
        w.writerow([ev["id"], fmt(ev["trigger_time"]), fmt(ev["p_time"]),
                    fmt(ev["end_time"]), ev["snr"], ev["coincident"]])


def handle_event(ev, all_events, cfg):
    all_events.append(ev)
    check_coincidence(ev, all_events, cfg)
    print_event(ev)

    if cfg["save_events"]:
        save_event(ev, cfg)

    cutoff = ev["trigger_time"] - 3600
    all_events[:] = [e for e in all_events if e["trigger_time"] >= cutoff]


def print_status(states, dropped):
    for sid, st in states.items():
        print("[status] %-16s state=%-8s bg=%.3e" % (sid, st["state"], st["bg"]))
    print("[status] dropped records:", dropped["records"])


# 11. PROCESSING LOOP

def processing_loop(q, cfg, all_events, dropped):
    states = {}
    last_status = time.time()

    while True:
        try:
            tr = q.get(timeout=5)
        except queue.Empty:
            tr = "idle"

        if tr is None:
            break

        try:
            if tr != "idle":
                for ev in process_trace(states, tr, cfg):
                    handle_event(ev, all_events, cfg)

            if time.time() - last_status >= cfg["status_every_sec"]:
                last_status = time.time()
                if states:
                    print_status(states, dropped)
                else:
                    print("[status] no data received yet")

        except Exception as exc:
            print("Processing error:", exc)


# 12. MAIN

def main():
    print("Starting live earthquake detector ...", flush=True)

    q = queue.Queue(maxsize=CONFIG["queue_size"])
    all_events = []
    dropped = {"records": 0}
    stop = threading.Event()

    worker = threading.Thread(
        target=processing_loop,
        args=(q, CONFIG, all_events, dropped),
        daemon=True,
    )
    worker.start()

    try:
        acquisition_loop(q, CONFIG, stop, dropped)
    except KeyboardInterrupt:
        print("Stopping ...")
    finally:
        stop.set()
        q.put(None)
        worker.join(timeout=10)


if __name__ == "__main__":
    main()