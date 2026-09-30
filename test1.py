from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy.signal import hilbert
from scipy.optimize import root
from obspy.geodetics import gps2dist_azimuth


client = Client("https://seiscomp.alertnepal.online")

stations = [
    {"sta": "KKN", "lat": 27.8000,"lon": 85.2790,"net": "NK","cha": "BHZ","loc": "*"},
    {"sta": "EQM08","lat": 27.831,"lon": 86.65,"net": "NP","cha": "EHZ","loc": "*",},
    {"sta": "EQM10", "lat": 28.299517, "lon": 83.960148, "net": "NP", "cha": "EHZ", "loc": "*"},
]

start = UTCDateTime("2026-09-22T07:45:00")
end = UTCDateTime("2026-09-22T07:55:00")

fs = 20.0
fmin = 1.0
fmax = 8.0

# P-wave velocity in km/s
P_VELOCITY = 6.5


def preprocess(station, start, end):

    stream = client.get_waveforms(network=station["net"],station=station["sta"],location=station["loc"],channel=station["cha"],starttime=start,endtime=end)

    if len(stream) == 0:
        raise RuntimeError("No waveform found")

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
        try:
            tr = preprocess(st_info, start, end)
            traces[name] = tr
        except Exception as e:
            print(f"Skipping {name}: {e}")
    return traces

def waveform_plot(traces):

    fig, axes = plt.subplots(len(traces),1,figsize=(12, 8))
    axes = np.atleast_1d(axes)
    for ax, (name, trace) in zip(axes, traces.items()):
        data = trace.data.astype(float)

        time = ( np.arange(len(data)) / trace.stats.sampling_rate)
        ax.plot(time,data,linewidth=0.8)
        ax.set_ylabel(name)
        ax.grid(True)

    axes[-1].set_xlabel("Time from start (s)")
    fig.tight_layout()
    plt.show()

def spectogram(traces):
    fig, axes = plt.subplots(len(traces),1,figsize=(12, 8))
    axes = np.atleast_1d(axes)
    for ax, (name, trace) in zip(axes,traces.items()):
        data = trace.data.astype(float)
        nperseg = min(512,len(data))
        noverlap = int(nperseg * 0.75)
        ax.specgram(data,NFFT=nperseg,Fs=trace.stats.sampling_rate,noverlap=noverlap)

        ax.set_ylabel(f"{name}\nFrequency (Hz)")
        ax.grid(True)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def pick_p(
    trace,
    sta=1.0,
    lta=15.0,
    on=3.5,
    off=1.5,
    skip=None
):

    fs = trace.stats.sampling_rate

    nsta = int(sta * fs)
    nlta = int(lta * fs)

    cft = classic_sta_lta(trace.data,nsta,nlta)
    if skip is None:
        skip = (lta+ 0.05 * len(trace.data) / fs)
    cft[:int(skip * fs)] = 0

    onsets = trigger_onset(cft,on,off)
    if len(onsets) == 0:
        return None, cft

    best = max(
        onsets,
        key=lambda o:
        cft[o[0]:o[1] + 1].max()
    )

    return (best[0] / fs,cft )


def pick_s(
    trace,
    p_time,
    sta=1.0,
    lta=8.0,
    min_gap=6.0,
    max_gap=60.0,
    thr=2.0
):

    fs = trace.stats.sampling_rate

    env = np.abs(
        hilbert(
            trace.data.astype(float)
        )
    )

    cft = classic_sta_lta(
        env,
        int(sta * fs),
        int(lta * fs)
    )

    i0 = int(
        (p_time + min_gap) * fs
    )

    i1 = min(
        int((p_time + max_gap) * fs),
        len(cft)
    )

    if i0 >= i1:

        return None, cft

    ipk = (
        i0
        + np.argmax(
            cft[i0:i1]
        )
    )

    if cft[ipk] < thr:

        return None, cft

    level = (
        1.0
        + 0.3 * (cft[ipk] - 1.0)
    )

    i = ipk

    while (
        i > i0
        and cft[i] > level
    ):

        i -= 1

    return (
        i / fs,
        cft
    )


def pick_all(traces):

    picks = {}

    for name, tr in traces.items():

        tp, cft_p = pick_p(tr)

        ts = None
        cft_s = None

        if tp is not None:

            ts, cft_s = pick_s(
                tr,
                tp
            )

        picks[name] = {
            "tp": tp,
            "ts": ts,
            "cft_p": cft_p,
            "cft_s": cft_s
        }

        msg = f"{name}: "

        if tp is not None:

            p_absolute = (
                tr.stats.starttime
                + tp
            )

            msg += (
                f"P = {p_absolute}"
            )

        else:

            msg += "P not found"

        if ts is not None:

            msg += (
                f" | S = "
                f"{tr.stats.starttime + ts}"
                f" | S-P = "
                f"{ts - tp:.2f} s"
            )

        print(msg)

    return picks


def picks_plot(traces, picks):

    fig, axes = plt.subplots(
        len(traces),
        2,
        figsize=(
            14,
            3 * len(traces)
        ),
        sharex="col"
    )

    axes = np.atleast_2d(axes)

    for row, (name, tr) in zip(
        axes,
        traces.items()
    ):

        t = (
            np.arange(
                tr.stats.npts
            )
            / tr.stats.sampling_rate
        )

        pk = picks[name]

        row[0].plot(
            t,
            tr.data,
            lw=0.7,
            color="k"
        )

        row[0].set_ylabel(name)

        row[1].plot(
            t,
            pk["cft_p"],
            lw=0.8,
            label="P STA/LTA"
        )

        if pk["cft_s"] is not None:

            row[1].plot(
                t,
                pk["cft_s"],
                lw=0.8,
                label="S STA/LTA (env)"
            )

        for ax in row:

            if pk["tp"] is not None:

                ax.axvline(
                    pk["tp"],
                    color="r",
                    label="P"
                )

            if pk["ts"] is not None:

                ax.axvline(
                    pk["ts"],
                    color="b",
                    label="S"
                )

            ax.grid(True)

        row[1].legend(
            loc="upper right",
            fontsize=7
        )

    axes[-1, 0].set_xlabel(
        "Time from start (s)"
    )

    axes[-1, 1].set_xlabel(
        "Time from start (s)"
    )

    fig.tight_layout()

    plt.show()


def distance_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    distance_m, _, _ = gps2dist_azimuth(
        lat1,
        lon1,
        lat2,
        lon2
    )

    return distance_m / 1000.0


def get_p_arrivals(
    traces,
    picks
):

    arrivals = {}

    for name, pk in picks.items():

        if pk["tp"] is None:
            continue

        tr = traces[name]

        arrivals[name] = (
            tr.stats.starttime
            + pk["tp"]
        )

    return arrivals


def find_hyperbola_intersection(
    stations,
    arrivals,
    velocity
):

    if len(arrivals) < 3:

        raise RuntimeError(
            "At least 3 P-wave arrivals "
            "are required."
        )

    station_dict = {}

    for st in stations:

        name = (
            f"{st['net']}."
            f"{st['sta']}"
        )

        if name in arrivals:

            station_dict[name] = st

    if len(station_dict) < 3:

        raise RuntimeError(
            "Three stations with P picks "
            "are required."
        )

    # Earliest station is the reference
    reference = min(
        arrivals,
        key=lambda x: arrivals[x]
    )

    other_stations = [
        name
        for name in station_dict
        if name != reference
    ]

    station2 = other_stations[0]
    station3 = other_stations[1]

    t1 = arrivals[reference]
    t2 = arrivals[station2]
    t3 = arrivals[station3]

    # Observed TDOA
    dt21 = t2 - t1
    dt31 = t3 - t1

    # Convert TDOA to distance differences
    distance_difference_21 = (
        velocity * dt21
    )

    distance_difference_31 = (
        velocity * dt31
    )

    print()
    print("TDOA hyperbola equations:")
    print()
    print(
        f"{station2} - {reference}: "
        f"{dt21:.3f} s"
    )

    print(
        f"Distance difference: "
        f"{distance_difference_21:.3f} km"
    )

    print()

    print(
        f"{station3} - {reference}: "
        f"{dt31:.3f} s"
    )

    print(
        f"Distance difference: "
        f"{distance_difference_31:.3f} km"
    )

    ref = station_dict[reference]
    st2 = station_dict[station2]
    st3 = station_dict[station3]

    def equations(x):

        lat = x[0]
        lon = x[1]

        d1 = distance_km(
            lat,
            lon,
            ref["lat"],
            ref["lon"]
        )

        d2 = distance_km(
            lat,
            lon,
            st2["lat"],
            st2["lon"]
        )

        d3 = distance_km(
            lat,
            lon,
            st3["lat"],
            st3["lon"]
        )

        # Hyperbola equation 1
        eq1 = (
            (d2 - d1)
            - distance_difference_21
        )

        # Hyperbola equation 2
        eq2 = (
            (d3 - d1)
            - distance_difference_31
        )

        return [
            eq1,
            eq2
        ]

    # Initial guess = center of stations
    initial_lat = np.mean([
        ref["lat"],
        st2["lat"],
        st3["lat"]
    ])

    initial_lon = np.mean([
        ref["lon"],
        st2["lon"],
        st3["lon"]
    ])

    solution = root(
        equations,
        [
            initial_lat,
            initial_lon
        ]
    )

    if not solution.success:

        raise RuntimeError(
            "Hyperbola intersection solver failed: "
            + solution.message
        )

    epicenter_lat = solution.x[0]
    epicenter_lon = solution.x[1]

    residuals = equations(
        solution.x
    )

    print()
    print("Hyperbola intersection result:")
    print(
        f"Latitude  : "
        f"{epicenter_lat:.6f}"
    )

    print(
        f"Longitude : "
        f"{epicenter_lon:.6f}"
    )

    print()
    print("Hyperbola residuals:")

    print(
        f"{station2}: "
        f"{residuals[0]:+.6f} km"
    )

    print(
        f"{station3}: "
        f"{residuals[1]:+.6f} km"
    )

    print()
    print(
        f"Reference station: "
        f"{reference}"
    )

    print(
        f"P-wave velocity: "
        f"{velocity:.2f} km/s"
    )

    return {
        "lat": epicenter_lat,
        "lon": epicenter_lon,
        "reference": reference,
        "station2": station2,
        "station3": station3,
        "dt21": dt21,
        "dt31": dt31,
        "dd21": distance_difference_21,
        "dd31": distance_difference_31,
        "residuals": residuals,
        "velocity": velocity
    }


def plot_hyperbola_intersection(
    stations,
    result
):

    ref_name = result["reference"]
    st2_name = result["station2"]
    st3_name = result["station3"]

    station_dict = {}

    for st in stations:

        name = (
            f"{st['net']}."
            f"{st['sta']}"
        )

        station_dict[name] = st

    ref = station_dict[ref_name]
    st2 = station_dict[st2_name]
    st3 = station_dict[st3_name]

    epicenter_lat = result["lat"]
    epicenter_lon = result["lon"]

    # Plot region
    all_lats = [
        st["lat"]
        for st in stations
    ]

    all_lons = [
        st["lon"]
        for st in stations
    ]

    lat_min = min(all_lats + [epicenter_lat]) - 1.0
    lat_max = max(all_lats + [epicenter_lat]) + 1.0

    lon_min = min(all_lons + [epicenter_lon]) - 1.0
    lon_max = max(all_lons + [epicenter_lon]) + 1.0

    lat_values = np.linspace(
        lat_min,
        lat_max,
        400
    )

    lon_values = np.linspace(
        lon_min,
        lon_max,
        400
    )

    lon_grid, lat_grid = np.meshgrid(
        lon_values,
        lat_values
    )

    d_ref = np.zeros_like(
        lat_grid
    )

    d2 = np.zeros_like(
        lat_grid
    )

    d3 = np.zeros_like(
        lat_grid
    )

    for i in range(
        lat_grid.shape[0]
    ):

        for j in range(
            lat_grid.shape[1]
        ):

            d_ref[i, j] = distance_km(
                lat_grid[i, j],
                lon_grid[i, j],
                ref["lat"],
                ref["lon"]
            )

            d2[i, j] = distance_km(
                lat_grid[i, j],
                lon_grid[i, j],
                st2["lat"],
                st2["lon"]
            )

            d3[i, j] = distance_km(
                lat_grid[i, j],
                lon_grid[i, j],
                st3["lat"],
                st3["lon"]
            )

    equation1 = (
        d2
        - d_ref
        - result["dd21"]
    )

    equation2 = (
        d3
        - d_ref
        - result["dd31"]
    )

    plt.figure(
        figsize=(10, 8)
    )

    # First hyperbola
    plt.contour(
        lon_grid,
        lat_grid,
        equation1,
        levels=[0],
        linewidths=2,
        label=f"{st2_name} - {ref_name}"
    )

    # Second hyperbola
    plt.contour(
        lon_grid,
        lat_grid,
        equation2,
        levels=[0],
        linewidths=2
    )

    # Stations
    for st in stations:

        name = (
            f"{st['net']}."
            f"{st['sta']}"
        )

        plt.scatter(
            st["lon"],
            st["lat"],
            marker="^",
            s=100
        )

        plt.text(
            st["lon"] + 0.05,
            st["lat"] + 0.05,
            name
        )

    # Intersection
    plt.scatter(
        epicenter_lon,
        epicenter_lat,
        marker="*",
        s=250,
        label="Hyperbola intersection"
    )

    plt.text(
        epicenter_lon + 0.05,
        epicenter_lat + 0.05,
        (
            f"Epicenter\n"
            f"{epicenter_lat:.4f}, "
            f"{epicenter_lon:.4f}"
        )
    )

    plt.xlabel(
        "Longitude (°E)"
    )

    plt.ylabel(
        "Latitude (°N)"
    )

    plt.title(
        "TDOA Hyperbola Intersection"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.show()


def main():

    print(
        "Fetching waveforms..."
    )

    traces = fetch_all_traces(
        stations,
        start,
        end
    )

    if len(traces) < 3:

        raise RuntimeError(
            "Could not obtain all three waveforms."
        )

    waveform_plot(
        traces
    )

    spectogram(
        traces
    )

    print()
    print(
        "Detecting P and S arrivals..."
    )

    picks = pick_all(
        traces
    )

    picks_plot(
        traces,
        picks
    )

    print()
    print(
        "Getting P-wave arrival times..."
    )

    arrivals = get_p_arrivals(
        traces,
        picks
    )

    if len(arrivals) < 3:

        print(
            "Triangulation cannot be performed."
        )

        print(
            "P picks found at:"
        )

        for name in arrivals:

            print(name)

        return

    result = find_hyperbola_intersection(
        stations,
        arrivals,
        velocity=P_VELOCITY
    )

    plot_hyperbola_intersection(
        stations,
        result
    )


if __name__ == "__main__":

    main()