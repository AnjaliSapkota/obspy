import base64
import io
import threading
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from obspy import Stream, UTCDateTime
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
from scipy.signal import butter, correlate, correlation_lags, filtfilt, hilbert, windows
from scipy.optimize import least_squares
from pyproj import Transformer

server = "ring.wscada.net:18000"

window = 60
RUN_EVERY = 10

target_fs = 20.0
freqmin = 1.0
freqmax = 8.0
smooth_cutoff = 0.5

shift = 30.0
velocity = 3.0


transformer_to_utm = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32645",
    always_xy=True
)

transformer_to_wgs = Transformer.from_crs(
    "EPSG:32645",
    "EPSG:4326",
    always_xy=True
)


class RingBuffer:

    def __init__(self, keep_seconds=900):

        self.keep_seconds = keep_seconds

        self.stream = Stream()

        self.lock = threading.Lock()

    def add(self, trace):

        if trace is None:
            return

        if trace.stats.npts == 0:
            return

        with self.lock:

            self.stream.append(trace)

    def snapshot(self, station):

        with self.lock:

            return self.stream.select(
                network=station["net"],
                station=station["sta"],
                channel=station["cha"]
            ).copy()

    def latest_end(self, station):

        stream = self.snapshot(station)

        if len(stream) == 0:
            return None

        return max(
            trace.stats.endtime
            for trace in stream
        )

    def prune(self):

        with self.lock:

            if len(self.stream) == 0:
                return

            newest = max(
                trace.stats.endtime
                for trace in self.stream
            )

            cutoff = newest - self.keep_seconds

            self.stream = Stream(
                trace.copy()
                for trace in self.stream
                if trace.stats.endtime >= cutoff
            )


class SeedLinkListener(EasySeedLinkClient):

    def __init__(self, server, buffer):

        super().__init__(
            server,
            autoconnect=False
        )

        self.buffer = buffer

        self.conn.timeout = 10

    def on_data(self, trace):

        self.buffer.add(trace)


def start_seedlink(stations, buffer):

    def worker():

        while True:

            client = None

            try:

                print(
                    f"Connecting to SeedLink: "
                    f"{server}"
                )

                client = SeedLinkListener(server, buffer)

                client.connect()

                print(
                    "SeedLink connected."
                )

                for station in stations:

                    print(
                        f"Selecting: "
                        f"{station['net']}."
                        f"{station['sta']}."
                        f"{station['cha']}"
                    )

                    client.select_stream(
                        station["net"],
                        station["sta"],
                        station["cha"]
                    )

                print(
                    "SeedLink streaming..."
                )

                client.run()

            except Exception as e:

                print(
                    f"SeedLink error: {e}"
                )

            finally:

                if client is not None:

                    try:
                        client.close()

                    except Exception:
                        pass

            print(
                "Reconnecting in 10 seconds..."
            )

            time.sleep(10)

    thread = threading.Thread(
        target=worker,
        daemon=True
    )

    thread.start()


def fetch_waveform(
    buffer,
    station,
    start,
    end,
    target_fs,
    freqmin,
    freqmax
):

    stream = buffer.snapshot(
        station
    )

    if len(stream) == 0:

        raise RuntimeError(
            f"No live data for "
            f"{station['sta']}."
        )

    stream.merge(
        method=1,
        fill_value="interpolate"
    )

    stream.trim(
        starttime=start,
        endtime=end
    )

    if len(stream) == 0:

        raise RuntimeError(
            f"No data in window for "
            f"{station['sta']}."
        )

    if stream[0].stats.npts < 10:

        raise RuntimeError(
            f"Not enough data for "
            f"{station['sta']}."
        )

    stream.detrend(
        "linear"
    )

    stream.detrend(
        "demean"
    )

    stream.taper(
        max_percentage=0.05,
        type="hann"
    )

    stream.interpolate(
        sampling_rate=target_fs,
        method="linear"
    )

    stream.filter(
        "bandpass",
        freqmin=freqmin,
        freqmax=freqmax,
        corners=4,
        zerophase=True
    )

    return stream[0]


def trim_to_common_window(traces):

    common_start = max(
        trace.stats.starttime
        for trace in traces.values()
    )

    common_end = min(
        trace.stats.endtime
        for trace in traces.values()
    )

    if common_start >= common_end:

        raise RuntimeError(
            "No common time window exists "
            "between stations."
        )

    for trace in traces.values():

        trace.trim(
            starttime=common_start,
            endtime=common_end
        )


def image_to_base64(fig):

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=140,
        bbox_inches="tight"
    )

    plt.close(fig)

    return base64.b64encode(
        buffer.getvalue()
    ).decode()


def create_waveform_plot(traces):

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 8),
        sharex=True
    )

    for ax, (name, trace) in zip(
        axes,
        traces.items()
    ):

        sampling_rate = (
            trace.stats.sampling_rate
        )

        data = trace.data.astype(float)

        time_axis = (
            np.arange(len(data))
            / sampling_rate
        )

        ax.plot(
            time_axis,
            data,
            linewidth=0.8
        )

        ax.set_ylabel(name)

        ax.grid(
            True,
            alpha=0.3
        )

    axes[-1].set_xlabel(
        "Time since common start (seconds)"
    )

    fig.suptitle(
        "Three-Station Seismic Waveforms"
    )

    fig.tight_layout()

    return image_to_base64(fig)


def create_spectrogram_plot(
    traces,
    fmin=1.0,
    fmax=8.0
):

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        sharex=True
    )

    for ax, (name, trace) in zip(
        axes,
        traces.items()
    ):

        sampling_rate = (
            trace.stats.sampling_rate
        )

        data = trace.data.astype(float)

        nperseg = min(
            512,
            len(data)
        )

        if nperseg < 2:

            raise RuntimeError(
                f"Not enough samples for "
                f"spectrogram at {name}."
            )

        noverlap = int(
            nperseg * 0.75
        )

        ax.specgram(
            data,
            NFFT=nperseg,
            Fs=sampling_rate,
            noverlap=noverlap
        )

        ax.set_ylabel(
            f"{name}\nFrequency (Hz)"
        )

        ax.set_ylim(
            fmin,
            fmax
        )

        ax.grid(
            True,
            alpha=0.2
        )

    axes[-1].set_xlabel(
        "Time since common start (seconds)"
    )

    fig.suptitle(
        f"Three-Station Frequency Spectrogram "
        f"({fmin:g}–{fmax:g} Hz)"
    )

    fig.tight_layout()

    return image_to_base64(fig)


def get_hilbert_envelope(
    trace,
    smooth_cutoff=0.5
):

    signal = trace.data.astype(float)

    signal = (
        signal
        - np.mean(signal)
    )

    analytic_signal = hilbert(
        signal
    )

    envelope = np.abs(
        analytic_signal
    )

    if smooth_cutoff is not None:

        fs = trace.stats.sampling_rate

        b, a = butter(
            N=2,
            Wn=smooth_cutoff
            / (0.5 * fs),
            btype="low"
        )

        envelope = filtfilt(
            b,
            a,
            envelope
        )

        envelope = np.maximum(
            envelope,
            0
        )

    return envelope


def calculate_envelope_lag(
    envelope_a,
    envelope_b,
    sampling_rate,
    max_shift_seconds
):

    min_len = min(
        len(envelope_a),
        len(envelope_b)
    )

    envelope_a = envelope_a[
        :min_len
    ]

    envelope_b = envelope_b[
        :min_len
    ]

    crop = int(
        0.05 * min_len
    )

    if crop > 0:

        envelope_a = envelope_a[
            crop:-crop
        ]

        envelope_b = envelope_b[
            crop:-crop
        ]

    envelope_a = (
        envelope_a
        - np.mean(envelope_a)
    )

    envelope_b = (
        envelope_b
        - np.mean(envelope_b)
    )

    taper = windows.tukey(
        len(envelope_a),
        alpha=0.05
    )

    a = (
        envelope_a
        * taper
    )

    b = (
        envelope_b
        * taper
    )

    correlation = correlate(
        a,
        b,
        mode="full"
    )

    lags = correlation_lags(
        len(a),
        len(b),
        mode="full"
    )

    norm_factor = np.sqrt(
        np.sum(a ** 2)
        *
        np.sum(b ** 2)
    )

    if norm_factor == 0:

        raise RuntimeError(
            "Envelope has zero energy."
        )

    correlation = (
        correlation
        / norm_factor
    )

    lag_seconds = (
        lags
        / sampling_rate
    )

    max_shift_samples = int(
        round(
            max_shift_seconds
            * sampling_rate
        )
    )

    valid = (
        (lags >= -max_shift_samples)
        &
        (lags <= max_shift_samples)
    )

    valid_corr = correlation[
        valid
    ]

    valid_lags = lags[
        valid
    ]

    if len(valid_corr) == 0:

        raise RuntimeError(
            "No valid lag values found."
        )

    peak_index = np.argmax(
        valid_corr
    )

    lag_samples = (
        valid_lags[
            peak_index
        ]
    )

    lag_seconds_value = (
        lag_samples
        / sampling_rate
    )

    max_correlation = (
        valid_corr[
            peak_index
        ]
    )

    return {

        "lag": float(
            lag_seconds_value
        ),

        "cc_max": float(
            max_correlation
        ),

        "lags_full": (
            lag_seconds.tolist()
        ),

        "cc_full": (
            correlation.tolist()
        )
    }


def latlon_to_xy(
    origin_lat,
    origin_lon,
    lat,
    lon
):

    origin_x, origin_y = (
        transformer_to_utm.transform(
            origin_lon,
            origin_lat
        )
    )

    target_x, target_y = (
        transformer_to_utm.transform(
            lon,
            lat
        )
    )

    return (
        (target_x - origin_x)
        / 1000.0,

        (target_y - origin_y)
        / 1000.0
    )


def xy_to_latlon(
    origin_lat,
    origin_lon,
    x_km,
    y_km
):

    origin_x, origin_y = (
        transformer_to_utm.transform(
            origin_lon,
            origin_lat
        )
    )

    target_x = (
        origin_x
        + x_km * 1000.0
    )

    target_y = (
        origin_y
        + y_km * 1000.0
    )

    lon, lat = (
        transformer_to_wgs.transform(
            target_x,
            target_y
        )
    )

    return lat, lon


def tdoa_residuals(
    xy,
    station_xy,
    delta,
    station_a,
    station_b,
    station_c
):

    x, y = xy

    x_a, y_a = (
        station_xy[station_a]
    )

    x_b, y_b = (
        station_xy[station_b]
    )

    x_c, y_c = (
        station_xy[station_c]
    )

    d_a = np.sqrt(
        (x - x_a) ** 2
        +
        (y - y_a) ** 2
    )

    d_b = np.sqrt(
        (x - x_b) ** 2
        +
        (y - y_b) ** 2
    )

    d_c = np.sqrt(
        (x - x_c) ** 2
        +
        (y - y_c) ** 2
    )

    res_1 = (
        (d_b - d_a)
        -
        delta[
            f"{station_a}-{station_b}"
        ]
    )

    res_2 = (
        (d_c - d_a)
        -
        delta[
            f"{station_a}-{station_c}"
        ]
    )

    res_3 = (
        (d_c - d_b)
        -
        delta[
            f"{station_b}-{station_c}"
        ]
    )

    return [
        res_1,
        res_2,
        res_3
    ]


def find_source_location(
    station_xy,
    delta,
    station_a,
    station_b,
    station_c
):

    initial_guess = [
        0.0,
        0.0
    ]

    xs = [
        xy[0]
        for xy in station_xy.values()
    ]

    ys = [
        xy[1]
        for xy in station_xy.values()
    ]

    min_x = min(xs)
    max_x = max(xs)

    min_y = min(ys)
    max_y = max(ys)

    margin = 300.0

    lower_bounds = [
        min_x - margin,
        min_y - margin
    ]

    upper_bounds = [
        max_x + margin,
        max_y + margin
    ]

    result = least_squares(
        tdoa_residuals,
        initial_guess,
        bounds=(
            lower_bounds,
            upper_bounds
        ),
        args=(
            station_xy,
            delta,
            station_a,
            station_b,
            station_c
        )
    )

    x_src, y_src = result.x

    residual_norm = np.linalg.norm(
        tdoa_residuals(
            result.x,
            station_xy,
            delta,
            station_a,
            station_b,
            station_c
        )
    )

    return (
        float(x_src),
        float(y_src),
        float(residual_norm)
    )


def create_hyperbola_plot(station_xy,delta,source_xy):
    if not np.all(np.isfinite(source_xy)):
        raise RuntimeError(
            "Invalid source coordinates."
    )

    xs = [
        xy[0]
        for xy in station_xy.values()
    ]

    ys = [
        xy[1]
        for xy in station_xy.values()
    ]

    all_x = (
        xs
        +
        [source_xy[0]]
    )

    all_y = (
        ys
        +
        [source_xy[1]]
    )

    margin = (
        max(
            max(all_x) - min(all_x),
            max(all_y) - min(all_y)
        )
        * 0.2
        +
        50.0
    )

    xmin = (
        min(all_x)
        - margin
    )

    xmax = (
        max(all_x)
        + margin
    )

    ymin = (
        min(all_y)
        - margin
    )

    ymax = (
        max(all_y)
        + margin
    )

    resolution = 0.5

    x = np.arange(
        xmin,
        xmax,
        resolution
    )

    y = np.arange(
        ymin,
        ymax,
        resolution
    )

    X, Y = np.meshgrid(
        x,
        y
    )

    D = {}

    for name, (
        station_x,
        station_y
    ) in station_xy.items():

        D[name] = np.sqrt(
            (X - station_x) ** 2
            +
            (Y - station_y) ** 2
        )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    colors = [
        "blue",
        "red",
        "green"
    ]

    legend_lines = []

    for i, (
        pair,
        distance_difference
    ) in enumerate(
        delta.items()
    ):

        station_a, station_b = (
            pair.split("-")
        )

        H = (
            D[station_b]
            -
            D[station_a]
            -
            distance_difference
        )

        ax.contour(
            X,
            Y,
            H,
            levels=[0],
            colors=colors[i],
            linewidths=2.5
        )

        legend_lines.append(
            Line2D(
                [0],
                [0],
                color=colors[i],
                linewidth=2.5,
                label=(
                    f"{station_a}–"
                    f"{station_b} hyperbola"
                )
            )
        )

    for name, (
        station_x,
        station_y
    ) in station_xy.items():

        ax.scatter(
            station_x,
            station_y,
            marker=".",
            s=160,
            zorder=5
        )

        ax.text(
            station_x + 5,
            station_y + 5,
            name,
            fontsize=12,
            fontweight="bold"
        )

    legend_lines.append(
        Line2D(
            [0],
            [0],
            marker=".",
            color="black",
            linestyle="None",
            markersize=10,
            label="Station"
        )
    )

    source_x, source_y = (
        source_xy
    )

    ax.scatter(
        source_x,
        source_y,
        marker="*",
        s=300,
        zorder=10
    )

    legend_lines.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            linestyle="None",
            markersize=15,
            label="Least-squares source"
        )
    )

    ax.legend(
        handles=legend_lines,
        loc="best"
    )

    ax.set_xlabel(
        "East-West distance from origin station (km)"
    )

    ax.set_ylabel(
        "North-South distance from origin station (km)"
    )

    ax.set_title(
        "Three-Station TDOA Hyperbola Intersection"
    )

    ax.grid(True)

    ax.axis("equal")

    fig.tight_layout()

    return image_to_base64(fig)


def run_triangulation(
    stations,
    buffer,
    start,
    end,
    fs=20.0,
    fmin=1.0,
    fmax=8.0,
    smooth_cutoff=0.5,
    max_shift_seconds=50.0,
    velocity=3
):

    if len(stations) != 3:

        raise ValueError(
            "This method needs exactly 3 stations."
        )

    start = UTCDateTime(
        start
    )

    end = UTCDateTime(
        end
    )

    print(
        "Fetching live waveforms"
    )

    traces = {}

    for station in stations:

        print(
            f"Reading {station['sta']}"
        )

        traces[
            station["sta"]
        ] = fetch_waveform(
            buffer,
            station,
            start,
            end,
            fs,
            fmin,
            fmax
        )

    print(
        "Trimming to common window"
    )

    trim_to_common_window(
        traces
    )

    print(
        "Creating waveform plots"
    )

    waveform_plot = (
        create_waveform_plot(
            traces
        )
    )

    spectrogram_plot = (
        create_spectrogram_plot(
            traces,
            fmin=fmin,
            fmax=fmax
        )
    )

    print(
        "Calculating Hilbert envelopes"
    )

    envelopes = {}

    for name, trace in (
        traces.items()
    ):

        envelopes[name] = (
            get_hilbert_envelope(
                trace,
                smooth_cutoff
            )
        )

    station_a = (
        stations[0]["sta"]
    )

    station_b = (
        stations[1]["sta"]
    )

    station_c = (
        stations[2]["sta"]
    )

    print(
        "Calculating TDOA"
    )

    lag_ab = calculate_envelope_lag(
        envelopes[station_a],
        envelopes[station_b],
        fs,
        max_shift_seconds
    )

    lag_ac = calculate_envelope_lag(
        envelopes[station_a],
        envelopes[station_c],
        fs,
        max_shift_seconds
    )

    lag_bc = calculate_envelope_lag(
        envelopes[station_b],
        envelopes[station_c],
        fs,
        max_shift_seconds
    )

    closure_error = (
        lag_ac["lag"]
        -
        (
            lag_ab["lag"]
            +
            lag_bc["lag"]
        )
    )

    print(
        "\nHilbert Envelope TDOA results:"
    )

    print(
        f"{station_a} - {station_b}: "
        f"lag = {lag_ab['lag']:+.3f} s, "
        f"CC = {lag_ab['cc_max']:.3f}"
    )

    print(
        f"{station_a} - {station_c}: "
        f"lag = {lag_ac['lag']:+.3f} s, "
        f"CC = {lag_ac['cc_max']:.3f}"
    )

    print(
        f"{station_b} - {station_c}: "
        f"lag = {lag_bc['lag']:+.3f} s, "
        f"CC = {lag_bc['cc_max']:.3f}"
    )

    print(
        f"\nTDOA closure error: "
        f"{closure_error:+.3f} s"
    )

    delta_ab = (
        velocity
        *
        (-lag_ab["lag"])
    )

    delta_ac = (
        velocity
        *
        (-lag_ac["lag"])
    )

    delta_bc = (
        velocity
        *
        (-lag_bc["lag"])
    )

    delta = {

        f"{station_a}-{station_b}":
            delta_ab,

        f"{station_a}-{station_c}":
            delta_ac,

        f"{station_b}-{station_c}":
            delta_bc
    }

    origin = stations[0]

    origin_lat = (
        origin["lat"]
    )

    origin_lon = (
        origin["lon"]
    )

    station_xy = {}

    for station in stations:

        station_xy[
            station["sta"]
        ] = latlon_to_xy(
            origin_lat,
            origin_lon,
            station["lat"],
            station["lon"]
        )

    print(
        "\nCalculating source location"
    )

    x_src, y_src, residual_norm = (
        find_source_location(
            station_xy,
            delta,
            station_a,
            station_b,
            station_c
        )
    )

    print(
        f"X (East) : "
        f"{x_src:+.3f} km"
    )

    print(
        f"Y (North): "
        f"{y_src:+.3f} km"
    )

    print(
        f"Residual norm: "
        f"{residual_norm:.4f} km"
    )

    source_lat, source_lon = (
        xy_to_latlon(
            origin_lat,
            origin_lon,
            x_src,
            y_src
        )
    )

    print(
        "\nLeast-squares source geographic location:"
    )

    print(
        f"Latitude  : "
        f"{source_lat:.6f}"
    )

    print(
        f"Longitude : "
        f"{source_lon:.6f}"
    )

    print(
        "Creating hyperbola plot"
    )

    plot_png = (
        create_hyperbola_plot(
            station_xy,
            delta,
            (x_src, y_src)
        )
    )

    pairs = [

        {
            "a": station_a,
            "b": station_b,
            **lag_ab
        },

        {
            "a": station_a,
            "b": station_c,
            **lag_ac
        },

        {
            "a": station_b,
            "b": station_c,
            **lag_bc
        }

    ]

    station_coordinates = {}

    for station in stations:

        station_coordinates[
            station["sta"]
        ] = {

            "lat": station["lat"],

            "lon": station["lon"]
        }

    return {

        "pairs": pairs,

        "closure_error":
            float(closure_error),

        "source": {

            "lat":
                float(source_lat),

            "lon":
                float(source_lon),

            "x_km":
                float(x_src),

            "y_km":
                float(y_src),

            "residual_norm":
                float(residual_norm),

            "velocity":
                float(velocity)
        },

        "stations":
            station_coordinates,

        "waveform_plot":
            waveform_plot,

        "spectrogram_plot":
            spectrogram_plot,

        "plot_png":
            plot_png,

        "settings": {

            "start":
                str(start),

            "end":
                str(end),

            "sampling_rate":
                float(fs),

            "freqmin":
                float(fmin),

            "freqmax":
                float(fmax),

            "smooth_cutoff":
                float(smooth_cutoff),

            "max_shift_seconds":
                float(max_shift_seconds),

            "velocity":
                float(velocity)
        }
    }


if __name__ == "__main__":

    stations = [

        {
            "sta": "EQM07",
            "lat": 27.814941,
            "lon": 86.713401,
            "net": "NP",
            "cha": "EHZ"
        },

        {
            "sta": "EQM08",
            "lat": 27.831,
            "lon": 86.65,
            "net": "NP",
            "cha": "EHZ"
        },

        {
            "sta": "EQM10",
            "lat": 28.299517,
            "lon": 83.960148,
            "net": "NP",
            "cha": "EHZ"
        }

    ]

    buffer = RingBuffer(
        keep_seconds=900
    )

    start_seedlink(
        stations,
        buffer
    )

    print("Waiting for live data")

    while True:
        time.sleep(RUN_EVERY)

        buffer.prune()

        ends = []

        for station in stations:

            end = buffer.latest_end(
                station
            )

            ends.append(end)

        if any(
            end is None
            for end in ends
        ):

            print(
                "Waiting for all stations..."
            )

            continue

        end = min(
            ends
        )

        start = (
            end
            - window
        )

        print(
            f"\nAnalysis window:"
            f"\n{start}"
            f" -> "
            f"{end}"
        )

        ready = True

        for station in stations:

            stream = buffer.snapshot(
                station
            )

            if len(stream) == 0:

                ready = False

                print(
                    f"{station['sta']}: "
                    f"no data"
                )

                continue

            stream.merge(
                method=1,
                fill_value="interpolate"
            )

            available = (
                stream[0].stats.endtime
                -
                stream[0].stats.starttime
            )

            print(
                f"{station['sta']}: "
                f"{available:.1f}s available"
            )

            if available < window:

                ready = False

        if not ready:

            print(
                "Not enough live data yet."
            )

            continue

        try:

            result = run_triangulation(

                stations=stations,

                buffer=buffer,

                start=start,

                end=end,

                fs=target_fs,

                fmin=freqmin,

                fmax=freqmax,

                smooth_cutoff=smooth_cutoff,

                max_shift_seconds=shift,

                velocity=velocity
            )

            source = result[
                "source"
            ]

            print(
                "\n"
                f"[LIVE RESULT] "
                f"Source: "
                f"{source['lat']:.6f}, "
                f"{source['lon']:.6f}"
            )

            print(
                f"Residual: "
                f"{source['residual_norm']:.3f} km"
            )

            print(
                f"Closure: "
                f"{result['closure_error']:+.3f} s"
            )

            with open(
                "latest_hyperbola.png",
                "wb"
            ) as file:

                file.write(
                    base64.b64decode(
                        result["plot_png"]
                    )
                )

        except Exception as e:

            print(
                f"Analysis failed: {e}"
            )