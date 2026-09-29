import base64
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import butter, correlate, correlation_lags, filtfilt, hilbert, windows
from obspy.geodetics import gps2dist_azimuth
from scipy.optimize import least_squares
from pyproj import Geod

# FDSN client
client = Client("https://seiscomp.alertnepal.online")
# Geodetic converter
geod = Geod(ellps="WGS84")

# Fetch and preprocess waveform
def fetch_waveform(station,start,end,target_fs,freqmin,freqmax):

    stream = client.get_waveforms(network=station["net"],station=station["sta"],location=station["loc"],channel=station["cha"],starttime=start,endtime=end)

    stream.merge( method=1, fill_value="interpolate")
    stream.detrend("linear")
    stream.detrend("demean")
    stream.taper( max_percentage=0.05, type="hann")
    stream.interpolate(sampling_rate=target_fs,method="linear")
    stream.filter("bandpass",freqmin=freqmin,freqmax=freqmax,corners=4,zerophase=True )

    if len(stream) == 0:
        raise RuntimeError(
            f"No processed {station['sta']} waveform found."
        )

    return stream[0]

# Trim all traces to common time window
def trim_to_common_window(traces):

    common_start = max(trace.stats.starttime for trace in traces.values())

    common_end = min(trace.stats.endtime for trace in traces.values())

    if common_start >= common_end:raise RuntimeError("No common time window exists between stations.")

    for trace in traces.values():trace.trim(starttime=common_start,endtime=common_end)

# Convert matplotlib figure to base64
def image_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer,format="png",dpi=140,bbox_inches="tight")
    plt.close(fig)

    return base64.b64encode(buffer.getvalue()).decode()

# Create waveform plot
def create_waveform_plot(traces):
    fig, axes = plt.subplots(3,1,figsize=(12, 8),sharex=True)
    for ax, (name, trace) in zip(axes,traces.items()):

        sampling_rate = (trace.stats.sampling_rate)
        data = trace.data.astype(float)
        time = (np.arange(len(data))/ sampling_rate)
        ax.plot(time, data, linewidth=0.8)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time since common start (seconds)" )

    fig.suptitle("Three-Station Seismic Waveforms")
    fig.tight_layout()
    return image_to_base64(fig)

# Create spectrogram plot
def create_spectrogram_plot(
    traces,
    fmin=1.0,
    fmax=8.0
):

    fig, axes = plt.subplots(3,1,figsize=(12, 9),sharex=True)

    for ax, (name, trace) in zip(axes,traces.items()):

        sampling_rate = (trace.stats.sampling_rate)
        data = trace.data.astype(float)
        nperseg = min(512,len(data))

        if nperseg < 2:
            raise RuntimeError(
                f"Not enough samples for spectrogram at {name}."
            )

        noverlap = int(nperseg * 0.75 )
        ax.specgram(
            data,
            NFFT=nperseg,
            Fs=sampling_rate,
            noverlap=noverlap
        )
        ax.set_ylabel( f"{name}\nFrequency (Hz)" )

        ax.set_ylim(fmin,fmax)

        ax.grid(True,alpha=0.2)

    axes[-1].set_xlabel( "Time since common start (seconds)")

    fig.suptitle(
        f"Three-Station Frequency Spectrogram "
        f"({fmin:g}–{fmax:g} Hz)"
    )

    fig.tight_layout()
    return image_to_base64(fig)

# Hilbert envelope computation
def get_hilbert_envelope(trace,smooth_cutoff=0.5):
    signal = trace.data.astype(float)
    signal = signal - np.mean(signal)

    # Hilbert transform
    analytic_signal = hilbert( signal)

    # Envelope
    envelope = np.abs(analytic_signal)

    # Smooth envelope
    if smooth_cutoff is not None:

        fs = trace.stats.sampling_rate
        b, a = butter(
            N=2,
            Wn=smooth_cutoff / (
                0.5 * fs
            ),
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


# Calculate envelope lag using cross-correlation
def calculate_envelope_lag(
    envelope_a,
    envelope_b,
    sampling_rate,
    max_shift_seconds
):

    # Make same length
    min_len = min(
        len(envelope_a),
        len(envelope_b)
    )

    envelope_a = envelope_a[:min_len]
    envelope_b = envelope_b[:min_len]

    # Remove 5% from both edges
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

    # Remove envelope mean
    envelope_a = (
        envelope_a
        - np.mean(envelope_a)
    )

    envelope_b = (
        envelope_b
        - np.mean(envelope_b)
    )

    # Tukey taper
    taper = windows.tukey(
        len(envelope_a),
        alpha=0.05
    )

    a = envelope_a * taper
    b = envelope_b * taper

    # Cross-correlation
    correlation = correlate(
        a,
        b,
        mode="full"
    )

    # Corresponding lags
    lags = correlation_lags(
        len(a),
        len(b),
        mode="full"
    )

    # Normalize
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

    # Convert lags to seconds
    lag_seconds = (
        lags
        / sampling_rate
    )

    # Limit lag search
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

    # Find maximum correlation
    peak_index = np.argmax(
        valid_corr
    )

    lag_samples = (
        valid_lags[peak_index]
    )

    lag_seconds_value = (
        lag_samples
        / sampling_rate
    )

    max_correlation = (
        valid_corr[peak_index]
    )

    return {
        "lag": float(
            lag_seconds_value
        ),

        "cc_max": float(
            max_correlation
        ),

        "lags_full": lag_seconds.tolist(),

        "cc_full": correlation.tolist()
    }


# Convert latitude/longitude to local X/Y coordinates
def latlon_to_xy(origin_lat,origin_lon,lat,lon):
    distance, azimuth, _ = gps2dist_azimuth(origin_lat, origin_lon, lat, lon)
    azimuth_rad = np.radians(azimuth)
    x = (distance / 1000.0) * np.sin(azimuth_rad)
    y = (distance / 1000.0) * np.cos(azimuth_rad)
    return x, y

# Convert local X/Y back to latitude/longitude
def xy_to_latlon(origin_lat,origin_lon,x,y):
    distance = np.sqrt(x ** 2 + y ** 2)
    azimuth = np.degrees(np.arctan2(x,y))
    source_lon, source_lat, _ = geod.fwd(origin_lon,origin_lat,azimuth,distance * 1000.0)

    return source_lat, source_lon

# Calculate TDOA residuals
def tdoa_residuals(xy,station_xy,delta,station_a,station_b,station_c):
    x, y = xy
    x_a, y_a = station_xy[station_a]
    x_b, y_b = station_xy[station_b]
    x_c, y_c = station_xy[station_c]
    d_a = np.sqrt((x - x_a) ** 2+(y - y_a) ** 2)
    d_b = np.sqrt((x - x_b) ** 2+(y - y_b) ** 2)

    d_c = np.sqrt((x - x_c) ** 2+(y - y_c) ** 2)
    res_1 = ((d_b - d_a)-delta[f"{station_a}-{station_b}"])
    res_2 = ((d_c - d_a)-delta[f"{station_a}-{station_c}"])
    res_3 = ((d_c - d_b)-delta[f"{station_b}-{station_c}"])

    return [res_1,res_2,res_3]

# Least-squares source location
def find_source_location(
    station_xy,
    delta,
    station_a,
    station_b,
    station_c
):

    initial_guess = [0.0,0.0]

    result = least_squares(
        tdoa_residuals,
        initial_guess,
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


# Create hyperbola plot
def create_hyperbola_plot(
    station_xy,
    delta,
    source_xy
):

    # Get station coordinates
    xs = [
        xy[0]
        for xy in station_xy.values()
    ]

    ys = [
        xy[1]
        for xy in station_xy.values()
    ]

    # Plotting region
    margin = 300.0
    resolution = 0.5

    xmin = min(xs) - margin
    xmax = max(xs) + margin

    ymin = min(ys) - margin
    ymax = max(ys) + margin

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

    # Distance from every grid point
    # to every station
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

    # Create figure
    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    colors = [
        "blue",
        "red",
        "green"
    ]

    legend_lines = []

    # Plot hyperbolas
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

    # Plot stations
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

    # Plot source
    source_x, source_y = source_xy

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

    # Labels
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


# Main analysis function
def run_triangulation(
    stations,
    start,
    end,
    fs=20.0,
    fmin=1.0,
    fmax=8.0,
    smooth_cutoff=0.5,
    max_shift_seconds=30.0,
    velocity=2.4
):

    if len(stations) != 3:

        raise ValueError(
            "This method needs exactly 3 stations."
        )

    # Convert times
    start = UTCDateTime(
        start
    )

    end = UTCDateTime(
        end
    )

    # Fetch waveforms
    print(
        "Fetching waveforms"
    )

    traces = {}

    for station in stations:

        print(
            f"Fetching {station['sta']}"
        )

        traces[
            station["sta"]
        ] = fetch_waveform(
            station,
            start,
            end,
            fs,
            fmin,
            fmax
        )

    # Trim to common window
    print(
        "Trimming to common window"
    )

    trim_to_common_window(
        traces
    )

    # Create waveform and spectrogram plots
    print(
        "Creating waveform plots"
    )

    waveform_plot = create_waveform_plot(
        traces
    )

    spectrogram_plot = create_spectrogram_plot(
        traces,
        fmin=fmin,
        fmax=fmax
    )

    # Calculate Hilbert envelopes
    print(
        "Calculating Hilbert envelopes"
    )

    envelopes = {}

    for name, trace in traces.items():

        envelopes[name] = (
            get_hilbert_envelope(
                trace,
                smooth_cutoff
            )
        )

    # Station names
    station_a = stations[0]["sta"]
    station_b = stations[1]["sta"]
    station_c = stations[2]["sta"]

    # Calculate pairwise TDOA
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

    # TDOA closure
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

    # Convert TDOA to distance differences
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
        f"{station_a}-{station_b}": delta_ab,
        f"{station_a}-{station_c}": delta_ac,
        f"{station_b}-{station_c}": delta_bc
    }

    # Use first station as local coordinate origin
    origin = stations[0]

    origin_lat = origin["lat"]
    origin_lon = origin["lon"]

    # Convert stations to X/Y
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

    # Least-squares source location
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
        f"X (East) : {x_src:+.3f} km"
    )

    print(
        f"Y (North): {y_src:+.3f} km"
    )

    print(
        f"Residual norm: "
        f"{residual_norm:.4f} km"
    )

    # Convert source X/Y to latitude/longitude
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
        f"Latitude  : {source_lat:.6f}"
    )

    print(
        f"Longitude : {source_lon:.6f}"
    )

    # Create hyperbola plot
    print(
        "Creating hyperbola plot"
    )

    plot_png = create_hyperbola_plot(
        station_xy,
        delta,
        (x_src, y_src)
    )

    # Prepare pairwise results for website
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

    # Prepare station coordinates
    station_coordinates = {}

    for station in stations:

        station_coordinates[
            station["sta"]
        ] = {
            "lat": station["lat"],
            "lon": station["lon"]
        }

    # Final result returned to Flask
    return {

        "pairs": pairs,

        "closure_error": float(
            closure_error
        ),

        "source": {

            "lat": float(
                source_lat
            ),

            "lon": float(
                source_lon
            ),

            "x_km": float(
                x_src
            ),

            "y_km": float(
                y_src
            ),

            "residual_norm": float(
                residual_norm
            ),

            "velocity": float(
                velocity
            )
        },

        "stations": station_coordinates,

        "waveform_plot": waveform_plot,

        "spectrogram_plot": spectrogram_plot,

        "plot_png": plot_png,

        "settings": {

            "start": str(
                start
            ),

            "end": str(
                end
            ),

            "sampling_rate": float(
                fs
            ),

            "freqmin": float(
                fmin
            ),

            "freqmax": float(
                fmax
            ),

            "smooth_cutoff": float(
                smooth_cutoff
            ),

            "max_shift_seconds": float(
                max_shift_seconds
            ),

            "velocity": float(
                velocity
            )
        }
    }


# Run directly from Python
if __name__ == "__main__":

    stations = [

        {
            "sta": "KKN",
            "lat": 27.8000,
            "lon": 85.2790,
            "net": "NK",
            "cha": "BHZ",
            "loc": "*"
        },

        {
            "sta": "EVN",
            "lat": 27.95865,
            "lon": 86.811653,
            "net": "IO",
            "cha": "BHZ",
            "loc": "*"
        },

        {
            "sta": "EQM10",
            "lat": 28.299517,
            "lon": 83.960148,
            "net": "NP",
            "cha": "EHZ",
            "loc": "*"
        }

    ]

    start = (
        "2026-08-26T02:52:00"
    )

    end = (
        "2026-08-26T02:56:00"
    )

    result = run_triangulation(

        stations=stations,

        start=start,

        end=end,

        fs=20.0,

        fmin=1.0,

        fmax=8.0,

        smooth_cutoff=0.5,

        max_shift_seconds=30.0,

        velocity=2.4
    )

    print(
        "\nAnalysis completed."
    )