import warnings
import numpy as np
import obspy

from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth
from obspy.signal.trigger import ar_pick
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")


CLIENT_NAME = "https://seiscomp.alertnepal.online"

START_TIME = "2026-09-15T20:00:00"
END_TIME   = "2026-09-15T20:25:00"

STATIONS = ["EVN", "KKN", "EQM10"]

# Request 3-component data:
# Z = vertical
# N = north
# E = east
CHANNEL_PATTERN = "BH?,HH?,EH?"

# Velocity model used for S-P -> distance conversion
V_P = 6.0       # km/s
V_S = 3.5       # km/s

# S-P distance relationship:
#
# D = (Vp * Vs / (Vp - Vs)) * (S-P)
#
K_FACTOR = (V_P * V_S) / (V_P - V_S)

# Maximum physically reasonable S-P interval
MIN_SP = 1.0
MAX_SP = 120.0

# Maximum acceptable station-to-station disagreement
MAX_DISTANCE_RESIDUAL_KM = 100.0


def get_three_components(stream):
    """
    Find one Z, one N and one E component from a station stream.

    Returns:
        (z, n, e)
    or
        None
    """

    z_traces = []
    n_traces = []
    e_traces = []

    for tr in stream:

        component = tr.stats.channel[-1].upper()

        if component == "Z":
            z_traces.append(tr)

        elif component in ["N", "1"]:
            n_traces.append(tr)

        elif component in ["E", "2"]:
            e_traces.append(tr)

    if not z_traces or not n_traces or not e_traces:
        return None

    # Prefer the longest traces
    z = max(z_traces, key=lambda x: x.stats.npts)
    n = max(n_traces, key=lambda x: x.stats.npts)
    e = max(e_traces, key=lambda x: x.stats.npts)

    return z, n, e



def prepare_components(z, n, e):
    """
    Make Z/N/E traces compatible for AR picking.
    """

    # Copy so original downloaded data are untouched
    z = z.copy()
    n = n.copy()
    e = e.copy()

    # Remove linear trend
    for tr in [z, n, e]:
        tr.detrend("linear")
        tr.detrend("demean")

    # Taper
    for tr in [z, n, e]:
        tr.taper(max_percentage=0.05)

    # Bandpass
    #
    # 1-15 Hz is a reasonable starting range for local/regional
    # earthquake phase picking.
    for tr in [z, n, e]:
        nyquist = tr.stats.sampling_rate / 2.0

        high_cut = min(15.0, nyquist * 0.8)

        if high_cut > 1.0:
            tr.filter(
                "bandpass",
                freqmin=1.0,
                freqmax=high_cut,
                corners=4,
                zerophase=True
            )

    # Trim all three to common time range
    common_start = max(
        z.stats.starttime,
        n.stats.starttime,
        e.stats.starttime
    )

    common_end = min(
        z.stats.endtime,
        n.stats.endtime,
        e.stats.endtime
    )

    if common_end <= common_start:
        raise ValueError("Z/N/E traces do not overlap in time.")

    z.trim(common_start, common_end)
    n.trim(common_start, common_end)
    e.trim(common_start, common_end)

    # Resample to same sampling rate
    target_fs = min(
        z.stats.sampling_rate,
        n.stats.sampling_rate,
        e.stats.sampling_rate
    )

    for tr in [z, n, e]:

        if abs(tr.stats.sampling_rate - target_fs) > 1e-6:
            tr.resample(target_fs)

    return z, n, e


# AUTOMATIC P/S PICKER

def detect_p_s(z, n, e):
    """
    Automatically detect P and S arrivals using ObsPy AR picker.

    Returns:
        p_time, s_time
    """

    fs = z.stats.sampling_rate

    print(f"      Sampling rate: {fs:.2f} Hz")

    # AR PICKER
    #
    # Parameters follow the documented ObsPy ar_pick usage.
    #
    # f1, f2       : frequency band
    # lta_p, sta_p : P picker windows
    # lta_s, sta_s : S picker windows
    # m_p, m_s     : AR model orders
    # l_p, l_s     : time windows
    #
    # The exact values may need tuning for your stations.

    p_pick, s_pick = ar_pick(
        z.data,
        n.data,
        e.data,
        fs,

        # frequency range
        1.0,
        15.0,

        # P-wave parameters
        1.0,
        0.1,

        # S-wave parameters
        4.0,
        1.0,

        # AR parameters
        2,
        8,

        # noise / signal thresholds
        0.1,
        0.2
    )

    # ar_pick returns seconds relative to trace start
    p_time = z.stats.starttime + p_pick
    s_time = z.stats.starttime + s_pick

    return p_time, s_time


# VALIDATE P/S PICKS
def validate_p_s(p_time, s_time):
    """
    Check whether detected P/S picks are physically reasonable.
    """

    if p_time is None or s_time is None:
        return False, "Missing P or S pick"

    sp = s_time - p_time

    if sp <= 0:
        return False, "S arrival is not after P arrival"

    if sp < MIN_SP:
        return False, f"S-P too small ({sp:.2f} s)"

    if sp > MAX_SP:
        return False, f"S-P too large ({sp:.2f} s)"

    return True, f"S-P = {sp:.2f} s"


# S-P -> HYPOCENTRAL DISTANCE

def sp_to_distance(sp_seconds):
    """
    Convert S-P time difference to approximate hypocentral distance.
    """

    return K_FACTOR * sp_seconds


# DOWNLOAD ONE STATION

def download_station(client, station, start, end):

    try:

        st = client.get_waveforms(
            network="*",
            station=station,
            location="*",
            channel=CHANNEL_PATTERN,
            starttime=start,
            endtime=end,
            attach_response=False
        )

    except Exception as e:

        print(f"  ERROR downloading {station}: {e}")
        return None

    print(f"  Downloaded traces: {len(st)}")

    if len(st) == 0:
        print("  No waveform data.")
        return None

    print(st)

    components = get_three_components(st)

    if components is None:

        print(
            "  ERROR: Could not find Z + N + E components."
        )

        print("  Available channels:")

        for tr in st:
            print(
                f"      {tr.id}"
            )

        return None

    z, n, e = components

    print(
        f"  Z: {z.id}"
    )
    print(
        f"  N: {n.id}"
    )
    print(
        f"  E: {e.id}"
    )

    try:

        z, n, e = prepare_components(z, n, e)

    except Exception as e:

        print(f"  ERROR preprocessing: {e}")
        return None

    return z, n, e


def run_calibrated_triangulation():

    print("=" * 80)
    print("AUTOMATED EARTHQUAKE P/S + 3D TRIANGULATION")
    print("=" * 80)

    print("\nConfiguration")
    print(f"  Start time : {START_TIME}")
    print(f"  End time   : {END_TIME}")
    print(f"  Stations   : {', '.join(STATIONS)}")
    print(f"  Vp         : {V_P:.2f} km/s")
    print(f"  Vs         : {V_S:.2f} km/s")
    print(f"  K factor   : {K_FACTOR:.3f}")

    client = Client(CLIENT_NAME)

    start = obspy.UTCDateTime(START_TIME)
    end = obspy.UTCDateTime(END_TIME)

    try:

        inventory = client.get_stations(
            network="*",
            station=",".join(STATIONS),
            location="*",
            channel=CHANNEL_PATTERN,
            starttime=start,
            endtime=end,
            level="station"
        )

    except Exception as e:

        print(f"Metadata error: {e}")
        return

    station_coords = {}

    for net in inventory:

        for sta in net:

            if sta.code in STATIONS:

                station_coords[sta.code] = {
                    "lat": sta.latitude,
                    "lon": sta.longitude,
                    "elevation": sta.elevation / 1000.0
                }

    print("\nStation coordinates:")

    for sta, info in station_coords.items():

        print(
            f"  {sta:4s} "
            f"Lat={info['lat']:.5f} "
            f"Lon={info['lon']:.5f} "
            f"Elev={info['elevation']:.3f} km"
        )


    picks = {}

    for station in STATIONS:

        if station not in station_coords:

            print(f"\n{station}: no station metadata")
            continue

        components = download_station(
            client,
            station,
            start,
            end
        )

        if components is None:
            continue

        z, n, e = components

        try:

            p_time, s_time = detect_p_s(
                z,
                n,
                e
            )

        except Exception as exc:

            print(
                f"  ERROR during automatic P/S picking: {exc}"
            )

            continue

        print("\n  AUTOMATIC PICKS")

        print(
            f"      P arrival: {p_time}"
        )

        print(
            f"      S arrival: {s_time}"
        )

        valid, message = validate_p_s(
            p_time,
            s_time
        )

        print(
            f"      {message}"
        )

        if not valid:

            print(
                "      Pick rejected."
            )

            continue

        sp = s_time - p_time

        distance = sp_to_distance(sp)

        print(
            f"      Estimated distance: "
            f"{distance:.2f} km"
        )

        picks[station] = {

            "p_time": p_time,
            "s_time": s_time,
            "sp": sp,
            "distance": distance,

            "lat": station_coords[station]["lat"],
            "lon": station_coords[station]["lon"],
            "elevation": station_coords[station]["elevation"]
        }


    if len(picks) < 3:

        print(
            f"Only {len(picks)} station(s) produced valid P/S picks."
        )

        print(
            "At least 3 stations are required for this 3D "
            "distance-based inversion."
        )

        return

    for sta, info in picks.items():

        print(
            f"\n{sta}"
        )

        print(
            f"  P  = {info['p_time']}"
        )

        print(
            f"  S  = {info['s_time']}"
        )

        print(
            f"  S-P = {info['sp']:.3f} s"
        )

        print(
            f"  Distance = {info['distance']:.3f} km"
        )



    def residuals(params):

        eq_lat, eq_lon, depth = params

        residual_list = []

        for station, info in picks.items():

            sta_lat = info["lat"]
            sta_lon = info["lon"]
            sta_elev = info["elevation"]

        
            distance_m, _, _ = gps2dist_azimuth(
                eq_lat,
                eq_lon,
                sta_lat,
                sta_lon
            )

            surface_distance_km = distance_m / 1000.0

            # -----------------------------------------------
            # Vertical separation
            #
            # depth is positive downward.
            #
            # station elevation is positive upward.
            #
            # So vertical separation is approximately:
            #
            # depth + station elevation
            # -----------------------------------------------

            vertical_difference = depth + sta_elev

            # -----------------------------------------------
            # Full 3D distance
            # -----------------------------------------------

            calculated_distance = np.sqrt(
                surface_distance_km ** 2
                +
                vertical_difference ** 2
            )

            # -----------------------------------------------
            # Difference between calculated and observed
            # -----------------------------------------------

            residual = (
                calculated_distance
                -
                info["distance"]
            )

            residual_list.append(residual)

        return np.array(residual_list)

    # Initial position = average station coordinates

    init_lat = np.mean([
        info["lat"]
        for info in picks.values()
    ])

    init_lon = np.mean([
        info["lon"]
        for info in picks.values()
    ])

    init_depth = 15.0

    print(
        f"\nInitial guess:"
    )

    print(
        f"  Latitude  = {init_lat:.4f}"
    )

    print(
        f"  Longitude = {init_lon:.4f}"
    )

    print(
        f"  Depth     = {init_depth:.1f} km"
    )

    # Bounds

    lower_bounds = [
        -90.0,
        -180.0,
        0.0
    ]

    upper_bounds = [
        90.0,
        180.0,
        100.0
    ]

    result = least_squares(
        residuals,
        [
            init_lat,
            init_lon,
            init_depth
        ],
        bounds=(
            lower_bounds,
            upper_bounds
        ),
        method="trf"
    )

    est_lat, est_lon, est_depth = result.x



    print(
        f"\nLatitude  : {est_lat:.5f}°"
    )

    print(
        f"Longitude : {est_lon:.5f}°"
    )

    print(
        f"Depth     : {est_depth:.2f} km"
    )

    print(
        f"\nOptimization cost : {result.cost:.4f}"
    )

    print(
        f"Optimization success: {result.success}"
    )

    print(
        f"Message: {result.message}"
    )


    print("\nStation residuals:")

    final_residuals = residuals(result.x)

    for station, residual in zip(
        picks.keys(),
        final_residuals
    ):

        print(
            f"  {station:4s}: "
            f"{residual:+.3f} km"
        )

    print(
        f"{'STA':<6}"
        f"{'P TIME':<30}"
        f"{'S TIME':<30}"
        f"{'S-P(s)':>10}"
        f"{'DIST(km)':>12}"
    )

    print("-" * 90)

    for station, info in picks.items():

        print(
            f"{station:<6}"
            f"{str(info['p_time']):<30}"
            f"{str(info['s_time']):<30}"
            f"{info['sp']:>10.3f}"
            f"{info['distance']:>12.2f}"
        )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":

    run_calibrated_triangulation()