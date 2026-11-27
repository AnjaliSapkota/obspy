
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth

# projects lat/lon to a local Cartesian frame (km)
def latlon_to_xy(lat: float, lon: float, lat0: float, lon0: float,) -> Tuple[float, float]:
    x = (lon - lon0) * 111.32 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110.574
    return x, y


def xy_to_latlon(x: float, y: float, lat0: float, lon0: float,) -> Tuple[float, float]:
    lat = lat0 + y / 110.574
    lon = lon0 + x / (111.32 * np.cos(np.radians(lat0)))
    return lat, lon


#   Calculate theoretical P or S travel time between an earthquake source and a station.
def calculate_theoretical_arrival(source_latitude, source_longitude, source_depth_km, station, travel_time_table, phase="P",):
    phase = phase.upper()

    if phase not in ("P", "S"):
        raise ValueError("Phase must be P or S.")

    if not np.isfinite(source_depth_km) or source_depth_km < 0:
        raise ValueError("Source depth must be non-negative.")

    if not -90 <= source_latitude <= 90:
        raise ValueError("Invalid source latitude.")

    if not -180 <= source_longitude <= 180:
        raise ValueError("Invalid source longitude.")

    distance_m, _, _ = gps2dist_azimuth(source_latitude, source_longitude, station["latitude"], station["longitude"],)

    distance_km = distance_m / 1000.0

    travel_time = travel_time_table.get_time(
        distance_km=distance_km,
        depth_km=source_depth_km,
        phase=phase,
    )

    return travel_time


def calculate_arrival_residual(observed_time, theoretical_travel_time, origin_time):
    theoretical_arrival = (origin_time + theoretical_travel_time)
    residual = observed_time - theoretical_arrival
    return float(residual)


def estimate_origin_time(arrival_times,travel_times,weights=None,):
    if len(arrival_times) != len(travel_times):
        raise ValueError("Arrival times and travel times must have equal lengths.")

    if len(arrival_times) == 0:
        raise ValueError("At least one arrival is required.")

    arrival_seconds = np.asarray([arrival.timestamp for arrival in arrival_times],dtype=float,)

    travel_seconds = np.asarray(travel_times,dtype=float,)

    if not np.all(np.isfinite(arrival_seconds)):
        raise ValueError("Arrival times must be finite.")

    if not np.all(np.isfinite(travel_seconds)):
        raise ValueError("Travel times must be finite.")

    if np.any(travel_seconds < 0):
        raise ValueError("Travel times cannot be negative.")

    origin_estimates = arrival_seconds - travel_seconds

    if weights is None:
        weights = np.ones(len(origin_estimates))

    weights = np.asarray(weights, dtype=float)

    if len(weights) != len(origin_estimates):
        raise ValueError("Weights must have the same length as arrivals.")

    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Weights must be positive and finite.")

    origin_seconds = np.average(origin_estimates,weights=weights,)

    return UTCDateTime(float(origin_seconds))


def evaluate_location(
    source_latitude,
    source_longitude,
    source_depth_km,
    picks,
    stations,
    travel_time_table,
    weights=None,
):
    """
    Evaluate one proposed earthquake source location.

    Parameters
    ----------
    source_latitude, source_longitude : float
        Proposed earthquake coordinates.

    source_depth_km : float
        Proposed earthquake depth.

    picks : list
        Pick objects containing station_id, phase,
        and arrival_time.

    stations : dict
        Station information keyed by station ID.

    travel_time_table : TravelTimeTable
        Precomputed theoretical travel-time table.

    weights : list, optional
        Weight assigned to each pick.

    Returns
    -------
    dict
        Estimated origin time, residuals, and RMS residual.
    """

    if not np.isfinite(source_latitude):
        raise ValueError("Source latitude must be finite.")

    if not np.isfinite(source_longitude):
        raise ValueError("Source longitude must be finite.")

    if not -90 <= source_latitude <= 90:
        raise ValueError("Invalid source latitude.")

    if not -180 <= source_longitude <= 180:
        raise ValueError("Invalid source longitude.")

    if not np.isfinite(source_depth_km) or source_depth_km < 0:
        raise ValueError("Invalid source depth.")

    if not picks:
        raise ValueError("At least one pick is required.")

    arrival_times = []
    travel_times = []
    pick_details = []

    for pick in picks:

        station_id = pick.station_id
        phase = pick.phase.upper()

        if phase not in ("P", "S"):
            raise ValueError(
                f"Invalid phase for {station_id}: {phase}"
            )

        if station_id not in stations:
            raise KeyError(
                f"Station not found: {station_id}"
            )

        station = stations[station_id]

        travel_time = calculate_theoretical_arrival(
            source_latitude,
            source_longitude,
            source_depth_km,
            station,
            travel_time_table,
            phase,
        )

        arrival_times.append(pick.arrival_time)
        travel_times.append(travel_time)

        pick_details.append({
            "station_id": station_id,
            "phase": phase,
            "observed_arrival": pick.arrival_time,
            "travel_time": travel_time,
        })

    origin_time = estimate_origin_time(
        arrival_times,
        travel_times,
        weights,
    )

    residuals = []

    for detail in pick_details:

        residual = calculate_arrival_residual(
            detail["observed_arrival"],
            detail["travel_time"],
            origin_time,
        )

        detail["residual_sec"] = residual

        residuals.append(residual)

    residuals = np.asarray(residuals, dtype=float)

    rms_sec = float(
        np.sqrt(np.mean(residuals ** 2))
    )

    return {
        "latitude": float(source_latitude),
        "longitude": float(source_longitude),
        "depth_km": float(source_depth_km),
        "origin_time": origin_time,
        "rms_sec": rms_sec,
        "residuals": residuals.tolist(),
        "picks": pick_details,
    }


def locate_epicenter_lsq(
    obs: List[Tuple[float, float, float]],
    source_depth_km: float = 0.0,
    *,
    max_rms_km: Optional[float] = None,
) -> Tuple[float, float, float, List[float]]:

    if not np.isfinite(source_depth_km) or source_depth_km < 0:
        raise ValueError("source_depth_km must be a non-negative finite number")

    if max_rms_km is not None:
        if not np.isfinite(max_rms_km) or max_rms_km < 0:
            raise ValueError("max_rms_km must be a non-negative finite number")

    n = len(obs)

    if n < 3:
        raise ValueError("Need at least 3 stations")

    lats = []
    lons = []
    hypos = []

    for i, entry in enumerate(obs):

        if len(entry) != 3:
            raise ValueError(
                f"obs[{i}] must be a (lat, lon, hypo) triple"
            )

        lat, lon, hypo = entry

        for name, val in (
            ("lat", lat),
            ("lon", lon),
            ("hypo", hypo),
        ):
            if not isinstance(val, (int, float, np.integer, np.floating)):
                raise ValueError(
                    f"obs[{i}].{name} must be numeric"
                )

            if not np.isfinite(val):
                raise ValueError(
                    f"obs[{i}].{name} must be finite"
                )

        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"obs[{i}].lat outside [-90, 90]")

        if not -180.0 <= lon <= 180.0:
            raise ValueError(f"obs[{i}].lon outside [-180, 180]")

        if hypo <= source_depth_km:
            raise ValueError(
                f"obs[{i}].hypo must exceed source_depth_km"
            )

        lats.append(float(lat))
        lons.append(float(lon))
        hypos.append(float(hypo))

    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    hypos = np.asarray(hypos, dtype=float)

    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))

    sx = np.empty(n, dtype=float)
    sy = np.empty(n, dtype=float)

    for i in range(n):
        sx[i], sy[i] = latlon_to_xy(
            lats[i],
            lons[i],
            lat0,
            lon0,
        )

    sr = np.sqrt(hypos**2 - source_depth_km**2)

    if len(set(zip(np.round(sx, 4), np.round(sy, 4)))) < n:
        raise ValueError("Two stations have identical coordinates")

    def resid(p):
        return np.hypot(
            p[0] - sx,
            p[1] - sy,
        ) - sr

    centroid = np.array([
        sx.mean(),
        sy.mean(),
    ])

    starts = [centroid]

    A = np.column_stack([
        2 * (sx[1:] - sx[0]),
        2 * (sy[1:] - sy[0]),
    ])

    b = (
        sr[0]**2 - sr[1:]**2
        + sx[1:]**2 - sx[0]**2
        + sy[1:]**2 - sy[0]**2
    )

    if np.linalg.matrix_rank(A) == 2:
        linear_start = np.linalg.lstsq(A, b, rcond=None)[0]
        starts.append(linear_start)

    spread = max(
        np.ptp(sx),
        np.ptp(sy),
        10.0,
    )

    for ang in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        offset = 0.5 * spread * np.array([
            np.cos(ang),
            np.sin(ang),
        ])

        starts.append(centroid + offset)

    best = None

    for s0 in starts:
        sol = least_squares(
            resid,
            s0,
            method="lm",
        )

        if not sol.success:
            continue

        if not np.all(np.isfinite(sol.x)):
            continue

        if not np.isfinite(sol.cost):
            continue

        if best is None or sol.cost < best.cost:
            best = sol

    if best is None:
        raise ValueError("Solver failed to converge to a finite solution")

    x_epi, y_epi = best.x

    residuals = resid(best.x)

    rms = float(np.sqrt(np.mean(residuals**2)))

    if not np.isfinite(rms):
        raise ValueError("Solver produced a non-finite RMS")

    if max_rms_km is not None and rms > max_rms_km:
        raise ValueError(
            f"RMS {rms:.2f} km exceeds max_rms_km={max_rms_km}"
        )

    lat_epi, lon_epi = xy_to_latlon(
        x_epi,
        y_epi,
        lat0,
        lon0,
    )

    return (
        float(lat_epi),
        float(lon_epi),
        rms,
        [float(r) for r in residuals],
    )


def grid_search_location(
    picks,
    stations,
    travel_time_table,
    latitude_range,
    longitude_range,
    depth_km,
    grid_spacing_km=10.0,
):
    """
    Search a geographical grid for the source location
    with the lowest arrival-time RMS residual.
    """

    if grid_spacing_km <= 0:
        raise ValueError(
            "Grid spacing must be positive."
        )

    min_lat, max_lat = latitude_range
    min_lon, max_lon = longitude_range

    if not -90 <= min_lat <= max_lat <= 90:
        raise ValueError("Invalid latitude range.")

    if not -180 <= min_lon <= max_lon <= 180:
        raise ValueError("Invalid longitude range.")

    center_lat = (min_lat + max_lat) / 2.0

    latitude_step = grid_spacing_km / 110.574

    longitude_step = (
        grid_spacing_km
        / (111.32 * np.cos(np.radians(center_lat)))
    )

    latitudes = np.arange(
        min_lat,
        max_lat + latitude_step,
        latitude_step,
    )

    longitudes = np.arange(
        min_lon,
        max_lon + longitude_step,
        longitude_step,
    )

    best_result = None

    for latitude in latitudes:

        for longitude in longitudes:

            try:
                result = evaluate_location(
                    source_latitude=float(latitude),
                    source_longitude=float(longitude),
                    source_depth_km=depth_km,
                    picks=picks,
                    stations=stations,
                    travel_time_table=travel_time_table,
                )

            except ValueError:
                continue

            if (
                best_result is None
                or result["rms_sec"] < best_result["rms_sec"]
            ):
                best_result = result

    if best_result is None:
        raise ValueError(
            "Grid search could not evaluate any candidate location."
        )

    best_result["grid_spacing_km"] = grid_spacing_km

    return best_result


def _best_over_depths(
    picks,
    stations,
    travel_time_table,
    depths,
    latitude_range,
    longitude_range,
    grid_spacing_km,
):
    """Run ``grid_search_location`` at every depth and keep the best RMS."""

    best_result = None

    for depth in depths:

        try:
            result = grid_search_location(
                picks=picks,
                stations=stations,
                travel_time_table=travel_time_table,
                latitude_range=latitude_range,
                longitude_range=longitude_range,
                depth_km=float(depth),
                grid_spacing_km=grid_spacing_km,
            )

        except ValueError:
            continue

        if (
            best_result is None
            or result["rms_sec"] < best_result["rms_sec"]
        ):
            best_result = result

    return best_result


def _rms_for_phase(result, phase):
    """RMS residual of one phase inside a location result."""

    values = [
        residual
        for residual, detail in zip(
            result["residuals"],
            result["picks"],
        )
        if detail["phase"] == phase
    ]

    if not values:
        return None

    return float(
        np.sqrt(np.mean(np.square(values)))
    )


def locate_event_from_picks(
    picks,
    stations,
    travel_time_table,
    depths,
    latitude_range,
    longitude_range,
    grid_spacing_km=10.0,
    min_observations=4,
    s_outlier_sec=5.0,
):
    """
    Locate an event from P and/or S picks, searching several depths.

    ``grid_search_location`` fixes the depth, leaving latitude, longitude
    and origin time as unknowns. A stable three-dimensional solution
    therefore needs at least four arrival times (for example three P plus
    one S). With only three P picks the epicenter, depth and origin time
    trade off against each other and a near-zero RMS is not evidence of a
    good location, so this refuses to report one.

    S picks add absolute distance information, but a late S pick (for
    example a surface wave picked as S) drags the epicenter away from the
    P arrivals. The P arrivals are the anchor, so:

    * S picks whose residual against the joint solution exceeds
      ``s_outlier_sec`` are removed one at a time, worst first;
    * if the P-only solution is overdetermined (four or more P picks) and
      the joint solution fits the P arrivals much worse, the P-only
      solution is returned instead.

    Parameters
    ----------
    picks : list
        Pick objects with ``station_id``, ``phase`` ("P" or "S") and
        ``arrival_time``.

    stations : dict
        Station information keyed by station ID (``geometry.stations``).

    travel_time_table : TravelTimeTable
        Precomputed P and S travel-time table.

    depths : iterable of float
        Source depths to try, in kilometres.

    latitude_range, longitude_range : tuple
        (minimum, maximum) search bounds in degrees.

    grid_spacing_km : float
        Horizontal grid spacing.

    min_observations : int
        Minimum number of arrival times required. Defaults to 4, the
        number of free parameters (latitude, longitude, depth, origin).

    s_outlier_sec : float
        Maximum allowed S residual before the pick is dropped.

    Returns
    -------
    dict
        Best location result, with ``n_observations``, ``n_p`` and
        ``n_s`` describing the arrivals actually used.
    """

    if not picks:
        raise ValueError("At least one pick is required.")

    if min_observations < 4:
        raise ValueError(
            "min_observations must be at least 4 "
            "(latitude, longitude, depth, origin)."
        )

    if len(picks) < min_observations:
        raise ValueError(
            f"Only {len(picks)} arrival(s); a stable location needs at "
            f"least {min_observations} (P and/or S)."
        )

    if s_outlier_sec <= 0:
        raise ValueError("s_outlier_sec must be positive.")

    p_picks = [
        pick for pick in picks
        if pick.phase.upper() == "P"
    ]

    s_picks = [
        pick for pick in picks
        if pick.phase.upper() == "S"
    ]

    p_only = None

    if p_picks:
        p_only = _best_over_depths(
            picks=p_picks,
            stations=stations,
            travel_time_table=travel_time_table,
            depths=depths,
            latitude_range=latitude_range,
            longitude_range=longitude_range,
            grid_spacing_km=grid_spacing_km,
        )

    if p_only is None:
        # No P arrivals: fall back to locating from S picks alone.
        result = _best_over_depths(
            picks=picks,
            stations=stations,
            travel_time_table=travel_time_table,
            depths=depths,
            latitude_range=latitude_range,
            longitude_range=longitude_range,
            grid_spacing_km=grid_spacing_km,
        )
        if result is None:
            raise ValueError(
                "Grid search could not evaluate any depth."
            )
        result["n_observations"] = len(result["picks"])
        result["n_p"] = 0
        result["n_s"] = result["n_observations"]
        return result

    joint = None

    if s_picks:
        joint = _best_over_depths(
            picks=p_picks + s_picks,
            stations=stations,
            travel_time_table=travel_time_table,
            depths=depths,
            latitude_range=latitude_range,
            longitude_range=longitude_range,
            grid_spacing_km=grid_spacing_km,
        )

    if joint is None:
        result = p_only
    else:
        # Remove S picks that do not fit the joint solution.
        while s_picks:

            residuals = []

            for s_pick in s_picks:

                station = stations[s_pick.station_id]

                distance_m, _, _ = gps2dist_azimuth(
                    joint["latitude"],
                    joint["longitude"],
                    station["latitude"],
                    station["longitude"],
                )

                travel_time = travel_time_table.get_time(
                    distance_m / 1000.0,
                    joint["depth_km"],
                    "S",
                )

                predicted = joint["origin_time"] + travel_time

                residuals.append((
                    s_pick,
                    float(s_pick.arrival_time - predicted),
                ))

            worst_pick, worst_residual = max(
                residuals,
                key=lambda item: abs(item[1]),
            )

            if abs(worst_residual) <= s_outlier_sec:
                break

            s_picks = [
                pick for pick in s_picks
                if pick is not worst_pick
            ]

            if s_picks:
                joint = _best_over_depths(
                    picks=p_picks + s_picks,
                    stations=stations,
                    travel_time_table=travel_time_table,
                    depths=depths,
                    latitude_range=latitude_range,
                    longitude_range=longitude_range,
                    grid_spacing_km=grid_spacing_km,
                )
            else:
                joint = p_only

        # S picks can still bias the epicenter. When the P-only solution is
        # overdetermined, keep it if the joint solution fits P much worse.
        if len(p_picks) >= 4:

            p_rms_only = _rms_for_phase(p_only, "P")
            p_rms_joint = _rms_for_phase(joint, "P")

            if (
                p_rms_only is not None
                and p_rms_joint is not None
                and p_rms_joint > max(1.0, 2.0 * p_rms_only)
            ):
                result = p_only
            else:
                result = joint

        else:
            result = joint

    result["n_observations"] = len(result["picks"])
    result["n_p"] = sum(
        1 for detail in result["picks"]
        if detail["phase"] == "P"
    )
    result["n_s"] = sum(
        1 for detail in result["picks"]
        if detail["phase"] == "S"
    )

    return result
