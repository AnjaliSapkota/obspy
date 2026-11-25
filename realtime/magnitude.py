import numpy as np
from obspy.geodetics import gps2dist_azimuth


def validate_number(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a valid number")

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")

    return value


def calculate_epicentral_distance(station_lat, station_lon, epi_lat, epi_lon):
    # Convert coordinates to finite floating-point values
    station_lat = validate_number(station_lat, "Station latitude")
    station_lon = validate_number(station_lon, "Station longitude")
    epi_lat = validate_number(epi_lat, "Epicenter latitude")
    epi_lon = validate_number(epi_lon, "Epicenter longitude")

    # Validate geographic coordinate ranges
    if not -90 <= station_lat <= 90:
        raise ValueError("Station latitude must be between -90 and 90")

    if not -90 <= epi_lat <= 90:
        raise ValueError("Epicenter latitude must be between -90 and 90")

    if not -180 <= station_lon <= 180:
        raise ValueError("Station longitude must be between -180 and 180")

    if not -180 <= epi_lon <= 180:
        raise ValueError("Epicenter longitude must be between -180 and 180")

    # Calculate geodesic distance in metres
    dist_m, _, _ = gps2dist_azimuth(
        epi_lat,
        epi_lon,
        station_lat,
        station_lon
    )

    # Convert metres to kilometres
    return float(dist_m / 1000.0)


def calculate_b_delta(delta_km):
    # Validate epicentral distance
    delta_km = validate_number(delta_km, "Epicentral distance")

    if delta_km <= 0:
        raise ValueError("Epicentral distance must be greater than 0")

    # Calculate distance-dependent attenuation correction
    return float(
        -1.85
        + 0.854 * np.log10(delta_km)
        + 0.00102 * delta_km
    )


def calculate_ml(amplitude, period, delta_km, station_correction=0.0):
    # Validate inputs
    amplitude = validate_number(amplitude, "Amplitude")
    period = validate_number(period, "Period")
    station_correction = validate_number(
        station_correction,
        "Station correction"
    )

    if amplitude <= 0:
        raise ValueError("Amplitude must be greater than 0")

    if period <= 0:
        raise ValueError("Period must be greater than 0")

    # Calculate attenuation correction
    b_delta = calculate_b_delta(delta_km)

    # Calculate station local magnitude
    ml_station = (
        np.log10(amplitude)
        - np.log10(period)
        + b_delta
        + station_correction
    )

    return float(ml_station)