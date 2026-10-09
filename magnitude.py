import numpy as np
from obspy.geodetics import gps2dist_azimuth


def calculate_epicentral_distance(station_lat, station_lon, epi_lat, epi_lon):
    # Calculate surface epicentral distance (in km) between a station and epicenter
    dist_m, _azimuth, _back_azimuth = gps2dist_azimuth(float(epi_lat), float(epi_lon), float(station_lat), float(station_lon))
    return float(dist_m / 1000.0)


def calculate_b_delta(delta_km):
    # Calculate the attenuation function B(Delta).
    # Equation: B(Delta) = -1.85 + 0.854 * log10(Delta) + 0.00102 * Delta
    if delta_km <= 0:
        raise ValueError("Epicentral distance must be greater than 0")
    
    return float(-1.85 + 0.854 * np.log10(delta_km) + 0.00102 * delta_km)


def calculate_ml(amplitude, period, delta_km, station_correction=0.0):
    # Calculate station Local Magnitude (ML).
    # Formula: ML = log10(A / T) + B(Delta) + C(i)
    if amplitude <= 0 or period <= 0:
        raise ValueError("Amplitude and Period must be positive values")

    b_delta = calculate_b_delta(delta_km)
    ml_station = np.log10(amplitude / period) + b_delta + station_correction
    return float(ml_station)

# if __name__ == "__main__":
#     stations = [
#         {"sta": "EQM06", "lat": 29.540123, "lon": 82.081913, "net": "NP", "cha": "EHZ", "loc": "*"},
#         {"sta": "EQM08", "lat": 27.831, "lon": 86.65, "net": "NP", "cha": "EHZ", "loc": "*"},
#         {"sta": "EQM11", "lat": 29.2859556, "lon": 81.2741972, "net": "NP", "cha": "EHZ", "loc": ""},
#     ]

#     server_url = "https://seiscomp.alertnepal.online"
#     start_time = UTCDateTime("2026-10-06T16:52:00")
#     end_time = UTCDateTime("2026-10-06T17:05:00")
