
import csv
import json
import numpy as np
from obspy import UTCDateTime
from pyproj import Geod

geod = Geod(ellps="WGS84")

source_lat = 28.4755
source_lon = 84.9644

reference_time = UTCDateTime("2026-10-10T22:37:00")

stations = [
    {
        "net": "NP",
        "sta": "EQM10",
        "lat": 28.299517,
        "lon": 83.960148,
        "peak_time": "2026-10-10T22:39:38.824765",
    },
    {
        "net": "NP",
        "sta": "EQM13",
        "lat": 28.256323,
        "lon": 85.367569,
        "peak_time": "2026-10-10T22:39:33.596284",
    },
    {
        "net": "NK",
        "sta": "KKN",
        "lat": 27.800000,
        "lon": 85.279000,
        "peak_time": "2026-10-10T22:39:44.460000",
    },
]

velocities = np.linspace(1.5, 3.0, 10001)

distances = []
peak_times = []

for station in stations:
    _, _, distance_m = geod.inv(
        source_lon,
        source_lat,
        station["lon"],
        station["lat"],
    )

    distances.append(distance_m / 1000.0)

    peak_time = UTCDateTime(station["peak_time"])
    peak_times.append(float(peak_time - reference_time))

distances = np.array(distances)
peak_times = np.array(peak_times)

best_velocity = None
best_rmse = float("inf")
best_intercept = None
best_residuals = None

for velocity in velocities:
    intercepts = peak_times - distances / velocity
    event_intercept = np.mean(intercepts)

    predicted_times = event_intercept + distances / velocity
    residuals = peak_times - predicted_times

    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    if rmse < best_rmse:
        best_rmse = rmse
        best_velocity = float(velocity)
        best_intercept = float(event_intercept)
        best_residuals = residuals.copy()

boundary_hit = (
    abs(best_velocity - velocities[0]) < 1e-8
    or abs(best_velocity - velocities[-1]) < 1e-8
)

model = {
    "event_id": "LAND01",
    "model_type": "empirical_effective_envelope_velocity",
    "source_lat": source_lat,
    "source_lon": source_lon,
    "initial_velocity_km_s": 1.6,
    "tested_velocity_range_km_s": [1.5, 3.0],
    "estimated_velocity_km_s": best_velocity,
    "event_reference_offset_seconds": best_intercept,
    "timing_rmse_seconds": best_rmse,
    "velocity_range_boundary_hit": bool(boundary_hit),    "number_of_stations": len(stations),
    "calibration_events": 1,
    "validated": False,
    "interpretation": (
        "Preliminary event-specific calibration only. "
        "Do not treat as a validated regional velocity model."
    ),
}

with open("velocity_model.json", "w", encoding="utf-8") as file:
    json.dump(model, file, indent=4)

with open(
    "velocity_model_residuals.csv",
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "event_id",
        "station",
        "distance_km",
        "observed_peak_offset_s",
        "predicted_peak_offset_s",
        "residual_s",
    ])

    for i, station in enumerate(stations):
        predicted = (
            best_intercept + distances[i] / best_velocity
        )

        writer.writerow([
            "LAND01",
            station["sta"],
            round(float(distances[i]), 3),
            round(float(peak_times[i]), 3),
            round(float(predicted), 3),
            round(float(best_residuals[i]), 3),
        ])

print("\nEffective envelope-velocity calibration")
print("----------------------------------------")
print(f"Estimated velocity: {best_velocity:.3f} km/s")
print(f"Timing RMSE: {best_rmse:.3f} seconds")
print(f"Boundary hit: {boundary_hit}")

print("\nStation residuals")

for i, station in enumerate(stations):
    print(
        f"{station['sta']}: "
        f"distance={distances[i]:.2f} km, "
        f"observed={peak_times[i]:.3f} s, "
        f"residual={best_residuals[i]:+.3f} s"
    )

if boundary_hit:
    print(
        "\nWARNING: Best velocity is at the edge of the tested "
        "range. The velocity is not reliably calibrated."
    )

if best_rmse > 2.0:
    print(
        "WARNING: Timing residuals are large. Check that all "
        "three peaks represent the same landslide signal."
    )

print("\nSaved velocity_model.json")
print("Saved velocity_model_residuals.csv")
