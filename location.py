
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares


def latlon_to_xy(
    lat: float,
    lon: float,
    lat0: float,
    lon0: float,
) -> Tuple[float, float]:

    x = (lon - lon0) * 111.32 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110.574

    return x, y


def xy_to_latlon(
    x: float,
    y: float,
    lat0: float,
    lon0: float,
) -> Tuple[float, float]:

    lat = lat0 + y / 110.574
    lon = lon0 + x / (111.32 * np.cos(np.radians(lat0)))

    return lat, lon


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

    # Reference point for the local coordinate system.
    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))

    # Convert station coordinates and distances.
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

    # Residual: predicted distance minus observed distance.
    def resid(p):
        return np.hypot(
            p[0] - sx,
            p[1] - sy,
        ) - sr

    # First starting position: station centroid.
    centroid = np.array([
        sx.mean(),
        sy.mean(),
    ])

    starts = [centroid]

    # Second starting position: linear trilateration estimate.
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

    # Additional starting positions around the centroid.
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

    # Try each starting position and keep the best converged solution.
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
