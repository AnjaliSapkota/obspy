
from dataclasses import dataclass
from typing import List, Tuple

from config import (
    DETECTION_SETTINGS,
    HISTORICAL_SETTINGS,
    REALTIME_SETTINGS,
)

from models import Pick


@dataclass
class AssociationParams:
    vp: float
    vs: float

    min_stations: int
    min_sp_sec: float
    max_sp_sec: float

    moveout_vmin: float
    moveout_tolerance: float

    dead_time_sec: float

    max_rms_sec: float

    travel_depths: Tuple[float, ...]
    travel_max_km: float
    travel_step_km: float

    wait_sec: float
    dedup_sec: float
    max_events: int

    min_stations_tentative: int = 2
    min_stations_confirmed: int = 4

    max_outlier_removals: int = 3
    outlier_sec: float = 2.0

    pick_expiry_sec: float = 120.0

    grid_coarse_km: float = 10.0
    grid_fine_km: float = 1.0
    grid_padding_km: float = 100.0

    azimuthal_gap_max: float = 180.0

    @property
    def sp_km_per_sec(self) -> float:
        return 1.0 / (1.0 / self.vs - 1.0 / self.vp)

    @classmethod
    def from_config(cls, **overrides):

        d = DETECTION_SETTINGS
        h = HISTORICAL_SETTINGS
        r = REALTIME_SETTINGS

        params = {
            "vp": d["vp"],
            "vs": d["vs"],
            "min_stations": d["min_stations"],
            "min_sp_sec": d["min_sp_sec"],
            "max_sp_sec": d["max_sp_sec"],
            "moveout_vmin": d["moveout_vmin"],
            "moveout_tolerance": d["moveout_tolerance"],
            "dead_time_sec": d["dead_time_sec"],
            "max_rms_sec": h["max_rms"],
            "travel_depths": tuple(h["travel_depths"]),
            "travel_max_km": h["travel_max_km"],
            "travel_step_km": h["travel_step_km"],
            "wait_sec": r["wait_sec"],
            "dedup_sec": r["dedup_sec"],
            "max_events": r["max_events"],
        }

        params.update(overrides)

        return cls(**params)

    def validate(self):

        if self.vp <= self.vs:
            raise ValueError(
                "P-wave velocity must be greater than S-wave velocity."
            )

        if self.vs <= 0:
            raise ValueError(
                "S-wave velocity must be greater than zero."
            )

        if self.moveout_vmin <= 0:
            raise ValueError(
                "Minimum moveout velocity must be positive."
            )

        if self.moveout_tolerance < 0:
            raise ValueError(
                "Moveout tolerance cannot be negative."
            )

        if self.min_sp_sec <= 0 or self.max_sp_sec <= self.min_sp_sec:
            raise ValueError(
                "Invalid S-P time range."
            )

        if self.max_rms_sec <= 0:
            raise ValueError(
                "Maximum RMS must be positive."
            )

        if self.min_stations_tentative < 2:
            raise ValueError(
                "Tentative events require at least two stations."
            )

        if self.min_stations_confirmed < self.min_stations_tentative:
            raise ValueError(
                "Confirmed station count cannot be below tentative count."
            )

        if self.max_outlier_removals < 0:
            raise ValueError(
                "Maximum outlier removals cannot be negative."
            )

        if self.grid_coarse_km <= 0 or self.grid_fine_km <= 0:
            raise ValueError(
                "Grid spacing must be positive."
            )

        if self.travel_max_km <= 0 or self.travel_step_km <= 0:
            raise ValueError(
                "Travel-time table distances must be positive."
            )

        if not self.travel_depths or any(
            depth < 0 for depth in self.travel_depths
        ):
            raise ValueError(
                "Travel depths must be non-negative."
            )

        if self.wait_sec < 0 or self.dedup_sec < 0:
            raise ValueError(
                "Realtime time settings cannot be negative."
            )

        if self.max_events < 1:
            raise ValueError(
                "Maximum events must be at least one."
            )

        return True


def _pick_quality(pick):

    snr = (
        pick.snr
        if pick.snr is not None
        else float("-inf")
    )

    contrast = (
        pick.aic_contrast
        if pick.aic_contrast is not None
        else float("-inf")
    )

    return snr, contrast


def prepare_picks(
    picks,
    dedup_sec=5.0,
) -> List[Pick]:
    """
    Prepare accepted P and S picks for event association.

    Picks from the same station and phase are considered duplicates
    when they occur within dedup_sec of the first pick in a cluster.

    The best-quality pick is retained from each cluster.
    """

    if dedup_sec < 0:
        raise ValueError(
            "Deduplication interval cannot be negative."
        )

    grouped = {}

    for pick in picks:

        if not pick.accepted:
            continue

        phase = pick.phase.upper()

        if phase not in ("P", "S"):
            continue

        if not pick.station_id:
            continue

        key = (pick.station_id, phase)

        grouped.setdefault(key, []).append(pick)

    prepared = []

    for key, station_picks in grouped.items():

        station_picks.sort(
            key=lambda pick: pick.arrival_time
        )

        cluster = []
        cluster_start = None

        for pick in station_picks:

            if not cluster:
                cluster = [pick]
                cluster_start = pick.arrival_time
                continue

            time_difference = (
                pick.arrival_time - cluster_start
            )

            if time_difference < dedup_sec:
                cluster.append(pick)

            else:
                best_pick = max(
                    cluster,
                    key=_pick_quality,
                )

                prepared.append(best_pick)

                cluster = [pick]
                cluster_start = pick.arrival_time

        if cluster:
            best_pick = max(
                cluster,
                key=_pick_quality,
            )

            prepared.append(best_pick)

    prepared.sort(
        key=lambda pick: pick.arrival_time
    )

    return prepared


def pair_ps(
    picks,
    min_sp_sec=2.0,
    max_sp_sec=60.0,
):
    """
    Pair P and S picks from the same station.

    Each S pick is assigned to the most recent preceding P pick,
    provided that the S-P interval is within the allowed range.

    Only one S pick can be paired with each P pick.

    Returns
    -------
    pairs : list
        List of (P_pick, S_pick) tuples.

    orphan_s : list
        S picks that could not be paired.
    """

    if min_sp_sec <= 0:
        raise ValueError(
            "Minimum S-P interval must be positive."
        )

    if max_sp_sec <= min_sp_sec:
        raise ValueError(
            "Maximum S-P interval must exceed minimum S-P interval."
        )

    station_picks = {}

    for pick in picks:

        phase = pick.phase.upper()

        if not pick.accepted:
            continue

        if phase not in ("P", "S"):
            continue

        station_picks.setdefault(
            pick.station_id,
            {"P": [], "S": []}
        )

        station_picks[pick.station_id][phase].append(pick)

    pairs = []
    orphan_s = []

    for station_id, phases in station_picks.items():

        p_picks = sorted(
            phases["P"],
            key=lambda pick: pick.arrival_time
        )

        s_picks = sorted(
            phases["S"],
            key=lambda pick: pick.arrival_time
        )

        used_s = set()

        for i, p_pick in enumerate(p_picks):

            p_time = p_pick.arrival_time

            if i + 1 < len(p_picks):
                next_p_time = p_picks[i + 1].arrival_time
            else:
                next_p_time = None

            for j, s_pick in enumerate(s_picks):

                if j in used_s:
                    continue

                s_time = s_pick.arrival_time

                if s_time <= p_time:
                    continue

                if next_p_time is not None and s_time >= next_p_time:
                    continue

                sp_time = s_time - p_time

                if sp_time < min_sp_sec:
                    continue

                if sp_time > max_sp_sec:
                    continue

                pairs.append((p_pick, s_pick))
                used_s.add(j)

                break

        for j, s_pick in enumerate(s_picks):

            if j not in used_s:
                orphan_s.append(s_pick)

    pairs.sort(
        key=lambda pair: pair[0].arrival_time
    )

    orphan_s.sort(
        key=lambda pick: pick.arrival_time
    )

    return pairs, orphan_s


def build_candidates(
    picks,
    geometry,
    min_stations=3,
):
    """
    Build possible earthquake candidates from P-wave picks.

    Two picks are compatible when:
    - They come from different stations.
    - Their arrival-time difference is within the pairwise
      maximum moveout.

    Every pair inside a candidate must be compatible.

    Returns a list of candidate groups. Each group is a list
    of Pick objects.
    """

    if min_stations < 2:
        raise ValueError(
            "At least two stations are required."
        )

    # Keep accepted P-wave picks only.
    p_picks = [
        pick
        for pick in picks
        if pick.accepted and pick.phase.upper() == "P"
    ]

    p_picks.sort(
        key=lambda pick: pick.arrival_time
    )

    if not p_picks:
        return []

    # Build a compatibility graph.
    # Each vertex represents one P pick.
    adjacency = {
        i: set()
        for i in range(len(p_picks))
    }

    for i, pick_a in enumerate(p_picks):

        for j in range(i + 1, len(p_picks)):

            pick_b = p_picks[j]

            # A candidate cannot contain two P picks
            # from the same station.
            if pick_a.station_id == pick_b.station_id:
                continue

            max_moveout = geometry.get_max_moveout(
                pick_a.station_id,
                pick_b.station_id,
            )

            time_difference = abs(
                pick_a.arrival_time - pick_b.arrival_time
            )

            if time_difference <= max_moveout:

                adjacency[i].add(j)
                adjacency[j].add(i)

    # Find maximal cliques in the compatibility graph.
    # A clique means every pick is compatible with every
    # other pick in that group.
    maximal_cliques = []

    def bron_kerbosch(current, possible, excluded):

        if not possible and not excluded:

            if len(current) >= min_stations:
                maximal_cliques.append(
                    sorted(current)
                )

            return

        if possible or excluded:

            pivot = max(
                possible | excluded,
                key=lambda vertex: len(
                    adjacency[vertex] & possible
                ),
            )

            candidates = possible - adjacency[pivot]

        else:
            candidates = set()

        for vertex in sorted(candidates):

            neighbors = adjacency[vertex]

            bron_kerbosch(
                current | {vertex},
                possible & neighbors,
                excluded & neighbors,
            )

            possible.remove(vertex)
            excluded.add(vertex)

    bron_kerbosch(
        current=set(),
        possible=set(range(len(p_picks))),
        excluded=set(),
    )

    # Convert clique indices back into Pick objects.
    candidates = []

    for clique in maximal_cliques:

        candidate = [
            p_picks[index]
            for index in clique
        ]

        candidate.sort(
            key=lambda pick: pick.arrival_time
        )

        candidates.append(candidate)

    # Sort candidates by their earliest arrival.
    candidates.sort(
        key=lambda group: group[0].arrival_time
    )

    return candidates


def geometry_precheck(
    candidate,
    pairs,
    geometry,
    vp=6.0,
    vs=3.5,
    tolerance_km=5.0,
    tolerance_fraction=0.20,
):
    """
    Check whether station S-P distances are geometrically
    compatible with one possible earthquake source.

    Parameters
    ----------
    candidate : list
        P picks belonging to one candidate.

    pairs : list
        P-S tuples returned by pair_ps().

    geometry : StationGeometry
        Contains distances between seismic stations.

    vp, vs : float
        Assumed P- and S-wave velocities in km/s.

    tolerance_km : float
        Minimum allowed geometric error in kilometres.

    tolerance_fraction : float
        Additional tolerance relative to the estimated
        source distances.

    Returns
    -------
    dict
        Geometric consistency report.
    """

    if vs <= 0 or vp <= vs:
        raise ValueError(
            "Velocities must satisfy vp > vs > 0."
        )

    if tolerance_km < 0 or tolerance_fraction < 0:
        raise ValueError(
            "Geometry tolerances cannot be negative."
        )

    denominator = (1.0 / vs) - (1.0 / vp)

    # Match P-S pairs to the P picks in this candidate.
    pair_lookup = {}

    for p_pick, s_pick in pairs:

        key = (
            p_pick.station_id,
            float(p_pick.arrival_time.timestamp),
        )

        pair_lookup[key] = s_pick

    # Calculate approximate source distance for each station.
    station_distances = {}

    for p_pick in candidate:

        key = (
            p_pick.station_id,
            float(p_pick.arrival_time.timestamp),
        )

        s_pick = pair_lookup.get(key)

        if s_pick is None:
            continue

        sp_time = (
            s_pick.arrival_time - p_pick.arrival_time
        )

        if sp_time <= 0:
            continue

        radius_km = sp_time / denominator

        station_distances[p_pick.station_id] = {
            "sp_sec": float(sp_time),
            "radius_km": float(radius_km),
        }

    station_ids = list(station_distances)

    pair_checks = []
    suspicious_stations = set()

    for i, station_a in enumerate(station_ids):

        radius_a = station_distances[station_a]["radius_km"]

        for station_b in station_ids[i + 1:]:

            radius_b = station_distances[station_b]["radius_km"]

            station_distance = geometry.get_distance(
                station_a,
                station_b,
            )

            lower_bound = abs(radius_a - radius_b)
            upper_bound = radius_a + radius_b

            tolerance = max(
                tolerance_km,
                tolerance_fraction * max(radius_a, radius_b),
            )

            lower_limit = max(
                0.0,
                lower_bound - tolerance,
            )

            upper_limit = upper_bound + tolerance

            consistent = (
                lower_limit <= station_distance <= upper_limit
            )

            pair_checks.append({
                "station_a": station_a,
                "station_b": station_b,
                "station_distance_km": station_distance,
                "radius_a_km": radius_a,
                "radius_b_km": radius_b,
                "lower_bound_km": lower_bound,
                "upper_bound_km": upper_bound,
                "tolerance_km": tolerance,
                "consistent": consistent,
            })

            if not consistent:
                suspicious_stations.add(station_a)
                suspicious_stations.add(station_b)

    if len(station_distances) < 2:
        status = "insufficient_data"
        consistent = None

    else:
        consistent = len(suspicious_stations) == 0

        status = (
            "consistent"
            if consistent
            else "suspicious"
        )

    return {
        "status": status,
        "consistent": consistent,
        "stations_checked": len(station_distances),
        "station_distances": station_distances,
        "pair_checks": pair_checks,
        "suspicious_stations": sorted(suspicious_stations),
    }

