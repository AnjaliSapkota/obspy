
import numpy as np

from obspy.taup import TauPyModel


class TravelTimeTable:

    def __init__(
        self,
        depths=(10.0, 20.0, 30.0),
        max_distance_km=1400.0,
        step_km=10.0,
        model_name="iasp91",
    ):

        if not depths:
            raise ValueError("At least one depth is required.")

        if any(depth < 0 for depth in depths):
            raise ValueError("Depths cannot be negative.")

        if max_distance_km <= 0 or step_km <= 0:
            raise ValueError("Distance settings must be positive.")

        self.depths = np.asarray(
            sorted(set(float(depth) for depth in depths)),
            dtype=float,
        )

        self.distances_km = np.arange(
            0.0,
            max_distance_km + step_km,
            step_km,
        )

        self.model = TauPyModel(model=model_name)

        self.phases = {
            "P": ["P", "p"],
            "S": ["S", "s"],
        }

        self.times = {
            "P": {},
            "S": {},
        }

        self._build_table()

    def _build_table(self):

        for phase, phase_list in self.phases.items():

            for depth in self.depths:

                travel_times = []

                for distance_km in self.distances_km:

                    distance_deg = distance_km / 111.195

                    arrivals = self.model.get_travel_times(
                        source_depth_in_km=float(depth),
                        distance_in_degree=float(distance_deg),
                        phase_list=phase_list,
                    )

                    if arrivals:
                        travel_times.append(
                            min(arrival.time for arrival in arrivals)
                        )
                    else:
                        travel_times.append(np.nan)

                self.times[phase][float(depth)] = np.asarray(
                    travel_times,
                    dtype=float,
                )

    def get_time(self, distance_km, depth_km, phase="P"):

        phase = phase.upper()

        if phase not in self.phases:
            raise ValueError("Phase must be P or S.")

        distance_km = float(distance_km)
        depth_km = float(depth_km)

        if distance_km < 0:
            raise ValueError("Distance cannot be negative.")

        if depth_km not in self.times[phase]:
            raise ValueError(
                f"Depth {depth_km} km is not in the table."
            )

        if distance_km > self.distances_km[-1]:
            raise ValueError(
                "Distance exceeds travel-time table range."
            )

        values = self.times[phase][depth_km]

        valid = np.isfinite(values)

        if not np.any(valid):
            raise ValueError(
                f"No valid {phase}-wave travel times "
                f"at depth {depth_km} km."
            )

        return float(
            np.interp(
                distance_km,
                self.distances_km[valid],
                values[valid],
            )
        )
