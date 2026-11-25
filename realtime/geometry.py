
from obspy.geodetics import gps2dist_azimuth

from config import STATIONS, DETECTION_SETTINGS


class StationGeometry:

    def __init__(self, stations=None, vmin=None, tolerance=None):

        if stations is None:
            stations = STATIONS

        if vmin is None:
            vmin = DETECTION_SETTINGS["moveout_vmin"]

        if tolerance is None:
            tolerance = DETECTION_SETTINGS["moveout_tolerance"]

        self.vmin = float(vmin)
        self.tolerance = float(tolerance)

        if self.vmin <= 0:
            raise ValueError("Minimum velocity must be positive.")

        if self.tolerance < 0:
            raise ValueError("Moveout tolerance cannot be negative.")

        self.stations = {}
        self.distances = {}
        self.moveout_limits = {}

        self._load_stations(stations)
        self._calculate_geometry()

    def _load_stations(self, stations):

        for station in stations:

            network = station["net"]
            code = station["sta"]

            station_id = f"{network}.{code}"

            latitude = float(station["lat"])
            longitude = float(station["lon"])

            if not -90 <= latitude <= 90:
                raise ValueError(
                    f"Invalid latitude for {station_id}"
                )

            if not -180 <= longitude <= 180:
                raise ValueError(
                    f"Invalid longitude for {station_id}"
                )

            if station_id in self.stations:
                raise ValueError(
                    f"Duplicate station: {station_id}"
                )

            self.stations[station_id] = {
                "station_id": station_id,
                "network": network,
                "code": code,
                "latitude": latitude,
                "longitude": longitude,
            }

        if len(self.stations) < 2:
            raise ValueError(
                "At least two stations are required."
            )

    def _calculate_geometry(self):

        station_ids = list(self.stations)

        for i, station_a in enumerate(station_ids):

            coordinates_a = self.stations[station_a]

            for station_b in station_ids[i + 1:]:

                coordinates_b = self.stations[station_b]

                distance_m, _, _ = gps2dist_azimuth(
                    coordinates_a["latitude"],
                    coordinates_a["longitude"],
                    coordinates_b["latitude"],
                    coordinates_b["longitude"],
                )

                distance_km = distance_m / 1000.0

                max_moveout = (
                    distance_km / self.vmin
                ) + self.tolerance

                self.distances[
                    (station_a, station_b)
                ] = distance_km

                self.distances[
                    (station_b, station_a)
                ] = distance_km

                self.moveout_limits[
                    (station_a, station_b)
                ] = max_moveout

                self.moveout_limits[
                    (station_b, station_a)
                ] = max_moveout

    def get_station(self, station_id):

        if station_id not in self.stations:
            raise KeyError(
                f"Unknown station: {station_id}"
            )

        return self.stations[station_id]

    def get_distance(self, station_a, station_b):

        if station_a == station_b:
            return 0.0

        key = (station_a, station_b)

        if key not in self.distances:
            raise KeyError(
                f"Unknown station pair: {station_a}, {station_b}"
            )

        return self.distances[key]

    def get_max_moveout(self, station_a, station_b):

        if station_a == station_b:
            return 0.0

        key = (station_a, station_b)

        if key not in self.moveout_limits:
            raise KeyError(
                f"Unknown station pair: {station_a}, {station_b}"
            )

        return self.moveout_limits[key]

    def get_all_distances(self):

        return self.distances.copy()

    def get_all_moveout_limits(self):

        return self.moveout_limits.copy()
