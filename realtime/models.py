
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from obspy import UTCDateTime


@dataclass
class Station:
    network: str
    code: str
    latitude: float
    longitude: float
    location: str = ""
    channel: str = ""
    elevation_m: float = 0.0
    clock_correction_sec: float = 0.0

    @property
    def station_id(self):
        return ".".join(
            part for part in (self.network, self.code, self.location)
            if part
        )


@dataclass
class Trigger:
    station_id: str
    detection_time: UTCDateTime
    cft_value: float
    duration_sec: float
    amplitude: float


@dataclass
class Pick:
    station_id: str
    phase: str
    arrival_time: UTCDateTime
    trigger_time: UTCDateTime
    snr: Optional[float] = None
    aic_shift_sec: Optional[float] = None
    aic_contrast: Optional[float] = None
    aic_boundary: bool = False
    accepted: bool = True
    rejection_reason: Optional[str] = None

# @dataclass
# class AssociationParams:
#     min_sp_sec: float
#     max_sp_sec: float

#     vp: float
#     vs: float

#     moveout_vmin: float
#     moveout_tolerance: float

#     min_stations: int
#     min_stations_tentative: int
#     min_stations_confirmed: int

#     max_rms_sec: float
#     outlier_sec: float
#     max_outlier_removals: int

#     pick_expiry_sec: float

#     travel_depths: tuple
#     travel_max_km: float
#     travel_step_km: float

#     grid_spacing_coarse_km: float
#     grid_spacing_fine_km: float
#     grid_padding_km: float

#     azimuthal_gap_max: float