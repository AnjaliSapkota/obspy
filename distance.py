from obspy.geodetics.base import locations2degrees
from obspy.geodetics import degrees2kilometers

# kkn
kkn_lat = 27.8
kkn_lon =  85.279

# evn
evn_lat = 27.95865
evn_lon = 86.811653

# EQM10
eqm10_lat = 28.299517
eqm10_lon = 83.960148

# Event
event_lat = 28.289
event_lon = 85.528

# kkn - Event
distance_kkn_event = locations2degrees(
    kkn_lat, kkn_lon,
    event_lat, event_lon
)
distance_kkn_event = degrees2kilometers(distance_kkn_event)

# evn - Event
distance_evn_event = locations2degrees(
    evn_lat, evn_lon,
    event_lat, event_lon
)
distance_evn_event = degrees2kilometers(distance_evn_event)

# EQM10 - Event
distance_eqm10_event = locations2degrees(
    eqm10_lat, eqm10_lon,
    event_lat, event_lon
)
distance_eqm10_event = degrees2kilometers(distance_eqm10_event)

print("kkn - Event")
print("Distance:", distance_kkn_event, "km")

print()

print("evn - Event")
print("Distance:", distance_evn_event, "km")

print()

print("EQM10 - Event")
print("Distance:", distance_eqm10_event, "km")