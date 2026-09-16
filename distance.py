from obspy.geodetics.base import locations2degrees
from obspy.geodetics import degrees2kilometers

# Event center
event_lat = 28.271
event_lon = 85.515

# KKN
kkn_lat = 27.800
kkn_lon = 85.279

# EVN
evn_lat = 27.95865
evn_lon = 86.811653



distance_kkn_deg = locations2degrees(
    event_lat,
    event_lon,
    kkn_lat,
    kkn_lon
)

distance_kkn_km = degrees2kilometers(distance_kkn_deg)



distance_evn_deg = locations2degrees(
    event_lat,
    event_lon,
    evn_lat,
    evn_lon
)

distance_evn_km = degrees2kilometers(distance_evn_deg)




print("Event → KKN")
print("Distance:", distance_kkn_km, "km")

print()

print("Event → EVN")
print("Distance:", distance_evn_km, "km")