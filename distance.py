from obspy.geodetics.base import locations2degrees
from obspy.geodetics import degrees2kilometers

# KNSET
knset_lat = 27.65337
knset_lon = 85.302528

# KKN
kkn_lat = 27.800
kkn_lon = 85.279

# EVN
evn_lat = 27.95865
evn_lon = 86.811653

# KKN - EVN
distance_kkn_evn = locations2degrees(kkn_lat,kkn_lon,evn_lat,evn_lon)

distance_kkn_evn = degrees2kilometers(distance_kkn_evn)

# EVN - KNSET
distance_evn_knset = locations2degrees(evn_lat,evn_lon,knset_lat,knset_lon)

distance_evn_knset = degrees2kilometers(distance_evn_knset)

# KKN - KNSET
distance_kkn_knset = locations2degrees(kkn_lat,kkn_lon,knset_lat,knset_lon)

distance_kkn_knset = degrees2kilometers(distance_kkn_knset)

print("KKN - KNSET")
print("Distance:", distance_kkn_knset, "km")

print()

print("KNSET - EVN")
print("Distance:", distance_evn_knset, "km")

print()

print("KKN - EVN")
print("Distance:", distance_kkn_evn, "km")