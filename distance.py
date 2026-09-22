from obspy.geodetics.base import locations2degrees
from obspy.geodetics import degrees2kilometers

# eqm10
eqm_lat = 28.299517
eqm_lon = 83.960148

# KKN
kkn_lat = 27.800
kkn_lon = 85.279

# EVN
evn_lat = 27.95865
evn_lon = 86.811653



distance_kkn_evn = locations2degrees(
    kkn_lat,
    kkn_lon,
    evn_lat,
    evn_lon
)

distance_kkn_evn = degrees2kilometers(distance_kkn_evn)



distance_evn_eqm = locations2degrees(
    eqm_lat,
    eqm_lon,
    evn_lat,
    evn_lon
)

distance_evn_eqm = degrees2kilometers(distance_evn_eqm)



# kkn - eqm
distance_kkn_eqm = locations2degrees(
    kkn_lat,
    kkn_lon,
    eqm_lat,
    eqm_lon
)

distance_kkn_eqm = degrees2kilometers(distance_kkn_eqm)




print("kkn → eqm")
print("Distance:", distance_kkn_eqm, "km")

print()

print("eqm → EVN")
print("Distance:", distance_evn_eqm, "km")

print()

print("kkn → EVN")
print("Distance:", distance_kkn_evn, "km")