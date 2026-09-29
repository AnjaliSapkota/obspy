from obspy import UTCDateTime
from obspy.clients.fdsn import Client

start = UTCDateTime("2026-08-26T02:30:00")
end   = UTCDateTime("2026-08-26T03:00:00")

client = Client("EARTHSCOPE")

print("Searching EarthScope stations")
print(f"Start: {start}")
print(f"End:  {end}")
print()

# Nepal bounding box
# latitude: 26–31 N
# longitude: 80–89 E

inventory = client.get_stations(
    starttime=start,
    endtime=end,
    minlatitude=26,
    maxlatitude=31,
    minlongitude=80,
    maxlongitude=89,
    level="channel"
)

for network in inventory:
    for station in network:
        for channel in station:

            print(
                f"{network.code:5s} "
                f"{station.code:6s} "
                f"{channel.location_code:4s} "
                f"{channel.code:5s}"
            )