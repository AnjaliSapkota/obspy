from obspy.clients.fdsn import Client

client = Client("https://seiscomp.alertnepal.online")

inventory = client.get_stations(
    network="NP",
    level="station"
)

for network in inventory:
    for station in network:
        print(
            station.code,
            "|",
            station.latitude,
            "|",
            station.longitude,
            "|",
            station.elevation
        )