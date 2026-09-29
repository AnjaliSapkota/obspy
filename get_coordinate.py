from obspy.clients.fdsn import Client

client = Client("https://seiscomp.alertnepal.online")

inventory = client.get_stations(
    network="NP",
    station="EQM11",
    level="station"
)

for network in inventory:
    for station in network:
        print("Network:", network.code)
        print("Station:", station.code)
        print("Latitude:", station.latitude)
        print("Longitude:", station.longitude)
        print("Elevation:", station.elevation)