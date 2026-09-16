from obspy.clients.fdsn import Client

client = Client("https://seiscomp.alertnepal.online")

inventory = client.get_stations(
    network="IO",
    station="EVN",
    level="station"
)

print(inventory)

station = inventory[0][0]

print("Station:", station.code)
print("Latitude:", station.latitude)
print("Longitude:", station.longitude)
print("Elevation:", station.elevation)

