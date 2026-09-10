from obspy import UTCDateTime
from obspy.clients.fdsn import Client

client = Client("EARTHSCOPE")

start = UTCDateTime("2026-08-26T00:00:00")
end   = UTCDateTime("2026-08-26T04:00:00")

# inventory = client.get_stations(latitude=28.2500, longitude=83.9400, maxradius=5, starttime=start, endtime=end, level="channel")

# print(inventory) 


stations = [
    ("IO", "EVN", "", "HHZ"),
    ("NK", "KKN", "", "BHZ"),
    ("NQ", "KNSET", "01", "HNZ"),
    ("NQ", "KTNP2", "01", "HNZ"),
]


for network, station, location, channel in stations:

    print(f"{network}.{station}.{location}.{channel}")
    st = client.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=start,
            endtime=end
        )

    print(st)
