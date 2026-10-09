import xml.etree.ElementTree as ET
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient

SERVER = "ring.wscada.net:18000"

print(f"Connecting to {SERVER} to retrieve station inventory...")

try:
    # Initialize client and connect
    client = EasySeedLinkClient(SERVER, autoconnect=False)
    client.conn.timeout = 10
    client.connect()

    # Retrieve raw XML string from SeedLink server
    raw_xml_str = client.get_info("STREAMS")

    # Parse string into an XML ElementTree
    root = ET.fromstring(raw_xml_str)

    # Iterate through <station> and <stream> nodes
    for station in root.findall(".//station"):
        net_code = station.get("network", "N/A")
        sta_code = station.get("name", "N/A")

        # Collect channel names from <stream> tags
        channels = []
        for stream in station.findall("stream"):
            seedname = stream.get("seedname") or stream.get("location", "")
            if seedname:
                channels.append(seedname)

        # Deduplicate and sort channel names
        channels = list(sorted(set(channels)))
        channel_str = ", ".join(channels) if channels else "No channels listed"

        print(
            f"Network: {net_code:<5} | Station: {sta_code:<8} | Channels: {channel_str}"
        )

    # Close connection
    client.close()

except Exception as e:
    print(f"\n[ERROR]: Could not retrieve station list: {e}")
    
    
# from obspy import UTCDateTime
# from obspy.clients.fdsn import Client

# start = UTCDateTime("2026-08-26T02:30:00")
# end   = UTCDateTime("2026-08-26T03:00:00")

# client = Client("EARTHSCOPE")

# print("Searching EarthScope stations")
# print(f"Start: {start}")
# print(f"End:  {end}")
# print()

# # Nepal bounding box
# # latitude: 26–31 N
# # longitude: 80–89 E

# inventory = client.get_stations(
#     starttime=start,
#     endtime=end,
#     minlatitude=26,
#     maxlatitude=31,
#     minlongitude=80,
#     maxlongitude=89,
#     level="channel"
# )

# for network in inventory:
#     for station in network:
#         for channel in station:

#             print(
#                 f"{network.code:5s} "
#                 f"{station.code:6s} "
#                 f"{channel.location_code:4s} "
#                 f"{channel.code:5s}"
#             )