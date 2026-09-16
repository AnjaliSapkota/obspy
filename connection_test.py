import xml.etree.ElementTree as ET
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient

SERVER = "ring.wscada.net:18000"

print(f"Connecting to {SERVER} to retrieve station inventory...")

try:
    # 1. Initialize client and connect
    client = EasySeedLinkClient(SERVER, autoconnect=False)
    client.conn.timeout = 10
    client.connect()

    # 2. Retrieve raw XML string from SeedLink server
    raw_xml_str = client.get_info("STREAMS")

    # 3. Parse string into an XML ElementTree
    root = ET.fromstring(raw_xml_str)

    print("\n--- AVAILABLE STATIONS & CHANNELS ---")

    # 4. Iterate through <station> and <stream> nodes
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

    # 5. Close connection
    client.close()

except Exception as e:
    print(f"\n[ERROR]: Could not retrieve station list: {e}")