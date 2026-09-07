from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient

client = EasySeedLinkClient(
    "ring.wscada.net:18000",
    autoconnect=False
)

client.conn.timeout = 30

client.connect()

streams = client.get_info("STREAMS")
print(streams)

client.close()