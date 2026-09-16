from obspy import Stream
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient

SERVER = "ring.wscada.net:18000"

TARGETS = {
    ("NP", "EQM23"): ["HNE", "HNN", "HNZ"],
    ("NP", "EQM24"): ["HNE", "HNN", "HNZ"],
}


class MultiStationClient(EasySeedLinkClient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream = Stream()

    def on_data(self, trace):
        self.stream += trace
        self.stream.merge(method=1)

        print(
            f"{trace.id:<15} | "
            f"Start: {trace.stats.starttime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]} | "
            f"Samples: {trace.stats.npts:<4} | "
            f"Rate: {trace.stats.sampling_rate} Hz"
        )


def main():

    client = MultiStationClient(SERVER, autoconnect=False)
    client.conn.timeout = 10
    client.connect()

    for (net, sta), channels in TARGETS.items():
        for cha in channels:
            client.select_stream(net, sta, cha)


    try:
        client.run()
    except KeyboardInterrupt:
        print("\nDisconnected by user.")
        client.close()


if __name__ == "__main__":
    main()