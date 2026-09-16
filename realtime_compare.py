from obspy import Stream
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
import matplotlib.pyplot as plt

SERVER = "ring.wscada.net:18000"

TARGETS = {
    ("NP", "EQM23"): ["HNZ"],
    ("NP", "EQM24"): ["HNZ"],
}

WINDOW_SEC = 60 

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.tight_layout()

STATION_AXES = {
    "EQM23": ax1,
    "EQM24": ax2,
}


class MultiStationClient(EasySeedLinkClient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream = Stream()

    def on_data(self, trace):
        self.stream += trace
        self.stream.merge(method=1)

        # Keep only the most recent WINDOW_SEC of data per trace so the
        for tr in self.stream:
            if tr.stats.endtime - tr.stats.starttime > WINDOW_SEC:
                tr.trim(starttime=tr.stats.endtime - WINDOW_SEC)

        print(
            f"{trace.id:<15} | "
            f"Start: {trace.stats.starttime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]} | "
            f"Samples: {trace.stats.npts:<4} | "
            f"Rate: {trace.stats.sampling_rate} Hz"
        )

        st = self.stream.copy()
        st.filter('bandpass', freqmin=0.5, freqmax=10)

        self.update_plot(st)

    def update_plot(self, st):
        for ax in (ax1, ax2):
            ax.clear()

        for (net, sta), channels in TARGETS.items():
            ax = STATION_AXES.get(sta)
            if ax is None:
                continue

            sub = st.select(network=net, station=sta)
            if len(sub) == 0:
                continue

            for cha in channels:
                tr = sub.select(channel=cha)
                if len(tr) == 0:
                    continue
                tr = tr[0]
                ax.plot(
                    tr.times("matplotlib"),
                    tr.data,
                    label=cha,
                    linewidth=0.8,
                )

            ax.set_title(f"{net}.{sta}")
            ax.set_ylabel("Counts")
            ax.legend(loc="upper right", fontsize=8)
            ax.xaxis_date()

        ax2.set_xlabel("Time (UTC)")
        fig.autofmt_xdate()

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)


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