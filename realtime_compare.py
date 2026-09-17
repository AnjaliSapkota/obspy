from obspy import Stream
from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
import matplotlib.pyplot as plt
from scipy.signal import welch
from obspy.signal.cross_correlation import correlate, xcorr_max


SERVER = "ring.wscada.net:18000"

WINDOW = 60

stations = [
    {"net": "NP", "sta": "EQM10", "cha": "HNZ"},
    {"net": "NP", "sta": "EQM11", "cha": "HNZ"}
]

# Create plots
plt.ion()
fig, axes = plt.subplots(2, 2, figsize=(12, 7))

class MyClient(EasySeedLinkClient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream = Stream()

    def on_data(self, trace):
        if trace is None or trace.stats.npts == 0:
            return

        try:
            # 2. Append and merge incoming packet
            self.stream += trace
            self.stream.merge(method=1, fill_value="interpolate")

            for tr in self.stream:
                if tr.stats.endtime - tr.stats.starttime > WINDOW:
                    tr.trim(starttime=tr.stats.endtime - WINDOW)

            st = self.stream.copy()

            st.detrend("demean").detrend("linear")
            st.filter("bandpass", freqmin=0.1, freqmax=8.0, zerophase=True)

        except Exception as e:
            # Handle short windows or filtering errors gracefully
            return

        # Get the two stations
        tr10 = st.select(network="NP", station="EQM10", channel="HNZ")
        tr11 = st.select(network="NP", station="EQM11", channel="HNZ")

        if len(tr10) == 0 or len(tr11) == 0:
            return

        tr10 = tr10[0]
        tr11 = tr11[0]

        # Make sure both have the same sampling rate
        if tr10.stats.sampling_rate != tr11.stats.sampling_rate:
            return

        if tr10.stats.endtime - tr10.stats.starttime < WINDOW:
            return

        if tr11.stats.endtime - tr11.stats.starttime < WINDOW:
            return

        # Use the same number of samples
        n = min(len(tr10.data), len(tr11.data))

        data10 = tr10.data[-n:]
        data11 = tr11.data[-n:]

        # Cross-correlation
        cc = correlate(data10, data11, int(10 * tr10.stats.sampling_rate))

        # Find maximum correlation
        shift, value = xcorr_max(cc)

        # Convert samples to seconds
        lag = shift / tr10.stats.sampling_rate

        print(
            f"EQM10 vs EQM11 | "
            f"Lag: {lag:.3f} s | "
            f"Correlation: {value:.3f}"
        )      

        # Update plots
        for row, target in enumerate(stations):

            ax_time = axes[row][0]
            ax_freq = axes[row][1]

            ax_time.clear()
            ax_freq.clear()

            data = st.select(
                network=target["net"],
                station=target["sta"],
                channel=target["cha"]
            )

            if len(data) == 0:
                continue

            tr = data[0]

            # Waveform - time domain
            ax_time.plot(tr.times(), tr.data)

            ax_time.set_title(f"{target['net']}.{target['sta']}.{target['cha']}")
            ax_time.set_ylabel("Amplitude")
            ax_time.set_xlabel("Time (s)")

            # Frequency
            fs = tr.stats.sampling_rate

            freqs, psd = welch(
                tr.data,
                fs=fs,
                nperseg=min(1024, len(tr.data))
            )

            ax_freq.semilogy(freqs, psd)

            ax_freq.set_title(f"PSD — {target['net']}.{target['sta']}")
            ax_freq.set_xlabel("Frequency (Hz)")
            ax_freq.set_ylabel("PSD (dB/Hz)")
            # ax_freq.set_xlim(0, 15)
            # ax_freq.set_xlim(0, fs / 2.0)

        plt.tight_layout()
        plt.pause(0.01)


# Connect to RingServer

client = MyClient(
    SERVER,
    autoconnect=False
)

client.conn.timeout = 10
try:
    client.connect()

    # Subscribe using network & station targets
    for target in stations:
        client.select_stream(target["net"], target["sta"], target["cha"])

    print("Receiving live data... Press Ctrl+C to stop.")
    client.run()

except KeyboardInterrupt:
    print("\nStream stopped by user.")
except Exception as e:
    print(f"\nConnection Error: {e}")
finally:
    client.close()
    plt.close("all")