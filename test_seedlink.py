from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
import matplotlib.pyplot as plt
from obspy import Stream, UTCDateTime
import numpy as np

start = UTCDateTime()
plt.ion()
fig, (ax, ax2 ) = plt.subplots(2,1, figsize=(10, 8))

class MyClient(EasySeedLinkClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream = Stream()

    def on_data(self, trace):
        self.stream = self.stream + trace
        self.stream.merge()

        endtime = self.stream[-1].stats.endtime
        self.stream.trim(starttime = endtime - 40)

        st = self.stream.copy()
        st.detrend("demean")
        st.detrend("linear")

        # Taper
        st.taper(max_percentage = 0.02, type = 'hann')
        # Filter
        st.filter('bandpass', freqmin = 0.1, freqmax = 30)
        st.trim(starttime = endtime - 30)
        tr = st[0]

        # Normalize
        tr.normalize()
        data = tr.data
        times = tr.times()
        sampling_rate = tr.stats.sampling_rate

        # Waveform plot
        ax.clear()
        ax.plot(tr.times(), tr.data)
        ax.set_title("Live seismic data")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(times[0], times[-1])

        # Spectrogram
        ax2.clear()
        nfft = int(sampling_rate) if sampling_rate > 0 else 128
        noverlap = int(nfft * 0.75)

        if len(data) >= nfft:
            ax2.specgram(
                data,
                Fs=sampling_rate,
                NFFT=nfft,
                noverlap=noverlap,
                mode = 'psd',
                scale = 'dB'
            )
            ax2.set_ylim(0, sampling_rate / 2)
            ax2.set_xlim(times[0], times[-1])
        else:
            ax2.text(0.5,0.5,"Waiting for enough data",ha="center",va="center")

        ax2.set_title("Spectrogram")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Frequency (Hz)")

        plt.tight_layout()
        plt.pause(0.01)

        if UTCDateTime() - start >= 30:
            self.close()

client = MyClient("ring.wscada.net:18000",autoconnect=False)

client.conn.timeout = 10
client.connect()
client.select_stream("NP", "EQM10", "EHZ")


client.run()
print(client.stream)