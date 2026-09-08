from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
import matplotlib.pyplot as plt
from obspy import Stream, UTCDateTime
import numpy as np
from obspy.signal.trigger import classic_sta_lta

# client = EasySeedLinkClient("ring.wscada.net:18000", autoconnect= False)

# client.conn.timeout = 10

# client.connect()

start = UTCDateTime()

plt.ion()
fig, (ax, ax2, ax3) = plt.subplots(3,1, figsize=(10, 8))

class MyClient(EasySeedLinkClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream = Stream() 

    def on_data(self, trace):
        self.stream = self.stream + trace
        self.stream.merge()

        endtime = self.stream[-1].stats.endtime
        self.stream.trim(starttime = endtime - 30)

        tr = self.stream[0].copy()
        tr.detrend("demean")
        tr.detrend("linear")

        data = tr.data
        times = tr.times()

        sampling_rate = tr.stats.sampling_rate

        # Taper
        tr.taper(max_percentage = 0.05, type = 'hann')
        # Filter
        tr.filter('bandpass', freqmin = 0.1, freqmax = 30)
        # Normalize
        tr.normalize()

        # peak = np.max(np.abs(data))
        # rms = np.sqrt(np.mean(data ** 2))

        # print(
        # f"{trace.stats.starttime} | "
        # f"Peak={peak:.8e} | "
        # f"RMS={rms:.8e} | "
        # f"Min={np.min(data):.8e} | "
        # f"Max={np.max(data):.8e}"
        # )

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
            ax2.text(0.5,0.5,"Waiting for enough data...",ha="center",va="center")

        ax2.set_title("Spectrogram")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Frequency (Hz)")

        # sta/lta
        sta_window = 0.5
        lta_window = 3
        nsta = int(sta_window * sampling_rate)
        nlta = int(lta_window * sampling_rate)

        if len(data) >= nlta:
            sta_lta = classic_sta_lta(data, nsta, nlta)
        else:
            sta_lta = np.zeros_like(data)

        ax3.clear()

        if sta_lta is not None:
            sta_lta_times = np.arange(len(sta_lta)) / sampling_rate
            ax3.plot(sta_lta_times, sta_lta)
            ax3.axhline(3, linestyle="--")

        else:
            ax3.text(0.5, 0.5, "Waiting for enough data...", ha="center", va="center")

        ax3.set_title("STA/LTA Ratio")
        ax3.set_xlabel("Time (seconds)")
        ax3.set_ylabel("STA/LTA")

        plt.tight_layout()
        plt.pause(0.01)
        
        # print(trace.data[:10])
        # print(trace)
        # print(trace.stats)

        if UTCDateTime() - start >= 30:
            self.close()

        # print("Samples:", len(trace.data))
        # print("sampling rate:", trace.stats.sampling_rate)

client = MyClient(
    "ring.wscada.net:18000",
    autoconnect=False
)

client.conn.timeout = 10

client.connect()


client.select_stream("NP", "EQM22", "HNZ")

# client.on_data = on_data

    # streams_xml = client.get_info('STREAMS')
    # print(streams_xml)

client.run()


print(client.stream)