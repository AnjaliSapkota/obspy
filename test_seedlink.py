from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
import matplotlib.pyplot as plt
from obspy import Stream, UTCDateTime
import numpy as np
from obspy.signal.trigger import classic_sta_lta

client = EasySeedLinkClient("ring.wscada.net:18000", autoconnect= False)

client.conn.timeout = 10

client.connect()

stream = Stream()

start = UTCDateTime()

plt.ion()
fig, (ax, ax2, ax3) = plt.subplots(3,1)


def on_data(trace):
    global stream
    stream = stream + trace
    stream.merge()

    stream.trim(starttime = stream[-1].stats.endtime - 10,
                endtime = stream[-1].stats.endtime)

    data = stream[0].data

    peak = np.max(np.abs(data))
    rms = np.sqrt(np.mean(data ** 2))

    print(
    f"{trace.stats.starttime} | "
    f"Peak={peak:.8e} | "
    f"RMS={rms:.8e} | "
    f"Min={np.min(data):.8e} | "
    f"Max={np.max(data):.8e}"
    )
    
    ax.clear()

    ax.plot(stream[0].times(), stream[0].data)
    ax.set_title("Live seismic data")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")

    sampling_rate = stream[0].stats.sampling_rate
    
    # Spectrogram
    ax2.clear()

    if len(data) >= 128:

        ax2.specgram(
            data,
            Fs=sampling_rate,
            NFFT=128,
            noverlap=96
        )

    else:

        ax2.text(
            0.5,
            0.5,
            "Waiting for enough data...",
            ha="center",
            va="center"
        )

    ax2.set_title("Spectrogram")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_ylim(0, 20)

    # sta/lta
    sta_window = 1
    lta_window = 5

    nsta = int(sta_window * sampling_rate)
    nlta = int(lta_window * sampling_rate)

    if len(data) >= nlta:

        sta_lta = classic_sta_lta(
            data,
            nsta,
            nlta
        )

    else:

        sta_lta = None

    ax3.clear()

    if sta_lta is not None:

        ax3.plot(sta_lta)

        ax3.axhline(
            3,
            linestyle="--"
        )

    else:

        ax3.text(
            0.5,
            0.5,
            "Waiting for enough data...",
            ha="center",
            va="center"
        )

    ax3.set_title("STA/LTA Ratio")
    ax3.set_xlabel("Sample")
    ax3.set_ylabel("STA/LTA")

    plt.tight_layout()
    plt.pause(0.01)
    
    # print(trace.data[:10])
    # print(trace)
    # print(trace.stats)

    if UTCDateTime() - start >= 30:
        client.close()

    # print("Samples:", len(trace.data))
    # print("sampling rate:", trace.stats.sampling_rate)

client.select_stream("NP", "EQM22", "HNZ")

client.on_data = on_data

# streams_xml = client.get_info('STREAMS')
# print(streams_xml)

client.run()


print(stream)