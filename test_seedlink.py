from obspy.clients.seedlink.easyseedlink import EasySeedLinkClient
import matplotlib.pyplot as plt
from obspy import Stream, UTCDateTime
import numpy as np

client = EasySeedLinkClient("ring.wscada.net:18000", autoconnect= False)

client.conn.timeout = 10


client.connect()

stream = Stream()
start = UTCDateTime()

plt.ion()
fig, (ax, ax2) = plt.subplots(2,1)


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


    # FFT
    # n = len(data)
    # sampling_rate = stream[0].stats.sampling_rate

    # frequencies = np.fft.rfftfreq(n, d=1/sampling_rate)
    # spectrum = np.abs(np.fft.rfft(data))

    # # ignore 0 hz component
    # frequencies = frequencies[1:]
    # spectrum = spectrum[1:]

    # dominant_frequency = frequencies[np.argmax(spectrum)]

    # print(f"Dominant frequency = {dominant_frequency:.2f} Hz")

    ax.clear()

    ax.plot(stream[0].times(), stream[0].data)
    ax.set_title("Live seismic data")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")

    ax2.clear()

    # ax2.plot(frequencies, spectrum)
    # ax2.set_title("Frequency Spectrum")
    # ax2.set_xlabel("Frequency (Hz)")
    # ax2.set_ylabel("Amplitude")

    # ax2.set_xlim(0, 20)
    sampling_rate = stream[0].stats.sampling_rate

    ax2.specgram(data, Fs = sampling_rate, NFFT = 128, noverlap = 96)

    ax2.set_ylim(0, 20)

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